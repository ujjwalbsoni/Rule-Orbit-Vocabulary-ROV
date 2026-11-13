#!/usr/bin/env python3
"""
Rule-Orbit Vocabulary (ROV) miner
========================================
Finds tokens in a tokenizer vocabulary that are just a *base token + one small
rewrite rule* (leading space, capitalisation, a short suffix edit), and reports
how much the embedding table / softmax could shrink if those tokens were stored
as (base_id, rule_tag) instead of separate vocabulary entries.

Nothing is re-split, so sequence length is unchanged and the mapping is exactly
lossless (verified at the end).

What is new in v2
-----------------
Candidate merges are judged by up to two independent "gates":

  corpus gate      PPMI context vectors from --corpus (the v1 gate).
  embedding gate   vectors from a pretrained model, checked two ways:
                     1. similarity   cos(base, variant) must beat random token pairs
                     2. consistency  the offset (variant - base) must point the same way
                                     as the offsets of the *other* pairs of the same rule
                                     (all "+ed" pairs should move embeddings the same way)
                   Both thresholds are calibrated automatically against random-pair nulls,
                   so they adapt to whichever embedding space you plug in.

Embedding sources (any one enables the embedding gate):
  --embed-model ID    input-embedding matrix of a Hugging Face model (downloads model.safetensors;
                      the tokenizer you analyse MUST be that model's tokenizer)
  --embed-file F      a [vocab, dim] .npy file, or a .safetensors file (+ --embed-key)
  --word-vectors F    GloVe / word2vec text file (.txt or .gz). Word-level, so it can only judge
                      word-initial tokens (' walked'); word-internal pieces stay "unjudged".

Usage
-----
  pip install tokenizers numpy scipy              # + safetensors huggingface_hub for --embed-model

  python rule_orbit_vocab.py --demo --demo-glove
  python rule_orbit_vocab.py --hf-model gpt2 --embed-model gpt2 --corpus sample.txt
  python rule_orbit_vocab.py --tokenizer tok.json --embed-file wte.npy
  python rule_orbit_vocab.py --vocab-txt vocab.txt             # no gate -> upper bound only

A tag is (space_op, case_op, strip_k, add_suffix), shown e.g. as [+ed], [Cap], [-1+ing], [-space|Cap].
"""
from __future__ import annotations

import argparse
import gzip
import json
import os
import sys
import urllib.request
import zipfile
from array import array
from collections import Counter, defaultdict

Decomp = tuple  # (has_leading_space: bool, case: 'lower'|'cap'|'upper', body_lowercase: str)
Tag = tuple     # (space_op, case_op, strip_k, add_suffix)

GLOVE_URL = "https://github.com/RaRe-Technologies/gensim-data/releases/download/glove-wiki-gigaword-50/glove-wiki-gigaword-50.gz"
DEMO_CORPUS_URL = "https://raw.githubusercontent.com/nltk/nltk_data/gh-pages/packages/corpora/gutenberg.zip"


# --------------------------------------------------------------------------- #
# 1. Loading a vocabulary and turning raw token strings into readable text
# --------------------------------------------------------------------------- #
def bytes_to_unicode() -> dict:
    """GPT-2 byte -> printable-unicode table (used by byte-level BPE)."""
    bs = (list(range(ord("!"), ord("~") + 1)) + list(range(ord("¡"), ord("¬") + 1))
          + list(range(ord("®"), ord("ÿ") + 1)))
    cs, n = bs[:], 0
    for b in range(256):
        if b not in bs:
            bs.append(b)
            cs.append(256 + n)
            n += 1
    return dict(zip(bs, map(chr, cs)))


B2U_INV = {v: k for k, v in bytes_to_unicode().items()}


def detect_style(vocab: dict) -> str:
    toks = list(vocab)
    n = max(len(toks), 1)
    if sum(t.startswith("Ġ") for t in toks) > 0.01 * n:
        return "bytelevel"
    if sum(t.startswith("▁") for t in toks) > 0.01 * n:
        return "sentencepiece"
    return "plain"


def to_surface(raw: str, style: str):
    """Raw vocab string -> the actual text the token stands for (or None)."""
    if style == "bytelevel":
        try:
            return bytes(B2U_INV[c] for c in raw).decode("utf-8")
        except (KeyError, UnicodeDecodeError):
            return None
    if style == "sentencepiece":
        return raw.replace("▁", " ")
    return raw


def load_tokenizer(a):
    if a.vocab_txt:
        with open(a.vocab_txt, encoding="utf-8") as f:
            return None, {line.rstrip("\n"): i for i, line in enumerate(f)}
    from tokenizers import Tokenizer
    tok = Tokenizer.from_file(a.tokenizer) if a.tokenizer else Tokenizer.from_pretrained(a.hf_model)
    return tok, tok.get_vocab(with_added_tokens=True)


def train_demo_tokenizer(corpus_path: str, vocab_size: int):
    from tokenizers import Tokenizer, decoders, models, pre_tokenizers, trainers
    tok = Tokenizer(models.BPE())
    tok.pre_tokenizer = pre_tokenizers.ByteLevel(add_prefix_space=False)
    tok.decoder = decoders.ByteLevel()
    trainer = trainers.BpeTrainer(
        vocab_size=vocab_size, special_tokens=["<|endoftext|>"],
        initial_alphabet=pre_tokenizers.ByteLevel.alphabet())
    tok.train([corpus_path], trainer)
    return tok


# --------------------------------------------------------------------------- #
# 2. Decomposing tokens and applying rule tags
# --------------------------------------------------------------------------- #
def decompose(s: str):
    """' The' -> (True, 'cap', 'the').  Returns None for anything that is not a plain word piece."""
    sp = s.startswith(" ")
    body = s[1:] if sp else s
    if not body or not body.isalpha():
        return None
    if body.islower():
        case = "lower"
    elif body.isupper():
        case = "upper" if len(body) > 1 else "cap"
    elif body[0].isupper() and body[1:].islower():
        case = "cap"
    else:
        return None  # mixed case such as 'iPhone' stays a root
    return sp, case, body.lower()


def render(sp: bool, case: str, body: str) -> str:
    b = body if case == "lower" else body.capitalize() if case == "cap" else body.upper()
    return (" " if sp else "") + b


def apply_tag(dec: Decomp, tag: Tag):
    """base decomposition + tag -> surface string of the variant (None if not applicable)."""
    sp, case, body = dec
    sp_op, case_op, k, add = tag
    if sp_op == "+sp":
        sp = True
    elif sp_op == "-sp":
        sp = False
    if case_op != "=":
        case = case_op
    if k > len(body):
        return None
    return render(sp, case, body[:len(body) - k] + add)


def tag_str(tag: Tag) -> str:
    sp_op, case_op, k, add = tag
    parts = []
    if sp_op != "=":
        parts.append({"+sp": "+space", "-sp": "-space"}[sp_op])
    if case_op != "=":
        parts.append({"lower": "lower", "cap": "Cap", "upper": "UPPER"}[case_op])
    if k or add:
        parts.append((f"-{k}" if k else "") + (f"+{add}" if add else ""))
    return "[" + "|".join(parts) + "]" if parts else "[identity]"


def is_orth_only(tag: Tag) -> bool:
    return tag[2] == 0 and tag[3] == ""


# --------------------------------------------------------------------------- #
# 3. Mining candidate (base, variant, tag) pairs
# --------------------------------------------------------------------------- #
def _eligible_for_suffix(dec_entry, min_stem_len: int) -> bool:
    """Suffix edits only on lowercase word stems: capitalised names (' Mar', ' Car') and very short stems are excluded."""
    sp, case, body = dec_entry
    return case == "lower" and len(body) >= min_stem_len


def mine_pairs(dec: dict, surf: dict, a) -> dict:
    """
    Group tokens by shared stem (prefix of the lowercase body) and compare tokens inside each group.
    The lower-id token (more frequent for BPE) is the base candidate.
    Returns {(base_id, variant_id): (cost, tag)}.
    """
    groups = defaultdict(list)
    for tid, entry in dec.items():
        _sp, _case, body = entry
        n = len(body)
        for j in range(max(a.min_stem, n - a.max_suffix), n + 1):
            groups[body[:j]].append((tid, body[j:]))

    best = {}
    for members in groups.values():
        if len(members) < 2 or len(members) > a.max_group:
            continue
        members.sort()
        for i, (bid, bx) in enumerate(members):
            bsp, bcase, _ = dec[bid]
            for vid, vx in members[i + 1:]:
                vsp, vcase, _ = dec[vid]
                sp_op = "=" if bsp == vsp else ("+sp" if vsp else "-sp")
                case_op = "=" if bcase == vcase else vcase
                changes = (sp_op != "=") + (case_op != "=") + (1 if (bx or vx) else 0)
                if changes == 0 or changes > a.max_changes:
                    continue
                if (bx or vx) and (sp_op != "=" or case_op != "=") and not a.allow_composite_suffix:
                    continue  # ' look'/'looked' or ' think'/' This': composites are tiny groups and mostly false
                cost = changes * 10 + len(bx) + len(vx)
                if (bid, vid) in best and best[(bid, vid)][0] <= cost:
                    continue
                tag = (sp_op, case_op, len(bx), vx)
                if bx or vx:  # suffix-edit rule: stems must be lowercase words of at least a.min_suffix_stem letters
                    if not (_eligible_for_suffix(dec[bid], a.min_suffix_stem)
                            and _eligible_for_suffix(dec[vid], a.min_suffix_stem)):
                        continue
                if apply_tag(dec[bid], tag) != surf[vid]:  # exact-reconstruction check
                    continue
                best[(bid, vid)] = (cost, tag)
    return best


def support_filter(pairs: dict, tau: int) -> dict:
    """Keep a suffix edit only if it recurs in >= tau pairs (space/case-only rules always pass)."""
    support = Counter((t[2], t[3]) for _c, t in pairs.values())
    return {k: v for k, v in pairs.items() if is_orth_only(v[1]) or support[(v[1][2], v[1][3])] >= tau}


# --------------------------------------------------------------------------- #
# 4a. Corpus gate: PPMI context vectors
# --------------------------------------------------------------------------- #
def encode_corpus(tok, corpus_path: str, vocab_size: int):
    import numpy as np
    ids = array("i")
    with open(corpus_path, encoding="utf-8", errors="ignore") as f:
        lines = [ln for ln in f.read().split("\n") if ln.strip()]
    for s in range(0, len(lines), 5000):
        for enc in tok.encode_batch(lines[s:s + 5000]):
            ids.extend(enc.ids)
            ids.append(-1)  # line separator; context windows must not cross it
    ids = np.frombuffer(ids, dtype=np.int32)
    valid = ids[ids >= 0]
    return ids, np.bincount(valid, minlength=vocab_size), int(valid.size)


def corpus_gate(pairs: dict, corp, vocab_size: int, a):
    """Returns ({pair: (cos, None, accepted)}, info_line) for the pairs it could judge."""
    import numpy as np
    from scipy.sparse import coo_matrix
    ids, counts, _total = corp

    cand = sorted({i for k in pairs for i in k if counts[i] >= a.gate_min_count})
    cand_index = np.full(vocab_size, -1, dtype=np.int64)
    cand_index[cand] = np.arange(len(cand))
    ctx_ids = np.argsort(-counts)[:a.gate_dims]
    ctx_index = np.full(vocab_size, -1, dtype=np.int64)
    ctx_index[ctx_ids] = np.arange(len(ctx_ids))

    rows, cols = [], []
    for d in range(1, a.gate_window + 1):
        left, right = ids[:-d], ids[d:]
        for centre, context in ((left, right), (right, left)):
            ok = (centre >= 0) & (context >= 0)
            ci, xi = cand_index[centre[ok]], ctx_index[context[ok]]
            m = (ci >= 0) & (xi >= 0)
            rows.append(ci[m])
            cols.append(xi[m])
    rows, cols = np.concatenate(rows), np.concatenate(cols)
    M = coo_matrix((np.ones(rows.size, dtype=np.float32), (rows, cols)),
                   shape=(len(cand), len(ctx_ids))).tocsr()

    pc = counts[ctx_ids].astype(np.float64) ** 0.75   # PPMI of p(c|w) vs smoothed p(c)
    pc /= pc.sum()
    row_tot = np.asarray(M.sum(1)).ravel()
    nnz_row = np.repeat(np.arange(M.shape[0]), np.diff(M.indptr))
    pmi = np.log((M.data / row_tot[nnz_row]) / pc[M.indices])
    M.data = np.maximum(pmi, 0).astype(np.float32)
    norms = np.sqrt(np.asarray(M.multiply(M).sum(1)).ravel())
    norms[norms == 0] = 1.0
    M = M.multiply(1.0 / norms[:, None]).tocsr()

    keys = [k for k in pairs if cand_index[k[0]] >= 0 and cand_index[k[1]] >= 0]
    out = {}
    for s in range(0, len(keys), 50000):
        chunk = keys[s:s + 50000]
        I = cand_index[[k[0] for k in chunk]]
        J = cand_index[[k[1] for k in chunk]]
        cos = np.asarray(M[I].multiply(M[J]).sum(1)).ravel().tolist()
        for k, c in zip(chunk, cos):
            theta = a.theta_orth if is_orth_only(pairs[k][1]) else a.theta
            out[k] = (c, None, c >= theta)
    return out, f"corpus-PPMI gate: cos >= {a.theta} (suffix) / {a.theta_orth} (space,case); judged {len(out):,}"


# --------------------------------------------------------------------------- #
# 4b. Embedding gate: similarity + rule-consistency of offsets
# --------------------------------------------------------------------------- #
EMBED_KEY_HINTS = ("embed_tokens", "wte", "word_embeddings", "tok_embeddings", "embeddings.weight", "shared")


def _pick_embed_key(keys, forced):
    if forced:
        return forced
    hits = [k for k in keys if k.endswith("weight") and any(h in k for h in EMBED_KEY_HINTS)]
    if not hits:
        raise SystemExit("could not find the embedding matrix; pass --embed-key (keys: " + ", ".join(list(keys)[:8]) + " ...)")
    return sorted(hits, key=len)[0]


def read_safetensors_embedding(path: str, forced_key=None):
    import numpy as np
    from safetensors import safe_open
    with safe_open(path, framework="np") as f:
        key = _pick_embed_key(list(f.keys()), forced_key)
        try:
            return np.asarray(f.get_tensor(key), dtype=np.float32)
        except Exception:  # e.g. bfloat16 is not representable in numpy
            pass
    with safe_open(path, framework="pt") as f:
        return f.get_tensor(key).float().numpy()


def download_hf_embedding(model_id: str, forced_key=None):
    from huggingface_hub import hf_hub_download
    try:
        return read_safetensors_embedding(hf_hub_download(model_id, "model.safetensors"), forced_key)
    except Exception as e:  # sharded checkpoint: find the shard that holds the embedding
        if "model.safetensors" not in str(e) and "Entry Not Found" not in str(e) and "404" not in str(e):
            raise
    idx = json.load(open(hf_hub_download(model_id, "model.safetensors.index.json"), encoding="utf-8"))["weight_map"]
    key = _pick_embed_key(list(idx), forced_key)
    return read_safetensors_embedding(hf_hub_download(model_id, idx[key]), key)


def load_word_vectors(path: str, needed: set):
    import numpy as np
    opener = gzip.open if path.endswith(".gz") else open
    vecs = {}
    with opener(path, "rt", encoding="utf-8", errors="ignore") as f:
        for n, line in enumerate(f):
            word, _, rest = line.partition(" ")
            if n == 0 and rest.strip().isdigit():  # word2vec header "N dim"
                continue
            if word in needed:
                vecs[word] = np.array(rest.split(), dtype=np.float32)
    return vecs


class EmbSource:
    """Centred, L2-normalised vectors for the tokens an embedding source can describe."""

    def __init__(self, E, rows, key_of, label):
        self.E, self.rows, self.key_of, self.label = E, rows, key_of, label


def build_embedding_source(a, ctx) -> EmbSource:
    import numpy as np
    surf, dec, vocab_size = ctx["surf"], ctx["dec"], ctx["vocab_size"]
    ids = sorted(dec)
    key_of = None
    if a.word_vectors:
        # word-level vectors can only describe word-initial tokens; case/space twins would get
        # identical vectors by construction, so tokens sharing a lowercase key are never "judged".
        want = {i: surf[i][1:] for i in ids if dec[i][0]}
        needed = set(want.values()) | {w.lower() for w in want.values()}
        wv = load_word_vectors(a.word_vectors, needed)
        have, vecs, key_of = [], [], {}
        for i, w in want.items():
            v = wv.get(w)
            v = wv.get(w.lower()) if v is None else v
            if v is not None:
                have.append(i)
                vecs.append(v)
                key_of[i] = w.lower()
        ids, E, label = have, np.stack(vecs), f"word vectors ({os.path.basename(a.word_vectors)})"
    else:
        if a.embed_file:
            M = np.load(a.embed_file, mmap_mode="r") if a.embed_file.endswith(".npy") \
                else read_safetensors_embedding(a.embed_file, a.embed_key)
            label = os.path.basename(a.embed_file)
        else:
            M = download_hf_embedding(a.embed_model, a.embed_key)
            label = a.embed_model
        if M.shape[0] != vocab_size:
            print(f"warning: embedding rows ({M.shape[0]:,}) != tokenizer vocab ({vocab_size:,}); "
                  f"make sure the tokenizer belongs to this model", file=sys.stderr)
        ids = [i for i in ids if i < M.shape[0]]
        E = np.asarray(M[ids], dtype=np.float32)
    E = E - E.mean(0, keepdims=True)  # remove the shared mean direction (LM embeddings are anisotropic)
    norms = np.linalg.norm(E, axis=1, keepdims=True)
    norms[norms == 0] = 1.0
    return EmbSource(E / norms, {i: r for r, i in enumerate(ids)}, key_of, label)


def embedding_gate(pairs: dict, src: EmbSource, a):
    """Returns ({pair: (cos, offset_consistency|None, accepted)}, info_line)."""
    import numpy as np
    rng = np.random.default_rng(0)
    rows, E = src.rows, src.E
    judged = [k for k in pairs if k[0] in rows and k[1] in rows
              and (src.key_of is None or src.key_of[k[0]] != src.key_of[k[1]])]
    if not judged:
        return {}, f"embedding gate ({src.label}): judged 0 pairs"
    B = E[[rows[k[0]] for k in judged]]
    V = E[[rows[k[1]] for k in judged]]
    sim = (B * V).sum(1)

    # --- null: how similar are two random tokens in this space?
    nr = E.shape[0]
    ia, ib = rng.integers(0, nr, 20000), rng.integers(0, nr, 20000)
    ok = ia != ib
    null_sim = (E[ia[ok]] * E[ib[ok]]).sum(1)
    theta_sim = a.emb_theta if a.emb_theta is not None else float(np.quantile(null_sim, a.emb_null_quantile))
    # the rescue override is only meaningful above the floor; with an explicit --emb-theta it is off unless asked for
    strong_q = a.emb_strong_quantile if a.emb_strong_quantile is not None else (None if a.emb_theta is not None else 0.999)
    theta_strong = float(np.quantile(null_sim, strong_q)) if strong_q else 9.0

    # --- rule consistency: leave-one-out cosine between a pair's offset and its rule's mean offset
    off = V - B
    nrm = np.linalg.norm(off, axis=1)
    has_off = nrm > 1e-6
    unit = off / np.where(has_off, nrm, 1.0)[:, None]
    by_tag = defaultdict(list)
    for n, k in enumerate(judged):
        if has_off[n] and sim[n] >= theta_sim:  # direction is estimated from plausible pairs only
            by_tag[pairs[k][1]].append(n)
    cons = np.full(len(judged), np.nan)
    dirs, S_by_tag = [], {}
    for tag, idx in by_tag.items():
        if len(idx) < a.offset_min_group:
            continue
        U = unit[idx]
        S = U.sum(0)
        S_by_tag[tag] = S
        L = S[None, :] - U                      # leave-one-out direction for pairs inside the estimate
        cons[idx] = (U * L).sum(1) / np.maximum(np.linalg.norm(L, axis=1), 1e-9)
        dirs.append(S / np.linalg.norm(S))
    in_estimate = set(i for idx in by_tag.values() for i in idx)
    for n, k in enumerate(judged):              # pairs that failed the similarity pre-filter: score vs full direction
        if n not in in_estimate and has_off[n] and pairs[k][1] in S_by_tag:
            S = S_by_tag[pairs[k][1]]
            cons[n] = float(unit[n] @ S / np.linalg.norm(S))
    if a.no_offset:
        theta_off = -2.0
    elif a.offset_theta is not None:
        theta_off = a.offset_theta
    elif dirs:
        D = np.stack(dirs)
        ra, rb = rng.integers(0, nr, 4000), rng.integers(0, nr, 4000)
        okk = ra != rb
        O = E[rb[okk]] - E[ra[okk]]
        O /= np.maximum(np.linalg.norm(O, axis=1, keepdims=True), 1e-9)
        theta_off = float(np.quantile((O @ D.T).ravel(), a.off_null_quantile))
    else:
        theta_off = -2.0

    out = {}
    for n, k in enumerate(judged):
        c = None if np.isnan(cons[n]) else float(cons[n])
        out[k] = (float(sim[n]), c, bool(sim[n] >= theta_sim and (c is None or c >= theta_off or sim[n] >= theta_strong)))
    n_cons = int((~np.isnan(cons)).sum())
    origin = "set by --emb-theta" if a.emb_theta is not None else f"p{a.emb_null_quantile * 100:.0f} of random pairs"
    line = (f"embedding gate ({src.label}): cos >= {theta_sim:.2f} ({origin})"
            + ("" if a.no_offset else f", offset-consistency >= {theta_off:.2f}")
            + ((f" [INACTIVE: override cos >= {theta_strong:.2f} is below the floor, so the test is skipped]"
                if theta_strong <= theta_sim else f" unless cos >= {theta_strong:.2f}") if strong_q and not a.no_offset else "")
            + f"; judged {len(out):,}, consistency testable for {n_cons:,}")
    return out, line


# --------------------------------------------------------------------------- #
# 5. Combine gates, build orbits (star-shaped: every variant is one rule from a root)
# --------------------------------------------------------------------------- #
def prepare(vocab, tok, a) -> dict:
    style = detect_style(vocab)
    surf, surf2id, dup = {}, {}, set()
    for raw, i in vocab.items():
        s = to_surface(raw, style)
        if s is None:
            continue
        if s in surf2id:
            dup.add(s)
        surf2id[s] = i
        surf[i] = s
    surf = {i: s for i, s in surf.items() if s not in dup}  # ambiguous strings stay untouched roots
    dec = {i: d for i, s in surf.items() if (d := decompose(s))}
    return {"vocab": vocab, "tok": tok, "style": style, "surf": surf, "surf2id": surf2id, "dec": dec,
            "vocab_size": max(vocab.values()) + 1, "_corpus": None, "_emb": {}}


def get_corpus(ctx, a):
    if ctx["_corpus"] is None:
        ctx["_corpus"] = encode_corpus(ctx["tok"], a.corpus, ctx["vocab_size"])
    return ctx["_corpus"]


def get_embedding(ctx, a):
    key = (a.embed_model, a.embed_file, a.word_vectors, a.embed_key)
    if key not in ctx["_emb"]:
        ctx["_emb"][key] = build_embedding_source(a, ctx)
    return ctx["_emb"][key]


def plan_gates(ctx, a):
    has_corpus = ctx["tok"] is not None and bool(a.corpus)
    has_emb = bool(a.embed_model or a.embed_file or a.word_vectors)
    return {"auto": (has_corpus and not has_emb, has_emb), "corpus": (has_corpus, False), "embeddings": (False, has_emb),
            "both": (has_corpus, has_emb), "none": (False, False)}[a.gate]


def apply_gates(pairs: dict, ctx, a):
    use_corpus, use_emb = plan_gates(ctx, a)
    results, lines = [], []
    if use_corpus:
        r, line = corpus_gate(pairs, get_corpus(ctx, a), ctx["vocab_size"], a)
        results.append(r)
        lines.append(line)
    if use_emb:
        r, line = embedding_gate(pairs, get_embedding(ctx, a), a)
        results.append(r)
        lines.append(line)
    if not results:
        return pairs, {}, {}, ["no gate (no corpus / embeddings) -> numbers are an UPPER BOUND"], len(pairs)
    accepted, scores, offs, unjudged = {}, {}, {}, 0
    for k, v in pairs.items():
        decisions = [r[k][2] for r in results if k in r]
        if decisions:
            if all(decisions):  # every gate that could judge the pair must agree
                accepted[k] = v
            emb = results[-1].get(k) if use_emb else None
            first = emb if emb else next(r[k] for r in results if k in r)
            scores[k] = first[0]
            if emb and emb[1] is not None:
                offs[k] = emb[1]
        else:
            unjudged += 1
            if a.gate_unknown == "accept":
                accepted[k] = v
    return accepted, scores, offs, lines, unjudged


def build_orbits(dec: dict, pairs: dict) -> dict:
    cands = defaultdict(list)
    for (bid, vid), (cost, tag) in pairs.items():
        cands[vid].append((cost, bid, tag))
    assign = {}  # variant_id -> (base_id, tag)
    for tid in sorted(dec):  # ascending id: bases are decided before their variants
        options = [(c, b, t) for (c, b, t) in cands.get(tid, []) if b not in assign]
        if options:
            _c, b, t = min(options)
            assign[tid] = (b, t)
    return assign


def analyse(ctx, a, mined=None) -> dict:
    """mined: optional pre-computed (raw_pairs, support_filtered_pairs) to reuse across gate configs."""
    if mined is None:
        raw = mine_pairs(ctx["dec"], ctx["surf"], a)
        mined = (raw, support_filter(raw, a.tau))
    raw, sup = mined
    fam = getattr(a, "families", "all")
    if fam != "all":  # restrict to one rule family: 'orth' = space/case only, 'suffix' = suffix edits only
        def keep(tag):
            orth = is_orth_only(tag)
            return {"suffix": not orth, "orth": orth, "case": orth and tag[0] == "=",
                    "space": orth and tag[0] != "=",
                    "cap": orth and tag[0] == "=" and tag[1] in ("cap", "upper")}[fam]
        sup = {k: v for k, v in sup.items() if keep(v[1])}
    pairs, scores, offs, lines, unjudged = apply_gates(sup, ctx, a)
    if not lines[0].startswith("no gate"):
        pairs = support_filter(pairs, a.tau)  # re-check recurrence once spurious pairs are gone
    assign = build_orbits(ctx["dec"], pairs)
    return {"raw": raw, "sup": sup, "pairs": pairs, "scores": scores, "offs": offs, "gate_lines": lines,
            "unjudged": unjudged, "assign": assign}


# --------------------------------------------------------------------------- #
# 6. Reporting + lossless verification
# --------------------------------------------------------------------------- #
def verify_lossless(res, ctx, a):
    dec, surf2id, tok, assign = ctx["dec"], ctx["surf2id"], ctx["tok"], res["assign"]
    bad = sum(1 for vid, (bid, tag) in assign.items() if surf2id.get(apply_tag(dec[bid], tag)) != vid)
    msg = (f"every one of the {len(assign):,} variants rebuilds from (base, tag) exactly" if not bad
           else f"{bad} variants failed to rebuild!")
    if tok is not None and a.corpus and not bad:
        with open(a.corpus, encoding="utf-8", errors="ignore") as f:
            sample = f.read(200_000)
        ids = tok.encode(sample).ids
        mapped = [(assign[i][0], assign[i][1]) if i in assign else (i, None) for i in ids]
        ids2 = [i if t is None else surf2id[apply_tag(dec[i], t)] for i, t in mapped]
        same = ids == ids2 and tok.decode(ids) == tok.decode(ids2)
        msg += f"; text round-trip ({len(ids):,} tokens, same length): {'OK' if same else 'MISMATCH'}"
        bad += 0 if same else 1
    return bad == 0, msg


def report(a, ctx, res, ok_msg):
    vocab, surf, dec, assign = ctx["vocab"], ctx["surf"], ctx["dec"], res["assign"]
    scores, offs, pairs = res["scores"], res["offs"], res["pairs"]
    V = len(vocab)
    V_base = V - len(assign)
    tags = Counter(t for _b, t in assign.values())
    n_sp = len({t[0] for t in tags} | {"="})
    n_case = len({t[1] for t in tags} | {"="})
    n_suf = len({(t[2], t[3]) for t in tags} | {(0, "")})
    T_comp, T_fact = len(tags) + 1, n_sp + n_case + n_suf
    d = a.d_model
    gated = not res["gate_lines"][0].startswith("no gate")

    print("=" * 76)
    print("RULE-ORBIT VOCABULARY REPORT")
    print("=" * 76)
    print(f"vocab size                         : {V:,}")
    print(f"decomposable word-like tokens      : {len(dec):,}")
    print(f"candidate pairs mined              : {len(res['raw']):,}")
    print(f"  after rule-support filter (>= {a.tau}): {len(res['sup']):,}")
    for line in res["gate_lines"]:
        print(f"  {line}")
    if gated:
        print(f"  pairs accepted by gates          : {len(pairs):,}   (unjudged: {res['unjudged']:,} -> {a.gate_unknown})")
    print(f"base tokens (roots)                : {V_base:,}")
    print(f"variants folded into an orbit      : {len(assign):,}  ({len(assign) / V:.1%} of vocab)")
    orth = sum(1 for _b, t in assign.values() if is_orth_only(t))
    print(f"  space/case-only rules            : {orth:,} ({orth / V:.1%} of vocab)   <- riskiest family")
    print(f"  suffix-edit rules                : {len(assign) - orth:,} ({(len(assign) - orth) / V:.1%} of vocab)")
    print(f"distinct rule tags                 : {T_comp:,} composite, or {T_fact} factored "
          f"({n_sp} space + {n_case} case + {n_suf} suffix-edit)")
    print(f"\nembedding rows x d_model={d} (per table: input embedding or output head)")
    print(f"  flat vocab                       : {V * d / 1e6:,.1f} M params")
    print(f"  ROV composite tags               : {(V_base + T_comp) * d / 1e6:,.1f} M ({1 - (V_base + T_comp) / V:.1%} smaller)")
    print(f"  ROV factored tags                : {(V_base + T_fact) * d / 1e6:,.1f} M ({1 - (V_base + T_fact) / V:.1%} smaller)")
    corp = ctx["_corpus"]
    if corp is not None:
        traffic = sum(int(corp[1][v]) for v in assign) / max(corp[2], 1)
        print(f"share of corpus tokens that are variants : {traffic:.1%}")
    print(f"lossless check                     : {ok_msg}")

    print("\nTOP RULES   pairs  mean-cos  mean-offset  example base -> variant")
    ex, cs, os_ = defaultdict(list), defaultdict(list), defaultdict(list)
    for vid, (bid, t) in assign.items():
        if len(ex[t]) < 3:
            ex[t].append(f"{surf[bid]!r}->{surf[vid]!r}")
        if (bid, vid) in scores:
            cs[t].append(scores[(bid, vid)])
        if (bid, vid) in offs:
            os_[t].append(offs[(bid, vid)])
    for t, c in tags.most_common(a.top):
        m1 = f"{sum(cs[t]) / len(cs[t]):5.2f}" if cs[t] else "  n/a"
        m2 = f"{sum(os_[t]) / len(os_[t]):5.2f}" if os_[t] else "  n/a"
        print(f"  {c:6,}  {m1}     {m2}     {tag_str(t):<16} {', '.join(ex[t])}")

    orbit = defaultdict(list)
    for vid, (bid, t) in assign.items():
        orbit[bid].append((t, vid))
    print("\nBIGGEST ORBITS")
    for bid, members in sorted(orbit.items(), key=lambda kv: -len(kv[1]))[:a.top // 2]:
        shown = ", ".join(f"{tag_str(t)}{surf[v]!r}" for t, v in members[:8])
        print(f"  {surf[bid]!r}: {shown}" + (f" (+{len(members) - 8} more)" if len(members) > 8 else ""))

    if scores:
        acc = sorted((scores[(b, v)], (b, v)) for v, (b, _t) in assign.items() if (b, v) in scores)
        print("\nAUDIT: lowest-similarity pairs ACCEPTED (look for false merges)")
        for sc, (b, v) in acc[:10]:
            print(f"  cos={sc:5.2f}  {surf[b]!r} -> {surf[v]!r}")
        rej = sorted(((s, k) for k, s in scores.items() if k not in pairs and not is_orth_only(res['sup'][k][1])),
                     reverse=True)
        print("AUDIT: highest-similarity suffix pairs REJECTED (look for missed merges)")
        for sc, (b, v) in rej[:10]:
            o = offs.get((b, v))
            print(f"  cos={sc:5.2f}  offset={'  n/a' if o is None else f'{o:5.2f}'}  {surf[b]!r} -> {surf[v]!r}")
    if offs:
        acc = sorted((offs[(b, v)], (b, v)) for v, (b, _t) in assign.items() if (b, v) in offs)
        print("AUDIT: lowest rule-consistency pairs ACCEPTED")
        for o, (b, v) in acc[:8]:
            print(f"  offset={o:5.2f}  {surf[b]!r} -> {surf[v]!r}")


def export_json(path, surf, assign):
    orbits = defaultdict(list)
    for vid, (bid, t) in assign.items():
        orbits[surf[bid]].append({"variant": surf[vid], "tag": list(t), "readable": tag_str(t)})
    with open(path, "w", encoding="utf-8") as f:
        json.dump({"tags": [list(t) for t in sorted({t for _b, t in assign.values()})], "orbits": orbits},
                  f, ensure_ascii=False, indent=1)
    print(f"\nwrote {path}")


def export_mapping(path, ctx, assign):
    """token id -> (base id, tag index) arrays for training code (see rov_ab_test.py). Tag 0 = identity."""
    import numpy as np
    if os.path.exists(path):
        print(f"warning: overwriting existing {path}")
    tags = sorted({t for _b, t in assign.values()})
    tag_idx = {t: i + 1 for i, t in enumerate(tags)}
    variant = np.array(sorted(assign), dtype=np.int64)
    np.savez(path, vocab_size=ctx["vocab_size"], variant=variant,
             base=np.array([assign[v][0] for v in variant], dtype=np.int64),
             tag=np.array([tag_idx[assign[v][1]] for v in variant], dtype=np.int64),
             tag_names=np.array(["[identity]"] + [tag_str(t) for t in tags]))
    print(f"wrote {path}  ({len(variant):,} variants, {len(tags) + 1} tags incl. identity)")


# --------------------------------------------------------------------------- #
# CLI
# --------------------------------------------------------------------------- #
def build_parser():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    src = p.add_argument_group("tokenizer (pick one)")
    src.add_argument("--demo", action="store_true", help="download ~12 MB of public-domain English, train a small BPE")
    src.add_argument("--tokenizer", help="path to a Hugging Face tokenizer.json")
    src.add_argument("--hf-model", help="Hugging Face model id whose tokenizer to load, e.g. gpt2")
    src.add_argument("--vocab-txt", help="plain text file, one token per line (no gate possible)")
    p.add_argument("--corpus", help="plain-text sample of the training domain (corpus gate, traffic stats, round-trip test)")
    p.add_argument("--demo-vocab-size", type=int, default=16000)
    p.add_argument("--demo-glove", action="store_true", help="download GloVe-50 (69 MB) and use it as --word-vectors")
    e = p.add_argument_group("embedding gate (any one)")
    e.add_argument("--embed-model", help="HF model id; uses its input-embedding matrix (tokenizer must match)")
    e.add_argument("--embed-file", help="[vocab, dim] .npy, or .safetensors (+ --embed-key)")
    e.add_argument("--embed-key", help="tensor name inside the safetensors file (auto-detected otherwise)")
    e.add_argument("--word-vectors", help="GloVe/word2vec text file (.txt/.gz); judges word-initial tokens only")
    e.add_argument("--emb-null-quantile", type=float, default=0.98, help="similarity must beat this quantile of random pairs")
    e.add_argument("--emb-strong-quantile", type=float, default=None,
                   help="pairs above this random-pair quantile skip the consistency test (e.g. 0.999)")
    e.add_argument("--emb-theta", type=float, help="override the calibrated similarity threshold")
    e.add_argument("--off-null-quantile", type=float, default=0.80, help="offset consistency must beat this null quantile")
    e.add_argument("--offset-theta", type=float, help="override the calibrated offset-consistency threshold")
    e.add_argument("--offset-min-group", type=int, default=4, help="pairs a rule needs before its offsets are tested")
    e.add_argument("--no-offset", action="store_true", help="similarity only, skip the consistency test")
    p.add_argument("--gate", choices=["auto", "corpus", "embeddings", "both", "none"], default="auto",
                   help="auto = embeddings if supplied, else corpus; 'both' = a pair must pass every gate that can judge it")
    p.add_argument("--gate-unknown", choices=["accept", "reject"], default="reject",
                   help="what to do with pairs that no gate could judge")
    m = p.add_argument_group("mining")
    m.add_argument("--min-stem", type=int, default=3, help="shortest shared stem (blocks be->bed)")
    m.add_argument("--min-suffix-stem", type=int, default=4,
                   help="suffix-edit rules need lowercase stems at least this long (blocks 'mar'->'marg')")
    m.add_argument("--max-suffix", type=int, default=4, help="longest suffix on either side of an edit")
    m.add_argument("--max-group", type=int, default=120, help="skip stems shared by more tokens than this")
    m.add_argument("--max-changes", type=int, default=2, help="max of {space, case, suffix} changed per rule")
    m.add_argument("--allow-composite-suffix", action="store_true",
                   help="allow a suffix edit together with a space/case change in one rule (mostly false merges)")
    m.add_argument("--families", choices=["all", "suffix", "orth", "case", "space", "cap"], default="all",
                   help="rule families allowed to merge: suffix edits; orth = space/case-only, split into "
                        "'case' (' the'/' The'), 'cap' (only [Cap]/[UPPER], no [lower]) and 'space' (' and'/'and'); or all")
    m.add_argument("--tau", type=int, default=8, help="min pairs a suffix edit needs to count as a rule")
    g = p.add_argument_group("corpus gate")
    g.add_argument("--theta", type=float, default=0.30, help="min cosine for suffix-edit pairs")
    g.add_argument("--theta-orth", type=float, default=0.30, help="min cosine for space/case-only pairs")
    g.add_argument("--gate-window", type=int, default=2)
    g.add_argument("--gate-dims", type=int, default=1500)
    g.add_argument("--gate-min-count", type=int, default=5)
    o = p.add_argument_group("output")
    o.add_argument("--d-model", type=int, default=4096)
    o.add_argument("--top", type=int, default=20)
    o.add_argument("--out", help="write orbits to this JSON file")
    o.add_argument("--save-mapping", help="write a .npz token->(base, tag) mapping for rov_ab_test.py")
    return p


def ensure_demo_corpus(path: str):
    if os.path.exists(path):
        return
    print(f"downloading ~12 MB of public-domain English (NLTK Gutenberg sample) -> {path}")
    zpath, _ = urllib.request.urlretrieve(DEMO_CORPUS_URL)
    with zipfile.ZipFile(zpath) as z, open(path, "w", encoding="utf-8") as out:
        for name in z.namelist():
            if name.endswith(".txt"):
                out.write(z.read(name).decode("latin-1").replace("\r", "") + "\n")


def main():
    p = build_parser()
    a = p.parse_args()
    if a.demo_glove and not a.word_vectors:
        a.word_vectors = "glove-wiki-gigaword-50.gz"
        if not os.path.exists(a.word_vectors):
            print("downloading GloVe-50 (69 MB) ...")
            urllib.request.urlretrieve(GLOVE_URL, a.word_vectors)
    if a.demo:
        a.corpus = a.corpus or "demo_corpus.txt"
        ensure_demo_corpus(a.corpus)
        print(f"training demo BPE (vocab {a.demo_vocab_size}) ...")
        tok = train_demo_tokenizer(a.corpus, a.demo_vocab_size)
        tok.save("demo_tokenizer.json")
        print("saved demo_tokenizer.json (try: --tokenizer demo_tokenizer.json --corpus demo_corpus.txt)")
        vocab = tok.get_vocab(with_added_tokens=True)
    elif a.tokenizer or a.hf_model or a.vocab_txt:
        tok, vocab = load_tokenizer(a)
    else:
        p.error("pick a tokenizer: --demo, --tokenizer, --hf-model or --vocab-txt")
    if tok is None and a.corpus:
        print("note: --corpus is ignored with --vocab-txt (a tokenizer is needed to encode it)")
        a.corpus = None
    if a.embed_model and not (a.hf_model or a.tokenizer or a.demo):
        p.error("--embed-model needs the matching tokenizer via --hf-model or --tokenizer")

    ctx = prepare(vocab, tok, a)
    print(f"vocab entries: {len(vocab):,}   token style: {ctx['style']}")
    res = analyse(ctx, a)
    ok, msg = verify_lossless(res, ctx, a)
    report(a, ctx, res, msg)
    if a.out:
        export_json(a.out, ctx["surf"], res["assign"])
    if a.save_mapping:
        export_mapping(a.save_mapping, ctx, res["assign"])
    sys.exit(0 if ok else 1)


if __name__ == "__main__":
    main()
