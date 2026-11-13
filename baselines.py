#!/usr/bin/env python3
"""
Baselines for Rule-Orbit Vocabulary (ROV)
=========================================
Is folding variants into base+tag better than the obvious ways of shrinking a vocabulary?
Every scheme below is trained with the SAME transformer, optimiser, number of steps and tokens per
step, on the same text, and scored in BITS PER BYTE of the same held-out text (bits per token is not
comparable when tokenisations differ).

  flat        the unmodified tokenizer, one embedding row per token         (reference)
  rov         same tokenisation, tokens in the mapping stored as base + tag (this project)
  prune       tokens in the mapping removed from the vocabulary; wherever one would be produced it is
              re-split into the merge parts it was built from (sequences get longer)
  plain       a fresh BPE trained on this corpus at the pruned vocabulary size (control for the next row)
  casemarker  a fresh BPE of the same size trained on lowercased text, with a <cap> / <upper> marker
              token inserted before capitalised / ALL-CAPS words (reversible; sequences get longer)

Comparisons that mean something:  rov vs prune (same tokenizer, same vocabulary size),
rov vs flat, and casemarker vs plain (isolates the marker idea from "a freshly trained vocabulary").

  python baselines.py --tokenizer tokenizer.json --corpus text.txt --mapping map_cap.npz --workdir bl

Compute is matched (same steps, same tokens per step). Schemes with longer sequences therefore see
less text per step; the table reports the bytes of training text each one consumed.
Finished schemes are cached in --workdir and reused when the settings match.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import re
import sys
import time
from array import array

import numpy as np
import torch

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import rov_ab_test as ab  # noqa: E402
from rule_orbit_vocab import B2U_INV  # noqa: E402

CAP, UP = "\ue000", "\ue001"           # private-use characters, never present in ordinary text
WORD = re.compile(r"[A-Za-z]+")
UNMARK = re.compile("([\ue000\ue001])( ?)([a-z]+)")
SCHEMES = ["flat", "rov", "prune", "plain", "casemarker"]


# --------------------------------------------------------------------------- #
# case-marker transform (exactly reversible)
# --------------------------------------------------------------------------- #
def to_marked(text: str) -> str:
    """'saw The cat' -> 'saw<CAP> the cat'. The marker goes before the word's leading space, so the
    word keeps its normal ' the' token. Mixed-case words (iPhone, McDonald) are left unchanged."""
    out, last = [], 0
    for m in WORD.finditer(text):
        w = m.group()
        if len(w) == 1:
            kind = CAP if w.isupper() else None
        elif w.isupper():
            kind = UP
        elif w[0].isupper() and w[1:].islower():
            kind = CAP
        else:
            kind = None
        if kind is None:
            continue
        s = m.start()
        ins = s - 1 if s > 0 and text[s - 1] == " " else s
        out += [text[last:ins], kind, text[ins:s], w.lower()]
        last = m.end()
    out.append(text[last:])
    return "".join(out)


def from_marked(text: str) -> str:
    return UNMARK.sub(lambda m: m.group(2) + (m.group(3).capitalize() if m.group(1) == CAP
                                              else m.group(3).upper()), text)


# --------------------------------------------------------------------------- #
# helpers
# --------------------------------------------------------------------------- #
def id_tables(vocab: dict):
    """raw bytes of every token, and how many bytes of ORIGINAL text it stands for (markers count 0)."""
    V = max(vocab.values()) + 1
    raw, orig = [b""] * V, np.zeros(V, dtype=np.int64)
    for t, i in vocab.items():
        try:
            b = bytes(B2U_INV[c] for c in t)
        except KeyError:
            b = t.encode("utf-8")
        raw[i] = b
        orig[i] = len(b) - 3 * (t.count(CAP) + t.count(UP))
    return raw, orig


def read_lines(path, max_mb=0.0):
    with open(path, encoding="utf-8", errors="ignore") as f:
        lines = [ln + "\n" for ln in f.read().split("\n") if ln.strip()]
    if max_mb:
        out, size = [], 0
        for ln in lines:
            out.append(ln)
            size += len(ln.encode("utf-8"))
            if size >= max_mb * 1e6:
                break
        lines = out
    return lines


def split_text(lines, chunk_bytes, val_bytes, seed):
    """Hold out whole ~chunk_bytes passages (contiguous text), identical for every scheme."""
    chunks, cur, size = [], [], 0
    for ln in lines:
        cur.append(ln)
        size += len(ln.encode("utf-8"))
        if size >= chunk_bytes:
            chunks.append(cur)
            cur, size = [], 0
    if cur:
        chunks.append(cur)
    val, vb = set(), 0
    for ci in np.random.default_rng(seed).permutation(len(chunks)):
        if vb >= val_bytes:
            break
        val.add(int(ci))
        vb += sum(len(x.encode("utf-8")) for x in chunks[ci])
    train_lines = [x for i, c in enumerate(chunks) if i not in val for x in c]
    val_lines = [x for i, c in enumerate(chunks) if i in val for x in c]
    return train_lines, val_lines


def encode_lines(tok, lines, transform=None):
    ids = array("i")
    for s in range(0, len(lines), 5000):
        chunk = lines[s:s + 5000]
        if transform:
            chunk = [transform(x) for x in chunk]
        for enc in tok.encode_batch(chunk):
            ids.extend(enc.ids)
    return np.frombuffer(ids, dtype=np.int32).astype(np.int64)


def train_bpe(lines, vocab_size, special, path):
    from tokenizers import Tokenizer, decoders, models, pre_tokenizers, trainers
    if os.path.exists(path):
        cached = Tokenizer.from_file(path)
        if cached.get_vocab_size() == vocab_size:
            return cached
        print(f"   {path} has {cached.get_vocab_size():,} tokens, need {vocab_size:,}: retraining", flush=True)
    tok = Tokenizer(models.BPE())
    tok.pre_tokenizer = pre_tokenizers.ByteLevel(add_prefix_space=False)
    tok.decoder = decoders.ByteLevel()
    trainer = trainers.BpeTrainer(vocab_size=vocab_size, special_tokens=list(special), show_progress=False,
                                  initial_alphabet=pre_tokenizers.ByteLevel.alphabet())
    tok.train_from_iterator(lines, trainer)
    tok.save(path)
    return tok


def bpe_parts(tok_path):
    """token string -> the two strings it was merged from."""
    merges = json.load(open(tok_path, encoding="utf-8"))["model"]["merges"]
    parts = {}
    for m in merges:
        a, b = m.split(" ") if isinstance(m, str) else m
        parts.setdefault(a + b, (a, b))
    return parts


def prune_stream(ids, removed, expand):
    """Replace every removed id by the ids it was built from (recursively)."""
    removed_mask = np.zeros(int(ids.max()) + 1 if len(ids) else 1, dtype=bool)
    removed_mask[[i for i in removed if i < len(removed_mask)]] = True
    pos = np.flatnonzero(removed_mask[ids])
    pieces, prev = [], 0
    for p in pos:
        pieces.append(ids[prev:p])
        pieces.append(np.asarray(expand(int(ids[p])), dtype=np.int64))
        prev = p + 1
    pieces.append(ids[prev:])
    return np.concatenate(pieces)


def blocks(stream, ctx, stride):
    n = (len(stream) - 1) // stride if stride == ctx else len(stream) // (ctx + 1)
    if stride == ctx:
        idx = np.arange(n)[:, None] * ctx + np.arange(ctx + 1)[None, :]
        return stream[idx]
    return stream[:n * (ctx + 1)].reshape(n, ctx + 1)


def bpb_with_se(bits, nbytes, reps=300, seed=0):
    rng = np.random.default_rng(seed)
    n = len(bits)
    draws = [bits[i].sum() / nbytes[i].sum() for i in (rng.integers(0, n, n) for _ in range(reps))]
    return float(bits.sum() / nbytes.sum()), float(np.std(draws))


# --------------------------------------------------------------------------- #
def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--tokenizer", required=True, help="the tokenizer.json the mapping was built for")
    p.add_argument("--corpus", required=True)
    p.add_argument("--mapping", required=True, help=".npz from rule_orbit_vocab.py --save-mapping")
    p.add_argument("--schemes", nargs="+", default=SCHEMES, choices=SCHEMES)
    p.add_argument("--workdir", default="baselines")
    p.add_argument("--vocab-size", type=int, default=0, help="size for plain/casemarker (default: the pruned size)")
    p.add_argument("--val-bytes", type=int, default=800_000, help="held-out text, in bytes")
    p.add_argument("--chunk-bytes", type=int, default=20_000)
    p.add_argument("--max-corpus-mb", type=float, default=0.0, help="use only the first N MB (quick tests)")
    p.add_argument("--tok-train-mb", type=float, default=40.0, help="text used to train plain/casemarker tokenizers")
    p.add_argument("--steps", type=int, default=2000)
    p.add_argument("--batch", type=int, default=24)
    p.add_argument("--ctx", type=int, default=96)
    p.add_argument("--d-model", type=int, default=512)
    p.add_argument("--n-layer", type=int, default=8)
    p.add_argument("--n-head", type=int, default=8)
    p.add_argument("--d-ff", type=int, default=2048)
    p.add_argument("--lr", type=float, default=1e-3)
    p.add_argument("--warmup", type=int, default=200)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--data-seed", type=int, default=0)
    p.add_argument("--eval-every", type=int, default=500)
    p.add_argument("--quick-val", type=int, default=64)
    p.add_argument("--threads", type=int, default=0)
    a = p.parse_args()

    dev = "cuda" if torch.cuda.is_available() else "cpu"
    if a.threads:
        torch.set_num_threads(a.threads)
    os.makedirs(a.workdir, exist_ok=True)
    W = lambda x: os.path.join(a.workdir, x)  # noqa: E731

    from tokenizers import Tokenizer
    gtok = Tokenizer.from_file(a.tokenizer)
    gvocab = gtok.get_vocab(with_added_tokens=True)
    V0 = max(gvocab.values()) + 1
    mapping = ab.load_mapping(a.mapping, V0)
    removed = set(int(i) for i in np.flatnonzero(mapping["is_variant"]) if i < V0)
    target_vocab = a.vocab_size or (V0 - len(removed))
    graw, gorig = id_tables(gvocab)

    lines = read_lines(a.corpus, a.max_corpus_mb)
    train_lines, val_lines = split_text(lines, a.chunk_bytes, a.val_bytes, a.data_seed)
    val_bytes = sum(len(x.encode("utf-8")) for x in val_lines)
    train_bytes = sum(len(x.encode("utf-8")) for x in train_lines)
    print(f"device {dev} | text: {train_bytes / 1e6:.1f} MB train, {val_bytes / 1e6:.2f} MB held out | "
          f"vocab {V0:,}: {len(removed):,} mapped tokens -> {V0 - len(removed):,} rows | "
          f"plain/casemarker vocab {target_vocab:,}", flush=True)

    streams = {}

    def gpt2_streams():
        if "gpt2" not in streams:
            streams["gpt2"] = (encode_lines(gtok, train_lines), encode_lines(gtok, val_lines))
        return streams["gpt2"]

    def build(scheme):
        """-> (vocab rows, train ids, val ids, original-bytes-per-id, mapping or None)"""
        if scheme in ("flat", "rov"):
            tr, va = gpt2_streams()
            return V0, tr, va, gorig, (mapping if scheme == "rov" else None)
        if scheme == "prune":
            tr, va = gpt2_streams()
            parts, memo = bpe_parts(a.tokenizer), {}
            inv = {i: t for t, i in gvocab.items()}

            def expand(i):
                if i not in removed:
                    return [i]
                if i not in memo:
                    t = inv[i]
                    if t not in parts:
                        raise SystemExit(f"token {t!r} is in the mapping but is not a BPE merge; cannot prune it")
                    x, y = parts[t]
                    memo[i] = expand(gvocab[x]) + expand(gvocab[y])
                return memo[i]

            tr2, va2 = prune_stream(tr, removed, expand), prune_stream(va, removed, expand)
            assert b"".join(graw[i] for i in va2) == b"".join(graw[i] for i in va), "pruned text differs from original"
            kept = np.ones(V0, dtype=bool)
            kept[list(removed)] = False
            new_id = np.cumsum(kept) - 1
            return int(kept.sum()), new_id[tr2], new_id[va2], gorig[kept], None
        # retrained tokenizers
        budget, size, sub = a.tok_train_mb * 1e6, 0, []
        for x in train_lines:
            sub.append(x)
            size += len(x)
            if size >= budget:
                break
        if scheme == "plain":
            tok = train_bpe(sub, target_vocab, [], W("tok_plain.json"))
            tf = None
        else:
            tok = train_bpe([to_marked(x) for x in sub], target_vocab, [CAP, UP], W("tok_casemarker.json"))
            tf = to_marked
        vocab = tok.get_vocab(with_added_tokens=True)
        raw, orig = id_tables(vocab)
        if scheme == "casemarker":
            bad = sum(from_marked(b"".join(raw[i] for i in tok.encode(to_marked(x)).ids).decode("utf-8", errors="replace"))
                      != x for x in val_lines[:300])
            print(f"   case-marker round trip on 300 held-out lines: {'OK' if bad == 0 else f'{bad} FAILURES'}", flush=True)
            if bad:
                raise SystemExit("case-marker transform is not reversible on this text")
        return (max(vocab.values()) + 1, encode_lines(tok, train_lines, tf), encode_lines(tok, val_lines, tf),
                orig, None)

    results = {}
    u = np.random.default_rng(a.seed).random((a.steps, a.batch))  # same relative positions for every scheme
    for scheme in a.schemes:
        print(f"\n######## {scheme}", flush=True)
        sig = json.dumps({"scheme": scheme, "steps": a.steps, "batch": a.batch, "ctx": a.ctx, "d": a.d_model,
                          "L": a.n_layer, "h": a.n_head, "ff": a.d_ff, "lr": a.lr, "warmup": a.warmup, "seed": a.seed,
                          "data_seed": a.data_seed, "val_bytes": a.val_bytes, "chunk": a.chunk_bytes,
                          "max_mb": a.max_corpus_mb, "tok_mb": a.tok_train_mb, "vocab": target_vocab,
                          "map": hashlib.md5(a.mapping.encode() + mapping["is_variant"].tobytes()).hexdigest()[:8],
                          "corpus": os.path.getsize(a.corpus)}, sort_keys=True)
        ck = W(f"res_{scheme}.npz")
        if os.path.exists(ck):
            c = np.load(ck)
            if "sig" in c.files and str(c["sig"]) == sig:
                results[scheme] = {k: c[k] for k in c.files if k != "sig"}
                print(f"loaded finished run from {ck}", flush=True)
                continue
            print(f"{ck} was made with different settings -> retraining", flush=True)

        V, tr, va, orig, mp = build(scheme)
        tot = int(orig[va].sum())
        if tot != val_bytes:
            print(f"!! held-out bytes seen by this scheme ({tot:,}) != text bytes ({val_bytes:,}): tokenisation is not "
                  f"byte-exact; bits per byte will be biased", flush=True)
        train_blocks, val_blocks = blocks(tr, a.ctx, a.ctx + 1), blocks(va, a.ctx, a.ctx)
        batches = (u * len(train_blocks)).astype(np.int64)
        res = ab.run_one("rov" if scheme == "rov" else "flat", a, V, mp, train_blocks, val_blocks, 0.0, dev, batches)
        bits = res["nll"].sum(1) / math.log(2)
        nbytes = orig[val_blocks[:, 1:]].sum(1).astype(np.float64)
        bpt_train = float(orig[tr].sum() / len(tr))
        out = dict(bits=bits, nbytes=nbytes, params=res["params"], emb=res["emb_params"], seconds=res["seconds"],
                   rows=V, val_tokens=len(va), train_tokens=len(tr), bytes_per_token=bpt_train,
                   train_mb_seen=a.steps * a.batch * a.ctx * bpt_train / 1e6)
        np.savez(ck, sig=sig, **out)
        results[scheme] = out

    # ------------------------------------------------------------------ report
    print("\n" + "=" * 100)
    print(f"RESULTS  (held-out text {val_bytes / 1e6:.2f} MB; {a.steps} steps x {a.batch} x {a.ctx} tokens; seed {a.seed})")
    print("=" * 100)
    print(f"{'scheme':<12}{'vocab size':>11}{'emb+head':>10}{'params':>9}{'bytes/token':>12}{'train MB seen':>14}"
          f"{'bits/token':>12}{'bits/BYTE':>14}{'vs flat':>10}")
    flat = results.get("flat")
    se = {}
    for s in a.schemes:
        r = results[s]
        bpb, se[s] = bpb_with_se(r["bits"], r["nbytes"])
        r["bpb"] = bpb
        bpt = float(r["bits"].sum() / (len(r["bits"]) * a.ctx))
        diff = f"{bpb - flat['bpb']:+.4f}" if flat is not None and s != "flat" else ""
        print(f"{s:<12}{int(r['rows']):>11,}{int(r['emb']) / 1e6:>9.2f}M{int(r['params']) / 1e6:>8.2f}M"
              f"{float(r['bytes_per_token']):>12.2f}{float(r['train_mb_seen']):>14.1f}{bpt:>12.3f}"
              f"{bpb:>9.4f} ±{se[s]:.4f}{diff:>10}")
    print("\n(± = bootstrap standard error over held-out windows, per scheme alone. The comparisons below are PAIRED"
          " over equal byte-range buckets of the same text, so their CIs are much tighter than these.)")

    # Paired comparison: cut the held-out text into fixed byte-range buckets (same ranges for every scheme, even
    # though token boundaries differ), then bootstrap whole buckets jointly for both schemes being compared.
    BUCKET = 8000
    nb = max(1, int(val_bytes // BUCKET))
    bucket = {}
    for s in a.schemes:
        r = results[s]
        start = np.cumsum(r["nbytes"]) - r["nbytes"]          # byte offset where each window begins
        idx = np.minimum((start // BUCKET).astype(np.int64), nb - 1)
        bucket[s] = (np.bincount(idx, weights=r["bits"], minlength=nb), np.bincount(idx, weights=r["nbytes"], minlength=nb))
    boot = np.random.default_rng(1).integers(0, nb, (2000, nb))

    def pair(x, y, why):
        if x in results and y in results:
            bx, nx = bucket[x]
            by, ny = bucket[y]
            dx = bx.sum() / nx.sum() - by.sum() / ny.sum()
            draws = (bx[boot].sum(1) / nx[boot].sum(1)) - (by[boot].sum(1) / ny[boot].sum(1))
            lo, hi = np.percentile(draws, [2.5, 97.5])
            verdict = f"{x} better" if hi < 0 else f"{y} better" if lo > 0 else "no clear difference"
            print(f"  {x:<11} minus {y:<11}{dx:+.4f}  95% CI [{lo:+.4f}, {hi:+.4f}] bits/byte   {verdict}   ({why})"
                  f"   [{nb} paired buckets]")

    print("\nthe comparisons that matter (negative = first is better):")
    pair("rov", "prune", "same tokenizer, same number of embedding rows")
    pair("rov", "flat", "the folded model vs the unmodified vocabulary")
    pair("casemarker", "plain", "marker idea vs a freshly trained vocabulary of the same size")
    pair("prune", "flat", "what plain pruning costs")
    pair("rov", "plain", "folding vs a freshly trained vocabulary of ~the same size")
    pair("rov", "casemarker", "folding vs the case-marker scheme")
    summ = {}
    for s, r in results.items():
        summ[s] = {k: (float(v) if np.ndim(v) == 0 else None) for k, v in r.items()}
        summ[s]["bpb_se"] = se[s]
        summ[s]["bucket_bits"] = bucket[s][0].tolist()     # per-bucket data so comparisons can be redone offline
        summ[s]["bucket_bytes"] = bucket[s][1].tolist()
    json.dump(summ, open(W("summary.json"), "w"), indent=1)
    print(f"\nsaved {W('summary.json')}")


if __name__ == "__main__":
    main()
