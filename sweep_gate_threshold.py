#!/usr/bin/env python3
"""
Threshold sweep for the embedding gate, scored against WordNet.

The default gate sets its cosine floor at a random-pair percentile. On GPT-2 that percentile is
~0.11, close to chance, so unrelated words like ' Matt'/' Match' get through. This script instead
sweeps fixed cosine floors and reports precision and recall for each, so you can choose a floor
from labelled data rather than from random pairs.

Labels: a suffix pair (' looked', ' look') is related if WordNet links the two words (shared
lemma or derivational relation). Unrelated = both words are in WordNet and nothing links them.
Words WordNet does not know are skipped.

  pip install nltk tokenizers numpy scipy safetensors huggingface_hub
  python -m nltk.downloader wordnet

  python sweep_gate_threshold.py --hf-model gpt2 --embed-model gpt2 --corpus your_text.txt
  python sweep_gate_threshold.py --tokenizer tok.json --embed-file wte.npy --corpus your_text.txt

Output: a table of cosine floors and the precision/recall at each, plus a suggested floor (the lowest
floor whose precision reaches --target-precision while keeping recall above --min-recall).
"""
import argparse
import functools
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import rule_orbit_vocab as rov  # noqa: E402


def wordnet():
    try:
        from nltk.corpus import wordnet as wn
        wn.synsets("dog")
        return wn
    except LookupError:
        sys.exit("WordNet not found. Run: python -m nltk.downloader wordnet")


WN = None


@functools.lru_cache(maxsize=None)
def known(w):
    return bool(WN.synsets(w))


@functools.lru_cache(maxsize=None)
def forms(w):
    out = {w}
    for p in "nvar":
        out |= set(WN._morphy(w, p))
    for syn in WN.synsets(w):
        for lem in syn.lemmas():
            if lem.name().lower() == w:
                out |= {r.name().lower() for r in lem.derivationally_related_forms()}
                out |= {r.name().lower() for r in lem.pertainyms()}
    return out


def related(a, b):
    return bool(forms(a) & forms(b)) or b in forms(a) or a in forms(b)


def main():
    global WN
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    src = p.add_argument_group("tokenizer and embeddings")
    src.add_argument("--hf-model", help="model id, e.g. gpt2 (its tokenizer and embedding matrix)")
    src.add_argument("--tokenizer", help="tokenizer.json (use with --embed-file)")
    src.add_argument("--embed-model", help="model id whose input embeddings to use (tokenizer must match)")
    src.add_argument("--embed-file", help="[vocab, dim] .npy or .safetensors")
    src.add_argument("--embed-key", help="tensor name in a safetensors file")
    p.add_argument("--corpus", help="text file (optional; only used if you add corpus-based gates later)")
    p.add_argument("--min-suffix-stem", type=int, default=4)
    p.add_argument("--floors", default="0.10,0.15,0.20,0.25,0.30,0.35,0.40,0.45,0.50",
                   help="comma-separated cosine floors to test")
    p.add_argument("--target-precision", type=float, default=0.90)
    p.add_argument("--min-recall", type=float, default=0.30)
    p.add_argument("--show-errors", type=int, default=12, help="false merges to print at the suggested floor")
    a = p.parse_args()

    WN = wordnet()
    if a.hf_model:
        a.tokenizer = a.tokenizer or None
    from tokenizers import Tokenizer
    if a.hf_model:
        tok = Tokenizer.from_pretrained(a.hf_model)
        vocab = tok.get_vocab(with_added_tokens=True)
    else:
        tok = Tokenizer.from_file(a.tokenizer)
        vocab = tok.get_vocab(with_added_tokens=True)

    if not (a.embed_model or a.embed_file or a.hf_model):
        p.error("give --hf-model, --embed-model or --embed-file")
    src_args = {"embed_model": a.embed_model or a.hf_model, "embed_file": a.embed_file, "embed_key": a.embed_key,
                "word_vectors": None}
    ctx_args = type("A", (), dict(src_args, corpus=a.corpus))
    ctx = rov.prepare(vocab, tok, a)
    emb = rov.build_embedding_source(ctx_args, ctx)

    # labelled candidate pairs: word-initial, lowercase, suffix-only, both words in WordNet
    raw = rov.mine_pairs(ctx["dec"], ctx["surf"], type("A", (), {"min_stem": 3, "max_suffix": 4, "max_group": 120,
                                                              "max_changes": 2, "allow_composite_suffix": False,
                                                              "min_suffix_stem": a.min_suffix_stem})())
    sup = rov.support_filter(raw, 8)
    labels, scores = {}, {}
    for (b, v), (_c, tag) in sup.items():
        if rov.is_orth_only(tag):
            continue
        db, dv = ctx["dec"][b], ctx["dec"][v]
        if not (db[0] and dv[0] and db[1] == "lower" and dv[1] == "lower"):
            continue
        if b not in emb.rows or v not in emb.rows:
            continue
        if not (known(db[2]) and known(dv[2])):
            continue
        labels[(b, v)] = related(db[2], dv[2])
        scores[(b, v)] = float(emb.E[emb.rows[b]] @ emb.E[emb.rows[v]])

    n_true = sum(labels.values())
    n = len(labels)
    print(f"labelled suffix pairs: {n:,}   related: {n_true:,}   unrelated: {n - n_true:,}   "
          f"(accept-all precision {n_true / max(n, 1):.1%})\n")
    print(f"{'floor':>7}{'accepted':>10}{'TP':>8}{'FP':>8}{'precision':>11}{'recall':>9}{'F1':>7}")
    best = None
    rows = []
    for f in [float(x) for x in a.floors.split(",")]:
        acc = [k for k in labels if scores[k] >= f]
        tp = sum(labels[k] for k in acc)
        fp = len(acc) - tp
        prec = tp / len(acc) if acc else float("nan")
        rec = tp / max(n_true, 1)
        f1 = 2 * prec * rec / (prec + rec) if acc and tp else 0.0
        rows.append((f, prec, rec))
        print(f"{f:>7.2f}{len(acc):>10,}{tp:>8,}{fp:>8,}{prec:>10.1%}{rec:>9.1%}{f1:>7.2f}")
        if acc and prec >= a.target_precision and rec >= a.min_recall:
            best = best or (f, prec, rec)

    print()
    if best:
        f, prec, rec = best
        print(f"suggested floor: {f:.2f}  (precision {prec:.1%}, recall {rec:.1%})")
        print(f"  pass to the miner:  --emb-theta {f:.2f} --no-offset   (or --gate embeddings with --emb-theta)")
        errs = sorted((k for k in labels if scores[k] >= f and not labels[k]), key=lambda k: -scores[k])
        print(f"\nfalse merges still accepted at {f:.2f} (highest similarity first):")
        for b, v in errs[:a.show_errors]:
            print(f"   cos={scores[(b, v)]:.2f}  {ctx['surf'][b]!r} -> {ctx['surf'][v]!r}")
    else:
        print(f"no floor reached precision {a.target_precision:.0%} with recall >= {a.min_recall:.0%}.")
        print("Lower --target-precision, or accept the trade-off in the table above.")


if __name__ == "__main__":
    main()
