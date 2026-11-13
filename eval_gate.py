#!/usr/bin/env python3
"""
Scores the ROV gates against WordNet.

Ground truth: a candidate suffix-edit pair (' looked' / ' look') is TRUE if WordNet links the two
words by inflection (morphy) or derivation (derivationally-related forms, pertainyms), FALSE if both
words are in WordNet and nothing links them (' thought' / ' though'). Pairs with a word WordNet
does not know are skipped. Only word-initial, lowercase, suffix-only pairs are scored - the family
where a gate has to do real work.

  pip install nltk
  export NLTK_DATA=/path/to/nltk_data      # must contain corpora/wordnet

  python eval_gate.py --tokenizer demo_tokenizer.json --corpus demo_corpus.txt --word-vectors glove-wiki-gigaword-50.gz
"""
import copy
import functools
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import rule_orbit_vocab as rov  # noqa: E402

from nltk.corpus import wordnet as wn  # noqa: E402


@functools.lru_cache(maxsize=None)
def known(w):
    return bool(wn.synsets(w))


@functools.lru_cache(maxsize=None)
def forms(w):
    out = {w}  # morphy variants of w, including w itself (so siblings like looked/looking share 'look')
    for p in "nvar":
        out |= set(wn._morphy(w, p))
    for syn in wn.synsets(w):
        for lem in syn.lemmas():
            if lem.name().lower() == w:
                out |= {r.name().lower() for r in lem.derivationally_related_forms()}
                out |= {r.name().lower() for r in lem.pertainyms()}
    return out


def related(a, b):
    """True if the two words share a lemma / derivational link (siblings count: same family)."""
    return bool(forms(a) & forms(b)) or b in forms(a) or a in forms(b)


def main():
    p = rov.build_parser()
    p.add_argument("--pick", help="config name whose surviving false merges to list")
    a = p.parse_args()
    tok, vocab = rov.load_tokenizer(a)
    ctx = rov.prepare(vocab, tok, a)
    raw = rov.mine_pairs(ctx["dec"], ctx["surf"], a)
    sup = rov.support_filter(raw, a.tau)

    labels = {}
    for (b, v), (_c, tag) in sup.items():
        if rov.is_orth_only(tag):
            continue
        db, dv = ctx["dec"][b], ctx["dec"][v]
        if not (db[0] and dv[0] and db[1] == "lower" and dv[1] == "lower"):
            continue
        if known(db[2]) and known(dv[2]):
            labels[(b, v)] = related(db[2], dv[2])
    n_true = sum(labels.values())
    print(f"labelled suffix pairs: {len(labels):,}   related: {n_true:,}   unrelated: {len(labels) - n_true:,}")
    print(f"(if you accepted every candidate: precision {n_true / len(labels):.1%}, recall 100%)\n")

    def cfg(**kw):
        c = copy.copy(a)
        for k, v in kw.items():
            setattr(c, k, v)
        return c

    glove = a.word_vectors
    has_emb = bool(glove or a.embed_model or a.embed_file)

    def emb(q, off=None, strong=None):
        """every threshold explicit, so the table never depends on CLI defaults"""
        return cfg(gate="embeddings", emb_null_quantile=q, no_offset=off is None,
                   off_null_quantile=off if off is not None else 0.8, emb_strong_quantile=strong)

    configs = [("no gate", cfg(gate="none"))]
    if a.corpus:
        configs += [("corpus PPMI cos>=0.30", cfg(gate="corpus"))]
    if has_emb:
        configs += [
            ("emb sim only        p95", emb(0.95)),
            ("emb sim only        p98", emb(0.98)),
            ("emb sim only        p99", emb(0.99)),
            ("emb sim only        p99.9", emb(0.999)),
            ("emb p95 + offset p50", emb(0.95, 0.50)),
            ("emb p95 + offset p80", emb(0.95, 0.80)),
            ("emb p95 + offset p95", emb(0.95, 0.95)),
            ("emb p98 + offset p80", emb(0.98, 0.80)),
            ("emb p99 + offset p95", emb(0.99, 0.95)),
            ("emb p95 + offset p80 + strong p99.9", emb(0.95, 0.80, 0.999)),
            ("emb p95 + offset p80 + strong p99", emb(0.95, 0.80, 0.99)),
            ("emb p98 + offset p80 + strong p99.9", emb(0.98, 0.80, 0.999)),
            ("emb p98 + offset p80 + strong p99", emb(0.98, 0.80, 0.99)),
            ("emb p99 + offset p95 + strong p99.9", emb(0.99, 0.95, 0.999)),
        ]
    if a.corpus and has_emb:
        configs += [("corpus AND emb p98+offset p80", cfg(gate="both", emb_null_quantile=0.98, off_null_quantile=0.8,
                                                        no_offset=False, emb_strong_quantile=None))]

    print(f"{'config':<44}{'accepted':>9}{'TP':>6}{'FP':>6}{'precision':>11}{'recall':>8}{'F1':>7}")
    worst = {}
    for name, c in configs:
        res = rov.analyse(ctx, c, mined=(raw, sup))
        acc = [k for k in labels if k in res["pairs"]]
        tp = sum(labels[k] for k in acc)
        fp = len(acc) - tp
        prec = tp / len(acc) if acc else float("nan")
        rec = tp / n_true
        f1 = 2 * prec * rec / (prec + rec) if acc and tp else 0.0
        print(f"{name:<44}{len(acc):>9,}{tp:>6,}{fp:>6,}{prec:>10.1%}{rec:>8.1%}{f1:>7.2f}")
        worst[name] = [k for k in acc if not labels[k]]

    pick = a.pick if hasattr(a, "pick") and a.pick else "emb p98 + offset p80 + strong p99.9"
    if pick in worst:
        print(f"\nfalse merges that survive '{pick}' (first 15):")
        for b, v in worst[pick][:15]:
            print(f"   {ctx['surf'][b]!r} -> {ctx['surf'][v]!r}")


if __name__ == "__main__":
    main()
