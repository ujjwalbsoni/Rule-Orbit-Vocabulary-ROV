#!/usr/bin/env python3
"""
Corpus-derived stand-in embeddings for the embedding gate (PPMI context vectors + truncated SVD).

Use this only when you cannot load a real model's embedding matrix. Unlike GloVe it covers every
token of YOUR tokenizer (fragments, ' the' vs 'the', ' The'), so the space/case family can be
judged. It is a weaker signal than a pretrained language model's embeddings, and rare tokens
(< --min-count occurrences) get random vectors so that they are rejected rather than merged.

  python make_proxy_embeddings.py --tokenizer tok.json --corpus text.txt --out proxy.npy
"""
import argparse
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--tokenizer", required=True)
    p.add_argument("--corpus", required=True)
    p.add_argument("--out", required=True, help=".npy, shape [vocab, dim], row i = token id i")
    p.add_argument("--dim", type=int, default=100)
    p.add_argument("--ctx-dims", type=int, default=3000, help="number of frequent tokens used as context features")
    p.add_argument("--window", type=int, default=2)
    p.add_argument("--min-count", type=int, default=5)
    a = p.parse_args()

    import numpy as np
    from scipy.sparse import coo_matrix
    from scipy.sparse.linalg import svds
    from tokenizers import Tokenizer

    import rule_orbit_vocab as rov

    tok = Tokenizer.from_file(a.tokenizer)
    V = max(tok.get_vocab(with_added_tokens=True).values()) + 1
    print("encoding corpus ...", flush=True)
    ids, counts, total = rov.encode_corpus(tok, a.corpus, V)
    ctx_ids = np.argsort(-counts)[:a.ctx_dims]
    ctx_index = np.full(V, -1, dtype=np.int64)
    ctx_index[ctx_ids] = np.arange(len(ctx_ids))

    print(f"counting co-occurrences over {total:,} tokens ...", flush=True)
    rows, cols = [], []
    for d in range(1, a.window + 1):
        left, right = ids[:-d], ids[d:]
        for centre, context in ((left, right), (right, left)):
            ok = (centre >= 0) & (context >= 0)
            xi = ctx_index[context[ok]]
            m = xi >= 0
            rows.append(centre[ok][m])
            cols.append(xi[m])
    rows, cols = np.concatenate(rows), np.concatenate(cols)
    M = coo_matrix((np.ones(rows.size, dtype=np.float32), (rows, cols)), shape=(V, len(ctx_ids))).tocsr()

    pc = counts[ctx_ids].astype(np.float64) ** 0.75
    pc /= pc.sum()
    row_tot = np.asarray(M.sum(1)).ravel()
    nnz_row = np.repeat(np.arange(V), np.diff(M.indptr))
    pmi = np.log((M.data / np.maximum(row_tot[nnz_row], 1)) / pc[M.indices])
    M.data = np.maximum(pmi, 0).astype(np.float32)

    print(f"truncated SVD to {a.dim} dims ...", flush=True)
    U, S, _ = svds(M, k=a.dim)
    E = (U * np.sqrt(S)).astype(np.float32)
    rng = np.random.default_rng(0)
    rare = counts < a.min_count
    E[rare] = rng.normal(scale=E[~rare].std(), size=(int(rare.sum()), a.dim)).astype(np.float32)
    np.save(a.out, E)
    print(f"wrote {a.out}  {E.shape}   ({int(rare.sum()):,} rare tokens got random vectors)")


if __name__ == "__main__":
    main()
