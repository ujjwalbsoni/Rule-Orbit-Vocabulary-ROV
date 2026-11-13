#!/usr/bin/env python3
"""
A/B test for Rule-Orbit Vocabulary
==================================
Trains small GPT-style language models on the SAME tokenised text and the SAME batches:

  flat         ordinary model: V embedding rows, softmax over V tokens
  rov          input  = E_base[base] + E_tag[tag]
               output = p(base | h) * p(tag | base, h)      (tag softmax masked to the tags that
                        exist for that base, so it is a proper distribution over the V tokens)
  rov-matched  same as rov, with the feed-forward layers widened until total parameters match flat

Because both factorisations are exact distributions over the same V tokens, NLL per token is directly
comparable. Reported: bits/token, bits/byte, and the loss on variant tokens and on rare tokens
(where sharing statistical strength with the base is supposed to help).

  python rov_ab_test.py --tokenizer tok.json --corpus text.txt --mapping map_gated.npz
  python rov_ab_test.py ... --runs flat rov rov-matched --steps 3000 --d-model 256 --n-layer 4

Make the mapping with:  rule_orbit_vocab.py ... --save-mapping map_gated.npz
Needs: torch, numpy, tokenizers.   Uses CUDA automatically if available.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import time

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F


# --------------------------------------------------------------------------- #
# data
# --------------------------------------------------------------------------- #
def load_data(a):
    from tokenizers import Tokenizer
    tok = Tokenizer.from_file(a.tokenizer)
    with open(a.corpus, encoding="utf-8", errors="ignore") as f:
        lines = [ln + "\n" for ln in f.read().split("\n") if ln.strip()]
    ids, nbytes = [], 0
    for s in range(0, len(lines), 5000):
        chunk = lines[s:s + 5000]
        nbytes += sum(len(x.encode("utf-8")) for x in chunk)
        for enc in tok.encode_batch(chunk):
            ids.extend(enc.ids)
    ids = np.asarray(ids, dtype=np.int64)
    if a.max_tokens:
        nbytes = int(nbytes * min(1.0, a.max_tokens / len(ids)))
        ids = ids[:a.max_tokens]
    blocks = len(ids) // (a.ctx + 1)
    arr = ids[:blocks * (a.ctx + 1)].reshape(blocks, a.ctx + 1)
    perm = np.random.default_rng(a.data_seed).permutation(blocks)
    n_val = min(a.max_val_blocks, max(1, int(blocks * a.val_frac)))
    return arr[perm[n_val:]], arr[perm[:n_val]], nbytes / len(ids), tok.get_vocab_size()


def load_mapping(path, vocab_size):
    m = np.load(path)
    V = max(int(m["vocab_size"]), vocab_size)
    is_variant = np.zeros(V, dtype=bool)
    is_variant[m["variant"]] = True
    roots = np.flatnonzero(~is_variant)
    root_index = np.full(V, -1, dtype=np.int64)
    root_index[roots] = np.arange(len(roots))
    tok_base = root_index.copy()
    tok_tag = np.zeros(V, dtype=np.int64)
    tok_base[m["variant"]] = root_index[m["base"]]
    tok_tag[m["variant"]] = m["tag"]
    assert (tok_base >= 0).all(), "a variant points at a base that is itself a variant"
    fingerprint = hashlib.md5(m["variant"].tobytes() + m["base"].tobytes() + m["tag"].tobytes()).hexdigest()[:10]
    T = int(m["tag"].max()) + 1
    valid = np.zeros((len(roots), T), dtype=bool)
    valid[:, 0] = True
    valid[tok_base[m["variant"]], m["tag"]] = True
    return {"V": V, "Vb": len(roots), "T": T, "tok_base": tok_base, "tok_tag": tok_tag, "valid": valid,
            "is_variant": is_variant, "tag_names": [str(x) for x in m["tag_names"]], "hash": fingerprint}


# --------------------------------------------------------------------------- #
# model
# --------------------------------------------------------------------------- #
class Block(nn.Module):
    def __init__(self, d, heads, d_ff):
        super().__init__()
        self.h = heads
        self.ln1, self.ln2 = nn.LayerNorm(d), nn.LayerNorm(d)
        self.qkv, self.proj = nn.Linear(d, 3 * d), nn.Linear(d, d)
        self.fc1, self.fc2 = nn.Linear(d, d_ff), nn.Linear(d_ff, d)

    def forward(self, x):
        B, T, D = x.shape
        q, k, v = self.qkv(self.ln1(x)).split(D, dim=2)
        q, k, v = (t.view(B, T, self.h, D // self.h).transpose(1, 2) for t in (q, k, v))
        att = F.scaled_dot_product_attention(q, k, v, is_causal=True).transpose(1, 2).reshape(B, T, D)
        x = x + self.proj(att)
        return x + self.fc2(F.gelu(self.fc1(self.ln2(x))))


class LM(nn.Module):
    def __init__(self, a, V, mapping=None, d_ff=None):
        super().__init__()
        d = a.d_model
        self.rov = mapping is not None
        self.emb = nn.Embedding(mapping["Vb"] if self.rov else V, d)  # tied input/output
        if self.rov:
            T = mapping["T"]
            self.tag_in = nn.Embedding(T, d, padding_idx=0)           # identity tag = zero vector
            self.tag_out = nn.Linear(d, T)
            self.tag_from_base = nn.Linear(d, T, bias=False)          # which tags a base tends to take
            self.register_buffer("tok_base", torch.from_numpy(mapping["tok_base"]))
            self.register_buffer("tok_tag", torch.from_numpy(mapping["tok_tag"]))
            self.register_buffer("valid", torch.from_numpy(mapping["valid"]))
        self.pos = nn.Embedding(a.ctx, d)
        self.blocks = nn.ModuleList(Block(d, a.n_head, d_ff or a.d_ff) for _ in range(a.n_layer))
        self.lnf = nn.LayerNorm(d)
        self.apply(self._init)
        for b in self.blocks:
            nn.init.normal_(b.proj.weight, std=0.02 / math.sqrt(2 * a.n_layer))
            nn.init.normal_(b.fc2.weight, std=0.02 / math.sqrt(2 * a.n_layer))
        if self.rov:
            with torch.no_grad():
                self.tag_in.weight[0].zero_()

    @staticmethod
    def _init(m):
        if isinstance(m, (nn.Linear, nn.Embedding)):
            nn.init.normal_(m.weight, std=0.02)
            if isinstance(m, nn.Linear) and m.bias is not None:
                nn.init.zeros_(m.bias)

    def nll(self, idx, tgt):
        """Per-token negative log-likelihood (nats), shape of tgt."""
        B, T = idx.shape
        if self.rov:
            x = self.emb(self.tok_base[idx]) + self.tag_in(self.tok_tag[idx])
        else:
            x = self.emb(idx)
        x = x + self.pos(torch.arange(T, device=idx.device))
        for b in self.blocks:
            x = b(x)
        h = self.lnf(x).reshape(B * T, -1)
        t = tgt.reshape(-1)
        if not self.rov:
            return F.cross_entropy(h @ self.emb.weight.T, t, reduction="none").view(B, T)
        b, g = self.tok_base[t], self.tok_tag[t]
        nb = F.cross_entropy(h @ self.emb.weight.T, b, reduction="none")
        tl = self.tag_out(h) + self.tag_from_base(self.emb.weight[b])
        tl = tl.masked_fill(~self.valid[b], float("-inf"))            # only tags that exist for this base
        return (nb + F.cross_entropy(tl, g, reduction="none")).view(B, T)


def config_sig(a, name, mapping, d_ff):
    """Everything that changes a run's result. Stored in checkpoints so stale ones are never reused silently."""
    keys = ["steps", "batch", "ctx", "d_model", "n_layer", "n_head", "lr", "warmup", "seed", "data_seed",
            "val_frac", "max_val_blocks", "max_tokens"]
    d = {k: getattr(a, k) for k in keys}
    d["run"], d["d_ff"] = name, d_ff or a.d_ff
    if name != "flat":
        d["map"] = mapping["hash"]
    return json.dumps(d, sort_keys=True)


def n_params(m):
    return sum(p.numel() for p in m.parameters())


def match_d_ff(a, V, mapping, target):
    per_unit = a.n_layer * (2 * a.d_model + 1)
    d_ff = a.d_ff
    while n_params(LM(a, V, mapping, d_ff)) < target:
        d_ff += max(8, (target - n_params(LM(a, V, mapping, d_ff))) // per_unit // 8 * 8)
    return d_ff


# --------------------------------------------------------------------------- #
# train / eval
# --------------------------------------------------------------------------- #
@torch.no_grad()
def evaluate(model, blocks, dev, bs=32):
    model.eval()
    out = []
    for i in range(0, len(blocks), bs):
        x = torch.from_numpy(blocks[i:i + bs]).to(dev)
        out.append(model.nll(x[:, :-1], x[:, 1:]).cpu())
    model.train()
    return torch.cat(out).numpy()


def run_one(name, a, V, mapping, train, val, bpt, dev, batches, d_ff=None):
    torch.manual_seed(a.seed)
    model = LM(a, V, mapping if name != "flat" else None, d_ff).to(dev)
    P = n_params(model)
    emb_params = sum(p.numel() for n, p in model.named_parameters()
                     if n.split(".")[0] in ("emb", "tag_in", "tag_out", "tag_from_base"))
    decay = [p for n, p in model.named_parameters() if p.dim() >= 2 and "emb" not in n]
    other = [p for n, p in model.named_parameters() if not (p.dim() >= 2 and "emb" not in n)]
    opt = torch.optim.AdamW([{"params": decay, "weight_decay": 0.1}, {"params": other, "weight_decay": 0.0}],
                            lr=a.lr, betas=(0.9, 0.95))
    print(f"\n== {name}: {P / 1e6:.2f}M params ({emb_params / 1e6:.2f}M in embedding/head), "
          f"d_ff={d_ff or a.d_ff}", flush=True)
    t0, curve = time.time(), []
    for step in range(a.steps):
        lr = a.lr * min(1.0, (step + 1) / a.warmup) * (0.1 + 0.9 * 0.5 * (1 + math.cos(math.pi * step / a.steps)))
        for g in opt.param_groups:
            g["lr"] = lr
        x = torch.from_numpy(train[batches[step]]).to(dev)
        loss = model.nll(x[:, :-1], x[:, 1:]).mean()
        opt.zero_grad(set_to_none=True)
        loss.backward()
        nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        opt.step()
        if (step + 1) % a.eval_every == 0 or step + 1 == a.steps:
            v = evaluate(model, val[:a.quick_val], dev).mean() / math.log(2)
            curve.append((step + 1, float(v)))
            print(f"  step {step + 1:5d}  train {loss.item() / math.log(2):.3f}  val {v:.3f} bits/token  "
                  f"({time.time() - t0:.0f}s)", flush=True)
    nll = evaluate(model, val, dev)
    return {"name": name, "params": P, "emb_params": emb_params, "nll": nll, "curve": curve,
            "seconds": time.time() - t0, "d_ff": d_ff or a.d_ff}


def compare(results, val, train, mapping, bpt, rare):
    tgt = val[:, 1:].reshape(-1)
    counts = np.bincount(train.reshape(-1), minlength=len(mapping["is_variant"]))
    var = mapping["is_variant"][tgt]
    rare_m = counts[tgt] < rare
    groups = [("all tokens", np.ones_like(var)), ("variant tokens", var), ("rare variants", var & rare_m),
              ("root tokens", ~var), ("rare roots", ~var & rare_m)]
    flat = next((r for r in results if r["name"] == "flat"), results[0])
    names = [r["name"] for r in results]
    stats = {}
    print("\n" + "=" * 78)
    print(f"RESULTS  ({len(tgt):,} validation tokens; bytes/token = {bpt:.2f}; rare = <{rare} train occurrences)")
    print("=" * 78)
    print(f"{'':<22}" + "".join(f"{n:>18}" for n in names))
    print(f"{'total params':<22}" + "".join(f"{r['params'] / 1e6:>17.2f}M" for r in results))
    print(f"{'embedding+head params':<22}" + "".join(f"{r['emb_params'] / 1e6:>17.2f}M" for r in results))
    print(f"{'train time (s)':<22}" + "".join(f"{r['seconds']:>18.0f}" for r in results))
    for gname, mask in groups:
        if mask.sum() == 0:
            continue
        row = f"{gname + f' (n={int(mask.sum()):,})':<22}"
        for r in results:
            row += f"{r['nll'].reshape(-1)[mask].mean() / math.log(2):>18.3f}"
        print(row + "   bits/token")
    print(f"{'bits per byte':<22}" + "".join(f"{r['nll'].mean() / math.log(2) / bpt:>18.4f}" for r in results))
    print("\npaired difference vs flat, bits/token (negative = better than flat), 95% CI:")
    for r in results:
        if r is flat:
            continue
        for gname, mask in groups:
            if mask.sum() < 30:
                continue
            d = (r["nll"].reshape(-1)[mask] - flat["nll"].reshape(-1)[mask]) / math.log(2)
            ci = 1.96 * d.std() / math.sqrt(len(d))
            verdict = "better" if d.mean() + ci < 0 else "worse" if d.mean() - ci > 0 else "no clear difference"
            print(f"  {r['name']:<12} {gname:<16} {d.mean():+.4f} +/- {ci:.4f}   {verdict}")
            stats.setdefault(r["name"], {})[gname] = {"diff": float(d.mean()), "ci": float(ci),
                                                      "n": int(mask.sum()), "verdict": verdict}
    return stats


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--tokenizer", required=True)
    p.add_argument("--corpus", required=True)
    p.add_argument("--mapping", required=True, help=".npz from rule_orbit_vocab.py --save-mapping")
    p.add_argument("--runs", nargs="+", default=["flat", "rov", "rov-matched"], choices=["flat", "rov", "rov-matched"])
    p.add_argument("--steps", type=int, default=1500)
    p.add_argument("--batch", type=int, default=24)
    p.add_argument("--ctx", type=int, default=96)
    p.add_argument("--d-model", type=int, default=128)
    p.add_argument("--n-layer", type=int, default=2)
    p.add_argument("--n-head", type=int, default=4)
    p.add_argument("--d-ff", type=int, default=512)
    p.add_argument("--lr", type=float, default=3e-3)
    p.add_argument("--warmup", type=int, default=100)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--data-seed", type=int, default=0)
    p.add_argument("--val-frac", type=float, default=0.05)
    p.add_argument("--max-val-blocks", type=int, default=400)
    p.add_argument("--quick-val", type=int, default=64, help="val blocks used for the training curve")
    p.add_argument("--eval-every", type=int, default=250)
    p.add_argument("--rare", type=int, default=30, help="'rare' = fewer than this many training occurrences")
    p.add_argument("--max-tokens", type=int, default=0, help="use only the first N tokens (quick tests)")
    p.add_argument("--threads", type=int, default=0)
    p.add_argument("--save", help="write results summary JSON")
    p.add_argument("--save-nll", help="prefix: each finished run is saved to PREFIX.<run>.npz and reused on re-launch (resume)")
    p.add_argument("--baseline", help=".npz from --save-nll; its 'flat' run is reused instead of retraining")
    a = p.parse_args()

    dev = "cuda" if torch.cuda.is_available() else "cpu"
    if a.threads:
        torch.set_num_threads(a.threads)
    train, val, bpt, V = load_data(a)
    mapping = load_mapping(a.mapping, V)
    print(f"device {dev} | train blocks {len(train):,} x {a.ctx} tokens | val blocks {len(val):,} | vocab {mapping['V']:,} "
          f"-> {mapping['Vb']:,} bases + {mapping['T']} tags ({mapping['is_variant'].mean():.1%} folded)")
    batches = np.random.default_rng(a.seed).integers(0, len(train), size=(a.steps, a.batch))  # same for every run

    results, flat_params = [], None
    if a.baseline:
        b = np.load(a.baseline)
        results.append({"name": "flat", "params": int(b["flat_params"]), "emb_params": int(b["flat_emb"]),
                        "nll": b["flat"], "curve": [], "seconds": float(b["flat_seconds"]), "d_ff": a.d_ff})
        flat_params = int(b["flat_params"])
        assert b["flat"].shape == (len(val), a.ctx), "baseline was made with a different data split / ctx"
    for name in a.runs:
        d_ff = None
        if name == "rov-matched":
            if flat_params is None:
                flat_params = n_params(LM(a, mapping["V"], None))
            d_ff = match_d_ff(a, mapping["V"], mapping, flat_params)
        ck = f"{a.save_nll}.{name}.npz" if a.save_nll else None
        sig = config_sig(a, name, mapping, d_ff)
        reuse = False
        if ck and os.path.exists(ck):  # resume: reuse a finished run only if its settings match
            c = np.load(ck)
            old = str(c["sig"]) if "sig" in c.files else None
            if old is not None and old != sig:
                print(f"\n== {name}: {ck} was made with DIFFERENT settings -> retraining\n   old: {old}\n   new: {sig}",
                      flush=True)
            else:
                assert c["nll"].shape == (len(val), a.ctx), f"{ck} was made with a different split/ctx; delete it"
                if old is None:
                    print(f"\n!! {ck} has no settings record (made by an older version): reusing it, "
                          f"but check it matches this run or delete it", flush=True)
                reuse = True
        if reuse:
            r = {"name": name, "params": int(c["params"]), "emb_params": int(c["emb"]), "nll": c["nll"],
                 "curve": [], "seconds": float(c["seconds"]), "d_ff": d_ff or a.d_ff}
            print(f"\n== {name}: loaded finished run from {ck}", flush=True)
        else:
            r = run_one(name, a, mapping["V"], mapping, train, val, bpt, dev, batches, d_ff)
            if ck:
                np.savez(ck, nll=r["nll"], params=r["params"], emb=r["emb_params"], seconds=r["seconds"], sig=sig)
        results.append(r)
        if name == "flat":
            flat_params = r["params"]
    stats = compare(results, val, train, mapping, bpt, a.rare)
    if a.save:
        json.dump({"runs": [{k: v for k, v in r.items() if k != "nll"}
                            | {"bits_per_token": float(r["nll"].mean() / math.log(2))} for r in results],
                   "paired_vs_flat": stats, "folded_fraction": float(mapping["is_variant"].mean()),
                   "bases": mapping["Vb"], "tags": mapping["T"]}, open(a.save, "w"), indent=1)


if __name__ == "__main__":
    main()
