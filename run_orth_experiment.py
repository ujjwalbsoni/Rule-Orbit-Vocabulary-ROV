#!/usr/bin/env python3
"""
One-command experiment: which kinds of variants can safely share a base token?
===============================================================================
For each rule family it
  1. mines merges with the embedding gate        (rule_orbit_vocab.py --families F)
  2. trains a flat model and a folded model on identical batches   (rov_ab_test.py)
  3. prints one summary table

Families:  suffix = walk/walked/walking     case = ' The'/' the'     space = ' and'/'and'
           orth = case + space      all = everything

Typical use (real model embeddings; downloads the weights of the model you name):

  python run_orth_experiment.py --hf-model gpt2 --embed-model gpt2 --corpus text.txt

Your own tokenizer + embedding matrix aligned to its ids:

  python run_orth_experiment.py --tokenizer tok.json --embed-file wte.npy --corpus text.txt

No access to model weights? Build corpus-derived stand-in embeddings first (weaker signal):

  python make_proxy_embeddings.py --tokenizer tok.json --corpus text.txt --out proxy.npy
  python run_orth_experiment.py --tokenizer tok.json --embed-file proxy.npy --corpus text.txt

Training defaults are tiny (2 layers, 1000 steps) so it finishes on a laptop CPU; raise
--steps / --d-model / --n-layer / --seed for anything you intend to believe.
Interrupted runs resume: finished models are cached in --workdir.
"""
import argparse
import json
import os
import shutil
import subprocess
import sys

HERE = os.path.dirname(os.path.abspath(__file__))


def run(cmd, log):
    print("  $ " + " ".join(cmd[1:3]) + " ...", flush=True)
    with open(log, "w", encoding="utf-8") as f:
        rc = subprocess.run(cmd, stdout=f, stderr=subprocess.STDOUT).returncode
    if rc != 0:
        print(open(log, encoding="utf-8").read()[-2500:])
        sys.exit(f"command failed (exit {rc}); full log: {log}")
    return open(log, encoding="utf-8").read()


def section(text, header, n):
    lines = text.splitlines()
    for i, ln in enumerate(lines):
        if ln.startswith(header):
            return lines[i + 1:i + 1 + n]
    return []


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    t = p.add_argument_group("tokenizer (one)")
    t.add_argument("--hf-model")
    t.add_argument("--tokenizer")
    e = p.add_argument_group("embeddings (default: --hf-model's own)")
    e.add_argument("--embed-model")
    e.add_argument("--embed-file")
    e.add_argument("--embed-key")
    e.add_argument("--word-vectors")
    p.add_argument("--corpus", required=True)
    p.add_argument("--workdir", default="orth_experiment")
    p.add_argument("--families", nargs="+", default=["suffix", "orth", "all"], choices=["suffix", "orth", "case", "space", "cap", "all"])
    p.add_argument("--matched", action="store_true", help="also train a parameter-matched folded model")
    p.add_argument("--reuse-flat", help="a finished flat checkpoint (.npz) from an earlier identical setup")
    p.add_argument("--train-python", default=sys.executable, help="python with torch, if different")
    p.add_argument("--miner-args", default="", help="extra flags for rule_orbit_vocab.py, e.g. '--emb-null-quantile 0.95'")
    g = p.add_argument_group("training (passed to rov_ab_test.py)")
    for name, default in [("steps", 1000), ("batch", 24), ("ctx", 96), ("d-model", 128), ("n-layer", 2),
                          ("n-head", 4), ("d-ff", 512), ("seed", 0), ("max-tokens", 0), ("threads", 0)]:
        g.add_argument(f"--{name}", type=int, default=default)
    g.add_argument("--lr", type=float, default=3e-3)
    a = p.parse_args()

    os.makedirs(a.workdir, exist_ok=True)
    W = lambda x: os.path.join(a.workdir, x)  # noqa: E731

    tok_path = a.tokenizer
    if a.hf_model:
        from tokenizers import Tokenizer
        tok_path = W("tokenizer.json")
        Tokenizer.from_pretrained(a.hf_model).save(tok_path)
    if not tok_path:
        p.error("give --hf-model or --tokenizer")
    emb = []
    if a.embed_model or (a.hf_model and not (a.embed_file or a.word_vectors)):
        emb = ["--embed-model", a.embed_model or a.hf_model]
    if a.embed_file:
        emb = ["--embed-file", a.embed_file] + (["--embed-key", a.embed_key] if a.embed_key else [])
    if a.word_vectors:
        emb = ["--word-vectors", a.word_vectors]
    if not emb:
        p.error("give an embedding source: --embed-model / --embed-file / --word-vectors")

    summary, flat_ck = {}, a.reuse_flat
    for fam in a.families:
        print(f"\n=== family: {fam} " + "=" * 50, flush=True)
        mp, ck = W(f"map_{fam}.npz"), W(f"ck_{fam}")
        out = run([sys.executable, os.path.join(HERE, "rule_orbit_vocab.py"), "--tokenizer", tok_path,
                   "--corpus", a.corpus, "--gate", "embeddings", "--families", fam, "--save-mapping", mp,
                   "--top", "30"] + emb + a.miner_args.split(), W(f"mine_{fam}.log"))
        for ln in out.splitlines():
            if any(k in ln for k in ("embedding gate", "variants folded", "space/case-only", "suffix-edit rules",
                                     "lossless")):
                print("   " + ln.strip())
        print("   closest-call merges that were ACCEPTED (check these by eye):")
        for ln in section(out, "AUDIT: lowest-similarity pairs ACCEPTED", 6):
            print("   " + ln.strip())
        if "variants folded into an orbit      : 0 " in out:
            print("   nothing folded for this family; skipping training")
            continue

        if flat_ck:  # train the flat baseline only once
            shutil.copy(flat_ck, ck + ".flat.npz")
        cmd = [a.train_python, os.path.join(HERE, "rov_ab_test.py"), "--tokenizer", tok_path, "--corpus", a.corpus,
               "--mapping", mp, "--runs", "flat", "rov"] + (["rov-matched"] if a.matched else []) + [
               "--save-nll", ck, "--save", W(f"ab_{fam}.json"), "--steps", str(a.steps), "--batch", str(a.batch),
               "--ctx", str(a.ctx), "--d-model", str(a.d_model), "--n-layer", str(a.n_layer),
               "--n-head", str(a.n_head), "--d-ff", str(a.d_ff), "--seed", str(a.seed), "--lr", str(a.lr)]
        if a.max_tokens:
            cmd += ["--max-tokens", str(a.max_tokens)]
        if a.threads:
            cmd += ["--threads", str(a.threads)]
        print("   training (this is the slow part) ...", flush=True)
        run(cmd, W(f"ab_{fam}.log"))
        flat_ck = flat_ck or ck + ".flat.npz"
        summary[fam] = json.load(open(W(f"ab_{fam}.json"), encoding="utf-8"))

    print("\n" + "=" * 86)
    print("SUMMARY  (bits/token difference vs flat model; negative = folded model is better; 95% CI)")
    print("=" * 86)
    print(f"{'family':<8}{'model':<13}{'folded':>8}{'emb+head':>11}{'all tokens':>18}{'variant tokens':>20}{'rare variants':>19}")
    for fam, s in summary.items():
        flat = next(r for r in s["runs"] if r["name"] == "flat")
        for r in s["runs"]:
            if r["name"] == "flat":
                continue
            st = s["paired_vs_flat"].get(r["name"], {})

            def cell(g):
                return f"{st[g]['diff']:+.3f}±{st[g]['ci']:.3f}" if g in st else "n/a"
            print(f"{fam:<8}{r['name']:<13}{s['folded_fraction']:>8.1%}{r['emb_params'] / flat['emb_params'] - 1:>+11.1%}"
                  f"{cell('all tokens'):>18}{cell('variant tokens'):>20}{cell('rare variants'):>19}")
    print(f"\nlogs, mappings and per-run checkpoints are in {a.workdir}/")


if __name__ == "__main__":
    main()
