#!/usr/bin/env python3
"""
Summarise rov_ab_test.py results (--save JSON files) across seeds.

Files are grouped by name with a trailing _s<seed> removed:
    bigval/big_cap_s0.json, big_cap_s1.json, big_cap_s2.json   ->  group "big_cap"

  python summarize_runs.py bigval/big_cap_s*.json bigval/big_case_s*.json
  python summarize_runs.py bigval/*_s*.json --compare big_cap big_case

Reports, per group and metric, the mean and standard deviation of the paired difference vs the flat
model ACROSS SEEDS. That spread is the honest error bar; the +/- printed inside each log only measures
sampling noise over validation tokens and is typically 5-10x smaller than the seed-to-seed spread.
--compare A B prints the per-seed difference between two groups (same seed = same data order and
flat baseline) with a standard error.
"""
import argparse
import glob
import json
import math
import os
import re
import statistics as st
from collections import defaultdict

METRICS = ["all tokens", "variant tokens", "rare variants", "root tokens", "rare roots"]


def load(paths, model):
    groups = defaultdict(dict)  # group -> seed -> {metric: (diff, ci, n)}
    for pat in paths:
        for path in sorted(glob.glob(pat)):
            base = os.path.splitext(os.path.basename(path))[0]
            m = re.match(r"(.*)_s(\d+)$", base)
            group, seed = (m.group(1), int(m.group(2))) if m else (base, len(groups[base]))
            d = json.load(open(path, encoding="utf-8"))["paired_vs_flat"]
            name = model if model in d else next(iter(d), None)
            if name is None:
                print(f"skipping {path}: no paired results")
                continue
            groups[group][seed] = {k: (v["diff"], v["ci"], v["n"]) for k, v in d[name].items()}
    return groups


def mean_sd(xs):
    return st.mean(xs), (st.stdev(xs) if len(xs) > 1 else float("nan"))


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("files", nargs="+")
    p.add_argument("--model", default="rov-matched", help="which folded run to read (default rov-matched)")
    p.add_argument("--compare", nargs=2, metavar=("A", "B"), help="group names to compare seed by seed")
    a = p.parse_args()

    groups = load(a.files, a.model)
    if not groups:
        raise SystemExit("no result files found")

    print(f"difference vs flat, bits/token, model '{a.model}' (negative = folded model is better)\n")
    print(f"{'group':<16}{'seeds':>6}" + "".join(f"{m:>22}" for m in METRICS))
    for g, seeds in sorted(groups.items()):
        row = f"{g:<16}{len(seeds):>6}"
        for m in METRICS:
            xs = [v[m][0] for v in seeds.values() if m in v]
            if not xs:
                row += f"{'n/a':>22}"
                continue
            mu, sd = mean_sd(xs)
            cell = f"{mu:+.3f} ± {sd:.3f}" if not math.isnan(sd) else f"{mu:+.3f} (1 seed)"
            row += f"{cell:>22}"
        print(row)
    print("\n  mean ± standard deviation across seeds; with fewer than 4 seeds the spread itself is very uncertain.")

    print("\nper-seed values (all tokens | rare variants | rare roots):")
    for g, seeds in sorted(groups.items()):
        for s in sorted(seeds):
            v = seeds[s]
            cells = [f"{v[m][0]:+.3f}" if m in v else "n/a" for m in ("all tokens", "rare variants", "rare roots")]
            print(f"  {g:<14} seed {s}:  " + "   ".join(cells) + f"    (validation-noise CI on all tokens: ±{v['all tokens'][1]:.3f})")

    if a.compare:
        A, B = a.compare
        for g in (A, B):
            if g not in groups:
                raise SystemExit(f"group '{g}' not found; have: {', '.join(sorted(groups))}")
        common = sorted(set(groups[A]) & set(groups[B]))
        print(f"\npaired comparison {A} minus {B}, seed by seed (positive = {A} gains LESS than {B}):")
        for m in ("all tokens", "rare variants", "rare roots"):
            ds = [groups[A][s][m][0] - groups[B][s][m][0] for s in common if m in groups[A][s] and m in groups[B][s]]
            if len(ds) < 2:
                print(f"  {m:<14} need at least 2 shared seeds (have {len(ds)})")
                continue
            mu, sd = mean_sd(ds)
            se = sd / math.sqrt(len(ds))
            verdict = "difference exceeds 2 SE" if abs(mu) > 2 * se else "within noise: treat as a tie"
            note = "" if m == "all tokens" else "   [NOT like-for-like: the two families fold different tokens]"
            print(f"  {m:<14} mean {mu:+.3f}  SE {se:.3f}  over seeds {common}   -> {verdict}{note}")
        print("  (this treats the models, and not only the seeds, as fixed: it says nothing about other"
              " vocabularies, data or scales)")


if __name__ == "__main__":
    main()
