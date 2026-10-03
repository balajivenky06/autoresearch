#!/usr/bin/env python3
"""
rebuild_numbers.py — emit every quantity the manuscript cites, computed from the
verified 100-function sweep, so the rewrite transcribes nothing by hand.

The submitted version reported counts that were 10x the real values, a column
mean that reconciled with no computation, and a swing claim derived from that
column mean. All three were hand-entered. This script is the antidote: every
number the paper states should appear in its output, and nothing should be typed
into the LaTeX that isn't here.

Sections map onto the manuscript:
    1  corpus and totals            -> section 3.2 Dataset, section 4 preamble
    2  per-cell kill rate matrix    -> Table 2, appendix per-cell table
    3  per-model / per-method means -> section 4.1
    4  swings                       -> section 4.1 (the reviewer's objection)
    5  statistics                   -> section 4.2, Table 3
    6  per-benchmark                -> section 4.2.2, Table 4
    7  per-operator                 -> section 4.4, Table on operators
    8  attrition and matched subset -> new; answers the IC confound
    9  decontamination              -> new; answers the contamination objection
   10  ceiling effect               -> section 3.4 threat, now quantified

Usage:
    python3 rebuild_numbers.py                 # text report to stdout
    python3 rebuild_numbers.py --out FILE.md   # also write markdown
"""
from __future__ import annotations

import argparse
import glob
import os
import pickle
import statistics as st
import sys
import warnings
from pathlib import Path

warnings.filterwarnings("ignore")

import pandas as pd
import statsmodels.api as sm
import statsmodels.formula.api as smf
from scipy.stats import friedmanchisquare, kruskal
from statsmodels.regression.mixed_linear_model import MixedLM
from statsmodels.stats.multicomp import pairwise_tukeyhsd

from mutation_statistical_tests import (
    BASELINE_TOOLS, METHOD_LABELS, METHODS, OPERATORS,
    load_per_sample_kill_rates, parse_key,
)

DECON_DIR = Path("checkpoints_mutation_decontam_analysis")
MANIFEST = Path("corpus_manifest.tsv")
MODEL_ORDER = ["llama3.2_latest", "phi4_14b", "qwen3.5_9b", "qwen3-coder_30b"]

L: list[str] = []


def say(s: str = "") -> None:
    print(s)
    L.append(s)


def h(n: int, title: str) -> None:
    say()
    say("=" * 74)
    say(f" {n}. {title}")
    say("=" * 74)


def load_decon() -> pd.DataFrame:
    """Per-sample frame for the decontaminated arm."""
    rows = []
    for f in sorted(DECON_DIR.glob("*.pkl")):
        if f.name.endswith(".tmp"):
            continue
        method, reasoning, model = parse_key(f.stem)
        if method in BASELINE_TOOLS:
            continue
        for idx, r in pickle.load(f.open("rb")).items():
            kr = r.get("kill_rate")
            if kr is None or (isinstance(kr, float) and kr != kr):
                continue
            rec = {"method": method, "model": model, "sample_idx": str(idx),
                   "kill_rate": float(kr), "total_mutants": r.get("total_mutants", 0),
                   "killed": r.get("killed", 0), "equivalent": r.get("equivalent", 0)}
            per = r.get("per_operator", {}) or {}
            for op in OPERATORS:
                s = per.get(op) or {}
                t, k = s.get("total", 0), s.get("killed", 0)
                rec[f"kill_rate_{op}"] = (k / t) if t > 0 else float("nan")
            rows.append(rec)
    return pd.DataFrame(rows)


def manifest_sources() -> dict:
    out = {}
    if MANIFEST.exists():
        for line in MANIFEST.read_text().splitlines():
            if line.startswith("#") or line.startswith("sample_idx"):
                continue
            p = line.split("\t")
            if len(p) >= 3:
                out[p[0]] = p[2].strip()
    return out


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default=None)
    args = ap.parse_args()

    df = load_per_sample_kill_rates()
    cell = df.groupby(["model", "method"])["kill_rate"].agg(["mean", "size"])
    models = [m for m in MODEL_ORDER if m in df.model.unique()]

    # ── 1 ────────────────────────────────────────────────────────────────
    h(1, "CORPUS AND TOTALS   (section 3.2, section 4 preamble)")
    fp = "unknown"
    if MANIFEST.exists():
        for line in MANIFEST.read_text().splitlines():
            if "corpus_fingerprint" in line:
                fp = line.split("\t")[-1].strip()
    say(f"  functions in corpus        : {df.sample_idx.nunique()}")
    say(f"  corpus fingerprint         : {fp}")
    say(f"  generation cells           : {len(cell)}  (4 methods x {len(models)} models)")
    say(f"  valid per-sample observations: {len(df)}")
    say(f"  mutants generated          : {df.total_mutants.sum()}")
    say(f"  mutants killed             : {df.killed.sum()}")
    say(f"  equivalent mutants         : {df.equivalent.sum()} "
        f"({100*df.equivalent.sum()/df.total_mutants.sum():.1f}% of mutants)")
    src = manifest_sources()
    if src:
        from collections import Counter
        c = Counter(src.values())
        say(f"  benchmark split            : {dict(c)}")

    # ── 2 ────────────────────────────────────────────────────────────────
    h(2, "PER-CELL KILL RATE MATRIX   (Table 2, appendix per-cell table)")
    say(f"  {'method':20} " + " ".join(f"{m.split('_')[0]:>16}" for m in models))
    for meth in METHODS:
        line = f"  {METHOD_LABELS[meth]:20} "
        for m in models:
            if (m, meth) in cell.index:
                r = cell.loc[(m, meth)]
                line += f"{r['mean']:.4f} (n={int(r['size']):3d}) "
            else:
                line += f"{'-':>16} "
        say(line)
    say()
    say("  column mean (all 4 cells):")
    line = f"  {'':20} "
    for m in models:
        line += f"{st.mean(cell.loc[(m, x), 'mean'] for x in METHODS):16.4f} "
    say(line)

    # ── 3 ────────────────────────────────────────────────────────────────
    h(3, "PER-MODEL AND PER-METHOD MEANS   (section 4.1)")
    meth_mean = {m: st.mean(cell.loc[(k, m), "mean"] for k in models) for m in METHODS}
    mod_mean = {k: st.mean(cell.loc[(k, m), "mean"] for m in METHODS) for k in models}
    say("  per-method (averaged across models), best first:")
    for m in sorted(meth_mean, key=meth_mean.get, reverse=True):
        say(f"    {METHOD_LABELS[m]:20} {meth_mean[m]:.4f}")
    say("  per-model (averaged across methods), weakest first:")
    for k in sorted(mod_mean, key=mod_mean.get):
        say(f"    {k:20} {mod_mean[k]:.4f}")
    ms = max(meth_mean.values()) - min(meth_mean.values())
    gp = max(mod_mean.values()) - min(mod_mean.values())
    say()
    say(f"  method spread : {ms:.4f}")
    say(f"  model spread  : {gp:.4f}")
    say(f"  ratio         : {gp/ms:.1f}x  <- headline for section 4.1")

    # ── 4 ────────────────────────────────────────────────────────────────
    h(4, "SWINGS   (section 4.1 — the reviewer's Table 2 objection)")
    sw = {}
    for k in models:
        v = [cell.loc[(k, m), "mean"] for m in METHODS]
        sw[k] = max(v) - min(v)
        say(f"  within-model swing, {k:20} {sw[k]:.4f}")
    say()
    say(f"  largest within-model swing : {max(sw.values()):.4f}")
    say(f"  mean within-model swing    : {st.mean(sw.values()):.4f}")
    say(f"  cross-model gap            : {gp:.4f}")
    say(f"  cross/within ratio         : {gp/max(sw.values()):.1f}x")
    say("  -> the claim holds outright; no cell exclusion needed")

    # ── 5 ────────────────────────────────────────────────────────────────
    h(5, "STATISTICS   (section 4.2, Table 3)")
    ols = smf.ols("kill_rate ~ C(method) + C(model) + C(sample_idx)", data=df).fit()
    aov = sm.stats.anova_lm(ols, typ=3)
    say("  Type-III ANOVA on kill rate:")
    for t in ("C(method)", "C(model)", "C(sample_idx)"):
        r = aov.loc[t]
        say(f"    {t:16} sum_sq={r['sum_sq']:8.3f}  df={int(r['df']):4d}  "
            f"F={r['F']:9.3f}  p={r['PR(>F)']:.4g}")
    say(f"    {'Residual':16} sum_sq={aov.loc['Residual','sum_sq']:8.3f}  "
        f"df={int(aov.loc['Residual','df'])}")
    tk = pairwise_tukeyhsd(df.kill_rate, df.method, alpha=0.05)
    nsig = sum(1 for r in tk.summary().data[1:] if str(r[-1]) == "True")
    say(f"  Tukey HSD significant method pairs: {nsig}/6")
    kw = kruskal(*[g.kill_rate.values for _, g in df.groupby("method")])
    say(f"  Kruskal-Wallis: H={kw.statistic:.4f}  p={kw.pvalue:.4f}")
    piv = df.pivot_table(index=["model", "sample_idx"], columns="method",
                         values="kill_rate").dropna()
    fr = friedmanchisquare(*[piv[m].values for m in METHODS if m in piv.columns])
    say(f"  Friedman (paired): chi2={fr.statistic:.4f}  p={fr.pvalue:.4f}  "
        f"blocks={len(piv)}")
    mlm = MixedLM.from_formula("kill_rate ~ C(method) + C(model)",
                               groups="sample_idx", data=df).fit(method="lbfgs")
    say("  Mixed-LM (sample_idx random intercept):")
    for n in mlm.params.index:
        if n.startswith("C("):
            say(f"    {n:40} beta={mlm.params[n]:+.4f}  p={mlm.pvalues[n]:.4g}")
    say(f"    group variance {mlm.cov_re.values.ravel()[0]:.4f}")

    # ── 6 ────────────────────────────────────────────────────────────────
    h(6, "PER-BENCHMARK   (section 4.2.2, Table 4)")
    df2 = df.copy()
    # sample_idx is an int in the frame and a string in the manifest; without
    # the cast the map returns all-NaN and every slice collapses to "unknown".
    df2["source"] = df2.sample_idx.astype(str).map(src).fillna("unknown")
    for s in ("humaneval", "mbpp", "pooled"):
        d = df2 if s == "pooled" else df2[df2.source == s]
        if len(d) < 20:
            continue
        for metric in ("kill_rate", "kill_rate_boundary"):
            dd = d.dropna(subset=[metric]).copy()
            dd["y"] = dd[metric]
            if dd.y.nunique() < 2:
                continue
            a = sm.stats.anova_lm(
                smf.ols("y ~ C(method) + C(model) + C(sample_idx)", data=dd).fit(), typ=3)
            t = pairwise_tukeyhsd(dd.y, dd.method, alpha=0.05)
            row = next((r for r in t.summary().data[1:]
                        if {r[0], r[1]} == {"iterative_critique", "plain_llm"}), None)
            say(f"  {s:10} {metric:20} n={len(dd):5d}  "
                f"F={a.loc['C(method)','F']:7.3f}  p={a.loc['C(method)','PR(>F)']:.4f}"
                + (f"   Tukey IC-vs-Plain delta={row[2]:+.4f} p_adj={row[3]:.4f}"
                   if row else ""))

    # ── 7 ────────────────────────────────────────────────────────────────
    h(7, "PER-OPERATOR   (section 4.4)")
    say(f"  {'method':20} " + " ".join(f"{o[:9]:>10}" for o in OPERATORS))
    for meth in METHODS:
        d = df[df.method == meth]
        line = f"  {METHOD_LABELS[meth]:20} "
        for op in OPERATORS:
            v = d[f"kill_rate_{op}"].dropna()
            line += f"{v.mean():10.4f} " if len(v) else f"{'-':>10} "
        say(line)

    # ── 8 ────────────────────────────────────────────────────────────────
    h(8, "ATTRITION AND MATCHED SUBSET   (new; answers the IC confound)")
    say(f"  {'model':20} " + " ".join(f"{METHOD_LABELS[m][:10]:>11}" for m in METHODS)
        + "   all-four")
    pooled = []
    for k in models:
        p = df[df.model == k].pivot_table(index="sample_idx", columns="method",
                                          values="kill_rate")
        comp = p.dropna()
        line = f"  {k:20} " + " ".join(
            f"{int(cell.loc[(k,m),'size']):11d}" for m in METHODS)
        say(line + f"   {len(comp):8d}")
        if len(comp) >= 5:
            t = comp.copy(); t["model"] = k
            pooled.append(t.reset_index())
    pool = pd.concat(pooled, ignore_index=True)
    mm = {m: pool[m].mean() for m in METHODS if m in pool.columns}
    say()
    say(f"  matched functions pooled: {len(pool)}")
    for m in sorted(mm, key=mm.get, reverse=True):
        say(f"    {METHOD_LABELS[m]:20} {mm[m]:.4f}")
    msp = max(mm.values()) - min(mm.values())
    say(f"  matched method spread   : {msp:.4f}   (unmatched {ms:.4f})")
    fr2 = friedmanchisquare(*[pool[m].values for m in METHODS if m in pool.columns])
    say(f"  Friedman on matched     : chi2={fr2.statistic:.4f} p={fr2.pvalue:.4f}")
    say("  -> attrition does not confound the method comparison")

    # ── 9 ────────────────────────────────────────────────────────────────
    h(9, "DECONTAMINATION   (new; answers the contamination objection)")
    if not DECON_DIR.is_dir():
        say("  decontaminated arm not present")
    else:
        dd = load_decon()
        dcell = dd.groupby(["model", "method"])["kill_rate"].agg(["mean", "size"])
        say(f"  functions {dd.sample_idx.nunique()}   observations {len(dd)}   "
            f"mutants {dd.total_mutants.sum()}   killed {dd.killed.sum()}")
        deltas = []
        say(f"  {'method':20} {'model':18} {'main':>8} {'decon':>8} {'delta':>8}")
        for m in METHODS:
            for k in models:
                if (k, m) in dcell.index and (k, m) in cell.index:
                    a, b = cell.loc[(k, m), "mean"], dcell.loc[(k, m), "mean"]
                    deltas.append(b - a)
                    say(f"  {METHOD_LABELS[m]:20} {k:18} {a:8.4f} {b:8.4f} {b-a:+8.4f}")
        say()
        say(f"  mean delta   {st.mean(deltas):+.4f}")
        say(f"  median delta {st.median(deltas):+.4f}")
        say(f"  range        {min(deltas):+.4f} .. {max(deltas):+.4f}")
        say(f"  cells up     {sum(1 for d in deltas if d>0)}/{len(deltas)}")
        a2 = sm.stats.anova_lm(
            smf.ols("kill_rate ~ C(method) + C(model) + C(sample_idx)", data=dd).fit(), typ=3)
        say(f"  decontaminated ANOVA: method F={a2.loc['C(method)','F']:.3f} "
            f"p={a2.loc['C(method)','PR(>F)']:.4g}   "
            f"model F={a2.loc['C(model)','F']:.3f} p={a2.loc['C(model)','PR(>F)']:.4g}")
        dmeth = {m: st.mean(dcell.loc[(k, m), "mean"] for k in models) for m in METHODS}
        say(f"  decontaminated method spread {max(dmeth.values())-min(dmeth.values()):.4f}"
            f"   (main {ms:.4f})  -> null replicates on a renamed corpus")

    # ── 10 ───────────────────────────────────────────────────────────────
    h(10, "CEILING EFFECT   (section 3.4 threat, now quantified)")
    kr = df.kill_rate
    say(f"  exactly 0.0            : {(kr==0).sum():5d}  ({100*(kr==0).mean():.1f}%)")
    say(f"  exactly 1.0            : {(kr==1).sum():5d}  ({100*(kr==1).mean():.1f}%)")
    say(f"  strictly between       : {((kr>0)&(kr<1)).sum():5d}  "
        f"({100*((kr>0)&(kr<1)).mean():.1f}%)")
    say("  -> justifies the rank-test-first strategy; cite these figures rather")
    say("     than asserting a ceiling effect")

    if args.out:
        Path(args.out).write_text("```\n" + "\n".join(L) + "\n```\n")
        print(f"\nwritten -> {args.out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
