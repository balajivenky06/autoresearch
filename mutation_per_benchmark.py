"""
mutation_per_benchmark.py — Split per-sample kill rates by source benchmark
(HumanEval vs MBPP) and re-run the mixed-effects analysis within each.

Question for the paper: does the "Iterative Critique vs Plain LLM on boundary
kill rate" significance hold on both HumanEval and MBPP, or is it driven by
one benchmark?

Method:
  1. Load per-sample analysis pkls (kill_rate per sample_idx).
  2. Recover source per sample_idx from the matching generation pkl.
  3. Build a long DataFrame with columns
        method, model, sample_idx, source, kill_rate, kill_rate_<op>
  4. For each source slice (humaneval, mbpp, pooled), fit
        kill_rate ~ C(method) + C(model) + C(sample_idx)
     plus Tukey HSD and Mixed-LM.

Note: with the seed-42 shuffle on the first 30 samples we have
  21 MBPP × 4 models × 4 methods = 336 cells max → 168-300 after filter
   9 HumanEval × 4 models × 4 methods = 144 cells max → 60-120 after filter
HumanEval is the smaller, less-powered slice.

Output (plots_mutation/):
    mutation_per_benchmark_report.txt
"""

from __future__ import annotations

import argparse
import math
import pickle
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import statsmodels.api as sm
import statsmodels.formula.api as smf
from statsmodels.formula.api import mixedlm
from statsmodels.stats.multicomp import pairwise_tukeyhsd

from mutation_statistical_tests import (
    METHODS,
    METHOD_LABELS,
    OPERATORS,
    parse_key,
)

GEN_DIR_CANDIDATES = [Path("checkpoints_mutation"),
                       Path(".checkpoints_mutation")]
ANALYSIS_DIR_CANDIDATES = [Path(".checkpoints_mutation_analysis"),
                            Path("checkpoints_mutation_analysis")]

OUTPUT_DIR  = Path("plots_mutation")
REPORT_FILE = OUTPUT_DIR / "mutation_per_benchmark_report.txt"


def _resolve(candidates: list[Path]) -> Path | None:
    for p in candidates:
        if p.is_dir():
            return p
    return None


# ---------------------------------------------------------------------------
# Data assembly
# ---------------------------------------------------------------------------

def load_source_map(gen_dir: Path) -> dict:
    """Build {(method, reasoning, model_normalised, sample_idx): source}.

    The generation pkls are lists of dicts with method/reasoning/model/sample_idx/source
    populated by regenerate_tests.
    """
    out = {}
    for f in sorted(gen_dir.glob("*.pkl")):
        if f.name.endswith(".tmp"):
            continue
        with open(f, "rb") as fp:
            data = pickle.load(fp)
        if not isinstance(data, list):
            continue
        for s in data:
            key = (s.get("method", ""), s.get("reasoning", ""),
                   s.get("model", "").replace(":", "_"),
                   s.get("sample_idx", -1))
            out[key] = s.get("source", "unknown")
    return out


def load_per_sample(analysis_dir: Path, source_map: dict) -> pd.DataFrame:
    rows = []
    for f in sorted(analysis_dir.glob("*.pkl")):
        if f.name.endswith(".tmp"):
            continue
        with open(f, "rb") as fp:
            data = pickle.load(fp)
        if not isinstance(data, dict):
            continue
        method, reasoning, model = parse_key(f.stem)
        # The model field on each generation pkl is the colon form; the
        # analysis pkl filename uses underscores. Build the lookup key in
        # the same underscore form.
        for sample_idx, result in data.items():
            kr = result.get("kill_rate", float("nan"))
            if isinstance(kr, float) and math.isnan(kr):
                continue
            src_key = (method, reasoning, model, sample_idx)
            source = source_map.get(src_key, "unknown")
            row = {
                "method":     method,
                "reasoning":  reasoning,
                "model":      model,
                "sample_idx": sample_idx,
                "source":     source,
                "kill_rate":  float(kr),
            }
            per_op = result.get("per_operator", {}) or {}
            for op in OPERATORS:
                stats_op = per_op.get(op) or {}
                t = stats_op.get("total", 0)
                k = stats_op.get("killed", 0)
                row[f"kill_rate_{op}"] = (k / t) if t > 0 else float("nan")
            rows.append(row)
    df = pd.DataFrame(rows)
    df["method"]    = df["method"].astype(str)
    df["model"]     = df["model"].astype(str)
    df["sample_idx"] = df["sample_idx"].astype(str)
    df["source"]    = df["source"].astype(str)
    return df


# ---------------------------------------------------------------------------
# Per-source analysis
# ---------------------------------------------------------------------------

def analyse_slice(df: pd.DataFrame, metric: str) -> dict:
    """ANOVA + Tukey HSD + Mixed-LM on a single source slice."""
    if df.empty or df[metric].dropna().empty:
        return {"n": 0, "anova": None, "tukey": None,
                "mlm": None, "error": "empty slice"}

    out = {"n": len(df)}
    if metric != "kill_rate":
        df = df.dropna(subset=[metric]).copy()
        df["kill_rate"] = df[metric]
    if df["kill_rate"].nunique() < 2:
        return {"n": len(df), "anova": None, "tukey": None,
                "mlm": None, "error": "constant metric"}

    formula = "kill_rate ~ C(method) + C(model) + C(sample_idx)"
    try:
        ols = smf.ols(formula, data=df).fit()
        out["anova"] = sm.stats.anova_lm(ols, typ=3)
    except Exception as e:
        out["anova"] = None
        out["anova_error"] = str(e)

    try:
        tk = pairwise_tukeyhsd(endog=df["kill_rate"].values,
                               groups=df["method"].values, alpha=0.05)
        out["tukey"] = pd.DataFrame(data=tk._results_table.data[1:],
                                    columns=tk._results_table.data[0])
    except Exception as e:
        out["tukey"] = None
        out["tukey_error"] = str(e)

    try:
        mlm = mixedlm("kill_rate ~ C(method) + C(model)",
                      data=df, groups=df["sample_idx"]).fit(method="lbfgs")
        out["mlm"] = mlm
    except Exception as e:
        out["mlm"] = None
        out["mlm_error"] = str(e)

    return out


# ---------------------------------------------------------------------------
# Reporting helpers
# ---------------------------------------------------------------------------

def fmt_p(p: float) -> str:
    if p is None or math.isnan(p):
        return "  nan"
    if p < 0.001:
        return "<0.001"
    return f"{p:.4f}"


def _h(title: str) -> str:
    return "=" * 78 + "\n " + title + "\n" + "=" * 78


def write_section(label: str, metric: str, res: dict) -> list[str]:
    lines = []
    lines.append(f"  [{label}]   metric = {metric}   n = {res.get('n', 0)}")
    if res.get("error"):
        lines.append(f"    {res['error']}")
        return lines

    if res.get("anova") is not None:
        aov = res["anova"]
        if "C(method)" in aov.index:
            row = aov.loc["C(method)"]
            sig = "*" if (not math.isnan(row["PR(>F)"]) and row["PR(>F)"] < 0.05) else " "
            lines.append(
                f"    ANOVA  method:  F={row['F']:.3f}, "
                f"p={fmt_p(row['PR(>F)'])}{sig}"
            )

    if res.get("tukey") is not None:
        sig_pairs = res["tukey"][res["tukey"]["reject"] == True]
        if not sig_pairs.empty:
            for _, row in sig_pairs.iterrows():
                g1 = METHOD_LABELS.get(row["group1"], row["group1"])
                g2 = METHOD_LABELS.get(row["group2"], row["group2"])
                lines.append(
                    f"    Tukey HSD *  {g1:<20} vs {g2:<20}  "
                    f"Δ={float(row['meandiff']):+.4f}  "
                    f"p_adj={fmt_p(float(row['p-adj']))}"
                )
        else:
            lines.append(f"    Tukey HSD: no pairs significant at α=0.05")

    if res.get("mlm") is not None:
        mlm = res["mlm"]
        # Print the method-vs-IC contrasts (first level is iter_critique alphabetically)
        method_terms = [t for t in mlm.params.index if t.startswith("C(method)")]
        for t in method_terms:
            p = mlm.pvalues[t]
            coef = mlm.params[t]
            sig = "*" if (not math.isnan(p) and p < 0.05) else " "
            lines.append(
                f"    Mixed-LM    {t:<40}  coef={coef:+.4f}  "
                f"p={fmt_p(p)}{sig}"
            )

    return lines


def write_report(df: pd.DataFrame, results: dict) -> str:
    lines = []
    lines.append(_h("MUTATION KILL RATE — PER-BENCHMARK ANALYSIS"))
    lines.append("")
    lines.append(f"Total observations: {len(df)}")
    lines.append(f"Methods: {sorted(df['method'].unique())}")
    lines.append(f"Models : {sorted(df['model'].unique())}")
    lines.append(f"Sources: {dict(df['source'].value_counts())}")
    lines.append("")

    # ---- Descriptive table per source × method × model -------------------
    lines.append(_h("1. PER-SOURCE × METHOD × MODEL MEANS"))
    grp = (df.groupby(["source", "model", "method"])["kill_rate"]
             .agg(["count", "mean", "std"]).round(4))
    lines.append(f"  {'source':<12}{'model':<22}{'method':<22}"
                 f"{'n':>5}{'mean':>9}{'std':>9}")
    lines.append("-" * 78)
    for (src, mdl, mth), row in grp.iterrows():
        lines.append(
            f"  {src:<12}{mdl:<22}{METHOD_LABELS.get(mth, mth):<22}"
            f"{int(row['count']):>5}{row['mean']:>9.4f}"
            f"{(row['std'] if not math.isnan(row['std']) else 0):>9.4f}"
        )
    lines.append("")

    # ---- Per-method × source means (collapsed across models) ------------
    lines.append("Per-method × source means (collapsed across models):")
    coll = (df.groupby(["source", "method"])["kill_rate"]
              .agg(["count", "mean", "std"]).round(4))
    for (src, mth), row in coll.iterrows():
        lines.append(
            f"  {src:<12}{METHOD_LABELS.get(mth, mth):<22}"
            f"n={int(row['count']):<5}  mean={row['mean']:.4f}  "
            f"std={row['std']:.4f}"
        )
    lines.append("")

    # ---- Statistical tests per metric × source --------------------------
    lines.append(_h("2. SIGNIFICANCE TESTS BY (metric × source)"))
    lines.append("  Each row: ANOVA on method, Tukey HSD post-hoc, Mixed-LM")
    lines.append("  contrasts vs Iterative Critique baseline (the alphabetical first).")
    lines.append("")
    for metric in ("kill_rate", "kill_rate_boundary"):
        lines.append(f"\nMetric: {metric}")
        lines.append("-" * 78)
        for src in ("humaneval", "mbpp", "pooled"):
            sub = df if src == "pooled" else df[df["source"] == src]
            lines += write_section(src, metric, results[(metric, src)])
            lines.append("")
    return "\n".join(lines)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", default=str(REPORT_FILE),
                        help="Report output path")
    args = parser.parse_args()

    OUTPUT_DIR.mkdir(exist_ok=True)

    gen_dir      = _resolve(GEN_DIR_CANDIDATES)
    analysis_dir = _resolve(ANALYSIS_DIR_CANDIDATES)
    if gen_dir is None:
        sys.exit("ERROR: no generation checkpoints dir found")
    if analysis_dir is None:
        sys.exit("ERROR: no analysis checkpoints dir found")
    print(f"Generation dir: {gen_dir}")
    print(f"Analysis dir  : {analysis_dir}")

    print("Building source map from generation pkls...")
    source_map = load_source_map(gen_dir)
    print(f"  {len(source_map)} sample entries")

    print("Loading per-sample analysis data...")
    df = load_per_sample(analysis_dir, source_map)
    print(f"  {len(df)} observations  "
          f"sources={dict(df['source'].value_counts())}")
    print()

    # Run per-source × per-metric analyses
    results = {}
    for metric in ("kill_rate", "kill_rate_boundary"):
        for src in ("humaneval", "mbpp", "pooled"):
            print(f"Analysing  metric={metric}  source={src}...")
            sub = df if src == "pooled" else df[df["source"] == src]
            results[(metric, src)] = analyse_slice(sub, metric)

    report = write_report(df, results)
    Path(args.out).parent.mkdir(exist_ok=True)
    Path(args.out).write_text(report)
    print(f"\nReport → {args.out}")

    # Headline echo
    print()
    print("=" * 70)
    print("HEADLINE — boundary kill rate: IC vs Plain LLM within each benchmark")
    print("=" * 70)
    for src in ("humaneval", "mbpp", "pooled"):
        r = results[("kill_rate_boundary", src)]
        if r.get("anova") is None:
            print(f"  {src:<10}  (no ANOVA: {r.get('error', 'fit failed')})")
            continue
        aov = r["anova"]
        if "C(method)" not in aov.index:
            continue
        anova_p = aov.loc["C(method)"]["PR(>F)"]
        anova_F = aov.loc["C(method)"]["F"]
        tk = r["tukey"]
        ic_pl = tk[
            (((tk["group1"] == "iterative_critique") & (tk["group2"] == "plain_llm")) |
             ((tk["group1"] == "plain_llm") & (tk["group2"] == "iterative_critique")))
        ] if tk is not None else pd.DataFrame()
        if not ic_pl.empty:
            row = ic_pl.iloc[0]
            ic_p = float(row["p-adj"])
            ic_d = float(row["meandiff"])
            sig = "*" if ic_p < 0.05 else " "
            print(f"  {src:<10}  n={r['n']:<4}  ANOVA F={anova_F:.3f} "
                  f"p={fmt_p(anova_p)}   Tukey IC-vs-PL Δ={ic_d:+.4f}  "
                  f"p_adj={fmt_p(ic_p)}{sig}")
        else:
            print(f"  {src:<10}  n={r['n']:<4}  ANOVA F={anova_F:.3f} "
                  f"p={fmt_p(anova_p)}   (no IC-vs-PL Tukey row)")
    return 0


if __name__ == "__main__":
    sys.exit(main())
