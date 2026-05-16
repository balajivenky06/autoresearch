"""
mutation_mixed_effects.py — Linear models that exploit the nested
structure of the mutation-testing data.

The paired-sample tests in mutation_statistical_tests.py undercount the
data because the Friedman / Wilcoxon design drops every sample_idx that
isn't valid in all four methods. The same sample appears across four LLMs
and (when the LLM-generated tests filter through) four methods, so the
right model treats method as a fixed effect and accounts for the
crossed grouping by model and sample_idx.

Two complementary fits:

  1. Type-III ANOVA on
       kill_rate ~ C(method) + C(model) + C(sample_idx)
     This is the OLS-with-blocking analogue of the Friedman test but
     it tolerates unbalanced cells, and the F-test on the method term
     is the test you want for "do methods differ on average?".
     Followed by Tukey HSD post-hoc comparisons on method.

  2. Mixed-LM with sample_idx as a variance component (random),
     model entered as a fixed factor with 4 specific levels. We're not
     trying to generalise beyond these 4 LLMs, so a random model term
     would be a misuse of the variance estimator (4 levels is too few).

Both can be run on any metric (--metric kill_rate, kill_rate_boundary,
etc.). The script reads per-sample data through
mutation_statistical_tests.load_per_sample_kill_rates so it stays in sync
with the rest of the analysis pipeline.

Output:
  plots_mutation/mutation_mixed_effects_report.txt
"""

from __future__ import annotations

import argparse
import math
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
    load_per_sample_kill_rates,
)

OUTPUT_DIR  = Path("plots_mutation")
REPORT_FILE = OUTPUT_DIR / "mutation_mixed_effects_report.txt"


# ---------------------------------------------------------------------------
# Formatting helpers
# ---------------------------------------------------------------------------

def _h(title: str) -> str:
    return "=" * 78 + "\n " + title + "\n" + "=" * 78

def _hline() -> str:
    return "-" * 78

def fmt_p(p: float) -> str:
    if p is None or (isinstance(p, float) and math.isnan(p)):
        return "  nan"
    if p < 0.001:
        return "<0.001"
    return f"{p:.4f}"

def fmt_f(x: float, w: int = 8, prec: int = 4) -> str:
    if x is None or (isinstance(x, float) and math.isnan(x)):
        return f"{'nan':>{w}}"
    return f"{x:>{w}.{prec}f}"


# ---------------------------------------------------------------------------
# Data prep
# ---------------------------------------------------------------------------

def prepare_dataframe(metric: str) -> pd.DataFrame:
    """Load per-sample data and project onto the requested metric column."""
    df = load_per_sample_kill_rates()
    if metric != "kill_rate":
        if metric not in df.columns:
            raise SystemExit(
                f"--metric {metric!r} not in dataframe columns. "
                f"Available metrics: kill_rate, " +
                ", ".join(f"kill_rate_{op}" for op in OPERATORS)
            )
        df = df.dropna(subset=[metric]).copy()
        df["kill_rate"] = df[metric]
    # Cast to plain Python str so statsmodels' formula parser is happy with
    # the model + sample_idx categorical levels.
    df["method"]    = df["method"].astype(str)
    df["model"]     = df["model"].astype(str)
    df["sample_idx"] = df["sample_idx"].astype(str)
    return df


# ---------------------------------------------------------------------------
# Test 1 — Type-III ANOVA + Tukey HSD on method
# ---------------------------------------------------------------------------

def run_anova_and_tukey(df: pd.DataFrame) -> dict:
    """Fit OLS with method, model, sample_idx as fixed factors. Type-III ANOVA."""
    formula = "kill_rate ~ C(method) + C(model) + C(sample_idx)"
    ols = smf.ols(formula, data=df).fit()
    aov = sm.stats.anova_lm(ols, typ=3)

    tukey = pairwise_tukeyhsd(
        endog=df["kill_rate"].values,
        groups=df["method"].values,
        alpha=0.05,
    )
    tukey_df = pd.DataFrame(data=tukey._results_table.data[1:],
                            columns=tukey._results_table.data[0])
    return {"ols": ols, "anova": aov, "tukey_df": tukey_df}


# ---------------------------------------------------------------------------
# Test 2 — Mixed-LM with sample_idx as variance component
# ---------------------------------------------------------------------------

def run_mixed_lm(df: pd.DataFrame) -> dict:
    """
    Mixed model:
        kill_rate ~ C(method) + C(model)        # fixed effects
        groups = sample_idx                       # random intercept per sample

    Model is a fixed factor (only 4 specific LLMs, not a sample from a
    population), sample_idx is the natural random-effects unit (30 source
    problems, sampled from HumanEval+MBPP).
    """
    try:
        mlm = mixedlm(
            "kill_rate ~ C(method) + C(model)",
            data=df,
            groups=df["sample_idx"],
        ).fit(method="lbfgs")
    except Exception as e:
        return {"error": f"Mixed-LM failed: {e}", "mlm": None}
    return {"mlm": mlm}


# ---------------------------------------------------------------------------
# Report writer
# ---------------------------------------------------------------------------

def write_report(metric: str, df: pd.DataFrame,
                 anova_res: dict, mlm_res: dict) -> str:
    lines = []
    lines.append(_h(f"MUTATION MIXED-EFFECTS REPORT  (metric = {metric})"))
    lines.append("")
    lines.append(f"Observations: {len(df)}")
    lines.append(f"Models      : {sorted(df['model'].unique())}")
    lines.append(f"Methods     : {sorted(df['method'].unique())}")
    lines.append(f"Source samples (sample_idx levels): {df['sample_idx'].nunique()}")
    lines.append("")

    # ---- Descriptive table -------------------------------------------------
    lines.append(_h("1. DESCRIPTIVES (per method × model)"))
    grp = df.groupby(["model", "method"])["kill_rate"].agg(["count", "mean", "std"])
    lines.append(f"  {'model':<22}{'method':<22}{'n':>5}{'mean':>9}{'std':>9}")
    lines.append(_hline())
    for (m, meth), row in grp.iterrows():
        lines.append(
            f"  {m:<22}{METHOD_LABELS.get(meth, meth):<22}"
            f"{int(row['count']):>5}"
            f"{row['mean']:>9.4f}{(row['std'] if not math.isnan(row['std']) else 0):>9.4f}"
        )
    lines.append("")

    # ---- ANOVA -------------------------------------------------------------
    lines.append(_h("2. TYPE-III ANOVA  (kill_rate ~ method + model + sample_idx)"))
    lines.append("  Method-effect F-test is the omnibus test of interest.")
    lines.append("")
    aov = anova_res["anova"]
    lines.append(f"  {'term':<22}{'sum_sq':>14}{'df':>6}{'F':>10}{'p':>10}{'  sig':>5}")
    lines.append(_hline())
    for term in aov.index:
        row = aov.loc[term]
        ss   = row.get("sum_sq", float("nan"))
        ddf  = row.get("df", float("nan"))
        Fval = row.get("F", float("nan"))
        p    = row.get("PR(>F)", float("nan"))
        sig  = "*" if (not math.isnan(p) and p < 0.05) else " "
        if isinstance(ddf, float) and not math.isnan(ddf):
            ddf_str = f"{int(ddf):>6}"
        else:
            ddf_str = f"{'':>6}"
        lines.append(
            f"  {term:<22}{ss:>14.4f}{ddf_str}{fmt_f(Fval, 10)}{fmt_p(p):>10}{sig:>5}"
        )
    lines.append("")

    # ---- Tukey HSD ---------------------------------------------------------
    lines.append(_h("3. TUKEY HSD POST-HOC  (method, family-wise α = 0.05)"))
    lines.append("  Tukey controls the family-wise error rate over all 6 method pairs.")
    lines.append("")
    tdf = anova_res["tukey_df"]
    if not tdf.empty:
        lines.append(f"  {'group1':<22}{'group2':<22}"
                     f"{'meandiff':>10}{'p_adj':>10}{'lower':>10}{'upper':>10}{'  reject':>9}")
        lines.append(_hline())
        for _, row in tdf.iterrows():
            g1 = METHOD_LABELS.get(row["group1"], row["group1"])
            g2 = METHOD_LABELS.get(row["group2"], row["group2"])
            p_adj = float(row["p-adj"]) if not pd.isna(row["p-adj"]) else float("nan")
            reject = "True" if bool(row["reject"]) else "False"
            lines.append(
                f"  {g1:<22}{g2:<22}"
                f"{float(row['meandiff']):>10.4f}{fmt_p(p_adj):>10}"
                f"{float(row['lower']):>10.4f}{float(row['upper']):>10.4f}{reject:>9}"
            )
    lines.append("")

    # ---- Mixed-LM ----------------------------------------------------------
    lines.append(_h("4. MIXED-LM  (sample_idx as random intercept)"))
    lines.append("  Random:  intercept | sample_idx  (30 source problems)")
    lines.append("  Fixed :  C(method) + C(model)")
    lines.append("  REML estimation; method t-tests use Satterthwaite df.")
    lines.append("")
    if mlm_res.get("error"):
        lines.append("  " + mlm_res["error"])
    else:
        mlm = mlm_res["mlm"]
        lines.append(f"  AIC = {mlm.aic:.2f}   BIC = {mlm.bic:.2f}   "
                     f"log-likelihood = {mlm.llf:.2f}")
        lines.append(f"  Variance(intercept | sample_idx) = "
                     f"{mlm.cov_re.iloc[0,0]:.6f}")
        lines.append(f"  Residual variance                = {mlm.scale:.6f}")
        lines.append("")
        # Fixed-effects coefficient table
        params = mlm.params
        ses    = mlm.bse
        zs     = mlm.tvalues
        ps     = mlm.pvalues
        lines.append(f"  {'term':<40}{'coef':>10}{'se':>10}{'z':>9}"
                     f"{'p':>10}{'  sig':>5}")
        lines.append(_hline())
        for term in params.index:
            sig = "*" if (not math.isnan(ps[term]) and ps[term] < 0.05) else " "
            lines.append(
                f"  {term:<40}{params[term]:>10.4f}{ses[term]:>10.4f}"
                f"{zs[term]:>9.3f}{fmt_p(ps[term]):>10}{sig:>5}"
            )
        lines.append("")

    # ---- Interpretation ----------------------------------------------------
    lines.append(_h("INTERPRETATION"))
    aov = anova_res["anova"]
    method_p = aov.loc["C(method)"]["PR(>F)"] if "C(method)" in aov.index else float("nan")
    method_F = aov.loc["C(method)"]["F"] if "C(method)" in aov.index else float("nan")
    if not math.isnan(method_p) and method_p < 0.05:
        lines.append(f"  Method has a significant main effect: F={method_F:.3f}, "
                     f"p={fmt_p(method_p)}.")
        rejected = anova_res["tukey_df"]
        sig_pairs = rejected[rejected["reject"] == True]
        if not sig_pairs.empty:
            lines.append(f"  Tukey HSD rejects H₀ for {len(sig_pairs)} method pair(s):")
            for _, row in sig_pairs.iterrows():
                g1 = METHOD_LABELS.get(row["group1"], row["group1"])
                g2 = METHOD_LABELS.get(row["group2"], row["group2"])
                lines.append(f"    - {g1} vs {g2}  "
                             f"(Δ={float(row['meandiff']):+.4f}, "
                             f"p={fmt_p(float(row['p-adj']))})")
    else:
        lines.append(f"  Method's main F-test is NOT significant at α=0.05  "
                     f"(F={method_F:.3f}, p={fmt_p(method_p)}).")
        lines.append("  The Tukey HSD post-hoc still reports raw pair-wise estimates,")
        lines.append("  but family-wise α prevents calling any pair significant.")
    lines.append("")
    if mlm_res.get("mlm") is not None:
        mlm = mlm_res["mlm"]
        method_terms = [t for t in mlm.params.index if t.startswith("C(method)")]
        n_sig = sum(1 for t in method_terms if mlm.pvalues[t] < 0.05)
        lines.append(f"  Mixed-LM contrasts (vs plain_llm baseline): "
                     f"{n_sig}/{len(method_terms)} method dummies have p<0.05.")
    lines.append("")

    return "\n".join(lines)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--metric", default="kill_rate",
                        help="Per-sample column to model. 'kill_rate' (default) "
                             "or one of kill_rate_arithmetic, kill_rate_boundary, "
                             "kill_rate_comparison, kill_rate_negate_bool, "
                             "kill_rate_return_none")
    parser.add_argument("--out", default=str(REPORT_FILE),
                        help="Report output path")
    args = parser.parse_args()

    OUTPUT_DIR.mkdir(exist_ok=True)

    df = prepare_dataframe(args.metric)
    print(f"Loaded {len(df)} observations for metric={args.metric}")
    print(f"  models  : {sorted(df['model'].unique())}")
    print(f"  methods : {sorted(df['method'].unique())}")
    print(f"  samples : {df['sample_idx'].nunique()}")
    print()

    print("Fitting Type-III ANOVA...")
    anova_res = run_anova_and_tukey(df)

    print("Fitting Mixed-LM (sample_idx as random intercept)...")
    mlm_res = run_mixed_lm(df)

    report = write_report(args.metric, df, anova_res, mlm_res)
    Path(args.out).parent.mkdir(exist_ok=True)
    Path(args.out).write_text(report)
    print(f"\nReport written → {args.out}")

    # Echo headline
    print()
    print("=" * 60)
    print(f"HEADLINE — {args.metric}")
    print("=" * 60)
    aov = anova_res["anova"]
    if "C(method)" in aov.index:
        Fval = aov.loc["C(method)"]["F"]
        p = aov.loc["C(method)"]["PR(>F)"]
        verdict = "REJECT H₀" if (not math.isnan(p) and p < 0.05) \
            else "FAIL TO REJECT"
        print(f"ANOVA method main effect: F={Fval:.3f}  p={fmt_p(p)}   ({verdict})")
    tdf = anova_res["tukey_df"]
    sig_pairs = tdf[tdf["reject"] == True]
    print(f"Tukey HSD significant method pairs: {len(sig_pairs)}/{len(tdf)}")
    for _, row in sig_pairs.iterrows():
        g1 = METHOD_LABELS.get(row["group1"], row["group1"])
        g2 = METHOD_LABELS.get(row["group2"], row["group2"])
        print(f"  {g1:<20} vs {g2:<20} "
              f"Δ={float(row['meandiff']):+.4f}  p_adj={fmt_p(float(row['p-adj']))}")

    return 0


if __name__ == "__main__":
    sys.exit(main())
