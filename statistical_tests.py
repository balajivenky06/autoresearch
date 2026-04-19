"""
statistical_tests.py — Publication-quality statistical analysis for PhD thesis.

Mirrors the statistical methodology from the companion Springer paper:
  Kruskal-Wallis H-test → pairwise Mann-Whitney U with Bonferroni correction
  → Cohen's d effect sizes.

Also performs val_score weight sensitivity analysis (RQ1 robustness).

Usage:
    python statistical_tests.py
    python statistical_tests.py --results results_unitest.tsv

Output (plots_generalizability/):
    statistical_report.txt  — full significance table for thesis appendix
    sensitivity_weights.png — val_score rank stability across weight perturbations
"""

import sys
import argparse
import math
import itertools
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path
from scipy import stats

RESULTS_FILE = "results_unitest.tsv"
OUTPUT_DIR   = Path("plots_generalizability")

METHODS = ["plain_llm", "random_rag", "simple_rag", "iterative_critique"]
METHOD_LABELS = {
    "plain_llm":          "Plain LLM",
    "random_rag":         "Random RAG",
    "simple_rag":         "Simple RAG",
    "iterative_critique": "Iterative Critique",
}

# ---------------------------------------------------------------------------
# Data loading
# ---------------------------------------------------------------------------

def load_results(path: str) -> pd.DataFrame:
    df = pd.read_csv(path, sep="\t")
    if df.empty:
        print("ERROR: results file is empty.")
        sys.exit(1)

    if "/" in str(df["method"].iloc[0]):
        df[["method_name", "reasoning"]] = df["method"].str.split("/", n=1, expand=True)
    else:
        df["method_name"] = df["method"]
        df["reasoning"]   = "base"

    df = df[df["status"] != "crash"].copy()
    for col in ["val_score", "avg_faithfulness", "avg_llm_judge_faithfulness",
                "avg_noise_rate", "avg_llm_secs", "avg_retrieval_secs"]:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce")

    return df


# ---------------------------------------------------------------------------
# Helper: Cohen's d (pooled SD)
# ---------------------------------------------------------------------------

def cohens_d(a: list, b: list) -> float:
    """Pooled Cohen's d effect size. Returns NaN if insufficient data."""
    if len(a) < 2 or len(b) < 2:
        return float("nan")
    mean_a, mean_b = np.mean(a), np.mean(b)
    var_a,  var_b  = np.var(a, ddof=1), np.var(b, ddof=1)
    pooled_sd = math.sqrt(((len(a) - 1) * var_a + (len(b) - 1) * var_b) /
                           (len(a) + len(b) - 2))
    if pooled_sd == 0:
        return float("nan")
    return (mean_a - mean_b) / pooled_sd


def effect_label(d: float) -> str:
    """Cohen's d magnitude label (Cohen 1988)."""
    if math.isnan(d):
        return "N/A"
    a = abs(d)
    if a < 0.2:
        return "negligible"
    if a < 0.5:
        return "small"
    if a < 0.8:
        return "medium"
    return "large"


# ---------------------------------------------------------------------------
# Core statistical tests
# ---------------------------------------------------------------------------

def run_kruskal_wallis(df: pd.DataFrame, metric: str) -> dict:
    """
    Kruskal-Wallis H-test across all methods for a given metric.
    Returns dict with H, p, df (degrees of freedom).
    """
    groups = []
    for method in METHODS:
        vals = df[df["method_name"] == method][metric].dropna().tolist()
        groups.append(vals)

    # Need at least 2 groups with data
    non_empty = [g for g in groups if len(g) >= 2]
    if len(non_empty) < 2:
        return {"H": float("nan"), "p": float("nan"), "df": len(METHODS) - 1}

    try:
        H, p = stats.kruskal(*non_empty)
        return {"H": H, "p": p, "df": len(non_empty) - 1}
    except ValueError:
        return {"H": float("nan"), "p": float("nan"), "df": len(METHODS) - 1}


def run_pairwise_mannwhitney(df: pd.DataFrame, metric: str) -> pd.DataFrame:
    """
    Pairwise Mann-Whitney U tests with Bonferroni correction.
    Returns DataFrame with columns: method_a, method_b, U, p_raw, p_adj, d, effect
    """
    pairs = list(itertools.combinations(METHODS, 2))
    n_comparisons = len(pairs)
    rows = []

    for m_a, m_b in pairs:
        vals_a = df[df["method_name"] == m_a][metric].dropna().tolist()
        vals_b = df[df["method_name"] == m_b][metric].dropna().tolist()

        if len(vals_a) < 2 or len(vals_b) < 2:
            rows.append({
                "method_a": m_a, "method_b": m_b,
                "U": float("nan"), "p_raw": float("nan"),
                "p_adj": float("nan"), "d": float("nan"), "effect": "N/A"
            })
            continue

        U, p_raw = stats.mannwhitneyu(vals_a, vals_b, alternative="two-sided")
        p_adj = min(p_raw * n_comparisons, 1.0)   # Bonferroni
        d = cohens_d(vals_a, vals_b)
        rows.append({
            "method_a": m_a, "method_b": m_b,
            "U": U, "p_raw": round(p_raw, 6), "p_adj": round(p_adj, 6),
            "d": round(d, 4) if not math.isnan(d) else float("nan"),
            "effect": effect_label(d),
        })

    return pd.DataFrame(rows)


# ---------------------------------------------------------------------------
# Val_score weight sensitivity analysis
# ---------------------------------------------------------------------------

# Baseline weights (from prepare_unitest.py)
BASELINE_WEIGHTS = {
    "syntactic_validity": 0.30,
    "edge_case_score":    0.25,
    "assert_density":     0.20,
    "semantic_sim":       0.15,
    "rouge_1_f1":         0.10,
}

# Weight perturbation scenarios (vary each weight ±50%, renormalize)
def _build_weight_scenarios() -> dict:
    scenarios = {"baseline": BASELINE_WEIGHTS.copy()}
    for key in BASELINE_WEIGHTS:
        for direction, factor in [("high", 1.5), ("low", 0.5)]:
            w = BASELINE_WEIGHTS.copy()
            w[key] = BASELINE_WEIGHTS[key] * factor
            total = sum(w.values())
            w = {k: v / total for k, v in w.items()}
            scenarios[f"{key}_{direction}"] = w
    return scenarios


def _compute_val_score(df_row: pd.Series, weights: dict) -> float:
    """Recompute val_score for a row using given weights."""
    col_map = {
        "syntactic_validity": "avg_syntax",
        "edge_case_score":    "avg_edge",
        "assert_density":     "avg_assert_density",
        "semantic_sim":       "avg_semantic_sim",
        "rouge_1_f1":         "avg_rouge",
    }
    score = 0.0
    for key, col in col_map.items():
        if col in df_row.index and not pd.isna(df_row[col]):
            score += weights[key] * float(df_row[col])
    return score


def run_sensitivity_analysis(df: pd.DataFrame) -> pd.DataFrame:
    """
    Vary val_score weights ±50% (one at a time), recompute val_scores,
    check whether method ranking changes.
    """
    # Aggregate to best-per-method/model (same as analyze_generalizability)
    best_rows = []
    for model in df["model"].unique():
        for method in METHODS:
            sub = df[(df["model"] == model) & (df["method_name"] == method)]
            if sub.empty:
                continue
            if sub["val_score"].notna().any():
                best_rows.append(sub.loc[sub["val_score"].idxmax()])
            else:
                best_rows.append(sub.iloc[0])
    best = pd.DataFrame(best_rows)

    scenarios = _build_weight_scenarios()
    records = []

    for scenario_name, weights in scenarios.items():
        for model in best["model"].unique():
            model_rows = best[best["model"] == model]
            method_scores = {}
            for method in METHODS:
                row = model_rows[model_rows["method_name"] == method]
                if row.empty:
                    method_scores[method] = float("nan")
                else:
                    method_scores[method] = _compute_val_score(row.iloc[0], weights)

            sorted_methods = sorted(
                [m for m in METHODS if not math.isnan(method_scores.get(m, float("nan")))],
                key=lambda m: method_scores[m], reverse=True
            )
            for rank, method in enumerate(sorted_methods, 1):
                records.append({
                    "scenario": scenario_name,
                    "model":    model,
                    "method":   method,
                    "rank":     rank,
                    "score":    method_scores[method],
                })

    return pd.DataFrame(records)


def plot_sensitivity(sensitivity_df: pd.DataFrame) -> None:
    if sensitivity_df.empty:
        print("  sensitivity_weights.png SKIPPED — no data")
        return

    scenarios = sensitivity_df["scenario"].unique()
    methods   = METHODS
    models    = sensitivity_df["model"].unique()

    fig, axes = plt.subplots(1, len(models), figsize=(5 * len(models), 6), sharey=True)
    if len(models) == 1:
        axes = [axes]

    colors = {"plain_llm": "#4C72B0", "random_rag": "#8172B2", "simple_rag": "#DD8452", "iterative_critique": "#55A868"}

    for ax, model in zip(axes, models):
        sub = sensitivity_df[sensitivity_df["model"] == model]
        for method in methods:
            method_sub = sub[sub["method"] == method]
            ranks = [method_sub[method_sub["scenario"] == s]["rank"].values[0]
                     if len(method_sub[method_sub["scenario"] == s]) > 0 else None
                     for s in scenarios]
            ax.plot(range(len(scenarios)), ranks, "o-", color=colors[method],
                    label=METHOD_LABELS[method], linewidth=1.5, markersize=5)

        ax.set_title(model, fontsize=10)
        ax.set_xticks(range(len(scenarios)))
        ax.set_xticklabels(scenarios, rotation=45, ha="right", fontsize=7)
        n_methods = len(METHODS)
        ax.set_yticks(range(1, n_methods + 1))
        labels = []
        for i in range(1, n_methods + 1):
            suffix = {1: "st", 2: "nd", 3: "rd"}.get(i, "th")
            labels.append(f"{i}{suffix}")
        ax.set_yticklabels(labels, fontsize=9)
        ax.invert_yaxis()
        ax.grid(axis="y", alpha=0.3)

    axes[0].set_ylabel("Rank (val_score)", fontsize=11)
    axes[-1].legend(fontsize=8)
    fig.suptitle("Val_score Weight Sensitivity Analysis\n(Rank stability across ±50% weight perturbations)",
                 fontsize=12, fontweight="bold")
    plt.tight_layout()
    fig.savefig(OUTPUT_DIR / "sensitivity_weights.png", dpi=150)
    plt.close(fig)
    print("  sensitivity_weights.png")


# ---------------------------------------------------------------------------
# Method × Reasoning interaction analysis
# ---------------------------------------------------------------------------

def run_interaction_analysis(df: pd.DataFrame) -> pd.DataFrame:
    """
    Compute mean val_score for each Method × Reasoning cell.
    Returns a pivot table; the write_report function detects crossing interactions.
    A 'crossing interaction' exists when the best reasoning mode differs
    between methods — indicating method × reasoning is not additive.
    """
    pivot = df.pivot_table(
        index="method_name", columns="reasoning",
        values="val_score", aggfunc="mean"
    ).round(4)
    return pivot


def detect_interactions(pivot: pd.DataFrame) -> list:
    """
    Detect crossing interactions: for each pair of methods, check if
    their best reasoning mode differs. Returns list of (method_a, method_b, note).
    """
    findings = []
    methods = [m for m in pivot.index if m in pivot.index]
    for m_a, m_b in itertools.combinations(methods, 2):
        if m_a not in pivot.index or m_b not in pivot.index:
            continue
        best_a = pivot.loc[m_a].idxmax() if not pivot.loc[m_a].isna().all() else None
        best_b = pivot.loc[m_b].idxmax() if not pivot.loc[m_b].isna().all() else None
        if best_a and best_b and best_a != best_b:
            findings.append((m_a, m_b,
                f"best reasoning differs: {METHOD_LABELS[m_a]}→{best_a}, "
                f"{METHOD_LABELS[m_b]}→{best_b}"))
        elif best_a and best_b:
            findings.append((m_a, m_b, f"no crossing: both best at {best_a}"))
    return findings


# ---------------------------------------------------------------------------
# Per-source (HumanEval vs MBPP) analysis
# ---------------------------------------------------------------------------

def run_source_analysis(df: pd.DataFrame) -> dict:
    """
    Compare val_score on HumanEval vs MBPP subsets per method.
    Requires val_score_humaneval / val_score_mbpp columns in TSV.
    Addresses reviewer concern that results could be dataset-specific.
    Returns dict with per-method scores for report.
    """
    has_cols = ("val_score_humaneval" in df.columns and "val_score_mbpp" in df.columns)
    if not has_cols:
        return {}

    df = df.copy()
    df["val_score_humaneval"] = pd.to_numeric(df["val_score_humaneval"], errors="coerce")
    df["val_score_mbpp"]      = pd.to_numeric(df["val_score_mbpp"],      errors="coerce")

    result = {}
    for method in METHODS:
        sub = df[df["method_name"] == method]
        if sub.empty:
            continue
        best = sub.loc[sub["val_score"].idxmax()]
        result[method] = {
            "humaneval": float(best.get("val_score_humaneval", float("nan"))),
            "mbpp":      float(best.get("val_score_mbpp",      float("nan"))),
            "classeval": float(best.get("val_score_classeval", float("nan"))),
            "overall":   float(best.get("val_score",           float("nan"))),
        }
    return result


# ---------------------------------------------------------------------------
# Report writer
# ---------------------------------------------------------------------------

def write_statistical_report(df: pd.DataFrame, sensitivity_df: pd.DataFrame) -> None:
    lines = []
    lines.append("=" * 70)
    lines.append("  STATISTICAL SIGNIFICANCE REPORT — Unit Test Generation")
    lines.append("  Methods: Kruskal-Wallis → Pairwise Mann-Whitney U (Bonferroni)")
    lines.append("  Effect sizes: Cohen's d (Cohen 1988)")
    lines.append("=" * 70)

    metrics_to_test = [
        ("val_score",       "Primary: val_score"),
        ("avg_faithfulness","Faithfulness (token-overlap)"),
    ]
    if "avg_llm_judge_faithfulness" in df.columns:
        metrics_to_test.append(("avg_llm_judge_faithfulness", "Faithfulness (LLM-judge)"))
    if "avg_exec_pass_rate" in df.columns:
        metrics_to_test.append(("avg_exec_pass_rate", "Execution pass rate (diagnostic)"))

    for model in sorted(df["model"].unique()):
        model_df = df[df["model"] == model]
        lines.append(f"\n{'─' * 70}")
        lines.append(f"  Model: {model}  (n={len(model_df)} runs)")
        lines.append(f"{'─' * 70}")

        for col, label in metrics_to_test:
            if col not in model_df.columns:
                continue

            kw = run_kruskal_wallis(model_df, col)
            lines.append(f"\n{label}")
            lines.append(f"  Kruskal-Wallis: H={kw['H']:.3f}, df={kw['df']}, p={kw['p']:.4f}" +
                         (" *" if not math.isnan(kw['p']) and kw['p'] < 0.05 else ""))

            pw = run_pairwise_mannwhitney(model_df, col)
            lines.append(f"  {'Comparison':<42} {'U':>8} {'p_raw':>8} {'p_adj':>8}  {'d':>7}  Effect")
            lines.append("  " + "-" * 78)
            for _, row in pw.iterrows():
                sig = " *" if not math.isnan(row["p_adj"]) and row["p_adj"] < 0.05 else "  "
                d_str = f"{row['d']:.3f}" if not math.isnan(row["d"]) else "  N/A"
                lines.append(
                    f"  {METHOD_LABELS[row['method_a']]} vs {METHOD_LABELS[row['method_b']]:<25}"
                    f"  {row['U']:>8.1f}  {row['p_raw']:>8.4f}  {row['p_adj']:>8.4f}{sig}"
                    f"  {d_str:>7}  {row['effect']}"
                )

    # Sensitivity summary
    if not sensitivity_df.empty:
        lines.append(f"\n{'=' * 70}")
        lines.append("  VAL_SCORE WEIGHT SENSITIVITY ANALYSIS")
        lines.append("  (Rankings across ±50% perturbation of each weight component)")
        lines.append("=" * 70)

        for model in sorted(sensitivity_df["model"].unique()):
            lines.append(f"\nModel: {model}")
            sub = sensitivity_df[sensitivity_df["model"] == model]
            stable = True
            baseline_ranks = {}
            for method in METHODS:
                br = sub[(sub["scenario"] == "baseline") & (sub["method"] == method)]["rank"]
                baseline_ranks[method] = int(br.values[0]) if len(br) else None

            for scenario in sub["scenario"].unique():
                if scenario == "baseline":
                    continue
                for method in METHODS:
                    r = sub[(sub["scenario"] == scenario) & (sub["method"] == method)]["rank"]
                    if len(r) and baseline_ranks.get(method) and int(r.values[0]) != baseline_ranks[method]:
                        stable = False
                        lines.append(f"  RANK CHANGE: {METHOD_LABELS[method]} rank "
                                     f"{baseline_ranks[method]}→{int(r.values[0])} under scenario '{scenario}'")

            if stable:
                lines.append(f"  Rankings STABLE across all weight perturbations.")

    # Interaction analysis
    pivot = run_interaction_analysis(df)
    interactions = detect_interactions(pivot)
    lines.append(f"\n{'=' * 70}")
    lines.append("  METHOD × REASONING INTERACTION ANALYSIS")
    lines.append("  (Parallel effects → additive model; crossing → interaction)")
    lines.append("=" * 70)
    lines.append("\nMean val_score pivot (method × reasoning):")
    lines.append(pivot.to_string())
    lines.append("\nInteraction findings:")
    for m_a, m_b, note in interactions:
        lines.append(f"  {METHOD_LABELS.get(m_a, m_a)} vs {METHOD_LABELS.get(m_b, m_b)}: {note}")

    # Per-source analysis
    source_results = run_source_analysis(df)
    if source_results:
        lines.append(f"\n{'=' * 70}")
        lines.append("  DATASET SOURCE ANALYSIS: HumanEval vs MBPP")
        lines.append("  (Ensures results are not driven by one benchmark)")
        lines.append("=" * 70)
        lines.append(f"\n  {'Method':<28} {'HumanEval':>12} {'MBPP':>10} {'ClassEval':>12} {'Overall':>10}")
        lines.append("  " + "-" * 74)
        for method, scores in source_results.items():
            he = f"{scores['humaneval']:.4f}" if not math.isnan(scores['humaneval']) else "  N/A"
            mb = f"{scores['mbpp']:.4f}"      if not math.isnan(scores['mbpp'])      else "  N/A"
            ce = f"{scores['classeval']:.4f}" if not math.isnan(scores.get('classeval', float('nan'))) else "  N/A"
            ov = f"{scores['overall']:.4f}"   if not math.isnan(scores['overall'])   else "  N/A"
            lines.append(f"  {METHOD_LABELS.get(method, method):<28} {he:>12} {mb:>10} {ce:>12} {ov:>10}")

    lines.append("\n" + "=" * 70)

    report_path = OUTPUT_DIR / "statistical_report.txt"
    text = "\n".join(lines)
    report_path.write_text(text)
    print("  statistical_report.txt")
    print()
    print(text)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main(results_file: str = RESULTS_FILE) -> None:
    if not Path(results_file).exists():
        print(f"ERROR: {results_file} not found. Run experiments first.")
        sys.exit(1)

    df = load_results(results_file)
    OUTPUT_DIR.mkdir(exist_ok=True)
    print(f"Loaded {len(df)} results from {results_file}\n")

    sensitivity_df = run_sensitivity_analysis(df)
    plot_sensitivity(sensitivity_df)
    write_statistical_report(df, sensitivity_df)

    print(f"\nDone. Open {OUTPUT_DIR}/ to view outputs.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--results", default=RESULTS_FILE,
                        help="Path to results_unitest.tsv")
    args = parser.parse_args()
    main(args.results)
