"""
analyze_mutation_generalizability.py — Cross-model generalizability for
mutation kill rate (mirrors analyze_generalizability.py for val_score).

Question: do method rankings produced by mutation kill rate hold across LLMs?
If Spearman ρ between every pair of models is ≥ 0.8, the kill-rate finding
is model-agnostic. If not, the paper's claim is necessarily conditioned on
the LLM.

Operates on the (already-merged) 16-row results_mutation.tsv with columns
method, reasoning, model, mean_kill_rate, kill_<operator>. Produces:
  - Overall ranking ρ matrix for mean_kill_rate
  - Per-operator ranking ρ matrix (boundary, arithmetic, etc.)
  - Rank-stability lines
  - Heatmap of ρ between every model pair

Output (plots_mutation/):
    mutation_rank_correlation.png      — heatmap of Spearman ρ (overall)
    mutation_rank_stability.png        — method rankings across models
    mutation_kill_rate_by_model.png    — grouped bar
    mutation_generalizability_report.txt
"""

from __future__ import annotations

import math
import sys
from itertools import combinations
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy import stats

RESULTS_CANDIDATES = [Path("results_mutation.tsv"),
                      Path("results/results_mutation.tsv")]
OUTPUT_DIR  = Path("plots_mutation")
REPORT_FILE = OUTPUT_DIR / "mutation_generalizability_report.txt"

METHODS = ["plain_llm", "random_rag", "simple_rag", "iterative_critique"]
METHOD_LABELS = {
    "plain_llm":          "Plain LLM",
    "random_rag":         "Random RAG",
    "simple_rag":         "Simple RAG",
    "iterative_critique": "Iterative Critique",
}
METHOD_COLORS = {
    "plain_llm":          "#4C72B0",
    "random_rag":         "#8172B2",
    "simple_rag":         "#DD8452",
    "iterative_critique": "#55A868",
}
PRETTY_TO_RAW = {"Plain LLM": "plain_llm",
                 "Random RAG": "random_rag",
                 "Simple RAG": "simple_rag",
                 "Iterative Critique": "iterative_critique"}

GENERALIZE_THRESHOLD = 0.8   # Same threshold as analyze_generalizability.py.

OPERATOR_COLS = ["kill_arithmetic", "kill_boundary", "kill_comparison",
                 "kill_negate_bool", "kill_return_none"]


# ---------------------------------------------------------------------------
# Data loading
# ---------------------------------------------------------------------------

def load_tsv() -> pd.DataFrame:
    best = pd.DataFrame()
    for p in RESULTS_CANDIDATES:
        if not p.exists():
            continue
        df = pd.read_csv(p, sep="\t")
        if "model" not in df.columns:
            continue
        if best.empty or df["model"].nunique() > best["model"].nunique():
            best = df
    if best.empty:
        sys.exit("ERROR: no results_mutation.tsv with a 'model' column found")
    best["method_raw"] = best["method"].map(PRETTY_TO_RAW).fillna(best["method"])
    if "reasoning" not in best.columns:
        best["reasoning"] = "base"
    return best


def score_vector(df: pd.DataFrame, model: str, metric: str) -> list[float]:
    """One value per method (canonical order), 0.0 if missing."""
    vec = []
    for m in METHODS:
        row = df[(df["model"] == model) & (df["method_raw"] == m)]
        if row.empty or row[metric].isna().all():
            vec.append(float("nan"))
        else:
            vec.append(float(row[metric].values[0]))
    return vec


def rho_matrix(df: pd.DataFrame, metric: str
               ) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Pairwise Spearman ρ between each pair of models on the given metric.

    NaN methods (e.g. an unfinished cell) are dropped pairwise — both vectors
    must have ≥2 non-NaN values for spearmanr.
    """
    models = sorted(df["model"].unique())
    n = len(models)
    rho = np.full((n, n), float("nan"))
    pval = np.full((n, n), float("nan"))
    for i in range(n):
        rho[i, i] = 1.0
    for i, j in combinations(range(n), 2):
        a = np.array(score_vector(df, models[i], metric), dtype=float)
        b = np.array(score_vector(df, models[j], metric), dtype=float)
        mask = ~(np.isnan(a) | np.isnan(b))
        if mask.sum() < 2:
            continue
        if np.nanstd(a[mask]) == 0 or np.nanstd(b[mask]) == 0:
            continue
        r, p = stats.spearmanr(a[mask], b[mask])
        rho[i, j] = rho[j, i] = float(r)
        pval[i, j] = pval[j, i] = float(p)
    rho_df  = pd.DataFrame(rho,  index=models, columns=models)
    pval_df = pd.DataFrame(pval, index=models, columns=models)
    return rho_df, pval_df


# ---------------------------------------------------------------------------
# Plots
# ---------------------------------------------------------------------------

def plot_rank_correlation_heatmap(rho_df: pd.DataFrame, pval_df: pd.DataFrame,
                                  out_path: Path, title: str) -> None:
    if rho_df.empty:
        return
    models = list(rho_df.index)
    n = len(models)
    fig, ax = plt.subplots(figsize=(6.5, 5.5))
    im = ax.imshow(rho_df.values, cmap="RdYlGn", vmin=-1, vmax=1)
    plt.colorbar(im, ax=ax, label="Spearman ρ")

    ax.set_xticks(range(n))
    ax.set_yticks(range(n))
    ax.set_xticklabels(models, rotation=30, ha="right", fontsize=9)
    ax.set_yticklabels(models, fontsize=9)
    for i in range(n):
        for j in range(n):
            r = rho_df.values[i, j]
            color = "white" if (not math.isnan(r) and abs(r) > 0.7) else "black"
            label = "" if math.isnan(r) else f"ρ={r:.2f}"
            if i != j and not math.isnan(rho_df.values[i, j]):
                p = pval_df.values[i, j]
                label += "\np<.001" if (not math.isnan(p) and p < 0.001) \
                    else f"\np={p:.3f}"
            ax.text(j, i, label, ha="center", va="center",
                    fontsize=9, fontweight="bold", color=color)
    ax.set_title(title, fontsize=12, fontweight="bold")
    plt.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close(fig)


def plot_rank_stability(df: pd.DataFrame, metric: str, out_path: Path) -> None:
    models = sorted(df["model"].unique())
    if len(models) < 2:
        return
    fig, ax = plt.subplots(figsize=(10, 5.5))
    for method in METHODS:
        ranks = []
        for model in models:
            sub = df[df["model"] == model]
            scores = {}
            for m in METHODS:
                row = sub[sub["method_raw"] == m]
                scores[m] = float(row[metric].values[0]) \
                    if not row.empty and not row[metric].isna().all() \
                    else float("-inf")
            # Rank descending — 1 = best
            order = sorted(scores, key=scores.get, reverse=True)
            try:
                ranks.append(order.index(method) + 1)
            except ValueError:
                ranks.append(float("nan"))
        ax.plot(models, ranks, "o-", linewidth=2.5, markersize=9,
                color=METHOD_COLORS[method], label=METHOD_LABELS[method])
        for x_pos, rk in enumerate(ranks):
            if not (isinstance(rk, float) and math.isnan(rk)):
                ax.text(x_pos, rk - 0.12, str(int(rk)), ha="center",
                        fontsize=8, fontweight="bold")
    ax.invert_yaxis()
    ax.set_yticks([1, 2, 3, 4])
    ax.set_ylabel("Rank (1 = best)", fontsize=11)
    ax.set_xlabel("Model", fontsize=11)
    ax.set_title(f"Method ranking stability across models — {metric}\n"
                 "(Flat lines = rankings generalize)",
                 fontsize=12, fontweight="bold")
    ax.legend(title="Method", fontsize=9, title_fontsize=9, loc="best")
    ax.grid(axis="y", alpha=0.3)
    plt.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close(fig)


def plot_grouped_bars(df: pd.DataFrame, metric: str, out_path: Path) -> None:
    models = sorted(df["model"].unique())
    if len(models) < 2:
        return
    x = np.arange(len(METHODS))
    width = 0.8 / len(models)
    fig, ax = plt.subplots(figsize=(11, 5.5))
    for i, model in enumerate(models):
        vals = [float(df[(df["model"] == model) &
                         (df["method_raw"] == m)][metric].values[0])
                if not df[(df["model"] == model) &
                          (df["method_raw"] == m)].empty
                else float("nan")
                for m in METHODS]
        offset = (i - len(models) / 2 + 0.5) * width
        bars = ax.bar(x + offset, vals, width * 0.9, label=model, alpha=0.88)
        for bar, v in zip(bars, vals):
            if not math.isnan(v):
                ax.text(bar.get_x() + bar.get_width() / 2,
                        bar.get_height() + 0.005,
                        f"{v:.3f}", ha="center", va="bottom", fontsize=7)
    ax.set_xticks(x)
    ax.set_xticklabels([METHOD_LABELS[m] for m in METHODS], fontsize=11)
    ax.set_ylabel(metric, fontsize=11)
    ax.set_ylim(0, 1.1)
    ax.set_title(f"{metric} by Method × Model", fontsize=13, fontweight="bold")
    ax.legend(title="Model", fontsize=9, title_fontsize=9)
    ax.grid(axis="y", alpha=0.3)
    plt.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close(fig)


# ---------------------------------------------------------------------------
# Report
# ---------------------------------------------------------------------------

def _verdict_line(rho_df: pd.DataFrame, label: str) -> str:
    """Min off-diagonal ρ → verdict."""
    if rho_df.empty:
        return f"  {label}: (no data)"
    n = len(rho_df)
    vals = []
    for i in range(n):
        for j in range(i + 1, n):
            r = rho_df.values[i, j]
            if not math.isnan(r):
                vals.append(r)
    if not vals:
        return f"  {label}: (insufficient pairs)"
    min_rho = min(vals)
    mean_rho = sum(vals) / len(vals)
    verdict = "GENERALIZES" if min_rho >= GENERALIZE_THRESHOLD \
        else "DOES NOT FULLY GENERALIZE"
    return (f"  {label:<28}  min ρ = {min_rho:+.3f}   "
            f"mean ρ = {mean_rho:+.3f}   → {verdict}")


def write_report(df: pd.DataFrame,
                 overall_rho: pd.DataFrame, overall_pval: pd.DataFrame,
                 op_rhos: dict) -> str:
    lines = []
    models = sorted(df["model"].unique())
    lines.append("=" * 78)
    lines.append("  MUTATION KILL RATE — CROSS-MODEL GENERALIZABILITY REPORT")
    lines.append("=" * 78)
    lines.append(f"\nModels tested: {', '.join(models)}")
    lines.append(f"Methods      : {', '.join(METHODS)}")
    lines.append(f"Threshold    : Spearman ρ ≥ {GENERALIZE_THRESHOLD} "
                 f"(Zar 1984; Jureczko & Madeyski 2015 IST)\n")

    # Score table (overall)
    lines.append("Mean kill rate per (method × model):")
    lines.append(f"  {'Method':<22}" + "".join(f"{m:<22}" for m in models))
    lines.append("  " + "-" * (22 + 22 * len(models)))
    for method in METHODS:
        row = f"  {METHOD_LABELS[method]:<22}"
        for model in models:
            r = df[(df["model"] == model) & (df["method_raw"] == method)]
            val = float(r["mean_kill_rate"].values[0]) if not r.empty else float("nan")
            row += f"{val:<22.4f}" if not math.isnan(val) else f"{'nan':<22}"
        lines.append(row)
    lines.append("")

    # Rank table (overall)
    lines.append("Method rankings per model on mean_kill_rate (1 = best):")
    lines.append(f"  {'Method':<22}" + "".join(f"{m:<22}" for m in models))
    lines.append("  " + "-" * (22 + 22 * len(models)))
    for method in METHODS:
        row = f"  {METHOD_LABELS[method]:<22}"
        for model in models:
            sub = df[df["model"] == model]
            scores = {}
            for m in METHODS:
                r = sub[sub["method_raw"] == m]
                scores[m] = float(r["mean_kill_rate"].values[0]) \
                    if not r.empty else float("-inf")
            order = sorted(scores, key=scores.get, reverse=True)
            try:
                row += f"{order.index(method) + 1:<22}"
            except ValueError:
                row += f"{'-':<22}"
        lines.append(row)
    lines.append("")

    # Overall ρ matrix
    lines.append("Spearman ρ between models on mean_kill_rate:")
    lines.append("  " + overall_rho.round(3).to_string().replace("\n", "\n  "))
    lines.append("\nSpearman p-values (two-sided):")
    lines.append("  " + overall_pval.round(4).to_string().replace("\n", "\n  "))
    lines.append("")

    # Verdicts
    lines.append("VERDICTS (using min off-diagonal ρ):")
    lines.append(_verdict_line(overall_rho, "overall (mean_kill_rate)"))
    for op_col, op_rho in op_rhos.items():
        lines.append(_verdict_line(op_rho, op_col))
    lines.append("")

    lines.append("Per-operator ρ matrices:")
    for op_col, op_rho in op_rhos.items():
        lines.append(f"\n  [{op_col}]")
        lines.append("  " + op_rho.round(3).to_string().replace("\n", "\n  "))
    lines.append("")

    lines.append("=" * 78)
    lines.append("INTERPRETATION")
    lines.append("=" * 78)
    lines.append("  - ρ ≥  0.8: strong agreement; ranking is model-agnostic.")
    lines.append("  - 0.6 ≤ ρ < 0.8: moderate agreement; some inversion.")
    lines.append("  - 0.0 ≤ ρ < 0.6: weak agreement; ranking depends on the LLM.")
    lines.append("  - ρ <  0.0: inverted ranking; methods rank inversely across LLMs.")
    lines.append("")
    lines.append("  A failing overall verdict alongside a passing boundary verdict")
    lines.append("  (or vice versa) is a useful finding: it identifies which")
    lines.append("  defect-type's ordering is universal vs. LLM-dependent.")
    lines.append("")

    return "\n".join(lines)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> int:
    df = load_tsv()
    if df["model"].nunique() < 2:
        print("ERROR: need ≥2 models in results_mutation.tsv")
        return 1

    OUTPUT_DIR.mkdir(exist_ok=True)

    print(f"Loaded {len(df)} rows, models = {sorted(df['model'].unique())}")

    # Overall analysis
    overall_rho, overall_pval = rho_matrix(df, "mean_kill_rate")
    plot_rank_correlation_heatmap(
        overall_rho, overall_pval,
        OUTPUT_DIR / "mutation_rank_correlation.png",
        "Spearman ρ across Models — mean_kill_rate",
    )
    plot_rank_stability(df, "mean_kill_rate",
                        OUTPUT_DIR / "mutation_rank_stability.png")
    plot_grouped_bars(df, "mean_kill_rate",
                      OUTPUT_DIR / "mutation_kill_rate_by_model.png")

    # Per-operator analyses
    op_rhos = {}
    for op_col in OPERATOR_COLS:
        if op_col not in df.columns:
            continue
        op_rho, _op_pval = rho_matrix(df, op_col)
        op_rhos[op_col] = op_rho
        plot_rank_correlation_heatmap(
            op_rho, _op_pval,
            OUTPUT_DIR / f"mutation_rank_correlation_{op_col}.png",
            f"Spearman ρ across Models — {op_col}",
        )

    report = write_report(df, overall_rho, overall_pval, op_rhos)
    REPORT_FILE.write_text(report)
    print(f"\nReport → {REPORT_FILE}")
    print(f"Plots  → {OUTPUT_DIR}/mutation_rank_correlation*.png, "
          f"mutation_rank_stability.png, mutation_kill_rate_by_model.png")

    # Headline
    print()
    print("=" * 60)
    print("HEADLINE — generalizability verdicts (Spearman ρ ≥ 0.8)")
    print("=" * 60)
    print(_verdict_line(overall_rho, "overall (mean_kill_rate)"))
    for op_col, op_rho in op_rhos.items():
        print(_verdict_line(op_rho, op_col))

    return 0


if __name__ == "__main__":
    sys.exit(main())
