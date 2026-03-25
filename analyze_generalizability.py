"""
analyze_generalizability.py — Cross-model generalizability analysis for PhD thesis.

Answers the key question: do method rankings (plain_llm vs simple_rag vs
iterative_critique) hold consistently across all tested models?

If Spearman rank correlation is high (>0.8) across all model pairs, the
findings generalize — the thesis claim is model-agnostic.

Usage:
    python analyze_generalizability.py
    python analyze_generalizability.py --results results_unitest.tsv

Output (plots_generalizability/):
    rank_correlation.png   — heatmap of Spearman ρ between all model pairs
    rank_stability.png     — method ranking lines across models
    val_score_by_model.png — val_score grouped bar: method × model
    faithfulness_by_model.png — faithfulness grouped bar: method × model
    generalizability_report.txt — written summary for thesis appendix
"""

import sys
import argparse
import math
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from pathlib import Path
from scipy import stats

# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------

RESULTS_FILE = "results_unitest.tsv"
OUTPUT_DIR   = Path("plots_generalizability")

METHODS = ["plain_llm", "simple_rag", "iterative_critique"]
METHOD_LABELS = {
    "plain_llm":          "Plain LLM",
    "simple_rag":         "Simple RAG",
    "iterative_critique": "Iterative Critique",
}
METHOD_COLORS = {
    "plain_llm":          "#4C72B0",
    "simple_rag":         "#DD8452",
    "iterative_critique": "#55A868",
}

GENERALIZE_THRESHOLD = 0.8   # Spearman ρ above this → rankings generalize


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
    for col in ["val_score", "avg_faithfulness", "avg_noise_rate",
                "avg_llm_secs", "avg_retrieval_secs"]:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce")

    return df


def _best_per_method_model(df: pd.DataFrame) -> pd.DataFrame:
    """Return one row per (method, model): highest val_score run."""
    rows = []
    for model in df["model"].unique():
        for method in METHODS:
            sub = df[(df["model"] == model) & (df["method_name"] == method)]
            if sub.empty:
                continue
            if sub["val_score"].notna().any():
                rows.append(sub.loc[sub["val_score"].idxmax()])
            else:
                rows.append(sub.iloc[0])
    return pd.DataFrame(rows) if rows else pd.DataFrame()


# ---------------------------------------------------------------------------
# Chart 1: val_score grouped bar — method × model
# ---------------------------------------------------------------------------

def plot_val_score_by_model(best: pd.DataFrame) -> None:
    models = sorted(best["model"].unique())
    if len(models) < 2:
        print("  val_score_by_model.png SKIPPED — need ≥2 models")
        return

    x     = np.arange(len(METHODS))
    width = 0.8 / len(models)

    fig, ax = plt.subplots(figsize=(10, 5))

    for i, model in enumerate(models):
        vals = []
        for method in METHODS:
            row = best[(best["model"] == model) & (best["method_name"] == method)]
            vals.append(float(row["val_score"].values[0]) if not row.empty else 0.0)

        offset = (i - len(models) / 2 + 0.5) * width
        bars = ax.bar(x + offset, vals, width * 0.9, label=model, alpha=0.85)
        for bar, v in zip(bars, vals):
            if v > 0:
                ax.text(bar.get_x() + bar.get_width() / 2,
                        bar.get_height() + 0.005,
                        f"{v:.3f}", ha="center", va="bottom", fontsize=7)

    ax.set_xticks(x)
    ax.set_xticklabels([METHOD_LABELS[m] for m in METHODS], fontsize=11)
    ax.set_ylabel("val_score", fontsize=11)
    ax.set_ylim(0, 1.1)
    ax.set_title("val_score by Method × Model\n(Best run per method/model)",
                 fontsize=13, fontweight="bold")
    ax.legend(title="Model", fontsize=9, title_fontsize=9)
    ax.grid(axis="y", alpha=0.3)
    plt.tight_layout()
    fig.savefig(OUTPUT_DIR / "val_score_by_model.png", dpi=150)
    plt.close(fig)
    print("  val_score_by_model.png")


# ---------------------------------------------------------------------------
# Chart 2: Faithfulness grouped bar — method × model
# ---------------------------------------------------------------------------

def plot_faithfulness_by_model(best: pd.DataFrame) -> None:
    if "avg_faithfulness" not in best.columns:
        print("  faithfulness_by_model.png SKIPPED — avg_faithfulness missing")
        return

    models = sorted(best["model"].unique())
    rag_methods = ["simple_rag", "iterative_critique"]

    x     = np.arange(len(rag_methods))
    width = 0.8 / len(models)

    fig, ax = plt.subplots(figsize=(8, 5))
    has_data = False

    for i, model in enumerate(models):
        vals = []
        for method in rag_methods:
            row = best[(best["model"] == model) & (best["method_name"] == method)]
            if row.empty:
                vals.append(float("nan"))
                continue
            fv = float(row["avg_faithfulness"].values[0])
            vals.append(fv if not math.isnan(fv) else float("nan"))

        offset = (i - len(models) / 2 + 0.5) * width
        for j, v in enumerate(vals):
            if not math.isnan(v):
                has_data = True
                bar = ax.bar(x[j] + offset, v, width * 0.9, label=model if j == 0 else "",
                             alpha=0.85)
                ax.text(x[j] + offset, v + 0.005, f"{v:.3f}",
                        ha="center", va="bottom", fontsize=7)

    if not has_data:
        print("  faithfulness_by_model.png SKIPPED — no faithfulness data")
        plt.close(fig)
        return

    ax.set_xticks(x)
    ax.set_xticklabels([METHOD_LABELS[m] for m in rag_methods], fontsize=11)
    ax.set_ylabel("Faithfulness (token overlap)", fontsize=11)
    ax.set_ylim(0, 1.1)
    ax.set_title("Faithfulness by Method × Model\n(RAG methods only — plain_llm=NaN)",
                 fontsize=13, fontweight="bold")
    ax.legend(title="Model", fontsize=9, title_fontsize=9)
    ax.grid(axis="y", alpha=0.3)
    plt.tight_layout()
    fig.savefig(OUTPUT_DIR / "faithfulness_by_model.png", dpi=150)
    plt.close(fig)
    print("  faithfulness_by_model.png")


# ---------------------------------------------------------------------------
# Chart 3: Rank stability — method rankings across models (line plot)
# ---------------------------------------------------------------------------

def plot_rank_stability(best: pd.DataFrame) -> None:
    models = sorted(best["model"].unique())
    if len(models) < 2:
        print("  rank_stability.png SKIPPED — need ≥2 models")
        return

    fig, ax = plt.subplots(figsize=(9, 5))

    for method in METHODS:
        ranks = []
        model_labels = []
        for model in models:
            sub = best[(best["model"] == model)]
            if sub.empty:
                continue
            # Rank methods by val_score within this model (1=best)
            model_scores = {}
            for m in METHODS:
                row = sub[sub["method_name"] == m]
                model_scores[m] = float(row["val_score"].values[0]) if not row.empty else 0.0
            sorted_methods = sorted(model_scores, key=model_scores.get, reverse=True)
            rank = sorted_methods.index(method) + 1
            ranks.append(rank)
            model_labels.append(model)

        ax.plot(model_labels, ranks, "o-", linewidth=2.5, markersize=8,
                label=METHOD_LABELS[method], color=METHOD_COLORS[method])
        for x_pos, rank in enumerate(ranks):
            ax.text(x_pos, rank - 0.08, str(rank),
                    ha="center", va="bottom", fontsize=9, fontweight="bold",
                    color=METHOD_COLORS[method])

    ax.set_yticks([1, 2, 3])
    ax.set_yticklabels(["1st (best)", "2nd", "3rd (worst)"], fontsize=10)
    ax.invert_yaxis()
    ax.set_xlabel("Model", fontsize=11)
    ax.set_ylabel("Rank (by val_score)", fontsize=11)
    ax.set_title("Method Ranking Stability Across Models\n(Flat lines = rankings generalize)",
                 fontsize=13, fontweight="bold")
    ax.legend(fontsize=10)
    ax.grid(axis="y", alpha=0.3)
    plt.tight_layout()
    fig.savefig(OUTPUT_DIR / "rank_stability.png", dpi=150)
    plt.close(fig)
    print("  rank_stability.png")


# ---------------------------------------------------------------------------
# Chart 4: Spearman rank correlation heatmap across models
# ---------------------------------------------------------------------------

def plot_rank_correlation(best: pd.DataFrame) -> pd.DataFrame:
    models = sorted(best["model"].unique())
    if len(models) < 2:
        print("  rank_correlation.png SKIPPED — need ≥2 models")
        return pd.DataFrame()

    # Build score vector per model (one val_score per method, fixed order)
    score_vecs = {}
    for model in models:
        vec = []
        for method in METHODS:
            row = best[(best["model"] == model) & (best["method_name"] == method)]
            vec.append(float(row["val_score"].values[0]) if not row.empty else 0.0)
        score_vecs[model] = vec

    # Compute Spearman ρ matrix
    n = len(models)
    corr_matrix = np.ones((n, n))
    for i in range(n):
        for j in range(i + 1, n):
            rho, _ = stats.spearmanr(score_vecs[models[i]], score_vecs[models[j]])
            corr_matrix[i, j] = corr_matrix[j, i] = rho

    corr_df = pd.DataFrame(corr_matrix, index=models, columns=models)

    fig, ax = plt.subplots(figsize=(6, 5))
    im = ax.imshow(corr_matrix, cmap="RdYlGn", vmin=-1, vmax=1)
    plt.colorbar(im, ax=ax, label="Spearman ρ")

    ax.set_xticks(range(n))
    ax.set_yticks(range(n))
    ax.set_xticklabels(models, rotation=30, ha="right", fontsize=9)
    ax.set_yticklabels(models, fontsize=9)

    for i in range(n):
        for j in range(n):
            color = "white" if abs(corr_matrix[i, j]) > 0.7 else "black"
            ax.text(j, i, f"{corr_matrix[i, j]:.2f}",
                    ha="center", va="center", fontsize=11, fontweight="bold", color=color)

    ax.set_title("Spearman Rank Correlation of Method Rankings\nAcross Models",
                 fontsize=12, fontweight="bold")
    plt.tight_layout()
    fig.savefig(OUTPUT_DIR / "rank_correlation.png", dpi=150)
    plt.close(fig)
    print("  rank_correlation.png")

    return corr_df


# ---------------------------------------------------------------------------
# Text report for thesis appendix
# ---------------------------------------------------------------------------

def write_report(best: pd.DataFrame, corr_df: pd.DataFrame) -> None:
    models = sorted(best["model"].unique())
    lines = []
    lines.append("=" * 65)
    lines.append("  GENERALIZABILITY REPORT — Unit Test Generation")
    lines.append("=" * 65)
    lines.append(f"\nModels tested: {', '.join(models)}")
    lines.append(f"Methods:       {', '.join(METHODS)}\n")

    # val_score table per model × method
    lines.append("val_score (best run per method/model):")
    lines.append(f"{'Method':<25}" + "".join(f"{m:<18}" for m in models))
    lines.append("-" * (25 + 18 * len(models)))
    for method in METHODS:
        row_str = f"{METHOD_LABELS[method]:<25}"
        for model in models:
            r = best[(best["model"] == model) & (best["method_name"] == method)]
            val = float(r["val_score"].values[0]) if not r.empty else float("nan")
            row_str += f"{val:<18.4f}"
        lines.append(row_str)

    # Rank table
    lines.append("\nMethod rankings per model (1=best):")
    lines.append(f"{'Method':<25}" + "".join(f"{m:<18}" for m in models))
    lines.append("-" * (25 + 18 * len(models)))
    for method in METHODS:
        row_str = f"{METHOD_LABELS[method]:<25}"
        for model in models:
            sub = best[best["model"] == model]
            model_scores = {}
            for m in METHODS:
                r = sub[sub["method_name"] == m]
                model_scores[m] = float(r["val_score"].values[0]) if not r.empty else 0.0
            sorted_methods = sorted(model_scores, key=model_scores.get, reverse=True)
            rank = sorted_methods.index(method) + 1 if method in sorted_methods else "-"
            row_str += f"{rank:<18}"
        lines.append(row_str)

    # Spearman correlation
    if not corr_df.empty:
        lines.append("\nSpearman rank correlation between models:")
        lines.append(corr_df.round(3).to_string())

        min_rho = corr_df.values[np.triu_indices(len(corr_df), k=1)].min()
        if min_rho >= GENERALIZE_THRESHOLD:
            verdict = f"GENERALIZES (min ρ={min_rho:.3f} ≥ {GENERALIZE_THRESHOLD})"
        else:
            verdict = f"DOES NOT FULLY GENERALIZE (min ρ={min_rho:.3f} < {GENERALIZE_THRESHOLD})"
        lines.append(f"\nVerdict: {verdict}")

    # Faithfulness consistency
    if "avg_faithfulness" in best.columns:
        lines.append("\nFaithfulness (avg) per method/model:")
        lines.append(f"{'Method':<25}" + "".join(f"{m:<18}" for m in models))
        lines.append("-" * (25 + 18 * len(models)))
        for method in ["simple_rag", "iterative_critique"]:
            row_str = f"{METHOD_LABELS[method]:<25}"
            for model in models:
                r = best[(best["model"] == model) & (best["method_name"] == method)]
                fv = float(r["avg_faithfulness"].values[0]) if not r.empty else float("nan")
                row_str += f"{'nan' if math.isnan(fv) else f'{fv:.4f}':<18}"
            lines.append(row_str)

    lines.append("\n" + "=" * 65)

    report_path = OUTPUT_DIR / "generalizability_report.txt"
    report_text = "\n".join(lines)
    report_path.write_text(report_text)
    print("  generalizability_report.txt")
    print()
    print(report_text)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main(results_file: str = RESULTS_FILE) -> None:
    if not Path(results_file).exists():
        print(f"ERROR: {results_file} not found. Run experiments first.")
        sys.exit(1)

    df = load_results(results_file)
    models = df["model"].unique()

    if len(models) < 2:
        print(f"Only 1 model found ({models[0]}). Need ≥2 models for generalizability analysis.")
        print("Run experiments with multiple models first (see Colab notebook Step 7).")
        sys.exit(0)

    best = _best_per_method_model(df)
    OUTPUT_DIR.mkdir(exist_ok=True)
    print(f"Loaded {len(df)} results across {len(models)} models → {OUTPUT_DIR}/\n")

    plot_val_score_by_model(best)
    plot_faithfulness_by_model(best)
    plot_rank_stability(best)
    corr_df = plot_rank_correlation(best)
    write_report(best, corr_df)

    print(f"\nDone. Open {OUTPUT_DIR}/ to view charts.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--results", default=RESULTS_FILE,
                        help="Path to results_unitest.tsv")
    args = parser.parse_args()
    main(args.results)
