"""
visualize_unitest.py — generate PhD comparison charts from results_unitest.tsv

Run after experiments:
    python visualize_unitest.py
    uv run visualize_unitest.py       # if using uv

Outputs charts to plots_unitest/:
  Single-model charts (Charts 1-7):
    heatmap.png           — val_score grid: method × reasoning
    grouped_bar.png       — val_score grouped bar by reasoning
    radar.png             — per-metric radar (best run per method)
    per_metric_bar.png    — per-metric bar (best run per method)
    noise_rate.png        — avg noise rate per RAG method (RQ2)
    cost_breakdown.png    — stacked retrieval + LLM time (RQ4)
    faithfulness.png      — token-overlap faithfulness per method (RQ3)

  Cross-model charts (Charts 8-10, requires ≥2 models in TSV):
    model_val_score.png   — val_score grouped by method × model
    model_faithfulness.png — faithfulness grouped by method × model
    model_rank_stability.png — method ranking lines across models

See program_unitest.md for the full TSV column spec.
"""

import math
import sys
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path

# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------

RESULTS_FILE = "results_unitest.tsv"
OUTPUT_DIR   = Path("plots_unitest")

METHODS    = ["plain_llm", "simple_rag", "iterative_critique"]
REASONINGS = ["base", "cot", "tot", "got"]

METHOD_LABELS = {
    "plain_llm":          "Plain LLM",
    "simple_rag":         "Simple RAG",
    "iterative_critique": "Iterative Critique",
}
REASONING_LABELS = {"base": "Base", "cot": "CoT", "tot": "ToT", "got": "GoT"}

COLORS = {
    "plain_llm":          "#4C72B0",
    "simple_rag":         "#DD8452",
    "iterative_critique": "#55A868",
}

# Columns logged in extended TSV (see program_unitest.md)
METRIC_COLS   = ["avg_syntax", "avg_edge", "avg_assert_density", "avg_semantic_sim", "avg_rouge"]
METRIC_LABELS = ["Syntax\nValidity", "Edge\nCoverage", "Assert\nDensity", "Semantic\nSim", "ROUGE-1"]


# ---------------------------------------------------------------------------
# Data loading
# ---------------------------------------------------------------------------

def load_results() -> pd.DataFrame:
    df = pd.read_csv(RESULTS_FILE, sep="\t")

    if df.empty:
        return df  # caller handles empty case

    # Parse "plain_llm/base" style method column into two columns
    if "/" in str(df["method"].iloc[0]):
        df[["method_name", "reasoning"]] = df["method"].str.split("/", n=1, expand=True)
    else:
        df["method_name"] = df["method"]
        df["reasoning"]   = "base"

    df = df[df["status"] != "crash"].copy()
    df["val_score"] = pd.to_numeric(df["val_score"], errors="coerce").fillna(0.0)
    return df


def _best_per_method(df: pd.DataFrame) -> dict:
    """Return {method_name: row_series} for the highest val_score run per method."""
    best = {}
    for method in METHODS:
        sub = df[df["method_name"] == method]
        if len(sub) > 0:
            best[method] = sub.loc[sub["val_score"].idxmax()]
    return best


# ---------------------------------------------------------------------------
# Chart 1: Heatmap — method × reasoning
# ---------------------------------------------------------------------------

def plot_heatmap(df: pd.DataFrame) -> None:
    pivot = df.pivot_table(
        index="method_name", columns="reasoning",
        values="val_score", aggfunc="max",
    ).reindex(index=METHODS, columns=REASONINGS)

    fig, ax = plt.subplots(figsize=(8, 4))
    im = ax.imshow(pivot.values, cmap="YlGn", aspect="auto", vmin=0, vmax=1)

    ax.set_xticks(range(len(REASONINGS)))
    ax.set_xticklabels([REASONING_LABELS.get(r, r) for r in REASONINGS], fontsize=11)
    ax.set_yticks(range(len(METHODS)))
    ax.set_yticklabels([METHOD_LABELS.get(m, m) for m in METHODS], fontsize=11)

    for i in range(len(METHODS)):
        for j in range(len(REASONINGS)):
            val = pivot.values[i, j]
            if not np.isnan(val):
                color = "white" if val > 0.6 else "black"
                ax.text(j, i, f"{val:.3f}", ha="center", va="center",
                        fontsize=11, fontweight="bold", color=color)

    plt.colorbar(im, ax=ax, label="val_score")
    ax.set_title("val_score Heatmap: Method × Reasoning", fontsize=13, fontweight="bold", pad=10)
    plt.tight_layout()
    fig.savefig(OUTPUT_DIR / "heatmap.png", dpi=150)
    plt.close(fig)
    print("  heatmap.png")


# ---------------------------------------------------------------------------
# Chart 2: Grouped bar — val_score per reasoning, grouped by method
# ---------------------------------------------------------------------------

def plot_grouped_bar(df: pd.DataFrame) -> None:
    x     = np.arange(len(REASONINGS))
    width = 0.25

    fig, ax = plt.subplots(figsize=(10, 5))

    for i, method in enumerate(METHODS):
        vals = []
        for r in REASONINGS:
            row = df[(df["method_name"] == method) & (df["reasoning"] == r)]
            vals.append(float(row["val_score"].max()) if len(row) > 0 else 0.0)

        bars = ax.bar(
            x + i * width, vals, width,
            label=METHOD_LABELS[method], color=COLORS[method], alpha=0.85,
        )
        for bar, v in zip(bars, vals):
            if v > 0:
                ax.text(
                    bar.get_x() + bar.get_width() / 2,
                    bar.get_height() + 0.005,
                    f"{v:.3f}", ha="center", va="bottom", fontsize=7.5,
                )

    ax.set_xticks(x + width * 1.5)  # center under all 3 bars
    ax.set_xticklabels([REASONING_LABELS.get(r, r) for r in REASONINGS], fontsize=11)
    ax.set_xlabel("Reasoning Technique", fontsize=11)
    ax.set_ylabel("val_score", fontsize=11)
    ax.set_ylim(0, 1.08)
    ax.set_title("val_score by Method and Reasoning", fontsize=13, fontweight="bold")
    ax.legend(fontsize=10)
    ax.grid(axis="y", alpha=0.3)
    plt.tight_layout()
    fig.savefig(OUTPUT_DIR / "grouped_bar.png", dpi=150)
    plt.close(fig)
    print("  grouped_bar.png")


# ---------------------------------------------------------------------------
# Chart 3: Radar — per-metric breakdown (best run per method)
# ---------------------------------------------------------------------------

def plot_radar(df: pd.DataFrame) -> None:
    if not all(c in df.columns for c in METRIC_COLS):
        print("  radar.png SKIPPED — per-metric columns missing from TSV (see program_unitest.md)")
        return

    best = _best_per_method(df)
    if not best:
        return

    N      = len(METRIC_COLS)
    angles = np.linspace(0, 2 * np.pi, N, endpoint=False).tolist()
    angles += angles[:1]  # close the polygon

    fig, ax = plt.subplots(figsize=(6, 6), subplot_kw=dict(polar=True))

    for method, row in best.items():
        vals = [float(row.get(m, 0.0)) for m in METRIC_COLS]
        vals += vals[:1]
        ax.plot(angles, vals, "o-", linewidth=2,
                label=METHOD_LABELS[method], color=COLORS[method])
        ax.fill(angles, vals, alpha=0.12, color=COLORS[method])

    ax.set_xticks(angles[:-1])
    ax.set_xticklabels(METRIC_LABELS, fontsize=10)
    ax.set_ylim(0, 1)
    ax.set_yticks([0.2, 0.4, 0.6, 0.8, 1.0])
    ax.set_yticklabels(["0.2", "0.4", "0.6", "0.8", "1.0"], fontsize=7, color="grey")
    ax.set_title("Per-Metric Breakdown\n(Best Run per Method)",
                 fontsize=12, fontweight="bold", pad=20)
    ax.legend(loc="upper right", bbox_to_anchor=(1.35, 1.15), fontsize=9)
    plt.tight_layout()
    fig.savefig(OUTPUT_DIR / "radar.png", dpi=150)
    plt.close(fig)
    print("  radar.png")


# ---------------------------------------------------------------------------
# Chart 4: Per-metric grouped bar (best run per method)
# ---------------------------------------------------------------------------

def plot_per_metric_bar(df: pd.DataFrame) -> None:
    if not all(c in df.columns for c in METRIC_COLS):
        print("  per_metric_bar.png SKIPPED — per-metric columns missing from TSV (see program_unitest.md)")
        return

    best = _best_per_method(df)
    if not best:
        return

    x     = np.arange(len(METRIC_COLS))
    width = 0.25

    fig, ax = plt.subplots(figsize=(11, 5))

    for i, (method, row) in enumerate(best.items()):
        vals = [float(row.get(m, 0.0)) for m in METRIC_COLS]
        bars = ax.bar(
            x + i * width, vals, width,
            label=METHOD_LABELS[method], color=COLORS[method], alpha=0.85,
        )
        for bar, v in zip(bars, vals):
            if v > 0:
                ax.text(
                    bar.get_x() + bar.get_width() / 2,
                    bar.get_height() + 0.005,
                    f"{v:.2f}", ha="center", va="bottom", fontsize=8,
                )

    ax.set_xticks(x + width * 1.5)  # center under all 3 bars
    ax.set_xticklabels(METRIC_LABELS, fontsize=11)
    ax.set_ylabel("Score", fontsize=11)
    ax.set_ylim(0, 1.12)
    ax.set_title("Per-Metric Comparison (Best Run per Method)",
                 fontsize=13, fontweight="bold")
    ax.legend(fontsize=10)
    ax.grid(axis="y", alpha=0.3)
    plt.tight_layout()
    fig.savefig(OUTPUT_DIR / "per_metric_bar.png", dpi=150)
    plt.close(fig)
    print("  per_metric_bar.png")


# ---------------------------------------------------------------------------
# Chart 5: Noise Rate bar — avg_noise_rate per method (RAG methods only)
# ---------------------------------------------------------------------------

def plot_noise_rate(df: pd.DataFrame) -> None:
    if "avg_noise_rate" not in df.columns:
        print("  noise_rate.png SKIPPED — avg_noise_rate column missing from TSV")
        return

    fig, ax = plt.subplots(figsize=(7, 4))
    rag_methods = ["simple_rag", "iterative_critique"]
    vals, labels, colors = [], [], []

    for method in rag_methods:
        sub = df[df["method_name"] == method]
        if len(sub) == 0:
            continue
        # Use best run per method
        best = sub.loc[sub["val_score"].idxmax()]
        nr = pd.to_numeric(best.get("avg_noise_rate", float("nan")), errors="coerce")
        if not np.isnan(nr):
            vals.append(float(nr))
            labels.append(METHOD_LABELS[method])
            colors.append(COLORS[method])

    if not vals:
        print("  noise_rate.png SKIPPED — no valid noise_rate data")
        return

    bars = ax.bar(labels, vals, color=colors, alpha=0.85, width=0.4)
    ax.axhline(0.3, color="red", linestyle="--", linewidth=1.2, label="Threshold (0.3)")
    for bar, v in zip(bars, vals):
        ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.005,
                f"{v:.3f}", ha="center", va="bottom", fontsize=11, fontweight="bold")

    ax.set_ylabel("Noise Rate (fraction irrelevant chunks)", fontsize=11)
    ax.set_ylim(0, 1.05)
    ax.set_title("Avg Noise Rate per RAG Method\n(Best Run)", fontsize=13, fontweight="bold")
    ax.legend(fontsize=10)
    ax.grid(axis="y", alpha=0.3)
    plt.tight_layout()
    fig.savefig(OUTPUT_DIR / "noise_rate.png", dpi=150)
    plt.close(fig)
    print("  noise_rate.png")


# ---------------------------------------------------------------------------
# Chart 6: Cost breakdown — stacked bar (retrieval_secs + llm_secs) per method
# ---------------------------------------------------------------------------

def plot_cost_breakdown(df: pd.DataFrame) -> None:
    has_cols = "avg_retrieval_secs" in df.columns and "avg_llm_secs" in df.columns
    if not has_cols:
        print("  cost_breakdown.png SKIPPED — cost columns missing from TSV")
        return

    best = _best_per_method(df)
    if not best:
        return

    methods_present = list(best.keys())
    labels   = [METHOD_LABELS[m] for m in methods_present]
    ret_vals = [float(pd.to_numeric(best[m].get("avg_retrieval_secs", 0), errors="coerce") or 0)
                for m in methods_present]
    llm_vals = [float(pd.to_numeric(best[m].get("avg_llm_secs", 0), errors="coerce") or 0)
                for m in methods_present]

    x = np.arange(len(methods_present))
    fig, ax = plt.subplots(figsize=(8, 5))

    b1 = ax.bar(x, ret_vals, label="Retrieval", color="#5B9BD5", alpha=0.9)
    b2 = ax.bar(x, llm_vals, bottom=ret_vals, label="LLM Inference", color="#ED7D31", alpha=0.9)

    for i, (rv, lv) in enumerate(zip(ret_vals, llm_vals)):
        total = rv + lv
        if total > 0:
            ax.text(i, total + 0.02, f"{total:.2f}s", ha="center", va="bottom",
                    fontsize=10, fontweight="bold")

    ax.set_xticks(x)
    ax.set_xticklabels(labels, fontsize=11)
    ax.set_ylabel("Avg seconds per sample", fontsize=11)
    ax.set_title("Cost Breakdown: Retrieval vs LLM Time\n(Best Run per Method)",
                 fontsize=13, fontweight="bold")
    ax.legend(fontsize=10)
    ax.grid(axis="y", alpha=0.3)
    plt.tight_layout()
    fig.savefig(OUTPUT_DIR / "cost_breakdown.png", dpi=150)
    plt.close(fig)
    print("  cost_breakdown.png")


# ---------------------------------------------------------------------------
# Chart 7: Faithfulness bar — avg_faithfulness per method (RAG only)
# ---------------------------------------------------------------------------

def plot_faithfulness(df: pd.DataFrame) -> None:
    if "avg_faithfulness" not in df.columns:
        print("  faithfulness.png SKIPPED — avg_faithfulness column missing from TSV")
        return

    best = _best_per_method(df)
    if not best:
        return

    labels, vals, colors = [], [], []
    for method in METHODS:
        if method not in best:
            continue
        fv = pd.to_numeric(best[method].get("avg_faithfulness", float("nan")), errors="coerce")
        if np.isnan(fv):
            continue  # plain_llm has no context → NaN, skip
        labels.append(METHOD_LABELS[method])
        vals.append(float(fv))
        colors.append(COLORS[method])

    if not vals:
        print("  faithfulness.png SKIPPED — no valid faithfulness data")
        return

    fig, ax = plt.subplots(figsize=(7, 4))
    bars = ax.bar(labels, vals, color=colors, alpha=0.85, width=0.4)
    for bar, v in zip(bars, vals):
        ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.005,
                f"{v:.3f}", ha="center", va="bottom", fontsize=11, fontweight="bold")

    ax.set_ylabel("Faithfulness (token overlap with context)", fontsize=11)
    ax.set_ylim(0, 1.05)
    ax.set_title("Avg Faithfulness per Method\n(Best Run — NaN for Plain LLM)",
                 fontsize=13, fontweight="bold")
    ax.grid(axis="y", alpha=0.3)
    plt.tight_layout()
    fig.savefig(OUTPUT_DIR / "faithfulness.png", dpi=150)
    plt.close(fig)
    print("  faithfulness.png")


# ---------------------------------------------------------------------------
# Chart 8: val_score grouped by method × model (cross-model)
# ---------------------------------------------------------------------------

def plot_model_val_score(df: pd.DataFrame) -> None:
    models = sorted(df["model"].unique()) if "model" in df.columns else []
    if len(models) < 2:
        print("  model_val_score.png SKIPPED — need ≥2 models in TSV")
        return

    x     = np.arange(len(METHODS))
    width = 0.8 / len(models)

    fig, ax = plt.subplots(figsize=(10, 5))
    for i, model in enumerate(models):
        vals = []
        for method in METHODS:
            sub = df[(df["model"] == model) & (df["method_name"] == method)]
            vals.append(float(sub["val_score"].max()) if not sub.empty else 0.0)
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
    fig.savefig(OUTPUT_DIR / "model_val_score.png", dpi=150)
    plt.close(fig)
    print("  model_val_score.png")


# ---------------------------------------------------------------------------
# Chart 9: faithfulness grouped by method × model (cross-model)
# ---------------------------------------------------------------------------

def plot_model_faithfulness(df: pd.DataFrame) -> None:
    if "avg_faithfulness" not in df.columns:
        print("  model_faithfulness.png SKIPPED — avg_faithfulness missing")
        return
    models = sorted(df["model"].unique()) if "model" in df.columns else []
    if len(models) < 2:
        print("  model_faithfulness.png SKIPPED — need ≥2 models in TSV")
        return

    rag_methods = ["simple_rag", "iterative_critique"]
    x     = np.arange(len(rag_methods))
    width = 0.8 / len(models)

    fig, ax = plt.subplots(figsize=(8, 5))
    has_data = False
    for i, model in enumerate(models):
        vals = []
        for method in rag_methods:
            sub = df[(df["model"] == model) & (df["method_name"] == method)]
            if sub.empty:
                vals.append(float("nan"))
                continue
            fv = pd.to_numeric(sub["avg_faithfulness"], errors="coerce").max()
            vals.append(float(fv))
        offset = (i - len(models) / 2 + 0.5) * width
        for j, v in enumerate(vals):
            if not math.isnan(v):
                has_data = True
                ax.bar(x[j] + offset, v, width * 0.9,
                       label=model if j == 0 else "", alpha=0.85)
                ax.text(x[j] + offset, v + 0.005, f"{v:.3f}",
                        ha="center", va="bottom", fontsize=7)

    if not has_data:
        print("  model_faithfulness.png SKIPPED — no faithfulness data")
        plt.close(fig)
        return

    ax.set_xticks(x)
    ax.set_xticklabels([METHOD_LABELS[m] for m in rag_methods], fontsize=11)
    ax.set_ylabel("Faithfulness (token overlap)", fontsize=11)
    ax.set_ylim(0, 1.1)
    ax.set_title("Faithfulness by Method × Model\n(RAG methods only)",
                 fontsize=13, fontweight="bold")
    ax.legend(title="Model", fontsize=9, title_fontsize=9)
    ax.grid(axis="y", alpha=0.3)
    plt.tight_layout()
    fig.savefig(OUTPUT_DIR / "model_faithfulness.png", dpi=150)
    plt.close(fig)
    print("  model_faithfulness.png")


# ---------------------------------------------------------------------------
# Chart 10: Rank stability — method rankings across models (line plot)
# ---------------------------------------------------------------------------

def plot_model_rank_stability(df: pd.DataFrame) -> None:
    models = sorted(df["model"].unique()) if "model" in df.columns else []
    if len(models) < 2:
        print("  model_rank_stability.png SKIPPED — need ≥2 models in TSV")
        return

    fig, ax = plt.subplots(figsize=(9, 5))
    for method in METHODS:
        ranks, x_labels = [], []
        for model in models:
            sub = df[df["model"] == model]
            if sub.empty:
                continue
            scores = {}
            for m in METHODS:
                r = sub[sub["method_name"] == m]
                scores[m] = float(r["val_score"].max()) if not r.empty else 0.0
            sorted_methods = sorted(scores, key=scores.get, reverse=True)
            rank = sorted_methods.index(method) + 1 if method in sorted_methods else 3
            ranks.append(rank)
            x_labels.append(model)

        ax.plot(x_labels, ranks, "o-", linewidth=2.5, markersize=9,
                label=METHOD_LABELS[method], color=COLORS[method])
        for xi, rank in enumerate(ranks):
            ax.text(xi, rank - 0.1, str(rank), ha="center", va="bottom",
                    fontsize=9, fontweight="bold", color=COLORS[method])

    ax.set_yticks([1, 2, 3])
    ax.set_yticklabels(["1st (best)", "2nd", "3rd (worst)"], fontsize=10)
    ax.invert_yaxis()
    ax.set_xlabel("Model", fontsize=11)
    ax.set_ylabel("Rank (by val_score)", fontsize=11)
    ax.set_title("Method Ranking Stability Across Models\n(Flat lines = findings generalize)",
                 fontsize=13, fontweight="bold")
    ax.legend(fontsize=10)
    ax.grid(axis="y", alpha=0.3)
    plt.tight_layout()
    fig.savefig(OUTPUT_DIR / "model_rank_stability.png", dpi=150)
    plt.close(fig)
    print("  model_rank_stability.png")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    if not Path(RESULTS_FILE).exists():
        print(f"ERROR: {RESULTS_FILE} not found. Run experiments first.")
        sys.exit(1)

    df = load_results()
    if len(df) == 0:
        print("No non-crash results to visualize yet.")
        sys.exit(0)

    OUTPUT_DIR.mkdir(exist_ok=True)
    print(f"Loaded {len(df)} results → generating charts in {OUTPUT_DIR}/")

    plot_heatmap(df)
    plot_grouped_bar(df)
    plot_radar(df)
    plot_per_metric_bar(df)
    plot_noise_rate(df)
    plot_cost_breakdown(df)
    plot_faithfulness(df)

    # Cross-model charts (auto-skipped if only 1 model in TSV)
    print("\n  --- Cross-model charts ---")
    plot_model_val_score(df)
    plot_model_faithfulness(df)
    plot_model_rank_stability(df)

    print(f"\nDone. Open {OUTPUT_DIR}/ to view charts.")


if __name__ == "__main__":
    main()
