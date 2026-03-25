"""
visualize_unitest.py — generate PhD comparison charts from results_unitest.tsv

Run after experiments:
    python visualize_unitest.py
    uv run visualize_unitest.py       # if using uv

Outputs 4 charts to plots_unitest/:
  heatmap.png         — val_score grid: method × reasoning
  grouped_bar.png     — val_score grouped bar by reasoning
  radar.png           — per-metric radar (best run per method)
  per_metric_bar.png  — per-metric bar (best run per method)

Radar and per-metric charts require the extended TSV format with metric columns.
See program_unitest.md for the full TSV column spec.
"""

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

    print(f"\nDone. Open {OUTPUT_DIR}/ to view charts.")


if __name__ == "__main__":
    main()
