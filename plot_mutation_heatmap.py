"""
plot_mutation_heatmap.py — 4×4 mutation kill-rate heatmaps for the paper.

Replaces the original 16-bar chart (`kill_rate_by_method.png`) with
proper method × model heatmaps. Three flavours saved:

  kill_rate_heatmap.png            — overall mean_kill_rate
  kill_rate_boundary_heatmap.png   — boundary (significant in mixed-LM)
  kill_rate_combined_heatmap.png   — 1×5 grid: overall + per-operator

Reads results_mutation.tsv (the merged 4×4 matrix) and looks up colour
values + sample counts to annotate each cell.
"""

from __future__ import annotations

import math
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

OUTPUT_DIR = Path("plots_mutation")
RESULTS_CANDIDATES = [Path("results_mutation.tsv"),
                      Path("results/results_mutation.tsv")]

METHODS = ["Plain LLM", "Random RAG", "Simple RAG", "Iterative Critique"]
MODELS  = ["llama3.2:latest", "phi4:14b", "qwen3.5:9b", "qwen3-coder:30b"]
MODEL_SHORT = {
    "llama3.2:latest":  "llama3.2\n(3B)",
    "phi4:14b":         "phi4\n(14B)",
    "qwen3.5:9b":       "qwen3.5\n(9B)",
    "qwen3-coder:30b":  "qwen3-coder\n(30B MoE)",
}


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
        sys.exit("ERROR: no results_mutation.tsv with model column found")
    return best


def make_grid(df: pd.DataFrame, value_col: str) -> tuple[np.ndarray, np.ndarray]:
    """Return (values_matrix, n_matrix) keyed by METHODS × MODELS."""
    vals = np.full((len(METHODS), len(MODELS)), float("nan"))
    ns   = np.full((len(METHODS), len(MODELS)), 0, dtype=int)
    for i, method in enumerate(METHODS):
        for j, model in enumerate(MODELS):
            row = df[(df["method"] == method) & (df["model"] == model)]
            if row.empty:
                continue
            v = row[value_col].iloc[0]
            if not (isinstance(v, float) and math.isnan(v)):
                vals[i, j] = float(v)
            if "n_samples_valid" in row.columns:
                ns[i, j] = int(row["n_samples_valid"].iloc[0])
    return vals, ns


def plot_single(values: np.ndarray, ns: np.ndarray, title: str,
                out_path: Path, cbar_label: str = "Kill rate") -> None:
    fig, ax = plt.subplots(figsize=(8.5, 5.5))
    im = ax.imshow(values, cmap="RdYlGn", vmin=0.0, vmax=1.0, aspect="auto")
    cbar = plt.colorbar(im, ax=ax, label=cbar_label)
    cbar.set_label(cbar_label, fontsize=11)

    ax.set_xticks(range(len(MODELS)))
    ax.set_yticks(range(len(METHODS)))
    ax.set_xticklabels([MODEL_SHORT[m] for m in MODELS], fontsize=10)
    ax.set_yticklabels(METHODS, fontsize=11)
    ax.set_xlabel("Model", fontsize=12)
    ax.set_ylabel("Method", fontsize=12)
    ax.set_title(title, fontsize=13, fontweight="bold", pad=12)

    for i in range(values.shape[0]):
        for j in range(values.shape[1]):
            v = values[i, j]
            if math.isnan(v):
                text = "—"
                color = "#666"
            else:
                text = f"{v:.3f}"
                color = "white" if v < 0.5 else "black"
            label = text
            if ns[i, j] > 0:
                label += f"\nn={ns[i, j]}"
            ax.text(j, i, label, ha="center", va="center",
                    fontsize=10, fontweight="bold", color=color)

    plt.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close(fig)
    print(f"  {out_path.name}")


def plot_combined(df: pd.DataFrame, out_path: Path) -> None:
    """1×5 panel: overall + 4 per-operator heatmaps. (Skip kill_return_none
    just so we fit 5 panels neatly: overall, arithmetic, boundary,
    comparison, negate_bool.)
    """
    panels = [
        ("mean_kill_rate",   "Overall"),
        ("kill_arithmetic",  "Arithmetic"),
        ("kill_boundary",    "Boundary"),
        ("kill_comparison",  "Comparison"),
        ("kill_negate_bool", "Negate Bool"),
    ]
    panels = [(c, l) for c, l in panels if c in df.columns]
    if not panels:
        return

    fig, axes = plt.subplots(1, len(panels),
                             figsize=(3.6 * len(panels), 5.5),
                             sharey=True)
    if len(panels) == 1:
        axes = [axes]

    for ax, (col, label) in zip(axes, panels):
        vals, ns = make_grid(df, col)
        im = ax.imshow(vals, cmap="RdYlGn", vmin=0.0, vmax=1.0, aspect="auto")
        ax.set_xticks(range(len(MODELS)))
        ax.set_yticks(range(len(METHODS)))
        ax.set_xticklabels([MODEL_SHORT[m] for m in MODELS], fontsize=8)
        ax.set_yticklabels(METHODS, fontsize=9)
        ax.set_title(label, fontsize=11, fontweight="bold")
        for i in range(vals.shape[0]):
            for j in range(vals.shape[1]):
                v = vals[i, j]
                if math.isnan(v):
                    text = "—"
                    color = "#555"
                else:
                    text = f"{v:.2f}"
                    color = "white" if v < 0.5 else "black"
                ax.text(j, i, text, ha="center", va="center",
                        fontsize=9, fontweight="bold", color=color)

    fig.colorbar(im, ax=axes, location="bottom", pad=0.18, shrink=0.6,
                 label="Kill rate")
    fig.suptitle("Mutation Kill Rate by Method × Model (overall + per operator)",
                 fontsize=13, fontweight="bold")
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  {out_path.name}")


def main() -> int:
    OUTPUT_DIR.mkdir(exist_ok=True)
    df = load_tsv()
    if df["model"].nunique() < 2:
        sys.exit("ERROR: need ≥2 models")

    print(f"Loaded {len(df)} rows; models = {sorted(df['model'].unique())}")

    # Overall kill rate
    vals, ns = make_grid(df, "mean_kill_rate")
    plot_single(vals, ns,
                "Mutation Kill Rate by Method × Model",
                OUTPUT_DIR / "kill_rate_heatmap.png",
                "Mean kill rate")

    # Boundary (where significance lives)
    if "kill_boundary" in df.columns:
        vals_b, _ = make_grid(df, "kill_boundary")
        # ns is meaningless for per-operator (subset of samples)
        plot_single(vals_b, np.zeros_like(ns),
                    "Boundary Mutation Kill Rate (off-by-one defects)",
                    OUTPUT_DIR / "kill_rate_boundary_heatmap.png",
                    "Boundary kill rate")

    # Combined per-operator
    plot_combined(df, OUTPUT_DIR / "kill_rate_combined_heatmap.png")

    return 0


if __name__ == "__main__":
    sys.exit(main())
