"""
plot_pynguin_comparison.py — Paper figures for the Pynguin tool comparison.

Two figures:

  pynguin_vs_llm_kill_rate.png      Mean kill rate per generator (5 bars)
                                    with std error bars. LLM methods are
                                    aggregated across the 4 models so the
                                    comparison is apples-to-apples against
                                    Pynguin's single row.

  pynguin_vs_llm_per_operator.png   5×5 grouped bar chart: 5 operators on
                                    the x-axis, one bar per generator
                                    (Pynguin + 4 LLM methods), highlighting
                                    the comparison-mutator gap where
                                    Pynguin drops to 0.33.

Reads results_mutation.tsv. Save under plots_mutation/.
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

GENERATORS = [
    ("pynguin",            "Pynguin",            "#777777"),
    ("Plain LLM",          "Plain LLM",          "#4C72B0"),
    ("Random RAG",         "Random RAG",         "#8172B2"),
    ("Simple RAG",         "Simple RAG",         "#DD8452"),
    ("Iterative Critique", "Iterative Critique", "#55A868"),
]
OPERATORS = [
    ("kill_arithmetic",  "Arithmetic"),
    ("kill_boundary",    "Boundary"),
    ("kill_comparison",  "Comparison"),
    ("kill_negate_bool", "Negate bool"),
    ("kill_return_none", "Return None"),
]


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


def aggregate_per_generator(df: pd.DataFrame) -> pd.DataFrame:
    """One row per generator: mean kill rate + std + per-operator means."""
    rows = []
    for raw_method, _label, _color in GENERATORS:
        sub = df[df["method"] == raw_method]
        if sub.empty:
            continue
        row = {"generator": raw_method, "n_cells": len(sub)}
        # Overall mean kill rate across cells (each cell is a model row for
        # LLMs; for Pynguin there's just one cell, its own row)
        row["mean_kill_rate"] = sub["mean_kill_rate"].mean()
        # Std across cells. Pynguin only has one cell so std=NaN; use its
        # own std_kill_rate (within-sample) instead.
        if len(sub) == 1:
            row["std_kill_rate"] = sub["std_kill_rate"].iloc[0]
        else:
            row["std_kill_rate"] = sub["mean_kill_rate"].std(ddof=1)
        row["n_samples_valid"] = sub["n_samples_valid"].mean()
        for col, _opname in OPERATORS:
            row[col] = sub[col].mean()
        rows.append(row)
    return pd.DataFrame(rows)


# ---------------------------------------------------------------------------
# Plot 1 — overall kill rate comparison
# ---------------------------------------------------------------------------

def plot_overall(agg: pd.DataFrame, out_path: Path) -> None:
    fig, ax = plt.subplots(figsize=(8.5, 5.5))
    labels  = [g[1] for g in GENERATORS if (agg["generator"] == g[0]).any()]
    means   = [agg.loc[agg["generator"] == g[0], "mean_kill_rate"].iloc[0]
               for g in GENERATORS if (agg["generator"] == g[0]).any()]
    stds    = [agg.loc[agg["generator"] == g[0], "std_kill_rate"].iloc[0]
               for g in GENERATORS if (agg["generator"] == g[0]).any()]
    ns      = [agg.loc[agg["generator"] == g[0], "n_samples_valid"].iloc[0]
               for g in GENERATORS if (agg["generator"] == g[0]).any()]
    colors  = [g[2] for g in GENERATORS if (agg["generator"] == g[0]).any()]

    xs = np.arange(len(labels))
    bars = ax.bar(xs, means, yerr=stds, color=colors, capsize=6,
                  edgecolor="black", linewidth=0.6, error_kw={"linewidth": 1.2})

    for x, m, n in zip(xs, means, ns):
        ax.text(x, m + 0.02, f"{m:.3f}", ha="center", va="bottom",
                fontsize=10, fontweight="bold")
        ax.text(x, -0.04, f"n≈{int(round(n))}", ha="center", va="top",
                fontsize=8, color="#555")

    ax.set_xticks(xs)
    ax.set_xticklabels(labels, fontsize=11, rotation=15, ha="right")
    ax.set_ylabel("Mean mutation kill rate", fontsize=12)
    ax.set_ylim(0, 1.1)
    ax.axhline(0.5, color="red", linestyle="--", alpha=0.4,
               label="random-baseline reference (0.5)")
    ax.set_title("Mutation kill rate — Pynguin vs LLM methods\n"
                 "(LLM rows averaged across 4 models; Pynguin = single 40-sample row)",
                 fontsize=12, fontweight="bold")
    ax.legend(loc="lower left", fontsize=9)
    ax.grid(axis="y", alpha=0.3)
    plt.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close(fig)
    print(f"  {out_path.name}")


# ---------------------------------------------------------------------------
# Plot 2 — per-operator grouped bars
# ---------------------------------------------------------------------------

def plot_per_operator(agg: pd.DataFrame, out_path: Path) -> None:
    fig, ax = plt.subplots(figsize=(11, 5.5))
    op_keys   = [op[0] for op in OPERATORS]
    op_labels = [op[1] for op in OPERATORS]
    present_gens = [g for g in GENERATORS if (agg["generator"] == g[0]).any()]
    n_gens = len(present_gens)
    width = 0.8 / n_gens
    xs = np.arange(len(op_keys))

    for i, (gen_key, gen_label, color) in enumerate(present_gens):
        vals = []
        for op_key in op_keys:
            v = agg.loc[agg["generator"] == gen_key, op_key].iloc[0]
            vals.append(v if not (isinstance(v, float) and math.isnan(v)) else 0.0)
        offset = (i - n_gens / 2 + 0.5) * width
        bars = ax.bar(xs + offset, vals, width * 0.95,
                      label=gen_label, color=color,
                      edgecolor="black", linewidth=0.4)
        for bar, v in zip(bars, vals):
            if v > 0:
                ax.text(bar.get_x() + bar.get_width() / 2,
                        bar.get_height() + 0.01,
                        f"{v:.2f}", ha="center", va="bottom",
                        fontsize=7, rotation=0)

    ax.set_xticks(xs)
    ax.set_xticklabels(op_labels, fontsize=10)
    ax.set_ylabel("Per-operator kill rate", fontsize=11)
    ax.set_ylim(0, 1.18)
    ax.set_title("Per-operator mutation kill rate — Pynguin vs LLM methods\n"
                 "(Pynguin's comparison-operator gap is the largest, 0.33 vs IC 0.96)",
                 fontsize=12, fontweight="bold")
    ax.legend(loc="upper right", fontsize=9, ncol=2)
    ax.grid(axis="y", alpha=0.25)
    plt.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close(fig)
    print(f"  {out_path.name}")


def main() -> int:
    OUTPUT_DIR.mkdir(exist_ok=True)
    df = load_tsv()
    agg = aggregate_per_generator(df)
    if agg.empty:
        sys.exit("ERROR: no generators found")
    print("Aggregates:")
    print(agg[["generator", "n_cells", "mean_kill_rate",
               "std_kill_rate", "n_samples_valid"]].to_string(index=False))

    plot_overall(agg, OUTPUT_DIR / "pynguin_vs_llm_kill_rate.png")
    plot_per_operator(agg, OUTPUT_DIR / "pynguin_vs_llm_per_operator.png")
    return 0


if __name__ == "__main__":
    sys.exit(main())
