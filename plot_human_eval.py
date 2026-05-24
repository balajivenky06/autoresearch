"""
plot_human_eval.py — Paper figures for the human-evaluation study.

Three figures:

  human_eval_method_ranking.png       Grouped bar chart: 4 methods × 3
                                      dimensions, mean rating across the
                                      3 annotators. Shows IC ranks
                                      highest on every dimension.

  human_eval_annotator_bias.png       Per-annotator mean rating per
                                      dimension — visualises the
                                      SaeshwaranA-vs-others scale-usage
                                      bias documented in §Limitations.

  human_eval_correlations.png         Scatter of mean human ratings vs
                                      mutation kill rate (one point per
                                      sample, coloured by method).

Reads human_eval_annotations/{GS,SA,BV}.csv plus the meta CSV.
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
from scipy import stats

OUTPUT_DIR = Path("plots_mutation")

ANNOTATORS = [
    ("human_eval_annotations/GS.csv", "GS"),
    ("human_eval_annotations/SA.csv", "SA"),
    ("human_eval_annotations/BV.csv", "BV"),
]
META_PATH = "human_eval_pairs.meta.csv"
DIMS = [
    ("human_test_idiom",    "Test idiom"),
    ("human_correctness",   "Correctness"),
    ("human_completeness",  "Completeness"),
]
METHODS = [
    ("plain_llm",          "Plain LLM",          "#4C72B0"),
    ("random_rag",         "Random RAG",         "#8172B2"),
    ("simple_rag",         "Simple RAG",         "#DD8452"),
    ("iterative_critique", "Iterative Critique", "#55A868"),
]
ANNOTATOR_COLORS = {"GS": "#1f77b4", "SA": "#ff7f0e", "BV": "#2ca02c"}


def load_long() -> pd.DataFrame:
    """Long-format DataFrame: one row per (annotator, sample_id, dim)."""
    parts = []
    for path, _short in ANNOTATORS:
        p = Path(path)
        if not p.exists():
            sys.exit(f"ERROR: {path} not found")
        df = pd.read_csv(p)
        parts.append(df)
    full = pd.concat(parts, ignore_index=True)

    meta = pd.read_csv(META_PATH)
    full = full.merge(meta[["sample_id", "method", "model", "source",
                            "kill_rate"]], on="sample_id", how="left")
    return full


# ---------------------------------------------------------------------------
# Plot 1 — method × dimension grouped bars
# ---------------------------------------------------------------------------

def plot_method_ranking(full: pd.DataFrame, out_path: Path) -> None:
    fig, ax = plt.subplots(figsize=(10, 5.5))
    method_keys   = [m[0] for m in METHODS]
    method_labels = [m[1] for m in METHODS]
    method_colors = [m[2] for m in METHODS]
    dim_keys      = [d[0] for d in DIMS]
    dim_labels    = [d[1] for d in DIMS]

    xs = np.arange(len(dim_keys))
    width = 0.8 / len(method_keys)

    for i, (m_key, m_label, m_color) in enumerate(METHODS):
        means = []
        stds  = []
        for d_key, _d_label in DIMS:
            sub = full[full["method"] == m_key][d_key].dropna()
            means.append(sub.mean() if len(sub) > 0 else 0.0)
            stds.append(sub.std(ddof=1) if len(sub) > 1 else 0.0)
        offset = (i - len(METHODS) / 2 + 0.5) * width
        bars = ax.bar(xs + offset, means, width * 0.95,
                      yerr=stds, capsize=4,
                      color=m_color, label=m_label,
                      edgecolor="black", linewidth=0.4,
                      error_kw={"linewidth": 0.8, "alpha": 0.5})
        for bar, v in zip(bars, means):
            ax.text(bar.get_x() + bar.get_width() / 2,
                    bar.get_height() + 0.06,
                    f"{v:.2f}", ha="center", va="bottom",
                    fontsize=8, fontweight="bold")

    ax.set_xticks(xs)
    ax.set_xticklabels(dim_labels, fontsize=11)
    ax.set_ylabel("Mean human rating (0–5)", fontsize=11)
    ax.set_ylim(0, 6.0)
    ax.set_title("Human-evaluation method ranking (3 annotators × 40 samples)\n"
                 "Iterative Critique ranks highest on all three dimensions",
                 fontsize=12, fontweight="bold")
    ax.legend(loc="upper left", fontsize=9, ncol=2)
    ax.grid(axis="y", alpha=0.25)
    plt.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close(fig)
    print(f"  {out_path.name}")


# ---------------------------------------------------------------------------
# Plot 2 — per-annotator scale-usage bias
# ---------------------------------------------------------------------------

def plot_annotator_bias(full: pd.DataFrame, out_path: Path) -> None:
    fig, ax = plt.subplots(figsize=(8.5, 5))
    annotators = sorted(full["annotator_id"].unique())
    dim_keys    = [d[0] for d in DIMS]
    dim_labels  = [d[1] for d in DIMS]
    xs = np.arange(len(dim_keys))
    width = 0.8 / len(annotators)
    for i, ann in enumerate(annotators):
        means = [full[full["annotator_id"] == ann][k].mean() for k in dim_keys]
        offset = (i - len(annotators) / 2 + 0.5) * width
        # Normalise SA name for label
        label = "SaeshwaranA" if ann.startswith("Saeshu") else ann
        color = ANNOTATOR_COLORS.get(ann[:2], "#888")
        bars = ax.bar(xs + offset, means, width * 0.9, label=label,
                      color=color, edgecolor="black", linewidth=0.4)
        for bar, v in zip(bars, means):
            ax.text(bar.get_x() + bar.get_width() / 2,
                    bar.get_height() + 0.05,
                    f"{v:.2f}", ha="center", va="bottom", fontsize=9)
    ax.set_xticks(xs)
    ax.set_xticklabels(dim_labels, fontsize=11)
    ax.set_ylabel("Mean rating across all 40 samples (0–5)", fontsize=11)
    ax.set_ylim(0, 5.5)
    ax.set_title("Per-annotator scale-usage — documented in §Limitations\n"
                 "SaeshwaranA uses the lower half of the scale; "
                 "GS and BV the upper half",
                 fontsize=11, fontweight="bold")
    ax.legend(title="Annotator", fontsize=9, title_fontsize=9)
    ax.grid(axis="y", alpha=0.25)
    plt.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close(fig)
    print(f"  {out_path.name}")


# ---------------------------------------------------------------------------
# Plot 3 — mean human rating vs mutation kill rate
# ---------------------------------------------------------------------------

def plot_correlations(full: pd.DataFrame, out_path: Path) -> None:
    # Per-sample mean across annotators
    mean_per_sample = (full.groupby("sample_id")[[d[0] for d in DIMS]
                                                  + ["kill_rate"]]
                       .first().reset_index())  # placeholder
    # Recompute mean of the dim columns (kill_rate is the same per sample)
    means = full.groupby("sample_id")[[d[0] for d in DIMS]].mean()
    kr = full.groupby("sample_id")["kill_rate"].first()
    method = (full[["sample_id", "method"]]
              .drop_duplicates(subset=["sample_id"])
              .set_index("sample_id")["method"])
    df = means.join(kr).join(method).reset_index()

    fig, axes = plt.subplots(1, 3, figsize=(15, 4.8), sharey=True)
    method_color = {m[0]: m[2] for m in METHODS}
    for ax, (d_key, d_label) in zip(axes, DIMS):
        sub = df.dropna(subset=[d_key, "kill_rate"])
        for m_key, m_label, _color in METHODS:
            cell = sub[sub["method"] == m_key]
            if cell.empty:
                continue
            ax.scatter(cell[d_key], cell["kill_rate"],
                       c=method_color[m_key], s=70, alpha=0.85,
                       edgecolor="black", linewidth=0.4,
                       label=m_label)
        if len(sub) >= 3:
            r, p = stats.pearsonr(sub[d_key], sub["kill_rate"])
            ax.set_title(f"{d_label}  (r={r:+.3f}, p={p:.3f}, n={len(sub)})",
                         fontsize=11, fontweight="bold")
        else:
            ax.set_title(f"{d_label}", fontsize=11, fontweight="bold")
        ax.set_xlabel("Mean human rating (0–5)", fontsize=10)
        ax.set_xlim(-0.2, 5.4)
        ax.grid(alpha=0.25)

    axes[0].set_ylabel("Mutation kill rate", fontsize=10)
    axes[-1].legend(loc="lower right", fontsize=8)
    fig.suptitle("Per-sample mean human rating vs mutation kill rate "
                 "(n=40 samples, 3 annotators averaged)",
                 fontsize=12, fontweight="bold")
    plt.tight_layout(rect=(0, 0, 1, 0.94))
    fig.savefig(out_path, dpi=150)
    plt.close(fig)
    print(f"  {out_path.name}")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> int:
    OUTPUT_DIR.mkdir(exist_ok=True)
    full = load_long()
    print(f"Loaded {len(full)} rating rows across "
          f"{full['annotator_id'].nunique()} annotators × "
          f"{full['sample_id'].nunique()} samples")

    plot_method_ranking(full, OUTPUT_DIR / "human_eval_method_ranking.png")
    plot_annotator_bias(full, OUTPUT_DIR / "human_eval_annotator_bias.png")
    plot_correlations(full, OUTPUT_DIR / "human_eval_correlations.png")
    return 0


if __name__ == "__main__":
    sys.exit(main())
