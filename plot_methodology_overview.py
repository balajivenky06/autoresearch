"""
plot_methodology_overview.py — Architecture / methodology pipeline figure
for the paper's §3 Methods opener.

Draws the end-to-end experimental design as a single PNG:
  Input dataset → 4 × 4 generation matrix → 3 parallel evaluation tracks
  (mutation testing, human evaluation, Pynguin baseline) → statistical
  analyses.

Output:
    plots_mutation/methodology_overview.png
"""

from __future__ import annotations

from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.patches as mpatches
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch

OUTPUT = Path("plots_mutation") / "methodology_overview.png"

# Colour scheme (consistent with the rest of the paper's figures)
COLORS = {
    "input":      "#E8EEF7",   # cool grey-blue
    "generation": "#FDE9DA",   # warm peach
    "method":     "#FFE9B0",   # light yellow
    "model":      "#D8E8D8",   # light green
    "mutation":   "#F4D3D3",   # light red
    "human":      "#D9E6F7",   # blue
    "pynguin":    "#E0D9F2",   # purple
    "analysis":   "#F2E8D2",   # cream
    "border":     "#333333",
}
TXT_BIG   = {"fontsize": 11, "fontweight": "bold", "ha": "center", "va": "center"}
TXT_MED   = {"fontsize": 9.5, "ha": "center", "va": "center"}
TXT_SMALL = {"fontsize": 8.5, "ha": "center", "va": "center"}


def rounded_box(ax, x, y, w, h, color, edge=None, lw=1.0, alpha=1.0):
    """Draw a rounded rectangle with the given fill colour."""
    edge = edge or COLORS["border"]
    box = FancyBboxPatch((x, y), w, h,
                          boxstyle="round,pad=0.02,rounding_size=0.08",
                          linewidth=lw, edgecolor=edge, facecolor=color,
                          alpha=alpha)
    ax.add_patch(box)


def arrow(ax, x1, y1, x2, y2, color="#444"):
    arr = FancyArrowPatch((x1, y1), (x2, y2),
                          arrowstyle="-|>", mutation_scale=14,
                          linewidth=1.4, color=color)
    ax.add_patch(arr)


def main() -> int:
    OUTPUT.parent.mkdir(exist_ok=True)

    fig, ax = plt.subplots(figsize=(12, 9))
    ax.set_xlim(0, 12)
    ax.set_ylim(0, 9.5)
    ax.set_aspect("equal")
    ax.axis("off")

    # ── Title ─────────────────────────────────────────────────────────────
    ax.text(6, 9.0,
            "Experimental methodology",
            fontsize=14, fontweight="bold", ha="center")
    ax.text(6, 8.65,
            "Inputs (top) → 4 × 4 generation matrix → three parallel "
            "evaluation tracks → statistical analyses (bottom)",
            fontsize=9.5, ha="center", color="#555")

    # ── Level 1: Input dataset ────────────────────────────────────────────
    rounded_box(ax, 3.0, 7.6, 6.0, 0.7, COLORS["input"], lw=1.4)
    ax.text(6.0, 7.95, "Input dataset", **TXT_BIG)
    ax.text(6.0, 7.72,
            "100 functions = HumanEval + MBPP (seed = 42 shuffle); "
            "first 30 used per cell\n"
            "→ 9 HumanEval + 21 MBPP per cell",
            **TXT_SMALL)

    arrow(ax, 6.0, 7.55, 6.0, 7.25)

    # ── Level 2: Generation matrix (4 × 4) ────────────────────────────────
    rounded_box(ax, 0.4, 5.3, 11.2, 1.95, COLORS["generation"], lw=1.5)
    ax.text(6.0, 7.05, "4 × 4 generation matrix  (480 cells; n ≈ 4 – 30 "
                       "valid per cell after filter)",
            fontsize=11, fontweight="bold", ha="center", va="center")

    # Method boxes
    method_labels = [
        ("Plain LLM",          "no retrieval"),
        ("Random RAG",         "random chunks"),
        ("Simple RAG",         "cosine top-3"),
        ("Iterative\nCritique RAG", "generate-critique-refine"),
    ]
    for i, (label, sub) in enumerate(method_labels):
        x = 0.85 + i * 2.65
        rounded_box(ax, x, 6.3, 2.25, 0.55, COLORS["method"], lw=0.9)
        ax.text(x + 1.12, 6.62, label, fontsize=9.5, fontweight="bold",
                ha="center", va="center")
        ax.text(x + 1.12, 6.40, sub, fontsize=8, ha="center", va="center",
                color="#444")

    # Model row
    model_labels = [
        ("llama3.2", "3B dense"),
        ("phi4",     "14B dense"),
        ("qwen3.5",  "9B dense"),
        ("qwen3-coder", "30B-MoE"),
    ]
    ax.text(0.6, 5.65, "×", fontsize=14, ha="center", va="center", color="#555")
    for i, (label, sub) in enumerate(model_labels):
        x = 0.85 + i * 2.65
        rounded_box(ax, x, 5.4, 2.25, 0.55, COLORS["model"], lw=0.9)
        ax.text(x + 1.12, 5.72, label, fontsize=9.5, fontweight="bold",
                ha="center", va="center")
        ax.text(x + 1.12, 5.50, sub, fontsize=8, ha="center", va="center",
                color="#444")

    arrow(ax, 6.0, 5.25, 6.0, 4.95)

    # ── Level 3: Three parallel evaluation tracks ─────────────────────────
    ax.text(6.0, 4.85, "Three parallel evaluation tracks",
            fontsize=11, fontweight="bold", ha="center", va="center")

    # Track 1 — Mutation testing
    rounded_box(ax, 0.4, 3.0, 3.5, 1.7, COLORS["mutation"], lw=1.2)
    ax.text(2.15, 4.45, "1. Mutation testing", fontsize=10.5,
            fontweight="bold", ha="center", va="center")
    ax.text(2.15, 4.15,
            "5 operator families:\n"
            "arithmetic, comparison,\n"
            "boundary, return-None,\n"
            "negate-bool",
            fontsize=8.4, ha="center", va="center", color="#333")
    ax.text(2.15, 3.30,
            "Metric: kill_rate per cell\n"
            "(after test-filter on original)",
            fontsize=8.4, ha="center", va="center", style="italic",
            color="#666")

    # Track 2 — Human evaluation
    rounded_box(ax, 4.25, 3.0, 3.5, 1.7, COLORS["human"], lw=1.2)
    ax.text(6.0, 4.45, "2. Human evaluation", fontsize=10.5,
            fontweight="bold", ha="center", va="center")
    ax.text(6.0, 4.15,
            "40 stratified pairs\n"
            "3 annotators × 3 dims\n"
            "(test idiom / correctness /\n"
            "completeness) on 0–5 BARS",
            fontsize=8.4, ha="center", va="center", color="#333")
    ax.text(6.0, 3.30,
            "Metric: mean rating, Cohen's κ,\n"
            "Krippendorff's α (ordinal)",
            fontsize=8.4, ha="center", va="center", style="italic",
            color="#666")

    # Track 3 — Pynguin baseline
    rounded_box(ax, 8.1, 3.0, 3.5, 1.7, COLORS["pynguin"], lw=1.2)
    ax.text(9.85, 4.45, "3. Pynguin baseline", fontsize=10.5,
            fontweight="bold", ha="center", va="center")
    ax.text(9.85, 4.15,
            "Pynguin 0.45.0 SBST,\n"
            "60s budget per function,\n"
            "same 40 functions as\n"
            "human-eval subset",
            fontsize=8.4, ha="center", va="center", color="#333")
    ax.text(9.85, 3.30,
            "Metric: kill_rate, fed through\n"
            "the same mutation pipeline",
            fontsize=8.4, ha="center", va="center", style="italic",
            color="#666")

    # Arrows from generation matrix to tracks
    arrow(ax, 2.15, 4.95, 2.15, 4.75)
    arrow(ax, 6.0, 4.95, 6.0, 4.75)
    arrow(ax, 9.85, 4.95, 9.85, 4.75)

    # Arrows from tracks down to analysis
    arrow(ax, 2.15, 2.95, 2.15, 2.55)
    arrow(ax, 6.0, 2.95, 6.0, 2.55)
    arrow(ax, 9.85, 2.95, 9.85, 2.55)

    # ── Level 4: Analyses ─────────────────────────────────────────────────
    rounded_box(ax, 0.4, 0.85, 11.2, 1.65, COLORS["analysis"], lw=1.4)
    ax.text(6.0, 2.30, "Statistical analyses",
            fontsize=11, fontweight="bold", ha="center", va="center")
    ax.text(6.0, 1.92,
            "Per-cell:   Type-III ANOVA (kill_rate ∼ method + model + sample_idx)   "
            "·   Tukey HSD post-hoc",
            fontsize=9.3, ha="center", va="center")
    ax.text(6.0, 1.62,
            "Per-sample:  Mixed-LM (sample_idx random intercept; model + method fixed)   "
            "·   Kruskal-Wallis + Friedman + Wilcoxon",
            fontsize=9.3, ha="center", va="center")
    ax.text(6.0, 1.32,
            "Cross-cell:  Spearman ρ across LLMs (overall + per-operator)   "
            "·   Pearson r (RAG quality ↔ kill rate)",
            fontsize=9.3, ha="center", va="center")
    ax.text(6.0, 1.02,
            "Per-benchmark:  HumanEval vs MBPP slices (re-run ANOVA + Mixed-LM separately)",
            fontsize=9.3, ha="center", va="center")

    # ── Footer caption ────────────────────────────────────────────────────
    ax.text(6.0, 0.35,
            "Replication package: github.com/balajivenky06/autoresearch",
            fontsize=8.5, ha="center", va="center",
            style="italic", color="#555")

    plt.tight_layout()
    fig.savefig(OUTPUT, dpi=180, bbox_inches="tight")
    plt.close(fig)
    print(f"  {OUTPUT}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
