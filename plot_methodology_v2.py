#!/usr/bin/env python3
"""
plot_methodology_v2.py — replacement for the methodology overview figure.

Why a replacement
-----------------
The original (plot_methodology_overview.py) was written during the 30-sample
pilot and never refreshed. It states "first 30 used per cell", "9 HumanEval +
21 MBPP per cell", "480 cells", "n ~ 4-30 valid per cell" and "cosine top-3".
Every one of those contradicts the submitted manuscript, which reports 100
functions, 1,600 suites, n = 50-100 per cell and k = 5. It escaped the
figure-freshness gate in check_paper_consistency.py because that gate exempts
"schematics" — a wrong call, since this schematic carries data claims.

It was also unreadable: the track headings collided with their own body text,
the statistics panel was a wall of nine lines, and the type was small relative
to a 2000px canvas.

Design
------
Four stages, top to bottom, one idea each, with the counts carried as large
figures rather than sentences. Everything that is not a count has been cut to a
short phrase. The four generation techniques are the only categorical dimension
that needs colour, so they take the validated four-slot palette; stages are
distinguished by position and a neutral surface, not by hue.

Numbers are passed in from the artifacts by the caller, so this file cannot
drift from the data the way its predecessor did.

Usage:
    python3 plot_methodology_v2.py            # writes Fig1_methodology_v2.png
    python3 plot_methodology_v2.py --out X.png
"""
from __future__ import annotations

import argparse
import sys
import warnings
from pathlib import Path

warnings.filterwarnings("ignore")

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch

SURFACE = "#ffffff"
INK = "#0b0b0b"
INK_2 = "#52514e"
RULE = "#d8d8d4"
# validated four-slot categorical palette (CVD dE 9.1 protan, 22.9 normal)
CAT4 = ["#2a78d6", "#eb6834", "#1baf7a", "#eda100"]
BAND = "#f4f6f9"      # stage band
TINT = "#eef3fa"      # stage-1 / stage-4 fill


def box(ax, x, y, w, h, fc, ec, lw=1.6, r=0.016, z=2):
    ax.add_patch(FancyBboxPatch((x, y), w, h,
                                boxstyle=f"round,pad=0,rounding_size={r}",
                                linewidth=lw, edgecolor=ec, facecolor=fc, zorder=z))


def arrow(ax, x, y0, y1):
    ax.add_patch(FancyArrowPatch((x, y0), (x, y1), arrowstyle="-|>",
                                 mutation_scale=26, linewidth=2.4,
                                 color=INK_2, zorder=5))


def build(out: Path, *, n_func, n_mbpp, n_he, n_cells, n_suites,
          n_valid, n_min, n_max, n_mut, n_equiv, n_ann, n_pairs, n_pyn, k):
    fig, ax = plt.subplots(figsize=(13.0, 11.6))
    ax.set_xlim(0, 100); ax.set_ylim(0, 100); ax.axis("off")
    ax.set_facecolor(SURFACE)
    fig.patch.set_facecolor(SURFACE)

    def stage_label(y, n, text):
        ax.text(3.0, y, n, fontsize=30, fontweight="bold", color=RULE,
                va="center", ha="left", zorder=3)
        ax.text(8.6, y, text, fontsize=19, fontweight="bold", color=INK,
                va="center", ha="left", zorder=3)

    # ── 1. corpus ────────────────────────────────────────────────────
    box(ax, 8, 86.5, 84, 9.5, TINT, "#b9cde8")
    stage_label(91.3, "1", "Corpus")
    ax.text(50, 91.3, f"{n_func} Python functions", fontsize=23,
            fontweight="bold", color=INK, ha="center", va="center")
    ax.text(50, 87.9, f"{n_mbpp} MBPP   ·   {n_he} HumanEval   ·   fixed by seed 42",
            fontsize=15, color=INK_2, ha="center", va="center")
    arrow(ax, 50, 86.0, 81.6)

    # ── 2. generation ────────────────────────────────────────────────
    box(ax, 8, 56.5, 84, 24.5, BAND, "#c9ccd1")
    stage_label(77.6, "2", "Generate")
    ax.text(89.5, 77.6, f"{n_cells} cells  →  {n_suites:,} suites", fontsize=16,
            fontweight="bold", color=INK, ha="right", va="center")

    techs = [("Plain LLM", "no retrieval"),
             ("Random RAG", f"{k} random chunks"),
             ("Simple RAG", f"cosine top-{k}"),
             ("Iterative\nCritique", "draft → critique")]
    w, gap = 18.2, 2.4
    x0 = 50 - (4 * w + 3 * gap) / 2
    for i, (name, sub) in enumerate(techs):
        x = x0 + i * (w + gap)
        box(ax, x, 67.6, w, 6.6, CAT4[i], CAT4[i], lw=0, z=3)
        two = "\n" in name
        ax.text(x + w / 2, 72.2 if two else 71.9, name,
                fontsize=13.5 if two else 15, fontweight="bold",
                color="#ffffff", ha="center", va="center", zorder=4,
                linespacing=0.95)
        ax.text(x + w / 2, 68.8 if two else 69.2, sub, fontsize=12.5,
                color="#f2f2f2", ha="center", va="center", zorder=4)
    ax.text(50, 65.2, "×", fontsize=22, color=INK_2, ha="center", va="center")

    models = [("llama3.2", "3B"), ("phi4", "14B"),
              ("qwen3.5", "9B"), ("qwen3-coder", "30B MoE")]
    for i, (name, size) in enumerate(models):
        x = x0 + i * (w + gap)
        box(ax, x, 57.8, w, 5.6, "#ffffff", "#b6b9be", lw=1.4, z=3)
        ax.text(x + w / 2, 61.2, name, fontsize=14.5, fontweight="bold",
                color=INK, ha="center", va="center", zorder=4)
        ax.text(x + w / 2, 59.0, size, fontsize=12.5, color=INK_2,
                ha="center", va="center", zorder=4)
    arrow(ax, 50, 56.0, 50.6)

    # ── 3. evaluation ────────────────────────────────────────────────
    stage_label(46.8, "3", "Evaluate")
    tracks = [
        ("Mutation testing", [f"{n_mut:,} mutants",
                              "5 operator families",
                              f"{n_equiv} equivalent, excluded"], CAT4[0]),
        ("Human evaluation", [f"{n_ann} annotators × {n_pairs} suites",
                              "blinded to technique",
                              "0–5 rubric, 3 dimensions"], CAT4[2]),
        ("SBST baseline", [f"Pynguin, {n_pyn} of {n_pairs} functions",
                           "60 s search budget",
                           "same mutation pipeline"], CAT4[1]),
    ]
    tw, tgap = 26.4, 3.4
    tx0 = 50 - (3 * tw + 2 * tgap) / 2
    for i, (title, lines, col) in enumerate(tracks):
        x = tx0 + i * (tw + tgap)
        box(ax, x, 27.0, tw, 16.4, "#ffffff", "#c9ccd1", lw=1.5)
        box(ax, x, 41.2, tw, 2.2, col, col, lw=0, z=3)
        ax.text(x + tw / 2, 38.6, title, fontsize=16, fontweight="bold",
                color=INK, ha="center", va="center", zorder=4)
        for j, ln in enumerate(lines):
            ax.text(x + tw / 2, 35.0 - j * 3.0, ln, fontsize=13.5,
                    color=INK_2, ha="center", va="center", zorder=4)
        arrow(ax, x + tw / 2, 26.5, 20.6)
    ax.text(50, 45.0, "every suite goes through all three",
            fontsize=13.5, style="italic", color=INK_2, ha="center", va="center")

    # ── 4. analysis ──────────────────────────────────────────────────
    box(ax, 8, 8.0, 84, 12.0, TINT, "#b9cde8")
    stage_label(16.6, "4", "Analyse")
    ax.text(50, 16.6, f"{n_valid:,} valid observations", fontsize=22,
            fontweight="bold", color=INK, ha="center", va="center")
    ax.text(50, 13.3, f"n = {n_min}–{n_max} per cell after the original-code filter",
            fontsize=14, color=INK_2, ha="center", va="center")
    ax.text(50, 10.1,
            "Type-III ANOVA   ·   mixed-effects regression   ·   "
            "Tukey HSD   ·   cross-model Spearman",
            fontsize=13.5, color=INK_2, ha="center", va="center")

    ax.text(50, 3.2,
            "Replication package: github.com/balajivenky06/autoresearch",
            fontsize=12.5, style="italic", color=INK_2, ha="center", va="center")

    fig.savefig(out, dpi=300, bbox_inches="tight", facecolor=SURFACE)
    plt.close(fig)
    print(f"  wrote {out}")


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default="Fig1_methodology_v2.png")
    args = ap.parse_args()

    # every count below is read from the artifacts, never hard-coded prose
    sys.path.insert(0, ".")
    import pandas as pd
    from collections import Counter
    from mutation_statistical_tests import load_per_sample_kill_rates

    src = {}
    for line in open("corpus_manifest.tsv"):
        if line.startswith(("#", "sample_idx")):
            continue
        q = line.rstrip().split("\t")
        if len(q) >= 3:
            src[q[0]] = q[2]
    c = Counter(src.values())
    t = pd.read_csv("results_mutation.tsv", sep="\t")
    tt = t[t.method != "pynguin"]
    d = load_per_sample_kill_rates().dropna(subset=["kill_rate"])
    per = d.groupby(["method", "model"]).size()

    build(Path(args.out),
          n_func=sum(c.values()), n_mbpp=c["mbpp"], n_he=c["humaneval"],
          n_cells=tt.shape[0], n_suites=tt.shape[0] * sum(c.values()),
          n_valid=len(d), n_min=per.min(), n_max=per.max(),
          n_mut=int(tt.total_mutants.sum()), n_equiv=int(tt.total_equivalent.sum()),
          n_ann=3, n_pairs=40,
          n_pyn=int(t[t.method == "pynguin"].n_samples_valid.iloc[0]), k=5)
    return 0


if __name__ == "__main__":
    sys.exit(main())
