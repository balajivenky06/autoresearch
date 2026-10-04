#!/usr/bin/env python3
"""
plot_stale_figures.py — regenerate the three manuscript figures that still
dated from the 30-sample era.

Why this exists
---------------
plot_rebuild_figures.py refreshed the heatmaps and added four new figures, but
three PNGs in plots_mutation/ were never regenerated and still carried
30-sample numbers:

    noise_vs_kill_scatter.png      the faithfulness relationship
    mutation_rank_correlation.png  pairwise cross-model Spearman
    mutation_rank_stability.png    method rank trajectories

All three now disagree with the corrected results, and the first one disagrees
with its own conclusion: the pooled faithfulness-vs-kill-rate correlation is
confounded by model, so a single pooled regression line over eleven points is
exactly the misleading artifact the figure must not reproduce.

Design choices, and why
-----------------------
The faithfulness figure's job is to show that the points *cluster by model*,
because that clustering is the finding. Colour therefore carries model identity
(the four-slot categorical palette validated in plot_rebuild_figures.py:
adjacent-pair CVD dE 9.1 protan, 22.9 normal, all checks pass). The contrast
WARN on the aqua and yellow slots is relieved by a direct label on every
cluster plus a legend, so identity is never colour-alone. The pooled fit is
drawn in neutral grey and dashed — deliberately recessive, because it is the
artifact, not the result — while the within-model fits are drawn in each
model's own hue. Reading the figure should make the Simpson's-paradox structure
obvious before the caption is read.

The degenerate avg_noise_rate panel from the old three-panel layout is dropped.
Eleven points on a vertical line at x=0 carries no information; the fact is
stated in the table and the prose instead.

The rank-correlation figure plots a signed quantity on [-1, +1], so it uses the
diverging blue/red pair with a neutral midpoint at zero, never a sequential ramp
and never a rainbow. Cells are annotated, so magnitude survives without colour
perception.

The rank-stability figure encodes rank by vertical position with the axis
inverted so rank 1 sits on top; colour is redundant with the direct label at
each line's right end.

Usage:
    python3 plot_stale_figures.py
    python3 plot_stale_figures.py --outdir plots_mutation
"""
from __future__ import annotations

import argparse
import itertools
import sys
import warnings
from pathlib import Path

warnings.filterwarnings("ignore")

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy import stats
import statsmodels.formula.api as smf

from plot_rebuild_figures import (
    CAT4, DIV_MID, DIV_NEG, DIV_POS, GRID, INK, INK_2, SURFACE, save, style,
)

MODELS = ["llama3.2:latest", "phi4:14b", "qwen3.5:9b", "qwen3-coder:30b"]
MODEL_SHORT = {"llama3.2:latest": "llama3.2 3B", "phi4:14b": "phi4 14B",
               "qwen3.5:9b": "qwen3.5 9B", "qwen3-coder:30b": "qwen3-coder 30B"}
MODEL_TICK = {"llama3.2:latest": "llama3.2\n3B", "phi4:14b": "phi4\n14B",
              "qwen3.5:9b": "qwen3.5\n9B", "qwen3-coder:30b": "qwen3-coder\n30B MoE"}
MODEL_COLOR = dict(zip(MODELS, CAT4))

METHODS = ["Plain LLM", "Random RAG", "Simple RAG", "Iterative Critique"]
METHOD_COLOR = dict(zip(METHODS, CAT4))

# The eleven RAG cells, as joined by noise_vs_kill.py. Faithfulness is logged
# per cell by the generation harness (results_unitest.tsv); kill rate comes
# from results_mutation.tsv. There is no function-level pairing, which is one
# reason the relationship cannot be estimated within model from these columns
# alone — see sec:results-faithfulness.
CELLS = """method,model,kill,faith,judge
iterative_critique,llama3.2:latest,0.696285,0.1785,0.8377
iterative_critique,phi4:14b,0.909250,0.1353,0.7600
iterative_critique,qwen3-coder:30b,0.929349,0.1002,0.8261
random_rag,llama3.2:latest,0.706140,0.1787,0.8384
random_rag,phi4:14b,0.909160,0.0909,0.7119
random_rag,qwen3-coder:30b,0.942859,0.0960,0.8253
random_rag,qwen3.5:9b,0.959786,0.0521,0.8125
simple_rag,llama3.2:latest,0.724633,0.1851,0.8442
simple_rag,phi4:14b,0.915734,0.1306,0.7710
simple_rag,qwen3-coder:30b,0.929311,0.1114,0.8043
simple_rag,qwen3.5:9b,0.959431,0.0904,0.7969"""


def load_matrix(col: str = "mean_kill_rate") -> pd.DataFrame:
    df = pd.read_csv("results_mutation.tsv", sep="\t")
    df = df[df.method != "pynguin"]
    return df.pivot_table(index="method", columns="model",
                          values=col).reindex(METHODS)[MODELS]


# ──────────────────────────────────────────────────────────────────────
# 1. faithfulness — the confound, made visible
# ──────────────────────────────────────────────────────────────────────
def fig_faithfulness(out: Path) -> None:
    import io
    d = pd.read_csv(io.StringIO(CELLS))

    fig, axes = plt.subplots(1, 2, figsize=(11.6, 4.9))
    panels = [
        ("faith", "Token-overlap faithfulness", True),
        ("judge", "LLM-judge faithfulness (semantic)", False),
    ]
    for ax, (col, title, annotate_clusters) in zip(axes, panels):
        style(ax)
        xs, ys = d[col].values, d.kill.values

        # pooled fit — recessive, because it is the artifact
        b, a = np.polyfit(xs, ys, 1)
        gx = np.linspace(xs.min() - 0.008, xs.max() + 0.008, 50)
        ax.plot(gx, a + b * gx, color=INK_2, lw=1.6, ls=(0, (5, 3)),
                zorder=2, label="pooled fit")

        # within-model fits — in each model's own hue
        for mod, g in d.groupby("model"):
            if len(g) < 2:
                continue
            bb, aa = np.polyfit(g[col].values, g.kill.values, 1)
            lx = np.linspace(g[col].min(), g[col].max(), 20)
            ax.plot(lx, aa + bb * lx, color=MODEL_COLOR[mod], lw=2.0,
                    alpha=0.95, zorder=3, solid_capstyle="round")

        for mod in MODELS:
            g = d[d.model == mod]
            ax.scatter(g[col], g.kill, s=108, color=MODEL_COLOR[mod],
                       edgecolor=SURFACE, linewidth=2.0, zorder=4,
                       label=MODEL_SHORT[mod] if annotate_clusters else None)

        r, p = stats.pearsonr(xs, ys)
        rk = smf.ols("kill ~ C(model)", data=d).fit().resid
        rf = smf.ols(f"{col} ~ C(model)", data=d).fit().resid
        pr, pp = stats.pearsonr(rf, rk)
        ax.set_title(title, color=INK, fontsize=11.5, pad=26, loc="left")
        ax.text(0, 1.015,
                f"pooled $r$ = {r:+.3f} (p = {p:.3f})      "
                f"partial $r$ | model = {pr:+.3f} (p = {pp:.3f})",
                transform=ax.transAxes, color=INK_2, fontsize=9.6, va="bottom")
        ax.set_xlabel("avg_faithfulness" if col == "faith"
                      else "avg_llm_judge_faithfulness",
                      color=INK_2, fontsize=10)
        ax.set_ylabel("mean mutation kill rate", color=INK_2, fontsize=10)

        # Direct labels on the clusters: identity never rests on colour alone.
        # Anchor and offset are set per model because the clusters sit close
        # together on the left panel and a uniform offset collides.
        if annotate_clusters:
            for mod, anchor, dx, dy in (
                ("qwen3.5:9b",      "min", -6, -16),
                ("qwen3-coder:30b", "max", 10,  13),
                ("phi4:14b",        "max", 13,  -3),
                ("llama3.2:latest", "max", 13,   1),
            ):
                g = d[d.model == mod]
                i = g[col].idxmin() if anchor == "min" else g[col].idxmax()
                ax.annotate(MODEL_SHORT[mod], (d.loc[i, col], d.loc[i, "kill"]),
                            textcoords="offset points", xytext=(dx, dy),
                            color=MODEL_COLOR[mod], fontsize=9.2,
                            weight="medium", va="center")

    axes[0].legend(frameon=False, fontsize=9, loc="lower left",
                   labelcolor=INK_2, handletextpad=0.6)
    fig.suptitle("Faithfulness tracks the model, not the retrieval method",
                 color=INK, fontsize=13, x=0.012, ha="left", y=1.045)
    fig.text(0.012, 0.995,
             "Each point is one of the 11 RAG cells. Points cluster by model; "
             "the pooled fit (grey, dashed) runs down that clustering, while "
             "the within-model fits run flat or upward.",
             color=INK_2, fontsize=9.6, ha="left", va="top")
    fig.tight_layout(rect=(0, 0, 1, 0.97))
    save(fig, out)


# ──────────────────────────────────────────────────────────────────────
# 2. pairwise cross-model Spearman on overall kill rate
# ──────────────────────────────────────────────────────────────────────
def fig_rank_correlation(out: Path) -> None:
    piv = load_matrix()
    n = len(MODELS)
    M = np.full((n, n), np.nan)
    pairs = {}
    for i, j in itertools.combinations(range(n), 2):
        rho = stats.spearmanr(piv[MODELS[i]], piv[MODELS[j]]).correlation
        M[i, j] = M[j, i] = rho
        pairs[(i, j)] = rho
    # The diagonal is self-comparison, not data. Leave it NaN and paint it as
    # surface so a trivially perfect 1.0 cannot read as a result.
    (wi, wj), worst = min(pairs.items(), key=lambda kv: kv[1])
    worst_label = f"{MODEL_SHORT[MODELS[wi]].split()[0]}–{MODEL_SHORT[MODELS[wj]].split()[0]}"

    from matplotlib.colors import LinearSegmentedColormap
    cmap = LinearSegmentedColormap.from_list("div", [DIV_NEG, DIV_MID, DIV_POS])
    cmap.set_bad(SURFACE)

    fig, ax = plt.subplots(figsize=(7.4, 5.9))
    ax.set_facecolor(SURFACE)
    im = ax.imshow(np.ma.masked_invalid(M), cmap=cmap, vmin=-1, vmax=1)
    for i in range(n):
        for j in range(n):
            if i == j:
                continue
            v = M[i, j]
            ax.text(j, i, f"{v:+.2f}", ha="center", va="center",
                    color=SURFACE if abs(v) > 0.55 else INK,
                    fontsize=11.5, weight="medium")
    ax.set_xticks(range(n), [MODEL_TICK[m] for m in MODELS], fontsize=8.6)
    ax.set_yticks(range(n), [MODEL_TICK[m] for m in MODELS], fontsize=8.6)
    ax.tick_params(colors=INK_2, length=0)
    for s in ax.spines.values():
        s.set_visible(False)
    ax.set_xticks(np.arange(-.5, n, 1), minor=True)
    ax.set_yticks(np.arange(-.5, n, 1), minor=True)
    ax.grid(which="minor", color=SURFACE, linewidth=2.5)
    ax.tick_params(which="minor", length=0)

    cb = fig.colorbar(im, ax=ax, shrink=0.62, aspect=22, pad=0.04,
                      ticks=[-1, -0.5, 0, 0.5, 1])
    cb.outline.set_visible(False)
    cb.ax.tick_params(colors=INK_2, length=0, labelsize=9)
    cb.set_label("Spearman $\\rho$ between method orderings", color=INK_2,
                 fontsize=9.5)

    fig.suptitle("No model pair agrees on how to rank the methods",
                 color=INK, fontsize=13, x=0.013, ha="left", y=0.985)
    n_clear = sum(1 for v in pairs.values() if v >= 0.8)
    fig.text(0.013, 0.925,
             "Correlation of the four methods' kill-rate ordering between each "
             f"pair of LLMs. Only {n_clear} of the six pairs reaches the\n"
             f"$\\rho \\geq 0.8$ generalization threshold, and the {worst_label} "
             f"pair inverts almost perfectly ($\\rho = {worst:+.2f}$). "
             "The diagonal is self-comparison.",
             color=INK_2, fontsize=9.6, ha="left", va="top")
    fig.subplots_adjust(top=0.80, left=0.14, right=0.97, bottom=0.08)
    save(fig, out)


# ──────────────────────────────────────────────────────────────────────
# 3. rank trajectories across models
# ──────────────────────────────────────────────────────────────────────
def fig_rank_stability(out: Path) -> None:
    piv = load_matrix()
    ranks = piv.rank(ascending=False)

    fig, ax = plt.subplots(figsize=(8.8, 5.2))
    style(ax, grid_axis="y")
    x = np.arange(len(MODELS))
    for meth in METHODS:
        y = ranks.loc[meth, MODELS].values.astype(float)
        ax.plot(x, y, color=METHOD_COLOR[meth], lw=2.0, zorder=3,
                solid_capstyle="round", marker="o", markersize=9,
                markeredgecolor=SURFACE, markeredgewidth=2.0)
        ax.annotate(meth, (x[-1], y[-1]), textcoords="offset points",
                    xytext=(14, -4), color=METHOD_COLOR[meth],
                    fontsize=10, weight="medium", va="center")

    ax.set_xticks(x, [MODEL_TICK[m] for m in MODELS], fontsize=9.5)
    ax.set_yticks([1, 2, 3, 4], ["1\nbest", "2", "3", "4\nworst"], fontsize=9.5)
    ax.set_ylim(4.55, 0.45)
    ax.set_xlim(-0.35, len(MODELS) - 0.35)
    ax.set_ylabel("rank on mean kill rate", color=INK_2, fontsize=10)
    fig.suptitle("The best method is a different method on three of four models",
                 color=INK, fontsize=13, x=0.012, ha="left", y=0.985)
    fig.text(0.012, 0.925,
             "Iterative Critique is rank 1 on none of them. Two crossings are "
             "ties in the fourth decimal place (phi4 and qwen3-coder),\n"
             "so the ordering there is arbitrary rather than meaningful.",
             color=INK_2, fontsize=9.6, ha="left", va="top")
    fig.subplots_adjust(top=0.80, right=0.79, left=0.11, bottom=0.12)
    save(fig, out)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--outdir", default="plots_mutation")
    args = ap.parse_args()
    out = Path(args.outdir)
    out.mkdir(parents=True, exist_ok=True)

    fig_faithfulness(out / "noise_vs_kill_scatter.png")
    fig_rank_correlation(out / "mutation_rank_correlation.png")
    fig_rank_stability(out / "mutation_rank_stability.png")
    print("\n3 figures regenerated from the 100-function results.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
