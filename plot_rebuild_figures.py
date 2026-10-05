#!/usr/bin/env python3
"""
plot_rebuild_figures.py — regenerate the manuscript's figures from the verified
100-function data, and add the four the rebuild earned.

Every figure is derived from the per-sample analysis checkpoints. The stale PNGs
in plots_mutation/ date from the 30-sample era and disagree with the corrected
results, so they must not be reused.

Figures produced (into plots_mutation/):
    kill_rate_heatmap.png           replaces stale  — method x model magnitude
    kill_rate_boundary_heatmap.png  replaces stale  — same, boundary operator
    fig_model_vs_method.png         NEW  — the 21.7x headline, made visual
    fig_attrition.png               NEW  — IC collapse as model size falls
    fig_contamination_delta.png     NEW  — the decontamination answer
    fig_kill_rate_distribution.png  NEW  — the ceiling that justifies rank tests

Design choices, and why
-----------------------
Form before colour. Where a categorical dimension is already carried by the
row/column position, colour carries nothing and is held to a single hue — that is
the case in fig_model_vs_method, where the message is the *spread of rows*, not
the identity of points. Only fig_attrition needs categorical colour (four method
lines); that four-slot palette was validated before this file was written
(adjacent-pair CVD dE 9.1, normal-vision 22.9, all checks pass) and the contrast
WARN on aqua/yellow is relieved by direct labels on every line, so identity is
never colour-alone.

Model capability is ordinal, so the heatmaps use a single-hue sequential ramp
rather than categorical hues, and every cell is annotated — magnitude is readable
without perceiving colour at all.

The delta figure is the one genuinely diverging quantity (does renaming help or
hurt?), so it uses the blue/red pair with a neutral zero line, never a rainbow.

Usage:
    python3 plot_rebuild_figures.py
    python3 plot_rebuild_figures.py --outdir plots_mutation
"""
from __future__ import annotations

import argparse
import pickle
import statistics as st
import sys
import warnings
from pathlib import Path

warnings.filterwarnings("ignore")

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.colors import LinearSegmentedColormap

from mutation_statistical_tests import (
    BASELINE_TOOLS, METHOD_LABELS, METHODS, OPERATORS,
    load_per_sample_kill_rates, parse_key,
)

# ── palette (validated reference instance, light mode on a white page) ──
SURFACE = "#ffffff"
INK = "#0b0b0b"
INK_2 = "#52514e"
GRID = "#dcdcd8"
SERIES = "#2a78d6"                                   # categorical slot 1
CAT4 = ["#2a78d6", "#eb6834", "#1baf7a", "#eda100"]  # slots 1-4, validated
SEQ = ["#cde2fb", "#9ec5f4", "#6da7ec", "#3987e5", "#256abf", "#184f95", "#0d366b"]
DIV_POS, DIV_NEG, DIV_MID = "#2a78d6", "#e34948", "#f0efec"

SEQ_CMAP = LinearSegmentedColormap.from_list("seq_blue", SEQ)
DECON_DIR = Path("checkpoints_mutation_decontam_analysis")
MODELS = ["llama3.2_latest", "phi4_14b", "qwen3.5_9b", "qwen3-coder_30b"]
MODEL_LABEL = {"llama3.2_latest": "llama3.2\n3B", "phi4_14b": "phi4\n14B",
               "qwen3.5_9b": "qwen3.5\n9B", "qwen3-coder_30b": "qwen3-coder\n30B MoE"}
SIZE_ORDER = ["llama3.2_latest", "qwen3.5_9b", "phi4_14b", "qwen3-coder_30b"]


def style(ax, *, grid_axis="both"):
    """Recessive chrome: no box, thin ticks, grid behind the marks."""
    ax.set_facecolor(SURFACE)
    for s in ("top", "right"):
        ax.spines[s].set_visible(False)
    for s in ("left", "bottom"):
        ax.spines[s].set_color(GRID)
        ax.spines[s].set_linewidth(0.8)
    ax.tick_params(colors=INK_2, labelsize=9, length=3, width=0.8)
    if grid_axis != "none":
        ax.grid(True, axis=grid_axis, color=GRID, linewidth=0.6, alpha=0.9)
        ax.set_axisbelow(True)


def save(fig, path: Path):
    fig.savefig(path, dpi=300, bbox_inches="tight", facecolor=SURFACE)
    plt.close(fig)
    print(f"  wrote {path}")


def load_decon():
    rows = []
    for f in sorted(DECON_DIR.glob("*.pkl")):
        if f.name.endswith(".tmp"):
            continue
        method, _, model = parse_key(f.stem)
        if method in BASELINE_TOOLS:
            continue
        for _, r in pickle.load(f.open("rb")).items():
            kr = r.get("kill_rate")
            if kr is None or (isinstance(kr, float) and kr != kr):
                continue
            rows.append({"method": method, "model": model, "kill_rate": float(kr)})
    import pandas as pd
    return pd.DataFrame(rows)


# ── 1. the headline: model separates, method does not ───────────────────
def fig_model_vs_method(cell, out: Path):
    fig, axes = plt.subplots(1, 2, figsize=(11.6, 3.9), sharex=True)
    # the right panel's row labels sit left of its axis; without this gap they
    # overprint the left panel's data
    fig.subplots_adjust(wspace=0.42)
    panels = [
        ("Grouped by model", MODELS, lambda k: [cell.loc[(k, m), "mean"] for m in METHODS],
         [MODEL_LABEL[m].replace("\n", " ") for m in MODELS]),
        ("Grouped by RAG technique", METHODS, lambda m: [cell.loc[(k, m), "mean"] for k in MODELS],
         [METHOD_LABELS[m] for m in METHODS]),
    ]
    for ax, (title, keys, getter, labels) in zip(axes, panels):
        style(ax, grid_axis="x")
        for i, key in enumerate(keys):
            vals = getter(key)
            y = len(keys) - 1 - i
            ax.plot([min(vals), max(vals)], [y, y], color=GRID, lw=5,
                    solid_capstyle="round", zorder=1)
            ax.scatter(vals, [y] * len(vals), s=46, color=SERIES, zorder=3,
                       edgecolor=SURFACE, linewidth=1.4)
            ax.scatter([st.mean(vals)], [y], marker="|", s=320, color=INK,
                       linewidth=1.8, zorder=4)
            ax.text(min(vals) - 0.012, y, f"{max(vals)-min(vals):.3f}", ha="right",
                    va="center", fontsize=8, color=INK_2,
                    fontfamily="monospace")
        ax.set_yticks(range(len(keys)))
        ax.set_yticklabels(labels[::-1], fontsize=9.5, color=INK)
        ax.set_title(title, fontsize=10.5, color=INK, pad=9, loc="left")
        ax.set_xlim(0.63, 1.0)
    axes[0].set_xlabel("mean mutation kill rate", fontsize=9.5, color=INK_2)
    axes[1].set_xlabel("mean mutation kill rate", fontsize=9.5, color=INK_2)
    fig.text(0.5, -0.09,
             "Each dot is one of the 16 technique x model cells; the bar spans its row's "
             "range, the rule marks the row mean,\nand the figure at left is that row's "
             "spread. Hold the model fixed (left) and the four techniques span 0.017-0.047; "
             "hold the\ntechnique fixed (right) and the four models span 0.185-0.255 - "
             "roughly five times wider, on the same axis.",
             ha="center", fontsize=8.5, color=INK_2)
    save(fig, out)


# ── 2. heatmaps: magnitude over two categoricals ────────────────────────
def fig_heatmap(cell, metric_cell, out: Path, title: str, note: str):
    M = np.array([[metric_cell.get((k, m), np.nan) for k in MODELS] for m in METHODS])
    fig, ax = plt.subplots(figsize=(7.0, 3.3))
    im = ax.imshow(M, cmap=SEQ_CMAP, vmin=np.nanmin(M) - 0.02, vmax=1.0, aspect="auto")
    ax.set_xticks(range(len(MODELS)))
    ax.set_xticklabels([MODEL_LABEL[m] for m in MODELS], fontsize=9, color=INK)
    ax.set_yticks(range(len(METHODS)))
    ax.set_yticklabels([METHOD_LABELS[m] for m in METHODS], fontsize=9.5, color=INK)
    for i in range(len(METHODS)):
        for j in range(len(MODELS)):
            v = M[i, j]
            if np.isnan(v):
                continue
            # annotate every cell: magnitude is legible without perceiving colour
            ax.text(j, i, f"{v:.3f}", ha="center", va="center", fontsize=9.5,
                    fontfamily="monospace",
                    color="#ffffff" if v > 0.90 else INK)
    ax.set_xticks(np.arange(-.5, len(MODELS), 1), minor=True)
    ax.set_yticks(np.arange(-.5, len(METHODS), 1), minor=True)
    ax.grid(which="minor", color=SURFACE, linewidth=2)   # 2px surface gap
    ax.tick_params(which="minor", length=0)
    ax.tick_params(colors=INK_2, length=0)
    for s in ax.spines.values():
        s.set_visible(False)
    # Marginal means, printed outside the grid. These carry the finding —
    # the row margin barely moves, the column margin spans 0.23 — and
    # folding them in here is what lets the equivalent body table go.
    rowmu = np.nanmean(M, axis=1)
    colmu = np.nanmean(M, axis=0)
    for i, v in enumerate(rowmu):
        ax.text(len(MODELS) - 0.35, i, f"{v:.3f}", ha="left", va="center",
                fontsize=9.5, fontfamily="monospace", color=INK)
    for j, v in enumerate(colmu):
        ax.text(j, len(METHODS) - 0.42, f"{v:.3f}", ha="center", va="top",
                fontsize=9.5, fontfamily="monospace", color=INK)
    ax.text(len(MODELS) - 0.35, -0.72, "mean", ha="left", va="center",
            fontsize=8.5, color=INK_2)
    ax.text(-0.62, len(METHODS) - 0.42, "mean", ha="right", va="top",
            fontsize=8.5, color=INK_2)
    ax.set_xlim(-0.5, len(MODELS) + 0.35)
    ax.set_ylim(len(METHODS) + 0.15, -0.5)

    ax.set_title(title, fontsize=11, color=INK, pad=10, loc="left")
    cb = fig.colorbar(im, ax=ax, fraction=0.025, pad=0.10)
    cb.outline.set_visible(False)
    cb.ax.tick_params(colors=INK_2, labelsize=8, length=2)
    fig.text(0.0, -0.14, note, ha="left", fontsize=8.5, color=INK_2)
    save(fig, out)


# ── 3. attrition: the new finding ───────────────────────────────────────
def fig_attrition(cell, out: Path):
    fig, ax = plt.subplots(figsize=(8.2, 4.0))
    style(ax, grid_axis="y")
    x = range(len(SIZE_ORDER))
    for slot, m in enumerate(METHODS):
        y = [int(cell.loc[(k, m), "size"]) for k in SIZE_ORDER]
        ax.plot(x, y, color=CAT4[slot], lw=2, marker="o", markersize=8,
                markeredgecolor=SURFACE, markeredgewidth=1.4, zorder=3)
        # Direct-label every series: identity is never colour-alone, and this
        # relieves the contrast WARN on the aqua and yellow slots. Labels go at
        # the LEFT end, where the four lines are well separated (95/87/80/50);
        # at the right they converge on ~99 and overprint into an unreadable pile.
        ax.annotate(METHOD_LABELS[m], (0, y[0]),
                    textcoords="offset points", xytext=(-12, 0), va="center",
                    ha="right", fontsize=9.5, color=INK, fontweight="semibold")
    ax.set_xticks(list(x))
    ax.set_xticklabels([MODEL_LABEL[k] for k in SIZE_ORDER], fontsize=9, color=INK)
    ax.set_ylim(0, 108)
    ax.set_ylabel("functions with a usable test suite  (of 100)", fontsize=9.5, color=INK_2)
    ax.set_title("Iterative critique fails on small models", fontsize=11, color=INK,
                 pad=10, loc="left")
    ax.set_xlim(-1.35, len(SIZE_ORDER) - 0.75)
    fig.text(0.0, -0.08,
             "A suite counts as usable only if it passes against the unmutated function. "
             "Iterative Critique loses half its\nsamples on a 3B model and almost none on "
             "30B; the other three techniques are flat across model scale.",
             ha="left", fontsize=8.5, color=INK_2)
    save(fig, out)


# ── 4. contamination delta: the only diverging quantity ─────────────────
def fig_contamination(cell, dcell, out: Path):
    rows = []
    for m in METHODS:
        for k in MODELS:
            if (k, m) in dcell.index and (k, m) in cell.index:
                rows.append((f"{METHOD_LABELS[m]} × {MODEL_LABEL[k].split(chr(10))[0]}",
                             dcell.loc[(k, m), "mean"] - cell.loc[(k, m), "mean"]))
    rows.sort(key=lambda r: r[1])
    labels = [r[0] for r in rows]
    vals = [r[1] for r in rows]
    fig, ax = plt.subplots(figsize=(7.4, 5.0))
    style(ax, grid_axis="x")
    y = range(len(rows))
    ax.axvline(0, color=INK_2, lw=1.1, zorder=2)
    for i, v in enumerate(vals):
        c = DIV_POS if v >= 0 else DIV_NEG
        ax.plot([0, v], [i, i], color=c, lw=2.4, solid_capstyle="round", zorder=3)
        ax.scatter([v], [i], s=52, color=c, zorder=4,
                   edgecolor=SURFACE, linewidth=1.4)
        ax.text(v + (0.0035 if v >= 0 else -0.0035), i, f"{v:+.3f}",
                ha="left" if v >= 0 else "right", va="center",
                fontsize=8.5, color=INK_2, fontfamily="monospace")
    mean = st.mean(vals)
    ax.axvline(mean, color=INK, lw=1, ls=(0, (4, 3)), zorder=2)
    ax.text(mean, len(rows) - 0.2, f" mean {mean:+.4f}", fontsize=8.5,
            color=INK, va="bottom")
    ax.set_yticks(list(y))
    ax.set_yticklabels(labels, fontsize=8.5, color=INK)
    ax.set_xlabel("change in kill rate after renaming function and parameters",
                  fontsize=9.5, color=INK_2)
    ax.set_title("No evidence of benchmark memorisation", fontsize=11, color=INK,
                 pad=10, loc="left")
    ax.set_xlim(min(vals) - 0.022, max(vals) + 0.022)
    fig.text(0.0, -0.055,
             "Semantics-preserving AST renaming across 98 functions. Memorised suites "
             "would lose kill rate when identifiers\nchange; 12 of 16 cells instead "
             "improve slightly, and no cell falls more than 0.02.",
             ha="left", fontsize=8.5, color=INK_2)
    save(fig, out)


# ── 5. the ceiling that justifies rank tests ────────────────────────────
def fig_distribution(df, out: Path):
    fig, ax = plt.subplots(figsize=(7.0, 3.3))
    style(ax, grid_axis="y")
    ax.hist(df.kill_rate, bins=np.linspace(0, 1, 41), color=SERIES,
            edgecolor=SURFACE, linewidth=0.8)
    n1 = int((df.kill_rate == 1).sum())
    n0 = int((df.kill_rate == 0).sum())
    ax.annotate(f"{n1} observations\nat exactly 1.0  ({100*n1/len(df):.1f}%)",
                (1.0, n1), textcoords="offset points", xytext=(-14, -26),
                ha="right", fontsize=9, color=INK, fontweight="semibold")
    ax.annotate(f"{n0} at 0.0", (0.0, n0), textcoords="offset points",
                xytext=(14, 14), ha="left", fontsize=9, color=INK_2)
    ax.set_xlabel("per-sample mutation kill rate", fontsize=9.5, color=INK_2)
    ax.set_ylabel("observations", fontsize=9.5, color=INK_2)
    ax.set_title("Kill rate is boundary-concentrated", fontsize=11, color=INK,
                 pad=10, loc="left")
    fig.text(0.0, -0.12,
             f"All {len(df)} valid observations. Only "
             f"{int(((df.kill_rate>0)&(df.kill_rate<1)).sum())} ("
             f"{100*((df.kill_rate>0)&(df.kill_rate<1)).mean():.0f}%) fall strictly "
             "between the extremes, which is why rank tests\nlead the analysis and the "
             "Gaussian mixed model is reported only as a robustness check.",
             ha="left", fontsize=8.5, color=INK_2)
    save(fig, out)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--outdir", default="plots_mutation")
    args = ap.parse_args()
    out = Path(args.outdir); out.mkdir(exist_ok=True)

    df = load_per_sample_kill_rates()
    cell = df.groupby(["model", "method"])["kill_rate"].agg(["mean", "size"])
    kr = {(k, m): cell.loc[(k, m), "mean"] for k in MODELS for m in METHODS}
    bd = df.dropna(subset=["kill_rate_boundary"]).groupby(
        ["model", "method"])["kill_rate_boundary"].mean().to_dict()

    print(f"figures from {len(df)} observations, {df.sample_idx.nunique()} functions")
    fig_model_vs_method(cell, out / "fig_model_vs_method.png")
    fig_heatmap(cell, kr, out / "kill_rate_heatmap.png",
                "Mutation kill rate by RAG technique and model",
                "Cells annotated; model capability is ordinal so a single-hue "
                "sequential ramp is used rather than categorical hues.")
    fig_heatmap(cell, bd, out / "kill_rate_boundary_heatmap.png",
                "Boundary-operator kill rate by RAG technique and model",
                "Boundary mutants (n plus or minus 1) are the hardest operator family.")
    fig_attrition(cell, out / "fig_attrition.png")
    fig_distribution(df, out / "fig_kill_rate_distribution.png")
    if DECON_DIR.is_dir():
        dd = load_decon()
        dcell = dd.groupby(["model", "method"])["kill_rate"].agg(["mean", "size"])
        fig_contamination(cell, dcell, out / "fig_contamination_delta.png")
    else:
        print("  (decontaminated arm absent — skipping the delta figure)")
    return 0


if __name__ == "__main__":
    sys.exit(main())
