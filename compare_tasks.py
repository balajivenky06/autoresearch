"""
compare_tasks.py — Cross-task faithfulness comparison for RQ4.

Reads results TSVs from all completed PhD tasks and generates
publication-quality charts for cross-task analysis.

Workflow:
    1. python extract_docstring_results.py       # creates results_docstring.tsv
    2. (run unit test experiments)               # populates results_unitest.tsv
    3. python compare_tasks.py                   # generates plots_compare/

Output charts (plots_compare/):
    faithfulness_by_task.png   — faithfulness per method × task (RQ4 main chart)
    noise_vs_faithfulness.png  — scatter: noise rate vs faithfulness (RQ2 diagnostic)
    pareto.png                 — cost vs quality Pareto frontier (RQ4 efficiency)
    summary_table.tsv          — cross-task comparison table for thesis appendix
"""

import sys
import math
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from pathlib import Path

# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------

TASK_FILES = {
    "Test Oracle": "results_unitest.tsv",
    "Docstring":   "results_docstring.tsv",
}

OUTPUT_DIR = Path("plots_compare")

METHODS    = ["plain_llm", "simple_rag", "iterative_critique"]
REASONINGS = ["base", "cot", "tot", "got"]

METHOD_LABELS = {
    "plain_llm":          "Plain LLM",
    "simple_rag":         "Simple RAG",
    "iterative_critique": "Iterative\nCritique",
}
METHOD_COLORS = {
    "plain_llm":          "#4C72B0",
    "simple_rag":         "#DD8452",
    "iterative_critique": "#55A868",
}
TASK_HATCHES = {
    "Test Oracle": "",
    "Docstring":   "///",
}


# ---------------------------------------------------------------------------
# Data loading
# ---------------------------------------------------------------------------

def _parse_method(method_str: str):
    """Split 'simple_rag/cot' into ('simple_rag', 'cot')."""
    parts = str(method_str).split("/", 1)
    return (parts[0], parts[1]) if len(parts) == 2 else (parts[0], "base")


def load_task(path: str, task_name: str) -> pd.DataFrame | None:
    p = Path(path)
    if not p.exists():
        print(f"  '{task_name}' — {path} not found, skipping.")
        return None
    df = pd.read_csv(p, sep="\t")
    if df.empty:
        print(f"  '{task_name}' — {path} is empty, skipping.")
        return None
    df[["method_name", "reasoning"]] = pd.DataFrame(
        df["method"].map(_parse_method).tolist(), index=df.index
    )
    df["task"] = task_name
    df = df[df["status"] != "crash"].copy()
    for col in ["val_score", "avg_faithfulness", "avg_noise_rate",
                "avg_retrieval_secs", "avg_llm_secs"]:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce")
    print(f"  '{task_name}' — {len(df)} rows loaded from {path}")
    return df


def _best_per_method(df: pd.DataFrame) -> pd.DataFrame:
    """Return one row per method: the run with highest val_score (or first if val_score missing)."""
    rows = []
    for method in METHODS:
        sub = df[df["method_name"] == method]
        if sub.empty:
            continue
        if "val_score" in sub.columns and sub["val_score"].notna().any():
            rows.append(sub.loc[sub["val_score"].idxmax()])
        else:
            rows.append(sub.iloc[0])
    return pd.DataFrame(rows) if rows else pd.DataFrame()


# ---------------------------------------------------------------------------
# Chart 1: Faithfulness by method × task (main RQ4 chart)
# ---------------------------------------------------------------------------

def plot_faithfulness_by_task(task_dfs: dict) -> None:
    fig, ax = plt.subplots(figsize=(10, 5))

    n_tasks   = len(task_dfs)
    n_methods = len(METHODS)
    group_w   = 0.7
    bar_w     = group_w / n_tasks
    x         = np.arange(n_methods)

    has_data = False
    for t_idx, (task_name, df) in enumerate(task_dfs.items()):
        best = _best_per_method(df)
        if best.empty or "avg_faithfulness" not in best.columns:
            continue
        for m_idx, method in enumerate(METHODS):
            row = best[best["method_name"] == method]
            if row.empty:
                continue
            fval = row["avg_faithfulness"].values[0]
            if math.isnan(fval):
                # Draw an "N/A" indicator for plain_llm
                ax.text(x[m_idx] + t_idx * bar_w - group_w / 2 + bar_w / 2,
                        0.02, "N/A", ha="center", va="bottom",
                        fontsize=8, color="grey", style="italic")
                continue
            has_data = True
            bar = ax.bar(
                x[m_idx] + t_idx * bar_w - group_w / 2 + bar_w / 2,
                fval, bar_w * 0.85,
                color=METHOD_COLORS[method],
                hatch=TASK_HATCHES[task_name],
                alpha=0.85, edgecolor="white",
            )
            ax.text(bar[0].get_x() + bar[0].get_width() / 2,
                    fval + 0.008, f"{fval:.3f}",
                    ha="center", va="bottom", fontsize=8)

    if not has_data:
        print("  faithfulness_by_task.png SKIPPED — no faithfulness data in any task")
        plt.close(fig)
        return

    ax.set_xticks(x)
    ax.set_xticklabels([METHOD_LABELS[m] for m in METHODS], fontsize=11)
    ax.set_ylabel("Faithfulness (token-overlap, unified metric)", fontsize=11)
    ax.set_ylim(0, 1.08)
    ax.set_title("Cross-Task Faithfulness by Retrieval Method\n(RQ4: unified metric)",
                 fontsize=13, fontweight="bold")
    ax.grid(axis="y", alpha=0.3)

    # Legend: task hatches
    task_patches = [
        mpatches.Patch(facecolor="grey", hatch=TASK_HATCHES[t], label=t, alpha=0.7)
        for t in task_dfs
    ]
    ax.legend(handles=task_patches, fontsize=10, title="Task", title_fontsize=10)

    plt.tight_layout()
    fig.savefig(OUTPUT_DIR / "faithfulness_by_task.png", dpi=150)
    plt.close(fig)
    print("  faithfulness_by_task.png")


# ---------------------------------------------------------------------------
# Chart 2: Noise rate vs faithfulness scatter (RQ2 diagnostic)
# ---------------------------------------------------------------------------

def plot_noise_vs_faithfulness(task_dfs: dict) -> None:
    fig, ax = plt.subplots(figsize=(8, 6))
    has_data = False

    for task_name, df in task_dfs.items():
        if "avg_noise_rate" not in df.columns or "avg_faithfulness" not in df.columns:
            continue
        sub = df[["method_name", "reasoning", "avg_noise_rate", "avg_faithfulness"]].dropna()
        if sub.empty:
            continue
        has_data = True
        for _, row in sub.iterrows():
            method = row["method_name"]
            ax.scatter(
                row["avg_noise_rate"], row["avg_faithfulness"],
                color=METHOD_COLORS.get(method, "grey"),
                marker="o" if task_name == "Test Oracle" else "s",
                s=80, alpha=0.85, edgecolors="white", linewidths=0.5,
                zorder=3,
            )
            label = f"{METHOD_LABELS.get(method, method)}/{row['reasoning']}\n({task_name})"
            ax.annotate(label, (row["avg_noise_rate"], row["avg_faithfulness"]),
                        textcoords="offset points", xytext=(6, 4), fontsize=7, color="grey")

    if not has_data:
        print("  noise_vs_faithfulness.png SKIPPED — no noise_rate + faithfulness data")
        plt.close(fig)
        return

    # Threshold line
    ax.axvline(0.3, color="red", linestyle="--", linewidth=1, alpha=0.7,
               label="Noise threshold (0.3)")
    ax.set_xlabel("Avg Noise Rate (fraction irrelevant chunks)", fontsize=11)
    ax.set_ylabel("Faithfulness (token-overlap, unified)", fontsize=11)
    ax.set_xlim(-0.05, 1.05)
    ax.set_ylim(-0.05, 1.05)
    ax.set_title("Noise Rate vs Faithfulness\n(RQ2: retrieval hurts when corpus mismatched)",
                 fontsize=12, fontweight="bold")
    ax.grid(alpha=0.3)

    # Method legend
    method_patches = [
        mpatches.Patch(color=METHOD_COLORS[m], label=METHOD_LABELS[m])
        for m in METHODS if m in METHOD_COLORS
    ]
    shape_patches = [
        plt.Line2D([0], [0], marker="o", color="w", markerfacecolor="grey",
                   markersize=8, label="Test Oracle"),
        plt.Line2D([0], [0], marker="s", color="w", markerfacecolor="grey",
                   markersize=8, label="Docstring"),
    ]
    ax.legend(handles=method_patches + shape_patches, fontsize=9, loc="upper right")

    plt.tight_layout()
    fig.savefig(OUTPUT_DIR / "noise_vs_faithfulness.png", dpi=150)
    plt.close(fig)
    print("  noise_vs_faithfulness.png")


# ---------------------------------------------------------------------------
# Chart 3: Cost vs quality Pareto (RQ4 efficiency frontier)
# ---------------------------------------------------------------------------

def plot_pareto(task_dfs: dict) -> None:
    fig, ax = plt.subplots(figsize=(9, 6))
    has_data = False

    TASK_MARKERS = {"Test Oracle": "o", "Docstring": "s"}

    for task_name, df in task_dfs.items():
        # Cost = retrieval + llm time per sample
        if "avg_llm_secs" not in df.columns:
            continue

        df = df.copy()
        ret = df.get("avg_retrieval_secs", pd.Series(0.0, index=df.index)).fillna(0)
        llm = df["avg_llm_secs"].fillna(0)
        df["total_secs"] = ret + llm

        score_col = "val_score" if "val_score" in df.columns else None
        if score_col is None:
            continue

        plot_df = df[["method_name", "reasoning", "total_secs", score_col]].dropna()
        if plot_df.empty:
            continue

        has_data = True
        for _, row in plot_df.iterrows():
            method = row["method_name"]
            ax.scatter(
                row["total_secs"], row[score_col],
                color=METHOD_COLORS.get(method, "grey"),
                marker=TASK_MARKERS.get(task_name, "o"),
                s=90, alpha=0.85, edgecolors="white", linewidths=0.5, zorder=3,
            )
            label = f"{METHOD_LABELS.get(method, method)}/{row['reasoning']}"
            ax.annotate(label, (row["total_secs"], row[score_col]),
                        textcoords="offset points", xytext=(5, 4), fontsize=7, color="grey")

    if not has_data:
        print("  pareto.png SKIPPED — no cost + quality data")
        plt.close(fig)
        return

    ax.set_xlabel("Avg total time per sample (retrieval + LLM, seconds)", fontsize=11)
    ax.set_ylabel("Quality score (val_score / rouge_1_f1)", fontsize=11)
    ax.set_title("Cost vs Quality Pareto Frontier\n(RQ4: cost-faithfulness trade-off)",
                 fontsize=12, fontweight="bold")
    ax.grid(alpha=0.3)

    method_patches = [
        mpatches.Patch(color=METHOD_COLORS[m], label=METHOD_LABELS[m])
        for m in METHODS if m in METHOD_COLORS
    ]
    shape_patches = [
        plt.Line2D([0], [0], marker="o", color="w", markerfacecolor="grey",
                   markersize=8, label="Test Oracle"),
        plt.Line2D([0], [0], marker="s", color="w", markerfacecolor="grey",
                   markersize=8, label="Docstring"),
    ]
    ax.legend(handles=method_patches + shape_patches, fontsize=9)

    plt.tight_layout()
    fig.savefig(OUTPUT_DIR / "pareto.png", dpi=150)
    plt.close(fig)
    print("  pareto.png")


# ---------------------------------------------------------------------------
# Summary table (TSV for thesis appendix)
# ---------------------------------------------------------------------------

def save_summary_table(task_dfs: dict) -> None:
    rows = []
    for task_name, df in task_dfs.items():
        best = _best_per_method(df)
        if best.empty:
            continue
        for _, row in best.iterrows():
            method = row.get("method_name", "")
            rows.append({
                "task":            task_name,
                "method":          METHOD_LABELS.get(method, method),
                "val_score":       round(float(row.get("val_score", float("nan")) or float("nan")), 4),
                "faithfulness":    round(float(row.get("avg_faithfulness", float("nan")) or float("nan")), 4),
                "noise_rate":      round(float(row.get("avg_noise_rate", float("nan")) or float("nan")), 4),
                "total_secs":      round(
                    float(row.get("avg_retrieval_secs", 0) or 0) +
                    float(row.get("avg_llm_secs", 0) or 0), 3
                ),
                "rouge":           round(float(row.get("avg_rouge", float("nan")) or float("nan")), 4),
            })

    if not rows:
        return
    out = pd.DataFrame(rows)
    out.to_csv(OUTPUT_DIR / "summary_table.tsv", sep="\t", index=False, float_format="%.4f")
    print("  summary_table.tsv")
    print()
    print(out.to_string(index=False))


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    task_dfs = {}
    print("Loading task results:")
    for task_name, path in TASK_FILES.items():
        df = load_task(path, task_name)
        if df is not None:
            task_dfs[task_name] = df

    if not task_dfs:
        print("\nNo task data found. Run experiments first:")
        print("  python extract_docstring_results.py  # for Docstring task")
        print("  (run unit test experiments)          # for Test Oracle task")
        sys.exit(0)

    OUTPUT_DIR.mkdir(exist_ok=True)
    print(f"\nGenerating charts in {OUTPUT_DIR}/")

    plot_faithfulness_by_task(task_dfs)
    plot_noise_vs_faithfulness(task_dfs)
    plot_pareto(task_dfs)
    save_summary_table(task_dfs)

    print(f"\nDone. Open {OUTPUT_DIR}/ to view charts.")


if __name__ == "__main__":
    main()
