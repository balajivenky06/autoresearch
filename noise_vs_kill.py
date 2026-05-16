"""
noise_vs_kill.py — RAG retrieval quality vs mutation kill rate correlation.

Joins results_unitest.tsv (RAG-quality metrics) with results_mutation.tsv
(per-(method, model) mean kill rates) on (method, model, reasoning='base'),
then asks: does retrieval quality predict defect-detection capability?

Three retrieval-quality columns are tested:

  avg_noise_rate              fraction of retrieved chunks with
                               cosine sim < 0.3 (RAG signal-to-noise).
                               In our setup this is identically 0.0 —
                               KB threshold never fires. Reported for
                               transparency but the correlation is
                               mechanically undefined.

  avg_faithfulness            token overlap between generated tests
                               and retrieved context (RQ3, automated).

  avg_llm_judge_faithfulness  DeepSeek-Coder 6.7B judge of how grounded
                               the tests are in retrieved context
                               (validated Pearson r=0.925 vs human in
                               the docstring companion paper).

For each available metric we compute Pearson r + Spearman ρ both pooled
(all RAG cells) and per-method, and plot a scatter against
mean_kill_rate.

Output (plots_mutation/):
    noise_vs_kill_report.txt
    noise_vs_kill_scatter.png
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

OUTPUT_DIR  = Path("plots_mutation")
REPORT_FILE = OUTPUT_DIR / "noise_vs_kill_report.txt"
SCATTER_PNG = OUTPUT_DIR / "noise_vs_kill_scatter.png"

UNITEST_CANDIDATES  = [Path("results_unitest.tsv"),
                       Path("results/results_unitest.tsv")]
MUTATION_CANDIDATES = [Path("results_mutation.tsv"),
                       Path("results/results_mutation.tsv")]

# Mapping the mutation TSV's pretty labels back to the raw method names
# used in results_unitest.tsv.
PRETTY_TO_RAW = {
    "Plain LLM":          "plain_llm",
    "Random RAG":         "random_rag",
    "Simple RAG":         "simple_rag",
    "Iterative Critique": "iterative_critique",
}
RAG_METHODS = ["random_rag", "simple_rag", "iterative_critique"]
METHOD_COLORS = {
    "random_rag":         "#8172B2",
    "simple_rag":         "#DD8452",
    "iterative_critique": "#55A868",
}
METHOD_LABEL = {
    "random_rag":         "Random RAG",
    "simple_rag":         "Simple RAG",
    "iterative_critique": "Iterative Critique",
}

QUALITY_COLS = [
    ("avg_noise_rate",             "Noise rate (cosine < 0.3 fraction)"),
    ("avg_faithfulness",           "Faithfulness (token overlap)"),
    ("avg_llm_judge_faithfulness", "Faithfulness (DeepSeek judge)"),
]


# ---------------------------------------------------------------------------
# I/O helpers
# ---------------------------------------------------------------------------

def _first_existing(paths: list[Path]) -> Path | None:
    for p in paths:
        if p.exists():
            return p
    return None


def load_unitest() -> pd.DataFrame:
    p = _first_existing(UNITEST_CANDIDATES)
    if p is None:
        sys.exit("ERROR: results_unitest.tsv not found")
    df = pd.read_csv(p, sep="\t")
    # The unitest TSV uses 'method/reasoning' in a single column.
    if df["method"].astype(str).str.contains("/").any():
        df[["method_raw", "reasoning"]] = df["method"].str.split("/", n=1, expand=True)
    else:
        df["method_raw"] = df["method"]
        df["reasoning"]  = "base"
    return df


def load_mutation() -> pd.DataFrame:
    """Prefer whichever TSV has the model column AND all 4 models populated."""
    best = pd.DataFrame()
    for p in MUTATION_CANDIDATES:
        if not p.exists():
            continue
        df = pd.read_csv(p, sep="\t")
        if "model" not in df.columns:
            continue
        if best.empty or df["model"].nunique() > best["model"].nunique():
            best = df
    if best.empty:
        sys.exit("ERROR: no results_mutation.tsv with a 'model' column found")
    # Normalise method to raw form so we can join on it.
    best["method_raw"] = best["method"].map(PRETTY_TO_RAW).fillna(best["method"])
    if "reasoning" not in best.columns:
        best["reasoning"] = "base"
    return best


def join(unitest: pd.DataFrame, mutation: pd.DataFrame) -> pd.DataFrame:
    """Inner-join on (method_raw, reasoning, model) for the RAG methods only."""
    keep_u = ["method_raw", "reasoning", "model",
              "avg_noise_rate", "avg_faithfulness",
              "avg_llm_judge_faithfulness", "val_score"]
    keep_u = [c for c in keep_u if c in unitest.columns]
    sub_u = unitest[keep_u].copy()

    keep_m = ["method_raw", "reasoning", "model",
              "mean_kill_rate", "n_samples_valid", "total_mutants",
              "total_killed"]
    keep_m = [c for c in keep_m if c in mutation.columns]
    sub_m = mutation[keep_m].copy()

    merged = sub_u.merge(sub_m, on=["method_raw", "reasoning", "model"],
                         how="inner")
    merged = merged[merged["method_raw"].isin(RAG_METHODS)].copy()
    merged = merged[merged["reasoning"] == "base"].copy()
    return merged


# ---------------------------------------------------------------------------
# Correlation helpers
# ---------------------------------------------------------------------------

def correlate(x: np.ndarray, y: np.ndarray) -> dict:
    """Return Pearson r + Spearman ρ with p-values. NaN if insufficient data."""
    if len(x) < 3 or len(y) < 3:
        return {"n": len(x), "r": float("nan"), "r_p": float("nan"),
                "rho": float("nan"), "rho_p": float("nan")}
    if np.nanstd(x) == 0 or np.nanstd(y) == 0:
        return {"n": len(x), "r": float("nan"), "r_p": float("nan"),
                "rho": float("nan"), "rho_p": float("nan"),
                "note": "constant input — correlation undefined"}
    try:
        r,   r_p   = stats.pearsonr(x, y)
        rho, rho_p = stats.spearmanr(x, y)
        return {"n": len(x), "r": float(r), "r_p": float(r_p),
                "rho": float(rho), "rho_p": float(rho_p)}
    except Exception as e:
        return {"n": len(x), "r": float("nan"), "r_p": float("nan"),
                "rho": float("nan"), "rho_p": float("nan"),
                "note": f"error: {e}"}


def fmt_p(p: float) -> str:
    if p is None or math.isnan(p):
        return "  nan"
    if p < 0.001:
        return "<0.001"
    return f"{p:.4f}"


def fmt_r(r: float) -> str:
    if math.isnan(r):
        return "  nan"
    return f"{r:+.4f}"


# ---------------------------------------------------------------------------
# Reporting
# ---------------------------------------------------------------------------

def _h(title: str) -> str:
    return "=" * 78 + "\n " + title + "\n" + "=" * 78


def write_report(joined: pd.DataFrame) -> str:
    lines = []
    lines.append(_h("RAG QUALITY vs MUTATION KILL RATE — CORRELATION REPORT"))
    lines.append("")
    lines.append(f"Cells joined (method × model × base reasoning): {len(joined)}")
    lines.append(f"RAG methods: {sorted(joined['method_raw'].unique())}")
    lines.append(f"Models      : {sorted(joined['model'].unique())}")
    lines.append("")

    # ---- Descriptive table ------------------------------------------------
    lines.append(_h("1. JOINED CELLS  (method × model)"))
    cols = ["method_raw", "model", "mean_kill_rate", "avg_noise_rate",
            "avg_faithfulness", "avg_llm_judge_faithfulness"]
    cols = [c for c in cols if c in joined.columns]
    text = joined[cols].sort_values(["method_raw", "model"]).to_string(index=False)
    for line in text.splitlines():
        lines.append("  " + line)
    lines.append("")

    # ---- Correlation table ------------------------------------------------
    lines.append(_h("2. CORRELATIONS  (predictor → mean_kill_rate)"))
    lines.append("  H₀: ρ = 0 (no monotonic association)")
    lines.append("  Note: n is small (≤12 cells). |r|≳0.6 needed for p<0.05 at n=10.")
    lines.append("")
    header = (f"  {'predictor':<35}{'scope':<22}{'n':>4}"
              f"{'Pearson r':>12}{'p_r':>10}{'Spearman ρ':>14}{'p_ρ':>10}")
    lines.append(header)
    lines.append("-" * 78)

    for col, _label in QUALITY_COLS:
        if col not in joined.columns:
            lines.append(f"  {col:<35}{'(missing in TSV)':<22}")
            continue

        # Pooled across all RAG methods
        sub = joined.dropna(subset=[col, "mean_kill_rate"])
        c = correlate(sub[col].values, sub["mean_kill_rate"].values)
        note = c.get("note", "")
        lines.append(
            f"  {col:<35}{'all RAG methods':<22}{c['n']:>4}"
            f"{fmt_r(c['r']):>12}{fmt_p(c['r_p']):>10}"
            f"{fmt_r(c['rho']):>14}{fmt_p(c['rho_p']):>10}"
            f"  {note}"
        )

        for m in RAG_METHODS:
            sub_m = joined[(joined["method_raw"] == m)].dropna(
                subset=[col, "mean_kill_rate"])
            cm = correlate(sub_m[col].values, sub_m["mean_kill_rate"].values)
            note = cm.get("note", "")
            lines.append(
                f"  {'':<35}{m:<22}{cm['n']:>4}"
                f"{fmt_r(cm['r']):>12}{fmt_p(cm['r_p']):>10}"
                f"{fmt_r(cm['rho']):>14}{fmt_p(cm['rho_p']):>10}"
                f"  {note}"
            )
        lines.append("")

    # ---- Interpretation ---------------------------------------------------
    lines.append(_h("INTERPRETATION"))
    if "avg_noise_rate" in joined.columns:
        nr = joined["avg_noise_rate"].dropna()
        if len(nr) > 0 and nr.nunique() == 1 and nr.iloc[0] == 0.0:
            lines.append(
                "  avg_noise_rate is identically 0.0 across every RAG cell.")
            lines.append(
                "  Our knowledge base never returns chunks below the cosine<0.3")
            lines.append(
                "  threshold, so the noise-rate signal is degenerate. This is a")
            lines.append(
                "  finding in itself — the testing-docs KB is well-curated")
            lines.append(
                "  relative to the per-task queries — but it means noise rate")
            lines.append(
                "  cannot be correlated with anything in this dataset.")
    lines.append("")
    lines.append("  avg_faithfulness measures token overlap between the generated")
    lines.append("  tests and the retrieved context (RQ3). A positive correlation")
    lines.append("  with mean_kill_rate would say: tests that lift more vocabulary")
    lines.append("  from the retrieved docs are also better at killing mutants.")
    lines.append("")
    lines.append("  avg_llm_judge_faithfulness is the DeepSeek-Coder 6.7B judgement")
    lines.append("  of how grounded the tests are in the retrieved context.")
    lines.append("  Validated Pearson r=0.925 vs human in the docstring companion")
    lines.append("  paper, so it's a higher-fidelity grounding signal than the")
    lines.append("  token-overlap proxy.")
    lines.append("")

    return "\n".join(lines)


# ---------------------------------------------------------------------------
# Plot
# ---------------------------------------------------------------------------

def write_scatter(joined: pd.DataFrame, out_path: Path) -> None:
    """3-panel scatter: one panel per RAG-quality column."""
    panels = [(c, l) for c, l in QUALITY_COLS if c in joined.columns]
    if not panels:
        return

    fig, axes = plt.subplots(1, len(panels), figsize=(5.5 * len(panels), 5.2))
    if len(panels) == 1:
        axes = [axes]

    for ax, (col, title) in zip(axes, panels):
        sub = joined.dropna(subset=[col, "mean_kill_rate"])
        for m in RAG_METHODS:
            d = sub[sub["method_raw"] == m]
            if d.empty:
                continue
            ax.scatter(d[col], d["mean_kill_rate"],
                       c=METHOD_COLORS[m], s=90, alpha=0.85,
                       edgecolor="black", linewidth=0.6,
                       label=METHOD_LABEL[m])
            for _, row in d.iterrows():
                ax.annotate(row["model"].split(":")[0],
                            (row[col], row["mean_kill_rate"]),
                            xytext=(4, 4), textcoords="offset points",
                            fontsize=7, alpha=0.7)

        # Pooled trend line if it's well-defined
        if not sub.empty and sub[col].nunique() > 1:
            x = sub[col].values.astype(float)
            y = sub["mean_kill_rate"].values.astype(float)
            try:
                slope, intercept = np.polyfit(x, y, 1)
                xs = np.linspace(x.min(), x.max(), 50)
                ax.plot(xs, slope * xs + intercept,
                        color="gray", linestyle="--", linewidth=1,
                        label="OLS fit (pooled)")
            except Exception:
                pass

        c = correlate(sub[col].values, sub["mean_kill_rate"].values) \
            if not sub.empty else {}
        sub_t = ""
        if c.get("r") is not None and not math.isnan(c.get("r", float("nan"))):
            sub_t = f"  r={c['r']:+.3f} (p={fmt_p(c['r_p'])}, n={c['n']})"
        ax.set_title(title + sub_t, fontsize=11)
        ax.set_xlabel(col, fontsize=10)
        ax.set_ylabel("Mean mutation kill rate", fontsize=10)
        ax.grid(alpha=0.25)

    axes[-1].legend(loc="lower right", fontsize=8)
    fig.suptitle("RAG retrieval quality vs mutation kill rate "
                 "(base reasoning, all RAG methods)",
                 fontsize=13, fontweight="bold")
    fig.tight_layout(rect=(0, 0, 1, 0.95))
    fig.savefig(out_path, dpi=150)
    plt.close(fig)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> int:
    OUTPUT_DIR.mkdir(exist_ok=True)

    unitest  = load_unitest()
    mutation = load_mutation()
    joined   = join(unitest, mutation)

    print(f"Joined {len(joined)} cells (RAG methods × models, base reasoning)")
    cols_have = [c for c, _ in QUALITY_COLS if c in joined.columns]
    print(f"Available quality columns: {cols_have}")
    print()

    report = write_report(joined)
    REPORT_FILE.write_text(report)
    print(f"Report  → {REPORT_FILE}")

    write_scatter(joined, SCATTER_PNG)
    print(f"Scatter → {SCATTER_PNG}")

    # Headline echo
    print()
    print("=" * 60)
    print("HEADLINE — best correlation across the 3 RAG-quality signals")
    print("=" * 60)
    best = None
    for col, label in QUALITY_COLS:
        if col not in joined.columns:
            continue
        sub = joined.dropna(subset=[col, "mean_kill_rate"])
        c = correlate(sub[col].values, sub["mean_kill_rate"].values)
        if math.isnan(c.get("r", float("nan"))):
            print(f"  {col:<32}  r=nan ({c.get('note', 'insufficient data')})")
            continue
        sig = " *" if (not math.isnan(c["r_p"]) and c["r_p"] < 0.05) else ""
        print(f"  {col:<32}  r={c['r']:+.3f}  p={fmt_p(c['r_p'])}  n={c['n']}{sig}")
        if best is None or (not math.isnan(c["r_p"]) and
                            c["r_p"] < best.get("r_p", 1.0)):
            best = {**c, "col": col}
    if best is not None and not math.isnan(best["r_p"]):
        verdict = "REJECT H₀ (correlation is real)" if best["r_p"] < 0.05 \
            else "FAIL TO REJECT"
        print()
        print(f"  Strongest signal: {best['col']}  r={best['r']:+.3f}  "
              f"p={fmt_p(best['r_p'])}  → {verdict}")

    return 0


if __name__ == "__main__":
    sys.exit(main())
