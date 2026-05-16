"""
mutation_statistical_tests.py — Statistical significance for mutation kill rates.

Same methodology as statistical_tests.py (used for val_score) but for the
SE-relevant mutation kill rate metric.  Operates on per-sample kill rates
loaded from .checkpoints_mutation_analysis/{key}.pkl (one per
(method, reasoning, model) cell, saved by mutation_testing.run_mutation_analysis).

Tests run, for each model and pooled across models:
    - Kruskal-Wallis H-test across the 4 methods (omnibus)
    - Mann-Whitney U pairwise (6 pairs) with Bonferroni correction
    - Cohen's d effect size for each pair

Output (plots_mutation/):
    mutation_statistical_report.txt — full significance tables for thesis
"""

import argparse
import itertools
import math
import pickle
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import stats

# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------

# mutation_testing.py writes to .checkpoints_mutation_analysis/ (dot prefix).
# When pulled from Drive locally the same files often land under
# checkpoints_mutation_analysis/ (no dot). Try both, prefer the runtime path.
ANALYSIS_CKPT_CANDIDATES = [
    Path(".checkpoints_mutation_analysis"),
    Path("checkpoints_mutation_analysis"),
]
RESULTS_FILE_CANDIDATES = [
    Path("results_mutation.tsv"),         # written by mutation_testing.py
    Path("results/results_mutation.tsv"), # Drive-synced copy
]
OUTPUT_DIR        = Path("plots_mutation")
REPORT_FILE       = OUTPUT_DIR / "mutation_statistical_report.txt"


def _resolve_analysis_dir() -> Path:
    for cand in ANALYSIS_CKPT_CANDIDATES:
        if cand.is_dir():
            return cand
    return ANALYSIS_CKPT_CANDIDATES[0]   # for error message

METHODS = ["plain_llm", "random_rag", "simple_rag", "iterative_critique"]
METHOD_LABELS = {
    "plain_llm":          "Plain LLM",
    "random_rag":         "Random RAG",
    "simple_rag":         "Simple RAG",
    "iterative_critique": "Iterative Critique",
}

ALPHA           = 0.05
N_PAIRS         = len(list(itertools.combinations(METHODS, 2)))   # 6
BONFERRONI_A    = ALPHA / N_PAIRS                                  # 0.00833...


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def cohens_d(a: list, b: list) -> float:
    """Pooled Cohen's d. NaN if either group has <2 samples or zero pooled SD."""
    if len(a) < 2 or len(b) < 2:
        return float("nan")
    mean_a, mean_b = np.mean(a), np.mean(b)
    var_a, var_b   = np.var(a, ddof=1), np.var(b, ddof=1)
    pooled_sd = math.sqrt(
        ((len(a) - 1) * var_a + (len(b) - 1) * var_b) / (len(a) + len(b) - 2)
    )
    if pooled_sd == 0:
        return float("nan")
    return (mean_a - mean_b) / pooled_sd


def effect_label(d: float) -> str:
    """Cohen (1988) magnitude labels."""
    if math.isnan(d):
        return "N/A"
    a = abs(d)
    if a < 0.2:
        return "negligible"
    if a < 0.5:
        return "small"
    if a < 0.8:
        return "medium"
    return "large"


def parse_key(key: str) -> tuple:
    """
    Parse a checkpoint filename stem like 'iterative_critique_base_qwen3.5_9b'
    into (method, reasoning, model). The method may contain underscores, so we
    match against the known method list.
    """
    for m in sorted(METHODS, key=len, reverse=True):  # longest first
        if key.startswith(m + "_"):
            rest = key[len(m) + 1:]
            # next token is reasoning, the rest is the model (with _ → .: where needed)
            parts = rest.split("_", 1)
            reasoning = parts[0]
            model = parts[1] if len(parts) > 1 else ""
            # The model key was written with ':' → '_'. We can't perfectly invert
            # without a model list, but we don't need the exact form for stats —
            # we just need stable group labels.
            return m, reasoning, model
    return key, "base", ""


# ---------------------------------------------------------------------------
# Data loading
# ---------------------------------------------------------------------------

def load_per_sample_kill_rates() -> pd.DataFrame:
    """
    Load every analysis checkpoint and return a long-format DataFrame:
        columns = [method, reasoning, model, sample_idx, kill_rate,
                   total_mutants, killed, equivalent]
    Skips .tmp files and rows with NaN kill_rate (those samples had all tests
    fail on the original function and got filtered out).
    """
    analysis_dir = _resolve_analysis_dir()
    if not analysis_dir.is_dir():
        print(f"ERROR: none of {[str(p) for p in ANALYSIS_CKPT_CANDIDATES]} "
              f"exist. Nothing to test.")
        sys.exit(1)

    rows = []
    files = sorted(analysis_dir.glob("*.pkl"))
    for f in files:
        if f.name.endswith(".tmp") or ".pkl.tmp" in f.name:
            continue
        with open(f, "rb") as fp:
            try:
                data = pickle.load(fp)
            except Exception as e:
                print(f"  WARNING: could not load {f.name}: {e}")
                continue
        if not isinstance(data, dict):
            print(f"  WARNING: unexpected format in {f.name}")
            continue

        method, reasoning, model = parse_key(f.stem)
        for sample_idx, result in data.items():
            kr = result.get("kill_rate", float("nan"))
            if isinstance(kr, float) and math.isnan(kr):
                continue
            rows.append({
                "method":        method,
                "reasoning":     reasoning,
                "model":         model,
                "sample_idx":    sample_idx,
                "kill_rate":     float(kr),
                "total_mutants": result.get("total_mutants", 0),
                "killed":        result.get("killed", 0),
                "equivalent":    result.get("equivalent", 0),
            })

    if not rows:
        print("ERROR: no per-sample kill rates extracted.")
        sys.exit(1)

    df = pd.DataFrame(rows)
    return df


def load_tsv_means() -> pd.DataFrame:
    """Load aggregated TSV (used for descriptive stats incl. llama3.2/phi4).

    Prefer whichever candidate has the 'model' column — the repo may still
    hold an old pre-5710135 results_mutation.tsv (no model column) alongside
    a fresh Drive-synced copy.
    """
    best = pd.DataFrame()
    for cand in RESULTS_FILE_CANDIDATES:
        if not cand.exists():
            continue
        try:
            df = pd.read_csv(cand, sep="\t")
        except Exception:
            continue
        if "model" in df.columns:
            return df
        if best.empty:
            best = df  # fallback if no candidate has model column
    return best


# ---------------------------------------------------------------------------
# Core tests
# ---------------------------------------------------------------------------

def kruskal_across_methods(df: pd.DataFrame) -> dict:
    """Kruskal-Wallis on the 4 methods within this slice of df."""
    groups = [df[df["method"] == m]["kill_rate"].tolist() for m in METHODS]
    non_empty = [g for g in groups if len(g) >= 2]
    if len(non_empty) < 2:
        return {"H": float("nan"), "p": float("nan"), "k": len(non_empty)}
    H, p = stats.kruskal(*non_empty)
    return {"H": float(H), "p": float(p), "k": len(non_empty)}


def pairwise_mannwhitney(df: pd.DataFrame) -> pd.DataFrame:
    """All 6 pairwise Mann-Whitney U tests; Bonferroni correction; Cohen's d."""
    out = []
    for a, b in itertools.combinations(METHODS, 2):
        va = df[df["method"] == a]["kill_rate"].tolist()
        vb = df[df["method"] == b]["kill_rate"].tolist()
        if len(va) < 2 or len(vb) < 2:
            out.append({"a": a, "b": b, "n_a": len(va), "n_b": len(vb),
                        "U": float("nan"), "p_raw": float("nan"),
                        "p_adj": float("nan"), "d": float("nan"),
                        "effect": "N/A", "sig": False})
            continue
        U, p_raw = stats.mannwhitneyu(va, vb, alternative="two-sided")
        p_adj = min(p_raw * N_PAIRS, 1.0)
        d = cohens_d(va, vb)
        out.append({
            "a":      a,
            "b":      b,
            "n_a":    len(va),
            "n_b":    len(vb),
            "U":      float(U),
            "p_raw":  float(p_raw),
            "p_adj":  float(p_adj),
            "d":      float(d) if not math.isnan(d) else float("nan"),
            "effect": effect_label(d),
            "sig":    bool(p_adj < ALPHA),
        })
    return pd.DataFrame(out)


# ---------------------------------------------------------------------------
# Report writer
# ---------------------------------------------------------------------------

def _h(title: str) -> str:
    return "=" * 78 + "\n" + f" {title}\n" + "=" * 78

def _hline() -> str:
    return "-" * 78

def fmt_p(p: float) -> str:
    if math.isnan(p):
        return "  nan"
    if p < 0.001:
        return "<0.001"
    return f"{p:.4f}"

def fmt_d(d: float) -> str:
    if math.isnan(d):
        return "  nan"
    return f"{d:+.3f}"


def write_report(per_sample: pd.DataFrame, tsv: pd.DataFrame) -> str:
    lines = []
    lines.append(_h("MUTATION KILL RATE — STATISTICAL SIGNIFICANCE REPORT"))
    lines.append("")
    lines.append(f"Per-sample data source: {_resolve_analysis_dir()}/*.pkl")
    lines.append(f"Total per-sample observations: {len(per_sample)} "
                 f"(NaN kill rates dropped)")
    models = sorted(per_sample["model"].unique())
    lines.append(f"Models with per-sample data: {', '.join(models)}")
    lines.append(f"Methods analyzed: {', '.join(METHODS)}")
    lines.append(f"Multiple-comparison correction: Bonferroni, "
                 f"α = {ALPHA}/{N_PAIRS} = {BONFERRONI_A:.4f}")
    lines.append("")

    # ---- Descriptive table from per-sample data ---------------------------
    lines.append(_h("1. PER-SAMPLE DESCRIPTIVES"))
    lines.append(f"{'method':<22}{'model':<22}{'n':>4}  {'mean':>7}  {'std':>7}  {'median':>7}")
    lines.append(_hline())
    for m in models:
        for method in METHODS:
            slice_ = per_sample[(per_sample["model"] == m) &
                                (per_sample["method"] == method)]
            if slice_.empty:
                continue
            vals = slice_["kill_rate"].values
            lines.append(
                f"  {METHOD_LABELS[method]:<20}"
                f"{m:<22}{len(vals):>4}  "
                f"{np.mean(vals):>7.4f}  {np.std(vals, ddof=1):>7.4f}  "
                f"{np.median(vals):>7.4f}"
            )
        lines.append("")

    # Pooled descriptives across models
    lines.append("Pooled across models:")
    lines.append(f"  {'method':<22}{'n':>4}  {'mean':>7}  {'std':>7}  {'median':>7}")
    for method in METHODS:
        vals = per_sample[per_sample["method"] == method]["kill_rate"].values
        if len(vals) == 0:
            continue
        lines.append(
            f"  {METHOD_LABELS[method]:<22}{len(vals):>4}  "
            f"{np.mean(vals):>7.4f}  {np.std(vals, ddof=1):>7.4f}  "
            f"{np.median(vals):>7.4f}"
        )
    lines.append("")

    # ---- Kruskal-Wallis per model -----------------------------------------
    lines.append(_h("2. KRUSKAL-WALLIS H-TEST (omnibus across 4 methods)"))
    lines.append(f"  H₀: kill_rate distribution is the same across methods.")
    lines.append("")
    lines.append(f"  {'scope':<26}{'k':>4}  {'H':>10}  {'p':>10}  {'reject H₀?':>12}")
    lines.append(_hline())
    for m in models:
        slice_ = per_sample[per_sample["model"] == m]
        kw = kruskal_across_methods(slice_)
        verdict = "yes" if (not math.isnan(kw["p"]) and kw["p"] < ALPHA) else "no"
        lines.append(
            f"  {('model = ' + m):<26}{kw['k']:>4}  "
            f"{kw['H']:>10.4f}  {fmt_p(kw['p']):>10}  {verdict:>12}"
        )
    # Pooled
    kw_pooled = kruskal_across_methods(per_sample)
    verdict = "yes" if (not math.isnan(kw_pooled["p"]) and
                        kw_pooled["p"] < ALPHA) else "no"
    lines.append(_hline())
    lines.append(
        f"  {'pooled (all models)':<26}{kw_pooled['k']:>4}  "
        f"{kw_pooled['H']:>10.4f}  {fmt_p(kw_pooled['p']):>10}  {verdict:>12}"
    )
    lines.append("")

    # ---- Mann-Whitney pairwise per model ----------------------------------
    lines.append(_h("3. PAIRWISE MANN-WHITNEY U (6 pairs, Bonferroni-corrected)"))
    lines.append(f"  H₀: kill_rate(method_a) == kill_rate(method_b).")
    lines.append(f"  Significant at α_adj = {BONFERRONI_A:.4f} marked with *.")
    lines.append("")

    def _write_table(scope_label: str, sub: pd.DataFrame):
        lines.append(f"  [{scope_label}]")
        lines.append(f"    {'method_a vs method_b':<42}"
                     f"{'n_a':>5} {'n_b':>5} {'U':>9} {'p_raw':>9} {'p_adj':>9} "
                     f"{'d':>8} {'effect':>11} sig")
        lines.append("    " + "-" * 116)
        df = pairwise_mannwhitney(sub)
        for _, r in df.iterrows():
            pair = f"{METHOD_LABELS[r['a']]} vs {METHOD_LABELS[r['b']]}"
            sig_mark = "*" if r["sig"] else " "
            lines.append(
                f"    {pair:<42}{r['n_a']:>5} {r['n_b']:>5} "
                f"{r['U']:>9.1f} {fmt_p(r['p_raw']):>9} {fmt_p(r['p_adj']):>9} "
                f"{fmt_d(r['d']):>8} {r['effect']:>11} {sig_mark}"
            )
        lines.append("")

    for m in models:
        _write_table(f"model = {m}", per_sample[per_sample["model"] == m])

    _write_table("pooled (all models)", per_sample)

    # ---- Aggregated TSV (covers llama3.2 + phi4 which lack per-sample) ----
    lines.append(_h("4. AGGREGATED MEANS FROM results_mutation.tsv"))
    lines.append("  (Per-sample stats above use .checkpoints_mutation_analysis/.")
    lines.append("   For llama3.2 and phi4, only the aggregated mean is available")
    lines.append("   because those models were analysed before per-sample resume")
    lines.append("   was added. Re-running `python mutation_testing.py` without")
    lines.append("   --regenerate will populate per-sample data for them.)")
    lines.append("")
    if not tsv.empty and "model" in tsv.columns:
        cols = ["method", "reasoning", "model", "mean_kill_rate",
                "std_kill_rate", "n_samples_valid", "total_killed",
                "total_mutants"]
        cols = [c for c in cols if c in tsv.columns]
        sort_cols = [c for c in ("model", "method") if c in tsv.columns]
        sub = tsv[cols].sort_values(sort_cols).copy()
        text = sub.to_string(index=False)
        for line in text.splitlines():
            lines.append("  " + line)
    else:
        lines.append("  (results_mutation.tsv not found or missing 'model' column)")
    lines.append("")

    # ---- Footnotes --------------------------------------------------------
    lines.append(_h("INTERPRETATION GUIDE"))
    lines.append("  Cohen's d magnitudes:")
    lines.append("    |d| < 0.2 negligible    0.2-0.5 small    0.5-0.8 medium    ≥0.8 large")
    lines.append("  Sign convention: d > 0 means method_a > method_b (higher kill rate).")
    lines.append("  Significance: p_adj < α (=0.05) after Bonferroni over 6 pairwise tests.")
    lines.append("")

    return "\n".join(lines)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", default=str(REPORT_FILE),
                        help="report output path")
    args = parser.parse_args()

    OUTPUT_DIR.mkdir(exist_ok=True)

    print("Loading per-sample kill rates...")
    per_sample = load_per_sample_kill_rates()
    print(f"  loaded {len(per_sample)} per-sample observations")
    print(f"  models: {sorted(per_sample['model'].unique())}")
    print(f"  methods: {sorted(per_sample['method'].unique())}")

    tsv = load_tsv_means()

    report = write_report(per_sample, tsv)
    out_path = Path(args.out)
    out_path.parent.mkdir(exist_ok=True)
    out_path.write_text(report)
    print(f"\nReport written → {out_path}")

    # Echo the headline result
    print("\n" + "=" * 60)
    print("HEADLINE RESULTS (pooled across models)")
    print("=" * 60)
    kw = kruskal_across_methods(per_sample)
    verdict = "REJECT H₀" if (not math.isnan(kw["p"]) and kw["p"] < ALPHA) \
        else "FAIL TO REJECT"
    print(f"Kruskal-Wallis:  H={kw['H']:.4f}  p={fmt_p(kw['p'])}   ({verdict})")
    print()
    print("Significant pairwise differences (Bonferroni-adjusted p<0.05):")
    pw = pairwise_mannwhitney(per_sample)
    sig = pw[pw["sig"]]
    if sig.empty:
        print("  (none)")
    else:
        for _, r in sig.iterrows():
            print(f"  {METHOD_LABELS[r['a']]:<20} vs {METHOD_LABELS[r['b']]:<20} "
                  f"p_adj={fmt_p(r['p_adj'])}  d={fmt_d(r['d'])} ({r['effect']})")

    return 0


if __name__ == "__main__":
    sys.exit(main())
