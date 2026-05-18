"""
human_eval_pair_sampler.py — Build the actual (function, generated_tests)
worksheet for human annotation, drawing from the mutation-testing
generation checkpoints.

The existing human_eval_sampler.py picks rows from results_unitest.tsv, i.e.,
40 *aggregate-stats* rows — useful for sampling experiment cells but not for
per-test annotation (annotators need the literal function code and the
literal generated tests, not val_score=0.68).

This script reads checkpoints_mutation/*.pkl (each is a list of dicts with
function_code, generated_tests, method, model, source, task_id, ...) and
emits a stratified worksheet with one row per (function, generated_tests)
pair the annotator should read.

Stratification:
  - Balance across methods (plain_llm, random_rag, simple_rag,
    iterative_critique)
  - Balance across models (llama3.2, phi4, qwen3.5, qwen3-coder)
  - Balance across source (HumanEval, MBPP)
  - For each (method, model), pick samples from kill-rate strata:
    best, median, worst (when analysis data exists) — falls back to
    random if no analysis data for that cell

Output:
  human_eval_pairs.csv      — annotator worksheet (one row per pair)
  human_eval_pairs.meta.csv — PRIVATE: sample_id ↔ method/model/source
                              mapping. Keep separate so the worksheet
                              can be shared with annotators blinded.

The displayed CSV columns are restricted to:
  sample_id, function_code, generated_tests, ground_truth_tests,
  human_test_idiom, human_correctness, human_completeness,
  annotator_notes

(retrieved chunks aren't stored in the generation pkls, so we don't
ask annotators to judge "faithfulness to retrieved docs" — the
human_test_idiom dimension judges idiomatic pytest style instead.)
"""

from __future__ import annotations

import argparse
import math
import pickle
import random
import sys
from pathlib import Path

import pandas as pd

GEN_DIR_CANDIDATES = [Path("checkpoints_mutation"),
                       Path(".checkpoints_mutation")]
ANALYSIS_DIR_CANDIDATES = [Path("checkpoints_mutation_analysis"),
                            Path(".checkpoints_mutation_analysis")]


def _resolve(candidates: list[Path]) -> Path | None:
    for p in candidates:
        if p.is_dir():
            return p
    return None


# ---------------------------------------------------------------------------
# Loaders
# ---------------------------------------------------------------------------

def load_generation_pairs(gen_dir: Path) -> pd.DataFrame:
    rows = []
    for f in sorted(gen_dir.glob("*.pkl")):
        if f.name.endswith(".tmp"):
            continue
        with open(f, "rb") as fp:
            data = pickle.load(fp)
        if not isinstance(data, list):
            continue
        for s in data:
            if not s.get("generated_tests", "").strip():
                continue
            rows.append({
                "method":             s.get("method", ""),
                "reasoning":          s.get("reasoning", "base"),
                "model":              s.get("model", ""),
                "sample_idx":         s.get("sample_idx", -1),
                "source":             s.get("source", "unknown"),
                "task_id":            s.get("task_id", f"sample_{s.get('sample_idx', -1)}"),
                "function_code":      s.get("function_code", ""),
                "generated_tests":    s.get("generated_tests", ""),
                "ground_truth_tests": s.get("ground_truth_tests", ""),
            })
    return pd.DataFrame(rows)


def load_kill_rates(analysis_dir: Path | None) -> dict:
    """Map (method, model_underscored, sample_idx) → kill_rate (NaN-skipped)."""
    out = {}
    if analysis_dir is None or not analysis_dir.is_dir():
        return out
    for f in sorted(analysis_dir.glob("*.pkl")):
        if f.name.endswith(".tmp"):
            continue
        with open(f, "rb") as fp:
            try:
                data = pickle.load(fp)
            except Exception:
                continue
        if not isinstance(data, dict):
            continue
        # Filename: {method}_{reasoning}_{model_underscored}
        stem = f.stem
        # Parse: find which known method prefix matches
        for m in ("iterative_critique", "plain_llm", "random_rag", "simple_rag"):
            if stem.startswith(m + "_"):
                rest = stem[len(m) + 1:]
                parts = rest.split("_", 1)
                if len(parts) < 2:
                    break
                _reasoning, model_under = parts[0], parts[1]
                for sample_idx, result in data.items():
                    kr = result.get("kill_rate", float("nan"))
                    if isinstance(kr, float) and math.isnan(kr):
                        continue
                    out[(m, model_under, sample_idx)] = float(kr)
                break
    return out


# ---------------------------------------------------------------------------
# Stratified sampling
# ---------------------------------------------------------------------------

def add_kill_rate(df: pd.DataFrame, kill_rates: dict) -> pd.DataFrame:
    df = df.copy()
    df["model_underscored"] = df["model"].str.replace(":", "_", regex=False)
    df["kill_rate"] = df.apply(
        lambda r: kill_rates.get((r["method"], r["model_underscored"],
                                  r["sample_idx"]), float("nan")),
        axis=1,
    )
    return df


def stratified_sample(df: pd.DataFrame, n: int, seed: int) -> pd.DataFrame:
    """Pick n rows balanced across (method × model) and within each cell across
    source (humaneval, mbpp) and kill-rate strata (best / mid / worst)."""
    rng = random.Random(seed)
    rows = []

    method_x_model = [(m, mod) for m in sorted(df["method"].unique())
                      for mod in sorted(df["model"].unique())]
    # n_per_cell: aim for evenly distributed, round up
    n_cells = len(method_x_model)
    n_per_cell = max(1, (n + n_cells - 1) // n_cells)

    for m, mod in method_x_model:
        sub = df[(df["method"] == m) & (df["model"] == mod)]
        if sub.empty:
            continue
        picks = []
        # Try to get one HumanEval + one MBPP if available
        for src in ("humaneval", "mbpp"):
            cand = sub[sub["source"] == src]
            if cand.empty:
                continue
            # Pick best by kill_rate if available, else random
            if cand["kill_rate"].notna().any():
                picks.append(cand.loc[cand["kill_rate"].idxmax()])
            else:
                picks.append(cand.iloc[rng.randrange(len(cand))])
        # Fill remaining slots randomly from sub
        remaining = max(0, n_per_cell - len(picks))
        if remaining > 0:
            available = sub.drop(index=[p.name for p in picks if p is not None])
            if not available.empty:
                k = min(remaining, len(available))
                idxs = rng.sample(list(available.index), k=k)
                for ix in idxs:
                    picks.append(available.loc[ix])
        rows.extend(picks)

    sampled = pd.DataFrame(rows).reset_index(drop=True)
    # Trim or pad to exactly n
    if len(sampled) > n:
        sampled = sampled.sample(n=n, random_state=seed).reset_index(drop=True)
    elif len(sampled) < n:
        leftover = df[~df.index.isin(sampled.index)]
        topup = leftover.sample(n=min(n - len(sampled), len(leftover)),
                                random_state=seed)
        sampled = pd.concat([sampled, topup], ignore_index=True)

    sampled = sampled.sample(frac=1, random_state=seed).reset_index(drop=True)
    sampled.insert(0, "sample_id",
                   [f"s_{i:03d}" for i in range(len(sampled))])
    return sampled


# ---------------------------------------------------------------------------
# Output
# ---------------------------------------------------------------------------

def write_worksheets(sampled: pd.DataFrame, blinded_path: Path,
                     meta_path: Path) -> None:
    # Annotation worksheet (visible to annotators) — no method/model
    blinded_cols = ["sample_id", "function_code", "generated_tests",
                    "ground_truth_tests"]
    for col in ("human_test_idiom", "human_correctness",
                "human_completeness", "annotator_notes"):
        sampled[col] = ""
        blinded_cols.append(col)
    sampled[blinded_cols].to_csv(blinded_path, index=False)

    # Private metadata (for analysis only) — keep AWAY from annotators
    meta_cols = ["sample_id", "method", "model", "source", "sample_idx",
                 "task_id", "kill_rate"]
    sampled[meta_cols].to_csv(meta_path, index=False)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--n", type=int, default=40,
                        help="number of samples in the worksheet (default 40)")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--worksheet", default="human_eval_pairs.csv",
                        help="annotator-facing CSV (blinded)")
    parser.add_argument("--meta", default="human_eval_pairs.meta.csv",
                        help="private metadata CSV (do NOT share with annotators)")
    args = parser.parse_args()

    gen_dir = _resolve(GEN_DIR_CANDIDATES)
    if gen_dir is None:
        sys.exit("ERROR: no generation checkpoints dir found "
                 "(expected checkpoints_mutation/ or .checkpoints_mutation/)")

    analysis_dir = _resolve(ANALYSIS_DIR_CANDIDATES)

    print(f"Loading generation pairs from {gen_dir}/ ...")
    df = load_generation_pairs(gen_dir)
    print(f"  {len(df)} pairs across "
          f"{df['method'].nunique()} methods × "
          f"{df['model'].nunique()} models × "
          f"{df['source'].nunique()} sources")

    if analysis_dir is not None:
        kill_rates = load_kill_rates(analysis_dir)
        print(f"  kill rates available for {len(kill_rates)} samples")
        df = add_kill_rate(df, kill_rates)
    else:
        df["model_underscored"] = df["model"].str.replace(":", "_", regex=False)
        df["kill_rate"] = float("nan")
        print("  no analysis dir → strata fall back to random within cell")

    print(f"Sampling {args.n} pairs (seed={args.seed})...")
    sampled = stratified_sample(df, args.n, args.seed)

    print("Sample distribution:")
    print(sampled.groupby(["method", "model"]).size().unstack(fill_value=0))
    print()
    print("Source split:")
    print(sampled["source"].value_counts().to_string())
    print()

    write_worksheets(sampled, Path(args.worksheet), Path(args.meta))
    print(f"\nBlinded worksheet → {args.worksheet}   "
          f"({len(sampled)} rows, no method/model visible)")
    print(f"Private metadata  → {args.meta}   "
          f"(DO NOT share with annotators)")

    return 0


if __name__ == "__main__":
    sys.exit(main())
