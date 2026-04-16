"""
human_eval_sampler.py — Stratified sample generator for human annotation.

Generates a human_eval_samples.csv with 30–50 samples selected to cover:
  • All 3 methods (plain_llm, simple_rag, iterative_critique)
  • All reasoning modes (base, cot, tot, got)
  • Best and worst performers per method (to ensure score range coverage)

Annotators rate each generated test suite on:
  1. faithfulness   [0-3] — how well tests reflect retrieved documentation patterns
  2. correctness    [0-3] — whether tests correctly target the function under test
  3. completeness   [0-3] — coverage of happy path, edge, and error cases
  4. overall        [0-3] — holistic quality judgment

Pearson r between automated val_score and mean human score validates the
automated metric (threshold: r ≥ 0.7 per Landis & Koch 1977 guidelines).

Usage:
    python human_eval_sampler.py
    python human_eval_sampler.py --results results_unitest.tsv --n 40 --seed 42

Output:
    human_eval_samples.csv  — annotation worksheet (open in Excel / Google Sheets)
    human_eval_guide.txt    — annotation instructions for annotators
"""

import sys
import argparse
import random
import numpy as np
import pandas as pd
from pathlib import Path
from scipy import stats

RESULTS_FILE    = "results_unitest.tsv"
SAMPLES_PER_METHOD = 5   # samples per method; total ≈ N_METHODS × N_REASONING × SAMPLES_PER_METHOD

METHODS = ["plain_llm", "simple_rag", "iterative_critique"]
REASONING_MODES = ["base", "cot", "tot", "got"]


# ---------------------------------------------------------------------------
# Data loading
# ---------------------------------------------------------------------------

def load_results(path: str) -> pd.DataFrame:
    df = pd.read_csv(path, sep="\t")
    if df.empty:
        print("ERROR: results file is empty.")
        sys.exit(1)
    if "/" in str(df["method"].iloc[0]):
        df[["method_name", "reasoning"]] = df["method"].str.split("/", n=1, expand=True)
    else:
        df["method_name"] = df["method"]
        df["reasoning"]   = "base"
    df = df[df["status"] != "crash"].copy()
    for col in ["val_score", "avg_faithfulness", "avg_llm_judge_faithfulness"]:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce")
    return df


# ---------------------------------------------------------------------------
# Stratified sampler
# ---------------------------------------------------------------------------

def sample_for_annotation(df: pd.DataFrame, n_total: int = 40, seed: int = 42) -> pd.DataFrame:
    """
    Stratified sample ensuring:
      - All method × reasoning combos represented
      - Mix of high/low val_score within each stratum
    """
    rng = random.Random(seed)
    rows = []

    for method in METHODS:
        for reasoning in REASONING_MODES:
            sub = df[(df["method_name"] == method) & (df["reasoning"] == reasoning)]
            if sub.empty:
                continue

            sub = sub.sort_values("val_score", ascending=False).reset_index(drop=True)
            # Take top and bottom performers from this stratum
            indices = []
            if len(sub) == 1:
                indices = [0]
            elif len(sub) == 2:
                indices = [0, 1]
            else:
                # Best, worst, and a random middle sample
                mid_idx = rng.randint(1, len(sub) - 2)
                indices = list({0, mid_idx, len(sub) - 1})

            for idx in indices:
                row = sub.iloc[idx].to_dict()
                row["stratum"] = f"{method}/{reasoning}"
                rows.append(row)

    # Drop exact duplicate (method+reasoning+model) rows only — keep all reasoning modes
    sampled = pd.DataFrame(rows).drop_duplicates(subset=["method", "model", "stratum"]).head(n_total)

    # Add annotation columns (blank for human to fill)
    sampled = sampled.copy()
    sampled["human_faithfulness"]  = ""   # 0-3: grounding in retrieved docs
    sampled["human_correctness"]   = ""   # 0-3: tests correctly target the function
    sampled["human_completeness"]  = ""   # 0-3: coverage of happy/edge/error cases
    sampled["human_overall"]       = ""   # 0-3: holistic quality
    sampled["annotator_notes"]     = ""

    return sampled


# ---------------------------------------------------------------------------
# Annotation guide
# ---------------------------------------------------------------------------

ANNOTATION_GUIDE = """
HUMAN EVALUATION ANNOTATION GUIDE
Unit Test Generation — PhD Research (2025)
==========================================

Each row represents one generated test suite. For each row, provide integer
scores 0–3 for the four criteria below. Leave annotator_notes for anything
notable (hallucinations, unusual patterns, etc.).

SCORING CRITERIA
----------------

1. human_faithfulness [0-3]
   How well do the tests reflect patterns from retrieved pytest/unittest docs?
   0 = No grounding; tests show no sign of documentation patterns
   1 = Weak grounding; occasional use of documented idioms
   2 = Moderate grounding; clearly uses some documented patterns (parametrize,
       pytest.raises, etc.)
   3 = Strong grounding; tests closely follow documentation examples and idioms

   NOTE: For plain_llm rows (no retrieval), score faithfulness based on
   general pytest best practices — not retrieval grounding.

2. human_correctness [0-3]
   Are the tests logically correct and targeting the right function behavior?
   0 = Tests are incorrect / test wrong behavior
   1 = Mostly correct with minor errors
   2 = Correct for most cases; minor edge case errors acceptable
   3 = Tests are fully correct and well-targeted

3. human_completeness [0-3]
   How thoroughly do the tests cover happy path, edge cases, and error cases?
   0 = Only one scenario tested (e.g., only happy path)
   1 = Two scenarios covered
   2 = Three or more scenarios, but some gaps
   3 = Comprehensive coverage: happy path, boundaries, errors, and edge cases

4. human_overall [0-3]
   Holistic judgment: would you accept this test suite in a professional code review?
   0 = Reject — too many problems
   1 = Major revision needed
   2 = Minor revision needed
   3 = Accept as-is

INTER-RATER RELIABILITY
------------------------
If two annotators are used, compute Cohen's κ per criterion. Target κ ≥ 0.6
(substantial agreement, Landis & Koch 1977).

VALIDATION TARGET
-----------------
Pearson r between mean human score and automated val_score ≥ 0.7 validates
the automated metric for use in the journal paper.
"""


# ---------------------------------------------------------------------------
# Post-annotation validation
# ---------------------------------------------------------------------------

def validate_against_human(
    annotated_csv: str,
    results_tsv: str = RESULTS_FILE,
) -> None:
    """
    After annotation, call this to compute Pearson r between val_score and
    mean human score. Prints correlation and saves summary.

    Usage:
        python human_eval_sampler.py --validate human_eval_samples.csv
    """
    try:
        ann = pd.read_csv(annotated_csv)
    except Exception as e:
        print(f"Could not read {annotated_csv}: {e}")
        return

    score_cols = ["human_faithfulness", "human_correctness",
                  "human_completeness", "human_overall"]
    for col in score_cols:
        ann[col] = pd.to_numeric(ann[col], errors="coerce")

    ann["mean_human"] = ann[score_cols].mean(axis=1)
    ann["val_score"]  = pd.to_numeric(ann["val_score"], errors="coerce")

    valid = ann[["val_score", "mean_human"]].dropna()
    if len(valid) < 5:
        print(f"Only {len(valid)} fully annotated rows — need at least 5 for validation.")
        return

    r, p = stats.pearsonr(valid["val_score"], valid["mean_human"])
    rho, p_rho = stats.spearmanr(valid["val_score"], valid["mean_human"])

    print(f"\nHuman Evaluation Validation (n={len(valid)} samples)")
    print(f"  Pearson  r  = {r:.3f}  (p={p:.4f})")
    print(f"  Spearman ρ  = {rho:.3f}  (p={p_rho:.4f})")
    threshold = 0.7
    if r >= threshold:
        print(f"  PASS: r={r:.3f} ≥ {threshold} — automated val_score is validated.")
    else:
        print(f"  WARN: r={r:.3f} < {threshold} — val_score may not align with human judgment.")

    # Per-method correlation
    print("\nPer-method Pearson r:")
    for method in ann["method_name"].dropna().unique() if "method_name" in ann.columns else []:
        sub = ann[ann["method_name"] == method][["val_score", "mean_human"]].dropna()
        if len(sub) < 3:
            continue
        r_m, p_m = stats.pearsonr(sub["val_score"], sub["mean_human"])
        print(f"  {method:<25} r={r_m:.3f}  (p={p_m:.4f}, n={len(sub)})")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main(results_file: str, n: int, seed: int, validate: str = None) -> None:
    if validate:
        validate_against_human(validate, results_file)
        return

    if not Path(results_file).exists():
        print(f"ERROR: {results_file} not found. Run experiments first.")
        sys.exit(1)

    df = load_results(results_file)
    sampled = sample_for_annotation(df, n_total=n, seed=seed)

    out_csv = "human_eval_samples.csv"
    sampled.to_csv(out_csv, index=False)
    print(f"Annotation worksheet: {out_csv}  ({len(sampled)} rows)")

    guide_path = "human_eval_guide.txt"
    Path(guide_path).write_text(ANNOTATION_GUIDE.strip())
    print(f"Annotation guide:     {guide_path}")

    print(f"\nStrata covered:")
    for stratum, group in sampled.groupby("stratum"):
        print(f"  {stratum:<35} {len(group)} rows")

    print(f"\nNext steps:")
    print(f"  1. Open {out_csv} in Excel or Google Sheets")
    print(f"  2. Annotate human_faithfulness, human_correctness, human_completeness, human_overall")
    print(f"  3. Run: python human_eval_sampler.py --validate {out_csv}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--results",  default=RESULTS_FILE)
    parser.add_argument("--n",        type=int, default=40,
                        help="Target number of annotation rows")
    parser.add_argument("--seed",     type=int, default=42)
    parser.add_argument("--validate", default=None,
                        help="Path to completed annotation CSV to validate against val_score")
    args = parser.parse_args()
    main(args.results, args.n, args.seed, args.validate)
