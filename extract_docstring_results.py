"""
extract_docstring_results.py — Extract RAG-Docstring results into standard TSV format.

Reads the comprehensive CSV files from the RAG-Docstring repo and outputs
results_docstring.tsv in the same column schema as results_unitest.tsv,
enabling cross-task faithfulness comparison in compare_tasks.py.

Usage:
    python extract_docstring_results.py
    python extract_docstring_results.py --docstring-dir /path/to/RAG-Docstring/results

Output:
    results_docstring.tsv
"""

import argparse
import math
import sys
from pathlib import Path

import pandas as pd

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------

DEFAULT_DOCSTRING_DIR = Path(
    "/Users/balajivenktesh/Desktop/Education/Docstring_RAG/RAG-Docstring/results"
)
OUTPUT_FILE = Path("results_docstring.tsv")

# ---------------------------------------------------------------------------
# Method name mapping: RAG-Docstring → unified autoresearch naming
# ---------------------------------------------------------------------------

METHOD_MAP = {
    "PlainLLM":             ("plain_llm",          "base"),
    "FewShotPlainLLM":      ("plain_llm",          "fewshot"),
    "CoTPlainLLM":          ("plain_llm",          "cot"),
    "ToTPlainLLM":          ("plain_llm",          "tot"),
    "GoTPlainLLM":          ("plain_llm",          "got"),
    "SimpleRAG":            ("simple_rag",         "base"),
    "CoTRAG":               ("simple_rag",         "cot"),
    "ToTRAG":               ("simple_rag",         "tot"),
    "GoTRAG":               ("simple_rag",         "got"),
    "SelfCorrectionRAG":    ("iterative_critique", "base"),
    "CoTSelfCorrectionRAG": ("iterative_critique", "cot"),
    "ToTSelfCorrectionRAG": ("iterative_critique", "tot"),
    "GoTSelfCorrectionRAG": ("iterative_critique", "got"),
}

# CSV files produced by RAG-Docstring's compare_all_strategies.py
CSV_FILES = [
    "comprehensive_plain_comparison_report.csv",
    "comprehensive_rag_comparison_report.csv",
    "comprehensive_selfcorrectiverag_comparison_report.csv",
]


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _nan_if_zero_plain(method_name: str, value: float) -> float:
    """
    RAG-Docstring stores token_overlap_score = 0.0 for plain_llm (no context).
    Convert to NaN so it aligns with our unified faithfulness convention.
    """
    base_method = METHOD_MAP.get(method_name, ("", ""))[0]
    if base_method == "plain_llm" and value == 0.0:
        return float("nan")
    return value


def _load_model_results(model_dir: Path, model_name: str) -> list:
    """Load all CSV files for one model directory, return list of row dicts."""
    rows = []
    for csv_name in CSV_FILES:
        csv_path = model_dir / csv_name
        if not csv_path.exists():
            continue
        try:
            df = pd.read_csv(csv_path)
        except Exception as e:
            print(f"  Warning: could not read {csv_path}: {e}")
            continue

        for _, row in df.iterrows():
            raw_method = str(row.get("Method", "")).strip()
            if raw_method not in METHOD_MAP:
                print(f"  Warning: unknown method '{raw_method}' — skipping")
                continue

            method_name, reasoning = METHOD_MAP[raw_method]
            method_str = f"{method_name}/{reasoning}"

            # token_overlap_score = our faithfulness (same formula, closest match)
            tok_overlap = float(row.get("token_overlap_score", float("nan")) or float("nan"))
            faithfulness = _nan_if_zero_plain(raw_method, tok_overlap)

            rows.append({
                "commit":               "docstring",         # no git hash for docstring runs
                "val_score":            float(row.get("rouge_1_f1", 0.0) or 0.0),
                "method":               method_str,
                "model":                model_name,
                "status":               "keep",
                "description":          f"docstring/{raw_method}",
                # Quality metrics
                "avg_syntax":           float("nan"),        # not applicable for docstring
                "avg_edge":             float("nan"),        # not applicable for docstring
                "avg_assert_density":   float("nan"),        # not applicable for docstring
                "avg_semantic_sim":     float(row.get("bert_score", float("nan")) or float("nan")),
                "avg_rouge":            float(row.get("rouge_1_f1", float("nan")) or float("nan")),
                # Diagnostic metrics
                "avg_noise_rate":       float("nan"),        # not tracked in docstring runs
                "avg_faithfulness":     faithfulness,
                "avg_retrieval_secs":   float("nan"),        # not tracked separately
                "avg_llm_secs":         float(row.get("Avg Time/Sample (s)", float("nan")) or float("nan")),
                "avg_tokens":           float("nan"),        # not tracked in docstring runs
                # Task tag for compare_tasks.py
                "task":                 "docstring",
            })
    return rows


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main(docstring_dir: Path) -> None:
    if not docstring_dir.exists():
        print(f"ERROR: Docstring results directory not found:\n  {docstring_dir}")
        sys.exit(1)

    all_rows = []
    model_dirs = sorted(p for p in docstring_dir.iterdir() if p.is_dir())

    if not model_dirs:
        print(f"ERROR: No model subdirectories found in {docstring_dir}")
        sys.exit(1)

    for model_dir in model_dirs:
        model_name = model_dir.name
        rows = _load_model_results(model_dir, model_name)
        if rows:
            print(f"  {model_name}: {len(rows)} method results loaded")
            all_rows.extend(rows)
        else:
            print(f"  {model_name}: no results found (CSVs missing?)")

    if not all_rows:
        print("ERROR: No results extracted. Check that CSV files exist in model directories.")
        sys.exit(1)

    df = pd.DataFrame(all_rows)
    df.to_csv(OUTPUT_FILE, sep="\t", index=False, float_format="%.6f")
    print(f"\nExtracted {len(df)} rows → {OUTPUT_FILE}")
    print("\nSample (faithfulness column):")
    print(df[["method", "model", "avg_faithfulness", "avg_rouge"]].to_string(index=False))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Extract RAG-Docstring results to TSV")
    parser.add_argument(
        "--docstring-dir", type=Path, default=DEFAULT_DOCSTRING_DIR,
        help="Path to RAG-Docstring results/ directory",
    )
    args = parser.parse_args()
    main(args.docstring_dir)
