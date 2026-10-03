#!/usr/bin/env python3
"""
rebuild_tsv.py — regenerate results_mutation.tsv purely from the per-sample
analysis checkpoints.

Why this exists
---------------
results_mutation.tsv is written only at the end of a mutation_testing.py run,
and it ACCUMULATES: rows not covered by the current run are carried over
untouched. That is how a stale row survived for months — the Iterative
Critique x llama3.2 row kept its 30-sample values while every other row moved to
100 samples, and the TSV disagreed with both mutation_report.txt and the
analysis checkpoints.

The analysis checkpoints are the ground truth: one entry per (cell, sample) with
killed / survived / equivalent / total_mutants / per-operator counts. This script
derives the TSV from them alone, so the file becomes a pure function of the
analysis state rather than a ledger with history. Run it after any sweep, and
after restoring checkpoints from Drive.

Usage:
    python3 rebuild_tsv.py                      # writes results_mutation.tsv
    python3 rebuild_tsv.py --dry-run            # show, write nothing
    python3 rebuild_tsv.py --analysis-dir DIR --out FILE
"""
from __future__ import annotations

import argparse
import math
import pickle
import statistics as st
import sys
from pathlib import Path

import pandas as pd

from mutation_statistical_tests import (
    METHOD_LABELS, OPERATORS, _resolve_analysis_dir, parse_key,
)

COLUMNS = ["method", "reasoning", "model", "mean_kill_rate", "std_kill_rate",
           "median_kill_rate", "total_mutants", "total_killed", "total_survived",
           "total_equivalent", "n_samples_valid"] + [f"kill_{o}" for o in OPERATORS]


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--analysis-dir", default=None)
    ap.add_argument("--out", default="results_mutation.tsv")
    ap.add_argument("--dry-run", action="store_true")
    args = ap.parse_args()

    adir = Path(args.analysis_dir) if args.analysis_dir else _resolve_analysis_dir()
    if not adir.is_dir():
        print(f"ERROR: analysis dir not found: {adir}", file=sys.stderr)
        return 2
    print(f"analysis dir: {adir}")

    rows = []
    for f in sorted(adir.glob("*.pkl")):
        if f.name.endswith(".tmp"):
            continue
        data = pickle.load(f.open("rb"))
        if not isinstance(data, dict) or not data:
            continue
        method, reasoning, model = parse_key(f.stem)

        krs, tm, tk, tsv_, te = [], 0, 0, 0, 0
        op_tot = {o: 0 for o in OPERATORS}
        op_kil = {o: 0 for o in OPERATORS}
        for r in data.values():
            kr = r.get("kill_rate")
            # Rows whose suite failed on the ORIGINAL function carry NaN and are
            # excluded from the mean, matching run_mutation_analysis. Their
            # mutants are excluded too, so totals describe analysed samples only.
            if kr is None or (isinstance(kr, float) and math.isnan(kr)):
                continue
            krs.append(float(kr))
            tm += r.get("total_mutants", 0)
            tk += r.get("killed", 0)
            tsv_ += r.get("survived", 0)
            te += r.get("equivalent", 0)
            for o, s in (r.get("per_operator", {}) or {}).items():
                if o in op_tot:
                    op_tot[o] += (s or {}).get("total", 0)
                    op_kil[o] += (s or {}).get("killed", 0)
        if not krs:
            print(f"  skip {f.name}: no analysed samples")
            continue

        rows.append({
            "method": METHOD_LABELS.get(method, method),
            "reasoning": reasoning,
            "model": model.replace("_latest", ":latest").replace("_14b", ":14b")
                          .replace("_9b", ":9b").replace("_30b", ":30b"),
            "mean_kill_rate": st.mean(krs),
            "std_kill_rate": st.stdev(krs) if len(krs) > 1 else 0.0,
            "median_kill_rate": st.median(krs),
            "total_mutants": tm, "total_killed": tk,
            "total_survived": tsv_, "total_equivalent": te,
            "n_samples_valid": len(krs),
            **{f"kill_{o}": (op_kil[o] / op_tot[o]) if op_tot[o] else float("nan")
               for o in OPERATORS},
        })

    df = pd.DataFrame(rows, columns=COLUMNS).sort_values(["method", "model"])
    print(f"\n{len(df)} rows derived from {adir}")
    print(df[["method", "model", "mean_kill_rate", "n_samples_valid",
              "total_killed", "total_mutants"]].to_string(index=False))
    print(f"\ntotals: {df.total_mutants.sum()} mutants, {df.total_killed.sum()} killed, "
          f"{df.total_equivalent.sum()} equivalent, "
          f"{df.n_samples_valid.sum()} analysed samples")

    if args.dry_run:
        print("\n--dry-run: nothing written")
        return 0
    out = Path(args.out)
    if out.exists():
        bak = out.with_suffix(".tsv.bak")
        out.replace(bak)
        print(f"\nprevious file → {bak}")
    df.to_csv(out, sep="\t", index=False, float_format="%.6f")
    print(f"written → {out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
