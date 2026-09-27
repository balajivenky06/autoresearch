"""
pynguin_runner.py — Generate Pynguin test suites for the same functions
the LLM methods were evaluated on, so mutation_testing.py can give us a
head-to-head Pynguin-vs-LLM comparison.

Selects N distinct functions, writes each as a temp Python module, runs Pynguin
with a per-function time budget, and collects the generated test file. Output
schema matches the existing checkpoints_mutation/*.pkl files so
mutation_testing.py can analyse Pynguin's tests with no changes.

Subset selection
----------------
Two modes, and the default is not the recommended one:

  --from-corpus  draws N distinct functions from the evaluation corpus under
                 --seed. USE THIS.
  (default)      draws from human_eval_pairs.csv.

The worksheet is a list of (function, generated-tests) PAIRS: a function recurs
once per method/model cell it was sampled in. Pynguin generates from the
function alone, so one job per row yields near-duplicate entries. The 40-row
checkpoint this replaces covered just 14 distinct functions, with HumanEval/51
and MBPP/67 contributing 12 rows each -- 60% of the comparison resting on two
problems, while the paper described it as "the same 40 functions". Worksheet mode
now deduplicates by function, which caps it at those 14; --from-corpus lifts the
ceiling to the whole corpus.

Run:
    pip install pynguin                                          # one-time
    python3 pynguin_runner.py --from-corpus --n 5  --budget 60    # smoke test
    python3 pynguin_runner.py --from-corpus --n 40 --budget 60    # full comparison

Output:
    checkpoints_mutation/pynguin_base_pynguin.pkl       — generation pkl
    logs/pynguin_runner.log                             — per-sample log
"""

from __future__ import annotations

import argparse
import json
import os
import pickle
import re
import shutil
import subprocess
import sys
import tempfile
import textwrap
import time
from pathlib import Path

import pandas as pd

# Layout matching the LLM-generated checkpoints so mutation_testing.py treats
# Pynguin as just another (method, model) pair.
PYNGUIN_METHOD    = "pynguin"
PYNGUIN_REASONING = "base"
PYNGUIN_MODEL     = "pynguin"   # the "model" slot is the tool name here

OUTPUT_PKL  = Path("checkpoints_mutation") / "pynguin_base_pynguin.pkl"
LOG_FILE    = Path("logs") / "pynguin_runner.log"
WORKSHEET   = Path("human_eval_pairs.csv")
META        = Path("human_eval_pairs.meta.csv")


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def extract_function_name(function_code: str) -> str | None:
    m = re.search(r"^\s*def\s+(\w+)\s*\(", function_code, re.MULTILINE)
    return m.group(1) if m else None


def write_module(function_code: str, mod_path: Path) -> None:
    """Write the function as a standalone importable module."""
    text = function_code
    # Strip any existing `if __name__ == "__main__"` blocks — Pynguin will
    # explore the module on import and we don't want main() side effects.
    text = re.sub(r"^if\s+__name__\s*==\s*['\"]__main__['\"].*", "", text,
                  flags=re.MULTILINE | re.DOTALL)
    mod_path.write_text(text)


def run_pynguin(module_dir: Path, module_name: str, budget_secs: int,
                out_dir: Path, log_handle) -> tuple[bool, str]:
    """
    Invoke Pynguin on a single-module project. Returns (success, error_msg).
    """
    env = os.environ.copy()
    env["PYNGUIN_DANGER_AWARE"] = "1"
    cmd = [
        sys.executable, "-m", "pynguin",
        "--project-path", str(module_dir),
        "--output-path", str(out_dir),
        "--module-name", module_name,
        "--maximum-search-time", str(budget_secs),
        "--output-variables", "TargetModule,Coverage",
        "--report-dir", str(out_dir / "report"),
        "--seed", "42",
    ]
    log_handle.write(f"\n  cmd: {' '.join(cmd)}\n")
    log_handle.flush()
    try:
        proc = subprocess.run(
            cmd, capture_output=True, text=True, env=env,
            timeout=budget_secs * 3 + 30,  # generous wall-clock cap
            cwd=str(module_dir),
        )
        log_handle.write("  stdout (last 600):\n")
        log_handle.write(textwrap.indent(proc.stdout[-600:], "    "))
        log_handle.write("\n  stderr (last 600):\n")
        log_handle.write(textwrap.indent(proc.stderr[-600:], "    "))
        log_handle.write(f"\n  exit code: {proc.returncode}\n")
        log_handle.flush()
        return (proc.returncode == 0,
                proc.stderr.splitlines()[-1] if proc.stderr else "")
    except subprocess.TimeoutExpired:
        return False, f"timeout > {budget_secs * 3 + 30}s"
    except Exception as e:
        return False, str(e)


def collect_test_file(out_dir: Path, module_name: str) -> str | None:
    """Find the generated test file under out_dir and return its contents."""
    # Pynguin writes test_<module>.py inside out_dir.
    candidates = list(out_dir.glob(f"test_{module_name}.py")) + \
                  list(out_dir.glob("test_*.py"))
    if not candidates:
        return None
    # Prefer the one matching the module name
    target = candidates[0]
    return target.read_text()


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--n", type=int, default=5,
                        help="number of samples to run Pynguin on (default 5 = smoke test)")
    parser.add_argument("--budget", type=int, default=60,
                        help="Pynguin search budget per function (seconds)")
    parser.add_argument("--worksheet", default=str(WORKSHEET),
                        help="path to the blinded worksheet")
    parser.add_argument("--meta", default=str(META),
                        help="path to the meta CSV (for sample_idx / source)")
    parser.add_argument("--out", default=str(OUTPUT_PKL),
                        help="output pkl path")
    parser.add_argument("--allow-duplicates", action="store_true",
                        help="run one Pynguin job per worksheet ROW rather than "
                             "per distinct function. Reproduces the original "
                             "behaviour, which over-weighted a few functions; "
                             "only useful for replicating the old numbers.")
    parser.add_argument("--from-corpus", action="store_true",
                        help="draw the subset from the evaluation corpus instead "
                             "of the human-eval worksheet. RECOMMENDED: the "
                             "worksheet holds only 14 distinct functions, which "
                             "is too few for a tool comparison.")
    parser.add_argument("--seed", type=int, default=42,
                        help="seed for --from-corpus sampling (default 42)")
    args = parser.parse_args()

    LOG_FILE.parent.mkdir(exist_ok=True)
    OUTPUT_PKL.parent.mkdir(exist_ok=True)

    if args.from_corpus:
        # Draw distinct functions straight from the evaluation corpus. The tool
        # comparison and the human evaluation are separate analyses and do not
        # need a shared subset — coupling them was a convenience that capped
        # Pynguin at the worksheet's 14 distinct functions.
        import pickle as _pickle
        import numpy as _np
        from mutation_testing import DATASET_CACHE

        if not DATASET_CACHE.exists():
            print(f"ERROR: corpus not found at {DATASET_CACHE}", file=sys.stderr)
            return 2
        corpus = _pickle.load(DATASET_CACHE.open("rb"))
        rng = _np.random.default_rng(args.seed)
        take = min(args.n, len(corpus))
        picked = sorted(rng.choice(len(corpus), size=take, replace=False).tolist())
        joined = pd.DataFrame([{
            "sample_id":          f"corpus_{i}",
            "function_code":      corpus[i]["function_code"],
            "ground_truth_tests": corpus[i].get("ground_truth_tests", ""),
            "sample_idx":         i,
            "task_id":            corpus[i].get("task_id", f"sample_{i}"),
            "source":             corpus[i].get("source", "unknown"),
        } for i in picked])
        print(f"Subset drawn from {DATASET_CACHE.name} "
              f"(n={len(corpus)}, seed={args.seed}): {len(joined)} distinct functions")
    else:
        # Load worksheet and metadata
        worksheet = pd.read_csv(args.worksheet)
        meta = pd.read_csv(args.meta) if Path(args.meta).exists() else \
            pd.DataFrame(columns=["sample_id", "source", "sample_idx", "task_id"])
        joined = worksheet.merge(
            meta[["sample_id", "source", "sample_idx", "task_id"]],
            on="sample_id", how="left",
        )

    # ── Deduplicate by function ─────────────────────────────────────────
    # The worksheet holds (function, generated-tests) PAIRS: a function recurs
    # once per method/model cell it was sampled in. Pynguin generates from the
    # function alone, so running it per worksheet row produces near-identical
    # entries for the same function and silently weights the comparison.
    #
    # The 40-row checkpoint this replaces covered only 14 distinct functions,
    # with HumanEval/51 and MBPP/67 contributing 12 rows each — 60% of the
    # comparison resting on two problems, while the paper described it as
    # "the same 40 functions".
    n_rows = len(joined)
    if not args.allow_duplicates:
        key = joined["task_id"].where(joined["task_id"].notna(),
                                      joined["function_code"])
        joined = joined.assign(_dedup_key=key) \
                       .drop_duplicates(subset="_dedup_key", keep="first") \
                       .drop(columns="_dedup_key") \
                       .reset_index(drop=True)
        if n_rows != len(joined):
            print(f"Deduplicated worksheet: {n_rows} rows → "
                  f"{len(joined)} distinct functions")

    if args.n > len(joined):
        print(f"NOTE: --n {args.n} exceeds the {len(joined)} distinct functions "
              f"available; running all {len(joined)}.")
    joined = joined.head(args.n).reset_index(drop=True)

    print(f"Running Pynguin on {len(joined)} distinct functions "
          f"with {args.budget}s budget each "
          f"(estimated total: {len(joined) * args.budget / 60:.1f} min)")
    if "source" in joined:
        counts = joined["source"].fillna("unknown").value_counts().to_dict()
        print(f"  benchmark split: {counts}")

    # Per-sample log
    log = open(LOG_FILE, "w")
    log.write(f"Pynguin runner — {len(joined)} samples × {args.budget}s budget\n")
    log.write(f"Pynguin version: {subprocess.run([sys.executable, '-m', 'pynguin', '--version'], env={**os.environ, 'PYNGUIN_DANGER_AWARE': '1'}, capture_output=True, text=True).stdout.strip()}\n\n")

    results = []
    succeeded = 0
    for i, row in joined.iterrows():
        sample_id = row["sample_id"]
        func_code = row["function_code"]
        fn_name = extract_function_name(func_code) or f"fn_{i}"
        sample_idx = int(row["sample_idx"]) if pd.notna(row.get("sample_idx")) else i
        source = row.get("source", "unknown")
        if pd.isna(source):
            source = "unknown"

        log.write(f"\n[{i+1}/{len(joined)}] {sample_id}  fn={fn_name}  "
                  f"source={source}\n")
        print(f"  [{i+1}/{len(joined)}] {sample_id} (fn={fn_name})  ", end="",
              flush=True)

        t0 = time.time()
        generated_tests = ""
        error = ""
        with tempfile.TemporaryDirectory() as td:
            tmp = Path(td)
            mod_dir = tmp / "project"
            mod_dir.mkdir()
            module_name = f"sut_{re.sub(r'[^A-Za-z0-9]', '_', fn_name)}"
            mod_path = mod_dir / f"{module_name}.py"
            write_module(func_code, mod_path)
            out_dir = tmp / "out"
            out_dir.mkdir()

            ok, err_line = run_pynguin(mod_dir, module_name, args.budget,
                                       out_dir, log)
            if ok:
                test_text = collect_test_file(out_dir, module_name)
                if test_text:
                    # Pynguin generates `import sut_xxx as module_0` and then
                    # references `module_0.fn_name(...)`. When mutation_testing.py
                    # inlines function_code at the top of the test file, those
                    # qualified references break because `module_0` is undefined.
                    # Rewrite both the import lines AND the qualified calls so
                    # the tests resolve symbols directly against the inlined
                    # function.

                    # 1. Comment out import lines that name our SUT module
                    test_text = re.sub(
                        rf"^\s*from\s+{re.escape(module_name)}\s+import\s+(.+)$",
                        r"# rewritten import: \1 provided by inlined function_code",
                        test_text, flags=re.MULTILINE,
                    )
                    test_text = re.sub(
                        rf"^\s*import\s+{re.escape(module_name)}(?:\s+as\s+\w+)?.*$",
                        "# rewritten import — module inlined by mutation_testing",
                        test_text, flags=re.MULTILINE,
                    )
                    # 2. Strip qualified calls like sut_xxx.foo(...) → foo(...)
                    test_text = re.sub(
                        rf"\b{re.escape(module_name)}\.", "", test_text)
                    # 3. Strip qualified calls via the alias `module_0.` (Pynguin's
                    # canonical alias name across all generated suites).
                    test_text = re.sub(r"\bmodule_0\.", "", test_text)

                    generated_tests = test_text
                    succeeded += 1
                    print(f"OK ({time.time()-t0:.1f}s)  {len(test_text)}b")
                else:
                    error = "Pynguin reported success but no test file"
                    print(f"NO_TEST  ({time.time()-t0:.1f}s)")
            else:
                error = err_line
                print(f"FAIL  ({time.time()-t0:.1f}s)  {err_line[:60]}")

        results.append({
            "sample_idx":         sample_idx,
            "task_id":            row.get("task_id", sample_id),
            "source":             source,
            "function_code":      func_code,
            "ground_truth_tests": row.get("ground_truth_tests", "") or "",
            "generated_tests":    generated_tests,
            "method":             PYNGUIN_METHOD,
            "reasoning":          PYNGUIN_REASONING,
            "model":              PYNGUIN_MODEL,
            "pynguin_error":      error,
        })

    log.write(f"\n\n=== SUMMARY: {succeeded}/{len(joined)} samples produced tests ===\n")
    log.close()

    # Never replace a good checkpoint with a failed run. Without this, a missing
    # `pynguin` install writes 40 rows of empty generated_tests over real data,
    # and the analysis then reports a kill rate of zero for the tool rather than
    # reporting that the tool never ran.
    out_path = Path(args.out)
    if succeeded == 0:
        print(f"\nERROR: no sample produced tests — refusing to write {out_path}.")
        if out_path.exists():
            print(f"       the existing checkpoint is left untouched.")
        print(f"       Check {LOG_FILE}. Most often: `pip install pynguin`.")
        return 2

    if out_path.exists():
        try:
            prev = pickle.load(out_path.open("rb"))
            prev_ok = sum(1 for r in prev if str(r.get("generated_tests", "")).strip())
            if prev_ok > succeeded:
                print(f"\nWARNING: the existing {out_path.name} has {prev_ok} suites, "
                      f"this run produced {succeeded}.")
                backup = out_path.with_suffix(".pkl.bak")
                shutil.copy2(out_path, backup)
                print(f"         previous version backed up → {backup}")
        except Exception:
            pass

    with open(out_path, "wb") as f:
        pickle.dump(results, f)
    print(f"\nSaved {len(results)} entries → {out_path}")
    print(f"Generated tests for {succeeded}/{len(results)} samples "
          f"({100*succeeded/len(results):.0f}%)")
    print(f"Full log: {LOG_FILE}")

    return 0


if __name__ == "__main__":
    sys.exit(main())
