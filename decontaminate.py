#!/usr/bin/env python3
"""
decontaminate.py — build a decontaminated variant of the evaluation corpus.

Why
---
HumanEval (2021) and MBPP (2021) predate every model in the sweep — llama3.2's
cutoff is December 2023 — and both ship their reference solutions *and* their
canonical test suites. A model may therefore be reciting a memorised suite
rather than reasoning about the function. The SQJ reviewer raised exactly this:
"whether the datasets being in the training data will have influenced the
results". The submitted manuscript never addressed it; the word "contamination"
does not appear in its 39 pages.

Approach
--------
Apply semantics-preserving transformations that break surface-level recall
while leaving behaviour identical, then re-run the sweep on the transformed
corpus and report the kill-rate delta. A large drop implicates memorisation;
a small one is evidence against it.

Transformations (all deterministic, so the corpus is reproducible):

    1. Rename the function            — AST, recursive references updated
    2. Rename parameters              — AST, via a fixed alias table
    3. Rewrite the reference tests     — so they call the renamed function

Deliberately NOT included: LLM docstring paraphrase. The roundtrip-closure
implementation uses one, but it makes the corpus non-reproducible and
unauditable, which is the wrong trade for a paper already under scrutiny for
verifiability. Docstring text is left intact and that limitation is stated
rather than papered over.

Sanity check
------------
Every transformed problem must still pass its own rewritten reference tests.
Anything that fails is dropped rather than silently carried forward, because a
transform that changes behaviour would confound the kill-rate comparison with
plain breakage.

Usage
-----
    python3 decontaminate.py --dry-run          # report, write nothing
    python3 decontaminate.py                    # write the decontaminated cache
    python3 decontaminate.py --limit 10         # quick check on 10 problems
"""
from __future__ import annotations

import argparse
import ast
import hashlib
import pickle
import re
import sys
from pathlib import Path

from mutation_testing import (
    DATASET_CACHE,
    corpus_fingerprint,
    run_tests_against_code,
    write_corpus_manifest,
)

DECONTAM_CACHE = DATASET_CACHE.with_name(
    DATASET_CACHE.stem + "_decontaminated" + DATASET_CACHE.suffix)
DECONTAM_MANIFEST = Path("corpus_manifest_decontaminated.tsv")

# Fixed parameter aliases. Semantically neutral, and chosen so the renamed
# parameter stays readable — an unreadable corpus would confound the human
# evaluation if it were ever reused there.
PARAM_ALIASES: dict[str, str] = {
    "n": "count_val", "m": "size_val", "k": "index_val", "x": "first_val",
    "y": "second_val", "z": "third_val", "s": "text_val", "t": "other_text",
    "a": "alpha_val", "b": "beta_val", "c": "gamma_val", "i": "idx_val",
    "j": "jdx_val", "l": "seq_val", "arr": "sequence", "lst": "items",
    "nums": "numbers", "num": "value", "string": "text", "str1": "text_a",
    "str2": "text_b", "list1": "items_a", "list2": "items_b",
    "test_list": "input_items", "test_tup": "input_tuple",
    "text": "content", "word": "token", "words": "tokens",
    "target": "goal_val", "val": "given_val", "data": "payload",
}


def _first_funcdef(tree: ast.Module):
    for node in tree.body:
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
            return node
    return None


def function_name(code: str) -> str | None:
    """Top-level function name, from the AST.

    `entry_point` is populated for HumanEval samples but empty for MBPP ones,
    so it cannot be relied on.
    """
    try:
        fd = _first_funcdef(ast.parse(code))
    except SyntaxError:
        return None
    return fd.name if fd else None


def new_name_for(name: str) -> str:
    """Deterministic renamed identifier, stable across runs."""
    return f"{name}_{hashlib.sha256(name.encode()).hexdigest()[:4]}"


def rename_function(code: str, old: str, new: str) -> str:
    """Rename the top-level function and every reference to it."""
    tree = ast.parse(code)

    class Renamer(ast.NodeTransformer):
        def visit_FunctionDef(self, node):
            self.generic_visit(node)
            if node.name == old:
                node.name = new
            return node

        def visit_Name(self, node):
            if node.id == old:
                node.id = new
            return node

    tree = Renamer().visit(tree)
    ast.fix_missing_locations(tree)
    return ast.unparse(tree)


def rename_params(code: str) -> tuple[str, dict[str, str]]:
    """Rename the top-level function's parameters via PARAM_ALIASES.

    Only declared parameters are touched, and only inside that function, so
    module-level names and attribute accesses are unaffected. Any alias that
    would collide with a name already used in the function is skipped.
    """
    tree = ast.parse(code)
    fd = _first_funcdef(tree)
    if fd is None:
        return code, {}

    declared = [a.arg for a in fd.args.args] + [a.arg for a in fd.args.kwonlyargs]
    used = {n.id for n in ast.walk(fd) if isinstance(n, ast.Name)} | set(declared)
    mapping = {
        old: PARAM_ALIASES[old]
        for old in declared
        if old in PARAM_ALIASES and PARAM_ALIASES[old] not in used
    }
    if not mapping:
        return code, {}

    for a in list(fd.args.args) + list(fd.args.kwonlyargs):
        if a.arg in mapping:
            a.arg = mapping[a.arg]

    class ParamRenamer(ast.NodeTransformer):
        def visit_Name(self, node):
            if node.id in mapping:
                node.id = mapping[node.id]
            return node

    ParamRenamer().visit(fd)
    ast.fix_missing_locations(tree)
    return ast.unparse(tree), mapping


def rewrite_tests(tests: str, old: str, new: str) -> str:
    """Point the reference tests at the renamed function.

    Only the function name is substituted. Parameter renames are irrelevant
    here because both benchmarks call positionally.
    """
    if not tests.strip():
        return tests
    return re.sub(rf"\b{re.escape(old)}\b", new, tests)


def decontaminate_sample(sample: dict) -> tuple[dict | None, str]:
    """Transform one sample. Returns (new_sample, reason_if_dropped)."""
    code = sample.get("function_code", "")
    tests = sample.get("ground_truth_tests", "")
    old = function_name(code)
    if not old:
        return None, "no parseable top-level function"

    new = new_name_for(old)
    try:
        renamed = rename_function(code, old, new)
        renamed, mapping = rename_params(renamed)
    except SyntaxError as e:
        return None, f"AST rewrite failed: {e}"

    new_tests = rewrite_tests(tests, old, new)

    # Behaviour must be unchanged: the rewritten reference tests have to pass.
    # run_tests_against_code returns "pass" | "fail" | "error" | "timeout".
    status = run_tests_against_code(new_tests, renamed)
    if status != "pass":
        return None, f"sanity check {status}"

    out = dict(sample)
    out["function_code"] = renamed
    out["ground_truth_tests"] = new_tests
    out["entry_point"] = new
    out["decontaminated"] = True
    out["original_entry_point"] = old
    out["original_task_id"] = sample.get("task_id")
    out["param_renames"] = mapping
    # Tag the id. A decontaminated problem is a different problem instance, and
    # the corpus guard compares task ids — without the tag it cannot tell the two
    # arms apart, and a decontaminated checkpoint would resume happily against
    # the main corpus. `source` is left untouched so the per-benchmark
    # decomposition still slices correctly.
    out["task_id"] = f"{sample.get('task_id')}#decontam"
    return out, ""


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset", default=str(DATASET_CACHE))
    ap.add_argument("--out", default=str(DECONTAM_CACHE))
    ap.add_argument("--limit", type=int, default=None)
    ap.add_argument("--dry-run", action="store_true")
    args = ap.parse_args()

    ds_path = Path(args.dataset)
    if not ds_path.exists():
        sys.exit(f"ERROR: dataset not found at {ds_path}")
    dataset = pickle.load(ds_path.open("rb"))
    if args.limit:
        dataset = dataset[:args.limit]
    print(f"Source corpus: {ds_path.name}  n={len(dataset)}  "
          f"fingerprint {corpus_fingerprint(dataset)}\n")

    kept, dropped = [], []
    for i, s in enumerate(dataset):
        new, reason = decontaminate_sample(s)
        tid = s.get("task_id", f"sample_{i}")
        if new is None:
            dropped.append((tid, reason))
            print(f"  [{i:3d}] DROP {tid:18} {reason}")
        else:
            kept.append(new)
            n_p = len(new["param_renames"])
            print(f"  [{i:3d}] ok   {tid:18} {new['original_entry_point']} → "
                  f"{new['entry_point']}"
                  f"{f'  (+{n_p} param renames)' if n_p else ''}")

    print(f"\n{'='*68}")
    print(f"  kept {len(kept)}/{len(dataset)}   dropped {len(dropped)}")
    if dropped:
        from collections import Counter
        for reason, c in Counter(r.split("(")[0].strip() for _, r in dropped).items():
            print(f"    {c:3d} × {reason}")
    if kept:
        renamed_params = sum(1 for k in kept if k["param_renames"])
        print(f"  {renamed_params}/{len(kept)} also had parameters renamed")
        print(f"  decontaminated fingerprint: {corpus_fingerprint(kept)}")
    print(f"{'='*68}")

    if args.dry_run:
        print("\n--dry-run: nothing written.")
        return 0
    if not kept:
        sys.exit("ERROR: no samples survived — refusing to write an empty corpus.")

    with Path(args.out).open("wb") as f:
        pickle.dump(kept, f)
    print(f"\nWritten → {args.out}")
    write_corpus_manifest(kept, DECONTAM_MANIFEST)
    print(f"\nRun the sweep against it with:")
    print(f"  python3 mutation_testing.py --regenerate \\")
    print(f"      --dataset {args.out} \\")
    print(f"      --regen-checkpoints-dir .checkpoints_mutation_decontam \\")
    print(f"      --manifest {DECONTAM_MANIFEST} \\")
    print(f"      --max-samples {len(kept)} --model <model>")
    print(f"\nThe separate checkpoint directory keeps this arm's generations out "
          f"of the\nmain sweep's. Task ids are tagged '#decontam', so the corpus "
          f"guard also\nrefuses to resume one arm against the other.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
