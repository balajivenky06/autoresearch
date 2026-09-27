#!/usr/bin/env python3
"""
align_corpus.py — realign the cached eval dataset to the ordering the existing
mutation checkpoints were generated against, so the sweep can be extended
instead of restarted.

The problem
-----------
`prepare_unitest.py` selects its 100-function subset with
`rng.choice(len(combined), 100)` under seed 42 and then emits it as
`[combined[i] for i in sorted(indices)]`. The *membership* of that subset is
reproducible — every task id in the existing checkpoints is present in the
freshly built cache. The *ordering* is not: the checkpoints were generated
against an unsorted permutation of the same 100 functions, so
`dataset[0]` is `HumanEval/16` locally while `sample_idx 0` in every checkpoint
is `MBPP/67`.

That matters because `regenerate_tests` resumes by sample *count*. Extending
30 → 100 against a differently-ordered cache would keep samples 0–29 from the
original corpus and append samples 30–99 from a different ordering, so a single
`sample_idx` would denote two different functions depending on which cell you
looked at. Every paired analysis — Friedman, Wilcoxon, and the `sample_idx`
random intercept — silently compares unrelated functions at that point.

Why realigning is legitimate
----------------------------
The subset is a random sample; its order carries no meaning. The only property
the analysis requires is that a given `sample_idx` denotes the same function in
every cell. Reordering the cache so positions 0–29 match the checkpoints, then
appending the remaining 70 functions, satisfies that for the whole 100 and
preserves all 480 generations already computed.

What this writes
----------------
  - the realigned dataset cache (backing up the original alongside it)
  - `corpus_manifest.tsv`, the committed record of sample_idx → task_id → source

Usage
-----
    python3 align_corpus.py --dry-run      # report, change nothing
    python3 align_corpus.py                # realign and write the manifest
"""
from __future__ import annotations

import argparse
import pickle
import shutil
import sys
from collections import Counter
from pathlib import Path

from mutation_testing import (
    CORPUS_MANIFEST,
    DATASET_CACHE,
    corpus_fingerprint,
    sample_ids,
    write_corpus_manifest,
)

GEN_DIR_CANDIDATES = [Path("checkpoints_mutation"), Path(".checkpoints_mutation")]
BASELINE_MARKERS = ("pynguin",)


def resolve_gen_dir() -> Path:
    for p in GEN_DIR_CANDIDATES:
        if p.is_dir() and any(p.glob("*.pkl")):
            return p
    sys.exit(f"ERROR: no generation checkpoints found in "
             f"{[str(p) for p in GEN_DIR_CANDIDATES]}")


def reference_order(gen_dir: Path) -> list[str]:
    """The task-id ordering shared by every RAG-technique checkpoint.

    Baseline-tool checkpoints (Pynguin) ran on their own subset and are skipped.
    Exits if the RAG checkpoints disagree, since there would then be no single
    ordering to align to.
    """
    orders: dict[str, list[str]] = {}
    for f in sorted(gen_dir.glob("*.pkl")):
        if f.name.endswith(".tmp") or any(m in f.name.lower() for m in BASELINE_MARKERS):
            continue
        data = pickle.load(f.open("rb"))
        if isinstance(data, list) and data:
            orders[f.name] = sample_ids(data)
    if not orders:
        sys.exit("ERROR: no RAG-technique checkpoints found.")

    ref_name, ref = next(iter(orders.items()))
    for name, ids in orders.items():
        k = min(len(ids), len(ref))
        if ids[:k] != ref[:k]:
            sys.exit(
                f"ERROR: checkpoints disagree on sample ordering.\n"
                f"  {ref_name} and {name} differ within their first {k} samples.\n"
                f"  There is no single ordering to align to — the existing\n"
                f"  checkpoints are already inconsistent and must be regenerated."
            )
        if len(ids) > len(ref):
            ref, ref_name = ids, name
    print(f"  reference ordering: {ref_name}  ({len(ref)} samples, "
          f"{len(orders)} checkpoints agree)")
    return ref


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--dry-run", action="store_true",
                    help="report what would change without writing")
    ap.add_argument("--dataset", default=str(DATASET_CACHE))
    args = ap.parse_args()

    ds_path = Path(args.dataset)
    if not ds_path.exists():
        sys.exit(f"ERROR: dataset cache not found at {ds_path}\n"
                 f"Run prepare_unitest.py first.")

    dataset = pickle.load(ds_path.open("rb"))
    print(f"Dataset : {ds_path}  ({len(dataset)} samples, "
          f"fingerprint {corpus_fingerprint(dataset)})")

    gen_dir = resolve_gen_dir()
    print(f"Checkpoints: {gen_dir}")
    ref = reference_order(gen_dir)

    by_id: dict[str, dict] = {}
    for s in dataset:
        by_id.setdefault(str(s.get("task_id")), s)

    missing = [t for t in ref if t not in by_id]
    if missing:
        sys.exit(
            f"ERROR: {len(missing)} checkpointed task id(s) are absent from the\n"
            f"dataset cache, e.g. {missing[:5]}.\n"
            f"The corpus membership itself differs, not just the ordering, so the\n"
            f"checkpoints cannot be extended. Regenerate the sweep from scratch."
        )

    # checkpointed functions first, in checkpoint order; then the remainder,
    # keeping their existing relative order for determinism.
    head = [by_id[t] for t in ref]
    tail = [s for s in dataset if str(s.get("task_id")) not in set(ref)]
    realigned = head + tail

    assert len(realigned) == len(dataset), "sample count changed during realignment"
    assert sample_ids(realigned)[:len(ref)] == ref, "head does not match checkpoints"

    print(f"\nRealignment:")
    print(f"  {len(head):3d} checkpointed functions → positions 0–{len(head)-1}")
    print(f"  {len(tail):3d} new functions          → positions {len(head)}–{len(realigned)-1}")
    print(f"  fingerprint {corpus_fingerprint(dataset)} → {corpus_fingerprint(realigned)}")
    src = Counter(s.get("source", "unknown") for s in realigned)
    print(f"  sources: {dict(src)}")
    print(f"\n  first 5 after realignment : {sample_ids(realigned)[:5]}")
    print(f"  first 5 in checkpoints    : {ref[:5]}")

    if args.dry_run:
        print("\n--dry-run: nothing written.")
        return 0

    backup = ds_path.with_suffix(ds_path.suffix + ".presort_backup")
    if not backup.exists():
        shutil.copy2(ds_path, backup)
        print(f"\n  original cache backed up → {backup}")
    with ds_path.open("wb") as f:
        pickle.dump(realigned, f)
    print(f"  realigned cache written  → {ds_path}")

    write_corpus_manifest(realigned)
    print(f"\nDone. `--max-samples {len(realigned)}` will now resume from the "
          f"existing {len(ref)} samples per cell.")
    print(f"Commit {CORPUS_MANIFEST} so the corpus is auditable.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
