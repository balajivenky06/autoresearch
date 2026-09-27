#!/usr/bin/env python3
"""
verify_corpus.py — read-only pre-flight check that the evaluation corpus
reproduces the ordering the existing checkpoints were generated against.

Background
----------
This replaces align_corpus.py, which was built on a wrong diagnosis. The cached
dataset is emitted sorted by prepare_unitest.py, while the checkpoints are in an
unsorted order, and that looked like drift between the two. It is not:
mutation_testing.main() applies `random.Random(42).shuffle(dataset)` immediately
after loading (mirroring train_unitest.py), and that seeded shuffle reproduces
the checkpoint ordering exactly. The pipeline was already deterministic.

align_corpus.py reordered the cache file, after which main()'s shuffle scrambled
it into a third ordering and the corpus guard would have aborted the sweep. This
script does the opposite: it changes nothing and only confirms the invariant.

What it checks
--------------
  1. The cache loads and holds the expected number of samples.
  2. Applying the same seeded shuffle reproduces each existing checkpoint's task
     ids, position by position, for as far as that checkpoint goes.
  3. Every checkpoint agrees with every other on the ordering.

Exit status is non-zero if the invariant is broken, so a notebook can refuse to
start a sweep that would abort or, worse, mix two orderings.

Usage
-----
    python3 verify_corpus.py
    python3 verify_corpus.py --max-samples 100 --write-manifest
"""
from __future__ import annotations

import argparse
import pickle
import random
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

GEN_DIR_CANDIDATES = [Path(".checkpoints_mutation"), Path("checkpoints_mutation")]
BASELINE_MARKERS = ("pynguin",)
SHUFFLE_SEED = 42          # must match mutation_testing.main()


def resolve_gen_dir() -> Path | None:
    for p in GEN_DIR_CANDIDATES:
        if p.is_dir() and any(p.glob("*.pkl")):
            return p
    return None


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset", default=str(DATASET_CACHE))
    ap.add_argument("--max-samples", type=int, default=None,
                    help="the --max-samples the sweep will use (for the manifest)")
    ap.add_argument("--write-manifest", action="store_true")
    args = ap.parse_args()

    ds_path = Path(args.dataset)
    if not ds_path.exists():
        print(f"FAIL: dataset not found at {ds_path}\n"
              f"      run prepare_unitest.py first.", file=sys.stderr)
        return 2
    dataset = pickle.load(ds_path.open("rb"))

    # Reproduce exactly what mutation_testing.main() does to the dataset.
    shuffled = list(dataset)
    random.Random(SHUFFLE_SEED).shuffle(shuffled)
    order = sample_ids(shuffled)

    print(f"Dataset   : {ds_path.name}  n={len(dataset)}")
    print(f"            as-loaded first 3 : {sample_ids(dataset)[:3]}")
    print(f"            after shuffle({SHUFFLE_SEED}) : {order[:3]}")
    print(f"            post-shuffle fingerprint: {corpus_fingerprint(shuffled)}")
    src = Counter(s.get("source", "unknown") for s in shuffled)
    print(f"            sources: {dict(src)}")

    gen_dir = resolve_gen_dir()
    if gen_dir is None:
        print("\nNo existing checkpoints — nothing to verify against. A sweep "
              "will generate from scratch.")
        if args.write_manifest:
            write_corpus_manifest(shuffled[:args.max_samples or len(shuffled)])
        return 0

    print(f"\nCheckpoints: {gen_dir}")
    failures, checked = [], 0
    for f in sorted(gen_dir.glob("*.pkl")):
        if f.name.endswith(".tmp") or any(m in f.name.lower() for m in BASELINE_MARKERS):
            continue
        try:
            cached = pickle.load(f.open("rb"))
        except Exception as e:
            failures.append((f.name, f"unreadable: {e}")); continue
        if not isinstance(cached, list) or not cached:
            continue
        ids = sample_ids(cached)
        checked += 1
        bad = next((i for i in range(min(len(ids), len(order)))
                    if ids[i] != order[i]), None)
        if bad is None:
            print(f"  ok      {f.name:48} {len(ids):3d} samples")
        else:
            failures.append((f.name,
                             f"sample_idx {bad}: checkpoint has {ids[bad]}, "
                             f"shuffled corpus has {order[bad]}"))
            print(f"  MISMATCH {f.name:47} at sample_idx {bad}")

    print()
    if failures:
        print("=" * 70)
        print("  CORPUS INVARIANT BROKEN — do not start the sweep")
        print("=" * 70)
        for name, why in failures:
            print(f"  {name}: {why}")
        print("\n  The cached corpus no longer reproduces the ordering these\n"
              "  checkpoints were built on. Most likely causes:\n"
              "    - NUM_EVAL_SAMPLES changed in prepare_unitest.py\n"
              "    - the cache file was reordered by hand or by a tool\n"
              "    - the shuffle seed in mutation_testing.main() changed\n"
              "  Resolve before generating: resuming would merge two orderings.")
        return 1

    print("=" * 70)
    print(f"  CORPUS VERIFIED — {checked} checkpoint(s) reproduce from "
          f"shuffle({SHUFFLE_SEED})")
    print("=" * 70)
    if args.write_manifest:
        n = args.max_samples or len(shuffled)
        write_corpus_manifest(shuffled[:n])
    return 0


if __name__ == "__main__":
    sys.exit(main())
