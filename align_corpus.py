#!/usr/bin/env python3
"""
align_corpus.py - DEPRECATED. DO NOT RUN. Superseded by verify_corpus.py.

This was built on a wrong diagnosis and is harmful if executed.

The cached dataset is written sorted by prepare_unitest.py while the mutation
checkpoints are in an unsorted order, and that looked like drift between the two.
It is not. mutation_testing.main() applies

    random.Random(42).shuffle(dataset)

immediately after loading the cache (mirroring train_unitest.py), and that
seeded shuffle is exactly what produced the checkpoint ordering. Verified:
16 of 16 checkpoints reproduce from it, position by position. The pipeline was
already deterministic and the cache must not be touched.

What this script did was reorder the cache file so its raw order matched the
checkpoints. main() then shuffled that reordered cache, producing a third
ordering that matched nothing, and the corpus guard in regenerate_tests would
have aborted the sweep.

Use verify_corpus.py instead: it changes nothing and only confirms the
invariant, writing the corpus manifest as a side effect.

Retained as a record of the mistake rather than deleted, so the commit history
and the notebook's earlier instructions remain explicable.
"""
from __future__ import annotations

import sys

MESSAGE = (
    "align_corpus.py is DEPRECATED and must not be run.\n"
    "It reorders the dataset cache, which breaks the seeded shuffle in\n"
    "mutation_testing.main() and would abort the sweep.\n\n"
    "Run this instead:\n"
    "    python3 verify_corpus.py --max-samples 100 --write-manifest\n"
)

if __name__ == "__main__":
    print(__doc__, file=sys.stderr)
    sys.exit(MESSAGE)
