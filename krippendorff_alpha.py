#!/usr/bin/env python3
"""
krippendorff_alpha.py — Krippendorff's alpha, self-contained and self-testing.

Why this exists
---------------
The manuscript reports ordinal Krippendorff's alpha as the primary three-rater
agreement statistic. The value that was reported could not be reproduced from
this repository: there was no code for it, and the `krippendorff` PyPI package
is not installable on the analysis machine. A reliability statistic that cannot
be recomputed from the artifacts is exactly the kind of number this rebuild
exists to eliminate.

So rather than depend on an external package, this module implements the
statistic directly and validates itself against the worked example published in
Krippendorff's own methodological note, for which the correct answers are
documented. Run the file to execute that validation plus the project's data:

    python3 krippendorff_alpha.py              # self-test, then the study data
    python3 krippendorff_alpha.py --selftest   # validation only

The computation follows the coincidence-matrix formulation:

    alpha = 1 - D_o / D_e

    D_o = (1/n)         * sum_c sum_k  o_ck     * delta^2(c, k)
    D_e = (1/(n(n-1)))  * sum_c sum_k  n_c n_k  * delta^2(c, k)

where o_ck is the coincidence matrix (ordered pairs within each unit, each
weighted 1/(m_u - 1) for a unit rated by m_u coders), n_c are its marginals,
and n = sum_c n_c is the number of pairable values.

Difference functions:
    nominal   delta^2 = 0 if c == k else 1
    ordinal   delta^2 = ( sum_{g=c..k} n_g - (n_c + n_k)/2 )^2
    interval  delta^2 = (c - k)^2
    ratio     delta^2 = ((c - k) / (c + k))^2

Note that the ordinal metric depends on the marginals n_g, so it is defined
only with respect to the observed value distribution — it is not a fixed
distance like the interval metric. This is the usual source of disagreement
between implementations and the reason the metric must be named whenever an
ordinal alpha is reported.
"""
from __future__ import annotations

import argparse
import glob
import itertools
import math
import os
import sys
from pathlib import Path

import numpy as np


# ──────────────────────────────────────────────────────────────────────
# core
# ──────────────────────────────────────────────────────────────────────
def coincidence_matrix(reliability_data: np.ndarray) -> tuple[np.ndarray, list]:
    """Build the coincidence matrix from a (raters x units) array.

    Missing values are np.nan. Units rated by fewer than two coders carry no
    pairable values and are dropped, per Krippendorff.
    """
    values = sorted({v for v in reliability_data.ravel() if not np.isnan(v)})
    index = {v: i for i, v in enumerate(values)}
    V = len(values)
    o = np.zeros((V, V), dtype=float)

    for u in range(reliability_data.shape[1]):
        col = [v for v in reliability_data[:, u] if not np.isnan(v)]
        m_u = len(col)
        if m_u < 2:
            continue
        for a, b in itertools.permutations(col, 2):
            o[index[a], index[b]] += 1.0 / (m_u - 1)
    return o, values


def _delta2(metric: str, values: list, marginals: np.ndarray):
    """Return a delta^2(i, j) function over value *indices*."""
    if metric == "nominal":
        return lambda i, j: 0.0 if i == j else 1.0
    if metric == "interval":
        return lambda i, j: (values[i] - values[j]) ** 2
    if metric == "ratio":
        return lambda i, j: (((values[i] - values[j]) /
                              (values[i] + values[j])) ** 2
                             if (values[i] + values[j]) != 0 else 0.0)
    if metric == "ordinal":
        def d(i, j):
            lo, hi = (i, j) if i <= j else (j, i)
            s = marginals[lo:hi + 1].sum() - (marginals[lo] + marginals[hi]) / 2.0
            return s ** 2
        return d
    raise ValueError(f"unknown metric: {metric}")


def alpha(reliability_data, level_of_measurement: str = "ordinal") -> float:
    """Krippendorff's alpha for a (raters x units) array; np.nan for missing."""
    data = np.asarray(reliability_data, dtype=float)
    o, values = coincidence_matrix(data)
    V = len(values)
    if V < 2:
        return float("nan")          # no variation: alpha undefined
    marginals = o.sum(axis=1)
    n = marginals.sum()
    d2 = _delta2(level_of_measurement, values, marginals)

    D_o = sum(o[i, j] * d2(i, j) for i in range(V) for j in range(V)) / n
    D_e = sum(marginals[i] * marginals[j] * d2(i, j)
              for i in range(V) for j in range(V)) / (n * (n - 1))
    if D_e == 0:
        return float("nan")
    return 1.0 - D_o / D_e


# ──────────────────────────────────────────────────────────────────────
# validation against Krippendorff's published worked example
# ──────────────────────────────────────────────────────────────────────
#
# This is the example distributed with the reference implementation and taken
# from Krippendorff's methodological note. Three coders, fifteen units, heavy
# missingness. The documented answers are nominal 0.691 and interval 0.811.
CANONICAL = np.array([
    [np.nan, np.nan, np.nan, np.nan, np.nan, 3, 4, 1, 2, 1, 1, 3, 3, np.nan, 3],
    [1, np.nan, 2, 1, 3, 3, 4, 3, np.nan, np.nan, np.nan, np.nan, np.nan, np.nan, np.nan],
    [np.nan, np.nan, 2, 1, 3, 4, 4, np.nan, 2, 1, 1, 3, 3, np.nan, 4],
])
EXPECTED = {"nominal": 0.691, "interval": 0.811}


def selftest(tol: float = 0.0005) -> bool:
    print("Validating against Krippendorff's published worked example")
    print(f"  data: {CANONICAL.shape[0]} coders x {CANONICAL.shape[1]} units, "
          f"{int(np.isnan(CANONICAL).sum())} missing\n")
    ok = True
    for metric, expected in EXPECTED.items():
        got = alpha(CANONICAL, metric)
        good = abs(got - expected) < tol
        ok &= good
        print(f"  {'PASS' if good else 'FAIL'}  {metric:9} "
              f"expected {expected:.3f}   got {got:.6f}")
    for metric in ("ordinal", "ratio"):
        print(f"  ----  {metric:9} {'':17} got {alpha(CANONICAL, metric):.6f}"
              "   (no published value to check against)")
    print()
    # degenerate cases
    perfect = np.array([[1., 2., 3., 4.], [1., 2., 3., 4.]])
    a = alpha(perfect, "ordinal")
    print(f"  {'PASS' if abs(a - 1.0) < 1e-12 else 'FAIL'}  "
          f"perfect agreement -> alpha = {a:.6f} (expect 1.0)")
    ok &= abs(a - 1.0) < 1e-12
    return ok


# ──────────────────────────────────────────────────────────────────────
# the study's own data
# ──────────────────────────────────────────────────────────────────────
DIMS = ["human_test_idiom", "human_correctness", "human_completeness"]


def study_data(ann_dir="human_eval_annotations"):
    import pandas as pd
    frames = {os.path.basename(f)[:-4]: pd.read_csv(f).set_index("sample_id")
              for f in sorted(glob.glob(os.path.join(ann_dir, "*.csv")))}
    if not frames:
        print(f"  no annotation CSVs under {ann_dir}/", file=sys.stderr)
        return
    common = sorted(set.intersection(*[set(f.index) for f in frames.values()]))
    raters = list(frames)
    print(f"Study data: {len(raters)} annotators {raters}, "
          f"{len(common)} units rated by all\n")
    print(f"  {'dimension':22}" + "".join(f"{m:>12}" for m in
                                          ("ordinal", "interval", "nominal")))
    for dim in DIMS:
        M = np.array([[frames[a].loc[u, dim] for u in common] for a in raters],
                     dtype=float)
        row = f"  {dim:22}"
        for metric in ("ordinal", "interval", "nominal"):
            row += f"{alpha(M, metric):>12.3f}"
        print(row)
    print("\n  Report the ordinal column; name the metric in the manuscript.")


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--selftest", action="store_true",
                    help="run the validation only and exit")
    ap.add_argument("--annotations", default="human_eval_annotations")
    args = ap.parse_args()

    ok = selftest()
    if not ok:
        print("SELF-TEST FAILED — do not use these values.", file=sys.stderr)
        return 1
    if args.selftest:
        return 0
    print("=" * 62)
    study_data(args.annotations)
    return 0


if __name__ == "__main__":
    sys.exit(main())
