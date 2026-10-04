#!/usr/bin/env python3
"""
check_paper_consistency.py — assert that every number in the manuscript is
derivable from the generated results, and that the results files agree with
each other.

Motivation
----------
The SQJ submission reported 300 functions per cell, 4,800 cells, ~4,090 valid
observations and ~800 GPU-hours. The experiment actually ran 30 functions per
cell (`MUTATION_SAMPLES = 30`), 480 generations, 2,233 mutants. Every count was
exactly 10x the truth, because the counts were hand-entered as placeholders and
then never refreshed. Separately, the Iterative Critique x llama3.2 cell was
printed as 1.00 when the true kill rate is 0.50 — and results_mutation.tsv
itself disagreed with mutation_report.txt for that cell.

This script makes both classes of error fail loudly:

  LAYER 1  TSV <-> report cross-check
           results_mutation.tsv vs mutation_report.txt, cell by cell. Catches a
           corrupted or hand-edited TSV.
  LAYER 2  table <-> TSV
           Every numeric cell of the manuscript's kill-rate tables must match
           the TSV within tolerance. Catches the 1.00-vs-0.50 error.
  LAYER 3  prose scale claims <-> TSV
           Sample counts, cell counts, mutant totals and benchmark splits
           asserted in prose must match what the TSV can support. Catches the
           10x inflation.

Usage
-----
    python3 check_paper_consistency.py
    python3 check_paper_consistency.py --tsv results_mutation.tsv \\
        --report results/mutation_report.txt --tex paper_draft.tex

Exit status is non-zero if any check fails, so this can gate a build.
"""
from __future__ import annotations

import argparse
import csv
import re
import sys
from collections import defaultdict
from pathlib import Path

TOL = 0.006          # kill rates are printed to 2dp
OPS = ("kill_arithmetic", "kill_boundary", "kill_comparison",
       "kill_negate_bool", "kill_return_none")

MODEL_COL_ORDER = ["llama3.2:latest", "phi4:14b", "qwen3.5:9b", "qwen3-coder:30b"]
METHOD_ROW_ORDER = ["Plain LLM", "Random RAG", "Simple RAG", "Iterative Critique"]
BASELINE_ROWS = {"pynguin", "Pynguin"}   # comparator tools, not treatment cells

failures: list[str] = []
warnings: list[str] = []


def fail(layer: str, msg: str) -> None:
    failures.append(f"[{layer}] {msg}")


def warn(layer: str, msg: str) -> None:
    warnings.append(f"[{layer}] {msg}")


# ──────────────────────────────────────────────────────────────────────
# Loaders
# ──────────────────────────────────────────────────────────────────────
def load_tsv(path: Path) -> dict[tuple[str, str], dict]:
    out: dict[tuple[str, str], dict] = {}
    with path.open() as f:
        for r in csv.DictReader(f, delimiter="\t"):
            out[(r["method"], r["model"])] = r
    return out


REPORT_ROW = re.compile(
    r"^\s{2}(?P<method>.+?)\s*\((?P<model>[^/]+)/\w+\)\s+"
    r"(?P<kr>[\d.]+)\s+(?P<sd>[\d.]+)\s+(?P<killed>\d+)\s+"
    r"(?P<total>\d+)\s+(?P<equiv>\d+)\s+(?P<n>\d+)\s*$", re.M)


def load_report(path: Path) -> dict[tuple[str, str], dict]:
    if not path.exists():
        return {}
    txt = path.read_text(errors="replace")
    out = {}
    for m in REPORT_ROW.finditer(txt):
        out[(m.group("method").strip(), m.group("model").strip())] = {
            "mean_kill_rate": float(m.group("kr")),
            "total_killed": int(m.group("killed")),
            "total_mutants": int(m.group("total")),
            "total_equivalent": int(m.group("equiv")),
            "n_samples_valid": int(m.group("n")),
        }
    return out


def tex_table(tex: str, label: str) -> list[list[str]]:
    """Return the data rows of the tabular containing \\label{label}."""
    i = tex.find("\\label{" + label + "}")
    if i == -1:
        return []
    start = tex.find("\\begin{tabular}", i)
    if start == -1:                     # label may sit after the tabular
        start = tex.rfind("\\begin{tabular}", 0, i)
    end = tex.find("\\end{tabular}", start)
    if start == -1 or end == -1:
        return []
    body = tex[start:end]
    body = body[body.find("}", body.find("{tabular}") + 9) + 1:]
    rows = []
    for raw in body.split("\\\\"):
        line = raw.strip()
        if not line or line.startswith("%"):
            continue
        for cmd in ("\\toprule", "\\midrule", "\\bottomrule", "\\hline"):
            line = line.replace(cmd, "")
        if not line.strip():
            continue
        if re.fullmatch(r"[lcrp@{}|\\\s\d.*]+", line):   # column spec, not data
            continue
        cells = [c.strip() for c in line.split("&")]
        rows.append(cells)
    return rows


def num(s: str) -> float | None:
    """Pull the first number out of a LaTeX table cell."""
    s = re.sub(r"\\textbf\{([^}]*)\}|\\mathbf\{([^}]*)\}", r"\1\2", s)
    s = s.replace("{,}", "").replace(",", "").replace("$", "")
    s = re.sub(r"\\,", "", s)
    s = re.sub(r"\^\{?[a-zA-Z\\dagger†*]+\}?", "", s)
    s = re.sub(r"\$?\\dagger\$?|†", "", s)
    m = re.search(r"-?\d+\.?\d*", s)
    return float(m.group()) if m else None


# ──────────────────────────────────────────────────────────────────────
# Layer 1 — TSV vs report
# ──────────────────────────────────────────────────────────────────────
def layer1(tsv: dict, rep: dict) -> None:
    if not rep:
        warn("L1", "mutation_report.txt not found — TSV is unverified against a second source")
        return
    for key, r in rep.items():
        if key not in tsv:
            fail("L1", f"{key} present in report but missing from TSV")
            continue
        t = tsv[key]
        for fld in ("mean_kill_rate", "total_killed", "total_mutants",
                    "total_equivalent", "n_samples_valid"):
            tv, rv = float(t[fld]), float(r[fld])
            if abs(tv - rv) > (TOL if fld == "mean_kill_rate" else 0.5):
                fail("L1", f"{key[0]} x {key[1]}: TSV {fld}={tv:g} but report says {rv:g}")
    for key in tsv:
        if key not in rep:
            # mutation_report.txt covers the 4x4 treatment matrix only; the
            # Pynguin baseline is a separate comparator reported in its own
            # section, so its absence from the report is by design.
            if key[0] in BASELINE_ROWS:
                continue
            warn("L1", f"{key} in TSV but not in report")


# ──────────────────────────────────────────────────────────────────────
# Layer 2 — manuscript tables vs TSV
# ──────────────────────────────────────────────────────────────────────
def layer2(tsv: dict, tex: str) -> None:
    # 2a. main 4x4 kill-rate matrix
    rows = tex_table(tex, "tab:killrate-matrix")
    if not rows:
        warn("L2", "tab:killrate-matrix not found in .tex")
    else:
        for cells in rows:
            method = re.sub(r"\\textbf\{([^}]*)\}", r"\1", cells[0]).strip()
            if method not in METHOD_ROW_ORDER:
                continue
            for j, model in enumerate(MODEL_COL_ORDER, start=1):
                if j >= len(cells):
                    break
                printed = num(cells[j])
                if printed is None:
                    continue
                key = (method, model)
                if key not in tsv:
                    fail("L2", f"table cites {method} x {model} with no TSV row")
                    continue
                actual = float(tsv[key]["mean_kill_rate"])
                if abs(printed - actual) > TOL:
                    fail("L2", f"tab:killrate-matrix {method} x {model}: "
                               f"printed {printed:.2f} but TSV says {actual:.4f}")

    # 2b. appendix per-cell table: n, mean, operators, mutants
    rows = tex_table(tex, "tab:percell-app")
    if not rows:
        warn("L2", "tab:percell-app not found in .tex")
        return
    for cells in rows:
        if len(cells) < 10:
            continue
        method = cells[0].strip()
        # Column specs ("llcccccccc@{}}") survive the row split; they are not data.
        if "@{" in method or method.startswith("\\"):
            continue
        model = cells[1].strip()
        if (method, model) not in tsv:
            if "Pynguin" not in method:
                warn("L2", f"appendix row {method} x {model} has no TSV row")
            continue
        t = tsv[(method, model)]
        checks = [
            ("n", num(cells[2]), float(t["n_samples_valid"]), 0.5),
            ("mean", num(cells[3]), float(t["mean_kill_rate"]), TOL),
            ("mutants", num(cells[9]), float(t["total_mutants"]), 0.5),
        ]
        for i, op in enumerate(OPS):
            printed = num(cells[4 + i])
            raw = t.get(op, "")
            actual = float(raw) if raw not in ("", None) else None
            checks.append((op, printed, actual, TOL))
        for name, printed, actual, tol in checks:
            if printed is None and actual is None:
                continue
            if printed is None or actual is None:
                warn("L2", f"appendix {method} x {model} {name}: "
                           f"printed={printed} TSV={actual} (one side empty)")
                continue
            if abs(printed - actual) > tol:
                ratio = printed / actual if actual else float("inf")
                extra = f"  [printed/actual = {ratio:.1f}x]" if actual and ratio > 1.5 else ""
                fail("L2", f"appendix {method} x {model} {name}: "
                           f"printed {printed:g} but TSV says {actual:g}{extra}")


# ──────────────────────────────────────────────────────────────────────
# Layer 3 — prose scale claims vs TSV
# ──────────────────────────────────────────────────────────────────────
def layer3(tsv: dict, tex: str) -> None:
    llm = {k: v for k, v in tsv.items() if k[0] in METHOD_ROW_ORDER}
    n_cells = len(llm)
    max_n = max((int(r["n_samples_valid"]) for r in llm.values()), default=0)
    tot_mut = sum(int(r["total_mutants"]) for r in llm.values())
    flat = re.sub(r"\s+", " ", tex)

    # Any "<N> functions/samples" claim must not exceed the largest per-cell n.
    for m in re.finditer(r"(\d[\d,{}\\]*)\s*(?:sampled\s+)?(functions|samples)\b", flat):
        v = num(m.group(1))
        if v is None or v <= max_n:
            continue
        ctx = flat[max(0, m.start() - 90):m.end() + 40]
        ratio = v / max_n if max_n else float("inf")
        msg = (f"prose claims {v:g} {m.group(2)} but the largest per-cell n is "
               f"{max_n} [{ratio:.1f}x] — …{ctx.strip()}…")
        (fail if ratio >= 2.0 else warn)("L3", msg)

    # Total-cell claims: 4 methods x 4 models x per-cell n.
    for m in re.finditer(r"(\d[\d,{}\\]*)\s*cells\s+in\s+total", flat):
        v = num(m.group(1))
        if v is None:
            continue
        plausible = 16 * max_n
        if abs(v - plausible) > 0.5:
            ratio = v / plausible if plausible else float("inf")
            (fail if ratio >= 2.0 else warn)(
                "L3", f"prose claims {v:g} cells in total; 16 x max n ({max_n}) "
                      f"= {plausible} [{ratio:.1f}x]")

    # Mutant-count claims anywhere near the true total.
    for m in re.finditer(r"(\d[\d,{}\\]*)\s*(?:total\s+)?(?:valid\s+)?mutants", flat):
        v = num(m.group(1))
        if v is None or v <= tot_mut:
            continue
        ctx = flat[max(0, m.start() - 90):m.end() + 40]
        ratio = v / tot_mut if tot_mut else float("inf")
        msg = f"prose claims {v:g} mutants but TSV totals {tot_mut} [{ratio:.1f}x] — …{ctx.strip()}…"
        (fail if ratio >= 2.0 else warn)("L3", msg)

    # GPU-hour arithmetic of the form 4x4xNx600 s
    for m in re.finditer(r"4\s*\\?times\s*4\s*\\?times\s*(\d+)\s*\\?times\s*600", flat):
        v = int(m.group(1))
        if v != max_n:
            fail("L3", f"compute estimate uses 4x4x{v}x600 s but per-cell n is {max_n}")

    print(f"  TSV basis: {n_cells} cells · max per-cell n = {max_n} · "
          f"{tot_mut} mutants total")


# ──────────────────────────────────────────────────────────────────────
def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--tsv", default="results_mutation.tsv")
    ap.add_argument("--report", default="results/mutation_report.txt")
    ap.add_argument("--tex", default="paper_draft.tex")
    args = ap.parse_args()

    tsv_p, rep_p, tex_p = Path(args.tsv), Path(args.report), Path(args.tex)
    for p in (tsv_p, tex_p):
        if not p.exists():
            print(f"ERROR: {p} not found", file=sys.stderr)
            return 2

    tsv = load_tsv(tsv_p)
    rep = load_report(rep_p)
    tex = tex_p.read_text(errors="replace")

    print(f"Checking {tex_p.name} against {tsv_p.name}"
          f"{' and ' + rep_p.name if rep else ''}\n")

    layer1(tsv, rep)
    layer2(tsv, tex)
    layer3(tsv, tex)

    print()
    if warnings:
        print(f"── {len(warnings)} warning(s) ──")
        for w in warnings[:20]:
            print("  " + w)
        if len(warnings) > 20:
            print(f"  … and {len(warnings)-20} more")
        print()
    if failures:
        print(f"── {len(failures)} FAILURE(S) ──")
        for f in failures:
            print("  " + f)
        print(f"\n{'='*70}\n  INCONSISTENT — do not submit\n{'='*70}")
        return 1
    print(f"{'='*70}\n  CONSISTENT — every checked number traces to the TSV\n{'='*70}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
