#!/usr/bin/env python3
"""check_tables.py — structural validation of every tabular in the manuscript.

Checks, per table: the column specification parses, every data row has exactly
as many cells as the spec declares, booktabs rules are present, and nothing
stray sits after the final row. Brace-matched throughout, because LaTeX column
specs contain braces (@{}, p{...}) that defeat a non-greedy regex.
"""
import re, sys
from pathlib import Path

def take_braced(src, i):
    """src[i] must be '{'; return (contents, index after the closing brace)."""
    assert src[i] == "{"
    d, j = 1, i + 1
    while d:
        if src[j] == "\\": j += 2; continue
        if src[j] == "{": d += 1
        elif src[j] == "}": d -= 1
        j += 1
    return src[i+1:j-1], j

def env_end(src, start, env):
    d, i = 1, start
    while d:
        nb, ne = src.find("\\begin{"+env+"}", i), src.find("\\end{"+env+"}", i)
        if ne == -1: return -1
        if nb != -1 and nb < ne: d += 1; i = nb + len("\\begin{"+env+"}")
        else: d -= 1; i = ne + len("\\end{"+env+"}")
    return i

def main() -> int:
    raw = Path(sys.argv[1] if len(sys.argv) > 1 else "paper_draft.tex").read_text()
    # Blank out comments: a commented-out \begin{tabular} would otherwise be
    # parsed as a real table and the scan would run past every genuine end.
    s = re.sub(r"(?<!\\)%[^\n]*", "", raw)
    problems, rowsum = [], []
    for m in re.finditer(r"\\begin\{(tabular\*?|tabularx)\}", s):
        env = m.group(1)
        end = env_end(s, m.end(), env)
        seg = s[m.start():end]
        i = m.end()
        if env in ("tabularx", "tabular*"):
            _, i = take_braced(s, s.index("{", i))      # width argument
        spec, i = take_braced(s, s.index("{", i))
        ncol = len(re.findall(r"[lcr]|[pmb]\{|X", re.sub(r"@\{(?:[^{}]|\{[^}]*\})*\}", "", spec)))
        lab = re.search(r"\\label\{(tab:[^}]*)\}", s[max(0, m.start()-1200):end+600])
        lab = lab.group(1) if lab else "(unlabelled)"

        body = s[i:end - len("\\end{"+env+"}")]
        for _c in ("\\toprule", "\\midrule", "\\bottomrule", "\\hline"):
            body = body.replace(_c, "")
        body = re.sub(r"\\cmidrule(\([^)]*\))?\{[^}]*\}", "", body)
        for cmd in ("\\toprule", "\\midrule", "\\bottomrule", "\\hline"):
            body = body.replace(cmd, "")
        body = re.sub(r"\\cmidrule(\([^)]*\))?\{[^}]*\}", "", body)
        parts = body.split("\\\\")
        rows, bad = [], []
        for raw in parts[:-1]:
            line = raw
            for cmd in ("\\toprule", "\\midrule", "\\bottomrule", "\\hline"):
                line = line.replace(cmd, "")
            line = re.sub(r"\\cmidrule(\([^)]*\))?\{[^}]*\}", "", line)
            if not line.strip(): continue
            rows.append(line)
            n = len(re.split(r"(?<!\\)&", line))
            if n != ncol: bad.append((len(rows), n))
        tail = parts[-1]
        for cmd in ("\\toprule", "\\midrule", "\\bottomrule", "\\hline"):
            tail = tail.replace(cmd, "")
        rowsum.append((lab, env, spec, ncol, len(rows), bad))
        for r, n in bad:
            problems.append(f"{lab}: row {r} has {n} cells, spec declares {ncol}")
        if tail.strip():
            problems.append(f"{lab}: stray content after last row: {tail.strip()[:50]!r}")
        for rule in ("\\toprule", "\\bottomrule"):
            if rule not in seg: problems.append(f"{lab}: missing {rule}")

    print(f"{'label':30}{'env':10}{'cols':>5}{'rows':>6}  column spec")
    for lab, env, spec, ncol, nrow, bad in rowsum:
        print(f"{lab:30}{env:10}{ncol:>5}{nrow:>6}  {spec}" + ("   <<< BAD" if bad else ""))
    print()
    if problems:
        print(f"{len(problems)} PROBLEM(S):")
        for p in problems: print("  " + p)
        return 1
    print(f"All {len(rowsum)} tables structurally valid.")
    return 0

if __name__ == "__main__":
    sys.exit(main())
