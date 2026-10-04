#!/usr/bin/env python3
"""
verify_citations.py — resolve every BibTeX entry against Crossref and report
any entry whose recorded metadata does not match the real publication.

Motivation
----------
The SQJ submission was rejected with the Editor-in-Chief noting that a
reference had been described as being about *test* generation when the real
paper is about *text* generation, and that two entries still carried literal
"TODO verify" notes. This script makes that class of error detectable
mechanically, before submission, rather than by a reviewer afterwards.

What it checks, per entry
------------------------
  PLACEHOLDER      any field contains TODO / verify / FIXME / ??? text
  NO_DOI           no DOI recorded (cannot be machine-verified)
  DOI_UNRESOLVED   DOI recorded but Crossref returns no such work
  TITLE_MISMATCH   recorded title differs from Crossref title
  YEAR_MISMATCH    recorded year differs from Crossref issued year
  AUTHOR_MISMATCH  recorded first-author surname differs from Crossref
  VENUE_NOTE       informational: recorded venue vs Crossref container

It also pulls every \\cite context out of the .tex so the *prose description*
of each reference can be eyeballed against the real title/abstract. The
Huang & Huang error was a description error, not a metadata error, so
metadata checking alone would not have caught it.

Usage
-----
    python3 verify_citations.py
    python3 verify_citations.py --bib references.bib --tex paper_draft.tex
    python3 verify_citations.py --offline      # skip network, placeholder scan only

Exit status is non-zero if any BLOCKER-severity problem is found, so this can
gate a build.
"""
from __future__ import annotations

import argparse
import json
import re
import sys
import time
import urllib.error
import urllib.parse
import urllib.request
from dataclasses import dataclass, field
from difflib import SequenceMatcher
from pathlib import Path

CROSSREF = "https://api.crossref.org/works/"
DATACITE = "https://api.datacite.org/dois/"
MAILTO = "balajivenky06@gmail.com"
UA = f"citation-verifier/1.0 (mailto:{MAILTO})"

PLACEHOLDER_RE = re.compile(r"\bTODO\b|\bFIXME\b|\?\?\?|\bverify\b|placeholder", re.I)
TITLE_SIM_THRESHOLD = 0.82

# Registries sometimes store an abbreviated title. ACM, for instance, registers
# Methods2Test as just "Methods2Test" and EvoSuite as "EvoSuite", so a correct
# citation scores ~0.3 on title similarity. These are keyed by DOI and were each
# confirmed by hand against authors, venue and year before being listed. Anything
# not on this list still has to match.
REGISTRY_ABBREVIATED = {
    "10.1145/3524842.3528009": "Methods2Test",   # Tufano et al., MSR 2022
    "10.1145/2025113.2025179": "EvoSuite",       # Fraser & Arcuri, ESEC/FSE 2011
}

# Problems that should block a submission outright.
BLOCKERS = {"PLACEHOLDER", "DOI_UNRESOLVED", "TITLE_MISMATCH", "AUTHOR_MISMATCH"}

# ── Topic-consistency checking ────────────────────────────────────────
# The Huang & Huang failure was NOT a metadata error: the bib entry is
# accurate ("...Retrieval-Augmented *Text* Generation..."). The error lived in
# the prose, which described it as a *test*-generation study evaluated on
# HumanEval. Metadata checks cannot catch that, so we additionally check
# whether topic claims made near a \cite are supported by the real title.
#
# Each marker maps to the tokens that would have to appear in a real title
# for the claim to be plausible. High recall by design — output is a review
# queue, not a verdict.
TOPIC_MARKERS: dict[str, tuple[str, ...]] = {
    "test generation": ("test", "testing"),
    "unit test": ("test", "testing"),
    "test-generation": ("test", "testing"),
    "mutation testing": ("mutation", "mutant"),
    "mutation score": ("mutation", "mutant"),
    "kill rate": ("mutation", "mutant"),
    "humaneval": ("humaneval", "code", "program"),
    "mbpp": ("mbpp", "program", "code"),
    "coverage": ("coverage", "test", "testing"),
    "search-based": ("search", "sbst", "genetic", "evolutionary"),
    "sbst": ("search", "sbst", "genetic", "evolutionary"),
}


# ──────────────────────────────────────────────────────────────────────
# BibTeX parsing (regex-based; avoids a bibtexparser dependency)
# ──────────────────────────────────────────────────────────────────────
@dataclass
class Entry:
    key: str
    etype: str
    fields: dict[str, str] = field(default_factory=dict)
    raw: str = ""

    def get(self, *names: str) -> str:
        for n in names:
            if n in self.fields:
                return self.fields[n]
        return ""


def _strip_braces(v: str) -> str:
    v = v.strip().rstrip(",").strip()
    while len(v) >= 2 and ((v[0] == "{" and v[-1] == "}") or (v[0] == '"' and v[-1] == '"')):
        v = v[1:-1].strip()
    return v


def parse_bib(path: Path) -> list[Entry]:
    text = path.read_text(encoding="utf-8", errors="replace")
    entries: list[Entry] = []
    # Locate each @type{key, ... } block by brace matching.
    for m in re.finditer(r"@(\w+)\s*\{\s*([^,\s]+)\s*,", text):
        etype, key = m.group(1).lower(), m.group(2)
        i = text.index("{", m.start())
        depth, j = 0, i
        while j < len(text):
            if text[j] == "{":
                depth += 1
            elif text[j] == "}":
                depth -= 1
                if depth == 0:
                    break
            j += 1
        body = text[i + 1 : j]
        e = Entry(key=key, etype=etype, raw=body)
        # field = value pairs at depth 0
        for fm in re.finditer(r"(\w+)\s*=\s*", body):
            name = fm.group(1).lower()
            start = fm.end()
            if start >= len(body):
                continue
            if body[start] in "{\"":
                open_ch = body[start]
                close_ch = "}" if open_ch == "{" else '"'
                d, k = 0, start
                while k < len(body):
                    if body[k] == open_ch and (open_ch == "{" or k == start):
                        d += 1
                    elif body[k] == close_ch:
                        d -= 1
                        if d == 0:
                            break
                    k += 1
                val = body[start : k + 1]
            else:
                nxt = body.find(",", start)
                val = body[start : nxt if nxt != -1 else len(body)]
            e.fields[name] = _strip_braces(val)
        entries.append(e)
    return entries


# ──────────────────────────────────────────────────────────────────────
# Crossref
# ──────────────────────────────────────────────────────────────────────
def _get_json(url: str, retries: int = 2) -> dict | None:
    for attempt in range(retries + 1):
        try:
            req = urllib.request.Request(url, headers={"User-Agent": UA})
            with urllib.request.urlopen(req, timeout=20) as r:
                return json.load(r)
        except urllib.error.HTTPError as ex:
            if ex.code in (404, 400):
                return None
            if attempt == retries:
                return None
            time.sleep(1.5 * (attempt + 1))
        except Exception:
            if attempt == retries:
                return None
            time.sleep(1.5 * (attempt + 1))
    return None


def resolve_doi(doi: str) -> tuple[dict | None, str]:
    """Crossref first; fall back to DataCite (arXiv and many non-Crossref DOIs).

    Returns (normalised-record, registry-name). The normalised record uses
    Crossref's shape so downstream comparisons stay uniform.
    """
    q = urllib.parse.quote(doi.strip(), safe="/:._-()")
    j = _get_json(CROSSREF + q)
    if j and j.get("message"):
        return j["message"], "crossref"

    j = _get_json(DATACITE + q)
    if j and j.get("data"):
        at = j["data"].get("attributes", {}) or {}
        titles = [t.get("title", "") for t in (at.get("titles") or []) if t.get("title")]
        creators = [
            {"family": (c.get("familyName") or c.get("name", "").split()[-1] if c.get("name") else "")}
            for c in (at.get("creators") or [])
        ]
        yr = at.get("publicationYear")
        return (
            {
                "title": titles or [""],
                "author": creators,
                "issued": {"date-parts": [[yr]]} if yr else {},
                "container-title": [at.get("publisher") or ""],
                "type": (at.get("types") or {}).get("resourceTypeGeneral", "dataset/preprint"),
            },
            "datacite",
        )
    return None, ""


# LaTeX accent commands and braces, e.g. Sch{\"a}fer -> Schafer
_ACCENT_RE = re.compile(r"\\[`'\"^~=.uvHtcdbk]\s*\{?([a-zA-Z])\}?|\\[a-zA-Z]+\s*\{?([a-zA-Z])\}?")


def de_latex(s: str) -> str:
    s = _ACCENT_RE.sub(lambda m: m.group(1) or m.group(2) or "", s)
    s = s.replace("{", "").replace("}", "").replace("\\", "")
    # strip remaining diacritics
    import unicodedata

    s = unicodedata.normalize("NFKD", s)
    return "".join(c for c in s if not unicodedata.combining(c))


def norm_title(s: str) -> str:
    s = de_latex(s)
    return " ".join(re.sub(r"[^a-z0-9 ]", " ", s.lower()).split())


def first_surname(author_field: str) -> str:
    a = de_latex(author_field).split(" and ")[0].strip()
    if "," in a:
        return a.split(",")[0].strip().lower()
    parts = a.split()
    return parts[-1].strip().lower() if parts else ""


def topic_conflicts(contexts: list[str], real_title: str) -> list[str]:
    """Topic claims appearing near a \\cite that the real title does not support."""
    t = norm_title(real_title)
    if not t:
        return []
    out: list[str] = []
    for ctx in contexts:
        c = " ".join(re.sub(r"[^a-z0-9 -]", " ", de_latex(ctx).lower()).split())
        for marker, required in TOPIC_MARKERS.items():
            if marker in c and not any(tok in t for tok in required):
                out.append(marker)
    return sorted(set(out))


# ──────────────────────────────────────────────────────────────────────
# \cite context extraction
# ──────────────────────────────────────────────────────────────────────
_SENT_END = re.compile(r"(?<=[.!?])\s+(?=[A-Z\\])")


def cite_contexts(tex: Path, keys: set[str]) -> tuple[dict[str, list[str]], dict[str, list[str]]]:
    """Return (sentences, wide_contexts) per citation key.

    `sentences` holds only the sentence containing the \\cite — this is what the
    topic check runs against, so markers belonging to a neighbouring citation
    do not produce false positives. `wide_contexts` is a larger window kept
    purely for human reading in the report.
    """
    if not tex.exists():
        return {}, {}
    t = re.sub(r"\s+", " ", tex.read_text(encoding="utf-8", errors="replace"))
    sents = _SENT_END.split(t)

    # character offset of each sentence start, to map a match back to a sentence
    offsets, pos = [], 0
    for s in sents:
        offsets.append(pos)
        pos += len(s) + 1

    def sentence_at(idx: int) -> str:
        lo, hi = 0, len(offsets) - 1
        while lo < hi:
            mid = (lo + hi + 1) // 2
            if offsets[mid] <= idx:
                lo = mid
            else:
                hi = mid - 1
        return sents[lo]

    sent_out: dict[str, list[str]] = {k: [] for k in keys}
    wide_out: dict[str, list[str]] = {k: [] for k in keys}
    for m in re.finditer(r"\\cite[a-zA-Z]*\*?(?:\[[^\]]*\])*\{([^}]*)\}", t):
        s = sentence_at(m.start()).strip()
        a, b = max(0, m.start() - 300), min(len(t), m.end() + 150)
        wide = t[a:b].strip()
        for c in (x.strip() for x in m.group(1).split(",")):
            if c in sent_out:
                sent_out[c].append(s)
                wide_out[c].append(wide)
    return sent_out, wide_out


# ──────────────────────────────────────────────────────────────────────
# Main
# ──────────────────────────────────────────────────────────────────────
def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--bib", default="references.bib")
    ap.add_argument("--tex", default="paper_draft.tex")
    ap.add_argument("--out", default="citation_audit.md")
    ap.add_argument("--offline", action="store_true")
    ap.add_argument("--sleep", type=float, default=0.3)
    args = ap.parse_args()

    bib_path, tex_path = Path(args.bib), Path(args.tex)
    if not bib_path.exists():
        print(f"ERROR: {bib_path} not found", file=sys.stderr)
        return 2

    entries = parse_bib(bib_path)
    ctx, wide = cite_contexts(tex_path, {e.key for e in entries})
    print(f"Parsed {len(entries)} entries from {bib_path}")
    if tex_path.exists():
        ncited = sum(1 for k, v in ctx.items() if v)
        print(f"Found \\cite contexts for {ncited}/{len(entries)} entries in {tex_path}")
    print()

    rows = []
    for i, e in enumerate(entries, 1):
        problems: list[str] = []
        notes: list[str] = []

        title = e.get("title")
        year = e.get("year", "date")
        doi = e.get("doi")
        authors = e.get("author", "editor")
        venue = e.get("journal", "booktitle", "publisher")

        # 1. placeholder scan (works offline)
        ph_fields = [f"{k}={v}" for k, v in e.fields.items() if PLACEHOLDER_RE.search(v)]
        if ph_fields:
            problems.append("PLACEHOLDER")
            notes.append("placeholder text in: " + "; ".join(ph_fields)[:300])

        cr, registry = None, ""
        if not doi:
            problems.append("NO_DOI")
        elif not args.offline:
            cr, registry = resolve_doi(doi)
            time.sleep(args.sleep)
            if cr is None:
                problems.append("DOI_UNRESOLVED")
                notes.append(f"neither Crossref nor DataCite has a record for doi={doi}")
            elif registry == "datacite":
                notes.append("resolved via DataCite (preprint / non-Crossref registrant)")

        if cr:
            cr_title = (cr.get("title") or [""])[0]
            if cr_title and title:
                sim = SequenceMatcher(None, norm_title(title), norm_title(cr_title)).ratio()
                if doi and doi.strip() in REGISTRY_ABBREVIATED and sim < TITLE_SIM_THRESHOLD:
                    notes.append(f"registry stores an abbreviated title "
                                 f"({cr_title!r}); verified by hand against authors, "
                                 f"venue and year")
                elif sim < TITLE_SIM_THRESHOLD:
                    problems.append("TITLE_MISMATCH")
                    notes.append(f"bib='{title[:90]}' vs crossref='{cr_title[:90]}' (sim={sim:.2f})")

            dp = (cr.get("issued") or {}).get("date-parts") or [[None]]
            cr_year = dp[0][0]
            if year and cr_year and str(cr_year) != str(year).strip():
                problems.append("YEAR_MISMATCH")
                notes.append(f"bib year={year} vs crossref={cr_year}")

            cr_auth = [de_latex(a.get("family", "")).lower() for a in (cr.get("author") or [])]
            bib_first = first_surname(authors)
            if bib_first and cr_auth and bib_first not in cr_auth:
                problems.append("AUTHOR_MISMATCH")
                notes.append(f"bib first author '{bib_first}' not among crossref {cr_auth[:6]}")

            cr_venue = (cr.get("container-title") or [""])[0]
            if cr_venue:
                notes.append(f"registry venue: {cr_venue[:110]}")
            notes.append(f"registry type: {cr.get('type')}")

        # Topic-consistency: does the prose near each \cite claim something the
        # real title cannot support? This is the Huang & Huang failure mode.
        real_title = ((cr.get("title") or [""])[0] if cr else "") or title
        conflicts = topic_conflicts(ctx.get(e.key, []), real_title)  # sentence-scoped
        if conflicts:
            problems.append("DESCRIPTION_SUSPECT")
            notes.append(
                "prose near \\cite claims " + ", ".join(f"'{c}'" for c in conflicts)
                + f" but real title is '{real_title[:95]}'"
            )

        severity = "BLOCKER" if set(problems) & BLOCKERS else ("WARN" if problems else "OK")
        rows.append(
            dict(
                key=e.key, etype=e.etype, severity=severity, problems=problems,
                notes=notes, title=title, doi=doi, year=year,
                crossref_title=(cr.get("title") or [""])[0] if cr else "",
                n_cites=len(ctx.get(e.key, [])), contexts=wide.get(e.key, []),
                sentences=ctx.get(e.key, []),
            )
        )
        flag = {"OK": "  ok", "WARN": "WARN", "BLOCKER": "BLOCK"}[severity]
        print(f"[{i:2d}/{len(entries)}] {flag}  {e.key:32} {','.join(problems) or '-'}")

    blockers = [r for r in rows if r["severity"] == "BLOCKER"]
    warns = [r for r in rows if r["severity"] == "WARN"]
    uncited = [r for r in rows if r["n_cites"] == 0]

    # ── report ──
    L: list[str] = ["# Citation audit", ""]
    L.append(f"- Entries: **{len(rows)}**")
    L.append(f"- Blockers: **{len(blockers)}** · Warnings: **{len(warns)}** · Clean: **{len(rows)-len(blockers)-len(warns)}**")
    L.append(f"- Entries never \\cite'd in `{tex_path.name}`: **{len(uncited)}**")
    L.append(f"- Mode: {'OFFLINE (placeholder scan only)' if args.offline else 'Crossref-verified'}")
    L.append("")

    for label, group in (("Blockers", blockers), ("Warnings", warns)):
        if not group:
            continue
        L += [f"## {label}", ""]
        for r in group:
            L.append(f"### `{r['key']}` — {', '.join(r['problems'])}")
            L.append(f"- bib title: {r['title']}")
            if r["crossref_title"]:
                L.append(f"- crossref title: **{r['crossref_title']}**")
            L.append(f"- doi: `{r['doi'] or '(none)'}`  · cited {r['n_cites']}x")
            for n in r["notes"]:
                L.append(f"- {n}")
            for c in r["contexts"][:3]:
                L.append(f"  - > …{c}…")
            L.append("")

    L += ["## Description check (manual)", "",
          "Metadata can match while the *prose* misdescribes the work — that is the",
          "Huang & Huang failure. Read each context below against the real title.", ""]
    for r in rows:
        if not r["contexts"]:
            continue
        L.append(f"### `{r['key']}`")
        L.append(f"- real title: **{r['crossref_title'] or r['title']}**")
        for c in r["contexts"][:2]:
            L.append(f"  - > …{c}…")
        L.append("")

    if uncited:
        L += ["## Never cited", ""] + [f"- `{r['key']}`" for r in uncited] + [""]

    Path(args.out).write_text("\n".join(L), encoding="utf-8")

    print()
    print(f"{'='*66}")
    print(f"  {len(blockers)} BLOCKER · {len(warns)} WARN · {len(uncited)} uncited")
    print(f"  report → {args.out}")
    print(f"{'='*66}")
    for r in blockers:
        print(f"  BLOCKER {r['key']:32} {','.join(r['problems'])}")
    return 1 if blockers else 0


if __name__ == "__main__":
    sys.exit(main())
