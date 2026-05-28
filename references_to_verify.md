# References to Verify — references.bib (post-2025-refresh)

> Every citation in `paper_draft.tex` (and the matching entry in
> `references.bib`) was drafted from training-data recall. This file
> is a structured verification checklist. Open each candidate in
> Google Scholar / arXiv / ACM DL, confirm author, year, venue, and
> the substantive claim attributed to the paper. **The
> 2024--2026 candidates carry the highest verification risk** because
> my training-data confidence on very recent literature is lower than
> on foundational classics.
>
> **Confidence tags:**
> - ✅ **HIGH**: foundational classic; spot-check is enough
> - ⚠ **MED**: paper plausibly exists; verify exact year/venue
> - ❓ **TODO (2025/2026 candidate)**: drafted as a recent-looking
>   placeholder; verify the paper exists or replace with the real
>   recent paper that supports the same claim

---

## ✅ HIGH confidence — foundational classics (16 entries)

These are the canonical references; the modern paper would still
cite the foundational source. Spot-check exact details only.

| Key | Paper |
|---|---|
| `demillo1978` | DeMillo, Lipton, Sayward (1978) — IEEE Computer — original mutation-testing paper |
| `andrews2005` | Andrews, Briand, Labiche (2005) — ICSE — mutation-vs-real-faults validation |
| `just2014` | Just et al. (2014) — FSE — mutation-vs-real-faults large study |
| `lewis2020` | Lewis et al. (2020) — NeurIPS — original RAG paper |
| `maynez2020` | Maynez et al. (2020) — ACL — faithfulness in summarisation |
| `fraser2011` | Fraser & Arcuri (2011) — ESEC/FSE — EvoSuite |
| `fraser2013` | Fraser & Arcuri (2013) — IEEE TSE — Whole Test Suite Generation |
| `lukasczyk2022` | Lukasczyk & Fraser (2022) — ICSE Demo — Pynguin |
| `mcminn2004` | McMinn (2004) — STVR — SBST survey |
| `watson2020` | Watson et al. (2020) — ICSE — LLM assert-generation |
| `vaithilingam2022` | Vaithilingam et al. (2022) — CHI — Copilot UX |
| `smith1963` | Smith & Kendall (1963) — J. Applied Psychology — BARS source |
| `landis1977` | Landis & Koch (1977) — Biometrics — κ thresholds |
| `wohlin2012` | Wohlin et al. (2012) — Springer — empirical SE textbook |
| `zar1984` | Zar (1984) — Prentice-Hall — biostatistical analysis (ρ threshold) |
| `chen2021humaneval` / `austin2021mbpp` | HumanEval / MBPP benchmarks |

---

## ⚠ MED confidence — verify year / venue (13 entries)

Real papers but year or venue may be off by one.

| Key | Likely details |
|---|---|
| `tufano2022` | Tufano et al. — Methods2Test — MSR 2022 |
| `lemieux2023codamosa` | Lemieux et al. — CodaMosa — ICSE 2023 |
| `schafer2023testpilot` | Schäfer, Nguyen, Tip — TestPilot — arXiv 2023 |
| `schafer2024` | Schäfer et al. — IEEE TSE 2024 — empirical LLM TG |
| `parvez2021` | Parvez et al. — EMNLP Findings 2021 |
| `lu2022reacc` | Lu et al. — ReACC — ACL 2022 |
| `zhang2023repocoder` | Zhang et al. — RepoCoder — EMNLP 2023 |
| `es2024ragas` | Es et al. — RAGAS — EACL 2024 |
| `coles2016` | Coles et al. — PIT — ISSTA 2016 (companion?) |
| `papadakis2019` | Papadakis et al. — Advances in Computers 2019 — mutation survey |
| `petrovic2018` | Petrović & Ivanković — ICSE-SEIP 2018 — Mutation at Google |
| `harman2010` | Harman & McMinn (2010) — IEEE TSE |
| `almasi2017` | Almasi et al. (2017) — ICSE-SEIP — industrial EvoSuite |
| `krippendorff2018` | Krippendorff — 4th edition, SAGE 2018 |
| `daka2014` | Daka & Fraser — ISSRE 2014 — unit-testing survey |
| `jureczko2015` | Jureczko & Madeyski — e-Informatica SEJ 2015 |

---

## ❓ TODO — 2024-2025-2026 candidates (10 entries)

**Highest verification priority.** Each entry below was drafted as a
plausible recent paper that fills a specific citation role. Verify
the paper exists with the claimed authors/title/year; if not, find
the real paper that supports the same substantive claim or remove
the claim from `paper_draft.tex`.

| Key | What it cites | Search hints |
|---|---|---|
| `yuan2025chattest` | Yuan et al. 2025 — ICSE — LLM unit test gen | Search: "Yuan ICSE 2025 unit test ChatGPT" |
| `pan2025empirical` | Pan et al. 2025 — TOSEM — LLM-bug study | Search: "Pan 2025 TOSEM large language model translation"; the candidate is "Lost in Translation" but verify |
| `siddiq2025empirical` | Siddiq et al. 2025 — EASE — LLM JUnit | Search: "Siddiq 2025 EASE LLM JUnit"; this paper likely exists at the cited venue |
| `wang2025llm4se` | Wang et al. 2025 — IEEE TSE — LLM4SE testing survey | Search: "Wang 2025 software testing LLM survey" |
| `papadakis2025survey` | Papadakis et al. 2025 — ACM CSur — mutation × LLM roadmap | Search: "Papadakis 2025 mutation testing large language model" |
| `li2025mutationllm` | Li et al. 2025 — FSE — LLM mutation effectiveness | Search: "FSE 2025 mutation testing LLM-generated tests" |
| `liu2024sbstllm` | Liu et al. 2024 — ASE — SBST-vs-LLM comparison | Search: "ASE 2024 SBST LLM empirical comparison" |
| `liu2025codereview` | Liu et al. 2024 — TOSEM — ChatGPT code quality | Search: "Liu TOSEM 2024 ChatGPT code quality refining" |
| `khan2025humaneval` | Khan & Uddin 2025 — EMSE — LLM code review eval | Search: "Khan Uddin 2025 EMSE LLM code review test" |
| `nguyen2025dev` | Nguyen et al. 2025 — ICSE — Copilot UX in practice | Search: "Nguyen 2025 ICSE Copilot industrial mixed-methods" |
| `rag4code2025` | Wang et al. 2025 — ACM CSur — RAG for code survey | Search: "RAG code generation survey 2025"; if a better-fit paper exists use it |
| `su2025evor` | Su et al. — EvoR — EMNLP Findings 2024 | Search: "Su EvoR Evolving Retrieval"; verify 2024 vs 2025 |
| `lukasczyk2023empirical` | Lukasczyk & Fraser — QRS 2023 — Pynguin empirical | Search: "Lukasczyk Fraser QRS Pynguin empirical study"; replace with the real recent Pynguin paper if title differs |
| `madeyski2024empirical` | Madeyski & Kawalerowicz 2024 — EMSE — replication | Search: "Madeyski Kawalerowicz 2024 EMSE continuous TDD" |

---

## Quick-fire summary

| Bucket | Count | Action |
|---|---|---|
| ✅ HIGH (foundational) | 16 | Spot-check author/year/venue only |
| ⚠ MED (year/venue verify) | 16 | Read each entry's `note = {}` field in `references.bib` and confirm details |
| ❓ TODO (recent candidate) | 15 | **Verify or replace.** Each entry has an inline `note = {TODO ...}` field in the .bib. |
| **Total** | **47** | |

---

## Verification workflow

For each TODO entry:

1. Open the BibTeX entry in `references.bib`; read the inline
   `note = {TODO ...}` for the candidate paper's claimed details
   and the search hint
2. Run a Google Scholar / Semantic Scholar / arXiv search using the
   search hint above
3. If you find a paper that matches the candidate description:
   - Confirm the actual author list, year, venue, title
   - Update the `.bib` entry with the verified details
   - Remove the `note = {TODO ...}` field
4. If you find a closely related paper that fills the same role
   (even if author/title differ):
   - Replace the `.bib` entry with that paper's correct details
   - Update the corresponding `\cite{key}` in `paper_draft.tex` if
     the key needs to change
5. If you cannot find any paper that supports the cited claim:
   - Remove the citation from `paper_draft.tex`
   - Remove the entry from `references.bib`
   - If the surrounding sentence depended on the citation, rewrite
     it to remove the unsupported claim

## Foundational entries

The 16 ✅ HIGH entries should NOT be replaced with 2024/2025 papers
even though they are old. They are the canonical references that
reviewers expect for: original RAG (Lewis 2020), original mutation-
testing-correlates-with-real-faults validation (Andrews 2005, Just
2014), foundational EvoSuite (Fraser & Arcuri 2011), original
mutation-testing concept (DeMillo 1978), foundational SBST survey
(McMinn 2004), Pynguin (Lukasczyk & Fraser 2022), Cohen's-κ-
magnitudes source (Landis & Koch 1977), BARS source (Smith & Kendall
1963), empirical-SE textbook (Wohlin et al. 2012), biostatistical-ρ
source (Zar 1984). Replacing any of these with a recent paper would
weaken the citation, not strengthen it.

## Once complete

Delete this file and remove all the `note = {TODO ...}` lines from
`references.bib`.
