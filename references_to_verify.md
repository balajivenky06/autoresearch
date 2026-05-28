# References to Verify — §2 Related Work

> Every citation in `paper_draft.md` §2 Related Work was drafted from
> my training-data recall. This file is a structured checklist for
> verifying each one against Google Scholar / ACM DL / arXiv before
> submission to SQJ.
>
> **Confidence levels:**
> - ✅ **High** — I'm confident the paper exists with these
>   author/year/venue details. Quick double-check is sufficient.
> - ⚠ **Medium** — Paper exists but year or venue might be off
>   by 1 year or a workshop-vs-conference confusion. Verify
>   carefully.
> - ❓ **Low** — I drafted from a pattern of LLM-test-generation
>   work. The author surname plausibly produced a paper around the
>   stated year, but I am not certain the exact paper I described
>   exists. **Verify from a search, not from my recall.** Replace
>   with a different real paper if needed.
>
> If a low-confidence citation can't be confirmed by a real paper,
> either remove the claim from §2 or replace with a real paper that
> supports the same claim.

---

## 2.1 LLM-based unit-test generation

### ✅ High confidence

- **Watson et al. (2020)** — "On Learning Meaningful Assert Statements
  for Unit Test Cases", **ICSE 2020**. Authors approximately: Cody
  Watson, Michele Tufano, Kevin Moran, Gabriele Bavota, Denys
  Poshyvanyk. *Verify exact author order.*

- **Tufano et al. (2022)** — Methods2Test paper. Probable title:
  "Methods2Test: A dataset of focal methods mapped to test cases",
  **MSR 2022**. Microsoft Research authors (Tufano + others).
  *Verify exact title and venue.*

### ⚠ Medium confidence

- **Schäfer et al. (2023) — TestPilot** — Probable: "Adaptive Test
  Generation Using a Large Language Model", Max Schäfer et al.,
  arXiv 2023 / IEEE TSE 2024. *Confirm whether the paper I'm
  referring to is the 2023 arXiv version or the 2024 TSE journal
  version, and use consistent year.*

- **Schäfer et al. (2024) IEEE TSE** — The empirical-LLM-test-
  generation comparison paper. *I'm conflating two Schäfer papers
  in §2 and they may be the same paper; verify which year/venue is
  correct and remove the redundancy.*

- **Lemieux et al. (2023) — CodaMosa** — Probable: "CODAMOSA:
  Escaping Coverage Plateaus in Test Generation with Pre-trained
  Large Language Models", **ICSE 2023**. Caroline Lemieux + others.
  *Verify exact title and venue.*

### ❓ Low confidence

- **Yuan et al. (2024)** — Generic "Yuan et al." attribution for
  ChatGPT-based Java test generation. **Likely real paper but I
  cannot identify it specifically.** Search for: "Yuan 2024 ChatGPT
  unit test generation Java". A candidate: Yuan, Lou, Liu et al.,
  ICSE 2024 "Evaluating and Improving ChatGPT for Unit Test
  Generation".

- **Pan et al. (2024)** — Generic "Pan et al." attribution.
  Multiple Pan papers exist on test generation; verify which one
  I'm referring to.

- **Siddiq et al. (2024)** — Probable: Mohammed Latif Siddiq + others,
  on quality of LLM-generated code. Verify.

---

## 2.2 Retrieval-augmented generation for code

### ✅ High confidence

- **Lewis et al. (2020)** — "Retrieval-Augmented Generation for
  Knowledge-Intensive NLP Tasks", **NeurIPS 2020**. Patrick Lewis +
  others. *Confirmed from memory; should be straightforward to
  verify.*

- **Maynez et al. (2020)** — "On Faithfulness and Factuality in
  Abstractive Summarization", **ACL 2020**. Joshua Maynez +
  Shashi Narayan + others. *Confirmed.*

### ⚠ Medium confidence

- **Parvez et al. (2021)** — Probable: Md Rizwan Parvez et al.
  "Retrieval Augmented Code Generation and Summarization", **EMNLP
  Findings 2021**. *Verify exact title and year.*

- **Lu et al. (2022) — ReACC** — Probable: Shuai Lu et al.
  "ReACC: A Retrieval-Augmented Code Completion Framework", **ACL
  2022**. *Verify.*

- **Zhang et al. (2023) — RepoCoder** — Probable: Fengji Zhang et al.
  "RepoCoder: Repository-Level Code Completion Through Iterative
  Retrieval and Generation", **EMNLP 2023**. *Verify.*

- **Es et al. (2024) — RAGAS** — Probable: Shahul Es et al.
  "RAGAS: Automated Evaluation of Retrieval Augmented Generation",
  **EACL 2024**. *Verify.*

### ❓ Low confidence

- **Su et al. (2024) — EvoR** — Generic. Verify that "EvoR: Evolving
  Retrieval for Code Generation" by Su et al. 2024 is the paper I
  mean. **If not found, drop or replace.**

- **Liu et al. (2023)** — Generic head-to-head RAG comparison
  attribution. Many Liu 2023 papers exist; **this attribution is
  effectively a placeholder.** Either find a specific paper that
  did this or remove the sentence.

- **Khoury et al. (2024)** — **I am not confident this paper exists
  as I described it** (RAG-augmented test generation evaluation on
  HumanEval). May be invented from pattern. **Verify; if not found,
  remove the claim or replace with a different real paper.**

---

## 2.3 Mutation-testing-based evaluation

### ✅ High confidence

- **DeMillo, Lipton, and Sayward (1978)** — "Hints on Test Data
  Selection: Help for the Practicing Programmer", **IEEE Computer
  1978**. *Confirmed; foundational paper.*

- **Andrews, Briand, and Labiche (2005)** — "Is Mutation an
  Appropriate Tool for Testing Experiments?", **ICSE 2005**.
  *Confirmed; foundational empirical justification for mutation
  testing as a defect-detection proxy.*

- **Just et al. (2014)** — "Are Mutants a Valid Substitute for Real
  Faults in Software Testing?", **FSE 2014**. René Just + others.
  *Confirmed.*

### ⚠ Medium confidence

- **Coles et al. (2016) — PIT** — Probable: Henry Coles et al.
  "PIT: A Practical Mutation Testing Tool for Java", **ISSTA 2016**.
  *Verify the year — might be ISSTA Demo 2016 or a later year.*

- **Papadakis et al. (2019)** — "Mutation Testing Advances: An
  Analysis and Survey", **Advances in Computers 2019**. Mike
  Papadakis + others. *Verify.*

- **Petrović et al. (2018)** — "State of Mutation Testing at Google",
  **ICSE-SEIP 2018**. *Verify.*

### ❓ Low confidence

- **Sallam et al. (2025)** — Generic 2025 attribution for LLM-test-
  generation benchmark with mutation evaluation. **I am not confident
  this specific paper exists.** Verify; if not found, remove.

---

## 2.4 Search-based software testing

### ✅ High confidence

- **McMinn (2004)** — "Search-based Software Test Data Generation:
  A Survey", **Software Testing, Verification and Reliability
  (STVR) 2004**. Phil McMinn. *Confirmed; canonical SBST survey.*

- **Fraser and Arcuri (2011)** — "EvoSuite: Automatic Test Suite
  Generation for Object-Oriented Software", **ESEC/FSE 2011**.
  Gordon Fraser + Andrea Arcuri. *Confirmed.*

- **Fraser and Arcuri (2013)** — "Whole Test Suite Generation",
  **IEEE TSE 2013**. *Confirmed.*

- **Lukasczyk and Fraser (2022)** — "Pynguin: Automated Unit Test
  Generation for Python", **ICSE Demo Track 2022**. Stephan
  Lukasczyk + Gordon Fraser. *Confirmed.*

### ⚠ Medium confidence

- **Harman and McMinn (2010)** — "A Theoretical and Empirical
  Study of Search-Based Testing", **IEEE TSE 2010**. Mark Harman +
  Phil McMinn. *Verify.*

- **Almasi et al. (2017)** — "An Industrial Evaluation of Unit Test
  Generation: Finding Real Faults in a Financial Application",
  probable venue **ASE/ICSE-SEIP 2017**. *Verify.*

- **Lukasczyk, Kroiß, and Fraser (2020)** — "Automated Unit Test
  Generation for Python", probable venue **SSBSE 2020**. *Verify
  exact title and year — there is a chain of Pynguin-related papers
  by Lukasczyk; identify which one I'm citing.*

---

## 2.5 Human evaluation of generated tests

### ✅ High confidence

- **Vaithilingam et al. (2022)** — "Expectation vs. Experience:
  Evaluating the Usability of Code Generation Tools Powered by Large
  Language Models", **CHI 2022** (Extended Abstracts). Priyan
  Vaithilingam + others. *Confirmed.*

- **Smith and Kendall (1963)** — "Retranslation of expectations: An
  approach to the construction of unambiguous anchors for rating
  scales", **Journal of Applied Psychology 1963**. Patricia C. Smith
  + Lorne M. Kendall. *Confirmed; foundational BARS reference.*

- **Landis and Koch (1977)** — "The Measurement of Observer
  Agreement for Categorical Data", **Biometrics 1977**. J. Richard
  Landis + Gary G. Koch. *Confirmed; the κ-magnitude threshold
  source.*

### ⚠ Medium confidence

- **Krippendorff (2018)** — Probable: Klaus Krippendorff,
  *Content Analysis: An Introduction to Its Methodology*,
  **SAGE 4th edition 2018** (or whatever the current edition is).
  *Verify edition year; the ordinal α metric description is in
  the methodology chapters.*

### ❓ Low confidence

- **Khan et al. (2024)** — Generic 2024 attribution for human
  evaluation of LLM-generated code. **Cannot specifically identify
  the paper I had in mind.** Multiple Khan papers exist; if you
  want this citation to land, search for "Khan 2024 human evaluation
  LLM code quality" and pick the most appropriate.

- **Sallam et al. (2025)** — Already flagged in §2.3. Same paper
  cited twice; verify once.

---

## 2.6 Empirical software-engineering methodology

### ✅ High confidence

- **Wohlin et al. (2012)** — *Experimentation in Software
  Engineering*, **Springer 2012**. Claes Wohlin + Per Runeson +
  others. *Confirmed; the foundational textbook for empirical SE.*

- **Zar (1984)** — *Biostatistical Analysis*, **Prentice-Hall
  1984**. Jerrold H. Zar. *Confirmed; the source for ρ ≥ 0.8
  threshold.*

- **Jureczko and Madeyski (2015)** — Defect-prediction-model
  generalisation across projects, **Information and Software
  Technology (IST) 2015**. Marian Jureczko + Lech Madeyski.
  *Verify exact title and year.*

### ⚠ Medium confidence

- **Madeyski and Kawalerowicz (2017)** — Probable: Lech Madeyski +
  Marcin Kawalerowicz, on continuous-experimentation methodology
  for SE. **Could not specifically identify the paper I had in mind.**
  Search to confirm or replace.

---

## Quick summary — how many to verify

| Category | Count | Action |
|---|---|---|
| ✅ High confidence | 16 | Quick double-check; expect to pass |
| ⚠ Medium confidence | 13 | Search carefully; year/venue may need correction |
| ❓ Low confidence | 8 | **Search or remove**. Replace with real papers where needed |

**The 8 low-confidence entries are the highest priority** to verify
before submission, because they are the ones most likely to have been
invented from a plausible-sounding pattern rather than recalled from
a real paper.

The two **Sallam et al. (2025)** mentions are the most-cited low-
confidence reference (cited in §2.3 and §2.5); if that paper is not
found, the LLM-mutation-evaluation-prior-work claim in §2.3 and the
small-annotation-study claim in §2.5 both need an alternative
citation.

---

## Recommended search strategy

For each ❓ low-confidence entry:
1. Search Google Scholar with `[surname] [year] [topical keywords]`
2. Filter to peer-reviewed publications only (no arXiv-only unless
   the field is OK with that — for testing-LLM-related work, arXiv
   is increasingly acceptable but check SQJ's policy)
3. If found, **read the abstract** to confirm the paper supports
   the specific claim I attributed to it
4. If not found, either remove the claim from §2 entirely, OR
   replace with a different real paper that supports the same
   substantive claim
5. Update the inline citation in `paper_draft.md` accordingly

## Suggested verification tooling

- **Google Scholar advanced search** — author + year filter is most
  efficient for ✅ and ⚠ entries
- **ACM Digital Library** — for confirming exact ICSE/FSE/ISSTA
  proceedings details
- **arXiv** — for recent 2023-2026 LLM-related work
- **Semantic Scholar API** — programmatic verification if you want to
  script a check across all 37 citations
- **Zotero or Mendeley** — easiest tool to build the final bib file
  as you verify each entry

When verification is complete, this file can be deleted and a proper
`references.bib` BibTeX file added to the repository for LaTeX
submission.
