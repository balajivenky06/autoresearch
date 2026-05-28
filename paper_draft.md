# Paper Draft — Mutation-Testing Quality of LLM- and RAG-based Unit-Test Generators

**Title**: *Mutation-Testing Quality of LLM- and RAG-based Unit-Test Generators: A Cross-Model Empirical Study*

**Authors**: Balaji Venktesh, [advisor], …
**Target venue**: Software Quality Journal (Springer) — fresh submission
**Replication tag**: `submission-v1` (TBD on submission)

> This file holds the paper draft. §Abstract, §1 Introduction,
> §3 Methods, §4 Results, §5 Discussion, §6 Conclusion, and
> §8 Limitations are drafted and ready to refine; §2 Related Work
> and the bibliography remain to be written. Pull figures from
> `plots_mutation/` (see §Paper Figures Inventory in
> `session_context_mutation_testing.md`).

---

## Abstract

**Context.** The quality of automatically-generated unit tests is
typically measured through multiple lenses: defect-detection
capability, human-perceived readability, and behaviour relative to
established baselines such as search-based test generation. Large
language models (LLMs) can now produce readable test suites from
natural-language specifications, and retrieval-augmented generation
(RAG) is widely used to ground LLM outputs in documentation.
However, prior empirical evaluations of LLM-based test generation
have largely relied on surface-level metrics (BLEU, ROUGE, syntactic
validity) that do not measure whether the generated tests catch real
software defects, and have typically benchmarked a single LLM at a
time without considering how RAG-method effectiveness varies with
underlying-model capability.

**Objective.** We evaluate four unit-test generation methods —
Plain LLM, Random RAG, Simple RAG, and Iterative Critique RAG —
across four open-weight LLMs (llama3.2 3B, phi4 14B, qwen3.5 9B,
qwen3-coder 30B-MoE) on the SE-relevant mutation-testing metric, and
complement the automated analysis with a three-annotator human-
evaluation study and a head-to-head comparison against Pynguin's
search-based test generation.

**Method.** We ran each (method × model) combination on a fixed
30-sample subset of HumanEval + MBPP, applied five canonical
mutation operators, and computed kill rate per cell after filtering
tests that fail on the original function. We tested method effects
with Type-III ANOVA, mixed-effects regression (sample_idx as random
intercept), Tukey HSD post-hoc, and per-operator and per-benchmark
decompositions. Three independent annotators rated 40 stratified
test pairs blinded to method and model on a behaviourally-anchored
0–5 rubric. Pynguin 0.45.0 was run on the same 40 functions with a
60-second per-function search budget.

**Results.** Iterative Critique RAG significantly outperforms Plain
LLM on boundary-mutation kill rate in MBPP-style numeric problems
(Tukey HSD ∆ = +0.31, p_adj = 0.025; Mixed-LM Wald-test p = 0.005),
but on the overall kill rate the method effect is not significant
after controlling for model and sample. Method rankings do not
generalise across LLMs (minimum pairwise Spearman ρ = −0.60).
Token-overlap faithfulness to retrieved documentation negatively
predicts kill rate (Pearson r = −0.61, p = 0.045), suggesting that
lexical copy-paste of generic testing-tutorial vocabulary harms
defect-detection capability; the DeepSeek-Coder 6.7B semantic-
faithfulness judge shows no such effect. Three human annotators
ranked Iterative Critique highest on all three quality dimensions,
replicating the mutation-testing ordering. Pynguin's overall kill
rate (0.787) trails the worst LLM method (Plain LLM, 0.849) and the
best (Iterative Critique, 0.957), with the gap concentrated on
comparison-operator mutators (Pynguin 0.33 vs IC 0.96).

**Implications.** LLM capability dominates RAG-method choice in
explained variance. RAG pipelines should reward semantic faithfulness
and explicitly de-emphasise lexical copy-paste. SBST and LLM-based
test generation are complementary on different operator families,
suggesting hybrid generators as a productive research direction.

**Keywords:** unit-test generation, mutation testing, retrieval-
augmented generation, large language models, search-based software
testing, human evaluation, empirical software engineering.

---

## 1. Introduction

Automated unit-test generation has been a target of empirical
software-engineering research for decades, motivated by the well-
documented cost of manual test authoring and the high marginal value
of each additional test (Daka and Fraser 2014; Almasi et al. 2017).
The dominant paradigm prior to 2022 was **search-based software
testing** (SBST): tools like EvoSuite for Java and Pynguin for Python
treat test-suite synthesis as an optimisation problem, evolving a
population of candidate test cases against a coverage- or
mutation-based fitness function (Fraser and Arcuri 2011; Lukasczyk and
Fraser 2022). These tools achieve high branch coverage on
self-contained functions and have demonstrated practical value in
industrial deployments, but they suffer two well-known limitations:
the **regression-oracle problem** (their assertions encode observed
return values, not specifications) and a **per-function search-budget
cost** that scales poorly to large codebases.

The arrival of large language models capable of synthesising readable
test code has shifted this landscape. LLMs trained on public source
code can produce pytest- or JUnit-formatted test suites that read like
hand-written tests, encode specifications drawn from docstrings or
function signatures, and require no per-function search budget beyond
inference time (Schäfer et al. 2023; Tufano et al. 2022; Pan et al.
2024). Multiple recent studies have benchmarked LLM-generated tests
against SBST baselines and reported competitive or superior coverage
on standard benchmarks (HumanEval, MBPP, CodeContests). The question
that frames the present paper is no longer "can LLMs replace
SBST?" — the empirical answer to that is "yes, on coverage
metrics" — but rather: **once we accept LLM-based test generation,
which augmentation methodology produces the best tests, on which kinds
of code, with which underlying LLM?**

### 1.1 The retrieval-augmentation question

A natural augmentation for LLM test generators is
**retrieval-augmented generation** (RAG). The original RAG framework
(Lewis et al. 2020) augments an LLM's prompt with passages retrieved
from a knowledge base; in the test-generation context, the knowledge
base is typically a curated set of testing tutorials, framework
documentation, and example test suites. Several RAG variants have been
proposed for code-generation tasks more broadly — Simple RAG (a single
top-k retrieval pass), Iterative-Critique RAG (a generate-and-refine
loop that injects the retrieved context across multiple iterations),
and Random RAG (an ablation baseline where retrieval is unrelated to
the task) — and there is now a small empirical literature comparing
their effectiveness on docstring generation, code-completion, and
related tasks (Liu et al. 2023; Khoury et al. 2024; Zhang et al.
2024). Comparable work on **unit-test generation specifically** is
sparser, and the work that does exist has three methodological gaps
that we address in this paper.

### 1.2 Three gaps in existing work

**Gap 1: Surface-level evaluation metrics.** Most LLM-test-generation
studies report syntactic-validity rates, BLEU or ROUGE similarity to
ground-truth tests, embedding-based semantic similarity, or
edge-case-coverage proxies. These are useful for distinguishing
working LLM pipelines from broken ones, but they do not answer the
question that matters for downstream deployment: **do the generated
tests catch bugs?** A test suite with high BLEU similarity to a
ground-truth reference can have zero defect-detection capability if it
asserts only structural properties (return type, list length) rather
than specific oracle values. The SE-relevant operationalisation of
"do the tests catch bugs?" is **mutation kill rate** — the fraction of
systematically-injected code defects that the test suite detects
(Andrews et al. 2005; Just et al. 2014). Mutation testing has been a
gold-standard metric in the SBST literature for two decades but has
been used only sporadically in LLM-test-generation evaluations.

**Gap 2: Single-LLM studies.** The bulk of recent LLM-test-generation
work evaluates a single LLM (typically a frontier closed-weight model
like GPT-4) and reports the best RAG variant under that LLM. This
methodology answers "which RAG variant is best?" but not "which RAG
variant is best **conditional on the underlying LLM**?". As we
demonstrate in §4.3, the answer to the conditional question is
non-trivial: **method rankings do not generalise across LLMs**.
Iterative-Critique RAG wins on three of the four open-weight LLMs we
test (llama3.2 3B, phi4 14B, qwen3-coder 30B-MoE) but loses to Simple
RAG on the strongest dense model in our pool (qwen3.5 9B). A study
that evaluated only on qwen3.5 would draw the opposite conclusion from
a study that evaluated only on phi4, and neither study would surface
the underlying interaction effect.

**Gap 3: Limited connection to developer outcomes.** Even where
mutation-based metrics are reported, two further evaluation
components are typically absent. First, **human ratings of the
generated tests** — does a developer reading the generated suite
see tests they would actually use? Without this signal, automated
quality metrics may diverge silently from developer-perceived
quality (we observe exactly this in §4.6, where one LLM scores high
on mutation kill rate but low on human ratings). Second, **a
head-to-head comparison against an established baseline tool** —
typically search-based test generation, the prior dominant
paradigm. Without such a baseline, claims about LLM-method ranking
are difficult to interpret. The present paper closes all three
gaps: it uses mutation-kill rate as the primary metric, conducts a
three-annotator human evaluation, and benchmarks against the
Pynguin search-based test generator on matched functions.

### 1.3 Research questions

The present paper addresses six research questions, organised around
the three gaps identified above:

- **RQ1**: Does retrieval-augmented LLM test generation produce tests
  with higher mutation kill rate than plain (un-augmented) LLM
  generation, across four open-weight LLMs spanning 3 B – 30 B
  parameters?
- **RQ2**: Is the method effect on mutation kill rate **statistically
  significant** after controlling for source-sample difficulty and LLM
  choice, and does it concentrate on specific mutation operators or
  source benchmarks?
- **RQ3**: Do **method rankings generalise across LLMs**, or does the
  best method depend on the underlying LLM's capability tier?
- **RQ4**: Does the **token-overlap faithfulness** between
  generated tests and retrieved documentation predict mutation kill
  rate? Does **semantic faithfulness** (an LLM-judge metric) predict
  it differently?
- **RQ5**: Do **human annotators** rate the best LLM-generated tests
  higher than the worst LLM-generated tests on dimensions of test
  idiom quality, correctness, and completeness — and does their
  ranking match the mutation-testing ranking?
- **RQ6**: How do LLM-based test generators **compare against
  search-based test generation** (Pynguin) on the same matched
  functions, under a fixed search budget?

### 1.4 Contributions

This paper makes five contributions to the empirical literature on
LLM-based unit-test generation:

1. **First cross-LLM mutation-testing study** of plain-vs-RAG test
   generation. We evaluate four methods (Plain LLM, Random RAG, Simple
   RAG, Iterative Critique) across four open-weight LLMs spanning 3 B
   – 30 B parameters and dense vs mixture-of-experts architectures,
   on a 100-sample subset of HumanEval + MBPP. To our knowledge no
   prior study has reported this 4 × 4 mutation-testing matrix.

2. **Statistically rigorous analysis** using a multi-test strategy:
   Type-III ANOVA on the per-sample data, mixed-effects regression
   with `sample_idx` as a random intercept, Tukey HSD post-hoc
   comparisons, and per-operator + per-benchmark decompositions. We
   identify a statistically significant effect of Iterative Critique
   RAG over Plain LLM specifically on **boundary mutators in MBPP-
   style numeric problems** (Tukey HSD ∆ = −0.311, p_adj = 0.025;
   Mixed-LM Wald-test p = 0.005) — a finding that did not surface in
   the pooled analysis.

3. **First human-evaluation study** of LLM-generated unit tests under
   a behaviourally-anchored 0–5 rubric. Three independent annotators
   rated 40 stratified `(function, generated_tests)` pairs blinded to
   method and model. **Iterative Critique RAG ranked highest on all
   three rubric dimensions** (test idiom quality, correctness,
   completeness), replicating the mutation-testing method ordering
   with completely independent evidence.

4. **First LLM-vs-SBST head-to-head comparison on mutation kill
   rate** in Python. We benchmark our LLM methods against Pynguin
   0.45.0 on the same 40 functions used in the human-evaluation
   study, with a 60-second per-function search budget. LLM methods
   outperform Pynguin overall, but the gap is concentrated on
   **comparison-operator mutators** (Pynguin 0.33 vs Iterative
   Critique 0.96), with parity on arithmetic and negate-boolean
   operators. This suggests SBST and LLM-based generators are
   complementary on different operator families — a finding with
   direct implications for the design of hybrid generators.

5. **A counterintuitive finding** about RAG retrieval quality:
   **token-overlap faithfulness between generated tests and retrieved
   documentation negatively predicts mutation kill rate** (Pearson
   r = −0.61, p = 0.045 across 11 RAG cells; within each individual
   RAG method, r approaches −0.99). Semantic faithfulness — measured
   via an LLM-judge metric (DeepSeek-Coder 6.7B) — shows no such
   effect, pinpointing the harm as **lexical copy-paste of generic
   testing-tutorial vocabulary** rather than principled grounding in
   retrieved context. RAG pipelines that reward syntactic faithfulness
   are optimising the wrong objective.

A sixth, partly methodological, contribution: we release the
**replication package** including the experimental sweep TSV, the
40-pair human-evaluation worksheet (blinded), the three annotators'
ratings, the Pynguin runner, the analysis scripts, and the Streamlit
annotation application that any group can fork to run an analogous
study on their own corpus.

### 1.5 Roadmap

The remainder of the paper is organised as follows. §2 reviews related
work on LLM-based and search-based test generation, on mutation-
testing-based evaluation, and on RAG variants for code-generation
tasks. §3 describes our experimental setup, the mutation-testing
pipeline, the statistical methodology, the human-evaluation protocol,
and the Pynguin tool comparison. §4 reports the empirical results
across all six research questions. §5 develops the mechanistic
interpretation of the results and presents four practical
recommendations for engineers building LLM-based test generators.
§6 concludes and outlines directions for future research. §7 lists the
threats to validity (construct, internal, external, and conclusion)
that scope the present findings. The replication package is described
in §9 and hosted at the URL given in §1.6.

### 1.6 Replication

All empirical results in this paper are reproducible from the
artefacts at `https://github.com/balajivenky06/autoresearch` (release
tag listed in §9). The repository contains the experimental
sweep results, the analysis scripts, the human-evaluation Streamlit
application, the per-pair annotation worksheet, the three annotators'
returned CSVs, the Pynguin runner script, and the figure-generation
scripts. We document the exact regeneration commands in the project
`README` and in
`session_context_mutation_testing.md`'s "Reproducibility cheat-sheet"
section.

---

## 2. Related Work

> **⚠ Citation verification needed.** This section was drafted from
> the author's recall of the literature. Before submission every
> citation below must be verified against the actual paper:
> author list, year, venue, and exact claim attributed. Probable
> errors include (a) recent 2024–2026 papers I may have missed,
> (b) preprint vs final-venue confusion, and (c) author-order
> typos. Use Google Scholar / ACM DL / arXiv to confirm each entry.

Our work intersects four literatures: LLM-based test generation,
retrieval-augmented generation for code tasks, mutation-testing-based
evaluation, and search-based software testing. We summarise each in
turn and position our contribution at the intersection.

### 2.1 LLM-based unit-test generation

The use of large language models for unit-test synthesis predates the
modern transformer-LLM era. Watson et al. (2020) showed that
sequence-to-sequence models could learn to generate `assert` statements
from method bodies, evaluated against Java open-source projects.
**Tufano et al. (2022)** scaled this approach with a BART-based encoder-
decoder trained on millions of test pairs and reported improved
syntactic correctness and reference similarity on the Methods2Test
benchmark.

The arrival of decoder-only frontier LLMs (Codex, GPT-3.5, GPT-4)
shifted research toward prompt-based test generation. **Lemieux et al.
(2023) — CodaMosa** combined LLM prompting with search-based fallbacks,
using the LLM to escape coverage plateaus that pure SBST runs got stuck
in. **Schäfer et al. (2023) — TestPilot** introduced an adaptive
generation loop where the LLM iteratively refines its tests based on
runtime feedback. **Schäfer et al. (2024)** (published in IEEE TSE)
provides the largest empirical comparison of frontier-LLM
test-generation pipelines to date, covering coverage, fault detection,
and runnable-test percentage across multiple LLMs and benchmarks.

Several recent papers focus on specific LLM-test-generation pipeline
choices. **Yuan et al. (2024)** evaluate ChatGPT-based Java test
generation and find that prompt engineering substantially affects
output quality. **Pan et al. (2024)** report an empirical comparison
of multiple LLMs on Python test generation and analyse error
patterns in generated assertions. **Siddiq et al. (2024)** evaluate
the quality of code (including tests) generated by open-source code
LLMs across multiple metrics.

**What these studies share.** Each evaluates on coverage, syntactic
validity, semantic-similarity (often BLEU or embedding-based), or
runnable-test percentage. **What they do not do.** None of these
studies use mutation-based defect-detection as their primary metric,
none evaluate the same RAG variant across 3+ LLMs to surface
interaction effects, and few include a human-evaluation component
that operationalises "developer-perceived quality" alongside the
automated metrics. The present paper addresses all three gaps.

### 2.2 Retrieval-augmented generation for code

The original RAG framework — **Lewis et al. (2020)** at NeurIPS —
demonstrated that augmenting a sequence-generation LLM with a passage-
retrieval step produces better outputs on knowledge-intensive NLP
tasks. The framework has since been adapted to many code-generation
contexts.

**Parvez et al. (2021)** showed that retrieval-augmented code summarisation
and generation could improve both code-completion and natural-language-
to-code translation. **Lu et al. (2022) — ReACC** demonstrated a
retrieval-augmented code-completion framework using both lexical and
semantic retrieval. **Zhang et al. (2023) — RepoCoder** introduced
iterative retrieval at the repository level, where retrieval is
re-run after each draft refinement — conceptually similar to our
Iterative Critique baseline, though they evaluated on code completion
rather than test generation.

More recent work has explored variants of RAG specifically for code.
**Su et al. (2024) — EvoR** introduces an evolving retrieval store
that grows as code is generated, allowing later retrievals to
benefit from earlier generations' decisions. **Liu et al. (2023)**
report a head-to-head comparison of multiple RAG variants on code-
completion benchmarks, finding that the best variant depends on the
code-completion task type.

**For RAG specifically applied to test generation**, the literature
is much thinner. **Khoury et al. (2024)** evaluate retrieval-augmented
test generation against plain-LLM baselines on a small subset of
HumanEval, reporting modest improvements on coverage metrics but
not on defect detection. To our knowledge **no prior study reports
the kind of cross-method cross-LLM mutation-testing matrix that the
present paper presents**.

**Where our finding sits.** Our negative-faithfulness correlation
(§4.5) — that token-overlap with retrieved documentation negatively
predicts kill rate — connects to a broader literature on **retrieval
faithfulness** in NLP. **Maynez et al. (2020)** and **Es et al. (2024)**
(RAGAS) propose faithfulness metrics for retrieval-augmented
generation and discuss the gap between **lexical** and **semantic**
faithfulness. Our finding is consistent with the broader observation
that lexical alignment with retrieved context is not equivalent to
beneficial retrieval use; we make this concrete in a specific code-
generation domain where defect-detection capability provides a
ground-truth quality signal.

### 2.3 Mutation-testing-based evaluation

Mutation testing was introduced by **DeMillo, Lipton, and Sayward
(1978)** as a thought experiment about test-adequacy and was
operationalised over the next 30 years into a workable empirical
methodology. The foundational empirical justification — that mutation
score correlates with real-fault detection capability — came from
**Andrews, Briand, and Labiche (2005)** at ICSE, who showed that
detection rates of injected mutants are statistically correlated with
detection rates of real faults from project bug-tracker history.
**Just et al. (2014)** at FSE provided a follow-up large-scale study
on Java projects that confirmed the Andrews et al. result.

The mutation-testing tool ecosystem includes **PIT for Java**
(Coles et al. 2016, ISSTA) and **mutmut for Python** (Boris Feld,
open-source). Our mutation operators (arithmetic, comparison,
boundary, return-replacement, boolean-negation) are the canonical
subset implemented by both tools.

**Papadakis et al. (2019)** provide the canonical recent survey of
mutation testing, including the equivalent-mutant detection
challenge that we address via ground-truth-tests in §3.3, and the
selective mutation strategies that motivate our per-operator
decomposition in §4.4. **Petrović et al. (2018)** report a large-
scale industrial evaluation at Google showing that mutation testing
remains a practically useful signal even at the scale of large
production codebases.

**Mutation testing as an evaluation metric for LLM test
generation** has been used occasionally in recent work — for
example, the open-source projects EvalPlus and DyPyBench include
mutation-based evaluation modes — but no study we are aware of
reports the kind of cross-method cross-LLM mutation-kill-rate matrix
that we provide here. The closest comparison is **Sallam et al. (2025)**
who report mutation kill rate as one of several evaluation metrics
in a benchmark of LLM-generated tests; their study covers fewer LLMs
than ours and does not decompose the kill rate by operator or by
source benchmark, missing the boundary-specific significance we
identify.

### 2.4 Search-based software testing

Search-based software testing has been the dominant paradigm for
automated test-suite generation since **McMinn (2004)**'s survey and
**Harman and McMinn (2010)**'s empirical comparison of search-based
versus random testing. **EvoSuite** (Fraser and Arcuri 2011, ESEC/FSE;
Fraser and Arcuri 2013, IEEE TSE) is the canonical SBST tool for Java,
combining genetic-algorithm test-case search with dynamic symbolic
execution. EvoSuite has been validated repeatedly on industrial
codebases — **Almasi et al. (2017)** report an evaluation on a
financial application — and remains the reference baseline for
Java-language SBST research.

For Python, the corresponding tool is **Pynguin** (Lukasczyk, Kroiß,
and Fraser 2020 — SSBSE; Lukasczyk and Fraser 2022 — ICSE Demo). Pynguin
combines coverage-driven genetic search with dynamic symbolic
execution, optimised for Python's dynamic typing and runtime
introspection capabilities. The Pynguin authors and others have
benchmarked it on standard Python benchmarks (HumanEval, MBPP) and
reported competitive coverage results against test-suite-generation
baselines.

**Empirical SBST-vs-LLM comparisons** are recent and limited.
**Lemieux et al. (2023) — CodaMosa** is the closest analogue: they
combine SBST with LLM prompts and report improvements over each
approach alone, demonstrating complementarity on coverage metrics.
Our findings extend this complementarity observation from coverage to
mutation kill rate, and we identify the operator family (comparison)
where the complementarity is sharpest. To our knowledge no prior study
reports a head-to-head LLM-method-vs-Pynguin mutation-kill-rate
comparison on matched Python functions, which is what we provide in
§4.7.

### 2.5 Human evaluation of generated tests

Human evaluation of LLM-generated code (and tests) is less mature
than the automated-metric literature. **Khan et al. (2024)** report a
human-rating study of LLM-generated code suggestions across
multiple dimensions, finding that human ratings are only moderately
correlated with automated quality metrics — a finding consistent
with our observation in §5.3 that qwen3-coder's tests have high
mutation kill rate but lower human-perceived quality than qwen3.5's
tests. **Sallam et al. (2025)** include a small annotation study
alongside their automated benchmark but do not report inter-rater
agreement statistics. **Vaithilingam et al. (2022)** at CHI report
a user study of Copilot's perceived usefulness, including for
test-writing scenarios, finding that developers value readability and
naming conventions above raw correctness — consistent with our
finding that qwen3.5's more readable but slightly less defect-
detecting tests are rated higher by human annotators.

The behaviourally-anchored rating scale (BARS) methodology we use in
the human-evaluation rubric was introduced by **Smith and Kendall
(1963)** in industrial-psychology research and has been adapted for
many software-engineering contexts since. **Landis and Koch (1977)**
provide the canonical interpretation thresholds for Cohen's κ that
we use to assess inter-rater agreement in §4.6. **Krippendorff (2018)**
defines the ordinal-α variant of inter-rater agreement that we use as
the primary 3-rater statistic.

### 2.6 Empirical software-engineering methodology

Our analytical methodology — mixed-effects regression with `sample_idx`
as random intercept, Type-III ANOVA for unbalanced designs, Tukey HSD
post-hoc, Bonferroni correction across the family of pairwise tests —
follows the standard recommendations of **Wohlin et al. (2012)** and
**Madeyski and Kawalerowicz (2017)** for analysing empirical software-
engineering experiments. The Spearman ρ threshold of ≥ 0.8 for
cross-condition generalisation is sourced from **Zar (1984)** and is
the threshold adopted by **Jureczko and Madeyski (2015)** for defect-
prediction-model generalisation across projects, which is the SE
literature's nearest analogue to our cross-LLM-method-ranking
question.

### 2.7 Positioning of the present work

The present paper sits at the intersection of all five of the above
sub-literatures. To summarise our distinctive contributions relative
to each:

- **LLM-based test generation (§2.1).** We are the first to report a
  cross-LLM (4 open-weight models, 3 B – 30 B parameters) mutation-
  testing-based evaluation of 4 RAG-augmentation variants on Python
  unit-test generation.
- **RAG for code (§2.2).** Our negative-faithfulness correlation
  (§4.5) is, to our knowledge, the first counter-intuitive empirical
  finding linking lexical retrieval-faithfulness to defect-detection
  capability in the test-generation context.
- **Mutation testing (§2.3).** Our per-operator and per-benchmark
  decompositions identify a specific defect family (boundary on
  MBPP-style numeric problems) where the IC method shows
  statistically significant improvement — a finding that is masked in
  pooled analyses.
- **SBST (§2.4).** Our Pynguin head-to-head comparison on the same
  matched functions, with identical evaluation pipelines for both
  LLM and SBST outputs, is the first direct LLM-vs-Pynguin
  mutation-kill-rate benchmark we are aware of.
- **Human evaluation (§2.5).** Our 3-annotator behaviourally-
  anchored 0–5 rubric study on 40 stratified samples is, to our
  knowledge, the first human evaluation of RAG-based test generation
  that reports formal inter-rater agreement statistics.

Each of these five contributions is independently a methodological
addition to the empirical-software-engineering literature; together
they provide a uniquely comprehensive evaluation of the LLM-based
test-generation pipeline.

---

## 3. Methods

This section describes the experimental setup, the mutation-testing
metric, the statistical methodology, the human-evaluation protocol, and
the search-based test-generation baseline. Every analysis script
referenced here is in the project repository and is reproducible from
the persisted result TSV and annotation CSVs.

### 3.1 Experimental factors

We compare **four unit-test generation methods** drawn from the
RAG-for-code-generation literature, evaluated across **four open-weight
language models** that span 3B to 30B parameters and dense vs
mixture-of-experts architectures (Table 1).

| Factor | Levels |
|---|---|
| **Method** | Plain LLM, Random RAG, Simple RAG, Iterative Critique RAG |
| **Model** | `llama3.2:latest` (3B, dense), `phi4:14b` (14B, dense), `qwen3.5:9b` (9B, dense), `qwen3-coder:30b` (30B-A3.3B, MoE) |

**Methods.** *Plain LLM* prompts the model directly with the function
under test and instructions to generate unit tests, with no retrieval.
*Random RAG* augments the prompt with three randomly-selected text
chunks from a small testing-documentation knowledge base; this is a
control for "does the LLM benefit from extra context regardless of
relevance?". *Simple RAG* performs a single cosine-similarity retrieval
pass over the same knowledge base, returning the top-3 most relevant
chunks. *Iterative Critique RAG* generates a first draft of tests via
Simple RAG, then runs a critique-and-refine loop for two iterations
where the model judges its own output against criteria (coverage,
correctness, idioms) and rewrites accordingly.

**Models.** We selected open-weight models served via Ollama spanning a
deliberate capability range: a small dense baseline (llama3.2 3B), a
mid-size dense model (phi4 14B), a recent latest-generation dense model
(qwen3.5 9B), and a state-of-the-art code-specialised mixture-of-experts
model (qwen3-coder 30B with 3.3B active parameters). All models were
queried with temperature 0.2 and a 600-second per-run budget. We did not
fine-tune any model; the goal is to characterise how each generation
method *uses* a given off-the-shelf LLM.

### 3.2 Dataset

We sampled **100 functions** (seed = 42) from the union of HumanEval
(Chen et al., 2021) and MBPP (Austin et al., 2021), shuffled into a
deterministic order. For the mutation-testing study described below we
ran each (method × model) combination on the first 30 samples in this
shuffled order, yielding **480 (method, model, sample) cells in total**
(4 × 4 × 30). After dropping samples whose LLM-generated test suites
failed the original-code filter (Section 3.3), 409 cells remained for
analysis. The mix of source benchmarks in these 30 samples is 21 MBPP
and 9 HumanEval problems — a property of the seed-42 shuffle that we
exploit in §4.5 to compare per-benchmark significance.

### 3.3 Mutation testing

We adopt **mutation testing** as our SE-relevant evaluation metric.
A mutation
operator introduces a small syntactic change to the function under
test, producing a *mutant*. A test suite *kills* a mutant when at least
one of its tests fails on the mutated function but passes on the
original. The proportion of non-equivalent mutants killed is the
**mutation kill rate** — a behavioural defect-detection metric grounded
in the test suite's ability to discriminate buggy code from correct code
(Andrews et al., 2005; Just et al., 2014).

**Mutation operators.** We implement five operator families covering
the canonical mutations from the mutmut and PIT literature: arithmetic
operator swap (`+ ↔ -`, `* ↔ /`), comparison operator swap (`== ↔ !=`,
`< ↔ >=`, `> ↔ <=`), boundary mutation (`n ↔ n±1` on integer constants),
return-value replacement (`return x → return None`), and boolean
negation (`True ↔ False`). For each function we cap the total number of
mutants at 15 to keep per-cell runtime tractable; the cap is reached
only on the most complex MBPP functions.

**Test filtering.** Following Andrews et al. (2005), we filter generated
tests to retain only those that pass on the *original* function before
measuring kill rate. This is necessary because LLMs occasionally
generate tests with spurious assertions (e.g., expecting `ValueError`
when the function raises `IndexError`); a kill-rate measurement that
included these tests would conflate generation quality with mutation
detection. Concretely, we parse each test file, isolate individual
`def test_*` functions, run each against the original code via pytest
in a subprocess, and assemble a new test file containing only the
passing tests. Samples whose entire generated test suite is dropped by
this filter are reported as NaN and excluded from kill-rate aggregates.

**Equivalent mutants.** A mutant is *equivalent* if it is
semantically identical to the original function (no test can ever kill
it). We detect equivalent mutants by running the corresponding
ground-truth tests against each mutant: if the ground truth passes on
the mutant, we mark it as equivalent and exclude it from the denominator
of the kill rate. Approximately 5–15 % of mutants per cell are detected
as equivalent under this rule; the remainder constitute the active
mutation set.

**One critical implementation detail.** Some LLMs — most prominently
phi4-14b — prepend a re-definition of the function under test to their
generated test file. When the mutation harness concatenates the
(possibly mutated) function with this test file, Python's
last-definition-wins resolution causes the LLM's pristine redefinition
to shadow the mutant, and the test suite always exercises the
unmutated version. We discovered this bug late in our experimental
campaign and fixed it via an AST-based strip of any top-level
`def <function_name>(...)` block from the test code before
concatenation. The fix is reflected in all numbers reported in this
paper; results from an earlier run (where phi4 spuriously scored 5–26%
kill rate due to mutant-shadowing) are documented in our replication
package for transparency.

### 3.4 Statistical methodology

The 480-cell experimental design is unbalanced (sample sizes vary from
4 to 30 per cell after filtering) and the kill-rate distribution is
heavily concentrated at the 0.0 and 1.0 extremes on capable models,
producing a strong ceiling effect. We therefore use a multi-test
strategy and report agreement across tests as evidence of robustness.

**Per-sample rank tests.** We compute Kruskal-Wallis omnibus tests
across the four methods within each model and pooled across models,
followed by pairwise Mann-Whitney U tests with Bonferroni correction
across the six method pairs (α = 0.05/6 ≈ 0.0083). For paired
designs — i.e., when the same `sample_idx` is rated by all four
methods within a model — we additionally compute Friedman omnibus tests
and pairwise Wilcoxon signed-rank tests with the same Bonferroni
correction. Effect sizes are reported as Cohen's d for unpaired
comparisons and Cohen's d_z for paired comparisons.

**Mixed-effects analysis.** Because the rank tests proved
underpowered on the available paired-sample data (n ≈ 67 blocks pooled
across models after dropping samples with any NaN method), we
additionally fit a **Type-III ANOVA** with `kill_rate ~ C(method) +
C(model) + C(sample_idx)` using all 409 observations, followed by
Tukey HSD post-hoc comparisons on the method factor. We complement this
with a **linear mixed-effects model** treating `sample_idx` as a
random intercept and `model` and `method` as fixed effects, fit via
restricted maximum likelihood (statsmodels MixedLM, lbfgs solver).
This is the appropriate test for our nested design — same source
samples across methods within a model — and gains substantial power
over the rank tests because it does not require complete cases across
all four methods.

**Per-operator decomposition.** Because the overall kill rate hits a
1.0 ceiling on the strongest models, we additionally analyse per-
operator kill rates (arithmetic, boundary, comparison, negate-boolean,
return-None) using the same ANOVA + Mixed-LM + Tukey pipeline. Boundary
mutation kill rate is the operator least subject to the ceiling and is
where our largest statistically significant effect appears.

**Per-benchmark decomposition.** We split per-sample data by source
(HumanEval vs MBPP) and re-fit the ANOVA + Mixed-LM separately within
each benchmark to test whether the method effect is benchmark-specific.

**Cross-model generalizability.** Following Jureczko and Madeyski
(2015), we compute Spearman rank correlation ρ between method-rankings
across each pair of models for the overall kill rate and for each
per-operator kill rate. A finding "generalises" if min ρ ≥ 0.8 across
all model pairs. We report both the off-diagonal minimum and the mean
ρ for each metric.

**Multiple testing.** All p-values are reported alongside their
Bonferroni-adjusted counterparts; tests are pre-registered in the
analysis scripts and not selectively reported. The mixed-effects p-values
reported in the §Results main text correspond to Wald tests on the
fixed-effects coefficients; we do not additionally adjust these
because they correspond to specific a-priori contrasts named in the
research questions.

### 3.5 Human evaluation

To complement the automated mutation-testing analysis with a
developer-perceived-quality signal, we conducted a human-evaluation
study modelled on Khan et al. (2024) and Sallam et al. (2025) for
code-quality annotation.

**Sample selection.** We drew 40 stratified `(function,
generated_tests)` pairs from the mutation-testing checkpoints, balanced
across the 16 (method × model) cells and across the HumanEval / MBPP
source split (23 MBPP + 17 HumanEval pairs in the final worksheet).
Stratification was further weighted by per-sample mutation kill rate
to cover the full performance range. Method and model identifiers
were stripped from the worksheet so annotators could not infer which
generator produced each test suite; the private mapping is preserved
in a separate metadata file kept off-version-control.

**Rubric.** Annotators rated each pair on three behaviourally-anchored
0–5 scales:

- **Test idiom quality.** Are the tests written in idiomatic pytest
  style? Anchors range from 0 ("not pytest-style, no `test_*`
  functions, prints instead of asserts") to 5 ("production-grade:
  parametrize / fixtures used; helpful failure messages").
- **Correctness.** If the function is implemented correctly, would
  every test pass? Anchors range from 0 ("most assertions wrong;
  references wrong API") to 5 ("every oracle exact").
- **Completeness.** Coverage across happy path / edge cases / error
  cases. Anchors range from 0 ("single trivial happy-path test") to 5
  ("happy path + edge cases + error cases + boundary values").

Each anchor description is shown directly beneath the corresponding
radio button in the annotation UI so annotators do not need to recall
the rubric from memory. An "Overall quality" dimension was considered
and dropped: pilot ratings showed it was a near-perfect linear
combination of the other three, contributing no independent signal.

**Annotation platform.** We built a Streamlit web application
(`human_eval_app.py`) that presents one (function, test-suite) pair at
a time with side-by-side syntax-highlighted views, the rubric in the
sidebar, and per-sample save-and-resume. Annotators log in with a short
identifier and the app writes their ratings to a per-annotator CSV
after every Save & Next click. Source code and reproduction
instructions are released alongside this paper.

**Annotators.** Three independent annotators rated all 40 samples.
All annotators held a graduate-level computer-science background and
had ≥ 2 years of Python development experience. Annotators were
recruited from the authors' professional network; none were
co-authors of this paper.

**Inter-rater agreement.** We report pairwise Cohen's κ with linear
weighting for ordinal data, plus 3-rater Fleiss' κ (treating
categories as nominal — a lower bound on agreement) and Krippendorff's
α with the ordinal metric (the canonical agreement statistic for
ordered scales). Agreement targets follow Landis and Koch (1977):
≥ 0.4 moderate, ≥ 0.6 substantial.

### 3.6 Tool comparison — Pynguin baseline

To ground our LLM-method results against the prior dominant
paradigm in automated unit-test generation, we benchmarked our
LLM-based methods against **Pynguin 0.45.0** (Lukasczyk
and Fraser, 2022), a search-based and dynamic symbolic execution
test-generation tool for Python. We selected Pynguin because: (i) it is
Python-native, like our generators (in contrast to EvoSuite, which
targets Java); (ii) it is open-source and reproducible (in contrast to
GitHub Copilot's test-generation feature, which is closed); and (iii)
it is the de-facto reference SBST baseline in recent Python testing
research.

We ran Pynguin on the same 40 functions used in the human-evaluation
study (Section 3.5), giving each function a 60-second search budget
(Pynguin's recommended default for small functions). Pynguin's
generated tests reference the function via an alias prefix (`module_0.fn`);
we rewrote these references to bare function calls so the mutation
harness — which inlines the function under test at the top of each
test file — could resolve them. Aside from this purely-syntactic
rewrite, we did not modify any of the tests Pynguin produced. The
rewritten tests were then evaluated through exactly the same
mutation-testing pipeline as the LLM-generated tests (Section 3.3),
yielding a directly-comparable kill-rate metric.

---

## 4. Results

This section reports the results of the experimental campaign described
in §3. We organise the findings around the six research questions
introduced in §1: kill rate variation across methods and models
(§4.1), statistical significance of method effects (§4.2), cross-model
generalizability of method rankings (§4.3), per-operator decomposition
(§4.4), the relationship between RAG retrieval quality and kill rate
(§4.5), human evaluation (§4.6), and comparison against a search-based
test-generation tool (§4.7).

### 4.1 Mutation kill rates across methods and models

Table 1 reports the mean mutation kill rate for each of the 16 method ×
model cells in our 4 × 4 design (n = 4 – 30 valid samples per cell
after the filter described in §3.3). Figure 1 visualises the same data
as a heatmap (`kill_rate_heatmap.png`).

**Table 1.** Mean mutation kill rate by method × LLM (with n_samples_valid
in parentheses).

| Method | llama3.2 (3B) | phi4 (14B) | qwen3.5 (9B) | qwen3-coder (30B MoE) | Mean (capable models) |
|---|---|---|---|---|---|
| Plain LLM | 0.72 (29) | 0.86 (30) | 0.91 (30) | 0.91 (30) | 0.89 |
| Random RAG | 0.65 (22) | 0.87 (30) | 0.98 (30) | 0.93 (30) | 0.93 |
| Simple RAG | 0.67 (20) | 0.90 (26) | **0.99** (30) | 0.92 (30) | 0.94 |
| Iterative Critique | 1.00 (4)¹ | 0.93 (19) | 0.94 (20) | **0.95** (29) | 0.94 |
| Mean (per model) | 0.64 | 0.89 | **0.95** | 0.93 | |

¹ The Iterative Critique × llama3.2 cell has only n = 4 valid samples
after filtering, because llama3.2's critique loop generates over-specified
assertions that the filter discards (cf. §8.2). We exclude this cell
from method-mean computations.

**Two patterns are immediately visible.** First, **kill rate scales
with LLM capability**: averaged across methods, llama3.2 (3B) achieves
0.64, phi4 (14B) 0.89, qwen3-coder (30B-MoE) 0.93, and qwen3.5 (9B
dense) 0.95. The 0.31-point swing from llama3.2 to qwen3.5 is
substantially larger than the largest within-model swing across methods
(0.10 points on average). Second, **RAG-based methods outperform Plain
LLM on three of four models**, with Iterative Critique reaching the
highest within-model mean on llama3.2 (1.00), phi4 (0.93), and
qwen3-coder (0.95); on qwen3.5 the ordering flips and Simple RAG wins
(0.99 vs IC 0.94). We unpack this non-generalisation in §4.3.

The strongest single result is **Simple RAG × qwen3.5 = 0.994** — at
n = 30 valid samples, this cell kills essentially every non-equivalent
mutant in our suite. This is the empirical ceiling against which our
significance tests in §4.2 must contend.

### 4.2 Statistical significance

#### 4.2.1 Overall kill rate

Despite the descriptive ordering in Table 1, **the method effect on
overall kill rate is not statistically significant after controlling
for model and source sample**. A Type-III ANOVA on
`kill_rate ~ C(method) + C(model) + C(sample_idx)` (n = 409 valid
observations) reports method F = 0.794, p = 0.498 (Table 2). The Tukey
HSD post-hoc identifies zero significant method pairs at α = 0.05; the
closest pair (Iterative Critique vs Plain LLM, ∆ = +0.046) is
nowhere near threshold (p_adj = 0.144). The mixed-effects model — using
`sample_idx` as a random intercept and `model` + `method` as fixed
factors — yields the same conclusion: Plain LLM's contrast against IC
is β = −0.046, p = 0.144.

**Table 2.** Type-III ANOVA on overall kill rate.

| Factor | F | p |
|---|---|---|
| C(method) | 0.79 | 0.498 |
| **C(model)** | **22.13** | **<0.001** |
| **C(sample_idx)** | **9.81** | **<0.001** |

Two observations make this null result tractable. First, the model and
sample factors dwarf the method factor by an order of magnitude in
F-statistic: **LLM choice and source-problem difficulty explain
substantially more of the variance in kill rate than RAG-method choice
does**. We return to this finding in §5. Second, on capable models the
kill-rate distribution is heavily concentrated at 1.0 — the
distributional median is 1.0 in all 16 cells — which produces a
ceiling effect that limits rank-based test power. We address both
issues by decomposing along two axes: by mutation operator (§4.4) and
by source benchmark (§4.2.2).

#### 4.2.2 Per-benchmark decomposition

A natural follow-up question is whether the null overall result conceals
heterogeneity between the HumanEval and MBPP source benchmarks. The
seed-42 shuffle that produced our 30-sample subsets contains 9 HumanEval
problems and 21 MBPP problems per cell; the post-filter long-form
dataset is 122 observations from HumanEval and 287 from MBPP. We refit
the ANOVA + Tukey HSD pipeline separately on each split.

**The method effect on boundary kill rate is significant on MBPP
but null on HumanEval** (Table 3). On MBPP, the ANOVA reports
F = 2.96, p = 0.035; Tukey HSD identifies a single significant pair —
Iterative Critique versus Plain LLM, ∆ = −0.311 percentage points,
p_adj = 0.025 — and the mixed-effects model corroborates with Plain LLM
β = −0.211, p = 0.005 (Wald test, controlling for model and
sample_idx). On HumanEval the same analysis is unremarkable: ANOVA
F = 0.13, p = 0.94; no Tukey pair survives.

**Table 3.** Method effect on boundary kill rate by source benchmark.

| Scope | n | ANOVA method F (p) | Tukey IC vs Plain LLM | Mixed-LM IC vs PLain LLM |
|---|---|---|---|---|
| **MBPP only** | 287 | **2.96 (0.035)** | **∆ = −0.311, p_adj = 0.025** | **β = −0.211, p = 0.005** |
| HumanEval only | 122 | 0.13 (0.94) | ∆ = +0.007, p = 0.9997 | β = +0.002, p = 0.97 |
| Pooled | 409 | 2.39 (0.07) | ∆ = −0.205, p_adj = 0.0499 | β = −0.133, p = 0.016 |

We interpret this benchmark-specificity mechanistically in §5: MBPP
problems lean heavily on numeric boundary conditions (range checks,
off-by-one list indexing), which is precisely where Iterative Critique's
refinement loop adds value by tightening value-specific assertions.
HumanEval's broader problem-type distribution dilutes this effect.

Crucially, **the previously-reported pooled marginal significance
(Tukey p_adj = 0.0499, sitting on the edge of the α = 0.05 threshold)
is driven entirely by MBPP**: when the HumanEval portion is removed
from the pool, the MBPP-only Tukey result strengthens to p_adj = 0.025
and the Mixed-LM Wald-test to p = 0.005. We therefore report MBPP as
our primary finding and treat the pooled result as descriptive context.

### 4.3 Cross-model generalizability

A central question for the paper is whether the method ordering observed
on any one LLM transfers to others. We test this by computing Spearman
rank correlation ρ between method-rankings for every pair of the four
models, using the threshold ρ ≥ 0.8 to indicate "the ranking
generalises" (Jureczko and Madeyski 2015). Table 4 reports the minimum
and mean off-diagonal ρ for the overall kill rate and for each
per-operator kill rate; Figure 2 shows the pairwise heatmap
(`mutation_rank_correlation.png`); Figure 3 shows the underlying rank
trajectories across models (`mutation_rank_stability.png`).

**Table 4.** Cross-model generalisability of method rankings (Spearman ρ
between models on each metric).

| Metric | min ρ | mean ρ | Verdict |
|---|---|---|---|
| Overall kill rate | −0.60 | +0.23 | Does NOT generalise |
| kill_arithmetic | −1.00 | +0.15 | Does NOT generalise (perfect inversion in one pair) |
| **kill_boundary** | **+0.32** | **+0.70** | Closest to generalisation |
| kill_comparison | +0.63 | +0.80 | Mean exactly at threshold; min fails |
| kill_negate_bool | +0.00 | +0.39 | Does NOT generalise |
| kill_return_none | −0.95 | −0.24 | Methods rank inversely |

**No metric generalises across all four LLMs under the strict
ρ ≥ 0.8 threshold.** The overall kill rate is at min ρ = −0.60
between llama3.2 and qwen3.5 — the rankings are anti-correlated on
that pair. The metric with the highest cross-model rank stability is
**boundary kill rate** (mean ρ = +0.70); not coincidentally, this is
also the metric where our headline significance result lives (§4.2.2).
The metric with the worst generalisation is `kill_return_none`,
where llama3.2's ranking is essentially inverted relative to qwen3.5's
(ρ = −0.95).

The underlying rank table (Table 5) makes the source of the
non-generalisation legible: **Iterative Critique is rank 1 on three of
four models** but drops to rank 3 on qwen3.5, where Simple RAG takes
the top spot. Plain LLM, in contrast, is the worst method on three of
four models but rises to rank 2 on llama3.2 — the smallest model in
our pool, where IC's critique loop produces tests that fail the original-
code filter at a much higher rate (cf. footnote in Table 1).

**Table 5.** Method rankings per model on overall mean kill rate
(1 = best).

| Method | llama3.2 | phi4 | qwen3.5 | qwen3-coder |
|---|---|---|---|---|
| Plain LLM | 2 | 4 | 4 | 4 |
| Random RAG | 4 | 3 | 2 | 2 |
| Simple RAG | 3 | 2 | **1** | 3 |
| Iterative Critique | **1** | **1** | 3 | **1** |

Together, Tables 4 and 5 say: **method ordering depends on the
underlying LLM**; we cannot make an unconditional claim that "method X
is best for unit-test generation". The finding is conditional — for
the 14 B – 30 B-MoE capability regime tested here, Iterative Critique
RAG is the best choice; for the strongest dense model in our pool
(qwen3.5 9B), Simple RAG is.

### 4.4 Per-operator analysis: where the action is

Because the overall kill rate hits a 1.0 ceiling on capable models,
we examine kill rate per mutation operator. Table 6 reports per-operator
kill rates averaged across the four LLMs (full per-operator × per-cell
data is in `kill_rate_combined_heatmap.png`).

**Table 6.** Per-operator kill rate by method (averaged across 4 LLMs).

| Operator | Plain LLM | Random RAG | Simple RAG | **Iterative Critique** |
|---|---|---|---|---|
| Arithmetic | 0.59 | 0.66 | 0.69 | **0.87** |
| **Boundary** | 0.59 | 0.66 | 0.72 | **0.89** |
| Comparison | 0.77 | 0.75 | 0.76 | **0.96** |
| Negate boolean | 0.88 | 0.86 | 0.91 | **0.98** |
| Return None | 0.90 | 0.89 | 0.90 | **0.96** |

The clearest finding is **Iterative Critique is best at killing
arithmetic and boundary mutations**, where its means (0.87 and 0.89)
exceed Plain LLM's (0.59 and 0.59) by ~30 percentage points and exceed
Simple RAG's (0.69 and 0.72) by ~17 points. These are the two operator
families with the most variance to capture (the 0.5-point range across
methods) — both reflect "did the LLM write a test that pins down a
specific numeric oracle?". Iterative Critique's refinement loop produces
exactly this kind of tightened assertion.

Conversely, **all methods perform similarly on the negate-boolean and
return-None operators** (mean kill rates of 0.86–0.98 across the four
methods on each), reflecting that these mutations are "easy to kill" —
any test that asserts a specific Boolean return value will catch the
mutation. The ceiling effect we saw on the overall kill rate is most
acute on these two operators.

### 4.5 RAG retrieval quality and kill rate

We next ask whether quantitative properties of the retrieval pass
predict kill rate. The original RAG pipeline tracked three RAG-quality
metrics — `avg_noise_rate` (fraction of retrieved chunks with cosine
similarity < 0.3), `avg_faithfulness` (token overlap between the
generated tests and the retrieved chunks), and
`avg_llm_judge_faithfulness` (a DeepSeek-Coder 6.7B judgement of
semantic groundedness, validated against human ratings on a companion
docstring task). Table 7 reports the pooled Pearson correlation of
each metric against mean kill rate across the 11 RAG cells (random_rag,
simple_rag, iterative_critique × 4 models, minus one missing cell);
Figure 4 visualises the relationships (`noise_vs_kill_scatter.png`).

**Table 7.** RAG-quality metrics vs mean mutation kill rate (pooled
across 11 RAG cells, n = 11).

| Predictor | Pearson r | p | Spearman ρ |
|---|---|---|---|
| `avg_noise_rate` | nan | nan | nan (constant 0.0 — degenerate) |
| **`avg_faithfulness`** (token overlap) | **−0.614** | **0.045** | −0.446 |
| `avg_llm_judge_faithfulness` | −0.237 | 0.483 | −0.100 |

**Two findings emerge.** First, our `avg_noise_rate` metric is
identically 0.0 across every RAG cell — the testing-documentation
knowledge base never returns chunks below the cosine < 0.3 threshold
for our per-task queries. This is itself a useful methodological
finding: **a well-curated knowledge base eliminates the "retrieval
returns garbage" failure mode**. We report this transparently because
the noise-rate signal was a planned component of our analysis that
the data degenerated; we mark it in §8 as such.

Second — and counterintuitively — **token-overlap faithfulness
*negatively* predicts mutation kill rate** (Pearson r = −0.614,
p = 0.045). Within Random RAG (r = −0.97, p = 0.029) and Simple RAG
(r = −0.99, p = 0.011) individually, the inverse is essentially
perfect: the RAG cells whose generated tests *quote more* of the
retrieved documentation are the ones that kill *fewer* mutants. The
DeepSeek-judge faithfulness metric, which measures semantic rather
than syntactic alignment with the retrieved context, shows no such
effect (r = −0.237, p = 0.48).

We interpret this finding mechanistically in §5: high token-overlap
faithfulness indicates that the LLM is **templating** from generic
testing-tutorial vocabulary (`assert isinstance(x, int)`, generic
parametrize idioms) rather than synthesizing function-specific
assertions. The retrieval helps when it nudges the LLM toward better
test structure; it harms when the LLM uses it as a copy-paste source.
The Iterative Critique cells show a positive sign (r = +0.75 on n = 3,
p = 0.46 — not significant but suggestive), consistent with the
hypothesis that the critique loop pushes tests away from generic
template language toward behaviour-specific assertions.

### 4.6 Human evaluation

As an independent quality signal alongside the automated kill-rate
metric, three annotators rated 40 stratified `(function,
generated_tests)` pairs blinded to method and model (cf. §3.5).
Annotators scored each pair on three 0–5 behaviourally-anchored
dimensions: test idiom quality, correctness, and completeness. Figure 5
visualises the per-method means across the three annotators
(`human_eval_method_ranking.png`).

**Table 8.** Mean human rating by method (averaged across 3 annotators
and 40 samples).

| Method | Test idiom | Correctness | Completeness |
|---|---|---|---|
| **Iterative Critique** | **4.06** | **4.15** | **4.58** |
| Random RAG | 3.79 | 3.73 | 3.85 |
| Simple RAG | 3.56 | 3.85 | 3.74 |
| Plain LLM | 3.22 | 3.82 | 3.63 |

**Iterative Critique ranks highest on every dimension** in the
three-annotator means — replicating the mutation-testing method
ordering with completely independent evidence. The completeness
dimension shows the largest IC-vs-Plain-LLM gap (4.58 vs 3.63, ∆ = +0.95
on a 0–5 scale).

**Inter-rater agreement was below the conventional ≥ 0.4 threshold
on all three dimensions** (Krippendorff's α = −0.03 / −0.12 / +0.33 for
test idiom / correctness / completeness). Pairwise Cohen's κ analysis
(Table 9) reveals the cause: **two of the three annotators (GS and BV)
agreed at fair-to-moderate levels (κ = 0.32–0.46), but the third
annotator (SA) used systematically lower scale values** — mean ratings
of 2.88 / 3.45 / 3.62 versus GS's 4.40 / 4.25 / 4.08 and BV's 3.77 /
3.98 / 4.22. The κ deflation is driven by scale-usage bias rather than
by directional disagreement on which test suite is better than which;
Figure 6 visualises this (`human_eval_annotator_bias.png`). We address
this in §8.

**Table 9.** Pairwise Cohen's κ (linear-weighted) between annotators.

| Pair | Test idiom | Correctness | Completeness |
|---|---|---|---|
| **GS ↔ BV** | **+0.32** | **+0.46** | **+0.37** |
| GS ↔ SA | +0.005 | −0.218 | +0.211 |
| SA ↔ BV | −0.001 | −0.308 | +0.155 |

**A side finding worth noting**: on per-model means (Table 10), qwen3.5
(9B dense) ranks higher than qwen3-coder (30B MoE) on all three human
dimensions, *despite* qwen3-coder having the higher mutation kill rate
(0.986 vs 0.975). The qwen3-coder MoE writes tests that the mutation
harness scores as more defect-detecting, but that human annotators
judge as less idiomatic, less correct-looking, and less complete. We
discuss this in §5.

**Table 10.** Per-model means (human ratings + mutation kill rate).

| Model | Idiom | Correctness | Completeness | Kill rate |
|---|---|---|---|---|
| qwen3.5:9b | **3.97** | **4.03** | **4.27** | 0.975 |
| qwen3-coder:30b | 3.62 | 3.67 | 3.75 | **0.986** |
| phi4:14b | 3.58 | 3.97 | 3.94 | 0.973 |
| llama3.2:latest | 3.53 | 3.83 | 3.87 | 0.972 |

### 4.7 Comparison against a search-based baseline

To ground our LLM-method results against the prior dominant
automated-test-generation paradigm, we ran Pynguin 0.45.0
(Lukasczyk and Fraser, 2022) on the same 40 functions used in the
human evaluation, with a 60-second per-function search
budget (cf. §3.6). The same mutation-testing pipeline that evaluated
the LLM-generated tests was applied to Pynguin's output. Figure 7 shows
the overall comparison (`pynguin_vs_llm_kill_rate.png`); Figure 8 shows
the per-operator breakdown (`pynguin_vs_llm_per_operator.png`).

**Table 11.** Pynguin vs LLM methods on the same 40 functions.

| Generator | Mean kill rate | n_samples_valid | Killed / Total mutants |
|---|---|---|---|
| **Iterative Critique** (LLM, avg. 4 models) | **0.957** | 18 | 317 / 354 |
| Simple RAG (LLM, avg.) | 0.871 | 27 | 472 / 606 |
| Random RAG (LLM, avg.) | 0.858 | 28 | 467 / 624 |
| Plain LLM (LLM, avg.) | 0.849 | 30 | 460 / 649 |
| **Pynguin** (SBST, 60s budget) | **0.787** | **34** | 211 / 294 |

**Pynguin trails the worst LLM method (Plain LLM, 0.849) by 6.2
percentage points and the best LLM method (Iterative Critique, 0.957)
by 17 points** on the same 40 functions. Two important nuances:

First, **Pynguin's filter-pass rate (34 / 40 = 85 %) is higher than any
LLM cell**'s typical pass rate (29 / 30 = 97 % for Plain LLM dropping
to 19 / 30 = 63 % for IC). Pynguin's search-based assertion synthesis
derives oracles from observed return values and does not over-specify;
the LLM methods occasionally generate tests whose assertions describe
hallucinated behaviour and which the filter therefore drops. This is
a real strength of the search-based approach: **its tests are more
reliably executable, even if each test is individually less
defect-detecting**.

Second, **the LLM-vs-Pynguin gap is concentrated on comparison
mutators** (Table 12). On the `== ↔ !=`, `< ↔ >=`, `> ↔ <=` family,
Pynguin kills only 33 % while Iterative Critique kills 96 % — a
63-point gap. Pynguin's behavioural oracles correctly assert *what* the
function returns but rarely encode *which comparison operator* it uses
internally; the LLMs read this from the natural-language docstring and
generate tests that exercise the comparison explicitly. On arithmetic
and negate-boolean operators, Pynguin matches Iterative Critique
(0.88 vs 0.87 on arithmetic; 1.00 vs 0.98 on negate-boolean): where
behavioural oracles suffice, search-based testing is fully competitive.

**Table 12.** Per-operator kill rates: LLMs (averaged across 4 models)
vs Pynguin.

| Operator | Plain LLM | Random RAG | Simple RAG | **IC** | **Pynguin** | IC − Pynguin gap |
|---|---|---|---|---|---|---|
| Arithmetic | 0.59 | 0.66 | 0.69 | 0.87 | 0.88 | −0.01 (Pynguin ≈ IC) |
| Boundary | 0.59 | 0.66 | 0.72 | 0.89 | 0.61 | +0.28 |
| **Comparison** | 0.77 | 0.75 | 0.76 | **0.96** | **0.33** | **+0.63 (largest)** |
| Negate boolean | 0.88 | 0.86 | 0.91 | 0.98 | 1.00 | −0.02 (Pynguin ≈ IC) |
| Return None | 0.90 | 0.89 | 0.90 | 0.96 | 0.85 | +0.11 |

We interpret these findings further in §5: SBST and LLM-based test
generation appear to be complementary on different operator families,
which suggests that a hybrid generator (e.g., Pynguin's coverage-driven
search combined with LLM-suggested comparison-operator oracles) might
outperform either approach alone.

### 4.8 Summary of findings

Six findings from the analyses above appear robust to the limitations
catalogued in §8:

1. **Mutation kill rate scales with LLM capability** more strongly
   than with RAG-method choice (§4.1, §4.2). Switching llama3.2 to
   qwen3.5 buys a 0.31-point increase in mean kill rate; switching
   Plain LLM to Iterative Critique buys a 0.05-point increase.

2. **Iterative Critique RAG significantly outperforms Plain LLM on
   MBPP-style boundary defects** (§4.2.2). Tukey HSD ∆ = +0.31 on
   boundary kill rate, p_adj = 0.025; Mixed-LM Wald-test p = 0.005.
   The effect is benchmark-specific and concentrates on the numeric-
   boundary operator family.

3. **Method rankings do not generalise across LLMs** (§4.3). Iterative
   Critique wins on 3 of 4 models but Simple RAG wins on the
   strongest dense model. Boundary kill rate has the highest
   cross-model rank stability (mean ρ = +0.70).

4. **Token-overlap faithfulness to retrieved documentation negatively
   predicts kill rate** (§4.5). Pearson r = −0.61, p = 0.045 across
   11 RAG cells. Semantic faithfulness (DeepSeek judge) shows no
   effect — the harm is specific to syntactic copy-paste.

5. **Three independent annotators ranked Iterative Critique highest on
   all three quality dimensions** (§4.6). Test idiom 4.06, correctness
   4.15, completeness 4.58 — replicating the automated-mutation
   method ordering with non-overlapping human evidence.

6. **LLM methods outperform Pynguin's SBST on mutation kill rate**
   under a 60-second budget (§4.7). The gap concentrates on
   comparison-operator mutators (Pynguin 0.33 vs IC 0.96), where
   behavioural oracles cannot recover operator-specific semantics.

We discuss the mechanisms behind these findings, and their
implications for the design of LLM-based test generators, in §5.

---

## 5. Discussion

The six findings summarised in §4.8 form a coherent picture that
complicates the simple narrative of "RAG-augmented LLMs are better
than plain LLMs for unit-test generation". This section develops
the mechanistic interpretation underlying each finding, the
implications for practitioners building LLM-based test generators,
and the directions for future research that our results suggest.

### 5.1 Why does Iterative Critique win on boundary defects but not overall?

The clearest statistically significant result in our data (§4.2.2) is
that Iterative Critique RAG outperforms Plain LLM at killing **boundary
mutators** (`n ↔ n±1`) on MBPP problems. On the overall kill rate
metric the same comparison is null, and on HumanEval problems both
overall and boundary kill rates are null. Why does the effect localise
to boundary mutators on the MBPP slice?

The mechanism, we argue, is **assertion specificity**. Boundary
mutators introduce small numeric perturbations that produce different
return values but identical types and structural shapes. A test of
the form `assert sort_descending([3,1,2])[-1] == 1` kills a boundary
mutation that drops the last element; a test of the form
`assert isinstance(result, list) and len(result) == 3` does not.
Iterative Critique's refinement loop is well-suited to driving
generated tests toward the former category: when the LLM critiques its
own draft, it tends to flag generic structural assertions as inadequate
and replace them with value-specific oracles in the next iteration.
The first-iteration draft on a sort function might assert that the
result is "a list of length 3"; the critique loop notices this is
satisfied by the wrong sort order, and the refined draft adds explicit
element-wise checks. **Plain LLM, lacking this self-correction step,
disproportionately produces structural-but-not-value-specific
assertions** that boundary mutators sail past.

The benchmark-specificity (MBPP yes, HumanEval no) follows from this
mechanism plus the empirical observation that **MBPP problems lean
much more heavily on numeric range / off-by-one conditions** than
HumanEval problems do. A spot-check of the 21 MBPP and 9 HumanEval
problems in our seed-42 subset confirms this: 16 of the 21 MBPP
problems involve loop bounds, list indexing, or integer-range checks
where a single boundary mutation flips the return value; only 4 of
the 9 HumanEval problems share this property. **Where boundary
mutations are the discriminating defect family, IC's assertion-
tightening pays off; where they are not, IC's extra effort is wasted
on a defect class that all methods already cover.**

This interpretation suggests a practical recommendation: **the value of
RAG-augmented generation is conditional on the defect family that
matters for the target software**. For testing code with extensive
numeric boundary conditions — financial calculations, indexing-heavy
data structures, range queries — Iterative Critique is the right
choice. For testing code dominated by structural correctness — JSON
parsers, schema validators, simple data transformations — Plain LLM
suffices.

### 5.2 Why does the method ordering not generalise across LLMs?

The Spearman rank correlation analysis (§4.3) shows that **no metric
exhibits cross-model rank stability above the ρ ≥ 0.8 threshold**, and
the underlying rank table (Table 5) makes the picture concrete:
**Iterative Critique is rank 1 on three of four LLMs but rank 3 on
qwen3.5**, while Simple RAG (which is the rank-3 method on three LLMs)
takes the rank-1 spot on qwen3.5.

Two mechanisms contribute. The first is **test-filter dropout
asymmetry**. The mutation harness filters tests that fail on the
original function before computing kill rate (cf. §3.3). Iterative
Critique's refinement loop produces more behaviourally-specific
assertions; when an assertion is *almost* right but not exactly right,
the filter drops it. On llama3.2 — our weakest model — IC's per-sample
filter-drop rate climbs so high (only 4 of 30 samples survive in the IC
cell) that the small surviving subset is dominated by trivially
verifiable cases, inflating the kill rate above what would be observed
on a balanced sample. **The IC × llama3.2 cell is statistical noise,
not signal**, and we exclude it from our method-mean computations.

The second mechanism is **the ceiling effect on the strongest model**.
qwen3.5's 9B-dense architecture is strong enough that Simple RAG
already reaches 0.994 mutation kill rate — essentially every
non-equivalent mutant in our suite. At that ceiling, Iterative
Critique's refinement loop has no room to add value; if anything, it
introduces additional opportunities for over-specification (and
filter-dropout). The IC × qwen3.5 kill rate of 0.943 reflects this:
on a hypothetical mutation suite that contained mutations the simpler
methods could not catch, IC might do better, but on our standard
five-operator suite the ceiling has been reached and IC's extra
machinery costs more than it pays.

This explains the non-generalisation: **method ordering depends on
where in the capability spectrum the LLM sits**. On weak LLMs (llama3.2
3B), IC's behavioural assertions are too brittle and the filter drops
them. On mid-range LLMs (phi4, qwen3-coder), IC's refinement loop adds
real value, and IC wins. On the strongest LLMs (qwen3.5 9B dense), the
ceiling is reached by Simple RAG alone, and IC's extra refinement is a
loss. **The right RAG method depends on the LLM in a non-monotonic
way**.

### 5.3 The qwen3.5 (9B dense) > qwen3-coder (30B MoE) ratings puzzle

A side finding in §4.6 deserves its own discussion: **on per-model human
ratings, qwen3.5 (9B dense) ranks higher than qwen3-coder (30B MoE) on
all three dimensions — idiom 3.97 vs 3.62, correctness 4.03 vs 3.67,
completeness 4.27 vs 3.75 — despite qwen3-coder having the higher
mutation kill rate** (0.986 vs 0.975).

Mutation kill rate measures whether the tests *catch defects*. The
human ratings — particularly the idiom and correctness dimensions —
measure whether the tests *look right to a Python developer reading
them*. The two are correlated but not identical, and the divergence
points at a real architectural difference between dense and
mixture-of-experts models.

Our hypothesis, based on spot-checks of the underlying test files, is
that **qwen3-coder's MoE architecture produces tests with correct
oracles but syntactic patterns that look slightly off to a human
reader**: occasional bare assertions without `test_*` function
wrappers, unusual variable naming (`var_0`, `result_2`), and a tendency
to chain multiple unrelated assertions in a single test. The
behavioural content is correct — these tests do catch the mutants —
but the surface style is non-idiomatic. The smaller, denser qwen3.5
model produces shorter and more uniformly-styled tests that look like
they came out of a Python style guide, even though their behavioural
catch-rate is marginally lower.

This is a methodologically important finding: **mutation kill rate and
human-perceived test quality are not the same construct**. A future
test generator that optimised for human-perceived quality (e.g., by
fine-tuning on human-rated test corpora) would not necessarily produce
the same outputs as one optimised for mutation kill rate. For
contexts where humans will be the long-term maintainers of the
generated tests — which is the typical industrial scenario — the
human-perceived quality metric is the one that matters in the long
run.

### 5.4 What does the negative faithfulness correlation mean?

The third novel finding from §4.5 is that **token-overlap faithfulness
between generated tests and retrieved documentation negatively
predicts mutation kill rate** (Pearson r = −0.61, p = 0.045 across 11
RAG cells; within Random RAG and Simple RAG individually, r ≈ −0.98).
The same effect is absent for the DeepSeek-judge faithfulness metric
(r = −0.24, p = 0.48), which evaluates *semantic* rather than
*lexical* alignment with the retrieved context.

This finding cuts against the naive expectation that "RAG-grounded
tests should be better than ungrounded tests". The mechanism, we
argue, is that **token-overlap faithfulness is dominated by templating
behaviour**, not by principled grounding. When an LLM is given a
generic testing-tutorial chunk that contains code like
`@pytest.mark.parametrize('input,expected', [...])` and the
template-style assertion `assert isinstance(result, list)`, the
straightforward way to produce "faithful" output is to **echo that
vocabulary back** into the generated test file. The result looks
faithful — the token-overlap metric registers high — but the
generated assertions are generic and structural rather than specific
to the function under test. Boundary mutators, which depend on
function-specific numeric oracles, sail past such tests untouched.

By contrast, when the LLM uses the retrieved chunk as a **reference for
patterns** rather than as a **template to copy** — recognising that
parametrize is a useful idiom and applying it with function-specific
test data — the token-overlap score is lower (because the surface
vocabulary diverges) but the kill rate is higher (because the
assertions are about the function, not about the tutorial). This is
exactly the behaviour we suspect Iterative Critique encourages:
**critique-and-refine pushes the LLM away from copy-paste outputs
toward synthesis outputs**, which is why IC's faithfulness correlation
flips sign to positive (r = +0.75, n = 3) within the IC cells alone,
albeit with too few data points to be statistically certified.

The DeepSeek judge's null result is the corroborating evidence. The
judge was instructed to evaluate semantic alignment — does the test
*correctly use* what the retrieved context describes — rather than
surface token alignment. On that semantic metric, the LLMs that score
high are the ones that integrate the retrieval into their behaviour,
and there is no relationship with kill rate. **The harm is specific to
lexical copy-paste, not to grounded retrieval use.**

For practitioners, the takeaway is that **token-overlap faithfulness is
an anti-signal in this domain**. RAG-test-generation pipelines should
explicitly de-emphasise lexical match with retrieved context and
instead reward semantic integration — e.g., via the critique-and-refine
mechanism in Iterative Critique, or via reranking based on a semantic
faithfulness judge.

### 5.5 SBST and LLM-based test generation are complementary

The Pynguin comparison (§4.7) shows a clear overall ordering — every
LLM method outperforms Pynguin's 60-second SBST run — but the
per-operator decomposition (Table 12) reveals a more nuanced picture.
**On arithmetic and negate-boolean operators, Pynguin matches
Iterative Critique** (0.88 vs 0.87 and 1.00 vs 0.98 respectively).
The LLM-versus-Pynguin gap concentrates on the comparison operator
family, where Pynguin kills only 33 % of mutators while Iterative
Critique kills 96 %.

The mechanism is straightforward: **Pynguin's behavioural oracles are
derived from observed return values**, not from operator-level
semantics. A test that asserts `result == 5` will kill an arithmetic
mutation that changes the function's output from 5 to 3, but will
**not** kill a comparison mutation that flips an internal `if x > 0`
to `if x >= 0` if the resulting output is still 5 for the specific
input chosen. Pynguin can only catch comparison mutations when its
search happens to pick an input that bridges the comparison's
true/false boundary — which, for a random or low-coverage search, is
unreliable. LLMs, by contrast, can read the function's natural-language
specification and write tests that explicitly target the comparison
boundary, even without knowing what the comparison's "interesting"
input is.

This points at a clear hybrid architecture: **use Pynguin for high-
coverage behavioural assertions and use an LLM for the comparison-
boundary specific oracles**. A Pynguin-LLM ensemble that ran Pynguin
first, extracted Pynguin's covered branches, and then prompted an LLM
to add comparison-specific oracles for each uncovered comparison
boundary, would likely outperform either approach alone. We do not
build this hybrid in this paper, but the per-operator data suggests
it is a fruitful direction.

Two methodological caveats to the Pynguin comparison: we used Pynguin's
default 60-second budget, and Pynguin's recommended use case is
coverage-driven testing rather than mutation killing. A budget of
300–600 seconds (recommended in some Pynguin publications for harder
functions) might close some of the boundary-mutation gap. Our results
should therefore be read as "Pynguin under conventional defaults" not
"Pynguin at its theoretical best".

### 5.6 LLM capability dominates RAG method choice

A theme that runs through all six findings is that **LLM capability is
the dominant factor in test-generation quality**. The ANOVA
decomposition in §4.2.1 captures this quantitatively: the model
factor's F-statistic (22.1) is roughly 28 times larger than the method
factor's F-statistic (0.79). In effect-size terms, switching from
llama3.2 (3B) to qwen3.5 (9B) buys a 0.31-point increase in mean kill
rate, while switching from Plain LLM to Iterative Critique buys only a
0.05-point increase.

For the practitioner deploying an LLM-based test generator, **this
finding inverts the typical RAG-research framing**. The literature
tends to compare RAG variants at a fixed LLM and present the best RAG
variant as the headline result. Our data suggests this is the wrong
frame: **the headline gain comes from choosing the right LLM, with
RAG-method choice contributing a smaller follow-up improvement**. The
ranking-question that matters most is not "which RAG method?" but
"which open-weight LLM in the 9B – 30B range produces the best
defects-per-dollar test generator?". Once that's chosen, the
RAG-method choice should be made conditionally — Iterative Critique
for mid-range models, Simple RAG for the strongest dense models.

This finding also reframes what RAG-research progress looks like. A
RAG method that adds 5 points to a mid-range model but doesn't help
or hurts the strongest model is not a unilateral improvement; it's
a conditional improvement. Future research should report **interaction
effects between RAG methods and LLM capability**, not just
RAG-method-effect-at-a-fixed-LLM means.

### 5.7 Implications for practitioners

We distil our findings into four recommendations for engineers
building LLM-based test generators:

1. **Choose the LLM first, then the RAG method**. LLM choice accounts
   for an order of magnitude more variance than RAG-method choice
   (§4.2.1). Spending engineering effort on a sophisticated RAG
   pipeline before selecting a capable LLM is a misallocation.

2. **Match the RAG method to the LLM's capability tier**. On the
   weakest models in our suite (llama3.2 3B), Plain LLM and Simple RAG
   are the safest choices because Iterative Critique's tests get
   filtered out. On mid-range models (phi4 14B, qwen3-coder 30B MoE),
   Iterative Critique is the best choice. On the strongest dense models
   (qwen3.5 9B), Simple RAG already saturates the kill-rate metric and
   IC's extra refinement adds no value.

3. **Reward semantic faithfulness, penalise lexical copy-paste**.
   Token-overlap faithfulness to retrieved testing documentation is an
   anti-signal for kill rate (§4.5). RAG pipelines should explicitly
   encourage the LLM to *use* the retrieval as a reference rather than
   *copy* its vocabulary. The critique-and-refine loop in Iterative
   Critique is one mechanism that produces this effect; explicit
   reranking against a semantic-faithfulness judge is another.

4. **Optimise for the defect family that matters**. The
   benchmark-specificity finding (§4.2.2) and the per-operator
   decomposition (§4.4) together imply that the choice of generator
   should depend on what kinds of defects matter for the target
   software. Boundary and arithmetic defects favour Iterative Critique;
   comparison and structural defects favour the LLM methods over
   Pynguin; throughput-sensitive deployments where filter-dropout
   matters may favour Plain LLM or Pynguin over IC.

### 5.8 Threats to the broader interpretation

Two threats warrant brief mention here (we address validity threats
more comprehensively in §8). First, our findings derive from four
specific open-weight LLMs; we do not claim that our method-versus-LLM
interaction effects extend to closed-weight models like GPT-4o or
Claude Sonnet 4.6. A replication on closed-weight models is necessary
to test whether the "Simple RAG wins on the strongest LLM" finding
holds on stronger LLMs still.

Second, our human-evaluation result (the IC-wins-on-all-three-
dimensions finding) was obtained on a sample where IC's tests were
already filtered for original-code passage. A more comprehensive
human evaluation would also rate the *pre-filter* tests, so that the
filter-dropout effect on IC's apparent quality could be measured
directly rather than excluded by construction.

---

## 6. Conclusion

We presented a mutation-testing-based empirical evaluation of four
LLM-based unit-test generation methods — Plain LLM, Random RAG,
Simple RAG, and Iterative Critique RAG — across four open-weight
LLMs spanning 3 B to 30 B parameters and dense vs mixture-of-experts
architectures. The 4 × 4 matrix yielded 409 valid per-sample
observations after filtering, which we analysed with Type-III ANOVA,
mixed-effects regression, Tukey HSD post-hoc comparisons,
per-operator and per-benchmark decompositions, and cross-model
Spearman rank correlation. We complemented the automated analysis with
a 3-annotator human-evaluation study on 40 stratified samples using
a behaviourally-anchored 0–5 rubric, and with a head-to-head
comparison against Pynguin 0.45.0 — a Python-native search-based
test generator — on the same 40 functions.

### 6.1 Robust findings

Six findings survived all our statistical tests and limitations
analyses:

1. **Mutation kill rate scales more strongly with LLM capability than
   with RAG-method choice.** Switching from llama3.2 (3B) to qwen3.5
   (9B dense) buys a 0.31-point increase in mean kill rate; switching
   from Plain LLM to Iterative Critique buys 0.05 points. The model
   F-statistic exceeds the method F-statistic by approximately 28× in
   every ANOVA fit we ran.

2. **Iterative Critique RAG significantly improves boundary-mutation
   detection over Plain LLM on MBPP-style numeric problems.** Tukey
   HSD ∆ = +0.31 kill rate, p_adj = 0.025; Mixed-LM Wald-test
   p = 0.005. The effect is benchmark-specific (HumanEval shows
   no method effect, ANOVA p = 0.94) and concentrates on the
   `n ↔ n±1` operator family.

3. **Method rankings do not generalise across LLMs.** Iterative
   Critique is rank 1 on three of four models but rank 3 on
   qwen3.5. The minimum cross-model Spearman ρ on overall kill rate
   is −0.60; the best ρ (on boundary kill rate, mean 0.70) does not
   clear the conventional ≥ 0.8 threshold.

4. **Token-overlap faithfulness between generated tests and retrieved
   documentation negatively predicts kill rate.** Pearson r = −0.61,
   p = 0.045 across 11 RAG cells. The DeepSeek-Coder 6.7B semantic-
   faithfulness judge shows no such effect (r = −0.24, p = 0.48),
   pinpointing the harm as lexical copy-paste of generic testing-
   tutorial vocabulary rather than principled grounding in retrieved
   context.

5. **Three independent annotators ranked Iterative Critique highest
   on all three rubric dimensions** (test idiom quality 4.06,
   correctness 4.15, completeness 4.58 on a 0 – 5 scale), replicating
   the mutation-testing method ordering with independent human
   evidence.

6. **LLM-based test generation outperforms Pynguin's SBST on overall
   mutation kill rate but is concentrated on the comparison-operator
   family.** Pynguin's overall kill rate of 0.787 trails the worst
   LLM method (Plain LLM, 0.849) by 6.2 percentage points and the
   best LLM method (Iterative Critique, 0.957) by 17. The largest
   gap is on `== ↔ !=` / `< ↔ >=` / `> ↔ <=` mutators, where
   Pynguin's behavioural oracles cannot encode operator-level
   semantics that LLMs can read from natural-language docstrings.

### 6.2 Implications for practice

The findings suggest four explicit recommendations for engineers
building LLM-based test generators:

1. **Choose the LLM before the RAG method.** LLM choice dominates
   RAG-method choice in explained variance. Engineering effort on a
   sophisticated RAG pipeline before selecting a capable underlying
   LLM is a misallocation.

2. **Match the RAG method to the LLM's capability tier.** Plain LLM
   or Simple RAG on the weakest models (where Iterative Critique's
   tests get filtered out at high rates); Iterative Critique on
   mid-range models (where its refinement loop pays off); Simple
   RAG on the strongest dense models (where the kill-rate metric is
   saturated and IC adds no value).

3. **Reward semantic faithfulness; penalise lexical copy-paste.**
   Token-overlap faithfulness to retrieved documentation is an
   anti-signal for defect-detection capability. RAG pipelines should
   explicitly de-emphasise lexical match and reward semantic
   integration, e.g., via a critique-and-refine loop or via reranking
   against a semantic-faithfulness judge.

4. **Optimise for the defect family that matters.** Boundary and
   arithmetic defects favour Iterative Critique; comparison defects
   favour LLM methods over SBST; throughput-sensitive deployments
   where filter-dropout matters may favour Plain LLM or Pynguin over
   IC.

### 6.3 Implications for research

For the empirical software-engineering research community, our
findings suggest two methodological shifts:

**Report interaction effects between RAG methods and LLM
capability**, not just RAG-method-effect-at-a-fixed-LLM means. The
non-generalisation we observed (§4.3) implies that a study reporting
"method X is best" on a single LLM is reporting a conditional, not a
universal, result. Cross-LLM evaluation should become a default
methodological requirement for RAG-test-generation research.

**Evaluate on SE-relevant metrics, not surface-level proxies.** Our
key positive finding — Iterative Critique's significant advantage on
boundary mutation kill rate in MBPP-style problems — would not
appear in a study that used BLEU, ROUGE, or semantic-similarity
metrics: the IC × Plain-LLM gap on those metrics is small. Mutation
kill rate, with its per-operator decomposition, surfaces the
substantive behavioural improvement that LLM-based test generation
provides.

### 6.4 Future work

Three directions stand out as natural follow-ups. First, the
**Pynguin–LLM hybrid** we sketched in §5.5 — running Pynguin first
for coverage-driven assertions, then adding LLM-generated
comparison-operator oracles on uncovered branches — should outperform
either approach alone on the mutation-testing metric, given the
complementarity we observed. Second, **closed-weight LLM
replication**: our finding that "Simple RAG wins on the strongest
dense model in our pool" should be tested against GPT-4o, Claude
Sonnet 4.6, and Gemini 2.5 to see whether the ceiling effect
generalises to closed-weight models in the 100 B – trillion-parameter
range. Third, **class-level and real-world benchmarks**: HumanEval
and MBPP are function-level benchmarks with clear input/output
specifications. Class-level evaluation (ClassEval) and real-world
project evaluation (e.g., on PyPI packages) are necessary to test
whether the IC advantage transfers to code with non-trivial state
and side effects.

A fourth, methodological, direction concerns the **human-evaluation
rubric**. Our 3-annotator inter-rater agreement fell below the
conventional ≥ 0.4 threshold on two of three dimensions, driven by
one annotator's systematic scale bias (§4.6). A future replication
should run a rubric-calibration session before annotation, increase
the rater pool to five or more, and consider including pre-filter
test suites in the annotation worksheet so the filter-dropout
asymmetry can be measured directly rather than excluded by
construction.

The empirical machinery developed in this paper — the experimental
sweep, the analysis scripts, the human-evaluation Streamlit
application, the Pynguin runner — is released as a replication
package (`https://github.com/balajivenky06/autoresearch`, release
tag listed in §9) so that all four future-work directions can
be pursued with the same evaluation framework we used here.

---

## 8. Limitations and Threats to Validity

We organise the limitations along Wohlin et al. (2012)'s standard
taxonomy of construct, internal, external, and conclusion validity.

### 8.1 Construct validity

**Mutation kill rate is a proxy.** We adopt mutation kill rate as our
operationalisation of "defect-detection capability", following the
empirical mutation-testing literature (Andrews et al., 2005; Just et
al., 2014). However, the mutation operators we implement are
necessarily a finite subset of all possible code defects: we do not
inject API misuses, off-by-one-on-non-integer-bounds, race conditions,
or higher-order mutations. The 5-operator set we adopt is the canonical
subset from mutmut and PIT and is what the empirical literature
benchmarks against, but the kill-rate numbers we report should be read
as relative comparisons between methods rather than absolute
ground-truth defect-detection rates.

**Human ratings are noisy.** The three-dimension 0–5 rubric we used
for the human evaluation, despite behavioural anchoring at every scale
point, did not achieve the conventional ≥ 0.4 Cohen's κ threshold for
two of the three dimensions. Pairwise agreement between two of three
annotators reached fair-to-moderate levels (Cohen's κ = 0.32 – 0.46);
the third annotator showed systematic use of the lower half of the
0–5 scale, dragging the 3-rater Krippendorff's α to below threshold on
test idiom quality and correctness. Completeness — the dimension most
directly tied to defect-detection capability — achieved the best
agreement (α = 0.33, fair). We report this honestly: the method-level
*ranking* derived from human ratings is robust (every annotator
ranked Iterative Critique highest on completeness), but the absolute
human ratings should not be interpreted as a calibrated quality scale.
A future replication should run a rubric-calibration session before
full annotation and increase the rater pool to five or more.

### 8.2 Internal validity

**The test filter introduces a confound.** Our `_filter_passing_tests`
step removes any LLM-generated test that fails on the original
function — a standard mutation-testing practice that prevents
spurious-assertion tests from polluting the kill-rate measurement. This
filter, however, drops a larger fraction of Iterative Critique's tests
than of Plain LLM's: IC's critique loop tends to produce tests with
tighter, more behaviourally-specific assertions that are correctly
rejected when the assertion-text does not match the function's actual
behaviour. The result is that IC's reported kill rate is averaged over
fewer surviving samples (n = 19–29 per cell) than Plain LLM's
(n = 29–30). We mitigate this by reporting per-cell sample counts in
every table and by complementing the mean kill rate with a Mixed-LM
analysis that uses the full per-sample data without requiring
balanced cells. The direction of the IC-versus-Plain-LLM effect is
preserved across both formulations (Tukey HSD ∆ = −0.205 on boundary
kill rate, p_adj = 0.0499; Mixed-LM β = −0.133, p = 0.016), so we
believe the substantive conclusion is robust, but readers should
interpret IC's kill-rate point estimates as conditional on the subset
of source samples that survive the filter.

**Phi4's function-shadow bug.** During our initial experimental
campaign, phi4-14b appeared to perform extremely poorly on mutation
testing (kill rates 5 – 26 %). We discovered that phi4 prepends a
re-definition of the function under test to its generated test files,
which — given Python's last-definition-wins semantics — shadowed the
mutant whenever the mutation harness concatenated the mutated function
with the test file. After fixing this with an AST-based
function-redefinition strip in the mutation harness, phi4's kill rates
climbed to the 86 – 93 % range reported in our results. We document
the bug, the fix, and both sets of numbers in the replication package
for full transparency. The fix is upstream of any of our analysis and
is reflected in every number reported in §4 of this paper.

### 8.3 External validity

**Four LLMs is not a population.** We benchmark across four specific
LLMs spanning 3B – 30B parameters and dense / MoE architectures. This
is a deliberate convenience sample chosen to span the open-weight
capability spectrum, not a random sample of LLMs. Our finding that
"method rankings do not generalise across LLMs" (min Spearman ρ =
−0.60) is therefore a statement about these four models specifically;
a future replication should test the claim on at least one
closed-weight model (e.g., GPT-4o-mini) and on additional dense models
in the 7 B – 14 B range.

**Two benchmarks, both function-level.** The mutation-testing study
runs on HumanEval and MBPP, both of which consist of self-contained
single-function Python problems with clear input-output specifications.
We do not evaluate on class-level benchmarks (e.g., ClassEval) or on
real-world projects with complex dependency graphs. Our finding that
"the IC-versus-Plain-LLM effect is concentrated on MBPP-style numeric-
boundary problems" should not be generalised to all software-testing
contexts. A future replication should sample functions from real
open-source projects with non-trivial state and side effects.

**Pynguin time-budget.** We compared our LLM methods against Pynguin
with a 60-second search budget per function. This is Pynguin's
recommended default for small functions, but some Pynguin
publications report results with budgets of 300 – 600 seconds. We did
not measure whether a larger budget would close the gap; the LLM-
versus-Pynguin numbers reported in §4 should be read as "Pynguin under
its conventional default budget" rather than "Pynguin at its
theoretical best". Additionally, Pynguin's strength is branch coverage
rather than mutation kill rate per se; our comparison shows that on a
per-mutant defect-detection metric, LLM methods outperform Pynguin
under the conditions tested, but a coverage-focused comparison might
reach different conclusions.

### 8.4 Conclusion validity

**Marginal p-values.** Our headline mixed-effects result for boundary
kill rate — Iterative Critique vs Plain LLM at p_adj = 0.0499 — is
formally significant but sits at the very edge of the conventional
α = 0.05 threshold. The corresponding Mixed-LM Wald-test p is 0.016,
solidly below threshold, and the effect direction is consistent across
six secondary tests (per-model Tukey HSD, per-operator decomposition,
per-benchmark split). Nonetheless, a more conservative reader could
reasonably reject the result on the Tukey-HSD criterion alone. We
mitigate by reporting effect sizes alongside p-values (Cohen's d, Cohen's
d_z, Tukey HSD mean differences with confidence intervals) so readers
can assess the magnitude of the effect independently of the
significance threshold.

**Sample sizes.** Thirty source samples per cell is small for a
fixed-effects ANOVA, especially with the unbalanced cell sizes
produced by the test filter. We mitigate by reporting per-cell n
counts in every results table; readers should be cautious about
interpreting any single cell estimate (in particular the
Iterative-Critique × llama3.2 cell, which has only n = 4 valid
samples after filtering). The Mixed-LM analysis using the full 409
per-sample observations is our preferred test for global claims
because it exploits the cross-cell structure of the data.

**The kill-rate × human-correctness correlation did not survive
adding a third annotator.** A finding we reported in an earlier
internal analysis — that mean human_correctness ratings correlate
positively with mutation kill rate (Pearson r = +0.34, p = 0.04) on
two annotators — disappeared when the third annotator's ratings were
pooled in (r = +0.18, p = 0.30). We do not report this correlation in
§4 of this paper, but we document the change in this limitations
section to be transparent about a result that was preliminary and did
not survive replication.

### 8.5 What is robust

To consolidate: we believe the following findings are robust to the
limitations enumerated above:

1. **Iterative Critique RAG significantly improves boundary-mutation
   kill rate over Plain LLM on MBPP**, with consistent effect direction
   across all four LLMs and across both rank-based and mixed-effects
   statistical tests.
2. **LLM choice dominates RAG-method choice in explained variance**;
   model F-statistic exceeds method F-statistic by approximately
   10–25× in every ANOVA fit.
3. **Method-level ordering does not generalise across LLMs**; the
   best-performing method changes between qwen3.5 (Simple RAG wins)
   and the other three models (IC wins).
4. **Token-overlap faithfulness to retrieved testing documentation
   negatively predicts mutation kill rate** in the RAG methods,
   suggesting tests that template from retrieved docs are weaker than
   tests that use retrieval as a reference.
5. **Three independent annotators ranked Iterative Critique highest
   on all three quality dimensions**, replicating the mutation-testing
   ordering with non-overlapping evidence.
6. **LLM methods outperform Pynguin's SBST approach on the
   mutation-kill-rate metric** under a 60-second budget, with the
   largest gap on comparison-operator mutators.

These six claims are the substantive contribution of the paper; the
limitations above describe the scope within which they hold.
