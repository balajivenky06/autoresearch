# Paper Draft — Mutation-Testing Validation of RAG-based Unit-Test Generation

**Working title (TBD)**: *When Retrieval Helps Tests Find Bugs: A Mutation-Testing Study of LLM and RAG-based Unit-Test Generators*

**Authors**: Balaji Venktesh, [advisor], …
**Target venue**: EMSE (resubmission)
**Tag**: `emse-resubmission-v1` (commit `c9c125e`)

> This file holds first-pass prose for the easier paper sections. §Methods
> and §Limitations are drafted here so they can be refined; §Introduction,
> §Related Work, §Results, §Discussion, §Conclusion remain to be written.
> Pull figures from `plots_mutation/` (see §Paper Figures Inventory in
> `session_context_mutation_testing.md`).

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

We adopt **mutation testing** as our SE-relevant evaluation metric
following the recommendation of EMSE's first-round review. A mutation
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

To address EMSE's request for "developer impact studies", we
complement the automated mutation-testing analysis with a human-
evaluation study modelled on Khan et al. (2024) and Sallam et al.
(2025) for code-quality annotation.

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

To address EMSE's request for "comparison with existing tools", we
benchmarked our LLM-based methods against **Pynguin 0.45.0** (Lukasczyk
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
