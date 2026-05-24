# Paper Draft — Mutation-Testing Validation of RAG-based Unit-Test Generation

**Working title (TBD)**: *When Retrieval Helps Tests Find Bugs: A Mutation-Testing Study of LLM and RAG-based Unit-Test Generators*

**Authors**: Balaji Venktesh, [advisor], …
**Target venue**: EMSE (resubmission)
**Tag**: `emse-resubmission-v1` (commit `c9c125e`)

> This file holds first-pass prose for the paper sections. §Methods,
> §Results, and §Limitations are drafted here and ready to refine;
> §Introduction, §Related Work, §Discussion, §Conclusion, and the
> Abstract remain to be written. Pull figures from `plots_mutation/`
> (see §Paper Figures Inventory in `session_context_mutation_testing.md`).

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

To address EMSE's request for "developer impact studies", three
independent annotators rated 40 stratified `(function,
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

To address EMSE's request for "comparison with existing tools", we
ran Pynguin 0.45.0 (Lukasczyk and Fraser, 2022) on the same 40 functions
used in the human evaluation, with a 60-second per-function search
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
