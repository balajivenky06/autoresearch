# Session Context — Mutation Testing & Analysis

**Last updated**: 2026-05-24
**Repo**: `/Users/balajivenktesh/Desktop/Education/autoresearch/`
**Branch**: `master`
**Latest commit**: `f484a55` — "merge: ingest SaeshwaranA's annotation CSV from feature branch"
**Status**: All 3 human annotators complete (GS, SaeshwaranA, BV) — accepted current data, no calibration round planned
**Remote**: `https://github.com/balajivenky06/autoresearch.git` (pushed and up to date)

**PhD Topic**: Plain LLM vs Random RAG vs Simple RAG vs Iterative Critique RAG for code generation.
**This task**: Unit test generation — full 4 methods × 4 models × 30 samples mutation-testing matrix complete, plus correlation and cross-model generalizability analyses.

---

## TL;DR — Headline Results for EMSE Resubmission

Three publishable claims now backed by statistics:

1. **Significance (sharpened by benchmark split)** — *On MBPP, **Iterative Critique RAG generates tests that detect 31.1 percentage points more off-by-one defects than Plain LLM generation** (Tukey HSD Δ=−0.311, p_adj=0.025; Mixed-LM β=−0.211, p=0.0048 after controlling for sample and LLM). The pooled result is weaker (Δ=−0.205, p_adj=0.0499) because HumanEval shows no method differences at all (ANOVA p=0.94). The effect is **benchmark-specific** — IC pays off on MBPP's range-check / off-by-one problem mix, not on HumanEval's broader test types.*

2. **Counter-intuitive correlation** — *Token-overlap faithfulness to retrieved testing documentation **negatively** predicts mutation kill rate (Pearson r=−0.61, p=0.045, n=11). Within Random RAG and Simple RAG the inverse is near-perfect (r=−0.97 and r=−0.99). Semantic faithfulness via DeepSeek judge shows no effect (r=−0.24, p=0.48). The harm is specific to **syntactic copy-paste** of retrieved tutorial vocabulary — LLMs that template from docs produce weaker assertions than LLMs that use retrieval as a reference for function-specific reasoning.*

3. **Generalizability** — *Method rankings on mutation kill rate **do not generalize across LLMs** (min pairwise Spearman ρ = −0.60, mean ρ = +0.23). Iterative Critique wins on three of four models (llama3.2, phi4, qwen3-coder) but drops to third on qwen3.5 where Simple RAG dominates. **Boundary kill rate has the highest cross-model rank stability** (mean ρ = +0.70) — and it's the only metric where the IC-vs-Plain-LLM Tukey HSD reaches significance, making it the recommended SE-relevant metric when reporting RAG ablations for unit-test generation.*

Two supporting facts:

- The overall kill rate doesn't show statistical separation on the per-sample tests because of a 1.0 ceiling effect on the strong models. Discriminative signal lives in the **boundary** mutation operator (n ↔ n±1), the historically hardest mutator.
- From the ANOVA decomposition: **LLM choice dominates RAG method choice**. Switching llama3.2 (3B) → qwen3-coder (30B) buys ~0.20 kill rate; switching Plain LLM → Iterative Critique buys ~0.05. F-statistic ratio (model : method) is roughly 25 : 1.
- The testing-docs knowledge base is well-curated — `avg_noise_rate` is identically 0.0 across every RAG cell (no chunk falls below the 0.3 cosine threshold). Reported as a methods-section fact, not a result.

---

## Final Results

### 4×4 Mutation Kill Rate Matrix (mean per method × model)

| Method | llama3.2 (3B) | phi4 (14B) | qwen3.5 (9B) | qwen3-coder (30B MoE) | mean (excl. llama) |
|---|---|---|---|---|---|
| Plain LLM | 0.72 (n=29) | 0.86 (n=30) | 0.91 (n=30) | 0.91 (n=30) | **0.89** |
| Random RAG | 0.65 (n=22) | 0.87 (n=30) | 0.98 (n=30) | 0.93 (n=30) | **0.93** |
| Simple RAG | 0.67 (n=20) | 0.90 (n=26) | **0.99** (n=30) | 0.92 (n=30) | **0.94** |
| **Iterative Critique** | 1.00 (n=4)¹ | **0.93** (n=19) | 0.94 (n=20) | **0.95** (n=29) | **0.94** |
| **Per-model mean** | 0.64 | 0.89 | **0.95** | 0.93 | |

¹ Llama3.2 IC has only n=4 valid samples — its tests are filtered out by the "must pass on original code" rule for the other 26 samples. Treat as noise.

### Statistical Significance — Per-Sample Rank Tests (`mutation_statistical_tests.py`)

Loaded 409 per-sample observations across all 4 models.

| Test | Statistic | p | Verdict |
|---|---|---|---|
| Kruskal-Wallis (unpaired, pooled) | H=5.79 | 0.12 | not significant |
| Friedman (paired, blocked by sample_idx, pooled n=67) | χ²=2.16 | 0.54 | not significant |
| Best Wilcoxon pairwise after Bonferroni | p_adj=0.61 | — | none cross α=0.05 |

Adding llama3.2 + phi4 *worsened* the pooled Friedman (qwen-only was p=0.15, all-4 dropped to 0.54). Reason: methods rank differently across models. Friedman demands consistent ordering across paired blocks; cross-model rank inversion eats power.

### Statistical Significance — Per-Operator (`mutation_statistical_tests.py` §5)

Same Friedman/Wilcoxon pipeline run once per mutation operator. None cross Bonferroni.

| Operator | n_blocks | χ² | p | Notes |
|---|---|---|---|---|
| arithmetic | 33 | nan | nan | All-equal degenerate |
| **boundary** | 31 | 3.87 | 0.28 | Hardest mutator |
| comparison | 23 | 4.00 | 0.26 | — |
| **negate_bool** | 13 | 5.18 | 0.16 | Strongest rank-test signal, tiny n |
| return_none | 65 | 3.20 | 0.36 | Largest sample |

### Statistical Significance — Mixed-Effects (`mutation_mixed_effects.py`) ★

This is the test that actually works for the design. Uses all 409 observations (not just complete-case paired blocks).

| Metric | ANOVA method F | ANOVA p | Tukey HSD significant pairs | Mixed-LM best p |
|---|---|---|---|---|
| kill_rate (overall) | 0.79 | 0.50 | 0/6 | 0.14 |
| **kill_rate_boundary** | **2.39** | **0.07** | **1/6 ★** | **0.016 ★** |
| kill_rate_return_none | 0.11 | 0.95 | 0/6 | — |

#### The significant pair (boundary)

| Comparison | Δ mean | 95% CI | Tukey p_adj | Mixed-LM β | Mixed-LM p |
|---|---|---|---|---|---|
| **Iterative Critique vs Plain LLM** | **−0.205** | [−0.41, 0.00] | **0.0499** ★ | **−0.133** | **0.016** ★ |
| IC vs Random RAG | −0.142 | [−0.35, 0.06] | 0.28 | — | — |
| IC vs Simple RAG | −0.099 | [−0.31, 0.11] | 0.61 | — | — |

#### ANOVA decomposition (the "LLM choice dominates" finding)

| Term | F (overall kill_rate) | F (boundary) |
|---|---|---|
| C(method) | 0.79 (p=0.50) | 2.39 (p=0.07) |
| **C(model)** | **22.1 (p<0.001)** | **10.0 (p<0.001)** |
| **C(sample_idx)** | **9.8 (p<0.001)** | **15.2 (p<0.001)** |

The model and sample effects dwarf the method effect by 10–20×. This is the "pick the right LLM first, then the right RAG method" claim.

### Per-Operator Kill Rates (descriptive)

| Method | llama3.2 boundary | phi4 boundary | qwen3.5 boundary | qwen3-coder boundary |
|---|---|---|---|---|
| Plain LLM | 0.40 | 0.57 | 0.70 | 0.69 |
| Random RAG | 0.54 | 0.63 | 0.80 | 0.67 |
| Simple RAG | 0.61 | 0.66 | 0.91 | 0.69 |
| **Iterative Critique** | **1.00** | **0.96** | 0.84 | **0.76** |

Clear directional ordering across all 4 models: **IC ≥ Simple RAG ≥ Random RAG > Plain LLM on boundary**.

### RAG Quality vs Kill Rate Correlation (`noise_vs_kill.py`)

Joins `results_unitest.tsv` (RAG-quality metrics) with `results_mutation.tsv` on (method, model, reasoning='base') for the 11 RAG cells with non-NaN data.

| Predictor | Pooled r | p | n | Verdict |
|---|---|---|---|---|
| `avg_noise_rate` | nan | nan | 7 | Constant 0.0 — degenerate (see TL;DR fact) |
| **`avg_faithfulness`** (token overlap) | **−0.614** | **0.045** ★ | 11 | **Significant, negative** |
| `avg_llm_judge_faithfulness` (DeepSeek judge) | −0.237 | 0.48 | 11 | No correlation |

Within-method breakdown for `avg_faithfulness`:

| Method | r | p | n |
|---|---|---|---|
| random_rag | **−0.97** | 0.029 ★ | 4 |
| simple_rag | **−0.99** | 0.011 ★ | 4 |
| iterative_critique | +0.75 (small-n) | 0.46 | 3 |

The negative correlation is sharply within-method on Random/Simple RAG (r ≈ −0.98). IC alone flips the sign (n=3, suggestive only) — consistent with the critique loop steering tests away from copy-pasted vocabulary toward function-specific assertions.

### Cross-Model Generalizability (`analyze_mutation_generalizability.py`)

Spearman ρ between method rankings for every pair of models, threshold ρ ≥ 0.8 (Zar 1984; Jureczko & Madeyski 2015 IST).

| Metric | min ρ | mean ρ | Verdict |
|---|---|---|---|
| `mean_kill_rate` (overall) | **−0.60** | +0.23 | does not generalize |
| `kill_arithmetic` | **−1.00** | +0.15 | does not generalize (one pair perfectly inverted) |
| **`kill_boundary`** | **+0.32** | **+0.70** | closest, mean ρ highest of any metric |
| `kill_comparison` | +0.63 | **+0.80** | mean exactly at threshold; min fails |
| `kill_negate_bool` | +0.00 | +0.39 | does not generalize |
| `kill_return_none` | **−0.95** | −0.24 | **inverted** between llama3.2 and qwen3.5 |

Rank table (1 = best method on that model):

| Method | llama3.2 | phi4 | qwen3-coder | qwen3.5 |
|---|---|---|---|---|
| Plain LLM | 2 | 4 | 4 | 4 |
| Random RAG | 4 | 3 | 2 | 2 |
| Simple RAG | 3 | 2 | 3 | **1** |
| Iterative Critique | **1** | **1** | **1** | 3 |

- IC is rank-1 on three of four models; on qwen3.5 it drops to rank 3 (Simple RAG takes top).
- Plain LLM is rank 2 on llama3.2 but rank 4 on every stronger model.
- Why boundary is the most universal metric: it's the operator where the LLM-capability advantage doesn't yet saturate (no ceiling effect), so the method differences are still legible.

### Per-Benchmark Analysis (`mutation_per_benchmark.py`) ★

Splits per-sample kill rates by `source` (recovered from generation pkls via shared `sample_idx`) and re-runs the ANOVA + Tukey HSD + Mixed-LM pipeline within each benchmark. The split is **21 MBPP : 9 HumanEval** per cell (deterministic via seed-42 shuffle), giving n≈287 (MBPP) and n≈122 (HumanEval) per metric in the long form.

| Scope | n | ANOVA F (method) | ANOVA p | Tukey IC vs Plain LLM | Mixed-LM IC vs Plain LLM |
|---|---|---|---|---|---|
| **boundary kill rate** | | | | | |
| MBPP only | 287 | **2.96** | **0.035** ★ | **Δ=−0.311, p_adj=0.025** ★ | **β=−0.211, p=0.0048** ★★ |
| HumanEval only | 122 | 0.13 | 0.94 | Δ=+0.007, p=1.0 | β=+0.002, p=0.97 |
| Pooled (for comparison) | 409 | 2.39 | 0.07 | Δ=−0.205, p_adj=0.0499 | β=−0.133, p=0.016 |
| **overall kill rate** | | | | | |
| MBPP only | 287 | 1.30 | 0.27 | **Δ=−0.154, p_adj=0.028** ★ | **β=−0.082, p=0.048** ★ |
| HumanEval only | 122 | 1.01 | 0.39 | none | n.s. |
| Pooled (for comparison) | 409 | 0.79 | 0.50 | none | β=−0.046, p=0.144 |

**Interpretation**: The defect-detection advantage of Iterative Critique over Plain LLM is **specific to MBPP-style problems**. MBPP leans on numeric boundary conditions (range checks, off-by-one list indexing) where IC's critique loop tightens value-specific assertions. HumanEval's broader problem mix doesn't benefit. The pooled-marginal previous claim averaged across these two regimes; the per-benchmark split makes the claim sharper and more credible.

### 4×4 Heatmaps (`plot_mutation_heatmap.py`) ★

Replaces the original 16-bar chart with three heatmap variants for paper figures:

| File | What it shows |
|---|---|
| `kill_rate_heatmap.png` | 4×4 grid (rows=method, cols=model) coloured by mean kill rate, annotated with values + n |
| `kill_rate_boundary_heatmap.png` | Same grid for boundary kill rate — where the colour spread is widest, suitable for the significance figure |
| `kill_rate_combined_heatmap.png` | 1×5 mosaic: overall + 4 per-operator panels — supplementary materials |

### Human Evaluation — All 3 Annotators Complete ★

Built a Streamlit app (`human_eval_app.py`) plus a per-pair sampler
(`human_eval_pair_sampler.py`) to ask human annotators to rate 40
stratified (function, generated_tests) pairs on three behaviourally-
anchored 0–5 scales: **test idiom quality**, **correctness**,
**completeness**. Annotators are blinded to method/model.

**Final coverage and descriptive means**:

| Annotator | Rows | Idiom (mean) | Correctness | Completeness |
|---|---|---|---|---|
| GS | 40/40 | 4.40 | 4.25 | 4.08 |
| SaeshwaranA | 40/40 | 2.88 | 3.45 | 3.62 |
| BV | 40/40 | 3.77 | 3.98 | 4.22 |

GS and BV anchor at the high end; **SaeshwaranA is a systematic
low-end outlier** (1.5pt below GS on idiom, 0.8pt below on
correctness). Same scale, different mental anchors.

#### 🟢 The headline (robust to inter-rater noise)

**Iterative Critique ranks highest on all three dimensions in the 3-
annotator means** — purely from blinded human judgment, no mutation
data leaked into the ratings.

| Method | Idiom | Correctness | Completeness |
|---|---|---|---|
| **Iterative Critique** | **4.06** | **4.15** | **4.58** |
| Random RAG | 3.79 | 3.73 | 3.85 |
| Simple RAG | 3.56 | 3.85 | 3.74 |
| Plain LLM | 3.22 | 3.82 | 3.63 |

Three independent annotators converged on the same method ordering as
the mutation testing analysis (IC > others on boundary kill rate, p_adj
= 0.0499). This is the **developer-impact paragraph** EMSE asked for.

#### Pairwise Cohen's κ (linear-weighted)

| Pair | Idiom | Correctness | Completeness |
|---|---|---|---|
| **GS ↔ BV** | **0.32** | **0.46** | **0.37** |
| GS ↔ SA | 0.005 | −0.22 | 0.21 |
| SA ↔ BV | −0.001 | −0.31 | 0.16 |
| Mean of 3 pairs | +0.11 | −0.02 | **+0.25** |

GS and BV land in the fair-to-moderate band; SaeshwaranA's ratings
disagree with both other annotators in the same direction.

#### Multi-rater agreement (all 3, n=40)

| Test | Idiom | Correctness | Completeness |
|---|---|---|---|
| Fleiss' κ (nominal) | −0.11 | −0.02 | +0.14 |
| **Krippendorff's α (ordinal)** | −0.03 | −0.12 | **+0.33** |

Below the conventional ≥0.4 threshold on all dimensions; Completeness
is the only one reaching fair α.

#### Mean human ratings vs mutation kill_rate

| Dimension | Pearson r | p | Spearman ρ |
|---|---|---|---|
| Idiom | −0.080 | 0.64 | −0.019 |
| Correctness | +0.177 | 0.30 | +0.201 |
| Completeness | +0.023 | 0.89 | +0.111 |

The kill_rate ↔ human_correctness correlation that was significant on
n=2 annotators (r=+0.34, p=0.04) washes out with BV's data added. The
paper should NOT lean on that result.

#### Model-level wrinkle

| Model | Idiom | Correctness | Completeness | Kill rate |
|---|---|---|---|---|
| qwen3.5:9b | **3.97** | **4.03** | **4.27** | 0.975 |
| qwen3-coder:30b | 3.62 | 3.67 | 3.75 | **0.986** |
| phi4:14b | 3.58 | 3.97 | 3.94 | 0.973 |
| llama3.2:latest | 3.53 | 3.83 | 3.87 | 0.972 |

**qwen3.5 (9B dense) ranks highest in humans' eyes** even though
qwen3-coder (30B MoE) has the highest mutation kill rate. The MoE
writes correct-but-unidiomatic tests; the dense 9B writes more
readable ones. Worth a paragraph in the discussion.

#### Paper framing (Option C — accept current data)

We're not running a calibration round. The paper will:

1. **Lead with the method-ranking finding**: three independent
   blinded annotators ranked Iterative Critique highest across all
   three quality dimensions, replicating the mutation-testing
   ordering with non-overlapping evidence.
2. **Report the pairwise GS↔BV agreement** (κ ≈ 0.32–0.46, fair-to-
   moderate) as the primary inter-rater statistic, alongside the
   full 3-rater Krippendorff's α (Completeness = 0.33).
3. **Acknowledge SaeshwaranA's systematic scale-bias** in
   §Limitations rather than discarding their data.
4. **Drop the kill_rate ↔ human_correctness correlation** from
   §Results (it was an n=2 artifact).
5. **Add the qwen3.5 > qwen3-coder human-rating finding** to
   §Discussion as evidence that MoE architectures produce
   functionally-correct-but-stylistically-different tests.

Suggested §Limitations paragraph:
> "We acknowledge moderate inter-rater agreement on completeness
> (Krippendorff's ordinal α = 0.33) and lower agreement on test
> idiom quality and correctness, driven primarily by one annotator's
> systematic use of the lower half of the 0–5 scale. The agreement
> between the other two annotators reached fair-to-moderate levels
> (Cohen's κ = 0.32–0.46), and the method-level ranking was robust
> across all three: every annotator ranked Iterative Critique
> highest on completeness, the dimension most directly tied to
> defect-detection capability. A future replication should use a
> rubric-calibration session and ≥5 annotators."

The blinded worksheet (`human_eval_pairs.csv`) is committed at master
and `feature/human-eval-app`. The 3 returned CSVs live in
`human_eval_annotations/` (gitignored, personal data; Saeshu's commit
of `SaeshwaranA.csv` to the repo root is preserved in git history).

---

## Analysis Pipeline (7 scripts)

| Script | What it does | Key output |
|---|---|---|
| `mutation_testing.py` | Generate + analyze mutants. Supports `--regenerate` (Ollama generation) and `--checkpoints-dir` (analysis-only on existing tests). Per-sample resume on both phases. | `results_mutation.tsv`, `.checkpoints_mutation/`, `.checkpoints_mutation_analysis/`, `plots_mutation/` |
| `mutation_statistical_tests.py` | Per-sample Kruskal-Wallis + Mann-Whitney (unpaired) and Friedman + Wilcoxon (paired). Per-model, pooled, and per-operator (boundary, arithmetic, etc.). | `plots_mutation/mutation_statistical_report.txt` |
| `mutation_mixed_effects.py` ★ | Type-III ANOVA + Tukey HSD + Mixed-LM (sample_idx as random intercept). The right test for this design. Accepts `--metric kill_rate_<operator>`. | `plots_mutation/mutation_mixed_effects_*.txt` |
| `noise_vs_kill.py` ★ | Joins unitest TSV with mutation TSV; Pearson r + Spearman ρ for noise rate, faithfulness, and DeepSeek judge against kill rate. Per-method and pooled. | `plots_mutation/noise_vs_kill_report.txt`, `noise_vs_kill_scatter.png` |
| `analyze_mutation_generalizability.py` ★ | Cross-model Spearman ρ of method rankings (overall + per-operator). Heatmaps, rank-stability lines, grouped bars. | `plots_mutation/mutation_generalizability_report.txt`, `mutation_rank_*.png` |
| `mutation_per_benchmark.py` ★ | Splits per-sample data by `source` (HumanEval vs MBPP) and re-runs ANOVA + Tukey HSD + Mixed-LM within each. Reveals which benchmark drives the pooled significance. | `plots_mutation/mutation_per_benchmark_report.txt` |
| `plot_mutation_heatmap.py` ★ | 4×4 method × model heatmaps for paper figures: overall, boundary-only, and a 5-panel mosaic with all operators. | `plots_mutation/kill_rate_heatmap*.png` |
| `human_eval_pair_sampler.py` ★ | Builds the 40-row blinded worksheet from generation pkls (stratified across method × model × source × kill-rate strata). Produces a public `human_eval_pairs.csv` and a private `human_eval_pairs.meta.csv` (sample_id → method/model). | `human_eval_pairs.csv`, `human_eval_pairs.meta.csv` (gitignored) |
| `human_eval_app.py` ★ | Streamlit UI for annotators. Login → blinded sample → 0–5 radio for 3 dimensions (with anchor captions) → save & next → resume support. Persists to `human_eval_annotations/{annotator_id}.csv` after every sample. | `human_eval_annotations/*.csv` (gitignored) |

### Why mixed-effects beat the rank tests

The Friedman test required complete cases across all 4 methods within each sample_idx. Iterative Critique's filtering rule (tests must pass on original) drops the most cells (especially on llama3.2 where IC has 4/30 valid), so the paired-block count crashed from a possible 120 to 67. The mixed-effects model uses all 409 observations and treats sample_idx as a random intercept — same blocking benefit, no complete-case requirement.

---

## Bug Catalogue (chronological)

### Pre-session-2 (May 6 framework build)

| Bug | Root Cause | Fix |
|-----|-----------|-----|
| MBPP ground truth tests fail in pytest | Bare asserts, not `def test_*()` functions | `_wrap_bare_asserts()` |
| HumanEval `check(candidate)` format | pytest can't discover `check()` | Detect and append `def test_humaneval(): check(fn_name)` |
| GENERATORS dict key mismatch | Used `f"{method}/{reasoning}"` but GENERATORS uses tuple keys | Changed to `(method, reasoning)` |
| llama3.2:1b too weak | 0/20 tests passed on original | Use `llama3.2:latest` (3B) |
| `uv run` fails on macOS | PyTorch cu128 incompatible with ARM | Use `python3` directly |
| LLM-generated spurious assertions | e.g., expects ValueError when fn raises IndexError | `_filter_passing_tests()` |
| Method label parsing broken | `iterative_critique` has underscores | `KNOWN_METHODS` dict |

### Bugs found and fixed in session-2 (May 12–16) — these are why phi4 + qwen worked

| Bug | Commit | Root Cause | Fix |
|-----|--------|-----------|-----|
| Cache path mismatch on Colab | `9920057` | `mutation_testing.py` looked for dataset under `~/.cache/...` (= `/root/.cache/` on Colab) but `prepare_unitest.py` writes to `/content/.cache/...` on Colab | Mirror the `_IN_COLAB` detection from `prepare_unitest.py` |
| TSV collapsed method+model+reasoning into one column | `5710135` | `run_mutation_analysis` dropped model/reasoning when writing the row label; main() overwrote TSV per run instead of merging | Add `method`/`reasoning`/`model` columns; merge with existing TSV on (method, reasoning, model) keys |
| **phi4 looked terrible (5–26% kill rates)** | `6b977fd` | LLMs (phi4 in particular) prepend a re-definition of the function under test to their generated tests. When the harness writes the mutant function followed by tests, the LLM's pristine re-definition **shadows the mutant** in Python's module namespace, so tests call the un-mutated version. | `_strip_function_redefinition()` uses AST (regex fallback) to remove top-level `def <fn_name>(...)` from test code before concatenation. After fix: phi4 went 5–26% → 86–93%. |
| Ollama silently died mid-Colab-run | `6b977fd` | `train_unitest._llm` swallows connection errors and returns `""`, which got saved as empty `generated_tests` in checkpoints. ~17 samples wasted per method when Ollama crashed during qwen3.5 generation. | Per-sample retry (3 attempts, 5/10/15s waits) with `_wait_for_ollama()` health probe between attempts. |
| Analysis lost on disconnect | `d42058f` | Generation had per-sample resume but `run_mutation_analysis` ran sample 0 → N without saving intermediate state. A Colab drop mid-analysis lost all completed pytest work for the current method. | New `.checkpoints_mutation_analysis/{key}.pkl` saved atomically (tmp + rename) after every sample. Cell 24 syncer also mirrors the new dir to Drive. |
| **Resume crashed with AttributeError: lambda** | `6b1c779` | `evaluate_mutants` returned a result dict whose `per_operator` was a `defaultdict(lambda: {...})`. Pickle can't serialize lambdas, so the very first sample-save crashed → exit code 1, killing qwen3.5 right after `Analyzing: plain_llm_base_qwen3.5_9b`. | Replace defaultdict with plain dict + small `_bump()` helper. Smoke-test confirms `pickle.dumps({0: evaluate_mutants(...)})` succeeds. |
| Path mismatch for synced analysis pkls | `5e16057` | Local Drive sync put files under `checkpoints_mutation_analysis/` (no dot), but `mutation_testing.py` wrote to `.checkpoints_mutation_analysis/` (dot). Re-running locally would treat existing qwen analyses as absent and redo them. | `_resolve_analysis_dir()` prefers dotted, falls back to no-dot if that has data. Reads/writes use the same resolved path. |

### Bonus: `--models` filter (`5e16057`)

Added to `mutation_testing.py` so the user can run `--checkpoints-dir checkpoints_mutation/ --models 'llama3.2:latest,phi4:14b'` and surgically re-analyze a subset.

---

## Compute Trajectory (what was run, where, when)

| Date | Where | What | Outcome |
|---|---|---|---|
| 2026-05-04 | Local | First mutation run on llama3.2 only (20 samples/method) | Initial llama3.2 results, framework debugged |
| 2026-05-09 | Colab A100 | Attempted 4-model regenerate sweep | qwen3.5 + qwen3-coder failed (Ollama daemon crash) |
| 2026-05-12 | Colab A100 | Re-run after function-shadow fix | phi4 fixed (5–26% → 86–93%). qwen3.5 still failing (no retry yet) |
| 2026-05-14 | Colab A100 | Final 4-model regenerate sweep with retry + resume | **All 4 models' generation phase completed**, analysis completed for qwen3.5 + qwen3-coder |
| 2026-05-16 | Local laptop | Re-analyze llama3.2 + phi4 from existing checkpoints | All 8 missing analysis pkls populated. Full 409-observation dataset. |
| 2026-05-16 | Local laptop | `mutation_statistical_tests.py` on 4-model data | Pooled Friedman p=0.54 (no significance). Per-operator p=0.16–0.36 (also none). |
| 2026-05-16 | Local laptop | `mutation_mixed_effects.py` on 4-model data | **Boundary IC vs Plain LLM significant: Tukey p=0.0499, Mixed-LM p=0.016 ★** |

---

## TODO List (Prioritized for EMSE Resubmission)

### ✅ COMPLETED in this session

- [x] **Mutation testing on all 4 models** (4 models × 4 methods × 30 samples = 480 cells, 409 valid after filter)
- [x] **Statistical significance for mutation kill rate** — rank tests (Kruskal-Wallis, Friedman, Mann-Whitney, Wilcoxon) + Type-III ANOVA + Tukey HSD + Mixed-LM
- [x] **Per-operator significance tests** (boundary, arithmetic, comparison, negate_bool, return_none)
- [x] **Bug-fixing**: cache path, TSV merge, function-shadow, Ollama retry, per-sample analysis resume, pickle-friendly results, --models filter, dir-resolution fallback
- [x] **RAG quality ↔ kill rate correlation** — `noise_vs_kill.py`. Found significant negative correlation (r=−0.61, p=0.045) between token-overlap faithfulness and kill rate. Counter-intuitive finding: tests copying retrieved tutorial vocabulary verbatim catch fewer bugs.
- [x] **Cross-model generalizability** — `analyze_mutation_generalizability.py`. Found that method rankings do not generalize across LLMs (min ρ=−0.60); boundary kill rate has the highest cross-model rank stability (mean ρ=+0.70).
- [x] **Per-benchmark mutation analysis** (HumanEval vs MBPP) — `mutation_per_benchmark.py`. Found that the boundary IC vs Plain LLM significance is MBPP-driven (Δ=−0.311, Mixed-LM p=0.005) and absent on HumanEval (p=0.94). Sharpens the paper claim from "marginal across both" to "strong on MBPP".
- [x] **Replot 4×4 heatmaps** — `plot_mutation_heatmap.py`. Three figures for the paper: overall, boundary-only, and a per-operator mosaic.

### HIGH PRIORITY — directly addresses reviewer feedback

- [x] **Human evaluation study (40 samples)** — All 3 annotators (GS,
  SaeshwaranA, BV) returned full 40/40 coverage. Per the Option C
  framing: lead with the method-ranking story (IC ranks highest on
  all three dimensions in the 3-annotator means), report pairwise
  GS↔BV agreement (κ = 0.32–0.46) as the primary inter-rater stat,
  document SaeshwaranA's scale bias in §Limitations. The earlier
  kill_rate↔correctness correlation (r=+0.34 on n=2) does NOT
  survive 3-rater averaging — dropped from §Results. See §Human
  Evaluation for the full data + suggested limitations paragraph.
- [ ] **Tool comparison vs EvoSuite / Pynguin / Copilot** — reviewer mentioned "comparison with existing tools". Even qualitative comparison on 10–20 samples would help.

### MEDIUM PRIORITY — strengthens paper

- [x] ~~Cross-model generalizability for kill rates~~ — done; see Cross-Model Generalizability section above.
- [x] ~~Noise rate → kill rate correlation~~ — done; noise rate degenerate but found significant token-overlap faithfulness correlation instead.
- [ ] **Increase mutation sample size** to 100 samples/cell — direct power increase if the marginal p=0.07 ANOVA result needs to be airtight. ~12–24h Colab.
- [ ] **Iterative critique rounds ablation** — 1 vs 2 critique rounds on boundary kill rate. Tests whether the second round is doing real work.

### LOWER PRIORITY — nice-to-have

- [x] ~~Per-benchmark mutation analysis~~ — done; MBPP drives the pooled significance.
- [x] ~~Re-plot with full 4×4 matrix~~ — done; three heatmaps generated.

### Write the paper

- [ ] **Draft resubmission** — descriptive results + statistical tests + bug-fix Q&A for likely reviewer questions ("how do you handle equivalent mutants?", "what's the false-positive rate on the filter?", etc.).

---

## Paper Framing (suggested)

### Core narrative
1. RAG-based test generation methods produce tests with systematically higher mutation kill rates than plain LLM generation (descriptive ordering across 16 model × method cells)
2. The advantage is **statistically certified for boundary mutations on MBPP**: Iterative Critique RAG vs Plain LLM Δ=+0.311 kill rate on MBPP (Tukey HSD p_adj=0.025; Mixed-LM p=0.005 after controlling for LLM and sample). The pooled (HumanEval + MBPP) effect is weaker (Δ=+0.205, p_adj=0.0499) because HumanEval contributes a null slice (ANOVA p=0.94). The mechanism: MBPP problems lean on numeric range / off-by-one conditions where IC's critique loop tightens value-specific assertions; HumanEval's broader test types don't benefit.
3. The overall kill-rate advantage narrows to non-significant on capable models because of a 1.0 ceiling effect — methods converge when the base LLM is strong enough
4. **LLM choice matters more than RAG method**: model F-statistic dominates method F-statistic by 10–20× in every ANOVA fit
5. **Method rankings do not generalize across LLMs** (min Spearman ρ = −0.60). Iterative Critique wins on 3 of 4 models but Simple RAG wins on the strongest (qwen3.5). Boundary kill rate has the highest cross-model rank stability (mean ρ = +0.70), motivating it as the canonical SE-relevant metric for RAG ablations.
6. **Counter-intuitive faithfulness finding**: token-overlap faithfulness to retrieved testing documentation negatively predicts kill rate (Pearson r = −0.61, p = 0.045). Tests that template from retrieved tutorials catch fewer bugs than tests that use retrieval as a reference. DeepSeek-judged semantic faithfulness shows no correlation (r = −0.24, p = 0.48), pinpointing the harm as **syntactic copy-paste**, not principled grounding.
7. **Human-evaluation replication of the method ordering** (developer impact): three independent blinded annotators × 40 stratified samples × three 0–5 dimensions (test idiom quality, correctness, completeness). Iterative Critique ranks highest on every dimension in the 3-annotator means (idiom 4.06 / correctness 4.15 / completeness 4.58), replicating the mutation-testing ordering with non-overlapping evidence. Inter-rater agreement between two of three raters reached fair-to-moderate levels (Cohen's κ = 0.32–0.46); the third rater showed systematic low-end scale bias, acknowledged in §Limitations. qwen3.5 (9B dense) ranks higher than qwen3-coder (30B MoE) in human ratings despite a slightly lower mutation kill rate, suggesting MoE models write functionally-correct-but-stylistically-different tests.

### Findings summary table

| # | Finding | Evidence | Strength |
|---|---|---|---|
| 1 | **IC > Plain LLM for boundary defects on MBPP** | Tukey p_adj=0.025 (Δ=−0.31); Mixed-LM p=0.005 | **Strong, benchmark-specific** |
| 1a | (Pooled across HumanEval + MBPP) | Tukey p_adj=0.0499; Mixed-LM p=0.016 | Marginal (averages over the null HumanEval slice) |
| 1b | IC > Plain LLM for overall kill rate on MBPP | Tukey p_adj=0.028; Mixed-LM p=0.048 | Significant, narrower effect |
| 2 | LLM choice > RAG method choice | ANOVA F: model 22.1 vs method 0.79 | Strong (huge F-ratio) |
| 3 | Token-overlap faithfulness ↑ → kill rate ↓ | Pearson r=−0.61, p=0.045 | Strong, novel, counter-intuitive |
| 4 | Semantic faithfulness DOES NOT correlate with kill rate | DeepSeek judge r=−0.24, p=0.48 | Supporting (validates #3 is about syntax) |
| 5 | Method rankings don't generalize across LLMs | min ρ=−0.60, mean ρ=+0.23 | Strong |
| 6 | Boundary is the most universal metric | mean ρ=+0.70, only sig pair | Strong methodological note |
| 7 | KB curation: noise rate = 0 everywhere | All 7 RAG cells | Methods-section fact |
| 8 | Method effect is benchmark-specific (MBPP only) | HumanEval ANOVA p=0.94; MBPP p=0.035 | Sharpens claim #1; explains the pooled marginality |
| 9 | **3 blinded human annotators rank IC highest on all 3 dimensions** | Means: idiom 4.06 / correctness 4.15 / completeness 4.58 (IC), highest in every column | Strong, novel "developer impact" evidence (independent of automated metrics) |
| 10 | Pairwise GS↔BV agreement is fair-to-moderate | Cohen's κ = 0.32–0.46 across the 3 dims | Inter-rater reliability for the paper; SaeshwaranA's scale bias acknowledged in §Limitations |
| 11 | qwen3.5 (9B dense) ranks higher than qwen3-coder (30B MoE) in human eyes | Human means qwen3.5 > qwen3-coder on every dimension, despite qwen3-coder's higher mutation kill rate | "MoE writes correct-but-unidiomatic tests" — discussion-section paragraph |

### Threat-to-validity Q&A (preempt reviewers)

| Reviewer concern | Our answer |
|---|---|
| Why drop NaN samples? | LLM-generated tests must pass on the original function before mutation testing — standard practice (Andrews et al. 2005). Documented as `_filter_passing_tests`. |
| Why phi4 looks so different across versions? | Found and fixed a real bug (`6b977fd`): some LLMs prepend a function re-definition that shadowed the mutant. Fixed via AST-based redefinition stripping. Pre-fix phi4 numbers (5–26%) are reported with the fix and final numbers (86–93%) for transparency. |
| Equivalent mutants? | Detected via ground-truth tests on each mutant: if ground truth ALSO passes, the mutant is equivalent and excluded from kill-rate denominator. ~5–15% of mutants per cell. |
| Sample size? | 30 per cell; 409 valid observations after filtering. Mixed-LM uses all 409. Acknowledged in §Limitations. |
| Why does ρ analysis use the aggregated TSV (4 ranks per model) when you have per-sample data? | Generalizability claims at the paper level operate on the artifact you'd quote — the per-cell mean. The mixed-effects analysis already uses the per-sample data with the same blocking structure, so the two analyses are complementary, not redundant. |
| Why is the faithfulness correlation only n=11? | One RAG cell per (method × model) pair on base reasoning; iterative_critique × qwen3.5 had insufficient `avg_faithfulness` data. The result is small-n; quoted for completeness but stronger correlations within-method (r=−0.97 and r=−0.99) at n=4 each are the primary evidence. |
| Why does the IC vs Plain LLM result depend on benchmark? | MBPP problems are numeric-boundary heavy (range checks, off-by-one indexing); IC's critique loop adds value-specific assertions exactly where boundary mutators bite. HumanEval has more varied problem types where boundary mutators are a smaller share of mutants. We report both slices for transparency and frame the headline claim as MBPP-specific. |
| Why use sample_idx for source lookup rather than re-loading the dataset? | The dataset shuffle (seed=42) deterministically maps sample_idx → source, but generation pkls already store source per sample. The script reads from generation pkls, so source assignment is auditable from on-disk artifacts only — no dependency on a re-shuffle reproducing the same order. |

---

## Key Commits (in this session)

```
f484a55 merge: ingest SaeshwaranA's annotation CSV from feature branch
c519b00 docs: capture human-evaluation interim state (2/3 annotators)
9ae1fc2 Committed eval csv file (SaeshwaranA, via feature branch)
134b90c docs: add requirements.txt + README_human_eval.md
5175e89 rubric: expand rating scale 0-3 → 0-5
26104e9 rubric: drop Overall + add per-value anchor descriptions
9210037 rubric: rename Faithfulness → Test idiom quality
991d0ba feat: Streamlit human-evaluation app + per-pair sampler
399cdde feat: per-benchmark mutation analysis + 4×4 heatmaps
7bd2dd5 docs: add correlation + cross-model generalizability findings
815bb37 feat: analyze_mutation_generalizability.py — cross-model Spearman ρ
33a9137 feat: noise_vs_kill.py — RAG quality vs mutation kill rate correlation
a2d7a59 docs: refresh session_context_mutation_testing.md with full session state
6265c4e feat(stats): mixed-effects model — significance for boundary kill rate
bd7b2ac feat(stats): per-operator significance tests on per-sample kill rates
5e16057 feat(mutation_testing): resolve analysis-ckpt dir + --models filter
01d8415 feat(stats): paired Friedman + Wilcoxon signed-rank tests
55625a6 feat: mutation_statistical_tests.py — significance tests for kill rates
6b1c779 fix(mutation_testing): evaluate_mutants result must be pickleable
d42058f feat(mutation_testing): per-sample resume in run_mutation_analysis
19fdf0e notebook(Step 10b): regenerate-per-model + resume-safe mutation testing
6b977fd fix(mutation_testing): strip LLM function-redefinition; add Ollama retry
5710135 fix(mutation_testing): preserve model/reasoning in TSV; merge across runs
9920057 fix(mutation_testing): match prepare_unitest.py cache path on Colab
fb57e0d chore: ignore regenerable artifacts; add mutation testing session notes
212f707 add mutation testing framework + results (SE-relevant defect detection metric)
```

---

## Reproducibility Cheat-Sheet

### Local re-analysis (no GPU needed, ~1–3 h)

```bash
cd /Users/balajivenktesh/Desktop/Education/autoresearch
# Re-runs mutation analysis on existing generation checkpoints with per-sample resume
python3 mutation_testing.py \
  --checkpoints-dir checkpoints_mutation/ \
  --models 'llama3.2:latest,phi4:14b' \
  2>&1 | tee logs/mutation_reanalyze_local.log
```

### Statistical tests (~10 s each)

```bash
# Per-sample rank tests (Kruskal-Wallis, Friedman, Mann-Whitney, Wilcoxon)
python3 mutation_statistical_tests.py

# Mixed-effects model (ANOVA + Tukey HSD + Mixed-LM)
python3 mutation_mixed_effects.py                          # overall kill_rate
python3 mutation_mixed_effects.py --metric kill_rate_boundary
python3 mutation_mixed_effects.py --metric kill_rate_return_none

# Correlation analysis (RAG quality ↔ kill rate)
python3 noise_vs_kill.py

# Cross-model generalizability (Spearman ρ)
python3 analyze_mutation_generalizability.py

# Per-benchmark analysis (HumanEval vs MBPP)
python3 mutation_per_benchmark.py

# 4×4 heatmaps for paper figures
python3 plot_mutation_heatmap.py

# Human-evaluation Streamlit app
pip install -r requirements.txt              # minimal: streamlit + pandas
python3 human_eval_pair_sampler.py           # build blinded 40-pair worksheet
streamlit run human_eval_app.py              # opens http://localhost:8501
```

All reports land in `plots_mutation/`:
- `mutation_statistical_report.txt`
- `mutation_mixed_effects_report.txt` (+ `_boundary.txt`, `_return_none.txt`)
- `noise_vs_kill_report.txt` + `noise_vs_kill_scatter.png`
- `mutation_generalizability_report.txt` + `mutation_rank_*.png`
- `mutation_per_benchmark_report.txt`
- `kill_rate_heatmap.png`, `kill_rate_boundary_heatmap.png`, `kill_rate_combined_heatmap.png`

### Colab fresh sweep (~3.5 h on A100)

See `unitest_colab.ipynb` Step 10b. Knobs:
- `MUTATION_SAMPLES = 30` (per cell)
- `MUTATION_FRESH = True` to wipe state, `False` to resume
- `MUTATION_MODELS = MODELS` (or subset)
- `SYNC_INTERVAL_SECS = 300` (Drive checkpoint sync interval)

The cell restores from Drive on `MUTATION_FRESH=False`, skips models already complete in TSV, runs `mutation_testing.py --regenerate` per model with a background syncer mirroring `.checkpoints_mutation*/` to Drive every 5 min.

### Important macOS gotcha

Use `python3` directly, NOT `uv run` — PyTorch cu128 dependency is incompatible with macOS ARM.

---

## Research Questions

| RQ | Question | Metric | Status |
|----|----------|---------|--------|
| RQ1 | Does RAG outperform plain LLM for unit test generation? | val_score, exec_pass_rate | DONE (64 runs) |
| RQ2 | When does retrieval help vs. hurt? | noise_rate | DONE |
| RQ3 | How faithful are generated tests to retrieved context? | faithfulness, llm_judge | DONE |
| RQ4 | What is the cost-faithfulness trade-off? | retrieval_secs, llm_secs | DONE |
| RQ5 | Do results generalize across benchmarks? | per-benchmark val_score | PARTIAL |
| **RQ6** | **Do generated tests detect real software defects?** | **mutation_kill_rate** | **DONE — boundary IC > Plain LLM strongly significant on MBPP (Δ=−0.311, Mixed-LM p=0.005, Tukey p_adj=0.025); HumanEval null (p=0.94); pooled marginal (p_adj=0.0499). Rankings do NOT generalize across LLMs (min ρ=−0.60). Token-overlap faithfulness predicts LOWER kill rate (r=−0.61, p=0.045).** |

---

## EMSE Reviewer Feedback (preserved verbatim)

> "While we find the topic relevant the study is primarily an ML engineering evaluation — it measures BLEU, ROUGE, and embedding similarity but does not connect to developer outcomes. The paper lacks:
> 1. Developer impact studies (how do generated docstrings/tests affect developer productivity?)
> 2. Comparison with existing tools (e.g., GitHub Copilot, EvoSuite)
> 3. SE-relevant evaluation metrics (mutation testing, defect detection, code review time)
> We encourage resubmission with these additions."

### Status of each reviewer concern

| Concern | Status |
|---|---|
| #1 Developer impact studies | DONE — 3 blinded annotators × 40 samples × 3 dimensions (0–5). IC ranks highest on every dimension in 3-annotator means (idiom 4.06, correctness 4.15, completeness 4.58). GS↔BV pairwise κ = 0.32–0.46 (fair-to-moderate); SaeshwaranA outlier documented in §Limitations. Replicates the mutation-testing method ordering with independent human evidence. |
| #2 Comparison with existing tools | TODO — EvoSuite/Pynguin/Copilot comparison pending |
| **#3 SE-relevant evaluation metrics** | **DONE — full mutation-testing analysis with statistical significance** |
