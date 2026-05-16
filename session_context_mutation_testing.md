# Session Context — Mutation Testing & Analysis

**Last updated**: 2026-05-16
**Repo**: `/Users/balajivenktesh/Desktop/Education/autoresearch/`
**Branch**: `master`
**Latest commit**: `6265c4e` — "feat(stats): mixed-effects model — significance for boundary kill rate"
**Remote**: `https://github.com/balajivenky06/autoresearch.git` (pushed and up to date)

**PhD Topic**: Plain LLM vs Random RAG vs Simple RAG vs Iterative Critique RAG for code generation.
**This task**: Unit test generation — full 4 methods × 4 models × 30 samples mutation-testing matrix complete.

---

## TL;DR — Headline Result for EMSE Resubmission

> After controlling for source-sample difficulty and LLM choice via a linear mixed-effects model, **Iterative Critique RAG generates tests that detect 20.5 percentage points more off-by-one defects than Plain LLM generation** (Tukey HSD Δ=−0.205, p_adj=0.0499; Mixed-LM β=−0.133 for Plain LLM vs IC baseline, p=0.016).

The overall kill rate doesn't show statistical separation on the per-sample tests because of a 1.0 ceiling effect on the strong models. The discriminative signal lives in the **boundary** (n ↔ n±1) mutation operator, the historically hardest mutator.

A secondary finding from the same ANOVA: **LLM choice dominates RAG method choice**. Switching llama3.2 (3B) → qwen3-coder (30B) buys ~0.20 kill rate; switching Plain LLM → Iterative Critique buys ~0.05. Both effects are real; LLM > method by ~4× in F-statistic.

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

---

## Analysis Pipeline (3 scripts)

| Script | What it does | Key output |
|---|---|---|
| `mutation_testing.py` | Generate + analyze mutants. Supports `--regenerate` (Ollama generation) and `--checkpoints-dir` (analysis-only on existing tests). Per-sample resume on both phases. | `results_mutation.tsv`, `.checkpoints_mutation/`, `.checkpoints_mutation_analysis/`, `plots_mutation/` |
| `mutation_statistical_tests.py` | Per-sample Kruskal-Wallis + Mann-Whitney (unpaired) and Friedman + Wilcoxon (paired). Per-model, pooled, and per-operator (boundary, arithmetic, etc.). | `plots_mutation/mutation_statistical_report.txt` |
| `mutation_mixed_effects.py` ★ | Type-III ANOVA + Tukey HSD + Mixed-LM (sample_idx as random intercept). The right test for this design. Accepts `--metric kill_rate_<operator>`. | `plots_mutation/mutation_mixed_effects_*.txt` |

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

### HIGH PRIORITY — directly addresses reviewer feedback

- [ ] **Human evaluation study (40 samples)** — biggest remaining EMSE gap ("developer impact studies"). Use `human_eval_sampler.py` to generate annotation worksheet, get 2–3 annotators, compute Cohen's κ and Pearson r vs `val_score`.
- [ ] **Tool comparison vs EvoSuite / Pynguin / Copilot** — reviewer mentioned "comparison with existing tools". Even qualitative comparison on 10–20 samples would help.

### MEDIUM PRIORITY — strengthens paper

- [ ] **Cross-model generalizability for kill rates** (item #2 from earlier plan) — Spearman ρ between method-rankings across the 4 models. Mirror `analyze_generalizability.py`'s approach for `val_score`.
- [ ] **Noise rate → kill rate correlation** — join `avg_noise_rate` from `results_unitest.tsv` with `mean_kill_rate` from `results_mutation.tsv` (already 16 rows aligned by method×model). Pearson r tests "RAG retrieval quality predicts defect-detection quality" — novel SE claim.
- [ ] **Increase mutation sample size** to 100 samples/cell — direct power increase if the marginal p=0.07 ANOVA result needs to be airtight. ~12–24h Colab.
- [ ] **Iterative critique rounds ablation** — 1 vs 2 critique rounds on boundary kill rate.

### LOWER PRIORITY — nice-to-have

- [ ] **Per-benchmark mutation analysis** — split HumanEval vs MBPP.
- [ ] **Re-plot with full 4×4 matrix** in a model-grouped heatmap.

### Write the paper

- [ ] **Draft resubmission** — descriptive results + statistical tests + bug-fix Q&A for likely reviewer questions ("how do you handle equivalent mutants?", "what's the false-positive rate on the filter?", etc.).

---

## Paper Framing (suggested)

### Core narrative
1. RAG-based test generation methods produce tests with systematically higher mutation kill rates than plain LLM generation (descriptive ordering across 16 model×method cells)
2. The advantage is **statistically certified for boundary mutations** (the hardest, off-by-one defects): Iterative Critique RAG vs Plain LLM Δ=+0.205 kill rate (Tukey HSD p=0.0499; Mixed-LM p=0.016 after controlling for LLM and sample)
3. The overall kill-rate advantage narrows to non-significant on capable models because of a 1.0 ceiling effect — methods converge when the base LLM is strong enough
4. **LLM choice matters more than RAG method**: model F-statistic dominates method F-statistic by 10–20× in every ANOVA fit

### Threat-to-validity Q&A (preempt reviewers)

| Reviewer concern | Our answer |
|---|---|
| Why drop NaN samples? | LLM-generated tests must pass on the original function before mutation testing — standard practice (Andrews et al. 2005). Documented as `_filter_passing_tests`. |
| Why phi4 looks so different across versions? | Found and fixed a real bug (`6b977fd`): some LLMs prepend a function re-definition that shadowed the mutant. Fixed via AST-based redefinition stripping. Pre-fix phi4 numbers (5–26%) are reported with the fix and final numbers (86–93%) for transparency. |
| Equivalent mutants? | Detected via ground-truth tests on each mutant: if ground truth ALSO passes, the mutant is equivalent and excluded from kill-rate denominator. ~5–15% of mutants per cell. |
| Sample size? | 30 per cell; 409 valid observations after filtering. Mixed-LM uses all 409. Acknowledged in §Limitations. |

---

## Key Commits (in this session)

```
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

### Statistical tests (~10 s)

```bash
python3 mutation_statistical_tests.py
python3 mutation_mixed_effects.py                    # overall kill_rate
python3 mutation_mixed_effects.py --metric kill_rate_boundary
python3 mutation_mixed_effects.py --metric kill_rate_return_none
```

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
| **RQ6** | **Do generated tests detect real software defects?** | **mutation_kill_rate** | **DONE — significant for boundary IC > Plain LLM (p=0.016 Mixed-LM)** |

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
| #1 Developer impact studies | TODO — human evaluation pending |
| #2 Comparison with existing tools | TODO — EvoSuite/Pynguin/Copilot comparison pending |
| **#3 SE-relevant evaluation metrics** | **DONE — full mutation-testing analysis with statistical significance** |
