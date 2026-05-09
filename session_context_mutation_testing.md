# Session Context — Mutation Testing & Analysis (2026-05-06)

## Project Overview

**Repo**: `/Users/balajivenktesh/Desktop/Education/autoresearch/`
**Branch**: `master`
**Latest commit**: `212f707` — "add mutation testing framework + results (SE-relevant defect detection metric)"
**Remote**: `https://github.com/balajivenky06/autoresearch.git` (pushed and up to date)

**PhD Topic**: Comparing Plain LLM vs Simple RAG vs Iterative Critique RAG for code generation tasks.
**This task**: Unit test generation — 4 methods × 4 reasoning × 4 models = 64 experiments completed on Colab A100.

---

## What Was Done This Session

### 1. EMSE Reviewer Feedback Received

The docstring paper was rejected from Empirical Software Engineering (EMSE/Springer) with this key criticism:

> "The study is primarily an ML engineering evaluation... lacking developer impact studies, comparison with existing tools, and SE-relevant evaluation metrics."

This motivated building mutation testing as an SE-relevant metric.

### 2. Mutation Testing Framework Built (`mutation_testing.py`)

Built a complete AST-based mutation testing pipeline from scratch:

**Mutation operators** (adapted from mutmut/PIT):
- **Arithmetic**: `+ ↔ -`, `* ↔ /`
- **Comparison**: `== ↔ !=`, `< ↔ >=`, `> ↔ <=`
- **Boundary**: integer constants ±1
- **Return**: `return x → return None`
- **Negate**: `True ↔ False`

**Key features**:
- `generate_mutants(code)` — AST-based, returns list of (mutant_code, description)
- `run_tests_against_code(test_code, function_code)` — pytest subprocess in tempdir
- `_wrap_bare_asserts()` — handles MBPP bare asserts and HumanEval `check(candidate)` format
- `_filter_passing_tests()` — filters to only tests that pass on original code (standard practice)
- `evaluate_mutants()` — runs tests against all mutants, computes kill rate with equivalent mutant detection
- `regenerate_tests()` — re-generates tests using Ollama with per-sample checkpoint/resume
- `plot_mutation_results()` — bar charts and per-operator breakdown
- CLI: `--checkpoints-dir`, `--regenerate`, `--max-samples`, `--model`, `--methods`, `--results-only`

### 3. Bugs Fixed During Development

| Bug | Root Cause | Fix |
|-----|-----------|-----|
| MBPP ground truth tests fail in pytest | Bare asserts, not `def test_*()` functions | `_wrap_bare_asserts()` wraps in `def test_wrapped():` |
| HumanEval `check(candidate)` format | pytest can't discover `check()` | Detect pattern, append `def test_humaneval(): check(fn_name)` |
| GENERATORS dict key mismatch | Used string `f"{method}/{reasoning}"` but GENERATORS uses tuple keys | Changed to `gen_key = (method, reasoning)` |
| Missing `import time` | `regenerate_tests()` uses `time.time()` | Added import |
| llama3.2:1b too weak | 0/20 tests passed on original code | Pulled `llama3.2:latest` (3B) instead |
| `uv run` fails on macOS | PyTorch cu128 incompatible with ARM | Used `python3` directly |
| Most generated tests fail on original | LLM generates spurious assertions (e.g., expects ValueError, gets IndexError) | Added `_filter_passing_tests()` — filters to only passing test functions before mutation analysis |
| Method label parsing broken | `iterative_critique_base_llama3.2_latest` split on `_` gives wrong labels | Added `KNOWN_METHODS` dict for clean label mapping |

### 4. Files Modified

| File | Changes |
|------|---------|
| `mutation_testing.py` | **NEW** — entire mutation testing framework (~750 lines) |
| `train_unitest.py` | Added `generated_tests`, `function_code`, `ground_truth_tests` to per-sample metrics; disabled checkpoint clearing after TSV write |
| `program_unitest.md` | Added mutation testing section with results, RQ6, usage docs |
| `unitest_colab.ipynb` | Added Step 10b for mutation testing after experiments |

### 5. Ground Truth Baseline Validated

- Ground truth tests: **91.9% kill rate** (100 samples)
- On 20-sample subset: **100% kill rate** (after excluding equivalent mutants)
- This establishes the ceiling for LLM-generated test comparison

---

## Mutation Testing Results (llama3.2:latest, 3B, 20 samples/method)

| Method | Kill Rate | ± Std | Valid/Total Samples | Total Mutants | Killed | Equivalent |
|--------|-----------|-------|---------------------|---------------|--------|------------|
| **Iterative Critique** | **0.8750** | 0.2500 | 4/11 | 19 | 15 | 3 |
| Simple RAG | 0.7882 | 0.3456 | 14/20 | 88 | 59 | 7 |
| Plain LLM | 0.7441 | 0.3412 | 20/20 | 108 | 62 | 8 |
| Random RAG | 0.6038 | 0.4023 | 18/20 | 104 | 53 | 8 |

### Per-Operator Kill Rates

| Method | Arithmetic | Boundary | Comparison | Negate Bool | Return None |
|--------|-----------|----------|------------|-------------|-------------|
| Iterative Critique | 1.000 | 0.400 | 0.750 | 1.000 | 1.000 |
| Simple RAG | 0.667 | 0.500 | 1.000 | 0.700 | 0.750 |
| Plain LLM | 0.560 | 0.371 | 0.583 | 0.700 | 0.808 |
| Random RAG | 0.480 | 0.314 | 0.600 | 0.700 | 0.708 |

**Boundary mutations are universally hardest** to kill (31–50%) — requires precise value assertions.

---

## 64-Experiment Sweep Results (Colab A100)

### val_score (best run per method/model)

| Method | llama3.2:latest | phi4:14b | qwen3-coder:30b | qwen3.5:9b |
|--------|-----------------|----------|-----------------|------------|
| Plain LLM | 0.6800 | 0.6843 | 0.7069 | 0.6904 |
| Random RAG | 0.6726 | 0.6819 | 0.7039 | 0.7009 |
| Simple RAG | 0.6720 | 0.6842 | 0.7020 | 0.6918 |
| Iterative Critique | 0.6633 | 0.6814 | **0.7236** | 0.6197 |

### Generalizability

- **Verdict: DOES NOT FULLY GENERALIZE** (min Spearman ρ = −0.800 < threshold 0.8)
- qwen3-coder:30b (MoE) has inverted method rankings compared to dense models
- llama3.2 and phi4 agree well (ρ = 0.8)

### Faithfulness (token overlap)

| Method | llama3.2 | phi4 | qwen3-coder | qwen3.5 |
|--------|----------|------|-------------|---------|
| Simple RAG | 0.2025 | 0.1288 | 0.1387 | 0.0754 |
| Iterative Critique | 0.1771 | 0.1388 | 0.1216 | 0.0850 |

### Key Findings from 64-Run Sweep

1. **Iterative Critique wins on qwen3-coder:30b** (0.7236) but performs worst on qwen3.5:9b (0.6197)
2. **Plain LLM is surprisingly competitive** — best on 2/4 models (llama3.2, phi4)
3. **val_score differences are small** (~0.02–0.04 range) — methods are more similar than different on this metric
4. **Mutation kill rate shows larger differences** (0.60–0.88 range) — more discriminative metric

---

## Analysis Files Produced

| File | Description |
|------|-------------|
| `results_unitest.tsv` | 62 rows of experiment results (64 minus 2 crashes) |
| `results_mutation.tsv` | Per-method mutation kill rates |
| `plots_unitest/` | 13 KPI charts (heatmap, radar, grouped bar, etc.) |
| `plots_generalizability/` | Spearman ρ heatmap, rank stability, statistical report |
| `plots_mutation/` | kill_rate_by_method.png, kill_rate_by_operator.png, mutation_report.txt |

---

## TODO List (Prioritized for EMSE Resubmission)

### HIGH PRIORITY — Directly addresses reviewer feedback

- [ ] **1. Run mutation testing on Colab with all 4 models**
  - Current results are only llama3.2:latest (3B)
  - Use Step 10b already in `unitest_colab.ipynb`
  - This enables cross-model mutation testing generalizability analysis
  - Will produce a much stronger finding with 4× more data

- [ ] **2. Human evaluation study (40 samples)**
  - EMSE reviewer specifically wanted "developer impact studies"
  - Run `python human_eval_sampler.py` to generate annotation worksheet
  - Need 2–3 annotators to rate test quality
  - Compute inter-rater agreement (Cohen's κ)
  - Compute Pearson r between human ratings and val_score (target r ≥ 0.7)

- [ ] **3. Statistical significance for mutation kill rate**
  - Current results lack p-values for mutation testing
  - With 4 models × 4 methods, enough data for Kruskal-Wallis + Mann-Whitney U
  - Add to `statistical_tests.py` or create `mutation_statistical_tests.py`

### MEDIUM PRIORITY — Strengthens paper

- [ ] **4. Increase mutation sample size**
  - 20 samples/method is thin (iterative_critique has only 4 valid)
  - Consider 50–100 samples on Colab where compute is free
  - More valid samples = narrower confidence intervals

- [ ] **5. Add execution-based pass@k metric**
  - `avg_exec_pass_rate` from 64-run sweep already measures this
  - Include in paper as complementary metric to mutation kill rate

- [ ] **6. Noise rate → kill rate correlation**
  - Test hypothesis: "higher noise rate → lower mutation kill rate"
  - Would be a novel SE finding connecting RAG retrieval quality to defect detection
  - Cross-reference `avg_noise_rate` from results_unitest.tsv with kill rates

### LOWER PRIORITY — Nice-to-have

- [ ] **7. Per-benchmark mutation analysis**
  - Split mutation results by HumanEval vs MBPP
  - Check if defect detection varies by benchmark complexity

- [ ] **8. Iterative critique rounds ablation**
  - Compare 1 vs 2 critique rounds on mutation kill rate
  - Currently only have val_score ablation

- [ ] **9. Write the paper**
  - Enough data for strong resubmission: val_score + mutation kill rate + generalizability + statistical tests
  - Add human evaluation results when available

### ALSO CONSIDER (from EMSE feedback)

- [ ] **10. Comparison with existing tools**
  - Reviewer mentioned "comparison with existing tools"
  - Consider comparing against EvoSuite, Pynguin, or GitHub Copilot test generation
  - Even a qualitative comparison would help

- [ ] **11. Developer survey/interview**
  - Reviewer wanted "developer impact"
  - Small survey (10–15 developers) evaluating LLM vs RAG-generated tests
  - Could be lightweight: show test pairs, ask which is more useful

---

## Technical Notes for Future Sessions

### Running mutation testing locally

```bash
cd /Users/balajivenktesh/Desktop/Education/autoresearch

# Regenerate tests with a specific model (needs Ollama running):
python3 mutation_testing.py --regenerate --max-samples 20 --model llama3.2:latest

# Analyze existing checkpoints:
python3 mutation_testing.py --checkpoints-dir .checkpoints_mutation

# Re-plot from existing results:
python3 mutation_testing.py --results-only
```

### Important: Use `python3` not `uv run` on macOS

`uv run` fails because PyTorch cu128 dependency is incompatible with macOS ARM. Use `python3` directly.

### Checkpoint locations

- **Local mutation checkpoints**: `.checkpoints_mutation/` (4 pkl files, one per method)
- **Colab experiment checkpoints**: `checkpoints/` (downloaded from Google Drive)
- **Results**: `results_unitest.tsv` (64 experiments), `results_mutation.tsv` (4 methods)

### Iterative critique hangs

The iterative_critique method can hang on complex samples (sample 6 took 33 min, sample 12 never completed). The `regenerate_tests()` function has no per-sample timeout. Consider adding a timeout wrapper for Colab runs.

### Test filtering rationale

LLMs (especially 3B) generate tests with spurious assertions — e.g., testing `bell_number(-1)` expects `ValueError` but function raises `IndexError`. The `_filter_passing_tests()` function extracts individual `def test_*()` functions, runs each against original code, and keeps only passing ones. This is standard mutation testing practice (you can't measure defect detection with tests that don't pass on correct code).

### Models available locally (Ollama)

```
llama3.2:latest    3B    2.0 GB
llama3.2:1b        1B    1.3 GB (too weak for test generation)
```

### Key files to read when resuming

1. `program_unitest.md` — full project docs, architecture, RQs, results
2. `mutation_testing.py` — mutation testing framework
3. `train_unitest.py` — experiment runner (methods, generators, checkpoints)
4. `results_mutation.tsv` — mutation results
5. `results_unitest.tsv` — 64-experiment results (not committed to git)

---

## Research Questions

| RQ | Question | Metrics | Status |
|----|----------|---------|--------|
| RQ1 | Does RAG outperform plain LLM for unit test generation? | val_score, exec_pass_rate | DONE (64 experiments) |
| RQ2 | When does retrieval help vs. hurt? | noise_rate | DONE |
| RQ3 | How faithful are generated tests to retrieved context? | faithfulness, llm_judge | DONE (token overlap); judge NaN (DeepSeek not pulled) |
| RQ4 | What is the cost-faithfulness trade-off? | retrieval_secs, llm_secs | DONE |
| RQ5 | Do results generalize across benchmarks? | per-benchmark val_score | PARTIAL (val_score_mbpp NaN for some) |
| RQ6 | Do generated tests detect real software defects? | mutation_kill_rate | DONE (1 model); need all 4 models |

---

## Git History (relevant commits)

```
212f707 add mutation testing framework + results (SE-relevant defect detection metric)
103542e add experiment results: llama3.2:latest, phi4:14b, qwen3.5:9b, qwen3-coder:30b — 36 runs
23825b5 progress [64/64]: results + logs after 64 experiments
1848ff9 comprehensive audit: fix all bugs across entire codebase
756f76f fix: audit bugs — missing function_code in random_rag_tot, hardcoded y-axis
0fe8fa6 fix: critical disconnect-resume bug + add pytest to Colab pip install
d4a64c4 update models: qwen2.5:14b → qwen3.5:9b, qwen2.5-coder:32b → qwen3-coder:30b
```

---

## EMSE Reviewer Feedback (Full Quote)

> "Thank you for your submission to the Empirical Software Engineering (EMSE) journal. While we find the topic relevant the study is primarily an ML engineering evaluation — it measures BLEU, ROUGE, and embedding similarity but does not connect to developer outcomes. The paper lacks:
> 1. Developer impact studies (how do generated docstrings/tests affect developer productivity?)
> 2. Comparison with existing tools (e.g., GitHub Copilot, EvoSuite)
> 3. SE-relevant evaluation metrics (mutation testing, defect detection, code review time)
> We encourage resubmission with these additions."

**What we've done to address this:**
- Built mutation testing framework (addresses point 3)
- Results show Iterative Critique achieves 87.5% mutation kill rate (SE-relevant finding)
- TODO: Human evaluation (point 1) and tool comparison (point 2)
