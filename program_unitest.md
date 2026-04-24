# autoresearch — Unit Test Generation

Autonomous experimentation for PhD research on LLM/RAG-based unit test generation.
Mirrors program.md but for the unit test task.

## Repository Structure

```
autoresearch/
├── prepare_unitest.py          — FIXED harness: dataset download, VectorStore, evaluation, val_score, execute_tests(). DO NOT MODIFY.
├── train_unitest.py            — AGENT-EDITABLE: METHOD, REASONING, prompts, RAG config, temperatures, models
├── faithfulness.py             — Shared metric: token-overlap + LLM-as-Judge faithfulness. Read-only reference.
├── visualize_unitest.py        — 13 KPI charts from results_unitest.tsv → plots_unitest/
├── analyze_generalizability.py — Spearman rank correlation across models → plots_generalizability/
├── statistical_tests.py        — Kruskal-Wallis, Mann-Whitney U, Bonferroni, Cohen's d, weight sensitivity
├── human_eval_sampler.py       — Stratified annotation sampler (40 samples) + post-annotation validation
├── compare_tasks.py            — Cross-task comparison: unit test vs docstring generation (RQ4)
├── test_run.py                 — Pipeline verification: smoke-tests all 4 methods on 2 samples
├── unitest_colab.ipynb         — Colab notebook: full 64-run multi-model sweep on A100
├── program_unitest.md          — This file: agent instructions and repo knowledge base
├── results_unitest.tsv         — Experiment results (auto-generated, not committed)
├── plots_unitest/              — 13 visualization charts (auto-generated)
├── plots_generalizability/     — Cross-model analysis charts + reports (auto-generated)
└── pyproject.toml              — Dependencies (Python ≥3.10)
```

## Architecture

### Data Pipeline
1. `prepare_unitest.py` downloads HumanEval + MBPP (+ ClassEval for v4) datasets
2. Subsets to 100 samples (seed=42), caches to `~/.cache/autoresearch_unitest/eval_dataset_v3.pkl` (or `v4`)
3. `train_unitest.py` shuffles the dataset at runtime (`random.Random(42)`) to interleave HumanEval + MBPP + ClassEval samples — ensures per-benchmark scores are valid even when TIME_BUDGET truncates evaluation early
4. Knowledge base: 14 testing documentation URLs → 500-char overlapping chunks (100-char overlap) → cached as `knowledge_base_v3.pkl`
5. VectorStore: in-memory numpy + `all-MiniLM-L6-v2` sentence-transformers embeddings, cosine similarity, top-k retrieval

### Generation Pipeline (`train_unitest.py`)
- 4 methods × 4 reasoning = 16 generator functions, registered in `GENERATORS` dispatch dict
- Each generator takes `function_code: str` and returns `generated_tests: str`
- Per-sample diagnostics tracked via module-level buffers: `_noise_rate_buf`, `_last_context`, `_retrieval_secs`, `_llm_secs`
- Checkpoint system: saves progress per-sample to resume after interruption

### Evaluation Pipeline (`prepare_unitest.py`)
- `evaluate_tests(generated, ground_truth, function_code)` → dict of 5 component metrics
- `compute_val_score(metrics_list)` → weighted composite score
- `execute_tests(generated, function_code)` → pytest subprocess execution (15s timeout, tempfile isolation)

## Methods (4)

| Method | Description | Noise Rate | Faithfulness |
|--------|-------------|------------|--------------|
| `plain_llm` | Direct LLM generation, no retrieval | NaN | NaN |
| `random_rag` | Same pipeline as simple_rag but retrieves RANDOM chunks (ablation baseline) | NaN | computed |
| `simple_rag` | Single retrieval pass from testing docs KB | computed | computed |
| `iterative_critique` | Generate → critique → refine loop with RAG context (default 2 rounds) | computed | computed |

## Reasoning Techniques (4)

| Technique | Description |
|-----------|-------------|
| `base` | Direct prompt — single generation pass |
| `cot` | Chain-of-Thought — step-by-step reasoning before writing tests |
| `tot` | Tree-of-Thought — generate 2 candidates, evaluate and select best |
| `got` | Graph-of-Thought — generate happy path, edge cases, error cases separately, then merge |

## Models (4)

| Model | Size | Architecture | Role |
|-------|------|-------------|------|
| `llama3.2:latest` | 3B | Dense | Fast baseline |
| `phi4:14b` | 14B | Dense | Mid-size general |
| `qwen3.5:9b` | 9B | Dense | Latest-gen general |
| `qwen3-coder:30b` | 30B (3.3B active) | MoE | Code-specialized SOTA |

**Full factorial design: 4 methods × 4 reasoning × 4 models = 64 experiments.**

## Evaluation Metric

**val_score** (higher is better, range 0.0–1.0):

```
val_score = 0.30 × syntactic_validity    — generated tests must be valid Python
          + 0.25 × edge_case_score        — tests cover edge cases (None, empty, zero, etc.)
          + 0.20 × assert_density         — meaningful assertions per test function
          + 0.15 × semantic_sim           — sentence-transformers cosine vs ground truth
          + 0.10 × rouge_1_f1            — lexical overlap with reference test suite
```

## Setup

1. **Agree on a run tag**: propose a tag (e.g. `unitest-apr16`). Branch `autoresearch/<tag>` must not exist.
2. **Create the branch**: `git checkout -b autoresearch/<tag>`
3. **Read the in-scope files**:
   - `prepare_unitest.py` — fixed harness. **DO NOT MODIFY.**
   - `train_unitest.py` — the only file you edit.
   - `faithfulness.py` — shared metric. Read-only reference.
4. **Verify data exists**: Check `~/.cache/autoresearch_unitest/` for `eval_dataset_v3.pkl` (or `v4`) and `knowledge_base_v3.pkl`. If missing, run `python prepare_unitest.py` (one-time, ~5 min).
5. **Initialize results**: Create `results_unitest.tsv` with just the header row if it doesn't exist.
6. **Confirm and go**.

## What you CAN change in `train_unitest.py`

- `METHOD`: `"plain_llm"` | `"random_rag"` | `"simple_rag"` | `"iterative_critique"`
- `REASONING`: `"base"` | `"cot"` | `"tot"` | `"got"`
- `GENERATOR_MODEL` / `HELPER_MODEL`: any Ollama model available locally
- `NUM_CRITIQUE_ROUNDS`: number of critique-refine iterations (default 2; ablation: try 1)
- `DATASET_VERSION`: `"v3"` (HumanEval+MBPP) | `"v4"` (HumanEval+MBPP+ClassEval)
- `TEMPERATURE`, `CRITIQUE_TEMPERATURE`, `REFINE_TEMPERATURE`
- `TOT_TEMP_EXPLORE`, `TOT_TEMP_REFINE`, `TOT_TEMP_SELECT`, `GOT_TEMP_AGGREGATE` — named constants for branch temperatures
- `TOP_K`: number of chunks retrieved for RAG (default 5; v3 KB has ~100-200 chunks)
- Any prompt string: `SYSTEM_PROMPT`, `GENERATION_PROMPT`, `CRITIQUE_PROMPT`, `REFINE_PROMPT`, `COT_PROMPT`, `TOT_*`, `GOT_*`
- `MAX_SAMPLES`: set to integer for quick trial; `None` = use full dataset

## What you CANNOT change

- `prepare_unitest.py` — fixed harness, ground truth metric, execute_tests()
- The `evaluate_tests()` and `compute_val_score()` functions
- The eval dataset subset (fixed seed=42, 100 samples, cache version v3/v4)
- `faithfulness.py` — shared metric used across PhD tasks

## Running an experiment

```bash
.venv/bin/python train_unitest.py > run_unitest.log 2>&1
grep "^val_score:\|^method:\|^model:\|^status:" run_unitest.log
```

For a quick local trial, set `MAX_SAMPLES = 3` at the top of `train_unitest.py`.

## Output format

```
---
val_score:              0.512345
method:                 iterative_critique/cot
model:                  llama3.2:latest
samples_evaluated:      100
total_seconds:          487.3
avg_syntax:             0.8800
avg_edge:               0.5200
avg_assert_density:     0.3400
avg_semantic_sim:       0.4120
avg_rouge:              0.1230
avg_noise_rate:         0.1100
avg_faithfulness:       0.4800
avg_llm_judge_faith:    0.6200
avg_retrieval_secs:     0.420
avg_llm_secs:           12.340
avg_tokens:             850.0
avg_exec_pass_rate:     0.6500
avg_exec_total:         8.2
val_score_humaneval:    0.5340
val_score_mbpp:         0.4890
val_score_classeval:    0.4720
samples_humaneval:      40
samples_mbpp:           35
samples_classeval:      25
dataset_version:        v4
Results appended → results_unitest.tsv
```

## Logging results

Results are **automatically written** to `results_unitest.tsv` (tab-separated) at the end of each run.

TSV columns (25 total):
```
method  model  status  val_score  avg_syntax  avg_edge  avg_assert_density  avg_semantic_sim  avg_rouge  avg_noise_rate  avg_faithfulness  avg_llm_judge_faithfulness  avg_retrieval_secs  avg_llm_secs  avg_tokens  samples_evaluated  val_score_humaneval  val_score_mbpp  samples_humaneval  samples_mbpp  val_score_classeval  samples_classeval  avg_exec_pass_rate  avg_exec_total_tests  dataset_version
```

Column notes:
- `method`: `"iterative_critique/cot"` format (method/reasoning)
- `model`: Ollama model name (e.g. `llama3.2:latest`)
- `status`: `"ok"` | `"partial"` (>50% samples failed) | `"crash"` (unhandled exception)
- `avg_noise_rate`: NaN for plain_llm and random_rag — fraction of retrieved chunks with cosine sim < 0.3 (RQ2)
- `avg_faithfulness`: NaN for plain_llm — token-overlap score, grounding in retrieved context (RQ3). random_rag, simple_rag, iterative_critique all compute this.
- `avg_llm_judge_faithfulness`: NaN for plain_llm — DeepSeek-Coder 6.7B judge score (validated Pearson r=0.925 vs human, RQ3)
- `avg_retrieval_secs` / `avg_llm_secs`: cost breakdown (RQ4 Pareto analysis)
- `avg_tokens`: 0 if Ollama < v0.4 (token counting unavailable)
- `val_score_humaneval` / `val_score_mbpp` / `val_score_classeval`: per-benchmark val_score (dataset source ablation)
- `samples_humaneval` / `samples_mbpp` / `samples_classeval`: sample counts per benchmark in this run
- `avg_exec_pass_rate`: fraction of generated tests that pass when executed via pytest (supplementary, not in val_score)
- `avg_exec_total_tests`: average number of test functions detected per sample
- `dataset_version`: `v3` or `v4` — which eval dataset was used for this run

## The experiment loop

LOOP FOREVER:

1. Check current branch and last result in results_unitest.tsv
2. Pick an experiment idea (see Ideas below, or think of new ones)
3. Modify `train_unitest.py`
4. `git commit -m "experiment: <brief description>"`
5. `.venv/bin/python train_unitest.py > run_unitest.log 2>&1`
6. `grep "^val_score:\|^method:\|^model:" run_unitest.log`
7. If grep empty → crash. Run `tail -50 run_unitest.log` to diagnose. Fix if trivial, else log as crash and move on.
8. Results are auto-written to `results_unitest.tsv` — no manual logging needed.
9. If `val_score` improved → keep commit, advance branch
10. If `val_score` equal or worse → `git reset --hard HEAD~1` (revert)

**Timeout**: If a run exceeds 15 minutes, kill it and treat as crash.

**NEVER STOP**: Run until manually interrupted. Do not ask for permission to continue.

## Ideas to try (in rough order of expected impact)

1. Baseline: run as-is to establish baseline val_score
2. Switch METHOD: plain_llm → random_rag → simple_rag → iterative_critique
3. Switch REASONING: base → cot → tot → got
4. Prompt engineering: make GENERATION_PROMPT more specific about edge cases
5. Temperature tuning: lower TEMPERATURE for more deterministic tests; adjust TOT_TEMP_* constants
6. TOP_K tuning: try TOP_K = 5 or 2
7. Combine best METHOD + REASONING + refined prompts
8. Try different Ollama models (phi4:14b, qwen3.5:9b, qwen3-coder:30b)
9. Adjust CRITIQUE_PROMPT to be stricter or more lenient
10. Add few-shot examples directly into SYSTEM_PROMPT
11. Run random_rag (same as simple_rag but random chunks) — isolates retrieval quality contribution
12. Run NUM_CRITIQUE_ROUNDS=1 vs 2 — quantify marginal gain of second critique round

**Simplicity criterion**: simpler changes that improve val_score are better than complex ones that barely move the needle.

## Diagnostic metrics (not in val_score)

| Metric | Purpose | RQ |
|--------|---------|-----|
| `avg_noise_rate` | Fraction of retrieved chunks with cosine sim < 0.3 (NaN for plain_llm, random_rag) | RQ2 |
| `avg_faithfulness` | Token overlap: generated tests ∩ retrieved context (NaN for plain_llm) | RQ3 |
| `avg_llm_judge_faithfulness` | DeepSeek-Coder 6.7B judge (Pearson r=0.925 vs human, NaN for plain_llm) | RQ3 |
| `avg_retrieval_secs` | Time spent in vector retrieval per sample | RQ4 |
| `avg_llm_secs` | Time spent in LLM generation per sample | RQ4 |
| `avg_exec_pass_rate` | Fraction of generated tests that pass when executed via pytest | RQ1 |
| `avg_exec_total_tests` | Average number of test functions detected per sample | RQ1 |
| `val_score_humaneval` | val_score on HumanEval subset only (dataset ablation) | RQ5 |
| `val_score_mbpp` | val_score on MBPP subset only (dataset ablation) | RQ5 |
| `val_score_classeval` | val_score on ClassEval subset only (class-level ablation, v4 only) | RQ5 |

## Analysis scripts (run after full experiment)

```bash
# Generalizability: Spearman ρ with p-values across models
.venv/bin/python analyze_generalizability.py

# Statistical significance + val_score sensitivity
.venv/bin/python statistical_tests.py

# Human annotation worksheet (40 stratified samples)
.venv/bin/python human_eval_sampler.py

# After annotating: validate automated metric vs human ratings
.venv/bin/python human_eval_sampler.py --validate human_eval_samples_annotated.csv

# Cross-task comparison (requires docstring results too)
.venv/bin/python compare_tasks.py

# Pipeline verification (smoke test before full run)
.venv/bin/python test_run.py
```

## Visualizations

After logging results to `results_unitest.tsv`, generate PhD comparison charts:

```bash
.venv/bin/python visualize_unitest.py
```

Outputs to `plots_unitest/` (13 charts):
- `heatmap.png`                — val_score grid: method × reasoning technique
- `grouped_bar.png`            — val_score grouped bar (all 16 combinations per model)
- `radar.png`                  — per-metric radar: best run per method
- `per_metric_bar.png`         — per-metric bar: best run per method
- `noise_rate.png`             — avg noise rate per RAG method (simple_rag, iterative_critique) (RQ2)
- `cost_breakdown.png`         — stacked retrieval + LLM time per method (RQ4)
- `faithfulness.png`           — avg faithfulness per method (simple_rag, iterative_critique) (RQ3)
- `interaction.png`            — Method × Reasoning interaction plot (parallel lines = no interaction)
- `source_split.png`           — HumanEval vs MBPP vs ClassEval val_score per method (dataset ablation, RQ5)
- `exec_pass_rate.png`         — execution pass rate per method (tests run via pytest, RQ1)
- `model_val_score.png`        — val_score grouped by method × model (cross-model)
- `model_faithfulness.png`     — faithfulness grouped by method × model
- `model_rank_stability.png`   — method ranking lines across models (flat = generalizable)

Outputs to `plots_generalizability/` (via `analyze_generalizability.py`):
- `rank_correlation.png`        — Spearman ρ heatmap with p-values
- `rank_stability.png`          — method ranking lines across models
- `val_score_by_model.png`      — grouped bar: method × model
- `faithfulness_by_model.png`   — faithfulness grouped bar
- `sensitivity_weights.png`     — val_score rank stability across weight perturbations (via `statistical_tests.py`)
- `statistical_report.txt`      — Kruskal-Wallis + Mann-Whitney + Cohen's d + interaction + source analysis
- `generalizability_report.txt` — written summary for thesis appendix

## Colab Notebook (`unitest_colab.ipynb`)

Runs the full 64-experiment sweep on Google Colab A100. Key features:

- **Three-level checkpoint system**:
  1. Per-sample checkpoint saved to Google Drive after every sample
  2. Per-experiment TSV appended to Drive after every experiment
  3. Git push every 5 experiments
- **Disconnect resilience**: On reconnect, Drive copy of `results_unitest.tsv` is source of truth. Always restores from Drive when Drive has more data than local copy.
- **Quick-test cleanup**: Before the main loop, rows with `samples_evaluated < 50` are automatically removed from the TSV to prevent Step 7 quick-test pollution.
- **Steps**: Mount Drive → Install deps → Pull 4 Ollama models → Clone repo → One-time setup → 64-run sweep → Analysis → Visualize → Push

## Research Questions

| RQ | Question | Metrics |
|----|----------|---------|
| RQ1 | Does RAG outperform plain LLM for unit test generation? | val_score, avg_exec_pass_rate |
| RQ2 | When does retrieval help vs. hurt? | avg_noise_rate |
| RQ3 | How faithful are generated tests to retrieved context? | avg_faithfulness, avg_llm_judge_faithfulness |
| RQ4 | What is the cost-faithfulness trade-off? | avg_retrieval_secs, avg_llm_secs |
| RQ5 | Do results generalize across benchmarks? | val_score_humaneval, val_score_mbpp, val_score_classeval |

## Statistical Methodology

1. **Kruskal-Wallis H-test** — omnibus test across all 4 methods (non-parametric, no normality assumption)
2. **Mann-Whitney U** — pairwise comparisons (6 pairs from 4 methods) with Bonferroni correction (α = 0.05/6 = 0.0083)
3. **Cohen's d** — effect size for each pairwise comparison (small ≥0.2, medium ≥0.5, large ≥0.8)
4. **Interaction analysis** — Method × Reasoning interaction (parallel lines in interaction plot = no interaction)
5. **Weight sensitivity** — ±50% perturbation of val_score weights, check if method rankings are stable
6. **Spearman ρ** — rank correlation across models (ρ ≥ 0.8 = findings generalize)

## Threats to Validity (addressed in paper)

### Internal validity
- **Attribution confound**: `random_rag` METHOD provides a random-retrieval baseline. If `simple_rag` >> `random_rag`, the improvement is from retrieval quality, not just context length.
- **Critique iterations**: `NUM_CRITIQUE_ROUNDS` is agent-editable (default 2). Run with `NUM_CRITIQUE_ROUNDS=1` to quantify marginal gain of the second round.

### External validity
- **Dataset generalisability**: `val_score_humaneval`, `val_score_mbpp`, and `val_score_classeval` are logged per run. Source split chart (`source_split.png`) shows whether results hold across all three benchmarks (function-level + class-level).
- **Model size confound**: llama3.2 (3B) vs phi4 (14B) vs qwen3.5 (9B) vs qwen3-coder (30B MoE) spans diverse parameter ranges and architectures (dense vs MoE). Mitigated by Spearman ρ analysis across all four models.

### Construct validity
- **Automated metric**: val_score weight sensitivity (±50% perturbation) and human evaluation (Pearson r ≥ 0.7 target) validate the composite metric.
- **Embedding model**: `all-MiniLM-L6-v2` used for retrieval — code-specific embedders (e.g. CodeBERT) may improve retrieval quality; noted as future work.
- **Execution-based validation**: `avg_exec_pass_rate` runs generated tests via pytest in a subprocess sandbox (15s timeout per sample). This supplements the proxy-based val_score with ground-truth execution results. Note: execution is best-effort — some tests may fail due to missing imports or environment differences rather than logical errors.
