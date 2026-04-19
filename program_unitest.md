# autoresearch — Unit Test Generation

Autonomous experimentation for PhD research on LLM/RAG-based unit test generation.
Mirrors program.md but for the unit test task.

## Setup

1. **Agree on a run tag**: propose a tag (e.g. `unitest-apr16`). Branch `autoresearch/<tag>` must not exist.
2. **Create the branch**: `git checkout -b autoresearch/<tag>`
3. **Read the in-scope files**:
   - `prepare_unitest.py` — fixed harness: dataset (HumanEval+MBPP+ClassEval, 100 samples, seed=42), evaluation, val_score, execution. **DO NOT MODIFY.**
   - `train_unitest.py` — the only file you edit. Prompts, RAG config, METHOD, REASONING, DATASET_VERSION, temperature constants.
   - `faithfulness.py` — shared metric. Read-only reference.
4. **Verify data exists**: Check `~/.cache/autoresearch_unitest/` for `eval_dataset_v3.pkl` (or `v4`) and `knowledge_base_v3.pkl`. If missing, run `python prepare_unitest.py` (one-time, ~5 min — v3 knowledge base has 14 URLs × chunked).
5. **Initialize results**: Create `results_unitest.tsv` with just the header row if it doesn't exist.
6. **Confirm and go**.

## Experimentation

Each experiment runs for a **fixed time budget of 600 seconds** (10 min generation time), on a fixed **100-sample** eval subset. This ensures all runs are directly comparable.

**What you CAN change in `train_unitest.py`:**
- `METHOD`: `"plain_llm"` | `"random_rag"` | `"simple_rag"` | `"iterative_critique"`
  - `random_rag` = same pipeline as simple_rag but retrieves RANDOM chunks (ablation baseline)
- `NUM_CRITIQUE_ROUNDS`: number of critique-refine iterations (default 2; ablation: try 1)
- `DATASET_VERSION`: `"v3"` (HumanEval+MBPP) | `"v4"` (HumanEval+MBPP+ClassEval)
- `REASONING`: `"base"` | `"cot"` | `"tot"` | `"got"`
- `GENERATOR_MODEL` / `HELPER_MODEL`: any Ollama model available locally
- `TEMPERATURE`, `CRITIQUE_TEMPERATURE`, `REFINE_TEMPERATURE`
- `TOT_TEMP_EXPLORE`, `TOT_TEMP_REFINE`, `TOT_TEMP_SELECT`, `GOT_TEMP_AGGREGATE` — named constants for branch temperatures
- `TOP_K`: number of chunks retrieved for RAG (default 5; v3 KB has ~100-200 chunks)
- Any prompt string: `SYSTEM_PROMPT`, `GENERATION_PROMPT`, `CRITIQUE_PROMPT`, `REFINE_PROMPT`, `COT_PROMPT`, `TOT_*`, `GOT_*`

**What you CANNOT change:**
- `prepare_unitest.py` — fixed harness, ground truth metric
- The `evaluate_tests()` and `compute_val_score()` functions
- The eval dataset subset (fixed seed=42, 100 samples, cache version v3/v4)
- `faithfulness.py` — shared metric used across PhD tasks

**The goal: maximize `val_score`** (higher is better, range 0.0–1.0).

The composite score weights:
- `syntactic_validity` × 0.30 — generated tests must be valid Python
- `edge_case_score`    × 0.25 — tests must cover edge cases (None, empty, zero, etc.)
- `assert_density`     × 0.20 — meaningful assertions per test function
- `semantic_sim`       × 0.15 — semantic similarity to reference (sentence-transformers cosine)
- `rouge_1_f1`         × 0.10 — lexical overlap with reference test suite

**Ideas to try** (in rough order of expected impact):
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

TSV columns:
```
method  model  status  val_score  avg_syntax  avg_edge  avg_assert_density  avg_semantic_sim  avg_rouge  avg_noise_rate  avg_faithfulness  avg_llm_judge_faithfulness  avg_retrieval_secs  avg_llm_secs  avg_tokens  samples_evaluated  val_score_humaneval  val_score_mbpp  samples_humaneval  samples_mbpp  val_score_classeval  samples_classeval  avg_exec_pass_rate  avg_exec_total_tests  dataset_version
```

Column notes:
- `method`: `"iterative_critique/cot"` format (method/reasoning)
- `model`: Ollama model name (e.g. `llama3.2:latest`)
- `status`: `"ok"` | `"partial"` (>50% samples failed) | `"crash"` (unhandled exception)
- `avg_noise_rate`: NaN for plain_llm and random_rag — fraction of retrieved chunks with cosine sim < 0.3 (RQ2)
- `avg_faithfulness`: NaN for plain_llm — token-overlap score, grounding in retrieved context (RQ3)
- `avg_llm_judge_faithfulness`: NaN for plain_llm — DeepSeek-Coder 6.7B judge score (validated, RQ3)
- `avg_retrieval_secs` / `avg_llm_secs`: cost breakdown (RQ4 Pareto analysis)
- `avg_tokens`: 0 if Ollama < v0.4 (token counting unavailable)
- `val_score_humaneval` / `val_score_mbpp` / `val_score_classeval`: per-benchmark val_score (dataset source ablation)
- `samples_humaneval` / `samples_mbpp` / `samples_classeval`: sample counts per benchmark in this run
- `avg_exec_pass_rate`: fraction of generated tests that pass when executed via pytest (supplementary)
- `avg_exec_total_tests`: average number of test functions detected per sample
- `dataset_version`: `v3` or `v4` — which eval dataset was used for this run

Extract all metrics:
```bash
grep "^avg_" run_unitest.log
```

## The experiment loop

LOOP FOREVER:

1. Check current branch and last result in results_unitest.tsv
2. Pick an experiment idea (see Ideas above, or think of new ones)
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
```

## Visualizations

After logging results to `results_unitest.tsv`, generate PhD comparison charts:

```bash
.venv/bin/python visualize_unitest.py
```

Outputs to `plots_unitest/`:
- `heatmap.png`         — val_score grid: method × reasoning technique
- `grouped_bar.png`     — val_score grouped bar (all 16 combinations)
- `radar.png`           — per-metric radar: best run per method
- `per_metric_bar.png`  — per-metric bar: best run per method
- `noise_rate.png`      — avg noise rate per RAG method (RQ2)
- `cost_breakdown.png`  — stacked retrieval + LLM time per method (RQ4)
- `faithfulness.png`    — avg faithfulness per method (RQ3)
- `interaction.png`     — Method × Reasoning interaction plot (parallel lines = no interaction)
- `source_split.png`    — HumanEval vs MBPP vs ClassEval val_score per method (dataset ablation)
- `exec_pass_rate.png`  — execution pass rate per method (tests run via pytest)
- `model_val_score.png`       — val_score grouped by method × model (cross-model)
- `model_faithfulness.png`    — faithfulness grouped by method × model
- `model_rank_stability.png`  — method ranking lines across models

Outputs to `plots_generalizability/`:
- `rank_correlation.png`        — Spearman ρ heatmap with p-values
- `rank_stability.png`          — method ranking lines across models
- `val_score_by_model.png`      — grouped bar: method × model
- `faithfulness_by_model.png`   — faithfulness grouped bar
- `sensitivity_weights.png`     — val_score rank stability across weight perturbations
- `statistical_report.txt`      — Kruskal-Wallis + Mann-Whitney + Cohen's d + interaction + source analysis
- `generalizability_report.txt` — written summary for thesis appendix

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
- **Execution-based validation**: `avg_exec_pass_rate` runs generated tests via pytest in a subprocess sandbox (10s timeout per sample). This supplements the proxy-based val_score with ground-truth execution results. Note: execution is best-effort — some tests may fail due to missing imports or environment differences rather than logical errors.
