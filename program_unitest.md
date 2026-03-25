# autoresearch — Unit Test Generation

Autonomous experimentation for PhD research on LLM/RAG-based unit test generation.
Mirrors program.md but for the unit test task.

## Setup

1. **Agree on a run tag**: propose a tag (e.g. `unitest-mar12`). Branch `autoresearch/<tag>` must not exist.
2. **Create the branch**: `git checkout -b autoresearch/<tag>`
3. **Read the in-scope files**:
   - `prepare_unitest.py` — fixed: dataset (HumanEval+MBPP), evaluation, val_score. DO NOT MODIFY.
   - `train_unitest.py` — the only file you edit. Prompts, RAG config, METHOD, REASONING.
4. **Verify data exists**: Check `~/.cache/autoresearch_unitest/` for `eval_dataset.pkl` and `knowledge_base.pkl`. If missing, run `uv run prepare_unitest.py` (one-time, ~3 min).
5. **Initialize results**: Create `results_unitest.tsv` with just the header row.
6. **Confirm and go**.

## Experimentation

Each experiment runs for a **fixed time budget of 600 seconds** (10 min generation time), on a fixed 25-sample eval subset. This ensures all runs are directly comparable.

**What you CAN change in `train_unitest.py`:**
- `METHOD`: `"plain_llm"` | `"simple_rag"` | `"iterative_critique"`
- `REASONING`: `"base"` | `"cot"` | `"tot"` | `"got"`
- `GENERATOR_MODEL` / `HELPER_MODEL`: any Ollama model available locally
- `TEMPERATURE`, `CRITIQUE_TEMPERATURE`, `REFINE_TEMPERATURE`
- `TOP_K`: number of docs retrieved for RAG
- Any prompt string: `SYSTEM_PROMPT`, `GENERATION_PROMPT`, `CRITIQUE_PROMPT`, `REFINE_PROMPT`, `COT_PROMPT`, `TOT_*`, `GOT_*`
- Prompt structure, wording, examples, instructions

**What you CANNOT change:**
- `prepare_unitest.py` — fixed harness, ground truth metric
- The `evaluate_tests()` and `compute_val_score()` functions
- The eval dataset subset (fixed seed=42, 25 samples)

**The goal: maximize `val_score`** (higher is better, range 0.0–1.0).

The composite score weights:
- `syntactic_validity` × 0.30 — generated tests must be valid Python
- `edge_case_coverage` × 0.25 — tests must cover edge cases (None, empty, zero, etc.)
- `assert_density`     × 0.20 — meaningful assertions per test function
- `semantic_sim`       × 0.15 — semantic similarity to reference (sentence-transformers cosine)
- `rouge_1_f1`         × 0.10 — lexical overlap with reference test suite

**Ideas to try** (in rough order of expected impact):
1. Baseline: run as-is to establish baseline val_score
2. Switch METHOD: plain_llm → simple_rag → iterative_critique
3. Switch REASONING: base → cot → tot → got
4. Prompt engineering: make GENERATION_PROMPT more specific about edge cases
5. Temperature tuning: lower TEMPERATURE for more deterministic tests
6. TOP_K tuning: try TOP_K = 5 or 2
7. Combine best METHOD + REASONING + refined prompts
8. Try different Ollama models (phi4-mini, qwen2.5, deepseek-coder)
9. Adjust CRITIQUE_PROMPT to be stricter or more lenient
10. Add few-shot examples directly into SYSTEM_PROMPT

**Simplicity criterion**: same as program.md — simpler changes that improve val_score are better than complex ones that barely move the needle.

## Running an experiment

```bash
uv run train_unitest.py > run_unitest.log 2>&1
grep "^val_score:\|^method:\|^model:" run_unitest.log
```

## Output format

```
---
val_score:          0.512345
method:             iterative_critique/cot
model:              llama3.2:latest
samples_evaluated:  25
total_seconds:      487.3
avg_syntax_valid:   0.8800
avg_edge_coverage:  0.5200
avg_assertions:     3.4
avg_semantic_sim:   0.4120
avg_rouge_1:        0.1230
```

## Logging results

Log to `results_unitest.tsv` (tab-separated, NOT comma-separated).

Header and columns:
```
commit	val_score	method	model	status	description	avg_syntax	avg_edge	avg_assert_density	avg_semantic_sim	avg_rouge
```

1. git commit hash (short, 7 chars)
2. val_score achieved (e.g. 0.512345) — use 0.000000 for crashes
3. method (e.g. `iterative_critique/cot`)
4. model (e.g. `llama3.2:latest`)
5. status: `keep`, `discard`, or `crash`
6. short description of what was tried
7–11. per-metric averages from the run log (`avg_syntax_valid`, `avg_edge_coverage`, `avg_assert_density` normalized by 5, `avg_semantic_sim`, `avg_rouge_1`) — use 0.0000 for crashes

Extract columns 7–11 from run log with:
```bash
grep "^avg_" run_unitest.log
```

Example:
```
commit	val_score	method	model	status	description	avg_syntax	avg_edge	avg_assert_density	avg_semantic_sim	avg_rouge
a1b2c3d	0.412300	plain_llm/base	llama3.2:latest	keep	baseline	0.8400	0.4500	0.5600	0.3800	0.1100
b2c3d4e	0.489100	simple_rag/base	llama3.2:latest	keep	add RAG retrieval	0.8800	0.5200	0.6000	0.4200	0.1300
c3d4e5f	0.521400	iterative_critique/cot	llama3.2:latest	keep	COT+critique improves edge coverage	0.9200	0.5800	0.6400	0.4600	0.1400
d4e5f6g	0.498000	iterative_critique/got	llama3.2:latest	discard	GOT slower, no improvement	0.8800	0.5400	0.6000	0.4200	0.1200
```

## The experiment loop

LOOP FOREVER:

1. Check current branch and last result in results_unitest.tsv
2. Pick an experiment idea (see Ideas above, or think of new ones)
3. Modify `train_unitest.py`
4. `git commit -m "experiment: <brief description>"`
5. `uv run train_unitest.py > run_unitest.log 2>&1`
6. `grep "^val_score:\|^method:\|^model:" run_unitest.log`
7. If grep empty → crash. Run `tail -50 run_unitest.log` to diagnose. Fix if trivial, else log as crash and move on.
8. Log result to `results_unitest.tsv` (do NOT commit this file)
9. If `val_score` improved → keep commit, advance branch
10. If `val_score` equal or worse → `git reset --hard HEAD~1` (revert)

**Timeout**: If a run exceeds 15 minutes, kill it and treat as crash.

**NEVER STOP**: Run until manually interrupted. Do not ask for permission to continue.

## Visualizations

After logging results to `results_unitest.tsv`, generate PhD comparison charts:

```bash
uv run visualize_unitest.py
```

Outputs to `plots_unitest/`:
- `heatmap.png`        — val_score grid: method × reasoning technique
- `grouped_bar.png`    — val_score grouped bar (all 12 combinations)
- `radar.png`          — per-metric radar: best run per method (requires extended TSV columns)
- `per_metric_bar.png` — per-metric bar: best run per method (requires extended TSV columns)
