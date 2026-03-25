# Unit Test Generation via RAG — PhD Research

Comparing **Plain LLM vs Simple RAG vs Iterative Critique RAG** for automated unit test generation, evaluated across reasoning techniques and models.

## Research Questions

- **RQ1** — Does RAG outperform plain LLM for unit test generation?
- **RQ2** — When does retrieval help vs. hurt (noise rate analysis)?
- **RQ3** — How faithful are generated tests to retrieved context?
- **RQ4** — What is the cost-faithfulness trade-off across methods?

## Methods

| Method | Description |
|--------|-------------|
| `plain_llm` | Direct LLM generation, no retrieval |
| `simple_rag` | Single retrieval pass from testing docs KB |
| `iterative_critique` | Generate → critique → refine loop with RAG context |

## Reasoning Techniques

| Technique | Description |
|-----------|-------------|
| `base` | Direct prompt |
| `cot` | Chain-of-Thought — step-by-step reasoning before writing tests |
| `tot` | Tree-of-Thought — generate two candidates, select the best |
| `got` | Graph-of-Thought — generate happy path, edge cases, and error cases separately, then merge |

**Full comparison: 3 methods × 4 reasoning × 3 models = 36 experiments.**

## Evaluation Metric

`val_score` (higher is better):

```
val_score = 0.30 × syntactic_validity
          + 0.25 × edge_case_score
          + 0.20 × assert_density
          + 0.15 × semantic_sim        (sentence-transformers cosine vs. ground truth)
          + 0.10 × rouge_1_f1
```

**Diagnostic metrics** (not in val_score):

| Metric | Purpose |
|--------|---------|
| `noise_rate` | Fraction of retrieved chunks with cosine sim < 0.3 (RQ2) |
| `faithfulness` | Token overlap between generated tests and retrieved context (RQ3) |
| `avg_retrieval_secs`, `avg_llm_secs` | Cost breakdown per method (RQ4) |

## Models

| Model | Size | Role |
|-------|------|------|
| `llama3.2:latest` | 3B | Fast baseline |
| `phi4:14b` | 14B | Mid-size |
| `qwen2.5:14b` | 14B | Mid-size |

Rankings across models compared using **Spearman rank correlation** (ρ ≥ 0.8 = findings generalize).

## Dataset

Fixed 25-sample evaluation subset from **HumanEval + MBPP** (seed=42). Same subset used for every experiment.

## Knowledge Base

8 pytest/unittest documentation pages, embedded with `all-MiniLM-L6-v2` (in-memory vector store, cosine similarity, top-k=3):

- pytest assert, parametrize, exception, getting-started docs
- Python `unittest` stdlib docs
- RealPython pytest guide, GeeksForGeeks unittest, Semaphore pytest tutorial

## Project Files

```
prepare_unitest.py          — fixed harness: dataset, VectorStore, evaluation (do not modify)
train_unitest.py            — edit this: METHOD, REASONING, prompts, RAG config
program_unitest.md          — agent instructions
unitest_colab.ipynb         — Colab notebook: full 36-run multi-model sweep
visualize_unitest.py        — generates 10 KPI charts from results_unitest.tsv
analyze_generalizability.py — Spearman rank correlation across models
faithfulness.py             — shared faithfulness metric
test_run.py                 — 22-check local pipeline verification
```

## Quick Start (Local)

```bash
# One-time setup (~3 min)
python prepare_unitest.py

# Set METHOD, REASONING, GENERATOR_MODEL at top of train_unitest.py, then run
python train_unitest.py

# Visualize
python visualize_unitest.py
open plots_unitest/
```

Set `MAX_SAMPLES = 3` in `train_unitest.py` for a fast trial run.

## Running on Google Colab (Full Experiment)

Open `unitest_colab.ipynb` on a Colab A100:

1. Mount Google Drive
2. Install dependencies
3. Install Ollama + pull all 3 models
4. Clone repo
5. One-time setup
6. Single experiment quick test
7. **Full sweep: 36 runs** (~6–10 hours on A100)
8. Generalizability analysis (Spearman ρ heatmap + rank stability)
9. Visualize results (10 charts)
10. Cross-task comparison with Docstring RAG results (RQ4)
11. Push results to GitHub

**Checkpoint/resume:** checkpoints are saved to Google Drive after every sample. On disconnect, re-run Steps 1–5 then Step 7 — resumes automatically from the last completed sample.

## Outputs

| File | Description |
|------|-------------|
| `results_unitest.tsv` | One row per experiment (not committed) |
| `plots_unitest/` | 10 charts: heatmap, grouped bar, radar, per-metric bar, noise rate, cost breakdown, faithfulness + 3 cross-model charts |
| `plots_generalizability/` | Spearman ρ heatmap, rank stability, val_score/faithfulness by model + written report |
| `summary_all_experiments.csv` | Pivot summary across all models (not committed) |

## License

MIT
