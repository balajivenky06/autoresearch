# autoresearch

![teaser](progress.png)

*One day, frontier AI research used to be done by meat computers in between eating, sleeping, having other fun, and synchronizing once in a while using sound wave interconnect in the ritual of "group meeting". That era is long gone. Research is now entirely the domain of autonomous swarms of AI agents running across compute cluster megastructures in the skies. The agents claim that we are now in the 10,205th generation of the code base, in any case no one could tell if that's right or wrong as the "code" is now a self-modifying binary that has grown beyond human comprehension. This repo is the story of how it all began. -@karpathy, March 2026*.

The idea: give an AI agent a small but real LLM training setup and let it experiment autonomously overnight. It modifies the code, trains for 5 minutes, checks if the result improved, keeps or discards, and repeats. You wake up in the morning to a log of experiments and (hopefully) a better model. The training code here is a simplified single-GPU implementation of [nanochat](https://github.com/karpathy/nanochat). The core idea is that you're not touching any of the Python files like you normally would as a researcher. Instead, you are programming the `program.md` Markdown files that provide context to the AI agents and set up your autonomous research org. The default `program.md` in this repo is intentionally kept as a bare bones baseline, though it's obvious how one would iterate on it over time to find the "research org code" that achieves the fastest research progress, how you'd add more agents to the mix, etc. A bit more context on this project is here in this [tweet](https://x.com/karpathy/status/2029701092347630069).

---

## PhD Extension: Unit Test Generation via RAG

This fork extends the autoresearch framework for PhD research comparing three LLM-based approaches to **automated unit test generation**.

**Research Question:** Does Retrieval-Augmented Generation (RAG) improve unit test quality over a plain LLM baseline — and do findings generalize across models and reasoning techniques?

### Methods Compared

| Method | Description |
|--------|-------------|
| `plain_llm` | Direct LLM generation with no retrieval |
| `simple_rag` | Single retrieval pass from testing docs knowledge base |
| `iterative_critique` | Generate → critique → refine loop with RAG context |

### Reasoning Techniques (crossed with each method)

| Technique | Description |
|-----------|-------------|
| `base` | Direct prompt |
| `cot` | Chain-of-Thought — step-by-step reasoning before writing tests |
| `tot` | Tree-of-Thought — generate two candidates, select the best |
| `got` | Graph-of-Thought — generate happy path, edge cases, and error cases separately, then merge |

**Full comparison: 3 methods × 4 reasoning = 12 experiments per model.**

### Evaluation Metric

`val_score` (higher is better) — a composite score computed per generated test suite:

```
val_score = 0.30 × syntactic_validity
          + 0.25 × edge_case_score
          + 0.20 × assert_density
          + 0.15 × semantic_sim        (sentence-transformers cosine vs. ground truth)
          + 0.10 × rouge_1_f1
```

**Diagnostic metrics** (not in val_score — used for RQ analysis):
- `noise_rate` — fraction of retrieved chunks with cosine sim < 0.3 (RAG quality, RQ2)
- `faithfulness` — token overlap between generated tests and retrieved context (RQ3)
- `avg_retrieval_secs`, `avg_llm_secs` — cost breakdown (RQ4)

### Multi-Model Generalizability

All 12 experiments are run across 3 models to test whether method rankings hold:

- `llama3.2:latest` (3B) — fast baseline
- `phi4:14b` (14B) — mid-size
- `qwen2.5:14b` (14B) — mid-size

Rankings are compared using **Spearman rank correlation** (ρ ≥ 0.8 across all model pairs = findings generalize).

### Dataset

Fixed 25-sample evaluation subset drawn from **HumanEval + MBPP** (seed=42, reproducible). The same subset is used for every experiment for fair comparison.

### Knowledge Base (for RAG methods)

8 pytest/unittest documentation URLs scraped and indexed using `sentence-transformers/all-MiniLM-L6-v2` (in-memory vector store, numpy cosine similarity, top-k=3):

- pytest assertion docs, parametrize docs, exception docs
- Python unittest docs (stdlib)
- pytest getting started, RealPython pytest guide
- GeeksForGeeks unittest, Semaphore pytest tutorial

### Key Files Added

```
prepare_unitest.py      — fixed harness: dataset, VectorStore, evaluation (do not modify)
train_unitest.py        — agent/researcher modifies: METHOD, REASONING, prompts, RAG config
program_unitest.md      — agent instructions (mirrors program.md)
unitest_colab.ipynb     — Google Colab notebook: full 36-run multi-model experiment
visualize_unitest.py    — 10 KPI charts from results_unitest.tsv
analyze_generalizability.py — Spearman rank correlation across models
faithfulness.py         — shared faithfulness metric (used by both PhD tasks)
test_run.py             — 22-check local trial: verifies pipeline end-to-end
```

### Running Locally (Quick Trial)

```bash
# One-time setup (~3 min: downloads HumanEval + MBPP, builds knowledge base)
python prepare_unitest.py

# Run one experiment (METHOD and REASONING set at top of train_unitest.py)
python train_unitest.py

# Check results
cat results_unitest.tsv

# Generate KPI charts
python visualize_unitest.py
open plots_unitest/
```

To run a fast 3-sample trial set `MAX_SAMPLES = 3` at the top of `train_unitest.py`.

### Running on Google Colab (Full Experiment)

Open `unitest_colab.ipynb` on a Colab A100. Steps in the notebook:

1. Mount Google Drive (all outputs saved to Drive for persistence across disconnects)
2. Install dependencies
3. Install Ollama + pull all 3 models
4. Clone repo
5. One-time setup (dataset + knowledge base)
6. Single experiment quick test
7. **Full sweep: 3 models × 12 experiments = 36 runs** (~6–10 hours on A100)
8. Generalizability analysis (Spearman ρ heatmap + rank stability charts)
9. Visualize results (10 charts)
10. Cross-task comparison with Docstring RAG results (RQ4)
11. Push results to GitHub

**Checkpoint/resume:** train_unitest.py saves a checkpoint to Google Drive after every sample. On Colab disconnect, re-run Steps 1–5 then Step 7 — it resumes from the last completed sample automatically.

### Output Files

| File | Description |
|------|-------------|
| `results_unitest.tsv` | One row per experiment run (not committed to git) |
| `summary_all_experiments.csv` | Pivot summary across all models (not committed) |
| `plots_unitest/` | 10 KPI charts (7 single-model + 3 cross-model) |
| `plots_generalizability/` | 4 generalizability charts + written report |
| `.checkpoints/` | Per-run resume state (not committed) |

---

## Original Autoresearch (LLM Training)

### How it works

The repo is deliberately kept small and only really has three files that matter:

- **`prepare.py`** — fixed constants, one-time data prep (downloads training data, trains a BPE tokenizer), and runtime utilities (dataloader, evaluation). Not modified.
- **`train.py`** — the single file the agent edits. Contains the full GPT model, optimizer (Muon + AdamW), and training loop. Everything is fair game: architecture, hyperparameters, optimizer, batch size, etc. **This file is edited and iterated on by the agent**.
- **`program.md`** — baseline instructions for one agent. Point your agent here and let it go. **This file is edited and iterated on by the human**.

By design, training runs for a **fixed 5-minute time budget** (wall clock, excluding startup/compilation), regardless of the details of your compute. The metric is **val_bpb** (validation bits per byte) — lower is better, and vocab-size-independent so architectural changes are fairly compared.

If you are new to neural networks, this ["Dummy's Guide"](https://x.com/hooeem/status/2030720614752039185) looks pretty good for a lot more context.

### Quick start

**Requirements:** A single NVIDIA GPU (tested on H100), Python 3.10+, [uv](https://docs.astral.sh/uv/).

```bash

# 1. Install uv project manager (if you don't already have it)
curl -LsSf https://astral.sh/uv/install.sh | sh

# 2. Install dependencies
uv sync

# 3. Download data and train tokenizer (one-time, ~2 min)
uv run prepare.py

# 4. Manually run a single training experiment (~5 min)
uv run train.py
```

If the above commands all work ok, your setup is working and you can go into autonomous research mode.

### Running the agent

Simply spin up your Claude/Codex or whatever you want in this repo (and disable all permissions), then you can prompt something like:

```
Hi have a look at program.md and let's kick off a new experiment! let's do the setup first.
```

The `program.md` file is essentially a super lightweight "skill".

### Project structure

```
prepare.py      — constants, data prep + runtime utilities (do not modify)
train.py        — model, optimizer, training loop (agent modifies this)
program.md      — agent instructions
pyproject.toml  — dependencies
```

### Design choices

- **Single file to modify.** The agent only touches `train.py`. This keeps the scope manageable and diffs reviewable.
- **Fixed time budget.** Training always runs for exactly 5 minutes, regardless of your specific platform. This means you can expect approx 12 experiments/hour and approx 100 experiments while you sleep. There are two upsides of this design decision. First, this makes experiments directly comparable regardless of what the agent changes (model size, batch size, architecture, etc). Second, this means that autoresearch will find the most optimal model for your platform in that time budget. The downside is that your runs (and results) become not comparable to other people running on other compute platforms.
- **Self-contained.** No external dependencies beyond PyTorch and a few small packages. No distributed training, no complex configs. One GPU, one file, one metric.

### Platform support

This code currently requires that you have a single NVIDIA GPU. In principle it is quite possible to support CPU, MPS and other platforms but this would also bloat the code. I'm not 100% sure that I want to take this on personally right now. People can reference (or have their agents reference) the full/parent nanochat repository that has wider platform support and shows the various solutions (e.g. a Flash Attention 3 kernels fallback implementation, generic device support, autodetection, etc.), feel free to create forks or discussions for other platforms and I'm happy to link to them here in the README in some new notable forks section or etc.

Seeing as there seems to be a lot of interest in tinkering with autoresearch on much smaller compute platforms than an H100, a few extra words. If you're going to try running autoresearch on smaller computers (Macbooks etc.), I'd recommend one of the forks below. On top of this, here are some recommendations for how to tune the defaults for much smaller models for aspiring forks:

1. To get half-decent results I'd use a dataset with a lot less entropy, e.g. this [TinyStories dataset](https://huggingface.co/datasets/karpathy/tinystories-gpt4-clean). These are GPT-4 generated short stories. Because the data is a lot narrower in scope, you will see reasonable results with a lot smaller models (if you try to sample from them after training).
2. You might experiment with decreasing `vocab_size`, e.g. from 8192 down to 4096, 2048, 1024, or even - simply byte-level tokenizer with 256 possibly bytes after utf-8 encoding.
3. In `prepare.py`, you'll want to lower `MAX_SEQ_LEN` a lot, depending on the computer even down to 256 etc. As you lower `MAX_SEQ_LEN`, you may want to experiment with increasing `DEVICE_BATCH_SIZE` in `train.py` slightly to compensate. The number of tokens per fwd/bwd pass is the product of these two.
4. Also in `prepare.py`, you'll want to decrease `EVAL_TOKENS` so that your validation loss is evaluated on a lot less data.
5. In `train.py`, the primary single knob that controls model complexity is the `DEPTH` (default 8, here). A lot of variables are just functions of this, so e.g. lower it down to e.g. 4.
6. You'll want to most likely use `WINDOW_PATTERN` of just "L", because "SSSL" uses alternating banded attention pattern that may be very inefficient for you. Try it.
7. You'll want to lower `TOTAL_BATCH_SIZE` a lot, but keep it powers of 2, e.g. down to `2**14` (~16K) or so even, hard to tell.

I think these would be the reasonable hyperparameters to play with. Ask your favorite coding agent for help and copy paste them this guide, as well as the full source code.

### Notable forks

- [miolini/autoresearch-macos](https://github.com/miolini/autoresearch-macos) (MacOS)
- [trevin-creator/autoresearch-mlx](https://github.com/trevin-creator/autoresearch-mlx) (MacOS)
- [jsegov/autoresearch-win-rtx](https://github.com/jsegov/autoresearch-win-rtx) (Windows)

## License

MIT
