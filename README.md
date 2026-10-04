# Mutation-Testing Quality of LLM and RAG-based Unit Test Generators

Replication package for a cross-model empirical study asking a narrow question:
**does retrieval augmentation make LLM-generated unit tests better at catching
bugs?**

The answer is no, and most of this repository exists to make that answer
checkable.

---

## Headline results

A 4 × 4 factorial experiment — four generation methods × four open-weight LLMs
— over 100 HumanEval and MBPP functions, producing 1,443 valid observations and
9,660 mutants.

| Finding | Evidence |
|---|---|
| **Method choice has no detectable effect** | spread 0.011 in mean kill rate; ANOVA *F* = 0.21, *p* = 0.890; no Tukey pair separates in any scope |
| **Model choice dominates it 21.7×** | model spread 0.232; *F* = 108.5, *p* < 10⁻⁶⁰ |
| **Semantic retrieval ≈ random retrieval** | Simple RAG beats a random-chunk placebo from the same corpus by **0.3 percentage points** |
| **A strong correlation that isn't one** | faithfulness vs kill rate pools to *r* = −0.907, but partial *r* controlling for model is **+0.079** (*p* = 0.818) |
| **Humans and mutants disagree** | annotators rank Iterative Critique top on all three rubric dimensions; its kill rate is indistinguishable from plain prompting, and ratings don't predict kill rate (*r* = +0.18, *p* = 0.30) |
| **SBST and LLMs split by operator family** | Pynguin leads on arithmetic (0.850 vs 0.809) and negate-boolean (1.000 vs 0.917); LLMs lead on comparison (0.917 vs 0.250) and boundary (0.817 vs 0.554) |

The null survives two robustness checks: a **decontaminated replication**
(functions and parameters renamed via AST; kill rates go *up*, method
*F* = 0.072) and a **matched-subset replication** controlling for differential
test-filter attrition (spread 0.011 on 273 matched function-model units).

---

## Experimental design

**Methods** (the only factor under test):

| Method | What it does |
|---|---|
| `plain_llm` | Direct generation, no retrieval |
| `random_rag` | **Placebo.** Identical to `simple_rag` but the 5 chunks are drawn uniformly at random instead of by cosine rank. Isolates retrieval *relevance* from extra context. |
| `simple_rag` | One cosine-similarity pass, top *k* = 5 chunks |
| `iterative_critique` | `simple_rag` draft, then up to 3 critique-and-refine rounds |

**Models:** `llama3.2:latest` (3B), `phi4:14b` (14B), `qwen3.5:9b` (9B),
`qwen3-coder:30b` (30B-A3.3B MoE) — all via Ollama, all Q4_K_M.

**Corpus:** 100 functions from HumanEval + MBPP, fixed by
`random.Random(42).shuffle` and recorded in `corpus_manifest.tsv` with a
fingerprint. `verify_corpus.py` checks any checkpoint against it.

**Metric:** mutation kill rate. Five AST operators — arithmetic, boundary,
comparison, negate-boolean, return-None. Suites that fail against the
*original* function are discarded before mutation, since a suite that can't
pass on correct code can't meaningfully detect a defect in it.

### Knowledge base

The object the whole null is about, so the real numbers rather than the
configured ones:

- **14 URLs configured, 2 failed to fetch → 12 contribute**
- **964 chunks** (500-char windows, 100-char overlap, <50 chars discarded)
- `all-MiniLM-L6-v2`, 384-dim, exact cosine over an in-memory NumPy matrix —
  no approximate index, so ranking is deterministic
- Query = `"pytest unit testing examples patterns for python function: "` +
  first 300 chars of the function

Composition is uneven in a way that bears on the result: `unittest` +
`unittest.mock` supply **46%** of chunks, while the Hypothesis quickstart —
the only edge-case-oriented source — supplies **1.1%**.

The `noise_rate` diagnostic (fraction of retrieved chunks below cosine 0.3) is
**identically zero** for every query. Reported as degenerate rather than
dropped: it rules out "the retriever returns garbage" as an explanation of the
null, without establishing that what it returns is useful.

---

## Reproducing the analysis

Everything below runs from the committed artifacts — no GPU, no network.

```bash
# regenerate results_mutation.tsv from the analysis checkpoints alone
python3 rebuild_tsv.py

# statistics: ANOVA, Tukey, mixed-effects, per-benchmark, generalizability
python3 mutation_statistical_tests.py
python3 mutation_mixed_effects.py
python3 mutation_per_benchmark.py
python3 analyze_mutation_generalizability.py

# inter-rater agreement (self-validates against Krippendorff's worked example)
python3 krippendorff_alpha.py

# figures
python3 plot_rebuild_figures.py
python3 plot_stale_figures.py
```

### The two gates

Run both before trusting any number, and before any submission:

```bash
python3 check_paper_consistency.py   # every figure, table and in-text
                                     # statistic must trace to the TSV
python3 verify_citations.py          # DOI resolution + topic match for
                                     # every bibliography entry
```

`check_paper_consistency.py` exists because a stale row once survived for
months: one cell kept its 30-sample values while every other moved to 100, and
the TSV disagreed with both the report and the checkpoints. It compares three
layers — TSV against report, tables against TSV, prose claims against TSV.

> **macOS note:** use `python3`, not `uv run`. The pinned PyTorch build is
> CUDA-only and fails on Apple Silicon.

---

## Re-running the sweep

Generation needs GPUs and Ollama; the published results came from a Colab A100.

```bash
python3 prepare_unitest.py                 # one-time: datasets + knowledge base
python3 train_unitest.py                   # one cell; edit METHOD/REASONING/model
python3 mutation_testing.py --help         # the sweep driver
python3 pynguin_runner.py --from-corpus --n 40 --budget 60
python3 decontaminate.py                   # build the renamed corpus
```

`mutation_testing.py` validates every existing checkpoint against the corpus
manifest *before* any side effect, so a run against a different corpus aborts
rather than silently mixing populations.

---

## What's in the package, and what isn't

| Artifact | Status |
|---|---|
| `results_mutation.tsv`, `results_mutation_decontaminated.tsv` | committed |
| `checkpoints_mutation_analysis/` (+ decontam) | committed — per-sample kill/survive/equivalent counts for all 100 functions |
| `corpus_manifest.tsv` | committed — the 100 task IDs and fingerprint |
| `human_eval_annotations/` | committed — three annotators × 40 pairs |
| `plots_mutation/` | committed |
| **`checkpoints_mutation/` (generated test source)** | **30-function pilot only** |

The last row is a real limitation. The 100-function sweep ran on hosted
accelerators and only the *analysis* checkpoints were synced back; those carry
per-sample counts but not the generated test code. **Every statistic in the
paper is recomputable from this repository**, but inspecting the generated
tests for all 100 functions would require re-running the sweep.

---

## Layout

```
mutation_testing.py              sweep driver, AST mutation operators, checkpointing
mutation_statistical_tests.py    ANOVA, Tukey, Kruskal-Wallis, per-sample loader
mutation_mixed_effects.py        mixed-effects regression, sample_idx random intercept
mutation_per_benchmark.py        HumanEval vs MBPP decomposition
analyze_mutation_generalizability.py   cross-model Spearman
krippendorff_alpha.py            self-validating inter-rater agreement
decontaminate.py                 AST rename pass for the contamination check
verify_corpus.py                 read-only corpus identity check
pynguin_runner.py                SBST baseline
rebuild_tsv.py                   derive the TSV from checkpoints only
check_paper_consistency.py       gate: numbers ↔ artifacts
verify_citations.py              gate: bibliography ↔ DOI registries
plot_rebuild_figures.py          heatmaps, attrition, contamination, distribution
plot_stale_figures.py            faithfulness scatter, rank correlation, rank stability
human_eval_app.py                Streamlit annotation interface
prepare_unitest.py               datasets + knowledge base (fixed harness)
train_unitest.py                 generation: method, reasoning, prompts, RAG config
paper_draft.tex / references.bib  manuscript (elsarticle, targeting IST)
```

## Human evaluation

Three annotators rated 40 stratified `(function, generated_tests)` pairs blinded
to method and model, on three 0–5 behaviourally-anchored dimensions.

Agreement is **below** the conventional threshold — ordinal Krippendorff's
α = −0.001 / −0.127 / +0.304 — driven by one annotator's systematically lower
scale use (means 2.88/3.45/3.62 against 4.40/4.25/4.08). This is reported
rather than hidden; the pairwise κ between the other two annotators is
0.32–0.46. Setup and rubric: [`README_human_eval.md`](README_human_eval.md).

```bash
python3 human_eval_pair_sampler.py    # build the blinded worksheet
streamlit run human_eval_app.py
```

## License

MIT
