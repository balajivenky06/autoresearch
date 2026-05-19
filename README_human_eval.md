# Human Evaluation — Unit-Test Quality Annotation

A small Streamlit app for human annotators to rate LLM-generated unit
tests. Used to validate the automated metrics (val_score, mutation kill
rate) for the EMSE resubmission of the RAG-for-unit-tests study.

You'll see **40 (function, generated_tests) pairs**, blinded — you don't
know which generation method produced each — and score each on three
dimensions on a 0–5 scale.

---

## For annotators — how to run the app

### 1. Clone or download this folder

```bash
git clone https://github.com/balajivenky06/autoresearch.git
cd autoresearch
git checkout feature/human-eval-app   # or master — same content
```

### 2. Install the (minimal) dependencies

```bash
pip install -r requirements.txt
```

Only two packages — `streamlit` and `pandas`. The full research stack
(torch, transformers, scipy, statsmodels) is NOT required for annotation.

> **macOS gotcha**: if `pip install` complains about PyTorch / cu128,
> ignore it — those come from the project's `pyproject.toml` and aren't
> needed here. Just use the `requirements.txt` install above.

### 3. Launch the app

```bash
streamlit run human_eval_app.py
```

Your browser opens at `http://localhost:8501`.

### 4. Log in and annotate

- Enter your initials or any short ID when prompted (e.g. `bv`,
  `jane.r`). This becomes your private save file
  (`human_eval_annotations/{your_id}.csv`) — pick something stable so
  you can resume.
- For each sample, read the **function under test** (left) and the
  **generated tests** (right), then assign 0–5 on each of three
  dimensions:
  - **Test idiom quality** — pytest style: parametrize, fixtures,
    naming, assertion structure
  - **Correctness** — would these tests pass on a *correct*
    implementation of the function?
  - **Completeness** — coverage across happy path / edge cases /
    error cases
  Each value (0..5) has an anchor description right under the radio
  buttons — pick the anchor that best matches what you see.
- Use the **Notes** box for anything qualitative (missing edge cases,
  spurious assertions, surprises).
- **Save & Next** persists immediately — close the tab any time and
  resume by re-entering the same ID.

### 5. When done

- The app shows a "Download my annotations CSV" button on the
  completion screen.
- Send that CSV to the researcher (rename it to `{your_id}.csv`
  if it isn't already).
- Total expected time: **~100 minutes** for all 40 samples (~2-3 min
  each). Feel free to split across sessions.

---

## For researchers — full workflow

### Pre-annotation

```bash
# 1. Generate the stratified worksheet from existing mutation checkpoints.
python3 human_eval_pair_sampler.py --n 40 --seed 42

# Produces:
#   human_eval_pairs.csv         <- blinded; safe to share with annotators
#   human_eval_pairs.meta.csv    <- private mapping; NEVER share
```

The blinded worksheet contains only `sample_id`, `function_code`,
`generated_tests`, `ground_truth_tests`, and the empty annotation
columns. The meta CSV is the private mapping from `sample_id` to
`method` / `model` / `source` / `kill_rate`; it's `.gitignore`d so it
can't accidentally land on GitHub.

### Distribution

Three deployment options:

| Option | Setup | Best for |
|---|---|---|
| Each annotator runs the app locally | They clone repo + `pip install -r requirements.txt` + `streamlit run` | Trusted collaborators with Python set up |
| Single shared instance | `streamlit run human_eval_app.py --server.address=0.0.0.0 --server.port=8501`, share the URL on your LAN / Tailscale / ngrok | When annotators don't want to install anything |
| Streamlit Community Cloud | Push branch, link via share.streamlit.io | Remote / international raters |

The blinded worksheet is committed to the repo; the meta CSV is not, so
even the public-cloud option won't leak method/model identity. Make
sure annotators don't open the meta file by accident.

### Post-annotation

After each annotator returns their CSV, drop the files into
`human_eval_annotations/`:

```
human_eval_annotations/
├── alice.csv
├── bob.csv
└── carol.csv
```

Then run the validation script (TODO, separate task) to compute:

- Cohen's κ + Krippendorff's ordinal α (inter-rater agreement)
- Pearson r between mean human ratings and `val_score`
- Pearson r between mean human ratings and `mean_kill_rate`
- Per-dimension and per-method breakdown

---

## Rubric (sidebar in the app)

Each dimension is **scored 0–5**. Anchor descriptions appear directly
beneath the radio buttons in the app.

### Test idiom quality
| Score | Anchor |
|---|---|
| 0 | not pytest-style (no `test_*` functions, prints instead of asserts) |
| 1 | raw assertions at module level, no proper test functions |
| 2 | basic `test_*` functions but generic names; no parametrize / fixtures |
| 3 | descriptive `test_*` names; mostly one logical assertion per test |
| 4 | good structure + uses parametrize OR fixtures appropriately |
| 5 | production-grade: parametrize / fixtures used; helpful failure messages |

### Correctness
| Score | Anchor |
|---|---|
| 0 | most assertions wrong; references wrong API / would fail on a correct function |
| 1 | many wrong; only ~25% would pass on a correct function |
| 2 | mixed; about half would pass; some wrong exceptions or expected values |
| 3 | most assertions sound; ~75% pass on a correct function; a few wrong |
| 4 | all assertions sound; pass on a correct function; minor oracle nits |
| 5 | every oracle exact; assertions match function behaviour perfectly |

### Completeness
| Score | Anchor |
|---|---|
| 0 | single trivial happy-path test; no edge cases |
| 1 | only happy-path tests (multiple values but no edge cases) |
| 2 | happy path + 1 edge case (empty OR zero OR None) |
| 3 | happy path + 2-3 edge cases |
| 4 | happy path + multiple edge cases + at least one error/exception test |
| 5 | happy path + edge cases + error cases + boundary values (full coverage) |

---

## File map

| File | Purpose | Tracked in git? |
|---|---|---|
| `human_eval_app.py` | Streamlit UI | yes |
| `human_eval_pair_sampler.py` | Builds the worksheet from mutation checkpoints | yes |
| `human_eval_pairs.csv` | Blinded worksheet shown to annotators | yes |
| `human_eval_pairs.meta.csv` | Private sample_id → method/model mapping | **no** (gitignored) |
| `human_eval_annotations/` | Per-annotator response CSVs | **no** (gitignored) |
| `requirements.txt` | Minimal deps to run the app | yes |
| `README_human_eval.md` | This file | yes |

---

## Privacy / blinding

- Annotators see only `sample_id`, `function_code`, `generated_tests`,
  and optionally `ground_truth_tests`. They do NOT see which method
  (Plain LLM / Random RAG / Simple RAG / Iterative Critique) or which
  LLM (llama3.2 / phi4 / qwen3.5 / qwen3-coder) produced the tests.
- The mapping between `sample_id` and (method, model) lives in
  `human_eval_pairs.meta.csv` — kept off-version-control and used only
  during post-annotation analysis.
- Per-annotator CSVs in `human_eval_annotations/` are also
  `.gitignore`d so individual ratings never leave the researcher's
  machine.

---

## Troubleshooting

**App says "Worksheet not found"** — Run
`python3 human_eval_pair_sampler.py` from the repo root first.
The app expects `human_eval_pairs.csv` next to it.

**Port 8501 already in use** — Another Streamlit app is running.
Either stop it (`pkill -f 'streamlit run'`) or use a different port:
`streamlit run human_eval_app.py --server.port=8502`.

**I want to revise a sample I already rated** — On the main page,
tick "Show samples I've already rated (lets you revise)". The
dropdown will then include rated samples and let you change scores.

**I lost my progress / the browser tab crashed** — Don't panic.
Re-enter the same annotator ID and the app re-loads your saved
ratings from disk. Save-and-Next writes after every sample, so the
worst case is losing the in-flight sample.

---

## Citation

If you use this app or the released annotations in academic work,
please cite the parent study:

> Balaji Venktesh. *Unit Test Generation via RAG with Iterative
> Critique — Cross-Model Mutation-Testing Validation.*
> (under resubmission, EMSE 2026).
