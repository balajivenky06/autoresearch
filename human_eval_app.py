"""
human_eval_app.py — Streamlit UI for the human annotation task.

Each annotator opens the app at http://localhost:8501, logs in with their
initials, and walks through the 40-pair worksheet from
human_eval_pair_sampler.py. Method and model are blinded — the annotator
only sees (function_code, generated_tests). Ratings (0-3 on four
dimensions) plus optional notes are saved after every Save & Next click
to human_eval_annotations/{annotator_id}.csv so the app is fully
resumable.

Run:
  pip install streamlit                      # one-time
  python3 human_eval_pair_sampler.py         # one-time, builds the CSV
  streamlit run human_eval_app.py            # opens browser at :8501

Share the URL only after `streamlit run --server.address=0.0.0.0 ...` if
you want remote annotators on the same network. Otherwise each annotator
runs the app locally on their own machine after cloning the repo.
"""

from __future__ import annotations

from datetime import datetime
from pathlib import Path

import pandas as pd
import streamlit as st

WORKSHEET_PATH    = Path("human_eval_pairs.csv")
ANNOTATIONS_DIR   = Path("human_eval_annotations")
GUIDE_PATH        = Path("human_eval_guide.txt")  # produced by human_eval_sampler.py

DIMENSIONS = [
    ("faithfulness",
     "Faithfulness",
     "Do the tests use patterns / vocabulary from the testing docs that "
     "would have been retrieved (e.g. pytest.parametrize, fixtures)? "
     "Score 0 for non-RAG-style tests. Skip if you have no retrieval context."),
    ("correctness",
     "Correctness",
     "Are the assertions sound? If the function under test is correct, "
     "would every test in the cell pass? Penalise tests that reference the "
     "wrong API, expect the wrong exception, or hallucinate function behaviour."),
    ("completeness",
     "Completeness",
     "Does the test suite cover happy path + edge cases (None / empty / zero) "
     "+ error cases (invalid input)? A single trivial happy-path test → 0; "
     "happy + 1-2 edges → 1-2; full coverage → 3."),
    ("overall",
     "Overall quality",
     "Would you actually use this test suite as a starting point for the "
     "function under test on a real codebase? 0 = useless, 3 = ship as-is."),
]
SCALE = [0, 1, 2, 3]


# ---------------------------------------------------------------------------
# Persistence
# ---------------------------------------------------------------------------

def annotations_path(annotator_id: str) -> Path:
    ANNOTATIONS_DIR.mkdir(exist_ok=True)
    safe = "".join(c for c in annotator_id if c.isalnum() or c in "_-")
    return ANNOTATIONS_DIR / f"{safe or 'anon'}.csv"


def load_annotations(annotator_id: str) -> pd.DataFrame:
    p = annotations_path(annotator_id)
    if p.exists():
        return pd.read_csv(p)
    return pd.DataFrame(columns=[
        "sample_id", "annotator_id", "timestamp",
        "human_faithfulness", "human_correctness",
        "human_completeness", "human_overall",
        "annotator_notes",
    ])


def upsert_annotation(annotator_id: str, sample_id: str,
                      ratings: dict, notes: str) -> None:
    df = load_annotations(annotator_id)
    new_row = {
        "sample_id":           sample_id,
        "annotator_id":        annotator_id,
        "timestamp":           datetime.now().isoformat(timespec="seconds"),
        "human_faithfulness":  ratings["faithfulness"],
        "human_correctness":   ratings["correctness"],
        "human_completeness":  ratings["completeness"],
        "human_overall":       ratings["overall"],
        "annotator_notes":     notes,
    }
    df = df[df["sample_id"] != sample_id]   # remove prior version if any
    df = pd.concat([df, pd.DataFrame([new_row])], ignore_index=True)
    df.to_csv(annotations_path(annotator_id), index=False)


# ---------------------------------------------------------------------------
# Page
# ---------------------------------------------------------------------------

st.set_page_config(page_title="Test Quality Annotation",
                   page_icon="🧪", layout="wide")

# --- Sidebar: rubric + progress -------------------------------------------
with st.sidebar:
    st.header("Rubric (0–3 each)")
    for _, label, desc in DIMENSIONS:
        with st.expander(label, expanded=False):
            st.write(desc)
    st.markdown("**Scale**")
    st.markdown(
        "- **0** — absent / wrong / unusable\n"
        "- **1** — partial / weak\n"
        "- **2** — solid / mostly right\n"
        "- **3** — excellent / production-ready"
    )
    st.markdown("---")
    st.markdown("**Tips**\n"
                "- You don't see which method / model produced these tests — "
                "rate purely on the code in front of you.\n"
                "- Use the Notes box to flag tricky cases or anything "
                "interesting. We'll read all of them.\n"
                "- Save & Next persists immediately — close the tab and resume "
                "any time with the same ID.")

# --- Worksheet + annotator login ------------------------------------------
st.title("🧪 Unit-Test Quality — Human Evaluation")

if not WORKSHEET_PATH.exists():
    st.error(f"Worksheet not found at `{WORKSHEET_PATH}`. "
             "Run `python3 human_eval_pair_sampler.py` first to build it.")
    st.stop()

@st.cache_data
def load_worksheet():
    return pd.read_csv(WORKSHEET_PATH)

worksheet = load_worksheet()
total = len(worksheet)

st.markdown(
    "You'll rate **{} samples** of LLM-generated unit tests. "
    "For each sample, read the function, read the generated tests, then "
    "score them on the four dimensions in the sidebar.".format(total)
)

if "annotator_id" not in st.session_state:
    st.session_state.annotator_id = ""

annotator_id = st.text_input(
    "Your initials or annotator ID  (used to save your work — pick something "
    "you'll remember, e.g. 'bv' or 'jane.r')",
    value=st.session_state.annotator_id,
    placeholder="e.g. bv",
).strip()

if not annotator_id:
    st.info("Enter an annotator ID above to start.")
    st.stop()

st.session_state.annotator_id = annotator_id
annotations = load_annotations(annotator_id)
done_ids = set(annotations["sample_id"].astype(str)) if not annotations.empty else set()
done = sum(1 for sid in worksheet["sample_id"].astype(str) if sid in done_ids)

st.progress(done / total if total else 0.0,
            text=f"{done} / {total} annotated by **{annotator_id}**")

if done == total:
    st.success("🎉 All samples annotated. Download your annotations below "
               "and share the CSV with whoever runs the analysis.")
    st.download_button(
        label="Download my annotations CSV",
        data=annotations.to_csv(index=False).encode(),
        file_name=f"{annotator_id}.csv",
        mime="text/csv",
    )
    with st.expander("Preview"):
        st.dataframe(annotations)
    st.stop()

# --- Pick current sample --------------------------------------------------
nav_col1, nav_col2 = st.columns([3, 1])
with nav_col1:
    show_done = st.checkbox(
        "Show samples I've already rated (lets you revise)", value=False)
with nav_col2:
    only_unrated = worksheet[~worksheet["sample_id"].astype(str).isin(done_ids)] \
        if not show_done else worksheet
    if only_unrated.empty:
        only_unrated = worksheet
    current_id = st.selectbox(
        "Sample",
        options=only_unrated["sample_id"].astype(str).tolist(),
        index=0,
    )

current = worksheet[worksheet["sample_id"].astype(str) == current_id].iloc[0]

# --- Display the sample ---------------------------------------------------
st.markdown(f"### Sample `{current_id}`  ({done} of {total} done)")

col_fn, col_test = st.columns(2)
with col_fn:
    st.markdown("**Function under test**")
    st.code(current["function_code"], language="python", line_numbers=True)
with col_test:
    st.markdown("**Generated tests**")
    st.code(current["generated_tests"], language="python", line_numbers=True)

if isinstance(current.get("ground_truth_tests"), str) \
        and current["ground_truth_tests"].strip():
    with st.expander("Reference / ground-truth tests (HIDE if you want to "
                     "rate blind first)"):
        st.code(current["ground_truth_tests"], language="python")

# --- Rating widgets -------------------------------------------------------
st.markdown("### Your ratings")

# Pre-fill from any prior annotation on this sample (so revisiting just edits)
prior = annotations[annotations["sample_id"] == current_id]
prior_row = prior.iloc[0] if not prior.empty else None
def _prior_val(col: str, default: int = 0) -> int:
    if prior_row is None:
        return default
    v = prior_row[col]
    if pd.isna(v):
        return default
    return int(v)

ratings = {}
cols = st.columns(4)
for (key, label, _desc), col in zip(DIMENSIONS, cols):
    with col:
        ratings[key] = st.radio(
            label,
            SCALE,
            index=_prior_val(f"human_{key}", 0),
            horizontal=True,
            key=f"rating_{current_id}_{key}",
        )

notes = st.text_area(
    "Notes (optional)",
    value=("" if prior_row is None or pd.isna(prior_row["annotator_notes"])
           else str(prior_row["annotator_notes"])),
    key=f"notes_{current_id}",
    placeholder="What did you notice? Missing edge cases? Spurious assertions?",
)

save_clicked = st.button("Save & next →", type="primary",
                          use_container_width=True)

if save_clicked:
    upsert_annotation(annotator_id, current_id, ratings, notes)
    st.success(f"Saved `{current_id}`. Loading next sample...")
    st.rerun()
