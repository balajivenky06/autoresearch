"""
train_unitest.py — the only file the agent edits for unit test generation experiments.
Mirrors train.py in the autoresearch repo.

The agent modifies: METHOD, REASONING, prompts, and RAG config below.
prepare_unitest.py is fixed — do NOT touch it.

Usage: python train_unitest.py
"""

import os
import pickle
import re
import time
from pathlib import Path

import numpy as np
import ollama

from prepare_unitest import (
    TIME_BUDGET, make_eval_dataset, build_knowledge_base,
    evaluate_tests, compute_val_score, compute_faithfulness,
)

# ---------------------------------------------------------------------------
# Hyperparameters — edit these directly (no CLI flags needed)
# ---------------------------------------------------------------------------

METHOD    = "plain_llm"
# Options: "plain_llm" | "simple_rag" | "iterative_critique"

REASONING = "base"
# Options: "base" | "cot" | "tot" | "got"

GENERATOR_MODEL = "llama3.2:latest"   # Ollama model for generation
HELPER_MODEL    = "llama3.2:latest"   # Ollama model for critique/rewrite

TEMPERATURE          = 0.5
CRITIQUE_TEMPERATURE = 0.0
REFINE_TEMPERATURE   = 0.3
TOP_K                = 3   # number of docs to retrieve (for RAG methods)

# Set to an integer (e.g. 3) for a quick local trial; None = use full dataset / time budget
MAX_SAMPLES          = None

# ---------------------------------------------------------------------------
# Prompts — edit these freely
# ---------------------------------------------------------------------------

SYSTEM_PROMPT = """You are an expert Python software engineer and QA specialist with deep knowledge of pytest and testing best practices.

Generate comprehensive pytest unit tests for the given Python function.
- Return ONLY valid Python pytest code (no markdown, no explanation)
- Each test function must start with 'test_'
- Cover: happy path, edge cases (None, empty, zero, negatives), error cases (exceptions)
- Use assert statements for all checks

Example of high-quality pytest tests:
```python
import pytest

def test_add_positive_numbers():
    assert add(2, 3) == 5

def test_add_zero():
    assert add(0, 5) == 5
    assert add(5, 0) == 5

def test_add_negative():
    assert add(-1, -2) == -3

def test_add_floats():
    assert add(1.5, 2.5) == pytest.approx(4.0)

def test_add_invalid_type_raises():
    with pytest.raises(TypeError):
        add("a", 1)

def test_add_none_raises():
    with pytest.raises(TypeError):
        add(None, 1)
```
"""

GENERATION_PROMPT = """Generate pytest unit tests for the following Python function.

Requirements:
- Return ONLY valid Python code
- Each test function starts with 'test_'
- Cover: at least 2 happy path cases, edge cases, and exception cases

Function:
```python
{function_code}
```

Output ONLY the test code:"""

CRITIQUE_PROMPT = """Review these pytest tests for the function below.

Function:
```python
{function_code}
```

Tests:
```python
{tests}
```

Reply with ONLY one of:
- HIGH QUALITY
- NEEDS IMPROVEMENT: <brief reason>"""

REFINE_PROMPT = """Improve these tests based on feedback: {feedback}

Function:
```python
{function_code}
```

Current tests:
```python
{tests}
```

Context from testing docs:
{context}

Return ONLY improved pytest code:"""

COT_PROMPT = """Think step by step, then write pytest tests.

Function:
```python
{function_code}
```

Step 1: What does this function do? What are inputs/outputs?
Step 2: List test scenarios: (a) normal cases (b) edge cases (c) error cases
Step 3: Write the tests.

Put final test code in a ```python``` block."""

TOT_DECOMPOSE_PROMPT = """Generate pytest tests focusing ONLY on happy path (normal inputs/outputs).

Function:
```python
{function_code}
```

Return ONLY test code (2-3 test functions):"""

TOT_EDGE_PROMPT = """Generate pytest tests focusing ONLY on edge cases (None, empty, zero, boundary).

Function:
```python
{function_code}
```

Return ONLY test code (2-3 test functions):"""

TOT_EVALUATE_PROMPT = """Select the BEST test suite from these candidates for the function below.

Function:
```python
{function_code}
```

Candidate A:
{candidate_a}

Candidate B:
{candidate_b}

Return ONLY the winning test code (no explanation):"""

GOT_AGGREGATE_PROMPT = """Merge these partial test suites into one clean pytest file.
Remove duplicates. Ensure all test function names are unique.

Happy path tests:
{happy_path}

Edge case tests:
{edge_cases}

Error handling tests:
{error_handling}

Return ONLY the merged test code:"""

# ---------------------------------------------------------------------------
# Generation helpers
# ---------------------------------------------------------------------------

_client = None

def _get_client():
    """Lazy Ollama client — safe to import even before Ollama server starts."""
    global _client
    if _client is None:
        _client = ollama.Client()
    return _client


# ---------------------------------------------------------------------------
# Per-sample diagnostic accumulators — reset before each sample in main loop
# ---------------------------------------------------------------------------

_noise_rate_buf: list  = []    # noise_rate per search_with_scores call this sample
_retrieval_secs: float = 0.0   # cumulative retrieval time this sample
_llm_secs:       float = 0.0   # cumulative LLM call time this sample
_tokens_used:    int   = 0     # cumulative token count this sample
_last_context:   str   = ""    # most recently retrieved context (for faithfulness)


def _reset_sample_diagnostics():
    global _noise_rate_buf, _retrieval_secs, _llm_secs, _tokens_used, _last_context
    _noise_rate_buf = []
    _retrieval_secs = 0.0
    _llm_secs       = 0.0
    _tokens_used    = 0
    _last_context   = ""


def _clean(raw: str) -> str:
    text = raw.strip()
    if text.startswith("```python"):
        text = text[len("```python"):].strip()
    elif text.startswith("```"):
        text = text[3:].strip()
    if text.endswith("```"):
        text = text[:-3].strip()
    return text


def _extract_code_block(text: str) -> str:
    blocks = re.findall(r"```python\s*(.*?)```", text, re.DOTALL)
    return blocks[-1].strip() if blocks else _clean(text)


def _call(model: str, messages: list, temperature: float) -> str:
    global _llm_secs, _tokens_used
    t0 = time.time()
    try:
        resp = _get_client().chat(model=model, messages=messages, options={"temperature": temperature})
        _llm_secs += time.time() - t0
        # Token counts: ollama v0.4+ exposes prompt_eval_count / eval_count
        try:
            _tokens_used += (resp.prompt_eval_count or 0) + (resp.eval_count or 0)
        except AttributeError:
            pass  # older dict-based response — skip token counting
        # ollama v0.4+ returns a ChatResponse object; older versions return a dict
        try:
            return resp.message.content.strip()
        except AttributeError:
            return resp.get("message", {}).get("content", "").strip()
    except Exception as e:
        _llm_secs += time.time() - t0
        print(f"  LLM error: {e}")
        return ""


# ---------------------------------------------------------------------------
# RAG retrieval
# ---------------------------------------------------------------------------

_kb = None
_emb_model = None


def _get_context(query: str) -> str:
    global _kb, _emb_model, _retrieval_secs, _noise_rate_buf, _last_context
    if _kb is None:
        _kb, _emb_model = build_knowledge_base()
    t0 = time.time()
    context_str, noise_rate = _kb.search_with_scores(query, _emb_model, top_k=TOP_K)
    _retrieval_secs += time.time() - t0
    if not np.isnan(noise_rate):
        _noise_rate_buf.append(noise_rate)
    _last_context = context_str
    return context_str


def _make_query(function_code: str) -> str:
    return f"pytest unit testing examples patterns for python function: {function_code[:300]}"


# ---------------------------------------------------------------------------
# Generation strategies
# ---------------------------------------------------------------------------

def generate_plain_base(function_code: str) -> str:
    return _clean(_call(GENERATOR_MODEL, [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": GENERATION_PROMPT.format(function_code=function_code)},
    ], TEMPERATURE))


def generate_plain_cot(function_code: str) -> str:
    raw = _call(GENERATOR_MODEL, [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": COT_PROMPT.format(function_code=function_code)},
    ], TEMPERATURE)
    return _extract_code_block(raw)


def generate_plain_tot(function_code: str) -> str:
    candidate_a = _clean(_call(GENERATOR_MODEL, [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": TOT_DECOMPOSE_PROMPT.format(function_code=function_code)},
    ], 0.7))
    candidate_b = _clean(_call(GENERATOR_MODEL, [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": GENERATION_PROMPT.format(function_code=function_code)},
    ], 0.3))
    if not candidate_a.strip():
        return candidate_b
    if not candidate_b.strip():
        return candidate_a
    best = _clean(_call(GENERATOR_MODEL, [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": TOT_EVALUATE_PROMPT.format(
            function_code=function_code, candidate_a=candidate_a, candidate_b=candidate_b)},
    ], 0.0))
    return best or candidate_a


def generate_plain_got(function_code: str) -> str:
    happy = _clean(_call(GENERATOR_MODEL, [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": TOT_DECOMPOSE_PROMPT.format(function_code=function_code)},
    ], TEMPERATURE))
    edges = _clean(_call(GENERATOR_MODEL, [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": TOT_EDGE_PROMPT.format(function_code=function_code)},
    ], TEMPERATURE))
    errors_prompt = f"Generate pytest tests for ERROR cases only (invalid inputs, exceptions).\n\nFunction:\n```python\n{function_code}\n```\n\nReturn ONLY test code:"
    errors = _clean(_call(GENERATOR_MODEL, [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": errors_prompt},
    ], TEMPERATURE))
    non_empty = {k: v for k, v in [("happy_path", happy), ("edge_cases", edges), ("error_handling", errors)] if v.strip()}
    if not non_empty:
        return "# ERROR: GoT generation failed"
    merged = _clean(_call(GENERATOR_MODEL, [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": GOT_AGGREGATE_PROMPT.format_map(
            {"happy_path": "", "edge_cases": "", "error_handling": "", **non_empty})},
    ], 0.2))
    return merged or "\n\n".join(non_empty.values())


def _simple_rag_base(function_code: str) -> str:
    """Simple RAG: retrieve context, then generate with base prompt."""
    context = _get_context(_make_query(function_code))
    return _clean(_call(GENERATOR_MODEL, [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": f"Relevant testing documentation:\n{context[:3000]}"},
        {"role": "user", "content": GENERATION_PROMPT.format(function_code=function_code)},
    ], TEMPERATURE))


def _simple_rag_cot(function_code: str) -> str:
    """Simple RAG + COT: retrieve context, then reason step-by-step."""
    context = _get_context(_make_query(function_code))
    raw = _call(GENERATOR_MODEL, [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": f"Relevant testing documentation:\n{context[:3000]}"},
        {"role": "user", "content": COT_PROMPT.format(function_code=function_code)},
    ], TEMPERATURE)
    return _extract_code_block(raw)


def _simple_rag_tot(function_code: str) -> str:
    """Simple RAG + TOT: retrieve context, generate two candidates, select best."""
    context = _get_context(_make_query(function_code))
    ctx_msg = {"role": "user", "content": f"Relevant testing documentation:\n{context[:3000]}"}
    candidate_a = _clean(_call(GENERATOR_MODEL, [
        {"role": "system", "content": SYSTEM_PROMPT}, ctx_msg,
        {"role": "user", "content": TOT_DECOMPOSE_PROMPT.format(function_code=function_code)},
    ], 0.7))
    candidate_b = _clean(_call(GENERATOR_MODEL, [
        {"role": "system", "content": SYSTEM_PROMPT}, ctx_msg,
        {"role": "user", "content": GENERATION_PROMPT.format(function_code=function_code)},
    ], 0.3))
    if not candidate_a.strip():
        return candidate_b
    if not candidate_b.strip():
        return candidate_a
    return _clean(_call(GENERATOR_MODEL, [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": TOT_EVALUATE_PROMPT.format(
            function_code=function_code, candidate_a=candidate_a, candidate_b=candidate_b)},
    ], 0.0)) or candidate_a


def _simple_rag_got(function_code: str) -> str:
    """Simple RAG + GOT: retrieve context, generate per-axis, aggregate."""
    context = _get_context(_make_query(function_code))
    ctx_msg = {"role": "user", "content": f"Relevant testing documentation:\n{context[:3000]}"}
    happy = _clean(_call(GENERATOR_MODEL, [
        {"role": "system", "content": SYSTEM_PROMPT}, ctx_msg,
        {"role": "user", "content": TOT_DECOMPOSE_PROMPT.format(function_code=function_code)},
    ], TEMPERATURE))
    edges = _clean(_call(GENERATOR_MODEL, [
        {"role": "system", "content": SYSTEM_PROMPT}, ctx_msg,
        {"role": "user", "content": TOT_EDGE_PROMPT.format(function_code=function_code)},
    ], TEMPERATURE))
    errors = _clean(_call(GENERATOR_MODEL, [
        {"role": "system", "content": SYSTEM_PROMPT}, ctx_msg,
        {"role": "user", "content": f"Generate pytest tests for ERROR cases only (invalid inputs, exceptions).\n\nFunction:\n```python\n{function_code}\n```\n\nReturn ONLY test code:"},
    ], TEMPERATURE))
    non_empty = {k: v for k, v in [("happy_path", happy), ("edge_cases", edges), ("error_handling", errors)] if v.strip()}
    if not non_empty:
        return "# ERROR: GoT+RAG generation failed"
    return _clean(_call(GENERATOR_MODEL, [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": GOT_AGGREGATE_PROMPT.format_map(
            {"happy_path": "", "edge_cases": "", "error_handling": "", **non_empty})},
    ], 0.2)) or "\n\n".join(non_empty.values())


def _iterative_critique(initial_tests: str, function_code: str) -> str:
    """Critique-refine loop: up to 2 rounds. RAG context only fetched when critique fails."""
    tests = initial_tests
    context = None  # lazy: only fetched if critique fails

    for _ in range(2):
        verdict = _call(HELPER_MODEL, [
            {"role": "system", "content": "You are a strict QA reviewer."},
            {"role": "user", "content": CRITIQUE_PROMPT.format(
                function_code=function_code[:1500], tests=tests[:2000])},
        ], CRITIQUE_TEMPERATURE)

        if verdict.upper().startswith("HIGH QUALITY"):
            return tests

        if context is None:
            context = _get_context(_make_query(function_code))

        refined = _clean(_call(GENERATOR_MODEL, [
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": REFINE_PROMPT.format(
                feedback=verdict, function_code=function_code[:1500],
                tests=tests[:2000], context=context[:3000])},
        ], REFINE_TEMPERATURE))

        if refined:
            tests = refined

    return tests


# ---------------------------------------------------------------------------
# Dispatch table
# ---------------------------------------------------------------------------

GENERATORS = {
    ("plain_llm",          "base"): generate_plain_base,
    ("plain_llm",          "cot"):  generate_plain_cot,
    ("plain_llm",          "tot"):  generate_plain_tot,
    ("plain_llm",          "got"):  generate_plain_got,
    ("simple_rag",         "base"): _simple_rag_base,
    ("simple_rag",         "cot"):  _simple_rag_cot,
    ("simple_rag",         "tot"):  _simple_rag_tot,
    ("simple_rag",         "got"):  _simple_rag_got,
    ("iterative_critique", "base"): lambda c: _iterative_critique(generate_plain_base(c), c),
    ("iterative_critique", "cot"):  lambda c: _iterative_critique(generate_plain_cot(c), c),
    ("iterative_critique", "tot"):  lambda c: _iterative_critique(generate_plain_tot(c), c),
    ("iterative_critique", "got"):  lambda c: _iterative_critique(generate_plain_got(c), c),
}

# ---------------------------------------------------------------------------
# Checkpoint helpers — resume from last completed sample on Colab restarts
# ---------------------------------------------------------------------------

# Checkpoint directory: Google Drive on Colab, local otherwise.
# On Colab, mount Drive first:  from google.colab import drive; drive.mount('/content/drive')
_IN_COLAB = os.path.exists("/content")
_CKPT_DIR = (
    Path("/content/drive/MyDrive/autoresearch_checkpoints") if _IN_COLAB
    else Path(".checkpoints")
)


def _ckpt_path() -> Path:
    safe_model = GENERATOR_MODEL.replace(":", "-").replace("/", "-")
    return _CKPT_DIR / f"{METHOD}_{REASONING}_{safe_model}.pkl"


def _save_checkpoint(metrics_list: list, step: int) -> None:
    _CKPT_DIR.mkdir(parents=True, exist_ok=True)
    with open(_ckpt_path(), "wb") as f:
        pickle.dump({"metrics_list": metrics_list, "step": step,
                     "method": METHOD, "reasoning": REASONING,
                     "model": GENERATOR_MODEL}, f)


def _load_checkpoint() -> tuple:
    """Return (metrics_list, step) from checkpoint, or ([], 0) if none/mismatch."""
    p = _ckpt_path()
    if not p.exists():
        return [], 0
    try:
        with open(p, "rb") as f:
            data = pickle.load(f)
        if (data.get("method") == METHOD and
                data.get("reasoning") == REASONING and
                data.get("model") == GENERATOR_MODEL):
            print(f"Checkpoint found: resuming from sample {data['step']} / already done.")
            return data["metrics_list"], data["step"]
    except Exception as e:
        print(f"Checkpoint load failed ({e}), starting fresh.")
    return [], 0


def _clear_checkpoint() -> None:
    p = _ckpt_path()
    if p.exists():
        p.unlink()


# ---------------------------------------------------------------------------
# Main experiment loop
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    t_start = time.time()

    dataset = make_eval_dataset()
    print(f"Eval dataset: {len(dataset)} samples")
    print(f"Method: {METHOD} | Reasoning: {REASONING} | Model: {GENERATOR_MODEL}")
    print(f"Time budget: {TIME_BUDGET}s")

    generate_fn = GENERATORS.get((METHOD, REASONING))
    if generate_fn is None:
        raise ValueError(f"Unknown METHOD/REASONING combo: {METHOD}/{REASONING}")

    # Resume from checkpoint if available (handles Colab disconnects)
    metrics_list, start_step = _load_checkpoint()
    total_generation_time = 0.0
    step = start_step

    if start_step > 0:
        print(f"Resuming from step {start_step}/{len(dataset)} — {start_step} samples already evaluated.")

    for i, sample in enumerate(dataset):
        if i < start_step:
            continue  # already processed in a previous run

        _reset_sample_diagnostics()
        t0 = time.time()
        fn_code = sample["function_code"]
        gt_tests = sample["ground_truth_tests"]

        tests = generate_fn(fn_code)
        dt = time.time() - t0

        if step > 2:
            total_generation_time += dt

        metrics = evaluate_tests(tests, gt_tests, fn_code)

        # Diagnostic metrics (not in val_score — used for RQ2/RQ3/RQ4 analysis)
        metrics["noise_rate"]     = float(np.mean(_noise_rate_buf)) if _noise_rate_buf else float("nan")
        metrics["faithfulness"]   = compute_faithfulness(tests, _last_context)
        metrics["retrieval_secs"] = _retrieval_secs
        metrics["llm_secs"]       = _llm_secs
        metrics["tokens_used"]    = float(_tokens_used)

        metrics_list.append(metrics)
        step += 1

        # Save after every sample — safe to interrupt at any point
        _save_checkpoint(metrics_list, step)

        val_so_far = compute_val_score(metrics_list)
        noise_str = f"{metrics['noise_rate']:.2f}" if not np.isnan(metrics["noise_rate"]) else "N/A"
        print(f"\rstep {step:03d}/{len(dataset)} | val_score: {val_so_far:.6f} | dt: {dt:.1f}s | syntax: {metrics['syntactic_validity']:.0f} | edges: {metrics['edge_case_score']:.2f} | noise: {noise_str}    ", end="", flush=True)

        if MAX_SAMPLES is not None and step >= MAX_SAMPLES:
            print(f"\nMAX_SAMPLES limit ({MAX_SAMPLES}) reached.")
            break

        if step > 2 and total_generation_time >= TIME_BUDGET:
            print(f"\nTime budget reached after {step} samples.")
            break

    print()  # newline

    # -----------------------------------------------------------------------
    # Final summary
    # -----------------------------------------------------------------------

    val_score = compute_val_score(metrics_list)
    total_time = time.time() - t_start

    if not metrics_list:
        print("---")
        print("val_score:          0.000000")
        print(f"method:             {METHOD}/{REASONING}")
        print(f"model:              {GENERATOR_MODEL}")
        print("samples_evaluated:  0")
        print(f"total_seconds:      {total_time:.1f}")
        raise SystemExit("No samples evaluated — dataset may be empty or corrupted.")

    avg_syntax        = sum(m["syntactic_validity"] for m in metrics_list) / len(metrics_list)
    avg_edges         = sum(m["edge_case_score"] for m in metrics_list) / len(metrics_list)
    avg_asserts       = sum(m["assertion_count"] for m in metrics_list) / len(metrics_list)
    avg_assert_density= sum(m["assert_density"] for m in metrics_list) / len(metrics_list)
    avg_rouge         = sum(m["rouge_1_f1"] for m in metrics_list) / len(metrics_list)
    avg_semsim        = sum(m.get("semantic_sim", 0.0) for m in metrics_list) / len(metrics_list)

    # Diagnostic averages (NaN-safe — NaN means metric not applicable for this method)
    _noise_vals = [m["noise_rate"]   for m in metrics_list if not np.isnan(m.get("noise_rate",   float("nan")))]
    _faith_vals = [m["faithfulness"] for m in metrics_list if not np.isnan(m.get("faithfulness", float("nan")))]
    avg_noise_rate     = float(np.mean(_noise_vals)) if _noise_vals else float("nan")
    avg_faithfulness   = float(np.mean(_faith_vals)) if _faith_vals else float("nan")
    avg_retrieval_secs = sum(m.get("retrieval_secs", 0.0) for m in metrics_list) / len(metrics_list)
    avg_llm_secs       = sum(m.get("llm_secs",       0.0) for m in metrics_list) / len(metrics_list)
    avg_tokens         = sum(m.get("tokens_used",    0.0) for m in metrics_list) / len(metrics_list)

    # Experiment complete — remove checkpoint so next run starts fresh
    _clear_checkpoint()

    print("---")
    print(f"val_score:          {val_score:.6f}")
    print(f"method:             {METHOD}/{REASONING}")
    print(f"model:              {GENERATOR_MODEL}")
    print(f"samples_evaluated:  {step}")
    print(f"total_seconds:      {total_time:.1f}")
    print(f"avg_syntax:         {avg_syntax:.4f}")
    print(f"avg_edge:           {avg_edges:.4f}")
    print(f"avg_assert_density: {avg_assert_density:.4f}")
    print(f"avg_assertions:     {avg_asserts:.1f}")
    print(f"avg_semantic_sim:   {avg_semsim:.4f}")
    print(f"avg_rouge:          {avg_rouge:.4f}")
    print(f"avg_noise_rate:     {avg_noise_rate:.4f}" if not np.isnan(avg_noise_rate) else "avg_noise_rate:     nan")
    print(f"avg_faithfulness:   {avg_faithfulness:.4f}" if not np.isnan(avg_faithfulness) else "avg_faithfulness:   nan")
    print(f"avg_retrieval_secs: {avg_retrieval_secs:.3f}")
    print(f"avg_llm_secs:       {avg_llm_secs:.3f}")
    print(f"avg_tokens:         {avg_tokens:.1f}")

    # -----------------------------------------------------------------------
    # Write / append one row to results_unitest.tsv
    # -----------------------------------------------------------------------
    import csv
    tsv_path = "results_unitest.tsv"
    _nan_str = lambda v: f"{v:.4f}" if not np.isnan(v) else "nan"
    result_row = {
        "method":             f"{METHOD}/{REASONING}",
        "model":              GENERATOR_MODEL,
        "status":             "ok",
        "val_score":          f"{val_score:.6f}",
        "avg_syntax":         f"{avg_syntax:.4f}",
        "avg_edge":           f"{avg_edges:.4f}",
        "avg_assert_density": f"{avg_assert_density:.4f}",
        "avg_semantic_sim":   f"{avg_semsim:.4f}",
        "avg_rouge":          f"{avg_rouge:.4f}",
        "avg_noise_rate":     _nan_str(avg_noise_rate),
        "avg_faithfulness":   _nan_str(avg_faithfulness),
        "avg_retrieval_secs": f"{avg_retrieval_secs:.3f}",
        "avg_llm_secs":       f"{avg_llm_secs:.3f}",
        "avg_tokens":         f"{avg_tokens:.1f}",
        "samples_evaluated":  str(step),
    }
    _write_header = not os.path.exists(tsv_path) or os.path.getsize(tsv_path) == 0
    with open(tsv_path, "a", newline="") as _f:
        _writer = csv.DictWriter(_f, fieldnames=list(result_row.keys()), delimiter="\t")
        if _write_header:
            _writer.writeheader()
        _writer.writerow(result_row)
    print(f"Results appended → {tsv_path}")
