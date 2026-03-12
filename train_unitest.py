"""
train_unitest.py — the only file the agent edits for unit test generation experiments.
Mirrors train.py in the autoresearch repo.

The agent modifies: METHOD, REASONING, prompts, and RAG config below.
prepare_unitest.py is fixed — do NOT touch it.

Usage: python train_unitest.py
"""

import os
import re
import time

import ollama

from prepare_unitest import (
    TIME_BUDGET, make_eval_dataset, build_knowledge_base,
    evaluate_tests, compute_val_score,
)

# ---------------------------------------------------------------------------
# Hyperparameters — edit these directly (no CLI flags needed)
# ---------------------------------------------------------------------------

METHOD    = "iterative_critique"
# Options: "plain_llm" | "simple_rag" | "iterative_critique"

REASONING = "base"
# Options: "base" | "cot" | "tot" | "got"

GENERATOR_MODEL = "llama3.2:latest"   # Ollama model for generation
HELPER_MODEL    = "llama3.2:latest"   # Ollama model for critique/rewrite

TEMPERATURE          = 0.5
CRITIQUE_TEMPERATURE = 0.0
REFINE_TEMPERATURE   = 0.3
TOP_K                = 3   # number of docs to retrieve (for RAG methods)

# ---------------------------------------------------------------------------
# Prompts — edit these freely
# ---------------------------------------------------------------------------

SYSTEM_PROMPT = """You are an expert Python software engineer and QA specialist with deep knowledge of pytest and testing best practices.

Generate comprehensive pytest unit tests for the given Python function.
- Return ONLY valid Python pytest code (no markdown, no explanation)
- Each test function must start with 'test_'
- Cover: happy path, edge cases (None, empty, zero, negatives), error cases (exceptions)
- Use assert statements for all checks
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
    try:
        resp = _get_client().chat(model=model, messages=messages, options={"temperature": temperature})
        return resp.get("message", {}).get("content", "").strip()
    except Exception as e:
        print(f"  LLM error: {e}")
        return ""


# ---------------------------------------------------------------------------
# RAG retrieval
# ---------------------------------------------------------------------------

_kb = None
_emb_model = None


def _get_context(query: str) -> str:
    global _kb, _emb_model
    if _kb is None:
        _kb, _emb_model = build_knowledge_base()
    return _kb.search(query, _emb_model, top_k=TOP_K)


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
        {"role": "user", "content": GOT_AGGREGATE_PROMPT.format(**{k: non_empty.get(k, "") for k in ["happy_path", "edge_cases", "error_handling"]})},
    ], 0.2))
    return merged or "\n\n".join(non_empty.values())


def _with_rag(base_fn, function_code: str) -> str:
    """Wrap any base generation function with RAG context injection."""
    context = _get_context(_make_query(function_code))

    # Inject context as an extra user message before generation
    messages = [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": f"Relevant testing documentation:\n{context[:3000]}"},
        {"role": "user", "content": GENERATION_PROMPT.format(function_code=function_code)},
    ]
    # For non-base reasoning, re-use the base prompt but with context prepended
    return _clean(_call(GENERATOR_MODEL, messages, TEMPERATURE))


def _iterative_critique(initial_tests: str, function_code: str) -> str:
    """Critique tests, retrieve context if needed, refine."""
    verdict = _call(HELPER_MODEL, [
        {"role": "system", "content": "You are a strict QA reviewer."},
        {"role": "user", "content": CRITIQUE_PROMPT.format(
            function_code=function_code[:1500], tests=initial_tests[:2000])},
    ], CRITIQUE_TEMPERATURE)

    if verdict.upper().startswith("HIGH QUALITY"):
        return initial_tests

    context = _get_context(_make_query(function_code))
    refined = _clean(_call(GENERATOR_MODEL, [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": REFINE_PROMPT.format(
            feedback=verdict, function_code=function_code[:1500],
            tests=initial_tests[:2000], context=context[:3000])},
    ], REFINE_TEMPERATURE))
    return refined or initial_tests


# ---------------------------------------------------------------------------
# Dispatch table
# ---------------------------------------------------------------------------

GENERATORS = {
    ("plain_llm",         "base"): generate_plain_base,
    ("plain_llm",         "cot"):  generate_plain_cot,
    ("plain_llm",         "tot"):  generate_plain_tot,
    ("plain_llm",         "got"):  generate_plain_got,
    ("simple_rag",        "base"): lambda c: _with_rag(generate_plain_base, c),
    ("simple_rag",        "cot"):  lambda c: _with_rag(generate_plain_cot, c),
    ("simple_rag",        "tot"):  lambda c: _with_rag(generate_plain_tot, c),
    ("simple_rag",        "got"):  lambda c: _with_rag(generate_plain_got, c),
    ("iterative_critique","base"): lambda c: _iterative_critique(generate_plain_base(c), c),
    ("iterative_critique","cot"):  lambda c: _iterative_critique(generate_plain_cot(c), c),
    ("iterative_critique","tot"):  lambda c: _iterative_critique(generate_plain_tot(c), c),
    ("iterative_critique","got"):  lambda c: _iterative_critique(generate_plain_got(c), c),
}

# ---------------------------------------------------------------------------
# Main experiment loop
# ---------------------------------------------------------------------------

t_start = time.time()

dataset = make_eval_dataset()
print(f"Eval dataset: {len(dataset)} samples")
print(f"Method: {METHOD} | Reasoning: {REASONING} | Model: {GENERATOR_MODEL}")
print(f"Time budget: {TIME_BUDGET}s")

generate_fn = GENERATORS.get((METHOD, REASONING))
if generate_fn is None:
    raise ValueError(f"Unknown METHOD/REASONING combo: {METHOD}/{REASONING}")

metrics_list = []
total_generation_time = 0.0
step = 0

for sample in dataset:
    t0 = time.time()
    fn_code = sample["function_code"]
    gt_tests = sample["ground_truth_tests"]

    tests = generate_fn(fn_code)
    dt = time.time() - t0

    if step > 2:
        total_generation_time += dt

    metrics = evaluate_tests(tests, gt_tests, fn_code)
    metrics_list.append(metrics)
    step += 1

    val_so_far = compute_val_score(metrics_list)
    print(f"\rstep {step:03d}/{len(dataset)} | val_score: {val_so_far:.6f} | dt: {dt:.1f}s | syntax: {metrics['syntactic_validity']:.0f} | edges: {metrics['edge_case_score']:.2f}    ", end="", flush=True)

    if step > 2 and total_generation_time >= TIME_BUDGET:
        print(f"\nTime budget reached after {step} samples.")
        break

print()  # newline

# ---------------------------------------------------------------------------
# Final summary
# ---------------------------------------------------------------------------

val_score = compute_val_score(metrics_list)
total_time = time.time() - t_start
avg_syntax   = sum(m["syntactic_validity"] for m in metrics_list) / len(metrics_list)
avg_edges    = sum(m["edge_case_score"] for m in metrics_list) / len(metrics_list)
avg_asserts  = sum(m["assertion_count"] for m in metrics_list) / len(metrics_list)
avg_rouge    = sum(m["rouge_1_f1"] for m in metrics_list) / len(metrics_list)

print("---")
print(f"val_score:          {val_score:.6f}")
print(f"method:             {METHOD}/{REASONING}")
print(f"model:              {GENERATOR_MODEL}")
print(f"samples_evaluated:  {step}")
print(f"total_seconds:      {total_time:.1f}")
print(f"avg_syntax_valid:   {avg_syntax:.4f}")
print(f"avg_edge_coverage:  {avg_edges:.4f}")
print(f"avg_assertions:     {avg_asserts:.1f}")
print(f"avg_rouge_1:        {avg_rouge:.4f}")
