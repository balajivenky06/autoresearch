"""
test_run.py — Trial run: 1 sample from HumanEval + 1 from MBPP.

Tests the full pipeline end-to-end before running the full experiment.
Runs all 3 methods (plain_llm, simple_rag, iterative_critique) on 2 samples.

Usage:
    python test_run.py
"""

import os, sys, time, traceback
import numpy as np

# Force 2-sample test subset — override before importing prepare_unitest
os.environ["UNITEST_TRIAL"] = "1"

# ---------------------------------------------------------------------------
# Patch NUM_EVAL_SAMPLES for trial run
# ---------------------------------------------------------------------------
import prepare_unitest as _pu
_pu.NUM_EVAL_SAMPLES = 2
_pu.DATASET_SEED     = 0   # different seed to get 1 HumanEval + 1 MBPP

from prepare_unitest import (
    make_eval_dataset, build_knowledge_base,
    evaluate_tests, compute_val_score, compute_faithfulness,
    VectorStore, NOISE_THRESHOLD,
)
from faithfulness import compute_faithfulness as cf_direct, faithfulness_summary

PASS = "  [PASS]"
FAIL = "  [FAIL]"

errors = []

def section(title):
    print(f"\n{'='*55}")
    print(f"  {title}")
    print('='*55)

def check(name, fn):
    try:
        result = fn()
        print(f"{PASS} {name}")
        return result
    except Exception as e:
        print(f"{FAIL} {name}")
        print(f"         {type(e).__name__}: {e}")
        traceback.print_exc()
        errors.append((name, e))
        return None

# ---------------------------------------------------------------------------
# Assert helpers — defined before any check() calls
# ---------------------------------------------------------------------------

def assert_nan(v):
    assert np.isnan(v), f"Expected NaN, got {v}"
    return v

def assert_range(v):
    assert 0.0 <= v <= 1.0, f"Expected [0,1], got {v}"
    return v

def assert_nonempty(v):
    assert v and len(v) > 0, "Expected non-empty string"
    return v

def assert_scores(result):
    ctx, nr = result
    assert isinstance(ctx, str), f"context should be str, got {type(ctx)}"
    assert 0.0 <= nr <= 1.0, f"noise_rate should be in [0,1], got {nr}"
    return result

def assert_equal(a, b):
    assert a == b, f"Expected {b!r}, got {a!r}"
    return a

def assert_metrics(m):
    expected = {"syntactic_validity","assertion_count","test_func_count",
                "edge_case_score","assert_density","rouge_1_f1","semantic_sim"}
    missing = expected - set(m.keys())
    assert not missing, f"Missing metric keys: {missing}"
    return m

def assert_positive(v):
    assert v > 0, f"Expected > 0, got {v}"
    return v

def assert_dataset(ds):
    assert len(ds) > 0, "Dataset is empty"
    required = {"task_id","source","function_code","ground_truth_tests"}
    for sample in ds:
        missing = required - set(sample.keys())
        assert not missing, f"Sample missing keys: {missing}"
        assert sample["function_code"].strip(), "function_code is empty"
        assert sample["ground_truth_tests"].strip(), "ground_truth_tests is empty"
    print(f"         Sources: {[s['source'] for s in ds]}")
    print(f"         Task IDs: {[s['task_id'] for s in ds]}")
    return ds

def assert_kb(result):
    kb, model = result
    assert isinstance(kb, VectorStore), f"Expected VectorStore, got {type(kb)}"
    print(f"         Loaded {len(kb.texts)} docs")
    if kb.texts:
        ctx, nr = kb.search_with_scores("pytest unit test assert raises", model, top_k=3)
        assert isinstance(ctx, str), "search context should be str"
        assert 0.0 <= nr <= 1.0, f"noise_rate out of range: {nr}"
        print(f"         Sample search noise_rate: {nr:.3f}")
    return result

# ===========================================================================
# 1. faithfulness.py
# ===========================================================================
section("1. faithfulness.py — unified metric")

check("NaN for empty context (plain_llm)",
      lambda: assert_nan(cf_direct("def test_foo(): assert foo() == 1", "")))

check("NaN for empty generated",
      lambda: assert_nan(cf_direct("", "some context")))

check("score in [0,1] for valid inputs",
      lambda: assert_range(cf_direct(
          "def test_add(): assert add(1,2)==3\ndef test_none(): assert add(None,1) raises TypeError",
          "pytest testing assert raises TypeError None edge cases"
      )))

check("stop-word filtering (all stop-words → NaN, not inflated score)",
      lambda: assert_nan(cf_direct("def return import class for if", "def return import class for if")))

check("faithfulness_summary on mixed list",
      lambda: faithfulness_summary([0.3, 0.5, float('nan'), 0.8]))

# ===========================================================================
# 2. VectorStore
# ===========================================================================
section("2. VectorStore — search_with_scores + noise rate")

embs = np.random.randn(5, 10).astype("float32")
embs /= np.linalg.norm(embs, axis=1, keepdims=True)
texts = [f"chunk {i}" for i in range(5)]

class FakeModel:
    def encode(self, queries):
        vec = np.random.randn(len(queries), 10).astype("float32")
        vec /= np.linalg.norm(vec, axis=1, keepdims=True)
        return vec

vs = VectorStore(texts, embs, texts)

check("search() returns non-empty string",
      lambda: assert_nonempty(vs.search("test query", FakeModel(), top_k=3)))

check("search_with_scores() returns (str, float)",
      lambda: assert_scores(vs.search_with_scores("test query", FakeModel(), top_k=3)))

check("search() on empty VectorStore returns ''",
      lambda: assert_equal(VectorStore([], np.array([]), []).search("q", FakeModel()), ""))

check("noise_rate in [0, 1]",
      lambda: assert_range(vs.search_with_scores("q", FakeModel(), top_k=3)[1]))

# ===========================================================================
# 3. evaluate_tests
# ===========================================================================
section("3. evaluate_tests — all metrics including semantic_sim")

SAMPLE_TESTS = """
import pytest

def test_add_positive():
    assert add(2, 3) == 5

def test_add_zero():
    assert add(0, 5) == 5

def test_add_none():
    with pytest.raises(TypeError):
        add(None, 1)
"""

SAMPLE_GT = """
def test_add(a, b):
    assert add(1, 2) == 3
    assert add(0, 0) == 0

def test_add_raises():
    with pytest.raises(TypeError):
        add("a", 1)
"""

SAMPLE_FN = "def add(a, b):\n    return a + b\n"

metrics = check("evaluate_tests returns 7 metrics",
    lambda: assert_metrics(evaluate_tests(SAMPLE_TESTS, SAMPLE_GT, SAMPLE_FN)))

check("syntactic_validity = 1.0 for valid code",
    lambda: assert_equal(evaluate_tests(SAMPLE_TESTS, SAMPLE_GT, SAMPLE_FN)["syntactic_validity"], 1.0))

check("syntactic_validity = 0.0 for invalid code",
    lambda: assert_equal(evaluate_tests("def (broken:", SAMPLE_GT, SAMPLE_FN)["syntactic_validity"], 0.0))

check("edge_case_score detects pytest.raises",
    lambda: assert_positive(evaluate_tests(SAMPLE_TESTS, SAMPLE_GT, SAMPLE_FN)["edge_case_score"]))

check("semantic_sim in [0,1]",
    lambda: assert_range(evaluate_tests(SAMPLE_TESTS, SAMPLE_GT, SAMPLE_FN)["semantic_sim"]))

check("compute_val_score with one sample",
    lambda: assert_range(compute_val_score([evaluate_tests(SAMPLE_TESTS, SAMPLE_GT, SAMPLE_FN)])))

check("compute_val_score empty list returns 0.0",
    lambda: assert_equal(compute_val_score([]), 0.0))

# ===========================================================================
# 4. Dataset loading (2 samples)
# ===========================================================================
section("4. Dataset loading — 2-sample trial subset")

dataset = check("make_eval_dataset() loads 2 samples",
    lambda: assert_dataset(make_eval_dataset(force_reload=True)))

# ===========================================================================
# 5. Knowledge base
# ===========================================================================
section("5. Knowledge base — build + search_with_scores")

kb_result = check("build_knowledge_base() builds VectorStore",
    lambda: assert_kb(build_knowledge_base(force_reload=True)))

# ===========================================================================
# 6. Ollama + full pipeline on 1 sample
# ===========================================================================
section("6. Full pipeline — 1 sample, all 3 methods (llama3.2:1b)")

if not dataset:
    print("  SKIPPED — dataset not loaded")
else:
    import train_unitest as tu
    tu.GENERATOR_MODEL = "llama3.2:1b"
    tu.HELPER_MODEL    = "llama3.2:1b"
    tu.TEMPERATURE     = 0.3

    sample = dataset[0]
    fn_code  = sample["function_code"]
    gt_tests = sample["ground_truth_tests"]
    print(f"\n  Function: {sample['task_id']} ({sample['source']})")
    print(f"  Code snippet: {fn_code[:80].strip()}...")

    for method_key, gen_fn in [
        ("plain_llm/base",          tu.generate_plain_base),
        ("simple_rag/base",         tu._simple_rag_base),
        ("iterative_critique/base", lambda c: tu._iterative_critique(tu.generate_plain_base(c), c)),
    ]:
        def run_method(fn=gen_fn, code=fn_code, gt=gt_tests, key=method_key):
            tu._reset_sample_diagnostics()
            t0 = time.time()
            tests = fn(code)
            elapsed = time.time() - t0
            assert isinstance(tests, str), f"generate_fn should return str, got {type(tests)}"
            m = evaluate_tests(tests, gt, code)
            m["noise_rate"]     = float(np.mean(tu._noise_rate_buf)) if tu._noise_rate_buf else float("nan")
            m["faithfulness"]   = compute_faithfulness(tests, tu._last_context)
            m["retrieval_secs"] = tu._retrieval_secs
            m["llm_secs"]       = tu._llm_secs
            print(f"\n         method:       {key}")
            print(f"         elapsed:      {elapsed:.1f}s")
            print(f"         syntax_valid: {m['syntactic_validity']:.0f}")
            print(f"         edge_score:   {m['edge_case_score']:.3f}")
            print(f"         val_score:    {compute_val_score([m]):.4f}")
            print(f"         noise_rate:   {m['noise_rate']:.3f}" if not np.isnan(m["noise_rate"]) else "         noise_rate:   N/A")
            print(f"         faithfulness: {m['faithfulness']:.3f}" if not np.isnan(m["faithfulness"]) else "         faithfulness: N/A")
            print(f"         llm_secs:     {m['llm_secs']:.2f}s")
            return m

        check(f"generate + evaluate: {method_key}", run_method)

# ===========================================================================
# Summary
# ===========================================================================
section("SUMMARY")

n_err = len(errors)

print(f"\n  Total checks run, errors: {n_err}")
if errors:
    print("\n  FAILED checks:")
    for name, exc in errors:
        print(f"    - {name}: {type(exc).__name__}: {exc}")
    sys.exit(1)
else:
    print("\n  All checks passed. Safe to run full experiment.")
    sys.exit(0)
