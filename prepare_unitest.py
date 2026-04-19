"""
prepare_unitest.py — fixed constants, dataset prep, and evaluation for unit test generation.
DO NOT MODIFY. This is the fixed harness; train_unitest.py is the only file the agent edits.

Mirrors prepare.py in the autoresearch repo but for unit test generation via LLM/RAG.

Usage (one-time setup):
    python prepare_unitest.py
"""

import os
import sys
import ast
import re
import time
import pickle
import subprocess
import tempfile
import numpy as np
from pathlib import Path

# ---------------------------------------------------------------------------
# Fixed constants — do not change
# ---------------------------------------------------------------------------

TIME_BUDGET      = 600   # seconds of generation time per experiment (10 min)
NUM_EVAL_SAMPLES = 100   # fixed eval subset size — 100 samples meets journal-quality bar
DATASET_SEED     = 42    # seed for reproducible subset selection
CACHE_VERSION    = "v3"  # bump to bust stale caches after config changes

NOISE_THRESHOLD  = 0.3   # cosine similarity below this → chunk is "noisy" (diagnostic only)

# Knowledge base chunking config
KB_CHUNK_SIZE    = 500   # characters per chunk
KB_CHUNK_OVERLAP = 100   # overlapping characters between consecutive chunks

# Cache dir: /content/.cache in Colab, ~/.cache elsewhere
_IN_COLAB = os.path.exists("/content")
CACHE_DIR     = Path("/content/.cache/autoresearch_unitest") if _IN_COLAB else Path.home() / ".cache" / "autoresearch_unitest"
DATASET_CACHE = CACHE_DIR / f"eval_dataset_{CACHE_VERSION}.pkl"
KB_CACHE      = CACHE_DIR / f"knowledge_base_{CACHE_VERSION}.pkl"

# Knowledge base URLs — testing documentation for RAG retrieval
# Each page is chunked into KB_CHUNK_SIZE-char overlapping windows (not truncated to 4000 chars)
KNOWLEDGE_BASE_URLS = [
    # Core pytest docs
    "https://docs.pytest.org/en/stable/how-to/assert.html",
    "https://docs.pytest.org/en/stable/how-to/parametrize.html",
    "https://docs.pytest.org/en/stable/getting-started.html",
    "https://docs.pytest.org/en/stable/how-to/fixtures.html",
    "https://docs.pytest.org/en/stable/how-to/monkeypatch.html",
    "https://docs.pytest.org/en/stable/how-to/capture-output.html",
    "https://docs.pytest.org/en/stable/how-to/tmp_path.html",
    # Python standard library
    "https://docs.python.org/3/library/unittest.html",
    "https://docs.python.org/3/library/unittest.mock.html",
    # Tutorials and patterns
    "https://realpython.com/pytest-python-testing/",
    "https://realpython.com/python-mock-library/",
    "https://www.geeksforgeeks.org/unit-testing-python-unittest/",
    "https://semaphoreci.com/community/tutorials/testing-python-applications-with-pytest",
    # Edge case and property-based testing
    "https://hypothesis.readthedocs.io/en/latest/quickstart.html",
]

# ---------------------------------------------------------------------------
# Dataset loading
# ---------------------------------------------------------------------------

def _load_humaneval():
    from datasets import load_dataset
    ds = load_dataset("openai/openai_humaneval", split="test")
    rows = []
    for item in ds:
        full_code = item["prompt"] + item["canonical_solution"]
        rows.append({
            "task_id":            item["task_id"],
            "source":             "humaneval",
            "function_code":      full_code.strip(),
            "entry_point":        item["entry_point"],
            "ground_truth_tests": item["test"].strip(),
        })
    return rows


def _load_mbpp():
    from datasets import load_dataset
    ds = load_dataset("google-research-datasets/mbpp", "sanitized", split="test")
    rows = []
    for item in ds:
        tests = "\n".join(item.get("test_list", []))
        rows.append({
            "task_id":            f"MBPP/{item['task_id']}",
            "source":             "mbpp",
            "function_code":      item["code"].strip(),
            "entry_point":        "",
            "ground_truth_tests": tests.strip(),
        })
    return rows


def _load_classeval():
    """Load ClassEval and extract individual methods as function-level samples.

    ClassEval contains 100 classes with ~412 methods total. Each method is
    extracted as a separate sample with enough class context (imports +
    constructor + target method) for the LLM to generate meaningful tests.
    This matches the function-level granularity of HumanEval/MBPP.

    Dataset: FudanSELab/ClassEval (HuggingFace).
    """
    from datasets import load_dataset
    ds = load_dataset("FudanSELab/ClassEval", split="test")
    rows = []
    for item in ds:
        class_name = item.get("class_name", "UnknownClass")
        import_stmt = item.get("import_statement", "")
        if isinstance(import_stmt, list):
            import_stmt = "\n".join(import_stmt)
        constructor = item.get("class_constructor", "")

        methods_info = item.get("methods_info", [])
        if not methods_info:
            continue

        for j, method_info in enumerate(methods_info):
            method_code = method_info.get("solution_code", "")
            if not method_code.strip():
                continue

            # Build self-contained function context:
            # imports + class definition + constructor + target method
            parts = []
            if import_stmt.strip():
                parts.append(import_stmt.strip())
            class_body = f"class {class_name}:\n"
            if constructor.strip():
                # Indent constructor under class
                constructor_lines = constructor.strip().split("\n")
                class_body += "\n".join(f"    {line}" for line in constructor_lines) + "\n\n"
            # Indent method under class
            method_lines = method_code.strip().split("\n")
            class_body += "\n".join(f"    {line}" for line in method_lines)
            parts.append(class_body)
            function_code = "\n\n".join(parts)

            test_code = method_info.get("test_code", "")
            method_name = method_info.get("method_name", f"method_{j}")

            if not test_code.strip():
                continue

            rows.append({
                "task_id":            f"ClassEval/{item.get('task_id', j)}/{method_name}",
                "source":             "classeval",
                "function_code":      function_code.strip(),
                "entry_point":        method_name,
                "ground_truth_tests": test_code.strip(),
            })
    return rows


def make_eval_dataset(force_reload: bool = False) -> list:
    """
    Load and cache the fixed evaluation subset (HumanEval + MBPP).
    Returns a list of dicts with keys: task_id, source, function_code,
    entry_point, ground_truth_tests.
    """
    CACHE_DIR.mkdir(parents=True, exist_ok=True)

    if DATASET_CACHE.exists() and not force_reload:
        with open(DATASET_CACHE, "rb") as f:
            return pickle.load(f)

    print("Downloading HumanEval...")
    humaneval = _load_humaneval()
    print(f"  {len(humaneval)} samples")

    print("Downloading MBPP...")
    mbpp = _load_mbpp()
    print(f"  {len(mbpp)} samples")

    combined = [r for r in humaneval + mbpp
                if r["function_code"].strip() and r["ground_truth_tests"].strip()]

    # Fixed reproducible subset
    rng = np.random.default_rng(DATASET_SEED)
    indices = rng.choice(len(combined), size=min(NUM_EVAL_SAMPLES, len(combined)), replace=False)
    subset = [combined[int(i)] for i in sorted(indices)]

    with open(DATASET_CACHE, "wb") as f:
        pickle.dump(subset, f)

    print(f"Eval subset: {len(subset)} samples saved to {DATASET_CACHE}")
    return subset


# Extended dataset with ClassEval (v4)
_CACHE_VERSION_V4 = "v4"
_DATASET_CACHE_V4 = CACHE_DIR / f"eval_dataset_{_CACHE_VERSION_V4}.pkl"

def make_eval_dataset_v4(force_reload: bool = False) -> list:
    """
    Load and cache the extended evaluation subset: HumanEval + MBPP + ClassEval.

    Same structure as make_eval_dataset() but draws from 3 benchmark sources.
    The v3 dataset (HumanEval+MBPP only) remains unchanged and available.
    Uses same seed=42 and NUM_EVAL_SAMPLES=100 for comparability.
    """
    CACHE_DIR.mkdir(parents=True, exist_ok=True)

    if _DATASET_CACHE_V4.exists() and not force_reload:
        with open(_DATASET_CACHE_V4, "rb") as f:
            return pickle.load(f)

    print("Downloading HumanEval...")
    humaneval = _load_humaneval()
    print(f"  {len(humaneval)} samples")

    print("Downloading MBPP...")
    mbpp = _load_mbpp()
    print(f"  {len(mbpp)} samples")

    print("Downloading ClassEval...")
    classeval = _load_classeval()
    print(f"  {len(classeval)} method-level samples")

    combined = [r for r in humaneval + mbpp + classeval
                if r["function_code"].strip() and r["ground_truth_tests"].strip()]

    rng = np.random.default_rng(DATASET_SEED)
    indices = rng.choice(len(combined), size=min(NUM_EVAL_SAMPLES, len(combined)), replace=False)
    subset = [combined[int(i)] for i in sorted(indices)]

    # Report source distribution
    from collections import Counter
    source_counts = Counter(s["source"] for s in subset)
    print(f"Eval subset v4: {len(subset)} samples — {dict(source_counts)}")

    with open(_DATASET_CACHE_V4, "wb") as f:
        pickle.dump(subset, f)

    print(f"Saved to {_DATASET_CACHE_V4}")
    return subset


# ---------------------------------------------------------------------------
# Knowledge base (for RAG retrieval in train_unitest.py)
# ---------------------------------------------------------------------------

class VectorStore:
    """Simple in-memory vector store using sentence-transformers + numpy."""

    def __init__(self, texts: list, embeddings: np.ndarray, sources: list):
        self.texts = texts
        self.embeddings = embeddings  # shape: (N, dim)
        self.sources = sources

    def search_with_scores(self, query: str, model, top_k: int = 3):
        """Return (context_str, noise_rate).

        context_str — top_k chunks joined by '---'.
        noise_rate  — fraction of retrieved chunks with cosine sim < NOISE_THRESHOLD.
                      Returns ('', nan) when the knowledge base is empty.
        """
        if not self.texts:
            return "", float("nan")
        from sklearn.metrics.pairwise import cosine_similarity
        q_emb = model.encode([query])
        sims = cosine_similarity(q_emb, self.embeddings)[0]
        top_idx = np.argsort(sims)[::-1][:top_k]
        top_sims = [float(sims[int(i)]) for i in top_idx]
        context_str = "\n\n---\n\n".join(self.texts[int(i)] for i in top_idx)
        noise_rate = sum(1 for s in top_sims if s < NOISE_THRESHOLD) / len(top_sims)
        return context_str, noise_rate

    def search(self, query: str, model, top_k: int = 3) -> str:
        """Return top_k most relevant text chunks concatenated (discards noise_rate)."""
        context_str, _ = self.search_with_scores(query, model, top_k=top_k)
        return context_str

    def random_sample(self, n: int) -> tuple:
        """Return n randomly-sampled chunks (for random_rag ablation baseline).

        Used to isolate whether retrieval QUALITY (cosine similarity ranking)
        matters vs simply having additional context tokens in the prompt.
        Returns (context_str, nan) — noise_rate undefined for random retrieval.
        """
        if not self.texts:
            return "", float("nan")
        rng = np.random.default_rng(seed=None)   # different random each call
        idx = rng.choice(len(self.texts), size=min(n, len(self.texts)), replace=False)
        context_str = "\n\n---\n\n".join(self.texts[int(i)] for i in idx)
        return context_str, float("nan")


def _chunk_text(text: str, chunk_size: int = KB_CHUNK_SIZE, overlap: int = KB_CHUNK_OVERLAP) -> list:
    """Split text into overlapping fixed-size character chunks.

    Overlapping windows ensure that relevant content at chunk boundaries is
    not split across two non-retrieved chunks. Each chunk is independently
    embedded and retrieved by cosine similarity.
    """
    chunks = []
    step = chunk_size - overlap
    for start in range(0, len(text), step):
        chunk = text[start:start + chunk_size].strip()
        if len(chunk) > 50:  # skip near-empty trailing chunks
            chunks.append(chunk)
    return chunks


def build_knowledge_base(force_reload: bool = False):
    """
    Fetch testing documentation URLs, chunk into overlapping windows,
    encode with sentence-transformers, and return (VectorStore, embedding_model).

    Knowledge base: 14 pytest/unittest/mock URLs × ~5-20 chunks each
    ≈ 100-200 chunks total. TOP_K=5 retrieves the 5 most relevant chunks.
    """
    CACHE_DIR.mkdir(parents=True, exist_ok=True)

    if KB_CACHE.exists() and not force_reload:
        with open(KB_CACHE, "rb") as f:
            cached = pickle.load(f)
        from sentence_transformers import SentenceTransformer
        model = SentenceTransformer("all-MiniLM-L6-v2")
        return VectorStore(cached["texts"], cached["embeddings"], cached["sources"]), model

    import requests
    from bs4 import BeautifulSoup
    from sentence_transformers import SentenceTransformer

    model = SentenceTransformer("all-MiniLM-L6-v2")

    texts, sources = [], []
    for url in KNOWLEDGE_BASE_URLS:
        try:
            resp = requests.get(url, timeout=20, headers={"User-Agent": "Mozilla/5.0"})
            resp.raise_for_status()
            soup = BeautifulSoup(resp.content, "html.parser")
            main = soup.find("main") or soup.find("article") or soup.find("body")
            text = main.get_text(separator="\n", strip=True) if main else soup.get_text(separator="\n", strip=True)
            if len(text) < 100:
                print(f"  Skipped (too short): {url}")
                continue
            chunks = _chunk_text(text)
            for chunk in chunks:
                texts.append(chunk)
                sources.append(url)
            print(f"  Loaded: {url}  ({len(chunks)} chunks)")
            time.sleep(0.5)
        except Exception as e:
            print(f"  Skipped {url}: {e}")

    embeddings = model.encode(texts, show_progress_bar=True, batch_size=64)

    with open(KB_CACHE, "wb") as f:
        pickle.dump({"texts": texts, "embeddings": embeddings, "sources": sources}, f)

    print(f"Knowledge base: {len(texts)} chunks from {len(KNOWLEDGE_BASE_URLS)} URLs cached → {KB_CACHE}")
    return VectorStore(texts, embeddings, sources), model


# ---------------------------------------------------------------------------
# Evaluation — the ground truth metric. DO NOT MODIFY.
# ---------------------------------------------------------------------------

def _check_syntax(code: str) -> float:
    if not code or not code.strip():
        return 0.0
    try:
        ast.parse(code)
        return 1.0
    except SyntaxError:
        return 0.0


def _count_assertions(code: str) -> int:
    if not code:
        return 0
    try:
        tree = ast.parse(code)
        return sum(1 for n in ast.walk(tree) if isinstance(n, ast.Assert))
    except SyntaxError:
        return len(re.findall(r"\bassert\b", code))


def _count_test_funcs(code: str) -> int:
    return len(re.findall(r"^def test_", code or "", re.MULTILINE))


_st_model_cache = None


def _get_st_model():
    global _st_model_cache
    if _st_model_cache is None:
        from sentence_transformers import SentenceTransformer
        _st_model_cache = SentenceTransformer("all-MiniLM-L6-v2")
    return _st_model_cache


def _semantic_similarity(text_a: str, text_b: str) -> float:
    """Cosine similarity between generated and reference tests using sentence-transformers."""
    if not text_a.strip() or not text_b.strip():
        return 0.0
    try:
        from sklearn.metrics.pairwise import cosine_similarity
        model = _get_st_model()
        embs = model.encode([text_a[:4000], text_b[:4000]])
        score = cosine_similarity([embs[0]], [embs[1]])[0][0]
        return float(max(0.0, score))
    except Exception:
        return 0.0


def _edge_case_score(code: str) -> float:
    if not code:
        return 0.0
    patterns = [r"\bNone\b", r'""', r"\[\]", r"\{\}", r"\b0\b",
                r"-\d+", r"\bpytest\.raises\b", r"\bValueError\b",
                r"\bTypeError\b", r"\bIndexError\b"]
    hits = sum(1 for p in patterns if re.search(p, code))
    return min(1.0, hits / 4.0)


# Unified faithfulness metrics — single source of truth for all PhD tasks.
# Defined in faithfulness.py so docstring and test-oracle tasks use identical formula.
from faithfulness import compute_faithfulness, llm_judge_faithfulness  # noqa: F401 (re-exported)


def evaluate_tests(generated: str, ground_truth: str, function_code: str) -> dict:
    """
    Compute all evaluation metrics for one generated test suite.
    Returns dict of metric_name -> score.
    """
    syntax  = _check_syntax(generated)
    asserts = _count_assertions(generated)
    nfuncs  = _count_test_funcs(generated)
    edges   = _edge_case_score(generated)

    # Assertion density: normalize by test function count
    assert_density = (asserts / max(nfuncs, 1)) / 5.0   # saturates at 5 asserts/test
    assert_density = min(1.0, assert_density)

    # ROUGE-1 vs ground truth
    rouge_score = 0.0
    try:
        from rouge import Rouge
        if generated.strip() and ground_truth.strip():
            r = Rouge()
            rouge_score = r.get_scores(generated.lower(), ground_truth.lower())[0]["rouge-1"]["f"]
    except Exception:
        pass

    # Semantic similarity vs ground truth (sentence-transformers cosine similarity)
    sem_sim = _semantic_similarity(generated, ground_truth)

    return {
        "syntactic_validity": syntax,
        "assertion_count":    float(asserts),
        "test_func_count":    float(nfuncs),
        "edge_case_score":    edges,
        "assert_density":     assert_density,
        "rouge_1_f1":         rouge_score,
        "semantic_sim":       sem_sim,
    }


def execute_tests(generated_tests: str, function_code: str,
                  timeout_secs: int = 10) -> dict:
    """
    Run generated tests against the function under test in an isolated subprocess.

    Creates a temp file combining function_code + generated_tests, then runs
    pytest on it. Returns execution metrics as a diagnostic (not part of val_score).

    Returns dict with keys:
        exec_pass_rate   : float  (0.0-1.0, fraction of tests passed; 0.0 on error)
        exec_total_tests : int    (number of test items collected)
        exec_passed      : int
        exec_failed      : int
        exec_errors      : int
        exec_status      : str    ("pass"|"partial"|"fail"|"timeout"|"syntax_error"|"no_tests"|"error")
    """
    result = {
        "exec_pass_rate": 0.0, "exec_total_tests": 0,
        "exec_passed": 0, "exec_failed": 0, "exec_errors": 0,
        "exec_status": "error",
    }

    if not generated_tests or not generated_tests.strip():
        result["exec_status"] = "no_tests"
        return result

    try:
        with tempfile.TemporaryDirectory() as tmpdir:
            test_file = os.path.join(tmpdir, "test_generated.py")
            # Combine function code + generated tests into one file
            content = f"# --- Function under test ---\n{function_code}\n\n# --- Generated tests ---\n{generated_tests}\n"
            with open(test_file, "w") as f:
                f.write(content)

            proc = subprocess.run(
                [sys.executable, "-m", "pytest", test_file, "--tb=short", "-q", "--no-header"],
                capture_output=True, text=True,
                timeout=timeout_secs, cwd=tmpdir,
            )
            output = proc.stdout + proc.stderr

            # Parse pytest summary line: "3 passed, 2 failed, 1 error"
            passed = 0
            failed = 0
            errors = 0

            for match in re.finditer(r"(\d+)\s+passed", output):
                passed = int(match.group(1))
            for match in re.finditer(r"(\d+)\s+failed", output):
                failed = int(match.group(1))
            for match in re.finditer(r"(\d+)\s+error", output):
                errors = int(match.group(1))

            total = passed + failed + errors

            if total == 0:
                # Check for collection errors (syntax errors in generated tests)
                if "SyntaxError" in output or "IndentationError" in output:
                    result["exec_status"] = "syntax_error"
                elif "no tests ran" in output or "collected 0 items" in output:
                    result["exec_status"] = "no_tests"
                else:
                    result["exec_status"] = "error"
                return result

            result["exec_passed"] = passed
            result["exec_failed"] = failed
            result["exec_errors"] = errors
            result["exec_total_tests"] = total
            result["exec_pass_rate"] = passed / total

            if failed == 0 and errors == 0:
                result["exec_status"] = "pass"
            elif passed > 0:
                result["exec_status"] = "partial"
            else:
                result["exec_status"] = "fail"

    except subprocess.TimeoutExpired:
        result["exec_status"] = "timeout"
    except Exception:
        result["exec_status"] = "error"

    return result


def compute_val_score(metrics_list: list) -> float:
    """
    Compute a single composite val_score from a list of per-sample metric dicts.
    Higher is better (opposite of val_bpb).

    Weights:
      syntactic_validity : 0.30  (must be valid Python)
      edge_case_score    : 0.25  (covers edge cases)
      assert_density     : 0.20  (meaningful assertions per test)
      semantic_sim       : 0.15  (semantic similarity to reference via sentence-transformers)
      rouge_1_f1         : 0.10  (lexical overlap with reference)
    """
    if not metrics_list:
        return 0.0

    def avg(key):
        vals = [m[key] for m in metrics_list if key in m]
        return sum(vals) / len(vals) if vals else 0.0

    score = (
        0.30 * avg("syntactic_validity") +
        0.25 * avg("edge_case_score") +
        0.20 * avg("assert_density") +
        0.15 * avg("semantic_sim") +
        0.10 * avg("rouge_1_f1")
    )
    return round(score, 6)


# ---------------------------------------------------------------------------
# One-time setup
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    print("=== prepare_unitest.py: one-time setup ===")
    print("\n1. Loading eval dataset...")
    dataset = make_eval_dataset(force_reload=True)
    print(f"   {len(dataset)} samples ready.")

    print("\n2. Building knowledge base...")
    kb, emb_model = build_knowledge_base(force_reload=True)
    print(f"   {len(kb.texts)} docs indexed.")

    print("\n3. Building v4 dataset (HumanEval + MBPP + ClassEval)...")
    make_eval_dataset_v4(force_reload=True)

    print("\nSetup complete. You can now run: python train_unitest.py")
