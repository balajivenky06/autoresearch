"""
prepare_unitest.py — fixed constants, dataset prep, and evaluation for unit test generation.
DO NOT MODIFY. This is the fixed harness; train_unitest.py is the only file the agent edits.

Mirrors prepare.py in the autoresearch repo but for unit test generation via LLM/RAG.

Usage (one-time setup):
    python prepare_unitest.py
"""

import os
import ast
import re
import time
import pickle
import numpy as np
from pathlib import Path

# ---------------------------------------------------------------------------
# Fixed constants — do not change
# ---------------------------------------------------------------------------

TIME_BUDGET      = 600   # seconds of generation time per experiment (10 min)
NUM_EVAL_SAMPLES = 25    # fixed eval subset size for fair comparison across runs
DATASET_SEED     = 42    # seed for reproducible subset selection

NOISE_THRESHOLD  = 0.3   # cosine similarity below this → chunk is "noisy" (diagnostic only)

# Cache dir: /content/.cache in Colab, ~/.cache elsewhere
_IN_COLAB = os.path.exists("/content")
CACHE_DIR     = Path("/content/.cache/autoresearch_unitest") if _IN_COLAB else Path.home() / ".cache" / "autoresearch_unitest"
DATASET_CACHE = CACHE_DIR / "eval_dataset.pkl"
KB_CACHE      = CACHE_DIR / "knowledge_base.pkl"

# Knowledge base URLs — testing documentation for RAG retrieval
KNOWLEDGE_BASE_URLS = [
    "https://docs.pytest.org/en/stable/how-to/assert.html",
    "https://docs.pytest.org/en/stable/how-to/parametrize.html",
    "https://docs.python.org/3/library/unittest.html",
    "https://realpython.com/pytest-python-testing/",
    "https://docs.pytest.org/en/stable/getting-started.html",
    "https://docs.pytest.org/en/stable/how-to/assert.html#assertions-about-expected-exceptions",
    "https://www.geeksforgeeks.org/unit-testing-python-unittest/",
    "https://semaphoreci.com/community/tutorials/testing-python-applications-with-pytest",
]

# ---------------------------------------------------------------------------
# Dataset loading
# ---------------------------------------------------------------------------

def _load_humaneval():
    from datasets import load_dataset
    ds = load_dataset("openai/openai_humaneval", split="test", trust_remote_code=True)
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
    ds = load_dataset("google-research-datasets/mbpp", "sanitized", split="test", trust_remote_code=True)
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


def build_knowledge_base(force_reload: bool = False):
    """
    Fetch testing documentation URLs, encode with sentence-transformers,
    and return (VectorStore, embedding_model).
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
            if len(text) > 100:
                texts.append(text[:4000])
                sources.append(url)
                print(f"  Loaded: {url}")
            time.sleep(0.5)
        except Exception as e:
            print(f"  Skipped {url}: {e}")

    embeddings = model.encode(texts, show_progress_bar=True)

    with open(KB_CACHE, "wb") as f:
        pickle.dump({"texts": texts, "embeddings": embeddings, "sources": sources}, f)

    print(f"Knowledge base: {len(texts)} docs cached.")
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
        embs = model.encode([text_a[:1000], text_b[:1000]])
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


def compute_faithfulness(generated: str, context: str) -> float:
    """Token-overlap faithfulness: fraction of unique generated tokens present in context.

    Returns NaN when context is empty (plain_llm has no retrieval context).
    Higher is better — output is grounded in retrieved docs, not hallucinated.
    """
    if not generated or not generated.strip() or not context or not context.strip():
        return float("nan")
    gen_tokens = set(re.findall(r"\w+", generated.lower()))
    ctx_tokens = set(re.findall(r"\w+", context.lower()))
    if not gen_tokens:
        return float("nan")
    return len(gen_tokens & ctx_tokens) / len(gen_tokens)


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

    print("\nSetup complete. You can now run: python train_unitest.py")
