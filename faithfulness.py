"""
faithfulness.py — Unified faithfulness metric for RQ4 cross-task comparison.
Provides both token-overlap (fast, offline) and LLM-as-Judge (validated)
faithfulness metrics. LLM-as-Judge uses DeepSeek-Coder 6.7B via Ollama,
validated against human ratings (Pearson r=0.925) in companion Springer paper.

Used by all PhD tasks:
    Docstring Generation  (RAG-Docstring repo)
    Test Oracle Generation (autoresearch/train_unitest.py)
    Code Completion        (future)

Definition
----------
    score = |content_tokens(generated) ∩ content_tokens(context)|
            / |content_tokens(generated)|

    content_tokens: alphanumeric words, length > 2, not in _STOP_TOKENS.
    Stop-word filtering prevents common Python keywords and English function
    words from inflating the score regardless of true grounding.

Returns
-------
    float in [0.0, 1.0]  — fraction of generated content grounded in context
    float NaN            — when context is empty (plain_llm: no retrieval used),
                           or when inputs are invalid / generation failed.

    NaN explicitly distinguishes "retrieval not used" from "zero faithfulness",
    which is important for RQ2 / RQ4 cross-task analysis.

Cross-task comparability
------------------------
    The same formula is applied to docstring text, test code, and completion
    code — making the score directly comparable across tasks for RQ4.
"""

import re
import math

# ---------------------------------------------------------------------------
# LLM-as-Judge config
# ---------------------------------------------------------------------------

LLM_JUDGE_MODEL = "deepseek-coder:6.7b"  # same model validated in Springer companion paper

# ---------------------------------------------------------------------------
# Stop tokens
# ---------------------------------------------------------------------------

_STOP_TOKENS: frozenset = frozenset({
    # Python keywords & builtins
    "def", "return", "import", "from", "class", "for", "if", "else", "elif",
    "in", "is", "not", "and", "or", "try", "except", "raise", "with", "as",
    "pass", "break", "continue", "while", "yield", "lambda", "assert",
    "global", "nonlocal", "del", "none", "true", "false", "self", "cls",
    "print", "len", "range", "type", "str", "int", "float", "list", "dict",
    "set", "bool", "tuple", "any", "all", "map", "zip", "sum", "min", "max",
    # English function words
    "the", "a", "an", "of", "to", "in", "is", "it", "be", "that", "this",
    "for", "on", "are", "was", "with", "at", "by", "from", "or", "but",
    "not", "what", "all", "were", "when", "we", "there", "can", "each",
    "which", "do", "how", "their", "will", "out", "so", "its", "then",
    "than", "into", "has", "have", "had", "been", "would", "could", "should",
    "may", "might", "about", "also", "use", "using", "used",
    "you", "your", "they", "them", "one", "two", "more", "any",
})


def _content_tokens(text: str) -> frozenset:
    """Extract meaningful tokens: alpha-start words, length > 2, not stop words."""
    tokens = re.findall(r"\b[a-zA-Z]\w*\b", text.lower())
    return frozenset(t for t in tokens if len(t) > 2 and t not in _STOP_TOKENS)


# ---------------------------------------------------------------------------
# Primary metric
# ---------------------------------------------------------------------------

def compute_faithfulness(generated: str, context: str) -> float:
    """
    Unified faithfulness metric — cross-task comparable.

    Parameters
    ----------
    generated : str
        The LLM output (docstring text, test code, or completion code).
    context : str
        The retrieved context passed to the LLM. Pass '' or None for
        plain_llm runs where no retrieval was performed.

    Returns
    -------
    float
        Score in [0.0, 1.0], or NaN if not applicable.
    """
    if not generated or not generated.strip():
        return float("nan")           # generation failure
    if not context or not context.strip():
        return float("nan")           # plain_llm: retrieval not used

    gen_tokens = _content_tokens(generated)
    ctx_tokens = _content_tokens(context)

    if not gen_tokens:
        return float("nan")

    overlap = len(gen_tokens & ctx_tokens)
    return round(overlap / len(gen_tokens), 6)


# ---------------------------------------------------------------------------
# LLM-as-Judge faithfulness (validated against human ratings)
# ---------------------------------------------------------------------------

def llm_judge_faithfulness(generated: str, context: str, function_code: str = "") -> float:
    """
    LLM-as-Judge faithfulness score using DeepSeek-Coder 6.7B.

    Rates how well the generated tests are grounded in retrieved documentation.
    Returns float in [0.0, 1.0], or NaN if not applicable (plain_llm / failure).

    Validated against human ratings in the companion Springer paper
    (Pearson r=0.925, Spearman ρ=0.839, N=51 annotated samples).

    Parameters
    ----------
    generated : str   — generated test code
    context   : str   — retrieved documentation context (empty → NaN)
    function_code : str — optional; prepended for additional grounding context
    """
    if not generated or not generated.strip():
        return float("nan")
    if not context or not context.strip():
        return float("nan")  # plain_llm: retrieval not used

    prompt = (
        "You are evaluating whether generated pytest tests are grounded in "
        "retrieved testing documentation.\n\n"
        f"Retrieved documentation:\n{context[:2000]}\n\n"
        + (f"Function under test:\n{function_code[:500]}\n\n" if function_code else "")
        + f"Generated tests:\n{generated[:1500]}\n\n"
        "Rate how faithfully the tests use patterns, idioms, and techniques "
        "from the retrieved documentation.\n"
        "Respond with ONLY a single decimal number from 0.0 to 1.0:\n"
        "  1.0 = tests directly use patterns/idioms from the documentation\n"
        "  0.5 = tests partially align with documentation patterns\n"
        "  0.0 = tests show no grounding in the documentation\n\n"
        "Score:"
    )
    try:
        import ollama
        client = ollama.Client()
        resp = client.chat(
            model=LLM_JUDGE_MODEL,
            messages=[{"role": "user", "content": prompt}],
            options={"temperature": 0.0},
        )
        try:
            raw = resp.message.content.strip()
        except AttributeError:
            raw = resp.get("message", {}).get("content", "").strip()

        match = re.search(r"\b(1\.0|0\.\d+|[01])\b", raw)
        if match:
            score = float(match.group(1))
            return round(max(0.0, min(1.0, score)), 6)
        return float("nan")
    except Exception:
        return float("nan")


# ---------------------------------------------------------------------------
# Batch helper (for offline re-scoring of existing result sets)
# ---------------------------------------------------------------------------

def batch_faithfulness(pairs: list) -> list:
    """
    Compute faithfulness for a list of (generated, context) pairs.

    Returns a list of floats (NaN where not applicable).
    Useful for re-scoring RAG-Docstring results with the unified formula.
    """
    return [compute_faithfulness(g, c) for g, c in pairs]


# ---------------------------------------------------------------------------
# Summary helper
# ---------------------------------------------------------------------------

def faithfulness_summary(scores: list) -> dict:
    """
    Summarise a list of faithfulness scores (ignores NaN).

    Returns dict with: mean, std, min, max, n_valid, n_nan.
    """
    valid = [s for s in scores if not math.isnan(s)]
    if not valid:
        return {"mean": float("nan"), "std": float("nan"),
                "min": float("nan"), "max": float("nan"),
                "n_valid": 0, "n_nan": len(scores)}
    n = len(valid)
    mean = sum(valid) / n
    std  = (sum((x - mean) ** 2 for x in valid) / max(n - 1, 1)) ** 0.5
    return {
        "mean":    round(mean, 6),
        "std":     round(std,  6),
        "min":     round(min(valid), 6),
        "max":     round(max(valid), 6),
        "n_valid": n,
        "n_nan":   len(scores) - n,
    }
