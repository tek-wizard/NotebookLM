"""Central configuration for the Advanced RAG pipeline.

Every advanced retrieval stage can be toggled independently. This is the
practical answer to the RAG-2 topic *"Tradeoff between speed and accuracy"*:
each stage adds an extra model call (latency + cost) in exchange for better
retrieval quality. Turn stages off to go faster, on to go more accurate.

All values can be overridden with environment variables so behaviour can be
tuned per deployment without touching code.
"""

import os


def _flag(name: str, default: bool) -> bool:
    raw = os.getenv(name)
    if raw is None:
        return default
    return raw.strip().lower() in {"1", "true", "yes", "on"}


def _int(name: str, default: int) -> int:
    raw = os.getenv(name)
    try:
        return int(raw) if raw is not None else default
    except ValueError:
        return default


# --- Advanced retrieval stage toggles --------------------------------------
ENABLE_QUERY_REWRITE = _flag("ENABLE_QUERY_REWRITE", True)   # Query Translation (SLM)
ENABLE_SUBQUERY = _flag("ENABLE_SUBQUERY", True)             # Sub-query enhancement
ENABLE_HYDE = _flag("ENABLE_HYDE", True)                     # HyDE principle
ENABLE_RERANK = _flag("ENABLE_RERANK", True)                 # Cross-encoder re-ranking
ENABLE_CORRECTIVE_RAG = _flag("ENABLE_CORRECTIVE_RAG", True) # Corrective RAG
ENABLE_LLM_JUDGE = _flag("ENABLE_LLM_JUDGE", True)           # LLM judge of the answer

# --- Retrieval sizing -------------------------------------------------------
# We over-fetch a wide candidate pool (CANDIDATE_POOL), then the cross-encoder
# re-ranker compresses it down to FINAL_CONTEXT_K chunks. This is the
# "retrieve wide, rerank narrow" pattern that keeps the final prompt inside the
# context window (RAG-2: "Context window & token bottlenecks").
CANDIDATE_POOL = _int("RAG_CANDIDATE_POOL", 12)
FINAL_CONTEXT_K = _int("RAG_FINAL_CONTEXT_K", 4)
SUBQUERY_MAX = _int("RAG_SUBQUERY_MAX", 3)

# Cross-encoder model used for re-ranking. ONNX-based via fastembed, so it is
# lightweight and needs no torch. Falls back gracefully if it cannot load.
RERANK_MODEL = os.getenv("RAG_RERANK_MODEL", "Xenova/ms-marco-MiniLM-L-6-v2")
