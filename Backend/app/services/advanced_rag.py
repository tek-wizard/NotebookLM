"""Advanced RAG pipeline (RAG-2).

This module wires together the advanced retrieval techniques from the syllabus
into a single pipeline. Plain (RAG-1) retrieval is just "embed query -> vector
search -> stuff into prompt". Each stage here improves on that:

    Query Translation (SLM)   -> clean, standalone retrieval query
    Sub-query enhancement     -> split complex questions into parts
    HyDE                       -> embed a hypothetical answer, not the question
    Hybrid retrieval           -> dense vectors + lexical keywords (vectorDB)
    Cross-encoder re-ranking   -> reorder candidates by true relevance
    Corrective RAG             -> grade context; broaden or refuse if weak
    LLM judge                  -> verify the final answer is grounded

Every stage records what it did into a ``trace`` dict so the pipeline is
observable end-to-end (handy for the assignment demo and for debugging).
"""

import os

from dotenv import load_dotenv
from huggingface_hub import AsyncInferenceClient

from app.models.llm import AnswerJudgement, RetrievalGrade, SubQueries
from app.services import config
from app.services.prompts import (
    grader_system_prompt,
    hyde_system_prompt,
    judge_system_prompt,
    subquery_system_prompt,
)
from app.services.vectorDB import (
    build_context_from_chunks,
    is_small_document,
    rerank_chunks,
    retrieve_candidates,
    session_chunks_map,
)

load_dotenv()

# A dedicated client for the auxiliary "small model" style calls (decompose,
# HyDE, grade, judge). Reuses the same HF model as generation by default.
client = AsyncInferenceClient(
    model=os.getenv("huggingface_model_id"),
    api_key=os.getenv("hugging_face_key"),
)


async def _json_completion(system_prompt: str, user_content: str, schema_model):
    """Call the model and parse its reply into ``schema_model``.

    Returns a validated pydantic object, or ``None`` if the call fails / the
    model returns unparseable JSON. Callers always have a graceful fallback so
    a flaky auxiliary call never breaks the main answer.
    """
    response_format = {
        "type": "json_schema",
        "json_schema": {
            "name": schema_model.__name__,
            "schema": schema_model.model_json_schema(),
            "strict": True,
        },
    }
    try:
        raw = (
            await client.chat_completion(
                messages=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": user_content},
                ],
                response_format=response_format,
                temperature=0.1,
            )
        ).choices[0].message.content
        return schema_model.model_validate_json(raw)
    except Exception:
        return None


# --- Stage: Sub-query enhancement ------------------------------------------
async def decompose_query(query: str) -> list[str]:
    """Return a list of retrieval queries: the original plus any sub-queries."""
    if not config.ENABLE_SUBQUERY:
        return [query]

    result = await _json_completion(subquery_system_prompt, query, SubQueries)
    if result and result.is_complex and result.sub_queries:
        # Keep the original query too -- it anchors the overall intent.
        return [query, *result.sub_queries[: config.SUBQUERY_MAX]]
    return [query]


# --- Stage: HyDE -----------------------------------------------------------
async def generate_hyde(query: str) -> str | None:
    """Draft a hypothetical answer passage to use as an extra search query."""
    if not config.ENABLE_HYDE:
        return None
    try:
        response = await client.chat_completion(
            messages=[
                {"role": "system", "content": hyde_system_prompt},
                {"role": "user", "content": query},
            ],
            temperature=0.3,
            max_tokens=160,
        )
        passage = (response.choices[0].message.content or "").strip()
        return passage or None
    except Exception:
        return None


# --- Stage: Corrective RAG grading -----------------------------------------
async def grade_retrieval(query: str, context: str) -> RetrievalGrade:
    if not config.ENABLE_CORRECTIVE_RAG or not context:
        return RetrievalGrade(verdict="correct", reason="grading disabled or no context")

    user_content = f"USER QUESTION:\n{query}\n\nRETRIEVED CONTEXT:\n{context}"
    result = await _json_completion(grader_system_prompt, user_content, RetrievalGrade)
    # Fail open: if grading itself fails, assume the context is usable.
    return result or RetrievalGrade(verdict="correct", reason="grader unavailable")


# --- Stage: LLM judge ------------------------------------------------------
async def judge_answer(context: str, answer: str) -> AnswerJudgement:
    if not config.ENABLE_LLM_JUDGE:
        return AnswerJudgement(grounded=True, reason="judge disabled")

    user_content = f"RETRIEVED CONTEXT:\n{context}\n\nANSWER:\n{answer}"
    result = await _json_completion(judge_system_prompt, user_content, AnswerJudgement)
    return result or AnswerJudgement(grounded=True, reason="judge unavailable")


async def retrieve_with_pipeline(
    session_id: str,
    retrieval_query: str,
    include_all: bool,
) -> tuple[str, dict]:
    """Run the full advanced retrieval pipeline.

    Returns ``(context, trace)`` where ``context`` is the final re-ranked,
    corrective-graded text block to feed the generator, and ``trace`` is a
    record of every stage that ran.
    """
    trace: dict = {
        "retrieval_query": retrieval_query,
        "stages": [],
    }

    chunks = session_chunks_map.get(session_id, [])
    if not chunks:
        return "", {**trace, "verdict": "no_document"}

    # Small docs / summary requests: skip retrieval, use the whole document.
    # (Retrieval can only lose information when the doc already fits the window.)
    if include_all or is_small_document(chunks):
        trace["stages"].append("full_context")
        context = build_context_from_chunks(chunks, limit=max(config.FINAL_CONTEXT_K, len(chunks)))
        return context, {**trace, "verdict": "full_context"}

    # 1) Sub-query enhancement.
    queries = await decompose_query(retrieval_query)
    if len(queries) > 1:
        trace["stages"].append("subquery")
    trace["sub_queries"] = queries

    # 2) HyDE: add a hypothetical-answer passage as an extra search query.
    hyde_passage = await generate_hyde(retrieval_query)
    if hyde_passage:
        trace["stages"].append("hyde")
        trace["hyde_passage"] = hyde_passage
        queries = [*queries, hyde_passage]

    # 3) Hybrid retrieval into a wide candidate pool.
    candidates = await retrieve_candidates(session_id, queries)
    trace["candidate_count"] = len(candidates)

    # 4) Cross-encoder re-ranking down to the final top-K.
    top_chunks = rerank_chunks(retrieval_query, candidates, limit=config.FINAL_CONTEXT_K)
    if config.ENABLE_RERANK:
        trace["stages"].append("rerank")
    context = build_context_from_chunks(top_chunks, limit=config.FINAL_CONTEXT_K)

    # 5) Corrective RAG: grade the context; broaden once if it looks weak.
    grade = await grade_retrieval(retrieval_query, context)
    trace["grade"] = grade.verdict
    if config.ENABLE_CORRECTIVE_RAG:
        trace["stages"].append("corrective_rag")

    if grade.verdict == "incorrect":
        # Correction: fall back to a wider view of the document before giving up.
        wider = build_context_from_chunks(candidates, limit=config.CANDIDATE_POOL)
        return wider, {**trace, "verdict": "corrected_broaden"}

    if grade.verdict == "ambiguous":
        # Add a few more candidates to fill likely gaps.
        extra = rerank_chunks(retrieval_query, candidates, limit=config.FINAL_CONTEXT_K + 2)
        context = build_context_from_chunks(extra, limit=config.FINAL_CONTEXT_K + 2)
        return context, {**trace, "verdict": "ambiguous_expanded"}

    return context, {**trace, "verdict": "correct"}
