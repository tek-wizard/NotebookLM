from pydantic import BaseModel


# ---------------------------------------------------------------------------
# Query Translation / rewriting (RAG-2: "Query rewriting using SLMs")
# The small/cheap model decides whether a search is needed and produces a
# clean, standalone, keyword-rich query for retrieval.
# ---------------------------------------------------------------------------
class Search(BaseModel):
    needs_search: bool
    search_query: str | None = None


# ---------------------------------------------------------------------------
# Sub-query enhancement (RAG-2: "Sub query enhancement")
# A complex question is decomposed into 1-3 simpler, self-contained sub-queries
# so each can be retrieved for independently and the evidence combined.
# ---------------------------------------------------------------------------
class SubQueries(BaseModel):
    is_complex: bool
    sub_queries: list[str] = []


# ---------------------------------------------------------------------------
# Corrective RAG grading (RAG-2: "Corrective RAG")
# After retrieval we grade how well the retrieved context supports the question.
#   - "correct"     -> context is relevant, generate normally
#   - "ambiguous"   -> partial signal, broaden retrieval before generating
#   - "incorrect"   -> nothing relevant, refuse instead of hallucinating
# ---------------------------------------------------------------------------
class RetrievalGrade(BaseModel):
    verdict: str  # "correct" | "ambiguous" | "incorrect"
    reason: str | None = None


# ---------------------------------------------------------------------------
# LLM judge (RAG-2: "LLM judges")
# A separate model call checks that the generated answer is actually grounded
# in the retrieved context (no hallucination) before it is returned.
# ---------------------------------------------------------------------------
class AnswerJudgement(BaseModel):
    grounded: bool
    reason: str | None = None
