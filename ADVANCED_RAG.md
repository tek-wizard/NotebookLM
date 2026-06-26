# Advanced RAG — Upgrade Notes

This document maps the **RAG 2 (Advanced Retrieval Strategies)** syllabus to the
exact code that implements each technique. The original app was a basic
(RAG 1) pipeline: *embed query → vector search → stuff into prompt*. This
upgrade turns it into an advanced, self-correcting retrieval pipeline.

## The problem with basic RAG (the bottlenecks)

A naive `query → embed → top-k vector search → LLM` pipeline fails in
predictable ways, and these are the **bottlenecks** the advanced techniques
attack:

1. **Bad queries in → bad chunks out.** Short, pronoun-heavy, or multi-part
   questions embed poorly. → *Query rewriting + Sub-query enhancement*.
2. **Question ≠ answer in embedding space.** A question embeds differently from
   the passage that answers it. → *HyDE*.
3. **Vector search is approximate.** A bi-encoder can rank a loosely-related
   chunk above the right one. → *Cross-encoder re-ranking*.
4. **Garbage context → confident hallucination.** If retrieval misses, the LLM
   invents an answer. → *Corrective RAG + LLM judge*.
5. **Context window / token limits.** You cannot stuff the whole doc in.
   → *Retrieve-wide-rerank-narrow + chunk sizing*.

## Pipeline overview

For every question (`app/services/LLM.py::handle_user_query`):

```
            ┌─────────────────────────── small talk? → answer directly
user query ─┤
            ▼
  1. Query Translation / rewrite (SLM)      build_retrieval_query()   [LLM.py]
  2. Sub-query enhancement                  decompose_query()         [advanced_rag.py]
  3. HyDE (hypothetical answer → embed)     generate_hyde()           [advanced_rag.py]
  4. Hybrid retrieval (vector + lexical)    retrieve_candidates()     [vectorDB.py]
  5. Cross-encoder re-ranking               rerank_chunks()           [vectorDB.py]
  6. Corrective RAG (grade → correct)       grade_retrieval()         [advanced_rag.py]
  7. Generate answer                        chat_completion()         [LLM.py]
  8. LLM judge (grounded? else refuse)      judge_answer()            [advanced_rag.py]
```

Every stage records what it did in a `trace` object that is returned in the API
response, so you can *see* the advanced pipeline working.

## Technique-by-technique

### 1. Query rewriting using SLMs / Query Translation
`build_retrieval_query()` in `LLM.py` + `search_system_prompt`.
A small/cheap model turns the raw chat turn ("tell me more about him") into a
clean, standalone, keyword-rich query before retrieval. It also decides when a
search is unnecessary (greetings).

### 2. Sub-query enhancement
`decompose_query()` in `advanced_rag.py` + `subquery_system_prompt`.
Complex/comparative questions are split into 2–3 standalone sub-queries; each is
retrieved for independently and the evidence is pooled. Simple questions skip
this (no extra latency).

### 3. HyDE (Hypothetical Document Embeddings)
`generate_hyde()` in `advanced_rag.py` + `hyde_system_prompt`.
We ask the model to *draft a hypothetical answer paragraph* and use **that** as
an additional search query. A fake answer lives much closer to real passages in
embedding space than the short question does — even if the draft is factually
wrong, it improves recall.

### 4. Re-ranking strategies (cross-encoders)
`rerank_chunks()` in `vectorDB.py` (uses `fastembed` `TextCrossEncoder`).
Vector search (a **bi-encoder**) embeds query and chunk separately — fast but
approximate. A **cross-encoder** reads the `(query, chunk)` pair *together* and
scores true relevance — accurate but slow. So we **retrieve wide** (a 12-chunk
candidate pool) and **rerank narrow** (down to the best 4). Falls back to the
original order if the model can't load.

### 5. Corrective RAG (CRAG)
`grade_retrieval()` in `advanced_rag.py` + `grader_system_prompt`.
After retrieval, a grader call judges whether the context can actually answer
the question: `correct` → generate; `ambiguous` → broaden retrieval and retry;
`incorrect` → fall back to a wider document view instead of hallucinating.

### 6. LLM judges
`judge_answer()` in `advanced_rag.py` + `judge_system_prompt`.
A separate, strict judge verifies the generated answer is **grounded** in
(entailed by) the retrieved context. If the answer smuggled in facts that
aren't supported, it is replaced with the honest "out of context" refusal. This
is the last line of defence against hallucination.

### 7. Context window & token bottlenecks
Addressed by the *retrieve-wide / rerank-narrow* shape: we only ever send the
top `RAG_FINAL_CONTEXT_K` chunks to the generator, keeping the prompt small.
Small documents skip retrieval entirely (`is_small_document`) and use full
context, because retrieval can only *lose* information when the doc already fits.

### 8. Chunk size & overlap tradeoffs
`RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=50)` in
`vectorDB.py`. Smaller chunks = more precise retrieval but more fragmentation;
larger chunks = more context per hit but noisier embeddings. 500/50 is a
balanced default; the overlap preserves sentences that straddle a boundary.

## Speed vs accuracy tradeoff

Each advanced stage is an extra model call (more latency + cost) bought for
better quality. **Every stage is independently toggleable** via
`app/services/config.py` (and environment variables), which *is* the
speed/accuracy dial:

| Env var | Stage | Default |
|---|---|---|
| `ENABLE_QUERY_REWRITE` | Query rewriting (SLM) | on |
| `ENABLE_SUBQUERY` | Sub-query enhancement | on |
| `ENABLE_HYDE` | HyDE | on |
| `ENABLE_RERANK` | Cross-encoder re-ranking | on |
| `ENABLE_CORRECTIVE_RAG` | Corrective RAG grading | on |
| `ENABLE_LLM_JUDGE` | LLM judge | on |
| `RAG_CANDIDATE_POOL` | candidates fetched before rerank | 12 |
| `RAG_FINAL_CONTEXT_K` | chunks sent to the generator | 4 |
| `RAG_RERANK_MODEL` | cross-encoder model | `Xenova/ms-marco-MiniLM-L-6-v2` |

Turn everything off → you get the original basic RAG. Turn everything on → max
accuracy, slowest. Set them per your latency budget.

> **Deployment note:** the cross-encoder downloads a small ONNX model on first
> use. On serverless cold starts that download may be slow; the code degrades
> gracefully (skips reranking) if it can't load, so the app never breaks.
