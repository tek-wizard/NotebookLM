# NotebookLM Advanced RAG Clone

A Google NotebookLM-style RAG application where users can upload a PDF or plain text file and chat with the content through a grounded interface.

It runs an **Advanced RAG (RAG 2)** pipeline: query rewriting, sub-query
decomposition, HyDE, hybrid retrieval, cross-encoder re-ranking, Corrective RAG,
and an LLM judge. See **[ADVANCED_RAG.md](./ADVANCED_RAG.md)** for the full
technique-by-technique breakdown and how each maps to the code.

## Live Links

- Frontend: https://resource-chat-gray.vercel.app/
- Backend: https://grounded-llm.vercel.app/

## What It Does

- Uploads PDF or TXT resources
- Extracts text and splits it into chunks
- Stores embeddings in Qdrant
- Retrieves relevant chunks for each question
- Uses an LLM to generate answers grounded in the uploaded document

## RAG Pipeline (Advanced / RAG 2)

**Ingestion & indexing**

1. Ingestion: PDF/TXT upload or raw text input
2. Chunking: `RecursiveCharacterTextSplitter` (size 500 / overlap 50)
3. Embedding: `sentence-transformers/all-MiniLM-L6-v2`
4. Storage: Qdrant vector database

**Advanced retrieval & generation** (per question)

5. **Query rewriting (SLM)** — rewrite the chat turn into a clean standalone query
6. **Sub-query enhancement** — split complex questions into sub-queries
7. **HyDE** — draft a hypothetical answer and embed *that* for retrieval
8. **Hybrid retrieval** — dense vector search merged with lexical keyword search
9. **Cross-encoder re-ranking** — retrieve a wide pool, rerank down to the best chunks
10. **Corrective RAG** — grade the context; broaden or refuse if it is weak
11. **Generation** — Hugging Face inference client with context-aware prompting
12. **LLM judge** — verify the answer is grounded; refuse if it hallucinated

Each stage is independently toggleable (see *Tuning* below) — that is the
speed-vs-accuracy dial. The `/sessions/query` response also returns a `trace`
showing which stages ran.

## Chunking Strategy

The app uses `RecursiveCharacterTextSplitter` with:

- Chunk size: `500`
- Chunk overlap: `50`
- Separators: paragraph, newline, sentence, and word boundaries

This keeps chunks compact while preserving enough local context for retrieval.

## Tech Stack

- Frontend: React + Vite
- Backend: FastAPI
- Vector DB: Qdrant
- Embeddings: FastEmbed / MiniLM
- Re-ranker: FastEmbed cross-encoder (`ms-marco-MiniLM`)
- LLM: Hugging Face Inference API

## Local Setup

### Backend

```bash
cd Backend
pip install -r requirements.txt
uvicorn main:app --reload
```

### Frontend

```bash
cd frontend
npm install
npm run dev
```

Set the required backend environment variables before running or deploying:

- `QDRANT_URL`
- `QDRANT_API_KEY`
- `huggingface_model_id`
- `hugging_face_key`

For deployed frontend builds, set:

- `VITE_API_BASE_URL=https://grounded-llm.vercel.app/api`

## Tuning the Advanced RAG pipeline

Every advanced stage can be switched on/off independently via environment
variables — this is the practical speed-vs-accuracy dial. All default to `on`.

| Env var | Stage | Default |
|---|---|---|
| `ENABLE_QUERY_REWRITE` | Query rewriting (SLM) | `true` |
| `ENABLE_SUBQUERY` | Sub-query enhancement | `true` |
| `ENABLE_HYDE` | HyDE | `true` |
| `ENABLE_RERANK` | Cross-encoder re-ranking | `true` |
| `ENABLE_CORRECTIVE_RAG` | Corrective RAG grading | `true` |
| `ENABLE_LLM_JUDGE` | LLM judge | `true` |
| `RAG_CANDIDATE_POOL` | candidates fetched before rerank | `12` |
| `RAG_FINAL_CONTEXT_K` | chunks sent to the generator | `4` |
| `RAG_RERANK_MODEL` | cross-encoder model | `Xenova/ms-marco-MiniLM-L-6-v2` |

Set them all to `false` to get the original basic (RAG 1) behaviour.

## Project Structure

```text
Backend/
  app/services/config.py        Advanced RAG stage toggles & sizing
  app/services/advanced_rag.py  Sub-query, HyDE, Corrective RAG, LLM judge
  app/services/vectorDB.py      Chunking, hybrid retrieval, cross-encoder rerank
  app/services/LLM.py           Query rewrite + orchestration + generation
  app/services/prompts.py       System prompts for every stage
frontend/  React chat UI
```

## Assignment Goal

This project was built for Assignment 03: Google NotebookLM RAG. The goal is to implement the full pipeline end to end so the system can answer questions about documents it has never seen before using retrieved context instead of model memory.
