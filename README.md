# NotebookLM RAG Clone

A simple Google NotebookLM-style RAG application where users can upload a PDF or plain text file and chat with the content through a grounded interface.

## Live Links

- Frontend: https://resource-chat-gray.vercel.app/
- Backend: https://grounded-llm.vercel.app/

## What It Does

- Uploads PDF or TXT resources
- Extracts text and splits it into chunks
- Stores embeddings in Qdrant
- Retrieves relevant chunks for each question
- Uses an LLM to generate answers grounded in the uploaded document

## RAG Pipeline

1. Ingestion: PDF/TXT upload or raw text input
2. Chunking: `RecursiveCharacterTextSplitter`
3. Embedding: `sentence-transformers/all-MiniLM-L6-v2`
4. Storage: Qdrant vector database
5. Retrieval: vector search with lexical fallback for better grounding
6. Generation: Hugging Face inference client with context-aware prompting

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

## Project Structure

```text
Backend/   FastAPI API, ingestion, retrieval, generation
frontend/  React chat UI
```

## Assignment Goal

This project was built for Assignment 03: Google NotebookLM RAG. The goal is to implement the full pipeline end to end so the system can answer questions about documents it has never seen before using retrieved context instead of model memory.
