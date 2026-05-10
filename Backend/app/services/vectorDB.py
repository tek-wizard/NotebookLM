import os
import re
import threading
import uuid

os.environ["FASTEMBED_CACHE_PATH"] = "/tmp"
from dotenv import load_dotenv

from langchain_text_splitters import RecursiveCharacterTextSplitter
from qdrant_client import QdrantClient

from app.services.messagesData import delete_session_id_from_map

load_dotenv()

client = QdrantClient(
    url=os.getenv("QDRANT_URL"),
    api_key=os.getenv("QDRANT_API_KEY"),
)

client.set_model("sentence-transformers/all-MiniLM-L6-v2")

lock = threading.Lock()
session_chunks_map: dict[str, list[str]] = {}

SMALL_DOCUMENT_CHUNK_COUNT = 3
SMALL_DOCUMENT_CHAR_COUNT = 2200

splitter = RecursiveCharacterTextSplitter(
    chunk_size=500,
    chunk_overlap=50,
    separators=["\n\n", "\n", ". ", " "],
)


def normalize_text(value: str) -> str:
    return re.sub(r"\s+", " ", value).strip()


def deduplicate_chunks(chunks: list[str]) -> list[str]:
    seen: set[str] = set()
    unique_chunks: list[str] = []

    for chunk in chunks:
        cleaned_chunk = normalize_text(chunk)
        if not cleaned_chunk or cleaned_chunk in seen:
            continue
        seen.add(cleaned_chunk)
        unique_chunks.append(cleaned_chunk)

    return unique_chunks


def extract_keywords(text: str) -> list[str]:
    return re.findall(r"[a-zA-Z0-9]{2,}", text.lower())


def rank_chunks_by_keywords(
    chunks: list[str],
    query_text: str,
    limit: int = 4,
) -> list[str]:
    keywords = set(extract_keywords(query_text))
    if not keywords:
        return []

    scored_chunks: list[tuple[int, int, str]] = []
    for index, chunk in enumerate(chunks):
        chunk_keywords = extract_keywords(chunk)
        if not chunk_keywords:
            continue

        overlap_score = sum(1 for keyword in chunk_keywords if keyword in keywords)
        if overlap_score == 0:
            continue

        scored_chunks.append((overlap_score, -index, chunk))

    scored_chunks.sort(reverse=True)
    return deduplicate_chunks([chunk for _, _, chunk in scored_chunks[:limit]])


def build_context_from_chunks(chunks: list[str], limit: int = 4) -> str:
    return "\n\n".join(deduplicate_chunks(chunks[:limit]))


def is_small_document(chunks: list[str]) -> bool:
    return (
        len(chunks) <= SMALL_DOCUMENT_CHUNK_COUNT
        or sum(len(chunk) for chunk in chunks) <= SMALL_DOCUMENT_CHAR_COUNT
    )


async def get_chunks(document_text: str) -> list[str]:
    cleaned_document = normalize_text(document_text)
    chunk_docs = splitter.create_documents([cleaned_document])
    return deduplicate_chunks([doc.page_content for doc in chunk_docs])


async def create_collection(document_text: str) -> str:
    cleaned_document = normalize_text(document_text)
    if not cleaned_document:
        raise ValueError("No text content was found in the uploaded resource.")

    chunks = await get_chunks(cleaned_document)
    if not chunks:
        raise ValueError("No text content was found in the uploaded resource.")

    session_id = str(uuid.uuid4())

    with lock:
        session_chunks_map[session_id] = chunks

    ids = list(range(len(chunks)))
    client.recreate_collection(
        collection_name=session_id,
        vectors_config=client.get_fastembed_vector_params(),
    )
    client.add(collection_name=session_id, documents=chunks, ids=ids)

    return session_id


async def get_information(
    session_id: str,
    query_text: str,
    include_all: bool = False,
    limit: int = 4,
) -> str:
    chunks = session_chunks_map.get(session_id, [])
    if not chunks:
        return ""

    if include_all or is_small_document(chunks):
        return build_context_from_chunks(chunks, limit=max(limit, len(chunks)))

    retrieved_chunks: list[str] = []
    try:
        search_result = client.query(
            collection_name=session_id,
            query_text=query_text,
            limit=limit,
        )
        retrieved_chunks = deduplicate_chunks(
            [hit.document for hit in search_result if hit.document]
        )
    except Exception:
        retrieved_chunks = []

    lexical_chunks = rank_chunks_by_keywords(chunks, query_text, limit=limit)
    merged_chunks = deduplicate_chunks(retrieved_chunks + lexical_chunks)

    if not merged_chunks:
        merged_chunks = chunks[:limit]

    return build_context_from_chunks(merged_chunks, limit=limit)


async def delete_session_id(session_id: str) -> None:
    delete_session_id_from_map(session_id)

    with lock:
        session_chunks_map.pop(session_id, None)

    try:
        client.delete_collection(collection_name=session_id)
    except Exception:
        pass


getChunks = get_chunks
createCollection = create_collection
getInformation = get_information
deleteSessionId = delete_session_id
