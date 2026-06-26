import pdfplumber
from typing import Annotated

from fastapi import APIRouter, File, Form, HTTPException, Path, UploadFile

from app.models.query import QueryRequest
from app.services.LLM import handle_user_query
from app.services.vectorDB import create_collection, delete_session_id

router = APIRouter(prefix="/sessions", tags=["sessions"])


async def create_session_from_text(document_text: str) -> str:
    return await create_collection(document_text)


@router.post("/upload-file")
async def upload_file_data(
    file: Annotated[UploadFile, File(description="Upload a PDF or TXT file.")]
):
    file_name = file.filename or ""
    if not file_name.endswith((".pdf", ".txt")):
        raise HTTPException(status_code=400, detail="file must be either .pdf or .txt")

    document_text = ""
    try:
        if file_name.endswith(".pdf"):
            with pdfplumber.open(file.file) as pdf:
                document_text = "\n".join(
                    (page.extract_text() or "") for page in pdf.pages
                )
        else:
            document_text = (await file.read()).decode("utf-8")

        session_id = await create_session_from_text(document_text)
    except ValueError as error:
        raise HTTPException(status_code=400, detail=str(error)) from error

    return {"session_id": session_id}


@router.post("/upload-text")
async def upload_text_data(
    text: Annotated[str, Form(description="Provide raw text to upload.")]
):
    try:
        session_id = await create_session_from_text(text)
    except ValueError as error:
        raise HTTPException(status_code=400, detail=str(error)) from error

    return {"session_id": session_id}


@router.post("/query")
async def query_session(request: QueryRequest):
    result = await handle_user_query(request.query, request.session_id)
    # `trace` exposes which Advanced RAG stages ran (query rewrite, sub-query,
    # HyDE, rerank, corrective RAG, LLM judge) so the client can show them.
    return {"response": result["answer"], "trace": result["trace"]}


@router.delete("/{session_id}")
async def delete_session(
    session_id: Annotated[str, Path(description="Session ID to delete.")]
):
    await delete_session_id(session_id)
    return {"message": "success!"}
