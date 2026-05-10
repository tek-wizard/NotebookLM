from pydantic import BaseModel

class QueryRequest(BaseModel):
    session_id: str
    query: str