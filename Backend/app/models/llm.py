from pydantic import BaseModel

class Search(BaseModel):
    needs_search: bool
    search_query: str | None = None
