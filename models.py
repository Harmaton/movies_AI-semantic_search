from typing import Optional
from pydantic import BaseModel

class SemanticSearchRequest(BaseModel):
    query: str
    limit: Optional[int] = 100

class SemanticSearchResponse(BaseModel):
    query: str
    answer: str

class HealthResponse(BaseModel):
    status: str
    kb_status: str
    kb_row_count: Optional[int] = None