from pydantic import BaseModel
from typing import List, Optional

class SearchQuery(BaseModel):
    query: str
    top_k: int = 5  # This should work fine

class SearchResult(BaseModel):
    title: str
    content: str
    score: float
    page_id: int

class TopicRequest(BaseModel):
    topic: str

class Document(BaseModel):
    title: str
    content: str
    page_id: int
    chunk_id: int