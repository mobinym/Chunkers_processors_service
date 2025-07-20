# app/models.py
from pydantic import BaseModel
from typing import List, Dict, Any

class Chunk(BaseModel):
    page_content: str
    metadata: Dict[str, Any]

class ProcessResponse(BaseModel):
    chunks: List[Chunk]
    total_chunks: int
    extractor_used: str
    chunker_used: str