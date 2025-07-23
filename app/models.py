from pydantic import BaseModel
from typing import List, Dict, Any

class Chunk(BaseModel):
    chunk_content: str
    metadata: Dict[str, Any]

class ProcessResponse(BaseModel):
    file_metadata: Dict[str, Any]
    chunks: List[Chunk]
    total_chunks: int
    extractor_used: str
    chunker_used: str
