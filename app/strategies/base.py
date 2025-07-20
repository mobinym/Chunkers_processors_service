# app/strategies/base.py
from abc import ABC, abstractmethod
from typing import List
from langchain_core.documents import Document

class ExtractorStrategy(ABC):
    @abstractmethod
    def extract(self, file_path: str) -> List[Document]:
        pass

class ChunkerStrategy(ABC):
    @abstractmethod
    def chunk(self, documents: List[Document]) -> List[Document]:
        pass