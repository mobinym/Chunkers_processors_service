# app/strategies/extractors.py
from docling.document_converter import DocumentConverter
from langchain_community.document_loaders import PyMuPDFLoader
from langchain_core.documents import Document
from typing import List
from .base import ExtractorStrategy

class DoclingExtractor(ExtractorStrategy):
    def extract(self, file_path: str) -> List[Document]:
        converter = DocumentConverter()
        result = converter.convert(file_path)
        return [Document(page_content=result.document.export_to_markdown(), metadata={"page": 0})]

class PyMuPDFExtractor(ExtractorStrategy):
    def extract(self, file_path: str) -> List[Document]:
        loader = PyMuPDFLoader(file_path)
        return loader.load()