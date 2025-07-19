# app/strategies/extractors.py
from docling.document_converter import DocumentConverter
from langchain_community.document_loaders import PyMuPDFLoader
from langchain_core.documents import Document
from typing import List
from .base import ExtractorStrategy

class DoclingExtractor(ExtractorStrategy):
    """استراتژی استخراج با Docling. (توجه: این روش شماره صفحه را برنمی‌گرداند)"""
    def extract(self, file_path: str) -> List[Document]:
        converter = DocumentConverter()
        result = converter.convert(file_path)
        # چون Docling یک متن یکپارچه می‌دهد، ما آن را در یک داکیومنت با صفحه ۰ قرار می‌دهیم
        return [Document(page_content=result.document.export_to_markdown(), metadata={"page": 0})]

class PyMuPDFExtractor(ExtractorStrategy):
    """استراتژی استخراج با PyMuPDF که شماره صفحه را به متادیتا اضافه می‌کند."""
    def extract(self, file_path: str) -> List[Document]:
        loader = PyMuPDFLoader(file_path)
        # متد load() به صورت خودکار برای هر صفحه یک داکیومنت با متادیتای شماره صفحه می‌سازد
        return loader.load()