# app/strategies/base.py
from abc import ABC, abstractmethod
from typing import List
from langchain_core.documents import Document

class ExtractorStrategy(ABC):
    """کلاس پایه برای استراتژی‌های استخراج متن."""
    @abstractmethod
    def extract(self, file_path: str) -> List[Document]:
        """یک مسیر فایل دریافت کرده و لیستی از آبجکت‌های Document (هر کدام برای یک صفحه) برمی‌گرداند."""
        pass

class ChunkerStrategy(ABC):
    """کلاس پایه برای استراتژی‌های قطعه‌قطعه کردن."""
    @abstractmethod
    def chunk(self, documents: List[Document]) -> List[Document]:
        """لیستی از اسناد (صفحات) را دریافت کرده و لیستی از اسناد کوچک‌تر (چانک‌ها) را برمی‌گرداند."""
        pass