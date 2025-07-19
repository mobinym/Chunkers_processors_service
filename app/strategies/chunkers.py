# app/strategies/chunkers.py
import re
from typing import List
import logging
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_ollama import OllamaLLM # ✅ استفاده از کلاس جدید و صحیح
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnablePassthrough
from .base import ChunkerStrategy

logger = logging.getLogger(__name__)

class RecursiveChunker(ChunkerStrategy):
    """استراتژی چانک کردن بازگشتی با استفاده از LangChain."""
    def __init__(self, chunk_size: int = 1000, chunk_overlap: int = 200):
        self.splitter = RecursiveCharacterTextSplitter(
            chunk_size=chunk_size,
            chunk_overlap=chunk_overlap,
            separators=["\n\n", "\n", ". ", " ", ""]
        )

    def chunk(self, text: str) -> List[str]:
        return self.splitter.split_text(text)

class CustomSentenceChunker(ChunkerStrategy):
    """استراتژی چانک کردن سفارشی شما بر اساس جملات با همپوشانی."""
    def __init__(self, max_chars: int = 900, overlap: int = 100):
        self.max_chars = max_chars
        self.overlap = overlap
        self._SENT_BOUNDARY = re.compile(r"(?<=[.!?؟])\s+")
        self._WS_CLEAN = re.compile(r"\s+")

    def chunk(self, text: str) -> List[str]:
        clean = self._WS_CLEAN.sub(" ", text).strip()
        sentences = self._SENT_BOUNDARY.split(clean)
        
        chunks, buff, length = [], [], 0
        for s in sentences:
            if not s:
                continue
            s_len = len(s)
            if length + s_len <= self.max_chars:
                buff.append(s)
                length += s_len
            else:
                chunk = " ".join(buff)
                chunks.append(chunk)
                tail = chunk[-self.overlap:]
                buff = [tail, s] if tail else [s]
                length = len(" ".join(buff))

        if buff:
            chunks.append(" ".join(buff))

        return [c for c in chunks if c]

# ✅ بازنویسی شده بر اساس کد موفق شما با استفاده از LangChain
class OllamaSemanticChunker(ChunkerStrategy):
    def __init__(self, model_name: str = "gemma3:4b", base_url: str = "http://services.aiopt.io:11434"):
        self.model = OllamaLLM(model=model_name, base_url=base_url)
        self.prompt_template = (
            "You are an expert in identifying semantic meaning of text. "
            "You wrap each chunk in <<<>>>.\n\n"
            "Example:\n"
            "Text: \"The curious cat perched on the windowsill, its eyes wide as it watched the fluttering birds outside. "
            "With a swift leap, it was on the ground, stealthily making its way towards the door. "
            "Suddenly, a noise startled it, causing the cat to freeze in place.\"\n"
            "Wrapped:\n"
            "<<<The curious cat perched on the windowsill, its eyes wide as it watched the fluttering birds outside.>>>\n"
            "<<<With a swift leap, it was on the ground, stealthily making its way towards the door.>>>\n"
            "<<<Suddenly, a noise startled it, causing the cat to freeze in place.>>>\n\n"
            "Now, process the following text:\n\n"
            "{paragraph}"
        )

        self.prompt = ChatPromptTemplate.from_template(self.prompt_template)
        self.output_parser = StrOutputParser()
        self.chain = (
            {"paragraph": RunnablePassthrough()}
            | self.prompt
            | self.model
            | self.output_parser
        )

    def chunk(self, text: str) -> List[str]:
        preliminary_chunks = text.split("\n\n")
        final_chunks = []
        for i, paragraph in enumerate(preliminary_chunks):
            if not paragraph.strip():
                continue
            
            logger.info(f"Processing paragraph {i+1}/{len(preliminary_chunks)} with Ollama...")
            try:
                response = self.chain.invoke(paragraph)
                newly_found_chunks = re.findall(r'<<<(.*?)>>>', response, re.DOTALL)
                
                if newly_found_chunks:
                    final_chunks.extend([chunk.strip() for chunk in newly_found_chunks if chunk.strip()])
                else:
                    logger.warning(f"Ollama model did not return valid chunks for paragraph {i+1}. Using the whole paragraph as a chunk.")
                    final_chunks.append(paragraph.strip())
            except Exception as e:
                logger.error(f"Error calling Ollama chain for paragraph {i+1}: {e}. Using the whole paragraph as a fallback.")
                final_chunks.append(paragraph.strip())

        return final_chunks