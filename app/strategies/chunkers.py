# document_processor/app/strategies/chunkers.py
import re
import logging
from typing import List
from langchain_ollama import OllamaLLM 
from langchain_core.documents import Document
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import RunnablePassthrough
from langchain_text_splitters import RecursiveCharacterTextSplitter
from .base import ChunkerStrategy
from langchain.text_splitter import SentenceTransformersTokenTextSplitter


logger = logging.getLogger(__name__)

class RecursiveChunker(ChunkerStrategy):
    def __init__(self, chunk_size: int = 1000, chunk_overlap: int = 200):
        self.splitter = RecursiveCharacterTextSplitter(
            chunk_size=chunk_size,
            chunk_overlap=chunk_overlap,
            separators=["\n\n", "\n", ". ", " ", ""]
        )

    def chunk(self, documents: List[Document]) -> List[Document]:
        return self.splitter.split_documents(documents)


class CustomSentenceChunker(ChunkerStrategy):
    def __init__(self, max_chars: int = 900, overlap: int = 100):
        self.max_chars = max_chars
        self.overlap = overlap
        self._SENT_BOUNDARY = re.compile(r"(?<=[.!?؟])\s+")
        self._WS_CLEAN = re.compile(r"\s+")

    def _chunk_text_logic(self, text: str) -> List[str]:
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

    def chunk(self, documents: List[Document]) -> List[Document]:
        final_docs = []
        for doc in documents:
            text_chunks = self._chunk_text_logic(doc.page_content)
            for chunk_content in text_chunks:
                new_doc = Document(
                    page_content=chunk_content,
                    metadata=doc.metadata.copy() 
                )
                final_docs.append(new_doc)
        return final_docs


class OllamaSemanticChunker(ChunkerStrategy):
    def __init__(self, model_name: str = "gemma3:4b", base_url: str = "http://services.aiopt.io:11434"):
        self.model = OllamaLLM(model=model_name, base_url=base_url)
        self.prompt = ChatPromptTemplate.from_template(
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
        self.output_parser = StrOutputParser()
        self.chain = (
            {"paragraph": RunnablePassthrough()}
            | self.prompt
            | self.model
            | self.output_parser
        )

    def chunk(self, documents: List[Document]) -> List[Document]:
        final_docs = []
        for doc in documents:
            paragraph = doc.page_content
            if not paragraph.strip():
                continue
            
            logger.info(f"Processing page {doc.metadata.get('page', 'N/A')} with Ollama...")
            try:
                response = self.chain.invoke(paragraph)
                newly_found_chunks = re.findall(r'<<<(.*?)>>>', response, re.DOTALL)
                
                if newly_found_chunks:
                    for chunk_content in newly_found_chunks:
                        if chunk_content.strip():
                            new_doc = Document(
                                page_content=chunk_content.strip(),
                                metadata=doc.metadata.copy() 
                            )
                            final_docs.append(new_doc)
                else:
                    logger.warning(f"Ollama model did not return valid chunks for page {doc.metadata.get('page', 'N/A')}. Using the whole page as a chunk.")
                    final_docs.append(doc)
            except Exception as e:
                logger.error(f"Error calling Ollama chain for page {doc.metadata.get('page', 'N/A')}: {e}. Using the whole page as a fallback.")
                final_docs.append(doc)
                
        return final_docs
    

class TokenTextChunker(ChunkerStrategy):
    def __init__(self, chunk_size: int = 256, chunk_overlap: int = 32, model_name: str = "BAAI/bge-m3"):
        self.splitter = SentenceTransformersTokenTextSplitter(
            chunk_size=chunk_size,
            chunk_overlap=chunk_overlap,
            model_name=model_name
        )

    def chunk(self, documents: List[Document]) -> List[Document]:
        return self.splitter.split_documents(documents)
