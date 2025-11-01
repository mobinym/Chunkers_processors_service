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
# from langchain.text_splitter import SentenceTransformersTokenTextSplitter
from langchain_text_splitters import SentenceTransformersTokenTextSplitter
from transformers import AutoTokenizer




logger = logging.getLogger(__name__)

class RecursiveChunker(ChunkerStrategy):
    def __init__(self, chunk_size: int = 800, chunk_overlap: int = 150):
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
    def __init__(self, model_name: str = "gemma3:4b", base_url: str = "http://services.aiopt.io:11434" , chunk_size: int = 1000, chunk_overlap: int = 200):
        self.model = OllamaLLM(model=model_name, base_url=base_url)
        # self.prompt = ChatPromptTemplate.from_template(
        #     "You are an expert in identifying semantic meaning of text. "
        #     "You wrap each chunk in <<<>>>.\n\n"
        #     "Example:\n"
        #     "Text: \"The curious cat perched on the windowsill, its eyes wide as it watched the fluttering birds outside. "
        #     "With a swift leap, it was on the ground, stealthily making its way towards the door. "
        #     "Suddenly, a noise startled it, causing the cat to freeze in place.\"\n"
        #     "Wrapped:\n"
        #     "<<<The curious cat perched on the windowsill, its eyes wide as it watched the fluttering birds outside.>>>\n"
        #     "<<<With a swift leap, it was on the ground, stealthily making its way towards the door.>>>\n"
        #     "<<<Suddenly, a noise startled it, causing the cat to freeze in place.>>>\n\n"
        #     "Now, process the following text:\n\n"
        #     "{paragraph}"
        # )
        self.prompt = ChatPromptTemplate.from_template(
    "You are an expert text analyst specializing in creating broad, high-level summaries. Your task is to consolidate text into the fewest possible semantically coherent chunks.\n\n"
    "Follow these rules:\n"
    "1. **Maximize Cohesion:** Group as many related sentences as possible into a single, comprehensive chunk. Each chunk should cover a broad theme or a complete section of an argument.\n"
    "2. **Major Topic Shifts Only:** Only create a new chunk when the text makes a clear and significant pivot to a completely new topic. Do not split on minor sub-topics or concluding sentences if they are related to the main theme of the chunk.\n"
    "3. **Prioritize Large Chunks:** Your primary goal is to produce large, all-encompassing chunks. Avoid creating small or medium-sized chunks if the ideas can be logically grouped.\n"
    "4. **Output Format:** Wrap each complete chunk in <<< and >>>.\n\n" \
    "5. *Preserve Original Language:** The output chunks MUST be in the same language as the input text. Do not translate the content between languages (e.g., from Persian to English or English to Persian).\n"

    "--- EXAMPLE ---\n"
    "Text: \"Project Titan faced significant hurdles. The engineering team struggled with battery efficiency, a core component of the design. "
    "Simultaneously, the software division reported delays in the core OS development due to unforeseen bugs. "
    "These two issues created a bottleneck that threatened the entire launch schedule.\"\n\n"
    "Wrapped:\n"
    "<<<Project Titan faced significant hurdles. The engineering team struggled with battery efficiency, a core component of the design. "
    "Simultaneously, the software division reported delays in the core OS development due to unforeseen bugs. "
    "These two issues created a bottleneck that threatened the entire launch schedule.>>>\n"
    "--- END EXAMPLE ---\n\n"

    "Now, process the following text based on these rules to create the largest possible coherent chunks:\n\n"
    "{paragraph}"
)


        self.output_parser = StrOutputParser()
        self.fallback_chunker = RecursiveChunker(chunk_size=chunk_size, chunk_overlap=chunk_overlap)
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
                    logger.warning(f"Ollama model did not return valid chunks for page {doc.metadata.get('page', 'N/A')}. Using the RecursiveChunker .")
                    fallback_chunks = self.fallback_chunker.chunk([doc])
                    final_docs.extend(fallback_chunks)
            except Exception as e:
                logger.error(f"Error calling Ollama chain for page {doc.metadata.get('page', 'N/A')}: {e}. Using the whole page as a fallback.")
                final_docs.append(doc)
                
        return final_docs
    

class OllamaSemanticChunkerPersian(ChunkerStrategy):
    def __init__(self, model_name: str = "gemma3:4b", base_url: str = "http://services.aiopt.io:11434" , chunk_size: int = 1000, chunk_overlap: int = 200):
        self.model = OllamaLLM(model=model_name, base_url=base_url)
  
        self.prompt = ChatPromptTemplate.from_template(
    "شما یک تحلیلگر متخصص متن هستید که در ایجاد خلاصه‌های کلی و سطح بالا تخصص دارید. وظیفه شما این است که متن را در کمترین تعداد ممکن از قطعات منسجم معنایی ادغام کنید.\n\n"
    "این قوانین را دنبال کنید:\n"
    "۱. **به حداکثر رساندن انسجام:** تا جایی که ممکن است جملات مرتبط را در یک قطعه واحد و جامع گروه‌بندی کنید. هر قطعه باید یک موضوع کلی یا بخش کاملی از یک استدلال را پوشش دهد.\n"
    "۲. **فقط در صورت تغییرات اساسی موضوع:** تنها زمانی یک قطعه جدید ایجاد کنید که متن به وضوح و به طور قابل توجهی به یک موضوع کاملاً جدید تغییر مسیر دهد. بر اساس موضوعات فرعی جزئی یا جملات نتیجه‌گیری، اگر به موضوع اصلی قطعه مرتبط هستند، تقسیم‌بندی نکنید.\n"
    "۳. **اولویت با قطعات بزرگ:** هدف اصلی شما تولید قطعات بزرگ و جامع است. از ایجاد قطعات کوچک یا متوسط در صورتی که بتوان ایده‌ها را به طور منطقی گروه‌بندی کرد، خودداری کنید.\n"
    "۴. **فرمت خروجی:** هر قطعه کامل را بین <<< و >>> قرار دهید.\n\n"
    "۵. **حفظ زبان اصلی:** قطعات خروجی باید الزاماً به همان زبان متن ورودی باشند. محتوا را بین زبان‌ها ترجمه نکنید (مثلاً از فارسی به انگلیسی یا از انگلیسی به فارسی).\n"

    "--- مثال ---\n"
    "متن: «پروژه تایتان با موانع قابل توجهی روبرو شد. تیم مهندسی با بازدهی باتری، که یکی از اجزای اصلی طراحی بود، مشکل داشت. "
    "همزمان، بخش نرم‌افزار به دلیل باگ‌های پیش‌بینی نشده، تأخیر در توسعه سیستم‌عامل اصلی را گزارش داد. "
    "این دو مشکل یک گلوگاه ایجاد کردند که کل برنامه عرضه را تهدید می‌کرد.»\n\n"
    "خروجی:\n"
    "<<<پروژه تایتان با موانع قابل توجهی روبرو شد. تیم مهندسی با بازدهی باتری، که یکی از اجزای اصلی طراحی بود، مشکل داشت. "
    "همزمان، بخش نرم‌افزار به دلیل باگ‌های پیش‌بینی نشده، تأخیر در توسعه سیستم‌عامل اصلی را گزارش داد. "
    "این دو مشکل یک گلوگاه ایجاد کردند که کل برنامه عرضه را تهدید می‌کرد.>>>\n"
    "--- پایان مثال ---\n\n"

    "اکنون، متن زیر را بر اساس این قوانین پردازش کنید تا بزرگترین قطعات منسجم ممکن را ایجاد نمایید:\n\n"
    "{paragraph}"
)

        self.output_parser = StrOutputParser()
        self.fallback_chunker = RecursiveChunker(chunk_size=chunk_size, chunk_overlap=chunk_overlap)
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
                    logger.warning(f"Ollama model did not return valid chunks for page {doc.metadata.get('page', 'N/A')}. Using the RecursiveChunker .")
                    fallback_chunks = self.fallback_chunker.chunk([doc])
                    final_docs.extend(fallback_chunks)
            except Exception as e:
                logger.error(f"Error calling Ollama chain for page {doc.metadata.get('page', 'N/A')}: {e}. Using the whole page as a fallback.")
                final_docs.append(doc)
                
        return final_docs
    
    

# class TokenTextChunker(ChunkerStrategy):
#     def __init__(self, chunk_size: int = 256, chunk_overlap: int = 32, model_name: str = "BAAI/bge-m3"):
#         self.splitter = SentenceTransformersTokenTextSplitter(
#             chunk_size=chunk_size,
#             chunk_overlap=chunk_overlap,
#             model_name=model_name
#         )

#     def chunk(self, documents: List[Document]) -> List[Document]:
#         return self.splitter.split_documents(documents)
#--------------------------------------------------------------------------------------------------------

class TokenTextChunker(ChunkerStrategy):
    def __init__(self, chunk_size: int = 300, chunk_overlap: int = 30, model_name: str = "BAAI/bge-m3"):
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)

    def chunk(self, documents: List[Document]) -> List[Document]:
        all_chunks = []

        for doc in documents:
            text = doc.page_content
            tokens = self.tokenizer.encode(text, add_special_tokens=False)
            total_tokens = len(tokens)

            start = 0
            while start < total_tokens:
                end = min(start + self.chunk_size, total_tokens)
                token_chunk = tokens[start:end]
                chunk_text = self.tokenizer.decode(token_chunk)

                all_chunks.append(Document(
                    page_content=chunk_text,
                    metadata=doc.metadata
                ))

                start += self.chunk_size - self.chunk_overlap

        return all_chunks