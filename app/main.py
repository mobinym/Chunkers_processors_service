# app/main.py
import logging
import os
import tempfile
from fastapi import FastAPI, UploadFile, File, Form, HTTPException, Request, status
from fastapi.responses import JSONResponse
from fastapi.exceptions import RequestValidationError

from .models import ProcessResponse, Chunk
from .errors import ServiceException

from .strategies.extractors import DoclingExtractor, LangChainPyMuPDFExtractor , PyPDFExtractor , DocxExtractor
from .strategies.chunkers import RecursiveChunker, CustomSentenceChunker, OllamaSemanticChunker , OllamaSemanticChunkerPersian , TokenTextChunker

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)
app = FastAPI(
    title="Advanced Document Processor Service",
    description="سرویسی برای پردازش اسناد با استراتژی‌های قابل انتخاب و متادیتای صفحه.",
    version="0.1"
)

EXTRACTORS = {
    "docling": DoclingExtractor(),
    "pypdf": LangChainPyMuPDFExtractor(),
    "pdfFA" : PyPDFExtractor(),
    "docx"  : DocxExtractor(),
}
CHUNKERS = {
    "recursive": RecursiveChunker(),
    "custom_sentence": CustomSentenceChunker(),
    "ollama_semantic": OllamaSemanticChunker(),
    "ollama_semantic_p" : OllamaSemanticChunkerPersian(),
    "token_based" : TokenTextChunker(),
}

@app.exception_handler(ServiceException)
async def service_exception_handler(request: Request, exc: ServiceException):
    return JSONResponse(status_code=exc.status_code, content={"success": False, "error": {"code": exc.error_code, "message": exc.message}})

@app.exception_handler(RequestValidationError)
async def validation_exception_handler(request: Request, exc: RequestValidationError):
    clean_errors = [f"Field '{' -> '.join(map(str, e['loc'][1:]))}': {e['msg']}" for e in exc.errors()]
    return JSONResponse(
        status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
        content={"success": False, "error": {"code": 20000, "message": "Invalid input provided.", "details": clean_errors}},
    )

@app.exception_handler(Exception)
async def generic_exception_handler(request: Request, exc: Exception):
    logger.error(f"An unexpected error occurred: {exc}", exc_info=True)
    return JSONResponse(
        status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
        content={"success": False, "error": {"code": 99999, "message": "An internal server error occurred."}},
    )

@app.post("/v1/chn/chunking/", response_model=ProcessResponse)
async def process_document(
    file: UploadFile = File(...),
    extractor_strategy: str = Form("pypdf", description="Supported: docling, pypdf, pdfFA, docx" ),
    chunker_strategy: str = Form("token_based", description="Supported: recursive, custom_sentence, ollama_semantic, ollama_semantic_p, token_based")
):
    #check docx file for extractor strategies
    filename = file.filename.lower()

    is_word_file = filename.endswith(".docx") or filename.endswith(".doc")

    if is_word_file:
        if extractor_strategy != "docx":
            raise ServiceException(
                status_code=400,
                error_code=20003,
                message="For Word documents (.docx), the extractor_strategy must be 'docx'."
            )
        extractor_strategy = "docx"  

   
    elif extractor_strategy == "docx":
        raise ServiceException(
            status_code=400,
            error_code=20004,
            message="Extractor strategy 'docx' is only valid for Word documents."
        )
    
    if extractor_strategy not in EXTRACTORS:
        raise ServiceException(status_code=400, error_code=20001, message=f"Extractor strategy '{extractor_strategy}' is not supported.")
    if chunker_strategy not in CHUNKERS:
        raise ServiceException(status_code=400, error_code=20002, message=f"Chunker strategy '{chunker_strategy}' is not supported.")

    temp_file_path = None
    try:
        with tempfile.NamedTemporaryFile(delete=False, suffix=f"_{file.filename}") as temp_file:
            temp_file.write(await file.read())
            temp_file_path = temp_file.name
        
        extractor = EXTRACTORS[extractor_strategy]
        pages_as_docs = extractor.extract(temp_file_path)
       

        chunker = CHUNKERS[chunker_strategy]
        chunks_as_docs = chunker.chunk(pages_as_docs)
        # استخراج metadata عمومی فایل فقط از اولین صفحه
        file_metadata = {
            k: v for k, v in pages_as_docs[0].metadata.items()
            if k not in ["page", "source"]
        }

        # پاکسازی metadata تکراری از هر chunk
        # response_chunks = []
        # for doc in chunks_as_docs:
        #     chunk_metadata = {
        #         "page": doc.metadata.get("page"),
        #         "source": doc.metadata.get("source")
        #     }
        #     response_chunks.append(
        #         Chunk(page_content=doc.page_content, metadata=chunk_metadata)
        #     )
        response_chunks = []

        for doc in chunks_as_docs:
            chunk_metadata = {
                "page": doc.metadata.get("page")
            }
            response_chunks.append(
                Chunk(
                    chunk_content=doc.page_content,  # درست!
                    metadata=chunk_metadata
                )
            )

    except Exception as e:
        logger.error(f"An error occurred during processing: {e}", exc_info=True)
        raise
    finally:
        if temp_file_path and os.path.exists(temp_file_path):
            os.remove(temp_file_path)

    # response_chunks = [Chunk(page_content=doc.page_content, metadata=doc.metadata) for doc in chunks_as_docs]

    return ProcessResponse(
        file_metadata=file_metadata,
        chunks=response_chunks,
        total_chunks=len(response_chunks),
        extractor_used=extractor_strategy,
        chunker_used=chunker_strategy
    )