# app/main.py
import logging
import os
import tempfile
from fastapi import FastAPI, Request, UploadFile, File, Form, status
from fastapi.responses import JSONResponse
from fastapi.exceptions import RequestValidationError

from .models import ProcessResponse
from .errors import ServiceException
from .strategies.extractors import DoclingExtractor, PyMuPDFExtractor
from .strategies.chunkers import RecursiveChunker, CustomSentenceChunker, OllamaSemanticChunker

# --- تنظیمات اولیه ---
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)
app = FastAPI(
    title="Advanced Document Processor Service",
    description="سرویسی برای پردازش اسناد با استراتژی‌های قابل انتخاب و مدیریت خطای پیشرفته.",
    version="2.0"
)

# --- دیکشنری استراتژی‌ها ---
EXTRACTORS = {
    "docling": DoclingExtractor(),
    "pypdf": PyMuPDFExtractor(),
}
CHUNKERS = {
    "recursive": RecursiveChunker(),
    "custom_sentence": CustomSentenceChunker(),
    "ollama_semantic": OllamaSemanticChunker(),
}

# --- مدیریت‌کننده‌های استثنا (Exception Handlers) ---

@app.exception_handler(ServiceException)
async def service_exception_handler(request: Request, exc: ServiceException):
    """خطاهای سفارشی و قابل پیش‌بینی سرویس را مدیریت می‌کند."""
    return JSONResponse(
        status_code=exc.status_code,
        content={"success": False, "error": {"code": exc.error_code, "message": exc.message}},
    )

@app.exception_handler(RequestValidationError)
async def validation_exception_handler(request: Request, exc: RequestValidationError):
    """خطاهای مربوط به ورودی نامعتبر را با ساختار استاندارد مدیریت می‌کند."""
    clean_errors = [f"Field '{' -> '.join(map(str, e['loc'][1:]))}': {e['msg']}" for e in exc.errors()]
    return JSONResponse(
        status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
        content={
            "success": False,
            "error": {
                "code": 20000, # کد خطای عمومی برای ورودی نامعتبر
                "message": "Invalid input provided.",
                "details": clean_errors,
            },
        },
    )

@app.exception_handler(Exception)
async def generic_exception_handler(request: Request, exc: Exception):
    """تمام خطاهای پیش‌بینی نشده دیگر را مدیریت می‌کند."""
    logger.error(f"An unexpected error occurred: {exc}", exc_info=True)
    return JSONResponse(
        status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
        content={
            "success": False,
            "error": {
                "code": 99999,
                "message": "An internal server error occurred.",
            },
        },
    )

# --- اندپوینت اصلی ---
@app.post("/v1/documents/process/", response_model=ProcessResponse)
async def process_document(
    file: UploadFile = File(...),
    extractor_strategy: str = Form("docling"),
    chunker_strategy: str = Form("recursive")
):
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
        text = extractor.extract(temp_file_path)

        chunker = CHUNKERS[chunker_strategy]
        chunks = chunker.chunk(text)

    except Exception:
        # تمام خطاهای داخلی (مثلاً خطای Docling یا Ollama) توسط generic_exception_handler گرفته می‌شوند
        # و یک خطای ۵۰۰ استاندارد برمی‌گردانند.
        raise # خطا را دوباره پرتاب می‌کنیم تا مدیریت‌کننده عمومی آن را بگیرد
    finally:
        if temp_file_path and os.path.exists(temp_file_path):
            os.remove(temp_file_path)

    return ProcessResponse(
        chunks=chunks,
        total_chunks=len(chunks),
        extractor_used=extractor_strategy,
        chunker_used=chunker_strategy
    )