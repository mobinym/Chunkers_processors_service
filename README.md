
-----

# Advanced Document Processor Service 📄⚙️

A standalone FastAPI service designed to extract text from various document formats and split it into clean, metadata-rich chunks using a flexible strategy pattern.

[](https://www.python.org/downloads/)
[](https://fastapi.tiangolo.com/)
[](https://opensource.org/licenses/MIT)

This service acts as a powerful pre-processing engine for RAG (Retrieval-Augmented Generation) pipelines. It accepts a file and allows the user to select from various strategies for both text extraction and text chunking, ensuring the output is perfectly tailored to the needs of the downstream embedding and language models.

-----

## ✨ Features

  * **Modular Strategy Pattern:** Easily extend the service by adding new extraction or chunking methods without changing the core API logic.
  * **Multiple Text Extractors:**
      * `docling`: A powerful library for converting various formats (PDF, DOCX, etc.) into clean Markdown.
      * `pypdf`: A fast and reliable method for extracting raw text and page numbers directly from PDF files using `PyMuPDFLoader`.
  * **Multiple Chunking Strategies:**
      * `recursive`: A robust character-based splitter from LangChain.
      * `custom_sentence`: A custom sentence-based splitter with configurable overlap.
      * `ollama_semantic`: An advanced LLM-based chunker that splits text based on semantic meaning.
  * **Metadata Preservation:** Automatically extracts and preserves the page number for each chunk, which is critical for providing sources in RAG responses.
  * **Standardized Error Handling:** All errors (invalid strategy, file processing issues, etc.) are returned in a consistent and machine-readable JSON format.

-----

## 🛠️ Setup and Installation

Follow these steps to get the service running locally.

### Prerequisites

  * Python 3.10 or higher
  * `pip` and `venv`

### Installation

1.  Create a project folder and place the `app` directory, `Dockerfile`, and `requirements.txt` inside it.

2.  Create and activate a virtual environment:

    ```bash
    # Create the virtual environment
    python -m venv venv

    # Activate on Windows
    .\venv\Scripts\activate

    # Or activate on macOS/Linux
    source venv/bin/activate
    ```

3.  Install the dependencies:

    ```bash
    pip install -r requirements.txt
    ```

-----

## 🚀 Running the Service

Once the installation is complete, run the following command to start the web server. This service does not require pre-loading any models and should start quickly.

```bash
uvicorn app.main:app --host 0.0.0.0 --port 8001
```

The service will be available at `http://localhost:8001`.

-----

## API Documentation & Usage

This service comes with automatic, interactive API documentation.

  * **Swagger UI:** [http://localhost:8001/docs](https://www.google.com/search?q=http://localhost:8001/docs)
  * **ReDoc:** [http://localhost:8001/redoc](https://www.google.com/search?q=http://localhost:8001/redoc)

### Endpoint: `POST /v1/documents/process/`

This is the main endpoint for processing documents. It accepts `multipart/form-data`.

#### Form Fields

  * `file` (File, **required**): The document file you want to process.
  * `extractor_strategy` (string, *optional*): The name of the text extraction strategy. Defaults to `"pypdf"`.
  * `chunker_strategy` (string, *optional*): The name of the text chunking strategy. Defaults to `"recursive"`.

### Available Strategies

#### Extractor Strategies

| Strategy Name | Description |
| :--- | :--- |
| `pypdf` | **(Default)** Fast and reliable text extraction from PDFs, includes page numbers. |
| `docling` | Powerful, converts various formats to Markdown but may not preserve page numbers. |

#### Chunker Strategies

| Strategy Name | Description |
| :--- | :--- |
| `recursive` | **(Default)** A robust, general-purpose character splitter. |
| `custom_sentence` | Your custom sentence-based splitter with overlap logic. |
| `ollama_semantic`| Advanced chunking based on semantic meaning using an LLM. |

#### Example `curl` Requests

**1. Basic request using default strategies (`pypdf` + `recursive`):**

```bash
curl -X POST "http://localhost:8001/v1/documents/process/" \
-F "file=@/path/to/your/document.pdf"
```

**2. Request with custom strategies (`pypdf` + `ollama_semantic`):**

```bash
curl -X POST "http://localhost:8001/v1/documents/process/" \
-F "file=@/path/to/your/document.pdf" \
-F "extractor_strategy=pypdf" \
-F "chunker_strategy=ollama_semantic"
```

#### Success Response

A successful request will return a `200 OK` status with a body structured like this:

```json
{
  "chunks": [
    {
      "page_content": "This is the first chunk from the first page...",
      "metadata": {
        "source": "document.pdf",
        "page": 0
      }
    },
    {
      "page_content": "This is the second chunk, still from the first page...",
      "metadata": {
        "source": "document.pdf",
        "page": 0
      }
    }
  ],
  "total_chunks": 2,
  "extractor_used": "pypdf",
  "chunker_used": "recursive"
}
```

-----

## ⚠️ Error Handling

All errors are returned in a standardized JSON format.

```json
{
  "success": false,
  "error": {
    "code": <error_code>,
    "message": "<human_readable_message>",
    "details": []
  }
}
```

### Error Codes

| Error Code | Message | HTTP Status |
| :--- | :--- | :--- |
| **20001** | `Extractor strategy '<name>' is not supported.` | `400 Bad Request` |
| **20002** | `Chunker strategy '<name>' is not supported.` | `400 Bad Request` |
| **20000** | `Invalid input provided.` (e.g., no file uploaded) | `422 Unprocessable Entity` |
| **99999** | `An internal server error occurred.` (e.g., `docling` fails) | `500 Internal Server Error` |