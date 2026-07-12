# AI SOP Validation Agent

A FastAPI-based SOP validation service that compares an uploaded SOP against an uploaded reference guideline using a Retrieval-Augmented Generation (RAG) workflow.

The system:
- ingests a reference guideline (`.pdf` or `.docx`)
- extracts section-wise reference content
- assigns section importance weights
- stores reference sections in Qdrant
- retrieves relevant guideline sections for an uploaded SOP
- validates the SOP with a pluggable LLM provider
- computes weighted section scores and returns a structured JSON result

## Key Features

- Upload reference guideline in `PDF` or `DOCX`
- Upload SOP in `PDF` or `DOCX`
- PDF text extraction with page-level processing
- DOCX-to-PDF conversion before parsing
- Embedded image caption extraction for PDF pages
- Section-based guideline chunking using `###` headers
- Section importance weighting using an LLM
- Vector storage and retrieval with Qdrant
- Provider-neutral LLM layer with support for:
  - Groq
  - OpenAI
  - Google Gemini
  - Ollama
- Strict JSON validation and normalization of model responses
- Weighted SOP scoring with per-section breakdown
- Detailed endpoint with token usage reporting
- Application logging to both console and `logs/app.log`

## Project Structure

```text
app/
  agents/
    llm_validator.py
  api/
    upload.py
    validate.py
  llm/
    base.py
    config.py
    factory.py
    groq_adapter.py
    openai_adapter.py
    gemini_adapter.py
    ollama_adapter.py
  parser/
    pdf_parser.py
    docx_to_pdf.py
    image_captioner.py
  rag/
    embedder.py
    qdrant_client.py
    retriever.py
    section_ranker.py
    text_splitter.py
    vector_store.py
  main.py
```

## End-to-End Workflow

### 1. Upload Reference Guideline
`POST /upload-standard`

Flow:
1. Save uploaded file
2. If DOCX, convert to PDF
3. Extract page-level text from PDF
4. Extract image captions from embedded images
5. Split guideline into sections using `###` headings
6. Generate section importance weights
7. Embed each section
8. Store sections in Qdrant with:
   - section title
   - page number
   - text
   - weight

### 2. Validate SOP
`POST /validate-sop`  
`POST /validate-sop-detailed`

Flow:
1. Save uploaded SOP
2. If DOCX, convert to PDF
3. Extract SOP text from PDF
4. Embed the SOP text
5. Retrieve relevant guideline sections from Qdrant
6. Send SOP text + retrieved reference sections to the selected LLM provider
7. Validate and normalize LLM JSON output
8. Compute:
   - section-level status
   - weighted score
   - score breakdown
   - comments
   - reference section
9. Return the final validation response

## LLM Provider Support

The application supports multiple LLM providers through a common abstraction layer.

Select the provider using:

```env
LLM_PROVIDER=groq
```

Supported values:
- `groq`
- `openai`
- `gemini`
- `ollama`

The validation workflow and API response structure remain the same regardless of provider. Only the underlying model changes.

## API Endpoints

### `GET /`
Health/root endpoint.

Response:
```json
{
  "message": "SOP Validation Agent Running"
}
```

### `POST /upload-standard`
Uploads and indexes the reference guideline.

Accepted files:
- `.pdf`
- `.docx`

Success response:
```json
{
  "message": "Reference SOP uploaded"
}
```

### `POST /validate-sop`
Validates an SOP and returns the standard response.

Accepted files:
- `.pdf`
- `.docx`

Response format:
```json
[
  {
    "STATUS": "MODIFY",
    "SCORE": 30.58,
    "COMMENTS": "The SOP partially addresses several key areas...",
    "REFERENCE": "Characteristics of a High-Quality IPC SOP (Page 6)"
  }
]
```

### `POST /validate-sop-detailed`
Validates an SOP and returns the detailed response.

Response format:
```json
[
  {
    "STATUS": "MODIFY",
    "SCORE": 30.58,
    "SCORE_BREAKDOWN": {
      "Roles and Responsibilities": {
        "STATUS": "PARTIAL",
        "WEIGHT": 8.89,
        "SCORE": 4.45
      }
    },
    "COMMENTS": "The SOP partially addresses several key areas...",
    "REFERENCE": "Characteristics of a High-Quality IPC SOP (Page 6)",
    "TOKEN_COUNT": {
      "INPUT": 7612,
      "OUTPUT": 408,
      "TOTAL": 8020
    }
  }
]
```

## Validation Status Semantics

Top-level SOP result:
- `ACCEPT`
- `MODIFY`
- `REJECT`
- `SYSTEM_ERROR` in failure scenarios

Section-level statuses used internally for scoring:
- `COMPLIANT`
- `PARTIAL`
- `MISSING`

Scoring logic:
- `COMPLIANT` = full section weight
- `PARTIAL` = 50% of section weight
- `MISSING` = 0

The final SOP score is the sum of normalized weighted section scores.

## Reference Guideline Formatting Requirement

The reference guideline is split into sections using headings that match:

```text
### Section Title
```

This is required for section extraction. If the uploaded reference guideline does not contain `###` section headers after parsing, section splitting will fail.

## Environment Variables

Create a `.env` file and configure the provider(s) you plan to use.

### Provider Selection
```env
LLM_PROVIDER=groq
```

### Groq
```env
GROQ_API_KEY=your_groq_api_key
GROQ_MODEL=llama-3.3-70b-versatile
```

### OpenAI
```env
OPENAI_API_KEY=your_openai_api_key
OPENAI_MODEL=gpt-4o
```

### Gemini
```env
GEMINI_API_KEY=your_gemini_api_key
GEMINI_MODEL=gemini-2.0-flash
```

### Ollama
```env
OLLAMA_BASE_URL=http://localhost:11434
OLLAMA_MODEL=llama3
```

## Qdrant

The application currently connects to a local Qdrant instance from code:

- host: `localhost`
- port: `6333`

Make sure Qdrant is running locally before uploading a reference guideline or validating an SOP.

## Logs

Logs are written to:
- console output
- `logs/app.log`

Logging includes:
- file upload events
- parsing progress
- retrieval flow
- LLM validation flow
- scoring and error details

## Run Locally

Install dependencies:

```bash
pip install -r requirements.txt
```

Start the API:

```bash
python -m uvicorn app.main:app --reload
```

Open Swagger UI:

```text
http://127.0.0.1:8000/docs
```

## Current Tech Stack

- Python
- FastAPI
- Uvicorn
- PyMuPDF
- Sentence Transformers
- Qdrant
- Groq API
- OpenAI API
- Google Gemini API
- Ollama
- python-dotenv

## Notes

- The detailed endpoint includes token usage from the validation LLM call.
- Provider-specific request formatting is handled inside adapter modules.
- The application uses a common provider-neutral LLM interface internally.
- Differences in validation results across providers are expected to come from model behavior, not from different application logic.
```
