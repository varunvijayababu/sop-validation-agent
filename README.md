# SOP Validation Agent

A Retrieval-Augmented Generation (RAG) based SOP Validation Agent built using FastAPI, Qdrant Cloud, Sentence Transformers, and Groq LLM.

## Features

- Upload reference SOP
- Upload user SOP
- Semantic retrieval using Qdrant
- SOP validation using LLM
- JSON output

## Tech Stack

- Python
- FastAPI
- Qdrant Cloud
- Sentence Transformers
- Groq API
- Swagger UI

## Workflow

Reference SOP
→ Chunking
→ Embedding
→ Qdrant Storage

User SOP
→ Embedding
→ Retrieval
→ LLM Validation
→ STATUS + COMMENTS

## Output Format

```json
[
  {
    "STATUS": "ACCEPT",
    "COMMENTS": "Detailed validation comments"
  }
]
```

## Run Locally

```bash
pip install -r requirements.txt

python -m uvicorn app.main:app --reload
```

Open:

http://127.0.0.1:8000/docs