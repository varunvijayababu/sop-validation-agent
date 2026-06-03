from fastapi import APIRouter
from fastapi import UploadFile
from fastapi import File

from app.parser.pdf_parser import extract_pdf_text
from app.parser.docx_parser import extract_docx_text

from app.rag.retriever import retrieve_context
from app.agents.groq_validator import validate_sop

import os

router = APIRouter()

UPLOAD_DIR = "uploads"

@router.post("/validate-sop")
async def validate(
    file: UploadFile = File(...)
):

    file_path = os.path.join(
        UPLOAD_DIR,
        file.filename
    )

    with open(file_path, "wb") as f:
        f.write(await file.read())

    if file.filename.endswith(".pdf"):
        sop_text = extract_pdf_text(file_path)

    elif file.filename.endswith(".docx"):
        sop_text = extract_docx_text(file_path)

    else:
        return {"error": "Unsupported file"}

    reference_text = retrieve_context(
        sop_text
    )

    result = validate_sop(
        sop_text,
        reference_text
    )

    return result