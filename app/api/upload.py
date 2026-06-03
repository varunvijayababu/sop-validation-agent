from fastapi import APIRouter
from fastapi import UploadFile
from fastapi import File

from app.parser.pdf_parser import extract_pdf_text
from app.parser.docx_parser import extract_docx_text

from app.rag.vector_store import store_document

import os

router = APIRouter()

UPLOAD_DIR = "uploads"

os.makedirs(
    UPLOAD_DIR,
    exist_ok=True
)

@router.post("/upload-standard")
async def upload_standard(
    file: UploadFile = File(...)
):

    file_path = os.path.join(
        UPLOAD_DIR,
        file.filename
    )

    with open(file_path, "wb") as f:
        f.write(await file.read())

    if file.filename.endswith(".pdf"):
        text = extract_pdf_text(file_path)

    elif file.filename.endswith(".docx"):
        text = extract_docx_text(file_path)

    else:
        return {"error": "Unsupported file"}

    store_document(text)

    return {
        "message": "Reference SOP uploaded"
    }