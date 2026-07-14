import uuid

from fastapi import APIRouter
from fastapi import UploadFile
from fastapi import File

from app.parser.pdf_parser import (
    extract_pdf_pages
)

from app.parser.docx_parser import (
    extract_docx_pages
)

from app.rag.vector_store import store_document

import os
import logging

logger = logging.getLogger(__name__)

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

    logger.info(
        f"Reference SOP upload received: {file.filename}"
    )

    try:

        file_path = os.path.join(
            UPLOAD_DIR,
            f"{uuid.uuid4()}_{file.filename}"
        )

        logger.info(
            f"Saving uploaded file to: {file_path}"
        )

        with open(file_path, "wb") as f:
            f.write(await file.read())

        filename = file.filename.lower()

        if filename.endswith(".pdf"):

            logger.info(
                "Detected PDF guideline"
            )

            pages = extract_pdf_pages(
                file_path
            )

            logger.info(
                f"Extracted {len(pages)} pages from PDF"
            )

            store_document(
                pages
            )

            logger.info(
                "Guideline stored successfully"
            )

            return {
                "message": "Reference SOP uploaded"
            }

        elif filename.endswith(".docx"):

            logger.info(
                "Detected DOCX guideline"
            )

            pages = extract_docx_pages(
                file_path
            )

            logger.info(
                f"Extracted {len(pages)} pages from DOCX"
            )

            store_document(
                pages
            )

            logger.info(
                "Guideline stored successfully"
            )

            return {
                "message": "Reference SOP uploaded"
            }

        else:

            logger.warning(
                f"Unsupported file type uploaded: {file.filename}"
            )

            return {
                "error": "Only PDF and DOCX files are supported"
            }

    except Exception as e:

        logger.exception(
            f"Guideline upload failed: {str(e)}"
        )

        return {
            "error": str(e)
        }