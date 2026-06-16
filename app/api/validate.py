import uuid

from fastapi import APIRouter
from fastapi import UploadFile
from fastapi import File

from app.parser.docx_to_pdf import convert_docx_to_pdf

from app.parser.pdf_parser import extract_pdf_text

from app.rag.retriever import retrieve_context
from app.agents.groq_validator import validate_sop

import os
import logging

logger = logging.getLogger(__name__)

router = APIRouter()

UPLOAD_DIR = "uploads"

@router.post("/validate-sop")
async def validate(
    file: UploadFile = File(...)
):

    logger.info(
        f"Validation request received: {file.filename}"
    )

    try:

        file_path = os.path.join(
            UPLOAD_DIR,
            f"{uuid.uuid4()}_{file.filename}"
        )

        logger.info(
            f"Saving uploaded SOP to: {file_path}"
        )

        with open(file_path, "wb") as f:
            f.write(await file.read())

        if file.filename.endswith(".pdf"):

            logger.info(
                "Detected PDF SOP"
            )

            sop_text = extract_pdf_text(
                file_path
            )

        elif file.filename.endswith(".docx"):

            pdf_path = file_path.replace(
                ".docx",
                ".pdf"
            )

            logger.info(
                "Converting DOCX to PDF"
            )

            logger.info(
                f"PDF path: {pdf_path}"
            )

            convert_docx_to_pdf(
                file_path,
                pdf_path
            )

            sop_text = extract_pdf_text(
                pdf_path
            )
            
        else:

            logger.warning(
                f"Unsupported file type: {file.filename}"
            )

            return {
                "error": "Unsupported file"
            }

        logger.info(
            f"SOP extracted successfully. Characters: {len(sop_text)}"
        )

        logger.info(
            "Retrieving reference sections from Qdrant"
        )

        reference_text = retrieve_context(
            sop_text
        )

        logger.info(
            f"Retrieved {len(reference_text)} reference sections"
        )

        logger.info(
            "Sending SOP and reference context to Groq validator"
        )

        logger.info(
            f"SOP TEXT SENT TO GROQ:\n{sop_text}"
        )

        result = validate_sop(
            sop_text,
            reference_text
        )

        logger.info(
            "Validation completed successfully"
        )

        return result

    except Exception as e:

        logger.exception(
            f"Validation failed: {str(e)}"
        )

        return [
            {
                "STATUS": "REJECT",
                "COMMENTS": f"System Error: {str(e)}"
            }
        ]