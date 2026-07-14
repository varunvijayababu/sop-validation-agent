import uuid

from fastapi import APIRouter
from fastapi import UploadFile
from fastapi import File

from app.parser.docx_parser import extract_docx_text

from app.parser.pdf_parser import extract_pdf_text

from app.rag.retriever import retrieve_context
from app.agents.llm_validator import validate_sop

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

        filename = file.filename.lower()

        if filename.endswith(".pdf"):

            logger.info(
                "Detected PDF SOP"
            )

            sop_text = extract_pdf_text(
                file_path
            )

        elif filename.endswith(".docx"):

            logger.info(
                "Extracting DOCX text directly"
            )

            sop_text = extract_docx_text(
                file_path
            )
            
        else:

            logger.warning(
                f"Unsupported file type: {file.filename}"
            )

            return {
                "error": "Only PDF and DOCX files are supported"
            }
        logger.info(
            f"SOP extracted successfully. Characters: {len(sop_text)}"
        )

        logger.info(
            "Retrieving reference sections from Qdrant"
        )

        reference_sections = retrieve_context(
            sop_text
        )

        logger.info(
            f"Retrieved {len(reference_sections)} reference sections"
        )

        logger.info(
            "Sending SOP and reference context to LLM validator"
        )

        logger.info(
            f"SOP text length sent to LLM: {len(sop_text)} characters"
        )

        result = validate_sop(
            sop_text,
            reference_sections,
            detailed=False
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
                "SCORE": 0.0,
                "COMMENTS": f"System Error: {str(e)}",
                "REFERENCE": "System Error (Page N/A)"
            }
        ]
    
@router.post("/validate-sop-detailed")
async def validate_detailed(
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

        filename = file.filename.lower()

        if filename.endswith(".pdf"):

            logger.info(
                "Detected PDF SOP"
            )

            sop_text = extract_pdf_text(
                file_path
            )

        elif filename.endswith(".docx"):

            logger.info(
                "Extracting DOCX text directly"
            )

            sop_text = extract_docx_text(
                file_path
            )
            
        else:

            logger.warning(
                f"Unsupported file type: {file.filename}"
            )

            return {
                "error": "Only PDF and DOCX files are supported"
            }
        logger.info(
            f"SOP extracted successfully. Characters: {len(sop_text)}"
        )

        logger.info(
            "Retrieving reference sections from Qdrant"
        )

        reference_sections = retrieve_context(
            sop_text
        )

        logger.info(
            f"Retrieved {len(reference_sections)} reference sections"
        )

        logger.info(
            "Sending SOP and reference context to LLM validator"
        )

        logger.info(
            f"SOP text length sent to LLM: {len(sop_text)} characters"
        )

        result = validate_sop(
            sop_text,
            reference_sections,
            detailed=True
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
                "SCORE": 0.0,
                "SCORE_BREAKDOWN": {},
                "COMMENTS": f"System Error: {str(e)}",
                "REFERENCE": "System Error (Page N/A)",
                "TOKEN_COUNT": {
                    "INPUT": 0,
                    "OUTPUT": 0,
                    "TOTAL": 0
                }
            }
        ]