from docx import Document
import logging

logger = logging.getLogger(__name__)


def extract_docx_pages(file_path: str):

    try:

        logger.info(
            f"Opening DOCX: {file_path}"
        )

        doc = Document(file_path)

        logger.info(
            f"DOCX opened successfully. "
            f"Paragraph count: {len(doc.paragraphs)}"
        )

        text = "\n".join(
            paragraph.text
            for paragraph in doc.paragraphs
        )

        logger.info(
            f"DOCX text extracted. "
            f"Characters: {len(text)}"
        )

        pages = [
            {
                "page": 1,
                "text": text
            }
        ]

        logger.info(
            "DOCX converted into single-page structure"
        )

        return pages

    except Exception as e:

        logger.exception(
            f"DOCX extraction failed: {str(e)}"
        )

        raise


def extract_docx_text(file_path: str):

    try:

        logger.info(
            f"Extracting full DOCX text: {file_path}"
        )

        pages = extract_docx_pages(file_path)

        text = pages[0]["text"]

        logger.info(
            f"Full DOCX text extracted. "
            f"Characters: {len(text)}"
        )

        return text

    except Exception as e:

        logger.exception(
            f"DOCX text extraction failed: {str(e)}"
        )

        raise