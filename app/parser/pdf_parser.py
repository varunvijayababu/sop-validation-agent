import fitz
import logging

logger = logging.getLogger(__name__)


def extract_pdf_pages(file_path: str):

    try:

        logger.info(
            f"Opening PDF: {file_path}"
        )

        doc = fitz.open(file_path)

        logger.info(
            f"PDF opened successfully. Total pages: {len(doc)}"
        )

        pages = []

        for page_num, page in enumerate(doc, start=1):

            text = page.get_text()

            logger.info(
                f"Extracted page {page_num} "
                f"({len(text)} characters)"
            )

            pages.append(
                {
                    "page": page_num,
                    "text": text
                }
            )

        logger.info(
            f"PDF extraction completed. "
            f"Pages extracted: {len(pages)}"
        )

        return pages

    except Exception as e:

        logger.exception(
            f"PDF extraction failed: {str(e)}"
        )

        raise


def extract_pdf_text(file_path: str):

    try:

        logger.info(
            f"Extracting complete PDF text: {file_path}"
        )

        pages = extract_pdf_pages(file_path)

        full_text = "\n".join(
            page["text"]
            for page in pages
        )

        logger.info(
            f"Combined PDF text length: "
            f"{len(full_text)} characters"
        )

        return full_text

    except Exception as e:

        logger.exception(
            f"PDF text extraction failed: {str(e)}"
        )

        raise