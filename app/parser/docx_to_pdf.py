from docx2pdf import convert
import logging

logger = logging.getLogger(__name__)


def convert_docx_to_pdf(
    docx_path,
    pdf_path
):

    try:

        logger.info(
            f"Starting DOCX to PDF conversion. "
            f"Source: {docx_path}"
        )

        convert(
            docx_path,
            pdf_path
        )

        logger.info(
            f"DOCX converted successfully. "
            f"Output: {pdf_path}"
        )

    except Exception as e:

        logger.exception(
            f"DOCX to PDF conversion failed: {str(e)}"
        )

        raise