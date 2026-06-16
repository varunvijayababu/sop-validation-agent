import fitz
import logging
import os

from app.parser.image_captioner import (
    generate_image_caption
)

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

        image_dir = os.path.join(
            "uploads",
            "extracted_images"
        )

        os.makedirs(
            image_dir,
            exist_ok=True
        )

        pages = []

        for page_num, page in enumerate(doc, start=1):

            text = page.get_text()

            logger.info(
                f"Extracted page {page_num} "
                f"({len(text)} characters)"
            )

            image_descriptions = []

            image_list = page.get_images(
                full=True
            )

            logger.info(
                f"Found {len(image_list)} images "
                f"on page {page_num}"
            )

            for image_index, image in enumerate(
                image_list,
                start=1
            ):

                try:

                    xref = image[0]

                    pix = fitz.Pixmap(
                        doc,
                        xref
                    )

                    image_path = os.path.join(
                        image_dir,
                        f"page_{page_num}_img_{image_index}.png"
                    )

                    if pix.n < 5:

                        pix.save(
                            image_path
                        )

                    else:

                        rgb_pix = fitz.Pixmap(
                            fitz.csRGB,
                            pix
                        )

                        rgb_pix.save(
                            image_path
                        )

                        rgb_pix = None

                    pix = None

                    logger.info(
                        f"Saved image: {image_path}"
                    )

                    caption = generate_image_caption(
                        image_path
                    )

                    image_descriptions.append(
                        caption
                    )

                    logger.info(
                        f"Caption generated: {caption}"
                    )

                except Exception as image_error:

                    logger.exception(
                        f"Image processing failed: "
                        f"{str(image_error)}"
                    )

            if image_descriptions:

                text += "\n\nIMAGE FINDINGS:\n"

                for idx, caption in enumerate(
                    image_descriptions,
                    start=1
                ):

                    text += (
                        f"\nImage {idx}: "
                        f"{caption}\n"
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