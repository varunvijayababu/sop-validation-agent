import re
import logging

logger = logging.getLogger(__name__)


def split_sections(pages):

    try:

        logger.info(
            f"Starting section splitting. Pages received: {len(pages)}"
        )

        chunks = []

        pattern = r"###\s+(.*?)(?=\n)"

        for page_data in pages:

            page_number = page_data["page"]
            text = page_data["text"]

            logger.info(
                f"Processing page {page_number}"
            )

            matches = list(
                re.finditer(
                    pattern,
                    text
                )
            )

            logger.info(
                f"Found {len(matches)} sections on page {page_number}"
            )

            for i, match in enumerate(matches):

                section_title = match.group(1).strip()

                start = match.end()

                if i < len(matches) - 1:
                    end = matches[i + 1].start()
                else:
                    end = len(text)

                section_text = text[start:end].strip()

                logger.info(
                    f"Section extracted: "
                    f"{section_title} "
                    f"(Page {page_number}) "
                    f"Characters: {len(section_text)}"
                )

                if not section_text:

                    logger.warning(
                        f"Skipping empty section: {section_title}"
                    )

                    continue

                chunks.append(
                    {
                        "section": section_title,
                        "page": page_number,
                        "text": section_text
                    }
                )

        if len(chunks) == 0:

            raise Exception(
                "No sections found. Expected ### section headers."
            )

        logger.info(
            f"Section splitting completed. Total chunks: {len(chunks)}"
        )

        return chunks

    except Exception as e:

        logger.exception(
            f"Section splitting failed: {str(e)}"
        )

        raise