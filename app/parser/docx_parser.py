import os
import uuid
import logging
import docx
from docx.text.paragraph import Paragraph
from docx.table import Table
from app.parser.image_captioner import generate_image_caption

logger = logging.getLogger(__name__)

def iter_block_items(parent):
    """
    Yield each paragraph and table child within parent, in document order.
    Each returned value is an instance of either Paragraph or Table.
    """
    if isinstance(parent, docx.document.Document):
        parent_elm = parent.element.body
    elif isinstance(parent, docx.table._Cell):
        parent_elm = parent._tc
    else:
        raise ValueError("Unsupported parent type for iteration")

    for child in parent_elm.iterchildren():
        if child.tag.endswith('p'):
            yield Paragraph(child, parent)
        elif child.tag.endswith('tbl'):
            yield Table(child, parent)

def check_page_break(paragraph: Paragraph) -> bool:
    """
    Checks if a paragraph triggers a page break either manually or as rendered.
    """
    try:
        # Check for manual page breaks or last rendered page breaks inside runs XML
        for run in paragraph.runs:
            if 'w:br' in run._r.xml and 'type="page"' in run._r.xml:
                return True
            if 'w:lastRenderedPageBreak' in run._r.xml:
                return True
        # Check if pageBreakBefore is set on paragraph formatting or in XML
        if paragraph.paragraph_format.page_break_before:
            return True
        if 'w:pageBreakBefore' in paragraph._p.xml:
            return True
    except Exception:
        pass
    return False

def extract_images_from_element(element, doc, image_dir) -> list:
    """
    Scans element XML for drawing blips, extracts image bytes,
    saves them to image_dir, and returns a list of local image file paths.
    """
    image_paths = []
    try:
        # Search for a:blip tags (drawing references) in the XML element
        blips = element.xpath('.//a:blip')
        for blip in blips:
            embed_id = blip.get('{http://schemas.openxmlformats.org/officeDocument/2006/relationships}embed')
            if embed_id and embed_id in doc.part.related_parts:
                try:
                    part = doc.part.related_parts[embed_id]
                    image_bytes = part.blob
                    ext = os.path.splitext(part.partname)[1] or ".png"
                    
                    image_path = os.path.join(
                        image_dir,
                        f"{uuid.uuid4()}{ext}"
                    )
                    with open(image_path, "wb") as f:
                        f.write(image_bytes)
                    logger.info(f"Extracted DOCX image saved to: {image_path}")
                    image_paths.append(image_path)
                except Exception as image_err:
                    logger.warning(f"Failed to extract image for relationship {embed_id}: {str(image_err)}")
    except Exception as e:
        logger.warning(f"Error scanning element for images: {str(e)}")
    return image_paths

def extract_docx_pages(file_path: str) -> list:
    """
    Extracts text and caption images from a DOCX file, preserving document order,
    and grouping content into virtual pages based on manual page breaks or content length.
    """
    try:
        logger.info(f"Opening DOCX file: {file_path}")
        doc = docx.Document(file_path)
        
        image_dir = os.path.join("uploads", "extracted_images")
        os.makedirs(image_dir, exist_ok=True)
        
        pages = []
        current_page_num = 1
        current_page_text = ""
        current_page_char_count = 0
        image_descriptions = []
        
        def flush_page(page_num, text_content, img_descs):
            if img_descs:
                text_content += "\n\nIMAGE FINDINGS:\n"
                for idx, caption in enumerate(img_descs, start=1):
                    text_content += f"\nImage {idx}: {caption}\n"
            return {
                "page": page_num,
                "text": text_content.strip()
            }
            
        for block in iter_block_items(doc):
            if isinstance(block, Paragraph):
                # Check for page break triggers
                if check_page_break(block) and (current_page_text.strip() or image_descriptions):
                    pages.append(flush_page(current_page_num, current_page_text, image_descriptions))
                    current_page_num += 1
                    current_page_text = ""
                    current_page_char_count = 0
                    image_descriptions = []
                
                text = block.text.strip()
                if text:
                    current_page_text += text + "\n"
                    current_page_char_count += len(text)
                
                # Extract inline drawings/images
                image_paths = extract_images_from_element(block._p, doc, image_dir)
                for path in image_paths:
                    caption = generate_image_caption(path)
                    if caption:
                        image_descriptions.append(caption)
                        
                # Split page if character limit (3000 chars) is exceeded
                if current_page_char_count >= 3000 and (current_page_text.strip() or image_descriptions):
                    pages.append(flush_page(current_page_num, current_page_text, image_descriptions))
                    current_page_num += 1
                    current_page_text = ""
                    current_page_char_count = 0
                    image_descriptions = []
                    
            elif isinstance(block, Table):
                table_text_list = []
                for row in block.rows:
                    row_cells_text = []
                    for cell in row.cells:
                        # Extract cell text
                        cell_text = cell.text.strip()
                        if cell_text:
                            row_cells_text.append(cell_text)
                        
                        # Extract drawings from table cells
                        image_paths = extract_images_from_element(cell._tc, doc, image_dir)
                        for path in image_paths:
                            caption = generate_image_caption(path)
                            if caption:
                                image_descriptions.append(caption)
                                
                    if row_cells_text:
                        row_cells_text_filtered = [c for c in row_cells_text if c]
                        if row_cells_text_filtered:
                            row_joined = " | ".join(row_cells_text_filtered)
                            row_cells_text.append(row_joined)
                            table_text_list.append(row_joined)
                        
                if table_text_list:
                    table_full_text = "\n".join(table_text_list)
                    current_page_text += table_full_text + "\n"
                    current_page_char_count += len(table_full_text)
                    
                # Split page after tables if character limit exceeded
                if current_page_char_count >= 3000 and (current_page_text.strip() or image_descriptions):
                    pages.append(flush_page(current_page_num, current_page_text, image_descriptions))
                    current_page_num += 1
                    current_page_text = ""
                    current_page_char_count = 0
                    image_descriptions = []

        # Flush final remaining page
        if current_page_text.strip() or image_descriptions or not pages:
            pages.append(flush_page(current_page_num, current_page_text, image_descriptions))
            
        logger.info(f"DOCX extraction completed. Pages extracted: {len(pages)}")
        return pages

    except Exception as e:
        logger.exception(f"DOCX extraction failed: {str(e)}")
        raise

def extract_docx_text(file_path: str) -> str:
    """
    Extracts text and image captions and joins them into a single string.
    """
    pages = extract_docx_pages(file_path)
    return "\n".join(page["text"] for page in pages)
