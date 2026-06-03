from docx import Document

def extract_docx_text(file_path: str):
    doc = Document(file_path)

    text = "\n".join(
        paragraph.text
        for paragraph in doc.paragraphs
    )

    return text