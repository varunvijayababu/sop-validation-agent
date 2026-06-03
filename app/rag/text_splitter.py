import re

def split_sections(text):

    sections = re.split(
        r"###\s+",
        text
    )

    return [
        section.strip()
        for section in sections
        if section.strip()
    ]