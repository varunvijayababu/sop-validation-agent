import os
import json

from groq import Groq
from dotenv import load_dotenv

import logging

logger = logging.getLogger(__name__)

load_dotenv()

client = Groq(
    api_key=os.getenv(
        "GROQ_API_KEY"
    )
)

def rank_sections(chunks):

    section_list = []

    for chunk in chunks:

        section_list.append(
            {
                "section": chunk["section"],
                "summary": chunk["text"][:300]
            }
        )

    logger.info(
        f"Sending {len(section_list)} sections to weight generator"
    )

    sections_json = json.dumps(
        section_list,
        indent=2
    )

    prompt = f"""
    You are an Infection Prevention and Control (IPC) compliance expert.

    Below is a list of sections extracted from an uploaded guideline.

    {sections_json}

    IMPORTANT:

    The guideline contains {len(section_list)} sections.

    You MUST return exactly {len(section_list)} JSON objects.

    One JSON object for every section provided.

    Do not omit any section.

    The total of all weights must equal exactly 100.

    If you return fewer than {len(section_list)} sections, the response is invalid.

    Your task:

    Assign a relative importance weight to EACH section.

    Weights must reflect:

    - Patient safety impact
    - Regulatory importance
    - Operational importance
    - Audit and accreditation importance

    Important Rules:

    - Weights must total EXACTLY 100.
    - Higher weights should be assigned to sections that directly influence infection prevention effectiveness, compliance, surveillance, training, auditing, accountability, and patient safety.
    - Lower weights should be assigned to introductions, purpose statements, scope sections, summaries, and conclusions.
    - Do NOT distribute weights evenly.
    - Every section must receive a weight.
    - Return ONLY JSON.

    Example:

    [
        {{
            "section": "Purpose",
            "weight": 4
        }},
        {{
            "section": "Scope",
            "weight": 5
        }},
        {{
            "section": "Policy and Compliance",
            "weight": 15
        }}
    ]
    """

    response = (
        client.chat.completions.create(
            model="llama-3.3-70b-versatile",
            messages=[
                {
                    "role": "user",
                    "content": prompt
                }
            ],
            temperature=0
        )
    )

    result = (
        response
        .choices[0]
        .message
        .content
    )

    result = (
        result
        .replace("```json", "")
        .replace("```", "")
        .strip()
    )

    parsed = json.loads(result)

    if not isinstance(parsed, list):

        raise Exception(
            "Groq returned invalid JSON format"
        )

    if len(parsed) == 0:

        raise Exception(
            "Groq returned empty section list"
        )

    total_weight = sum(
        item["weight"]
        for item in parsed
    )

    logger.info(
        f"TOTAL WEIGHT FROM GROQ: {total_weight}"
    )

    if total_weight <= 0:

        raise Exception(
            "Invalid weight total returned by Groq"
        )

    for item in parsed:

        normalized_weight = (

            item["weight"]

            / total_weight

        ) * 100

        item["weight"] = round(
            normalized_weight,
            2
        )

    normalized_total = sum(
        item["weight"]
        for item in parsed
    )

    logger.info(
        f"NORMALISED TOTAL: {normalized_total}"
    )

    expected_count = len(section_list)

    if len(parsed) != expected_count:

        raise Exception(
            f"Expected {expected_count} sections "
            f"but Groq returned {len(parsed)}"
        )

    return parsed