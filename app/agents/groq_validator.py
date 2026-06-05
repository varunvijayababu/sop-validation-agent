import os
import json
import logging

from groq import Groq
from dotenv import load_dotenv

logger = logging.getLogger(__name__)

load_dotenv()

client = Groq(
    api_key=os.getenv("GROQ_API_KEY")
)

logger.info(
    "Groq client initialized"
)


def validate_sop(sop_text, reference_context):

    try:

        logger.info(
            "Starting SOP validation"
        )

        logger.info(
            f"Retrieved reference sections: {len(reference_context)}"
        )

        formatted_reference = ""

        for item in reference_context:

            logger.info(
                f"Reference section: "
                f"{item['section']} "
                f"(Page {item['page']})"
            )

            formatted_reference += f"""
REFERENCE_ID:
{item['section']} (Page {item['page']})

CONTENT:
{item['text']}

--------------------------------
"""

        logger.info(
            f"SOP length: {len(sop_text)} characters"
        )

        prompt = f"""
You are an SOP validation agent.

Compare the submitted SOP against the reference SOP content.

Reference SOP:
{formatted_reference}

Submitted SOP:
{sop_text}

Decision Rules:

ACCEPT:
- SOP aligns almost completely with reference SOP.

MODIFY:
- SOP is mostly correct but requires additions or corrections.

REJECT:
- SOP has major missing sections, procedures, responsibilities, or compliance requirements.

COMMENTS should:

- Mention specific missing sections.
- Mention compliance gaps.
- Be concise.
- Maximum 3 sentences.
- Be written in professional audit language.

REFERENCE RULES:

You MUST choose exactly one reference from the retrieved sections below.

You MUST copy the section title exactly as provided.

You MUST copy the page number exactly as provided.

Do NOT invent section names.

Do NOT invent page numbers.

Output format:

Section Title (Page X)

Example:

Policy and Compliance Expectations (Page 2)

Roles and Responsibilities (Page 3)

Return ONLY valid JSON.

[
    {{
        "STATUS": "MODIFY",
        "COMMENTS": "Reason",
        "REFERENCE": "Policy and Compliance Expectations (Page 2)"
    }}
]

REFERENCE must exactly match one REFERENCE_ID shown above.

Do not return markdown.
Do not return explanations.
Return JSON only.
"""

        logger.info(
            "Sending request to Groq"
        )

        response = client.chat.completions.create(
            model="llama-3.3-70b-versatile",
            messages=[
                {
                    "role": "user",
                    "content": prompt
                }
            ],
            temperature=0
        )

        logger.info(
            "Groq response received"
        )

        result = response.choices[0].message.content

        logger.info(
            f"Raw Groq response: {result}"
        )

        result = result.replace(
            "```json",
            ""
        )

        result = result.replace(
            "```",
            ""
        ).strip()

        parsed = json.loads(result)

        logger.info(
            "JSON parsed successfully"
        )

        return parsed

    except json.JSONDecodeError as e:

        logger.exception(
            f"JSON parsing failed: {str(e)}"
        )

        return [
            {
                "STATUS": "REJECT",
                "COMMENTS": f"JSON Parsing Error: {str(e)}",
                "REFERENCE": "System"
            }
        ]

    except Exception as e:

        logger.exception(
            f"Groq validation failed: {str(e)}"
        )

        return [
            {
                "STATUS": "REJECT",
                "COMMENTS": f"Validation Error: {str(e)}",
                "REFERENCE": "System"
            }
        ]