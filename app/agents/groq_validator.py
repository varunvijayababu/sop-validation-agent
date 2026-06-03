import os
import json

from groq import Groq
from dotenv import load_dotenv

print("GROQ_VALIDATOR.PY LOADED")
print(__file__)

load_dotenv()

client = Groq(
    api_key=os.getenv("GROQ_API_KEY")
)

def validate_sop(sop_text, reference_context):

    prompt = f"""
You are an SOP validation agent.

Compare the submitted SOP against the reference SOP content.

Reference SOP:
{reference_context}

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
- Be 3-6 sentences.
- Be written in professional audit language.

Return ONLY valid JSON.

Output format:

[
    {{
        "STATUS": "ACCEPT",
        "COMMENTS": "Reason"
    }}
]

OR

[
    {{
        "STATUS": "MODIFY",
        "COMMENTS": "Reason"
    }}
]

OR

[
    {{
        "STATUS": "REJECT",
        "COMMENTS": "Reason"
    }}
]

Do not return markdown.
Do not return explanations.
Return JSON only.
"""

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

    result = response.choices[0].message.content

    try:
        return json.loads(result)

    except Exception as e:
        return [
            {
                "STATUS": "REJECT",
                "COMMENTS": f"JSON Parsing Error: {str(e)}"
            }
        ]