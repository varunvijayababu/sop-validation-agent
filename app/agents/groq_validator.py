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

DOCUMENT QUALITY RULES:

Before evaluating compliance, first determine whether the SOP content is readable and meaningful.

If the SOP contains large amounts of corrupted, nonsensical, random, encoded, or unreadable text, the SOP must be REJECTED.

Do NOT assume unreadable content satisfies a requirement.

Do NOT claim sections are missing if section headings exist but the content is unreadable.

In such cases, COMMENTS must explain that the SOP content cannot be evaluated because it is corrupted or unreadable.

IMAGE VALIDATION RULES:

The SOP may contain image findings extracted from embedded images.

Image findings appear in the following format:

IMAGE FINDINGS:

Image 1: description

Image 2: description

You MUST evaluate every image finding.

Determine whether each image:

* Supports the SOP content
* Illustrates an SOP procedure
* Demonstrates an infection prevention practice
* Provides operational or training value

Examples of relevant images:

* Hand hygiene procedures
* PPE usage demonstrations
* Isolation signage
* Biomedical waste segregation charts
* Infection surveillance dashboards
* Cleaning and disinfection workflows
* Medical device sterilization procedures
* Infection prevention training materials

Examples of irrelevant images:

* Personal photographs
* Team photographs
* Office meetings
* Portraits
* Selfies
* Decorative images
* Images unrelated to infection prevention and control

MANDATORY IMAGE FINDING RULE:

If any image is irrelevant:

* You MUST mention the image issue in COMMENTS.
* You MUST explain why the image does not support the SOP.
* You MUST NOT ignore image findings.
* Image findings must be included even when larger SOP deficiencies exist.

COMMENTS must contain:

1. Image issue (if present)
2. SOP content issue (if present)

When reporting image findings:

- Refer to the image as an embedded image, figure, illustration, or visual content.
- Do not refer to it as an "image finding".
- Use professional audit language.

Good example:
"The SOP contains an embedded image depicting individuals in an office environment that does not support or illustrate any infection prevention and control procedure."

Bad example:
"The image finding of two men sitting in a room..."

If the image is relevant:

* Do not mention it.
* Continue normal SOP validation.

Image issues alone should normally result in MODIFY.

However, if the SOP also contains major compliance deficiencies, the overall status may be REJECT.

Before determining STATUS:

1. Evaluate all image findings.
2. Record any image relevance issues.
3. Then evaluate SOP content.
4. Include both findings in COMMENTS when applicable.

SECTION COMPLIANCE EVALUATION RULES

The reference SOP is divided into multiple sections.

Each retrieved section represents one compliance requirement.

For EVERY retrieved reference section:

Determine whether the submitted SOP is:

COMPLIANT
- Requirement is adequately addressed.

PARTIAL
- Requirement is partially addressed.
- Some required information is missing.
- The section exists but does not fully satisfy the guideline requirements.
- Related content is present, but important controls, responsibilities, procedures, monitoring requirements, documentation requirements, or implementation details are missing.
- Any section that is present but incomplete MUST be classified as PARTIAL.

MISSING
- Requirement is absent.
- The section cannot be identified in the SOP.
- The section heading is absent and the corresponding content is not meaningfully addressed elsewhere.
- Use MISSING only when the requirement is essentially absent.

IMPORTANT

Do NOT classify a section as MISSING if relevant content exists.

If the section is present but incomplete:
STATUS = PARTIAL

If the section heading exists and contains meaningful content:
STATUS = PARTIAL or COMPLIANT

Never classify such sections as MISSING unless the content is effectively empty.

IMPORTANT:

Do NOT assign numeric scores.

Do NOT calculate totals.

Only classify each retrieved section as:

COMPLIANT
PARTIAL
MISSING

Image content should contribute only to the section it supports.

Images are not scored separately.

A section may be COMPLIANT through:

- text
- image
- text + image

If an image is irrelevant, mention it in COMMENTS but do not create a separate image score.

IMPORTANT:

You MUST evaluate EVERY retrieved reference section.

For each retrieved reference section, create exactly one entry in SECTION_RESULTS.

Do not omit sections.

The number of SECTION_RESULTS entries must equal the number of retrieved reference sections.

If a section cannot be identified in the SOP:

STATUS = MISSING

Decision Rules:

ACCEPT:
- All major guideline requirements are present.
- No material compliance gaps exist.
- Minor wording differences, formatting differences, additional content, or organizational customizations MUST NOT result in MODIFY.
- If the SOP would reasonably pass an internal quality review or accreditation review, return ACCEPT.

MODIFY:
- One or more guideline requirements are partially addressed.
- A specific section, responsibility, procedure, compliance requirement, or control is missing or insufficient.
- COMMENTS must clearly identify the gap.

REJECT:

* Multiple major guideline sections are missing.
* Critical infection prevention controls are absent.
* The SOP cannot be considered compliant with the guideline.
* The SOP is only a high-level summary and lacks sufficient procedural detail to function as an operational SOP.
* The SOP contains headings but does not provide enough instructions, controls, responsibilities, monitoring requirements, or implementation details.
* The SOP is too brief to demonstrate compliance with the guideline requirements.
* A document consisting primarily of a cover page, metadata, titles, introductory text, images, figures, or other non-operational content without substantive SOP procedures MUST be classified as REJECT.
* If the majority of required SOP sections are absent, classify as REJECT and not MODIFY.
* If the document does not contain sufficient operational guidance for staff to perform the described process, classify as REJECT.

CLASSIFICATION CLARIFICATION:

MODIFY:

* Use MODIFY only when a functioning SOP exists and specific requirements, controls, responsibilities, procedures, or compliance elements need improvement.

REJECT:

* Use REJECT when the SOP is largely absent, incomplete, non-operational, or cannot reasonably be implemented by staff.
* A cover page, outline, template, summary, or placeholder document is not a valid SOP and must be classified as REJECT.

IMPORTANT:

Decision Boundary:

Choose MODIFY only when the SOP is substantially complete and operationally usable, but contains one or more identifiable compliance gaps.

Choose REJECT when the SOP lacks sufficient detail to function as an implementable SOP, even if section headings are present.

A document that contains only short descriptions, summaries, or placeholders for major sections should be REJECTED rather than MODIFY.

Do NOT choose MODIFY unless you can identify a specific compliance gap.

Do NOT choose REJECT unless multiple major requirements are missing. 

When in doubt between ACCEPT and MODIFY, choose ACCEPT if all major requirements are present.

SOP COMPLETENESS RULE:

A valid SOP must contain actionable procedures, responsibilities, controls, monitoring requirements, and implementation guidance.

The existence of section headings alone does not satisfy a requirement.

However, the existence of a section heading with meaningful content is strong evidence that the requirement is at least partially addressed.

A section should not be classified as MISSING solely because it lacks some guideline requirements.

If most sections contain only brief descriptive statements rather than operational procedures, the SOP should be classified as REJECT.

COMMENTS should:

For ACCEPT:
- Briefly state that the SOP satisfies the major guideline requirements.
- Maximum 2 sentences.

For MODIFY:
- Clearly identify the missing or insufficient requirement.
- Mention the specific compliance gap.
- Maximum 3 sentences.

For REJECT:
- Identify the major missing areas.
- Maximum 3 sentences.

Never give generic comments such as:
"requires additions or corrections"
"needs improvement"
"mostly correct"

Every MODIFY or REJECT decision must identify a specific gap.

REFERENCE RULES:

For ACCEPT:
- Choose the guideline section that best supports overall compliance.

For MODIFY:
- Choose the guideline section containing the identified gap.

For REJECT:
- Choose the most critical missing section.

If the SOP is unreadable, corrupted, nonsensical, contains random text, or cannot be meaningfully evaluated:
- Do NOT claim specific sections are missing.
- Do NOT infer compliance gaps from unreadable content.
- Choose the guideline section that best represents overall SOP quality and completeness.
- Prefer "Characteristics of a High-Quality IPC SOP" when available.

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

CLASSIFICATION PRIORITY

When deciding between COMPLIANT, PARTIAL, and MISSING:

1. Determine whether the section exists.
2. If it exists but is incomplete:
   PARTIAL
3. If it exists and satisfies the guideline:
   COMPLIANT
4. Use MISSING only when the requirement is absent.

When uncertain between PARTIAL and MISSING, choose PARTIAL if any meaningful related content exists.

For introductory, scope, purpose, summary, and conclusion sections:

If the section heading exists and contains meaningful content:

STATUS = COMPLIANT or PARTIAL

Do not classify these sections as MISSING unless the section is completely absent.

Example ACCEPT:

[
  {{
    "STATUS": "ACCEPT",
    "SECTION_RESULTS": [
      {{
        "SECTION": "Purpose of an Infection Prevention and Control SOP",
        "STATUS": "COMPLIANT"
      }},
      {{
        "SECTION": "Policy and Compliance Expectations",
        "STATUS": "COMPLIANT"
      }}
    ],
    "COMMENTS": "The SOP comprehensively addresses the major IPC requirements defined in the reference guideline.",
    "REFERENCE": "Characteristics of a High-Quality IPC SOP (Page 6)"
  }}
]

Example MODIFY:

[
  {{
    "STATUS": "MODIFY",
    "SECTION_RESULTS": [
      {{
        "SECTION": "Purpose of an Infection Prevention and Control SOP",
        "STATUS": "COMPLIANT"
      }},
      {{
        "SECTION": "Risk Assessment and Infection Surveillance",
        "STATUS": "PARTIAL"
      }}
    ],
    "COMMENTS": "The SOP does not sufficiently define notifiable disease reporting obligations and reporting timelines required by the guideline.",
    "REFERENCE": "Risk Assessment and Infection Surveillance (Page 4)"
  }}
]

Example REJECT:

[
  {{
    "STATUS": "REJECT",
    "SECTION_RESULTS": [
      {{    
        "SECTION": "Training and Competency Requirements",
        "STATUS": "MISSING"
      }},
      {{
        "SECTION": "Standard Infection Prevention Practices",
        "STATUS": "MISSING"
      }}
    ],
    "COMMENTS": "The SOP omits key infection prevention controls including surveillance, training requirements, and biomedical waste management procedures.",
    "REFERENCE": "Standard Infection Prevention Practices (Page 3)"
  }}
]

Return ONLY valid JSON.

[
    {{
        "STATUS": "MODIFY",
        "SECTION_RESULTS": [
            {{
                "SECTION": "Section Name",
                "STATUS": "COMPLIANT | PARTIAL | MISSING"
            }}
        ],
        "COMMENTS": "Reason",
        "REFERENCE": "Policy and Compliance Expectations (Page 2)"
    }}
]

SECTION_RESULTS must contain one entry for every retrieved reference section.

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
            f"Groq response length: {len(result)} characters"
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

        if not isinstance(parsed, list):
            raise Exception(
                "Groq returned invalid format"
            )

        if len(parsed) == 0:
            raise Exception(
                "Groq returned empty response"
            )

        item = parsed[0]
        
        weight_map = {
            section["section"]: section["weight"]
            for section in reference_context
        }

        section_results = item.get(
            "SECTION_RESULTS",
            []
        )

        logger.info(
            f"Groq returned {len(section_results)} section evaluations"
        )

        expected_sections = set(
            weight_map.keys()
        )

        returned_sections = set(
            s["SECTION"]
            for s in section_results
        )

        missing_sections = (
            expected_sections
            - returned_sections
        )

        logger.info(
            f"Sections missing from Groq response: "
            f"{missing_sections}"
        )

        for missing_section in missing_sections:

            section_results.append(
                {
                    "SECTION": missing_section,
                    "STATUS": "MISSING"
                }
            )

        logger.info(
            f"Added {len(missing_sections)} "
            f"missing sections as MISSING"
        )

        score = 0

        score_breakdown = {}

        for section in section_results:

            section_name = section["SECTION"]

            section_status = (
                section["STATUS"]
                .upper()
                .strip()
            )

            section_weight = (
                weight_map.get(
                    section_name,
                    0
                )
            )

            if section_status == "COMPLIANT":

                section_score = (
                    section_weight
                )

            elif section_status == "PARTIAL":

                section_score = (
                    section_weight * 0.5
                )

            else:

                section_score = 0

            score += section_score

            score_breakdown[
                section_name
            ] = {

                "STATUS": section_status,

                "WEIGHT": section_weight,

                "SCORE": round(
                    section_score,
                    2
                )
            }

        max_possible_score = sum(
            weight_map.values()
        )

        if max_possible_score == 0:
            final_score = 0
        else:
            final_score = (
                score / max_possible_score
            ) * 100

        item["SCORE"] = round(
            final_score,
            2
        )

        logger.info(
            "JSON parsed successfully"
        )

        logger.info(
            f"Calculated Score: {item['SCORE']}"
        )

        logger.info(
            f"Section Breakdown: {score_breakdown}"
        )
        
        item["SCORE_BREAKDOWN"] = score_breakdown

        return parsed

    except json.JSONDecodeError as e:

        logger.exception(
            f"JSON parsing failed: {str(e)}"
        )

        return [
            {
                "STATUS": "SYSTEM_ERROR",
                "COMMENTS": f"JSON Parsing Error: {str(e)}",
                "REFERENCE": "Validation System (Page N/A)"
            }
        ]

    except Exception as e:

        logger.exception(
            f"Groq validation failed: {str(e)}"
        )

        error_text = str(e)

        if (
            "rate_limit_exceeded" in error_text
            or "Rate limit reached" in error_text
            or "429" in error_text
        ):

            return [
                {
                    "STATUS": "SYSTEM_ERROR",
                    "COMMENTS": "Groq API rate limit reached. Please retry later.",
                    "REFERENCE": "System Error (Page N/A)"
                }
            ]

        if (
            "Request too large" in error_text
            or "413" in error_text
            or "tokens per minute" in error_text
        ):

            return [
                {
                    "STATUS": "SYSTEM_ERROR",
                    "COMMENTS": "Validation request exceeds model token limits. Reduce SOP size or retrieved context.",
                    "REFERENCE": "System Error (Page N/A)"
                }
            ]

        return [
            {
                "STATUS": "SYSTEM_ERROR",
                "COMMENTS": f"Validation Error: {error_text}",
                "REFERENCE": "System Error (Page N/A)"
            }
        ]