import os
import json
import logging
import re

from dotenv import load_dotenv
from app.llm.factory import get_llm_provider

logger = logging.getLogger(__name__)

load_dotenv()

logger.info(
    "LLM validator module loaded"
)


from functools import lru_cache
import copy

class ValidationRetryFailedError(Exception):
    """Custom exception raised when validation fails after retries, carrying token usage."""
    def __init__(self, message, token_count):
        super().__init__(message)
        self.token_count = token_count


def _normalize_section_status(status_str: str) -> str:
    """Normalize section status case-insensitively and trim whitespace to canonical values."""
    cleaned = status_str.upper().strip()
    if cleaned == "COMPLETE":
        return "COMPLIANT"
    if cleaned == "INCOMPLETE":
        return "PARTIAL"
    return cleaned


def _extract_json_string(text: str) -> str:
    text_stripped = text.strip()
    first_bracket = text_stripped.find('[')
    first_brace = text_stripped.find('{')
    
    if first_bracket != -1 and first_brace != -1:
        start_idx = min(first_bracket, first_brace)
    elif first_bracket != -1:
        start_idx = first_bracket
    elif first_brace != -1:
        start_idx = first_brace
    else:
        return text_stripped
        
    last_bracket = text_stripped.rfind(']')
    last_brace = text_stripped.rfind('}')
    
    if last_bracket != -1 and last_brace != -1:
        end_idx = max(last_bracket, last_brace)
    elif last_bracket != -1:
        end_idx = last_bracket
    elif last_brace != -1:
        end_idx = last_brace
    else:
        return text_stripped
        
    return text_stripped[start_idx:end_idx + 1]


def _normalize_strict(s: str) -> str:
    s = s.lower().strip()
    s = re.sub(r'[^\w\s]', '', s)
    s = re.sub(r'\s+', ' ', s).strip()
    return s

def _resolve_section(sec_val: str, reference_context) -> str:
    valid_sections = {sec["section"] for sec in reference_context}
    
    # 1. Exact match
    if sec_val in valid_sections:
        return sec_val
        
    # 2. Case-insensitive exact match
    sec_val_lower = sec_val.lower().strip()
    ci_matches = [s for s in valid_sections if s.lower().strip() == sec_val_lower]
    if len(ci_matches) == 1:
        return ci_matches[0]
    elif len(ci_matches) > 1:
        raise ValueError(f"Ambiguous section name (case-insensitive): '{sec_val}'")
        
    # 3. Whitespace & Punctuation normalized match
    norm_val = _normalize_strict(sec_val)
    norm_matches = [s for s in valid_sections if _normalize_strict(s) == norm_val]
    if len(norm_matches) == 1:
        return norm_matches[0]
    elif len(norm_matches) > 1:
        raise ValueError(f"Ambiguous section name (normalized): '{sec_val}'")
        
    # 4. Strip trailing page suffix and retry resolution steps 1-3
    clean_sec = re.sub(r'\s*\(?p(age|\.)?\s*\d+\)?\s*$', '', sec_val, flags=re.IGNORECASE).strip()
    if clean_sec != sec_val:
        # Exact match of clean title
        if clean_sec in valid_sections:
            return clean_sec
        # Case-insensitive
        clean_sec_lower = clean_sec.lower().strip()
        ci_matches = [s for s in valid_sections if s.lower().strip() == clean_sec_lower]
        if len(ci_matches) == 1:
            return ci_matches[0]
        elif len(ci_matches) > 1:
            raise ValueError(f"Ambiguous section name after page suffix strip (case-insensitive): '{sec_val}'")
        # Whitespace/punctuation normalized
        norm_clean = _normalize_strict(clean_sec)
        norm_matches = [s for s in valid_sections if _normalize_strict(s) == norm_clean]
        if len(norm_matches) == 1:
            return norm_matches[0]
        elif len(norm_matches) > 1:
            raise ValueError(f"Ambiguous section name after page suffix strip (normalized): '{sec_val}'")
            
    raise ValueError(f"Invalid section name: '{sec_val}'")

def _resolve_reference(ref_val: str, reference_context) -> str:
    valid_references = {f"{sec['section']} (Page {sec['page']})" for sec in reference_context}
    
    # 1. Exact match
    if ref_val in valid_references:
        return ref_val
        
    # 2. Case-insensitive exact match
    ref_val_lower = ref_val.lower().strip()
    ci_matches = [r for r in valid_references if r.lower().strip() == ref_val_lower]
    if len(ci_matches) == 1:
        return ci_matches[0]
    elif len(ci_matches) > 1:
        raise ValueError(f"Ambiguous reference (case-insensitive): '{ref_val}'")
        
    # 3. Whitespace & Punctuation normalized match
    norm_val = _normalize_strict(ref_val)
    norm_matches = [r for r in valid_references if _normalize_strict(r) == norm_val]
    if len(norm_matches) == 1:
        return norm_matches[0]
    elif len(norm_matches) > 1:
        raise ValueError(f"Ambiguous reference (normalized): '{ref_val}'")
        
    # 4. Strip trailing page suffix and attempt to match unique section title
    clean_ref = re.sub(r'\s*\(?p(age|\.)?\s*\d+\)?\s*$', '', ref_val, flags=re.IGNORECASE).strip()
    
    try:
        resolved_sec = _resolve_section(clean_ref, reference_context)
        matching_refs = [f"{sec['section']} (Page {sec['page']})" 
                         for sec in reference_context if sec["section"] == resolved_sec]
        if len(matching_refs) == 1:
            return matching_refs[0]
        elif len(matching_refs) > 1:
            raise ValueError(f"Ambiguous reference mapping for section: '{resolved_sec}'")
    except ValueError as e:
        if "Ambiguous" in str(e):
            raise
        
    raise ValueError(f"Invalid reference: '{ref_val}'")


def _make_hashable_context(reference_context):
    items = []
    for item in reference_context:
        items.append((
            item.get("section", ""),
            item.get("page", 0),
            item.get("weight", 0.0),
            item.get("text", "")
        ))
    return tuple(sorted(items, key=lambda x: (x[0], x[1])))

@lru_cache(maxsize=128)
def _cached_validate_sop_internal(sop_text, hashable_context):
    prompt_tokens = 0
    completion_tokens = 0
    total_tokens = 0

    reference_context = [
        {
            "section": item[0],
            "page": item[1],
            "weight": item[2],
            "text": item[3]
        }
        for item in hashable_context
    ]

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

        checklist_items = "\n".join(f"* {sec['section']}" for sec in reference_context)

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

Return ONLY a valid JSON list containing exactly one object.
Do not return any explanations, markdown fences, or prefix/suffix prose.

CRITICAL RULES:
1. Top-level "STATUS" must be exactly "ACCEPT", "MODIFY", or "REJECT".
2. Example placeholders (such as "Section Name" or "Reason") must NEVER be copied or used in your output. You must use actual section titles from REFERENCE_ID and provide real audit findings in COMMENTS.
3. "SECTION_RESULTS" must contain exactly one entry for every retrieved reference section. Specifically, you MUST include the following sections:
{checklist_items}
4. Each entry in "SECTION_RESULTS" must have a "SECTION" copying the retrieved section title EXACTLY. Do not paraphrase, summarize, or rename section titles. "STATUS" must be exactly "COMPLIANT", "PARTIAL", or "MISSING".
5. "REFERENCE" must copy one of the shown REFERENCE_ID values EXACTLY (including the exact section title and page suffix). Do not paraphrase, summarize, or rename references.

JSON Schema format:
[
  {{
    "STATUS": "ACCEPT | MODIFY | REJECT",
    "SECTION_RESULTS": [
      {{
        "SECTION": "Actual Section Title from Reference guidelines",
        "STATUS": "COMPLIANT | PARTIAL | MISSING"
      }}
    ],
    "COMMENTS": "Your specific audit comments here",
    "REFERENCE": "Exact Section Title (Page X) matching the most critical reference"
  }}
]
"""

        from app.llm.config import LLM_PROVIDER
        is_ollama = (LLM_PROVIDER.lower().strip() == "ollama")
        max_attempts = 2 if is_ollama else 1

        messages = [
            {
                "role": "user",
                "content": prompt
            }
        ]

        llm = get_llm_provider()
        provider_name = llm.name

        response = None
        for attempt in range(1, max_attempts + 1):
            llm_call_success = False
            try:
                try:
                    logger.info(
                        f"Sending request to LLM provider (attempt {attempt}/{max_attempts})"
                    )
                    response = llm.generate(messages=messages, temperature=0.0)
                    llm_call_success = True
                except Exception as e:
                    # LLM invocation failure (connection, API error, etc.) - fail immediately
                    raise e

                prompt_tokens += response.token_usage.get("INPUT", 0)
                completion_tokens += response.token_usage.get("OUTPUT", 0)
                total_tokens += response.token_usage.get("TOTAL", 0)

                result = response.content
                logger.info(
                    f"LLM response length: {len(result)} characters"
                )

                result = result.replace("```json", "")
                result = result.replace("```", "").strip()

                # Extract JSON from potential narrative wrappers
                result = _extract_json_string(result)

                if not result:
                    raise json.JSONDecodeError(
                        "Model returned an empty or whitespace-only response",
                        result,
                        0
                    )

                parsed = json.loads(result)

                # Normalize JSON object output to single-item list format
                if isinstance(parsed, dict):
                    parsed = [parsed]

                if not isinstance(parsed, list):
                    raise Exception(f"{provider_name} returned invalid format")

                if len(parsed) == 0:
                    raise Exception(f"{provider_name} returned empty response")

                item = parsed[0]

                # 1. Validate top-level STATUS is exactly ACCEPT, MODIFY, or REJECT
                status = item.get("STATUS")
                if not isinstance(status, str) or status.upper().strip() not in {"ACCEPT", "MODIFY", "REJECT"}:
                    raise Exception(f"Invalid top-level STATUS: {status}")
                item["STATUS"] = status.upper().strip()

                # 2. Validate COMMENTS is a non-empty string
                comments = item.get("COMMENTS")
                if not isinstance(comments, str) or not comments.strip():
                    raise Exception("COMMENTS must be a non-empty string")
                if comments.strip() in {"Reason", "Your reason here", "Brief explanation"}:
                    raise Exception(f"Placeholder COMMENTS detected: '{comments}'")

                # 3. Validate and resolve REFERENCE using strict resolver
                reference_val = item.get("REFERENCE")
                if not isinstance(reference_val, str) or not reference_val.strip():
                    raise Exception("REFERENCE must be a non-empty string")
                try:
                    resolved_reference = _resolve_reference(reference_val.strip(), reference_context)
                    item["REFERENCE"] = resolved_reference
                except ValueError as e:
                    raise Exception(f"Invalid REFERENCE '{reference_val}': {str(e)}")

                # 4. Validate SECTION_RESULTS is valid list
                section_results = item.get("SECTION_RESULTS", [])
                if not isinstance(section_results, list):
                    raise Exception("SECTION_RESULTS must be a list")

                resolved_section_results = []
                for sec_res in section_results:
                    if not isinstance(sec_res, dict):
                        raise Exception("Each entry in SECTION_RESULTS must be an object")
                    sec_name = sec_res.get("SECTION")
                    sec_status = sec_res.get("STATUS")
                    if not isinstance(sec_name, str) or not sec_name.strip():
                        raise Exception("SECTION name must be a non-empty string")
                    if sec_name.strip() in {"Section Name", "SOP Section Name"}:
                        raise Exception(f"Placeholder SECTION name detected: '{sec_name}'")
                    if not isinstance(sec_status, str):
                        raise Exception(f"Invalid section STATUS: {sec_status}")
                    sec_status_normalized = _normalize_section_status(sec_status)
                    if sec_status_normalized not in {"COMPLIANT", "PARTIAL", "MISSING"}:
                        raise Exception(f"Invalid section STATUS: {sec_status}")
                    
                    # Resolve section name using strict resolver
                    try:
                        resolved_sec_name = _resolve_section(sec_name.strip(), reference_context)
                    except ValueError as e:
                        raise Exception(f"Section name '{sec_name}' could not be resolved: {str(e)}")
                        
                    resolved_section_results.append({
                        "SECTION": resolved_sec_name,
                        "STATUS": sec_status_normalized
                    })
                    
                item["SECTION_RESULTS"] = resolved_section_results

                # Enforce strict section completeness for Ollama
                if is_ollama:
                    if len(resolved_section_results) != len(reference_context):
                        raise ValueError(
                            f"Ollama returned {len(resolved_section_results)} section results, "
                            f"but expected {len(reference_context)}."
                        )

                # Successfully validated. Break loop.
                break

            except (json.JSONDecodeError, Exception, ValueError) as e:
                if not llm_call_success:
                    raise e
                if is_ollama:
                    if attempt == 1:
                        logger.warning(
                            f"Ollama validation failed on attempt 1: {str(e)}. "
                            f"Retrying with correction prompt..."
                        )
                        expected_count = len(reference_context)
                        section_titles = [sec["section"] for sec in reference_context]
                        correction_prompt = (
                            f"Validation failed: {str(e)}\n"
                            f"You MUST evaluate every single retrieved reference section.\n"
                            f"Expected count of retrieved sections: {expected_count}\n"
                            f"The exact retrieved section titles are:\n"
                            + "\n".join(f"- {title}" for title in section_titles) + "\n"
                            f"Omitting any section is invalid. You MUST return exactly one entry per section in SECTION_RESULTS."
                        )
                        assistant_content = response.content if response else ""
                        messages.append({
                            "role": "assistant",
                            "content": assistant_content
                        })
                        messages.append({
                            "role": "user",
                            "content": correction_prompt
                        })
                    else:
                        e.token_count = {
                            "INPUT": prompt_tokens,
                            "OUTPUT": completion_tokens,
                            "TOTAL": total_tokens or (prompt_tokens + completion_tokens)
                        }
                        raise e
                else:
                    raise
        
        weight_map = {
            section["section"]: section["weight"]
            for section in reference_context
        }

        section_results = item.get(
            "SECTION_RESULTS",
            []
        )

        logger.info(
            f"{provider_name} returned {len(section_results)} section evaluations"
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
            f"Sections missing from {provider_name} response: "
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

        max_possible_score = sum(
            weight_map.values()
        )

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

            # Calculate the section's contribution to the overall score (normalized)
            if max_possible_score == 0:
                normalized_score = 0.0
            else:
                normalized_score = (section_score / max_possible_score) * 100

            score_breakdown[
                section_name
            ] = {

                "STATUS": section_status,

                "WEIGHT": section_weight,

                "SCORE": round(
                    normalized_score,
                    2
                )
            }

        # Calculate final overall SCORE as the sum of rounded section scores
        final_score = sum(info["SCORE"] for info in score_breakdown.values())
        item["SCORE"] = round(final_score, 2)

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

        item["TOKEN_COUNT"] = {
            "INPUT": prompt_tokens,
            "OUTPUT": completion_tokens,
            "TOTAL": total_tokens or (prompt_tokens + completion_tokens)
        }

        return parsed

    except json.JSONDecodeError as e:

        logger.exception(
            f"JSON parsing failed inside cached: {str(e)}"
        )
        raise

    except Exception as e:

        from app.llm.config import LLM_PROVIDER
        provider_map = {
            "groq": "Groq",
            "openai": "OpenAI",
            "gemini": "Gemini",
            "ollama": "Ollama"
        }
        raw_provider = LLM_PROVIDER.strip() if LLM_PROVIDER else ""
        provider_name = provider_map.get(raw_provider.lower(), raw_provider.capitalize() if raw_provider else "LLM")
        logger.exception(
            f"{provider_name} validation failed inside cached: {str(e)}"
        )
        raise


def validate_sop(sop_text, reference_context, detailed=False):
    from app.llm.config import LLM_PROVIDER
    provider_map = {
        "groq": "Groq",
        "openai": "OpenAI",
        "gemini": "Gemini",
        "ollama": "Ollama"
    }
    raw_provider = LLM_PROVIDER.strip() if LLM_PROVIDER else ""
    provider_name = provider_map.get(raw_provider.lower(), raw_provider.capitalize() if raw_provider else "LLM")

    try:
        # Lazy client instantiation to catch provider setup errors gracefully
        try:
            llm = get_llm_provider()
        except Exception as e:
            logger.exception(f"Failed to instantiate LLM provider {provider_name}: {str(e)}")
            comments = f"{provider_name} request failed: {str(e)}"
            if detailed:
                return [
                    {
                        "STATUS": "SYSTEM_ERROR",
                        "SCORE": 0.0,
                        "SCORE_BREAKDOWN": {},
                        "COMMENTS": comments,
                        "REFERENCE": "System Error (Page N/A)",
                        "TOKEN_COUNT": {
                            "INPUT": 0,
                            "OUTPUT": 0,
                            "TOTAL": 0
                        }
                    }
                ]
            else:
                return [
                    {
                        "STATUS": "SYSTEM_ERROR",
                        "SCORE": 0.0,
                        "COMMENTS": comments,
                        "REFERENCE": "System Error (Page N/A)"
                    }
                ]

        hashable_context = _make_hashable_context(reference_context)
        cached_parsed = _cached_validate_sop_internal(sop_text, hashable_context)

        parsed = copy.deepcopy(cached_parsed)
        item = parsed[0]

        if detailed:
            formatted_item = {
                "STATUS": item.get("STATUS"),
                "SCORE": item.get("SCORE"),
                "SCORE_BREAKDOWN": item.get("SCORE_BREAKDOWN"),
                "COMMENTS": item.get("COMMENTS"),
                "REFERENCE": item.get("REFERENCE"),
                "TOKEN_COUNT": item.get("TOKEN_COUNT")
            }
        else:
            formatted_item = {
                "STATUS": item.get("STATUS"),
                "SCORE": item.get("SCORE"),
                "COMMENTS": item.get("COMMENTS"),
                "REFERENCE": item.get("REFERENCE")
            }
        parsed[0] = formatted_item

        return parsed

    except json.JSONDecodeError as e:

        logger.exception(
            f"JSON parsing failed: {str(e)}"
        )

        token_count = getattr(e, "token_count", {
            "INPUT": 0,
            "OUTPUT": 0,
            "TOTAL": 0
        })

        if detailed:
            return [
                {
                    "STATUS": "SYSTEM_ERROR",
                    "SCORE": 0.0,
                    "SCORE_BREAKDOWN": {},
                    "COMMENTS": f"JSON Parsing Error: {str(e)}",
                    "REFERENCE": "Validation System (Page N/A)",
                    "TOKEN_COUNT": token_count
                }
            ]
        else:
            return [
                {
                    "STATUS": "SYSTEM_ERROR",
                    "SCORE": 0.0,
                    "COMMENTS": f"JSON Parsing Error: {str(e)}",
                    "REFERENCE": "Validation System (Page N/A)"
                }
            ]

    except Exception as e:

        logger.exception(
            f"{provider_name} validation failed: {str(e)}"
        )

        error_text = str(e)

        try:
            llm = get_llm_provider()
            if hasattr(llm, "map_exception"):
                comments = llm.map_exception(e)
            else:
                comments = f"{provider_name} request failed: {error_text}"
        except Exception:
            comments = f"{provider_name} request failed: {error_text}"

        token_count = getattr(e, "token_count", {
            "INPUT": 0,
            "OUTPUT": 0,
            "TOTAL": 0
        })

        if detailed:
            return [
                {
                    "STATUS": "SYSTEM_ERROR",
                    "SCORE": 0.0,
                    "SCORE_BREAKDOWN": {},
                    "COMMENTS": comments,
                    "REFERENCE": "System Error (Page N/A)",
                    "TOKEN_COUNT": token_count
                }
            ]
        else:
            return [
                {
                    "STATUS": "SYSTEM_ERROR",
                    "SCORE": 0.0,
                    "COMMENTS": comments,
                    "REFERENCE": "System Error (Page N/A)"
                }
            ]