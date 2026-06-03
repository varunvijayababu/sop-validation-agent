def validate_sop(
    sop_text: str,
    reference_text: str
):
    score = 50

    if len(sop_text) > 1000:
        score += 20

    if "purpose" in sop_text.lower():
        score += 10

    if "scope" in sop_text.lower():
        score += 10

    if score >= 90:
        decision = "ACCEPT"
    elif score >= 70:
        decision = "MODIFY"
    else:
        decision = "REJECT"

    return {
        "decision": decision,
        "compliance_score": score,
        "findings": [
            "Basic validation completed"
        ],
        "recommendations": [
            "Add more detailed sections"
        ]
    }