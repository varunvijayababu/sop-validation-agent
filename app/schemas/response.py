from pydantic import BaseModel

class ValidationResponse(BaseModel):
    decision: str
    compliance_score: int
    findings: list[str]
    recommendations: list[str]