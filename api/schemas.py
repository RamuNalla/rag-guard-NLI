from pydantic import BaseModel, Field
from typing import List

class RAGRequest(BaseModel):
    generated_text: str = Field(..., description="The LLM's generated response to evaluate.")
    source_text: str = Field(..., description="The retrieved context or source documents.")

class ClaimResult(BaseModel):
    claim: str
    nli_label: str
    confidence: float

class HallucinationReport(BaseModel):
    execution_time_seconds: float
    total_claims_checked: int
    details: List[ClaimResult]