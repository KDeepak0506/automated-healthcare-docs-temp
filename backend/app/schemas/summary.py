from uuid import UUID
from pydantic import BaseModel, ConfigDict, Field, field_validator


class SummaryResult(BaseModel):
    """Structured LLM output validation schema for M5 Document Summarization."""
    summary: str = Field(..., description="Factual, grounded medical document summary")
    key_findings: list[str] = Field(default_factory=list, description="List of key clinical findings and lab values")

    @field_validator("summary")
    @classmethod
    def validate_summary_non_empty(cls, v: str) -> str:
        if not v or not v.strip():
            raise ValueError("Summary must not be empty")
        return v.strip()


class SummaryResponse(BaseModel):
    """API response schema for document summarization endpoint."""
    model_config = ConfigDict(from_attributes=True)

    document_id: UUID
    summary: str
    key_findings: list[str]
    cached: bool = False
    truncated: bool = False
