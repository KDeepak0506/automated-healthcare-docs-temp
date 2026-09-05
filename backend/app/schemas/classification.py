from uuid import UUID
from pydantic import BaseModel, ConfigDict, Field, field_validator

ALLOWED_DOCUMENT_TYPES = [
    "Laboratory Report",
    "Prescription",
    "Discharge Summary",
    "Radiology Report",
    "Insurance Form",
    "Other",
]


class ClassificationResult(BaseModel):
    """Structured LLM output validation schema for M3 Document Type Identification."""
    document_type: str = Field(..., description="Identified healthcare document category")
    classification_confidence: float = Field(..., ge=0.0, le=1.0, description="Confidence score between 0.0 and 1.0")

    @field_validator("document_type")
    @classmethod
    def validate_document_type(cls, v: str) -> str:
        # Match case-insensitively or exact match
        for allowed in ALLOWED_DOCUMENT_TYPES:
            if v.strip().lower() == allowed.lower():
                return allowed
        raise ValueError(
            f"Unsupported document_type '{v}'. Must be one of: {', '.join(ALLOWED_DOCUMENT_TYPES)}"
        )


class ClassificationResponse(BaseModel):
    """API response schema for document classification endpoint."""
    model_config = ConfigDict(from_attributes=True)

    document_id: UUID
    document_type: str
    classification_confidence: float
    cached: bool = False
    truncated: bool = False
