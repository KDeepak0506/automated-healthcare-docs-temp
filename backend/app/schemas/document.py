from uuid import UUID
from pydantic import BaseModel, ConfigDict, Field
from enum import Enum
from datetime import datetime


class DocumentProcessingStatus(str, Enum):
    PENDING = "Pending"
    PROCESSING = "Processing"
    COMPLETED = "Completed"
    FAILED = "Failed"


class DocumentResponse(BaseModel):
    model_config = ConfigDict(from_attributes=True)

    document_id: UUID
    patient_id: UUID | None
    uploaded_by: UUID
    file_name: str
    file_type: str
    file_url: str
    document_type: str | None = None
    classification_confidence: float | None = None
    summary: str | None = None
    key_findings: list[str] | None = None
    processing_status: DocumentProcessingStatus
    privacy_status: str = "pending"
    uploaded_at: datetime


class PaginatedDocumentResponse(BaseModel):
    """Paginated wrapper for document listing (M7)."""

    items: list[DocumentResponse]
    page: int
    page_size: int
    total: int
    total_pages: int