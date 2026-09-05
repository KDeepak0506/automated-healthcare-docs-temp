from datetime import datetime
from uuid import UUID

from pydantic import BaseModel, ConfigDict


class DocumentTextResponse(BaseModel):
    model_config = ConfigDict(from_attributes=True)

    id: UUID
    document_id: UUID
    raw_text: str
    sanitized_text: str | None = None
    privacy_metadata: dict | None = None
    page_count: int | None = None
    confidence: float | None = None
    layout: dict | None = None
    ocr_engine: str
    processing_time_ms: int | None = None
    created_at: datetime
