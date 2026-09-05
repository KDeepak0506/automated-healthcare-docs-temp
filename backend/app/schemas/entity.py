from datetime import datetime
from uuid import UUID
from pydantic import BaseModel, ConfigDict, Field


class EntityResponse(BaseModel):
    model_config = ConfigDict(from_attributes=True)

    entity_id: UUID
    document_id: UUID
    text: str
    label: str
    confidence: float = Field(..., description="Exact confidence score returned by the model as a float")
    start_char: int | None = None
    end_char: int | None = None
    created_at: datetime


class DocumentEntitiesResponse(BaseModel):
    document_id: UUID
    entities: list[EntityResponse]
    total_count: int
