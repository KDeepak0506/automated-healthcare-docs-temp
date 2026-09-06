from uuid import UUID
from pydantic import BaseModel, ConfigDict, Field


class SearchRequest(BaseModel):
    query: str = Field(..., min_length=1, max_length=1000, description="The user query or question about the document")
    top_k: int = Field(default=5, ge=1, le=20, description="Number of relevant chunks to retrieve")


class SourceReference(BaseModel):
    model_config = ConfigDict(from_attributes=True)

    chunk_id: UUID
    chunk_index: int
    page_number: int | None = None
    similarity_score: float | None = None
    content_preview: str | None = None


class SearchResponse(BaseModel):
    answer: str
    sources: list[SourceReference]


class IndexResponse(BaseModel):
    document_id: UUID
    chunks_created: int
    status: str
