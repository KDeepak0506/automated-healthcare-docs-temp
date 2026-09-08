from datetime import date, datetime
from uuid import UUID

from pydantic import BaseModel, ConfigDict, Field



class PatientCreate(BaseModel):
    mrn: str = Field(..., min_length=1, max_length=50)
    first_name: str = Field(..., min_length=1, max_length=100)
    last_name: str = Field(..., min_length=1, max_length=100)
    date_of_birth: date | None = None
    gender: str | None = None


class PatientUpdate(BaseModel):
    first_name: str | None = None
    last_name: str | None = None
    date_of_birth: date | None = None
    gender: str | None = None


class PatientAssignmentCreate(BaseModel):
    user_id: UUID


class PatientResponse(BaseModel):
    model_config = ConfigDict(from_attributes=True)

    patient_id: UUID
    mrn: str
    first_name: str
    last_name: str
    date_of_birth: date | None = None
    gender: str | None = None
    created_at: datetime
    updated_at: datetime
    document_count: int = 0



class PaginatedPatientResponse(BaseModel):
    items: list[PatientResponse]
    page: int
    page_size: int
    total: int
    total_pages: int
