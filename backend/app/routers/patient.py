from uuid import UUID

from fastapi import APIRouter, Depends, Query, status as http_status
from sqlalchemy.orm import Session

from app.core.dependencies import get_current_user, require_roles
from app.db.session import get_db
from app.models.user import User
from app.schemas.patient import (
    PaginatedPatientResponse,
    PatientAssignmentCreate,
    PatientCreate,
    PatientResponse,
    PatientUpdate,
)
from app.schemas.rag import SearchRequest, SearchResponse
from app.schemas.user import UserRole
from app.services.patient_service import (
    assign_patient_to_user,
    create_patient,
    list_accessible_patients,
    update_patient,
    verify_patient_access,
)
from app.services.rag_service import rag_service

router = APIRouter(
    prefix="/patients",
    tags=["Patients"],
)


@router.post(
    "",
    response_model=PatientResponse,
    status_code=http_status.HTTP_201_CREATED,
    dependencies=[Depends(require_roles(UserRole.ADMIN.value, UserRole.RECORDS_STAFF.value, UserRole.DOCTOR.value))],
)
def create_patient_endpoint(
    patient_in: PatientCreate,
    current_user: User = Depends(get_current_user),
    db: Session = Depends(get_db),
):
    """Create a new patient record and assign creator automatically."""
    return create_patient(db=db, patient_in=patient_in, creator=current_user)


@router.get(
    "",
    response_model=PaginatedPatientResponse,
)
def list_patients_endpoint(
    page: int = Query(default=1, ge=1, description="Page number"),
    page_size: int = Query(default=20, ge=1, le=100, description="Items per page"),
    search: str | None = Query(default=None, description="Search by MRN, first or last name"),
    current_user: User = Depends(get_current_user),
    db: Session = Depends(get_db),
):
    """List patients accessible to the current user."""
    return list_accessible_patients(
        db=db,
        user=current_user,
        page=page,
        page_size=page_size,
        search=search,
    )


@router.get(
    "/{patient_id}",
    response_model=PatientResponse,
)
def get_patient_endpoint(
    patient_id: UUID,
    current_user: User = Depends(get_current_user),
    db: Session = Depends(get_db),
):
    """Get patient details by ID if authorized."""
    patient = verify_patient_access(db=db, user=current_user, patient_id=patient_id)
    doc_count = len(patient.documents) if patient.documents else 0
    return PatientResponse(
        patient_id=patient.patient_id,
        mrn=patient.mrn,
        first_name=patient.first_name,
        last_name=patient.last_name,
        date_of_birth=patient.date_of_birth,
        gender=patient.gender,
        created_at=patient.created_at,
        updated_at=patient.updated_at,
        document_count=doc_count,
    )


@router.put(
    "/{patient_id}",
    response_model=PatientResponse,
)
def update_patient_endpoint(
    patient_id: UUID,
    patient_in: PatientUpdate,
    current_user: User = Depends(get_current_user),
    db: Session = Depends(get_db),
):
    """Update patient details."""
    patient = update_patient(db=db, patient_id=patient_id, patient_in=patient_in, user=current_user)
    doc_count = len(patient.documents) if patient.documents else 0
    return PatientResponse(
        patient_id=patient.patient_id,
        mrn=patient.mrn,
        first_name=patient.first_name,
        last_name=patient.last_name,
        date_of_birth=patient.date_of_birth,
        gender=patient.gender,
        created_at=patient.created_at,
        updated_at=patient.updated_at,
        document_count=doc_count,
    )


@router.post(
    "/{patient_id}/assign",
    status_code=http_status.HTTP_200_OK,
    dependencies=[Depends(require_roles(UserRole.ADMIN.value, UserRole.RECORDS_STAFF.value))],
)
def assign_patient_endpoint(
    patient_id: UUID,
    assignment_in: PatientAssignmentCreate,
    current_user: User = Depends(get_current_user),
    db: Session = Depends(get_db),
):
    """Assign patient access to a user. Restricted to Admin or Records Staff."""
    assignment = assign_patient_to_user(
        db=db,
        patient_id=patient_id,
        target_user_id=assignment_in.user_id,
        assigner=current_user,
    )
    return {
        "status": "assigned",
        "assignment_id": assignment.assignment_id,
        "user_id": assignment.user_id,
        "patient_id": assignment.patient_id,
    }


@router.post(
    "/{patient_id}/search",
    response_model=SearchResponse,
)
def search_patient_documents_endpoint(
    patient_id: UUID,
    request: SearchRequest,
    current_user: User = Depends(get_current_user),
    db: Session = Depends(get_db),
):
    """
    Perform patient-scoped M6 RAG AI search across all authorized documents of the patient.
    Enforces server-side patient access authorization before retrieval.
    """
    verify_patient_access(db=db, user=current_user, patient_id=patient_id)
    return rag_service.search_patient(
        db=db,
        patient_id=patient_id,
        query=request.query,
        top_k=request.top_k,
    )
