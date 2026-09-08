import math
from datetime import datetime, timezone
from uuid import UUID

from fastapi import HTTPException, status
from sqlalchemy import func
from sqlalchemy.orm import Session

from app.models.document import Document
from app.models.patient import Patient, PatientAssignment
from app.models.user import User
from app.schemas.patient import PatientCreate, PatientUpdate
from app.schemas.user import UserRole


def has_patient_access(db: Session, user: User, patient_id: UUID) -> bool:
    """
    Check resource-level authorization for a patient.
    User is authorized if:
    1. User is Admin or Records Staff (system-wide clinical staff)
    2. An assignment exists in user_patient_assignments mapping user to patient
    3. User uploaded at least one document assigned to patient_id
    """
    if user.role in (UserRole.ADMIN.value, UserRole.RECORDS_STAFF.value):
        return True

    assignment = (
        db.query(PatientAssignment)
        .filter(
            PatientAssignment.user_id == user.user_id,
            PatientAssignment.patient_id == patient_id,
        )
        .first()
    )
    if assignment is not None:
        return True

    uploader_doc = (
        db.query(Document)
        .filter(
            Document.uploaded_by == user.user_id,
            Document.patient_id == patient_id,
        )
        .first()
    )
    return uploader_doc is not None


def has_document_access(db: Session, user: User, document: Document) -> bool:
    """
    Check access permission for a document.
    User is authorized if:
    1. User is document uploader
    2. User is Admin
    3. Document belongs to a patient and user has patient access
    """
    if document.uploaded_by == user.user_id:
        return True

    if user.role == UserRole.ADMIN.value:
        return True

    if document.patient_id is not None:
        return has_patient_access(db, user, document.patient_id)

    return False


def verify_patient_access(db: Session, user: User, patient_id: UUID) -> Patient:
    """Verify patient existence and user authorization. Raises 404 or 403 HTTP exception."""
    patient = db.query(Patient).filter(Patient.patient_id == patient_id).first()
    if patient is None:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="Patient not found",
        )

    if not has_patient_access(db, user, patient_id):
        raise HTTPException(
            status_code=status.HTTP_403_FORBIDDEN,
            detail="Access denied: You are not authorized to view this patient's records",
        )

    return patient


def create_patient(db: Session, patient_in: PatientCreate, creator: User) -> Patient:
    """Create a new patient record and assign creator automatically."""
    existing_mrn = db.query(Patient).filter(Patient.mrn == patient_in.mrn).first()
    if existing_mrn:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=f"Patient with MRN '{patient_in.mrn}' already exists",
        )

    patient = Patient(
        mrn=patient_in.mrn,
        first_name=patient_in.first_name,
        last_name=patient_in.last_name,
        date_of_birth=patient_in.date_of_birth,
        gender=patient_in.gender,
    )
    db.add(patient)
    db.commit()
    db.refresh(patient)

    # Auto-assign creator
    assignment = PatientAssignment(
        user_id=creator.user_id,
        patient_id=patient.patient_id,
    )
    db.add(assignment)
    db.commit()

    return patient


def update_patient(
    db: Session,
    patient_id: UUID,
    patient_in: PatientUpdate,
    user: User,
) -> Patient:
    patient = verify_patient_access(db, user, patient_id)

    if patient_in.first_name is not None:
        patient.first_name = patient_in.first_name
    if patient_in.last_name is not None:
        patient.last_name = patient_in.last_name
    if patient_in.date_of_birth is not None:
        patient.date_of_birth = patient_in.date_of_birth
    if patient_in.gender is not None:
        patient.gender = patient_in.gender

    patient.updated_at = datetime.now(timezone.utc)
    db.commit()
    db.refresh(patient)
    return patient


def assign_patient_to_user(
    db: Session,
    patient_id: UUID,
    target_user_id: UUID,
    assigner: User,
) -> PatientAssignment:
    """Assign a user to a patient. Restricted to Admin or Records Staff."""
    if assigner.role not in (UserRole.ADMIN.value, UserRole.RECORDS_STAFF.value):
        raise HTTPException(
            status_code=status.HTTP_403_FORBIDDEN,
            detail="Only Admin or Records Staff can assign patient access",
        )

    patient = db.query(Patient).filter(Patient.patient_id == patient_id).first()
    if not patient:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="Patient not found",
        )

    target_user = db.query(User).filter(User.user_id == target_user_id).first()
    if not target_user:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="Target user not found",
        )

    existing = (
        db.query(PatientAssignment)
        .filter(
            PatientAssignment.user_id == target_user_id,
            PatientAssignment.patient_id == patient_id,
        )
        .first()
    )
    if existing:
        return existing

    assignment = PatientAssignment(
        user_id=target_user_id,
        patient_id=patient_id,
    )
    db.add(assignment)
    db.commit()
    db.refresh(assignment)
    return assignment


def list_accessible_patients(
    db: Session,
    user: User,
    page: int = 1,
    page_size: int = 20,
    search: str | None = None,
) -> dict:
    """List patients accessible to the user with search and pagination."""
    query = db.query(Patient)

    if user.role not in (UserRole.ADMIN.value, UserRole.RECORDS_STAFF.value):
        # Filter patients assigned to user or containing user's uploaded documents
        assigned_patient_ids = (
            db.query(PatientAssignment.patient_id)
            .filter(PatientAssignment.user_id == user.user_id)
            .scalar_subquery()
        )
        uploader_patient_ids = (
            db.query(Document.patient_id)
            .filter(
                Document.uploaded_by == user.user_id,
                Document.patient_id.isnot(None),
            )
            .scalar_subquery()
        )

        query = query.filter(
            (Patient.patient_id.in_(assigned_patient_ids))
            | (Patient.patient_id.in_(uploader_patient_ids))
        )

    if search:
        search_term = f"%{search.strip()}%"
        query = query.filter(
            (Patient.mrn.ilike(search_term))
            | (Patient.first_name.ilike(search_term))
            | (Patient.last_name.ilike(search_term))
        )

    total = query.count()
    total_pages = math.ceil(total / page_size) if page_size > 0 else 1

    patients = (
        query.order_by(Patient.created_at.desc())
        .offset((page - 1) * page_size)
        .limit(page_size)
        .all()
    )

    # Attach document count to each patient item
    items = []
    for p in patients:
        doc_count = (
            db.query(func.count(Document.document_id))
            .filter(Document.patient_id == p.patient_id)
            .scalar()
            or 0
        )
        item_dict = {
            "patient_id": p.patient_id,
            "mrn": p.mrn,
            "first_name": p.first_name,
            "last_name": p.last_name,
            "date_of_birth": p.date_of_birth,
            "gender": p.gender,
            "created_at": p.created_at,
            "updated_at": p.updated_at,
            "document_count": doc_count,
        }
        items.append(item_dict)

    return {
        "items": items,
        "page": page,
        "page_size": page_size,
        "total": total,
        "total_pages": max(total_pages, 1),
    }
