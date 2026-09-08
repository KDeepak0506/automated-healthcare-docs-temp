"""Tests for patient-scoped M6 RAG AI search across multiple documents."""
import pytest
from sqlalchemy.orm import Session, sessionmaker

from app.models.document import Document
from app.models.document_text import DocumentText
from app.models.user import User
from app.schemas.patient import PatientCreate
from app.schemas.user import UserRole
from app.services.patient_service import create_patient
from app.services.rag_service import rag_service


def _make_user(db: Session, email: str, role: str) -> User:
    user = User(name=f"User {email}", email=email, password_hash="hashed", role=role)
    db.add(user)
    db.commit()
    db.refresh(user)
    return user


def test_patient_rag_returns_answer_from_both_docs(database: sessionmaker[Session], monkeypatch: pytest.MonkeyPatch) -> None:
    """Patient-scoped RAG should search across all authorized patient documents."""
    def mock_generate_text(prompt: str, system_prompt: str | None = None, temperature: float = 0.1) -> str:
        return "The patient has a hemoglobin of 14.2 g/dL and is diagnosed with Type 2 Diabetes."

    monkeypatch.setattr(rag_service.llm, "generate_text", mock_generate_text)

    with database() as db:
        doctor = _make_user(db, "rag.doctor@test.org", UserRole.DOCTOR.value)

        patient = create_patient(
            db,
            PatientCreate(mrn="PAT-RAG-01", first_name="Gregory", last_name="House"),
            creator=doctor,
        )

        # Create two documents for this patient
        doc1 = Document(
            patient_id=patient.patient_id,
            uploaded_by=doctor.user_id,
            file_name="Lab_Report.pdf",
            file_type="application/pdf",
            file_url="/tmp/doc1.pdf",
            processing_status="Completed",
            privacy_status="completed",
        )
        doc2 = Document(
            patient_id=patient.patient_id,
            uploaded_by=doctor.user_id,
            file_name="Discharge_Summary.pdf",
            file_type="application/pdf",
            file_url="/tmp/doc2.pdf",
            processing_status="Completed",
            privacy_status="completed",
        )
        db.add_all([doc1, doc2])
        db.commit()

        dt1 = DocumentText(
            document_id=doc1.document_id,
            raw_text="RAW_ONLY",
            sanitized_text="Hemoglobin level is 14.2 g/dL. Platelet count is within normal range.",
            ocr_engine="test",
        )
        dt2 = DocumentText(
            document_id=doc2.document_id,
            raw_text="RAW_ONLY",
            sanitized_text="Patient diagnosed with Type 2 Diabetes mellitus. Prescribed Metformin 500mg daily.",
            ocr_engine="test",
        )
        db.add_all([dt1, dt2])
        db.commit()

        # Index both docs
        rag_service.index_document(db, doc1)
        rag_service.index_document(db, doc2)

        # Patient-scoped search
        response = rag_service.search_patient(
            db=db,
            patient_id=patient.patient_id,
            query="What is the patient's hemoglobin and diagnosis?",
        )

        assert response.answer is not None
        assert len(response.answer) > 10
        assert len(response.sources) > 0

        # Source document IDs should be from patient's docs
        source_doc_ids = {s.document_id for s in response.sources if s.document_id}
        assert source_doc_ids.issubset({doc1.document_id, doc2.document_id})


def test_patient_rag_returns_empty_answer_when_no_docs(database: sessionmaker[Session]) -> None:
    """Patient-scoped RAG returns graceful empty message when no processed docs exist."""
    with database() as db:
        doctor = _make_user(db, "empty.rag@test.org", UserRole.DOCTOR.value)

        patient = create_patient(
            db,
            PatientCreate(mrn="PAT-RAG-EMPTY", first_name="Empty", last_name="Case"),
            creator=doctor,
        )

        # No documents added
        response = rag_service.search_patient(
            db=db,
            patient_id=patient.patient_id,
            query="What medications is this patient on?",
        )

        assert response.answer is not None
        assert "no" in response.answer.lower() or "unavailable" in response.answer.lower()
        assert len(response.sources) == 0
