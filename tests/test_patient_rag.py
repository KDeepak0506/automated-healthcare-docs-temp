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


def test_query_classification() -> None:
    """Verify clinical query intent classification routes correctly."""
    assert rag_service.classify_query("What were the imaging findings?") == "IMAGING_DIAGNOSTIC"
    assert rag_service.classify_query("Did the patient have an ECG or chest X-ray?") == "IMAGING_DIAGNOSTIC"
    assert rag_service.classify_query("What did the ultrasound show?") == "IMAGING_DIAGNOSTIC"
    assert rag_service.classify_query("List all abnormal findings") == "COMPREHENSIVE_ABNORMAL"
    assert rag_service.classify_query("List all H-flagged values") == "COMPREHENSIVE_ABNORMAL"
    assert rag_service.classify_query("Summarize all lab results") == "COMPREHENSIVE_ABNORMAL"
    assert rag_service.classify_query("What is the patient's hemoglobin level?") == "NARROW_FACTUAL"
    assert rag_service.classify_query("What dosage of metformin was prescribed?") == "NARROW_FACTUAL"


def test_patient_rag_balanced_cross_document_retrieval(database: sessionmaker[Session], monkeypatch: pytest.MonkeyPatch) -> None:
    """Narrow factual queries should retrieve relevant chunks from multiple documents without one document dominating."""
    captured_prompt = []

    def mock_generate_text(prompt: str, system_prompt: str | None = None, temperature: float = 0.1) -> str:
        captured_prompt.append(prompt)
        return "Hemoglobin is 14.5 g/dL in Doc1, 13.6 gm/dL in Doc2, and 15 g/dl in Doc3."

    monkeypatch.setattr(rag_service.llm, "generate_text", mock_generate_text)

    with database() as db:
        doctor = _make_user(db, "balanced.rag@test.org", UserRole.DOCTOR.value)

        patient = create_patient(
            db,
            PatientCreate(mrn="PAT-BALANCED-01", first_name="Balanced", last_name="Patient"),
            creator=doctor,
        )

        doc1 = Document(
            patient_id=patient.patient_id,
            uploaded_by=doctor.user_id,
            file_name="Lab_A.pdf",
            file_type="application/pdf",
            file_url="/tmp/docA.pdf",
            processing_status="Completed",
            privacy_status="completed",
        )
        doc2 = Document(
            patient_id=patient.patient_id,
            uploaded_by=doctor.user_id,
            file_name="Lab_B.pdf",
            file_type="application/pdf",
            file_url="/tmp/docB.pdf",
            processing_status="Completed",
            privacy_status="completed",
        )
        doc3 = Document(
            patient_id=patient.patient_id,
            uploaded_by=doctor.user_id,
            file_name="Lab_C.pdf",
            file_type="application/pdf",
            file_url="/tmp/docC.pdf",
            processing_status="Completed",
            privacy_status="completed",
        )
        db.add_all([doc1, doc2, doc3])
        db.commit()

        dt1 = DocumentText(
            document_id=doc1.document_id,
            raw_text="RAW",
            sanitized_text="CBC Panel: Hemoglobin: 14.5 g/dL (Reference 13.0 - 16.5). WBC Count: 7,000 /cmm.",
            ocr_engine="test",
        )
        dt2 = DocumentText(
            document_id=doc2.document_id,
            raw_text="RAW",
            sanitized_text="Blood Examination: Hemoglobin 13.6 gm/dL (Reference 11.5 - 14.5). Platelets: 250,000.",
            ocr_engine="test",
        )
        dt3 = DocumentText(
            document_id=doc3.document_id,
            raw_text="RAW",
            sanitized_text="Haematology Report: HEMOGLOBIN 15 g/dl (Reference 13 - 17). RBC: 5.0.",
            ocr_engine="test",
        )
        db.add_all([dt1, dt2, dt3])
        db.commit()

        rag_service.index_document(db, doc1)
        rag_service.index_document(db, doc2)
        rag_service.index_document(db, doc3)

        response = rag_service.search_patient(
            db=db,
            patient_id=patient.patient_id,
            query="What is the patient's hemoglobin level?",
            top_k=5,
        )

        assert response.answer is not None
        # All three documents should be represented in the sources
        source_doc_ids = {s.document_id for s in response.sources if s.document_id}
        assert source_doc_ids == {doc1.document_id, doc2.document_id, doc3.document_id}
        assert len(captured_prompt) == 1
        assert "Lab_A.pdf" in captured_prompt[0]
        assert "Lab_B.pdf" in captured_prompt[0]
        assert "Lab_C.pdf" in captured_prompt[0]


def test_patient_rag_comprehensive_abnormal_routing(database: sessionmaker[Session], monkeypatch: pytest.MonkeyPatch) -> None:
    """Comprehensive queries for abnormal findings should route and gather flagged chunks across all docs."""
    captured_prompt = []

    def mock_generate_text(prompt: str, system_prompt: str | None = None, temperature: float = 0.1) -> str:
        captured_prompt.append(prompt)
        return "Abnormal findings: Fasting blood sugar is High (H) at 142 mg/dL. Urinary Glucose is 1+ (abnormal)."

    monkeypatch.setattr(rag_service.llm, "generate_text", mock_generate_text)

    with database() as db:
        doctor = _make_user(db, "abnormal.rag@test.org", UserRole.DOCTOR.value)

        patient = create_patient(
            db,
            PatientCreate(mrn="PAT-ABN-01", first_name="Abnormal", last_name="Findings"),
            creator=doctor,
        )

        doc1 = Document(
            patient_id=patient.patient_id,
            uploaded_by=doctor.user_id,
            file_name="Lab_Report_Flags.pdf",
            file_type="application/pdf",
            file_url="/tmp/doc_flags.pdf",
            processing_status="Completed",
            privacy_status="completed",
        )
        doc2 = Document(
            patient_id=patient.patient_id,
            uploaded_by=doctor.user_id,
            file_name="Urine_Report.pdf",
            file_type="application/pdf",
            file_url="/tmp/doc_urine.pdf",
            processing_status="Completed",
            privacy_status="completed",
        )
        db.add_all([doc1, doc2])
        db.commit()

        dt1 = DocumentText(
            document_id=doc1.document_id,
            raw_text="RAW",
            sanitized_text="Fasting Blood Sugar H 142 mg/dL (Reference 74 - 106). HbA1c H 8.2 %.",
            ocr_engine="test",
        )
        dt2 = DocumentText(
            document_id=doc2.document_id,
            raw_text="RAW",
            sanitized_text="Urinary Glucose 1+ Negative. Urinary pH 5.5 (6.0 - 8.0).",
            ocr_engine="test",
        )
        db.add_all([dt1, dt2])
        db.commit()

        rag_service.index_document(db, doc1)
        rag_service.index_document(db, doc2)

        response = rag_service.search_patient(
            db=db,
            patient_id=patient.patient_id,
            query="List all abnormal findings",
        )

        assert response.answer is not None
        source_doc_ids = {s.document_id for s in response.sources if s.document_id}
        assert source_doc_ids == {doc1.document_id, doc2.document_id}
        assert len(captured_prompt) == 1
        assert "Lab_Report_Flags.pdf" in captured_prompt[0]
        assert "Urine_Report.pdf" in captured_prompt[0]
