import pytest
from uuid import uuid4
from sqlalchemy.orm import sessionmaker, Session

from app.models.document import Document
from app.models.document_text import DocumentText
from app.schemas.document import DocumentProcessingStatus
from app.services.document_service import store_ocr_result
from app.services.ocr_service import OCRResult


def test_privacy_pipeline_success_and_sanitization(database: sessionmaker[Session]):
    db = database()
    try:
        # 1. Create document
        doc = Document(
            patient_id=uuid4(),
            uploaded_by=uuid4(),
            file_name="test_clinical_note.pdf",
            file_type="application/pdf",
            file_url="/tmp/test_clinical_note.pdf",
            processing_status=DocumentProcessingStatus.PROCESSING.value,
            privacy_status="pending",
        )
        db.add(doc)
        db.commit()

        # 2. Simulate M2 OCR Result
        raw_ocr = """Patient Name: Rahul Kumar
DOB: 14/05/1998
MRN: MRN12345
Phone: 9876543210
Email: rahul@example.com

HbA1c: 8.2%
Creatinine: 1.4 mg/dL
Diagnosis: Type 2 Diabetes
Medication: Metformin 500 mg
"""
        ocr_result = OCRResult(
            raw_text=raw_ocr,
            page_count=1,
            confidence=0.98,
            layout={"blocks": []},
            ocr_engine="pytesseract",
            processing_time_ms=120,
        )

        # 3. Process via store_ocr_result (which runs PrivacyService automatically)
        doc_text = store_ocr_result(db, doc.document_id, ocr_result)

        db.refresh(doc)
        assert doc.processing_status == "Completed"
        assert doc.privacy_status == "completed"

        # Raw OCR preserved
        assert doc_text.raw_text == raw_ocr

        # Sanitized OCR produced
        assert doc_text.sanitized_text is not None
        assert "Rahul Kumar" not in doc_text.sanitized_text
        assert "rahul@example.com" not in doc_text.sanitized_text
        assert "[PATIENT]" in doc_text.sanitized_text

        # Clinical data preserved
        assert "HbA1c: 8.2%" in doc_text.sanitized_text
        assert "Creatinine: 1.4 mg/dL" in doc_text.sanitized_text
        assert "Metformin 500 mg" in doc_text.sanitized_text
    finally:
        db.close()


def test_privacy_pipeline_fail_safe(database: sessionmaker[Session], monkeypatch):
    db = database()
    try:
        doc = Document(
            patient_id=uuid4(),
            uploaded_by=uuid4(),
            file_name="test_fail_note.pdf",
            file_type="application/pdf",
            file_url="/tmp/test_fail_note.pdf",
            processing_status=DocumentProcessingStatus.PROCESSING.value,
            privacy_status="pending",
        )
        db.add(doc)
        db.commit()

        def mock_sanitize_failure(text):
            raise RuntimeError("Presidio execution failure")

        from app.services.privacy_service import privacy_service
        monkeypatch.setattr(privacy_service, "sanitize", mock_sanitize_failure)

        ocr_result = OCRResult(
            raw_text="Patient Name: Rahul Kumar",
            page_count=1,
            confidence=0.9,
            layout={},
            ocr_engine="pytesseract",
            processing_time_ms=50,
        )

        doc_text = store_ocr_result(db, doc.document_id, ocr_result)
        db.refresh(doc)

        # Fail-safe verification: status must be Failed, sanitized_text MUST be None
        assert doc.privacy_status == "failed"
        assert doc.processing_status == "Failed"
        assert doc_text.sanitized_text is None
    finally:
        db.close()


def test_sanitized_text_api_auth_and_privacy_checks(
    client,
    auth_headers,
    database: sessionmaker[Session],
    registered_user: dict,
):
    from uuid import UUID as _UUID
    db = database()
    try:
        # Create a completed privacy doc
        doc = Document(
            uploaded_by=_UUID(registered_user["user_id"]),
            file_name="completed.pdf",
            file_type="application/pdf",
            file_url="/tmp/completed.pdf",
            processing_status="Completed",
            privacy_status="completed",
        )
        db.add(doc)
        db.commit()
        db.refresh(doc)

        doc_text = DocumentText(
            document_id=doc.document_id,
            raw_text="Patient Name: Rahul Kumar\nHbA1c: 8.2%",
            sanitized_text="Patient Name: [PATIENT]\nHbA1c: 8.2%",
            ocr_engine="pytesseract",
        )
        db.add(doc_text)
        db.commit()

        document_id = str(doc.document_id)
    finally:
        db.close()

    # 1. Unauthenticated -> 401
    unauth_resp = client.get(f"/api/v1/documents/{document_id}/sanitized-text")
    assert unauth_resp.status_code == 401

    # 2. Authenticated owner -> 200 with sanitized_text
    auth_resp = client.get(
        f"/api/v1/documents/{document_id}/sanitized-text",
        headers=auth_headers,
    )
    assert auth_resp.status_code == 200
    assert auth_resp.json()["sanitized_text"] == "Patient Name: [PATIENT]\nHbA1c: 8.2%"
    assert "Rahul Kumar" not in auth_resp.json()["sanitized_text"]

    # 3. If privacy failed / pending -> 400 Bad Request
    db = database()
    try:
        failed_doc = Document(
            uploaded_by=_UUID(registered_user["user_id"]),
            file_name="failed.pdf",
            file_type="application/pdf",
            file_url="/tmp/failed.pdf",
            processing_status="Failed",
            privacy_status="failed",
        )
        db.add(failed_doc)
        db.commit()
        failed_id = str(failed_doc.document_id)
    finally:
        db.close()

    failed_resp = client.get(
        f"/api/v1/documents/{failed_id}/sanitized-text",
        headers=auth_headers,
    )
    assert failed_resp.status_code == 400
    assert "Sanitized text unavailable" in failed_resp.json()["detail"]
