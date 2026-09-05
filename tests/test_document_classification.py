from unittest.mock import MagicMock
from uuid import UUID, uuid4
import pytest
from fastapi.testclient import TestClient
from sqlalchemy.orm import Session, sessionmaker

from app.core.config import settings
from app.models.document import Document
from app.models.document_text import DocumentText
from app.schemas.classification import ClassificationResult
from app.services import classification_service as class_service_module
from app.services.llm_service import (
    LLMConfigurationError,
    LLMResponseParsingError,
    LLMServiceError,
    LLMTimeoutError,
)


def _create_test_document(
    db: Session,
    user_id: UUID,
    privacy_status: str = "completed",
    raw_text: str = "Patient Name: Rahul Kumar\nMRN: MRN12345\nHbA1c: 8.2%",
    sanitized_text: str | None = "Patient Name: [PATIENT]\nMRN: [MRN]\nHbA1c: 8.2%",
) -> Document:
    doc = Document(
        uploaded_by=user_id,
        file_name="lab_report.pdf",
        file_type="application/pdf",
        file_url="/tmp/lab_report.pdf",
        processing_status="Completed" if privacy_status == "completed" else "Processing",
        privacy_status=privacy_status,
    )
    db.add(doc)
    db.commit()
    db.refresh(doc)

    doc_text = DocumentText(
        document_id=doc.document_id,
        raw_text=raw_text,
        sanitized_text=sanitized_text,
        ocr_engine="pymupdf-native",
    )
    db.add(doc_text)
    db.commit()
    db.refresh(doc)
    return doc


def test_classify_document_happy_path(
    client: TestClient,
    auth_headers: dict[str, str],
    database: sessionmaker[Session],
    registered_user: dict,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    db = database()
    try:
        user_id = UUID(registered_user["user_id"])
        doc = _create_test_document(db, user_id)
        doc_id = str(doc.document_id)
    finally:
        db.close()

    mock_generate = MagicMock(
        return_value=ClassificationResult(
            document_type="Laboratory Report",
            classification_confidence=0.96,
        )
    )
    monkeypatch.setattr(class_service_module.classification_service.llm, "generate_structured", mock_generate)

    response = client.post(f"/api/v1/documents/{doc_id}/classify", headers=auth_headers)

    assert response.status_code == 200
    data = response.json()
    assert data["document_id"] == doc_id
    assert data["document_type"] == "Laboratory Report"
    assert data["classification_confidence"] == 0.96
    assert data["cached"] is False
    assert data["truncated"] is False
    mock_generate.assert_called_once()

    # Verify persisted in database
    check_db = database()
    try:
        saved_doc = check_db.query(Document).filter(Document.document_id == UUID(doc_id)).first()
        assert saved_doc is not None
        assert saved_doc.document_type == "Laboratory Report"
        assert saved_doc.classification_confidence == 0.96
    finally:
        check_db.close()


def test_classify_document_caching_idempotency(
    client: TestClient,
    auth_headers: dict[str, str],
    database: sessionmaker[Session],
    registered_user: dict,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    db = database()
    try:
        user_id = UUID(registered_user["user_id"])
        doc = _create_test_document(db, user_id)
        doc_id = str(doc.document_id)
    finally:
        db.close()

    mock_generate = MagicMock(
        return_value=ClassificationResult(
            document_type="Prescription",
            classification_confidence=0.92,
        )
    )
    monkeypatch.setattr(class_service_module.classification_service.llm, "generate_structured", mock_generate)

    # First call calls LLM
    resp1 = client.post(f"/api/v1/documents/{doc_id}/classify", headers=auth_headers)
    assert resp1.status_code == 200
    assert resp1.json()["cached"] is False
    assert mock_generate.call_count == 1

    # Second call returns cached result without invoking LLM
    resp2 = client.post(f"/api/v1/documents/{doc_id}/classify", headers=auth_headers)
    assert resp2.status_code == 200
    assert resp2.json()["cached"] is True
    assert resp2.json()["document_type"] == "Prescription"
    assert resp2.json()["classification_confidence"] == 0.92
    assert mock_generate.call_count == 1


def test_classify_document_unauthorized_access(
    client: TestClient,
    auth_headers: dict[str, str],
    database: sessionmaker[Session],
) -> None:
    db = database()
    try:
        other_user_id = uuid4()
        doc = _create_test_document(db, other_user_id)
        doc_id = str(doc.document_id)
    finally:
        db.close()

    response = client.post(f"/api/v1/documents/{doc_id}/classify", headers=auth_headers)
    assert response.status_code == 403
    assert response.json()["detail"] == "Access denied"


def test_classify_document_unauthenticated(
    client: TestClient,
    database: sessionmaker[Session],
) -> None:
    db = database()
    try:
        doc = _create_test_document(db, uuid4())
        doc_id = str(doc.document_id)
    finally:
        db.close()

    response = client.post(f"/api/v1/documents/{doc_id}/classify")
    assert response.status_code == 401


def test_classify_document_privacy_boundary_strictly_enforced(
    client: TestClient,
    auth_headers: dict[str, str],
    database: sessionmaker[Session],
    registered_user: dict,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Explicitly verify that raw PHI/OCR is NEVER sent to the LLM; only sanitized text is sent."""
    raw_ocr = "Patient Name: Rahul Kumar\nMRN: SECRETPATIENT123\nPhone: 9876543210\nHbA1c: 8.2%"
    sanitized_ocr = "Patient Name: [PATIENT]\nMRN: [MRN]\nPhone: [PHONE]\nHbA1c: 8.2%"

    db = database()
    try:
        user_id = UUID(registered_user["user_id"])
        doc = _create_test_document(
            db,
            user_id,
            privacy_status="completed",
            raw_text=raw_ocr,
            sanitized_text=sanitized_ocr,
        )
        doc_id = str(doc.document_id)
    finally:
        db.close()

    captured_prompt = []

    def mock_llm_call(prompt, response_schema, system_prompt=None):
        captured_prompt.append(prompt)
        return ClassificationResult(
            document_type="Laboratory Report",
            classification_confidence=0.95,
        )

    monkeypatch.setattr(class_service_module.classification_service.llm, "generate_structured", mock_llm_call)

    response = client.post(f"/api/v1/documents/{doc_id}/classify", headers=auth_headers)
    assert response.status_code == 200

    assert len(captured_prompt) == 1
    sent_text = captured_prompt[0]

    # Must contain sanitized tags
    assert "[PATIENT]" in sent_text
    assert "[MRN]" in sent_text
    assert "[PHONE]" in sent_text
    assert "HbA1c: 8.2%" in sent_text

    # Must NEVER contain raw PHI
    assert "Rahul Kumar" not in sent_text
    assert "SECRETPATIENT123" not in sent_text
    assert "9876543210" not in sent_text


def test_classify_document_blocked_when_privacy_incomplete(
    client: TestClient,
    auth_headers: dict[str, str],
    database: sessionmaker[Session],
    registered_user: dict,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    db = database()
    try:
        user_id = UUID(registered_user["user_id"])
        doc = _create_test_document(db, user_id, privacy_status="pending", sanitized_text=None)
        doc_id = str(doc.document_id)
    finally:
        db.close()

    mock_generate = MagicMock()
    monkeypatch.setattr(class_service_module.classification_service.llm, "generate_structured", mock_generate)

    response = client.post(f"/api/v1/documents/{doc_id}/classify", headers=auth_headers)
    assert response.status_code == 400
    assert "Sanitized text unavailable" in response.json()["detail"]
    mock_generate.assert_not_called()


def test_classify_document_empty_sanitized_text(
    client: TestClient,
    auth_headers: dict[str, str],
    database: sessionmaker[Session],
    registered_user: dict,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    db = database()
    try:
        user_id = UUID(registered_user["user_id"])
        doc = _create_test_document(db, user_id, privacy_status="completed", sanitized_text="   ")
        doc_id = str(doc.document_id)
    finally:
        db.close()

    mock_generate = MagicMock()
    monkeypatch.setattr(class_service_module.classification_service.llm, "generate_structured", mock_generate)

    response = client.post(f"/api/v1/documents/{doc_id}/classify", headers=auth_headers)
    assert response.status_code == 400
    assert "Sanitized OCR text is empty" in response.json()["detail"]
    mock_generate.assert_not_called()


def test_classify_document_oversized_text_truncation(
    client: TestClient,
    auth_headers: dict[str, str],
    database: sessionmaker[Session],
    registered_user: dict,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    db = database()
    try:
        user_id = UUID(registered_user["user_id"])
        oversized_text = "Clinical note line. " * 3000  # ~60,000 characters
        doc = _create_test_document(
            db,
            user_id,
            privacy_status="completed",
            raw_text=oversized_text,
            sanitized_text=oversized_text,
        )
        doc_id = str(doc.document_id)
    finally:
        db.close()

    captured_prompt = []

    def mock_llm_call(prompt, response_schema, system_prompt=None):
        captured_prompt.append(prompt)
        return ClassificationResult(
            document_type="Discharge Summary",
            classification_confidence=0.88,
        )

    monkeypatch.setattr(class_service_module.classification_service.llm, "generate_structured", mock_llm_call)

    response = client.post(f"/api/v1/documents/{doc_id}/classify", headers=auth_headers)
    assert response.status_code == 200
    data = response.json()
    assert data["truncated"] is True
    assert data["document_type"] == "Discharge Summary"
    # Ensure sent text is within limit
    assert len(captured_prompt[0]) < 55000


def test_classify_document_other_category_supported(
    client: TestClient,
    auth_headers: dict[str, str],
    database: sessionmaker[Session],
    registered_user: dict,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    db = database()
    try:
        user_id = UUID(registered_user["user_id"])
        doc = _create_test_document(db, user_id)
        doc_id = str(doc.document_id)
    finally:
        db.close()

    mock_generate = MagicMock(
        return_value=ClassificationResult(
            document_type="Other",
            classification_confidence=0.80,
        )
    )
    monkeypatch.setattr(class_service_module.classification_service.llm, "generate_structured", mock_generate)

    response = client.post(f"/api/v1/documents/{doc_id}/classify", headers=auth_headers)
    assert response.status_code == 200
    assert response.json()["document_type"] == "Other"
    assert response.json()["classification_confidence"] == 0.80


def test_classify_document_timeout_error(
    client: TestClient,
    auth_headers: dict[str, str],
    database: sessionmaker[Session],
    registered_user: dict,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    db = database()
    try:
        user_id = UUID(registered_user["user_id"])
        doc = _create_test_document(db, user_id)
        doc_id = str(doc.document_id)
    finally:
        db.close()

    def mock_timeout(*args, **kwargs):
        raise LLMTimeoutError("Groq timeout")

    monkeypatch.setattr(class_service_module.classification_service.llm, "generate_structured", mock_timeout)

    response = client.post(f"/api/v1/documents/{doc_id}/classify", headers=auth_headers)
    assert response.status_code == 504
    assert "timed out" in response.json()["detail"]


def test_classify_document_provider_error(
    client: TestClient,
    auth_headers: dict[str, str],
    database: sessionmaker[Session],
    registered_user: dict,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    db = database()
    try:
        user_id = UUID(registered_user["user_id"])
        doc = _create_test_document(db, user_id)
        doc_id = str(doc.document_id)
    finally:
        db.close()

    def mock_error(*args, **kwargs):
        raise LLMServiceError("Groq 500 error")

    monkeypatch.setattr(class_service_module.classification_service.llm, "generate_structured", mock_error)

    response = client.post(f"/api/v1/documents/{doc_id}/classify", headers=auth_headers)
    assert response.status_code == 502
    assert "Failed to classify" in response.json()["detail"]


def test_classify_document_missing_api_key(
    client: TestClient,
    auth_headers: dict[str, str],
    database: sessionmaker[Session],
    registered_user: dict,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    db = database()
    try:
        user_id = UUID(registered_user["user_id"])
        doc = _create_test_document(db, user_id)
        doc_id = str(doc.document_id)
    finally:
        db.close()

    def mock_config_error(*args, **kwargs):
        raise LLMConfigurationError("Missing API key")

    monkeypatch.setattr(class_service_module.classification_service.llm, "generate_structured", mock_config_error)

    response = client.post(f"/api/v1/documents/{doc_id}/classify", headers=auth_headers)
    assert response.status_code == 503
    assert "not configured" in response.json()["detail"]
