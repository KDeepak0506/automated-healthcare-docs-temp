from unittest.mock import MagicMock
from uuid import UUID, uuid4
import pytest
from fastapi.testclient import TestClient
from sqlalchemy.orm import Session, sessionmaker

from app.core.config import settings
from app.models.document import Document
from app.models.document_text import DocumentText
from app.schemas.summary import SummaryResult
from app.services import summary_service as sum_service_module
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
    raw_text: str = "Patient Name: Rahul Kumar\nMRN: MRN12345\nHbA1c: 8.2%\nCreatinine: 1.41 mg/dL",
    sanitized_text: str | None = "Patient Name: [PATIENT]\nMRN: [MRN]\nHbA1c: 8.2%\nCreatinine: 1.41 mg/dL",
) -> Document:
    doc = Document(
        uploaded_by=user_id,
        file_name="lab_summary.pdf",
        file_type="application/pdf",
        file_url="/tmp/lab_summary.pdf",
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


def test_summarize_document_happy_path(
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
        return_value=SummaryResult(
            summary="Clinical laboratory report indicates elevated HbA1c and serum creatinine levels.",
            key_findings=[
                "HbA1c: 8.2%",
                "Creatinine: 1.41 mg/dL",
            ],
        )
    )
    monkeypatch.setattr(sum_service_module.summary_service.llm, "generate_structured", mock_generate)

    response = client.post(f"/api/v1/documents/{doc_id}/summarize", headers=auth_headers)

    assert response.status_code == 200
    data = response.json()
    assert data["document_id"] == doc_id
    assert "elevated HbA1c" in data["summary"]
    assert "HbA1c: 8.2%" in data["key_findings"]
    assert "Creatinine: 1.41 mg/dL" in data["key_findings"]
    assert data["cached"] is False
    assert data["truncated"] is False
    mock_generate.assert_called_once()

    # Verify persisted in database
    check_db = database()
    try:
        saved_doc = check_db.query(Document).filter(Document.document_id == UUID(doc_id)).first()
        assert saved_doc is not None
        assert saved_doc.summary == "Clinical laboratory report indicates elevated HbA1c and serum creatinine levels."
        assert "HbA1c: 8.2%" in saved_doc.key_findings
    finally:
        check_db.close()


def test_summarize_document_caching_idempotency(
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
        return_value=SummaryResult(
            summary="Initial summary generation.",
            key_findings=["Finding 1"],
        )
    )
    monkeypatch.setattr(sum_service_module.summary_service.llm, "generate_structured", mock_generate)

    # First call invokes LLM
    resp1 = client.post(f"/api/v1/documents/{doc_id}/summarize", headers=auth_headers)
    assert resp1.status_code == 200
    assert resp1.json()["cached"] is False
    assert mock_generate.call_count == 1

    # Second call returns cached result without LLM call
    resp2 = client.post(f"/api/v1/documents/{doc_id}/summarize", headers=auth_headers)
    assert resp2.status_code == 200
    assert resp2.json()["cached"] is True
    assert resp2.json()["summary"] == "Initial summary generation."
    assert mock_generate.call_count == 1


def test_summarize_document_unauthorized_access(
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

    response = client.post(f"/api/v1/documents/{doc_id}/summarize", headers=auth_headers)
    assert response.status_code == 403
    assert response.json()["detail"] == "Access denied"


def test_summarize_document_unauthenticated(
    client: TestClient,
    database: sessionmaker[Session],
) -> None:
    db = database()
    try:
        doc = _create_test_document(db, uuid4())
        doc_id = str(doc.document_id)
    finally:
        db.close()

    response = client.post(f"/api/v1/documents/{doc_id}/summarize")
    assert response.status_code == 401


def test_summarize_document_privacy_boundary_strictly_enforced(
    client: TestClient,
    auth_headers: dict[str, str],
    database: sessionmaker[Session],
    registered_user: dict,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Explicitly verify that raw OCR containing PHI is NEVER passed to the LLM during summarization."""
    raw_ocr = "Patient Name: Rahul Kumar\nMRN: PRIVATE123\nPhone: 9876543210\nHbA1c: 8.2%\nCreatinine: 1.41 mg/dL"
    sanitized_ocr = "Patient Name: [PATIENT]\nMRN: [MRN]\nPhone: [PHONE]\nHbA1c: 8.2%\nCreatinine: 1.41 mg/dL"

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
        return SummaryResult(
            summary="Grounded summary of laboratory results.",
            key_findings=["HbA1c: 8.2%", "Creatinine: 1.41 mg/dL"],
        )

    monkeypatch.setattr(sum_service_module.summary_service.llm, "generate_structured", mock_llm_call)

    response = client.post(f"/api/v1/documents/{doc_id}/summarize", headers=auth_headers)
    assert response.status_code == 200

    assert len(captured_prompt) == 1
    sent_text = captured_prompt[0]

    # Sanitized tags must be present
    assert "[PATIENT]" in sent_text
    assert "[MRN]" in sent_text
    assert "[PHONE]" in sent_text
    assert "HbA1c: 8.2%" in sent_text
    assert "Creatinine: 1.41 mg/dL" in sent_text

    # Raw PHI must NEVER be present
    assert "Rahul Kumar" not in sent_text
    assert "PRIVATE123" not in sent_text
    assert "9876543210" not in sent_text


def test_summarize_document_blocked_when_privacy_incomplete(
    client: TestClient,
    auth_headers: dict[str, str],
    database: sessionmaker[Session],
    registered_user: dict,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    db = database()
    try:
        user_id = UUID(registered_user["user_id"])
        doc = _create_test_document(db, user_id, privacy_status="failed", sanitized_text=None)
        doc_id = str(doc.document_id)
    finally:
        db.close()

    mock_generate = MagicMock()
    monkeypatch.setattr(sum_service_module.summary_service.llm, "generate_structured", mock_generate)

    response = client.post(f"/api/v1/documents/{doc_id}/summarize", headers=auth_headers)
    assert response.status_code == 400
    assert "Sanitized text unavailable" in response.json()["detail"]
    mock_generate.assert_not_called()


def test_summarize_document_numerical_faithfulness(
    client: TestClient,
    auth_headers: dict[str, str],
    database: sessionmaker[Session],
    registered_user: dict,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Verify prompt instructs exact preservation of numerical measurements and units."""
    db = database()
    try:
        user_id = UUID(registered_user["user_id"])
        doc = _create_test_document(
            db,
            user_id,
            sanitized_text="WBC: 13,200 cells/uL\nCreatinine: 1.41 mg/dL\nHbA1c: 8.2%",
        )
        doc_id = str(doc.document_id)
    finally:
        db.close()

    captured_prompt = []

    def mock_llm_call(prompt, response_schema, system_prompt=None):
        captured_prompt.append(prompt)
        return SummaryResult(
            summary="Patient has elevated WBC of 13,200 cells/uL, Creatinine 1.41 mg/dL, and HbA1c 8.2%.",
            key_findings=[
                "WBC: 13,200 cells/uL",
                "Creatinine: 1.41 mg/dL",
                "HbA1c: 8.2%",
            ],
        )

    monkeypatch.setattr(sum_service_module.summary_service.llm, "generate_structured", mock_llm_call)

    response = client.post(f"/api/v1/documents/{doc_id}/summarize", headers=auth_headers)
    assert response.status_code == 200

    # Ensure system prompt and user prompt emphasize numerical preservation
    prompt_text = captured_prompt[0]
    assert "Numerical Faithfulness" in prompt_text or "numerical values" in prompt_text.lower()
    assert "13,200 cells/uL" in response.json()["key_findings"][0]
    assert "1.41 mg/dL" in response.json()["key_findings"][1]
    assert "8.2%" in response.json()["key_findings"][2]


def test_summarize_document_oversized_text_truncation(
    client: TestClient,
    auth_headers: dict[str, str],
    database: sessionmaker[Session],
    registered_user: dict,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    db = database()
    try:
        user_id = UUID(registered_user["user_id"])
        oversized_text = "Medical discharge notes section. " * 3000
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
        return SummaryResult(
            summary="Summary of truncated discharge notes.",
            key_findings=["Key finding 1"],
        )

    monkeypatch.setattr(sum_service_module.summary_service.llm, "generate_structured", mock_llm_call)

    response = client.post(f"/api/v1/documents/{doc_id}/summarize", headers=auth_headers)
    assert response.status_code == 200
    data = response.json()
    assert data["truncated"] is True
    assert len(captured_prompt[0]) < 55000


def test_summarize_document_timeout_error(
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

    monkeypatch.setattr(sum_service_module.summary_service.llm, "generate_structured", mock_timeout)

    response = client.post(f"/api/v1/documents/{doc_id}/summarize", headers=auth_headers)
    assert response.status_code == 504
    assert "timed out" in response.json()["detail"]


def test_summarize_document_parsing_error(
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

    def mock_parse_err(*args, **kwargs):
        raise LLMResponseParsingError("Invalid JSON structure")

    monkeypatch.setattr(sum_service_module.summary_service.llm, "generate_structured", mock_parse_err)

    response = client.post(f"/api/v1/documents/{doc_id}/summarize", headers=auth_headers)
    assert response.status_code == 502
    assert "Invalid response" in response.json()["detail"]


def test_summarize_document_missing_api_key(
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

    def mock_config_err(*args, **kwargs):
        raise LLMConfigurationError("Missing API key")

    monkeypatch.setattr(sum_service_module.summary_service.llm, "generate_structured", mock_config_err)

    response = client.post(f"/api/v1/documents/{doc_id}/summarize", headers=auth_headers)
    assert response.status_code == 503
    assert "not configured" in response.json()["detail"]
