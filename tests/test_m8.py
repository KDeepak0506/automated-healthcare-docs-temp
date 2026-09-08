from uuid import UUID, uuid4
from unittest.mock import MagicMock, patch
import pytest
from fastapi.testclient import TestClient
from sqlalchemy.orm import Session, sessionmaker

from app.models.document import Document
from app.models.document_chunk import DocumentChunk
from app.models.document_text import DocumentText


def _create_test_doc_and_chunk(db: Session, user_id) -> tuple[UUID, UUID]:
    uid = UUID(user_id) if isinstance(user_id, str) else user_id
    doc = Document(
        uploaded_by=uid,
        file_name="clinical_note.pdf",
        file_type="application/pdf",
        file_url="/tmp/clinical_note.pdf",
        processing_status="Completed",
        privacy_status="completed",
    )
    db.add(doc)
    db.commit()
    db.refresh(doc)

    doc_text = DocumentText(
        document_id=doc.document_id,
        raw_text="Patient: Rahul Kumar, MRN: 998811, HbA1c: 7.5%",
        sanitized_text="Patient: [PATIENT], MRN: [MRN], HbA1c: 7.5%",
        ocr_engine="pymupdf-native",
    )
    db.add(doc_text)

    chunk = DocumentChunk(
        document_id=doc.document_id,
        chunk_index=0,
        chunk_text="Patient: [PATIENT], MRN: [MRN], HbA1c: 7.5%",
        start_offset=0,
        end_offset=42,
        page_number=1,
    )
    db.add(chunk)
    db.commit()
    db.refresh(chunk)

    return doc.document_id, chunk.chunk_id


def test_get_chunk_source_success(
    client: TestClient,
    database: sessionmaker[Session],
    auth_headers: dict[str, str],
    registered_user: dict,
) -> None:
    """Test retrieving source chunk details for owned document (M8)."""
    user_id = UUID(registered_user["user_id"])
    with database() as db:
        doc_id, chunk_id = _create_test_doc_and_chunk(db, user_id)

    res = client.get(
        f"/api/v1/documents/{doc_id}/sources/{chunk_id}",
        headers=auth_headers,
    )
    assert res.status_code == 200
    data = res.json()
    assert data["document_id"] == str(doc_id)
    assert data["chunk_id"] == str(chunk_id)
    assert data["chunk_index"] == 0
    assert data["page_number"] == 1
    assert data["text"] == "Patient: [PATIENT], MRN: [MRN], HbA1c: 7.5%"
    # Ensure raw text is never exposed
    assert "Rahul Kumar" not in data["text"]


def test_get_chunk_source_unauthorized_user(
    client: TestClient,
    database: sessionmaker[Session],
    auth_headers: dict[str, str],
    registered_user: dict,
) -> None:
    """Test that another user cannot retrieve source chunks of a document they don't own."""
    user_id = UUID(registered_user["user_id"])
    with database() as db:
        doc_id, chunk_id = _create_test_doc_and_chunk(db, user_id)

    # Register second user
    client.post(
        "/api/v1/auth/register",
        json={
            "name": "Doctor Two",
            "email": "doctor2@example.com",
            "password": "password123",
            "role": "doctor",
        },
    )
    user2_login = client.post(
        "/api/v1/auth/login",
        data={"username": "doctor2@example.com", "password": "password123"},
    )
    user2_headers = {"Authorization": f"Bearer {user2_login.json()['access_token']}"}

    res = client.get(
        f"/api/v1/documents/{doc_id}/sources/{chunk_id}",
        headers=user2_headers,
    )
    assert res.status_code == 403
    assert res.json()["detail"] == "Access denied"


def test_get_chunk_source_not_found(
    client: TestClient,
    database: sessionmaker[Session],
    auth_headers: dict[str, str],
    registered_user: dict,
) -> None:
    """Test 404 response when requesting non-existent chunk ID or document ID."""
    user_id = UUID(registered_user["user_id"])
    with database() as db:
        doc_id, chunk_id = _create_test_doc_and_chunk(db, user_id)

    missing_chunk_id = uuid4()
    res = client.get(
        f"/api/v1/documents/{doc_id}/sources/{missing_chunk_id}",
        headers=auth_headers,
    )
    assert res.status_code == 404
    assert res.json()["detail"] == "Source chunk not found"
