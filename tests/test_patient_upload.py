"""Tests for patient-specific document upload: authorization, association, and rejection."""
from pathlib import Path

import pytest
from fastapi.testclient import TestClient

from app.services import document_service
from app.services.ocr_service import OCRResult


# ---------------------------------------------------------------------------
# Shared helpers
# ---------------------------------------------------------------------------

def _mock_ocr(monkeypatch, tmp_path: Path) -> None:
    """Patch UPLOAD_DIR and extract_text so uploads do not require Tesseract."""
    monkeypatch.setattr(document_service, "UPLOAD_DIR", tmp_path)
    monkeypatch.setattr(
        document_service,
        "extract_text",
        lambda file_path, file_type: OCRResult(
            raw_text="Patient document text",
            page_count=1,
            confidence=0.99,
            ocr_engine="pymupdf-native",
            processing_time_ms=5,
            layout={"pages": 1, "tables_detected": 0, "page_confidences": [], "word_confidences": None},
        ),
    )


def _register_and_login(client: TestClient, email: str, role: str = "doctor") -> dict:
    """Register a new user and return auth headers."""
    client.post(
        "/api/v1/auth/register",
        json={"name": f"User {email}", "email": email, "password": "pass1234", "role": role},
    )
    resp = client.post(
        "/api/v1/auth/login",
        data={"username": email, "password": "pass1234"},
    )
    assert resp.status_code == 200
    return {"Authorization": f"Bearer {resp.json()['access_token']}"}


def _create_patient(client: TestClient, headers: dict, mrn: str) -> dict:
    resp = client.post(
        "/api/v1/patients",
        json={"mrn": mrn, "first_name": "Jane", "last_name": "Doe"},
        headers=headers,
    )
    assert resp.status_code == 201
    return resp.json()


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------


def test_patient_upload_associates_document_with_patient(
    client: TestClient,
    auth_headers: dict,
    tmp_path: Path,
    monkeypatch,
) -> None:
    """Uploading with a valid patient_id stores the patient association."""
    _mock_ocr(monkeypatch, tmp_path)

    patient = _create_patient(client, auth_headers, "UP-ASSOC-001")
    patient_id = patient["patient_id"]

    response = client.post(
        "/api/v1/documents",
        headers=auth_headers,
        files={"file": ("report.pdf", b"pdf-data", "application/pdf")},
        data={"patient_id": patient_id},
    )
    assert response.status_code == 201
    doc = response.json()
    assert doc["patient_id"] == patient_id, "Document must be associated with the patient"

    listing = client.get(
        "/api/v1/documents",
        headers=auth_headers,
        params={"patient_id": patient_id},
    )
    assert listing.status_code == 200
    items = listing.json()["items"]
    assert any(d["document_id"] == doc["document_id"] for d in items), (
        "Document should appear in the patient-scoped listing"
    )


def test_unauthorized_patient_upload_rejected(
    client: TestClient,
    tmp_path: Path,
    monkeypatch,
) -> None:
    """An unrelated user must not upload a document to another user's patient."""
    _mock_ocr(monkeypatch, tmp_path)

    owner_headers = _register_and_login(client, "owner@pattest.org")
    intruder_headers = _register_and_login(client, "intruder@pattest.org")

    patient = _create_patient(client, owner_headers, "UP-AUTH-002")
    patient_id = patient["patient_id"]

    response = client.post(
        "/api/v1/documents",
        headers=intruder_headers,
        files={"file": ("evil.pdf", b"pdf-data", "application/pdf")},
        data={"patient_id": patient_id},
    )
    assert response.status_code == 403, (
        "Backend must reject uploads to patients the authenticated user cannot access"
    )


def test_general_upload_without_patient_id_still_works(
    client: TestClient,
    auth_headers: dict,
    tmp_path: Path,
    monkeypatch,
) -> None:
    """General upload (no patient_id) must still succeed."""
    _mock_ocr(monkeypatch, tmp_path)

    response = client.post(
        "/api/v1/documents",
        headers=auth_headers,
        files={"file": ("general.pdf", b"pdf-data", "application/pdf")},
    )
    assert response.status_code == 201
    doc = response.json()
    assert doc["patient_id"] is None, "General upload must not associate with any patient"


def test_patient_upload_not_visible_under_other_patient(
    client: TestClient,
    auth_headers: dict,
    tmp_path: Path,
    monkeypatch,
) -> None:
    """Document uploaded for Patient A must not appear in Patient B's listing."""
    _mock_ocr(monkeypatch, tmp_path)

    patient_a = _create_patient(client, auth_headers, "UP-ISO-A")
    patient_b = _create_patient(client, auth_headers, "UP-ISO-B")

    upload = client.post(
        "/api/v1/documents",
        headers=auth_headers,
        files={"file": ("for_a.pdf", b"pdf-data", "application/pdf")},
        data={"patient_id": patient_a["patient_id"]},
    )
    assert upload.status_code == 201

    listing_b = client.get(
        "/api/v1/documents",
        headers=auth_headers,
        params={"patient_id": patient_b["patient_id"]},
    )
    assert listing_b.status_code == 200
    doc_ids = [d["document_id"] for d in listing_b.json()["items"]]
    assert upload.json()["document_id"] not in doc_ids, (
        "Patient A's document must not appear in Patient B's listing"
    )
