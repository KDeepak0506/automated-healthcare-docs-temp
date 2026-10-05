"""
Tests for patient-document identity validation (Part 1).
Verifies:
1. Correct patient's document -> upload succeeds.
2. Clearly different patient's document -> upload rejected with PATIENT_IDENTITY_MISMATCH.
3. Document without identifiable patient information -> follows existing intended behavior.
4. Rejected document is not associated with the wrong patient.
5. No orphan/partially-created document remains after rejection.
6. Doctor/clinician names in non-patient context do not cause false rejections.
7. Existing nurse/admin/doctor/records-staff permissions continue to work.
"""
from pathlib import Path
from uuid import uuid4

import pytest
from fastapi.testclient import TestClient
from app.services import document_service
from app.services.ocr_service import OCRResult


def _mock_ocr_with_text(monkeypatch, tmp_path: Path, raw_text: str) -> None:
    """Mock OCR text returned during document extraction."""
    monkeypatch.setattr(document_service, "UPLOAD_DIR", tmp_path)
    monkeypatch.setattr(
        document_service,
        "extract_text",
        lambda file_path, file_type: OCRResult(
            raw_text=raw_text,
            page_count=1,
            confidence=0.99,
            ocr_engine="pymupdf-native",
            processing_time_ms=5,
            layout={"pages": 1, "tables_detected": 0, "page_confidences": [], "word_confidences": None},
        ),
    )


def _register_and_login(client: TestClient, email: str, role: str = "doctor") -> dict:
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


def _create_patient(
    client: TestClient,
    headers: dict,
    first_name: str,
    last_name: str,
    mrn: str,
) -> dict:
    resp = client.post(
        "/api/v1/patients",
        json={"mrn": mrn, "first_name": first_name, "last_name": last_name},
        headers=headers,
    )
    assert resp.status_code == 201
    return resp.json()


def test_correct_patient_document_upload_succeeds(
    client: TestClient,
    auth_headers: dict,
    tmp_path: Path,
    monkeypatch,
) -> None:
    """Requirement 1: Uploading a document that matches the target patient succeeds."""
    content = (
        "Springfield Hospital\n"
        "Patient Name: David Lee\n"
        "MRN: MRN-DL-001\n"
        "Procedure: CT Head Without Contrast\n"
        "Findings: Normal intracranial appearance."
    )
    _mock_ocr_with_text(monkeypatch, tmp_path, content)

    patient = _create_patient(client, auth_headers, "David", "Lee", "MRN-DL-001")
    patient_id = patient["patient_id"]

    response = client.post(
        "/api/v1/documents",
        headers=auth_headers,
        files={"file": ("David_Lee_CT.pdf", b"dummy-pdf-data", "application/pdf")},
        data={"patient_id": patient_id},
    )

    assert response.status_code == 201
    doc = response.json()
    assert doc["patient_id"] == patient_id
    assert doc["file_name"] == "David_Lee_CT.pdf"


def test_different_patient_name_rejected_with_structured_error(
    client: TestClient,
    auth_headers: dict,
    tmp_path: Path,
    monkeypatch,
) -> None:
    """Requirement 2: Uploading a document belonging to a different patient is rejected."""
    content = (
        "General Medical Center\n"
        "Patient Name: John Smith\n"
        "DOB: 1975-04-12\n"
        "Procedure: Chest X-Ray\n"
        "Findings: Clear lung fields."
    )
    _mock_ocr_with_text(monkeypatch, tmp_path, content)

    # Current workspace is David Lee
    patient = _create_patient(client, auth_headers, "David", "Lee", "MRN-DL-002")
    patient_id = patient["patient_id"]

    response = client.post(
        "/api/v1/documents",
        headers=auth_headers,
        files={"file": ("John_Smith.pdf", b"dummy-pdf-data", "application/pdf")},
        data={"patient_id": patient_id},
    )

    assert response.status_code == 400
    err_body = response.json()
    assert err_body["code"] == "PATIENT_IDENTITY_MISMATCH"
    assert "John Smith" in err_body["message"]
    assert "David Lee" in err_body["message"]


def test_different_patient_mrn_rejected(
    client: TestClient,
    auth_headers: dict,
    tmp_path: Path,
    monkeypatch,
) -> None:
    """Requirement 2b: MRN mismatch between document and workspace patient is rejected."""
    content = (
        "Radiology Report\n"
        "MRN: MRN-OTHER-999\n"
        "Procedure: MRI Brain"
    )
    _mock_ocr_with_text(monkeypatch, tmp_path, content)

    patient = _create_patient(client, auth_headers, "Alice", "Brown", "MRN-AB-003")
    patient_id = patient["patient_id"]

    response = client.post(
        "/api/v1/documents",
        headers=auth_headers,
        files={"file": ("report.pdf", b"dummy-pdf-data", "application/pdf")},
        data={"patient_id": patient_id},
    )

    assert response.status_code == 400
    err_body = response.json()
    assert err_body["code"] == "PATIENT_IDENTITY_MISMATCH"


def test_document_without_identifiable_patient_information_succeeds(
    client: TestClient,
    auth_headers: dict,
    tmp_path: Path,
    monkeypatch,
) -> None:
    """Requirement 3: Documents without patient identity info follow existing behavior and succeed."""
    content = (
        "Clinical Laboratory Reference Manual\n"
        "Standard reference intervals for adult complete blood count.\n"
        "Hemoglobin: 13.8 - 17.2 g/dL."
    )
    _mock_ocr_with_text(monkeypatch, tmp_path, content)

    patient = _create_patient(client, auth_headers, "David", "Lee", "MRN-DL-004")
    patient_id = patient["patient_id"]

    response = client.post(
        "/api/v1/documents",
        headers=auth_headers,
        files={"file": ("lab_reference.pdf", b"dummy-pdf-data", "application/pdf")},
        data={"patient_id": patient_id},
    )

    assert response.status_code == 201
    doc = response.json()
    assert doc["patient_id"] == patient_id


def test_rejected_document_not_associated_and_no_orphan_remains(
    client: TestClient,
    auth_headers: dict,
    tmp_path: Path,
    monkeypatch,
) -> None:
    """Requirements 4 & 5: Rejected document leaves no DB record and no orphan file."""
    content = "Patient Name: Robert Jones\nDiagnosis: Hypertension"
    _mock_ocr_with_text(monkeypatch, tmp_path, content)

    patient = _create_patient(client, auth_headers, "David", "Lee", "MRN-DL-005")
    patient_id = patient["patient_id"]

    # Verify no documents initially
    list_before = client.get("/api/v1/documents", headers=auth_headers, params={"patient_id": patient_id})
    assert len(list_before.json()["items"]) == 0

    response = client.post(
        "/api/v1/documents",
        headers=auth_headers,
        files={"file": ("Robert_Jones.pdf", b"dummy-pdf-data", "application/pdf")},
        data={"patient_id": patient_id},
    )
    assert response.status_code == 400

    # 1. No document associated with patient
    list_after = client.get("/api/v1/documents", headers=auth_headers, params={"patient_id": patient_id})
    assert len(list_after.json()["items"]) == 0

    # 2. No orphan record in database (check global accessible listing)
    all_docs = client.get("/api/v1/documents", headers=auth_headers).json()["items"]
    assert not any(d["patient_id"] == patient_id for d in all_docs)
    assert not any("Robert_Jones" in d["file_name"] for d in all_docs)

    # 3. No temp files left in upload directory
    temp_files = list(tmp_path.glob("temp_*"))
    assert len(temp_files) == 0


def test_doctor_in_non_patient_context_does_not_falsely_reject(
    client: TestClient,
    auth_headers: dict,
    tmp_path: Path,
    monkeypatch,
) -> None:
    """Non-patient names like Referring Doctor or Hospital do not cause false rejections."""
    content = (
        "Memorial General Hospital\n"
        "Referring Physician: Dr. Marcus Vance, MD\n"
        "Attending Doctor: Dr. Alice Walker\n"
        "Patient Name: David Lee\n"
        "MRN: MRN-DL-006\n"
        "Clinical Impression: Normal study."
    )
    _mock_ocr_with_text(monkeypatch, tmp_path, content)

    patient = _create_patient(client, auth_headers, "David", "Lee", "MRN-DL-006")
    patient_id = patient["patient_id"]

    response = client.post(
        "/api/v1/documents",
        headers=auth_headers,
        files={"file": ("clinical_report.pdf", b"dummy-pdf-data", "application/pdf")},
        data={"patient_id": patient_id},
    )

    assert response.status_code == 201
    assert response.json()["patient_id"] == patient_id


def test_permissions_with_identity_validation(
    client: TestClient,
    tmp_path: Path,
    monkeypatch,
) -> None:
    """Requirement 6: Role permissions continue to work alongside identity validation."""
    doctor_headers = _register_and_login(client, "doctor_perm@hospital.org", role="doctor")
    nurse_headers = _register_and_login(client, "nurse_perm@hospital.org", role="nurse")
    staff_headers = _register_and_login(client, "staff_perm@hospital.org", role="records_staff")

    patient = _create_patient(client, doctor_headers, "Jane", "Doe", "MRN-JD-007")
    patient_id = patient["patient_id"]

    # 1. Doctor (assigned creator) uploading correct document -> 201
    _mock_ocr_with_text(monkeypatch, tmp_path, "Patient Name: Jane Doe\nNotes: Follow-up")
    resp_doc = client.post(
        "/api/v1/documents",
        headers=doctor_headers,
        files={"file": ("Jane_Doe_notes.pdf", b"pdf-data", "application/pdf")},
        data={"patient_id": patient_id},
    )
    assert resp_doc.status_code == 201

    # 2. Records staff (system-wide clinical access) uploading mismatch document -> 400 PATIENT_IDENTITY_MISMATCH
    _mock_ocr_with_text(monkeypatch, tmp_path, "Patient Name: John Smith\nNotes: Other patient")
    resp_staff = client.post(
        "/api/v1/documents",
        headers=staff_headers,
        files={"file": ("John_Smith.pdf", b"pdf-data", "application/pdf")},
        data={"patient_id": patient_id},
    )
    assert resp_staff.status_code == 400
    assert resp_staff.json()["code"] == "PATIENT_IDENTITY_MISMATCH"

    # 3. Unassigned nurse uploading to patient -> 403 Forbidden
    resp_nurse = client.post(
        "/api/v1/documents",
        headers=nurse_headers,
        files={"file": ("doc.pdf", b"pdf-data", "application/pdf")},
        data={"patient_id": patient_id},
    )
    assert resp_nurse.status_code == 403
