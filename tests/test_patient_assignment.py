"""Tests for patient care team assignment, access control, and nurse permissions."""
from pathlib import Path
from uuid import uuid4

import pytest
from fastapi.testclient import TestClient

from app.services import document_service
from app.services.ocr_service import OCRResult


def _mock_ocr(monkeypatch, tmp_path: Path) -> None:
    """Mock OCR text extraction and upload dir to prevent disk/tesseract dependency."""
    monkeypatch.setattr(document_service, "UPLOAD_DIR", tmp_path)
    monkeypatch.setattr(
        document_service,
        "extract_text",
        lambda file_path, file_type: OCRResult(
            raw_text="Clinical nurse notes",
            page_count=1,
            confidence=0.98,
            ocr_engine="test-mock",
            processing_time_ms=5,
            layout={"pages": 1, "tables_detected": 0, "page_confidences": [], "word_confidences": None},
        ),
    )


def _register_and_login(client: TestClient, name: str, email: str, role: str) -> tuple[dict, dict[str, str]]:
    """Register a new user and login, returning user dict and Authorization headers."""
    reg = client.post(
        "/api/v1/auth/register",
        json={"name": name, "email": email, "password": "password123", "role": role},
    )
    assert reg.status_code == 201, reg.text
    user_data = reg.json()

    login = client.post(
        "/api/v1/auth/login",
        data={"username": email, "password": "password123"},
    )
    assert login.status_code == 200, login.text
    token = login.json()["access_token"]
    return user_data, {"Authorization": f"Bearer {token}"}


def _create_patient(client: TestClient, headers: dict[str, str], mrn: str) -> dict:
    """Create a patient as staff/doctor."""
    res = client.post(
        "/api/v1/patients",
        headers=headers,
        json={
            "mrn": mrn,
            "first_name": "Test",
            "last_name": "Patient",
            "date_of_birth": "1990-01-01",
            "gender": "Other",
        },
    )
    assert res.status_code == 201, res.text
    return res.json()


def test_unassigned_nurse_access(client: TestClient) -> None:
    """Unassigned nurse: GET /patients returns total 0, GET /patients/{id} is 403."""
    _, staff_headers = _register_and_login(client, "Staff Member", "staff1@hospital.org", "records_staff")
    patient = _create_patient(client, staff_headers, "MRN-UNASSIGNED-1")

    _, nurse_headers = _register_and_login(client, "Nurse One", "nurse1@hospital.org", "nurse")

    # List accessible patients for unassigned nurse
    list_res = client.get("/api/v1/patients", headers=nurse_headers)
    assert list_res.status_code == 200
    data = list_res.json()
    assert data["total"] == 0
    assert len(data["items"]) == 0

    # Direct detail access for unassigned nurse
    detail_res = client.get(f"/api/v1/patients/{patient['patient_id']}", headers=nurse_headers)
    assert detail_res.status_code == 403


def test_assigned_nurse_access(client: TestClient) -> None:
    """Assigned nurse: list shows the patient, GET /patients/{id} is 200."""
    _, staff_headers = _register_and_login(client, "Staff Member", "staff2@hospital.org", "records_staff")
    patient = _create_patient(client, staff_headers, "MRN-ASSIGNED-1")

    nurse, nurse_headers = _register_and_login(client, "Nurse Two", "nurse2@hospital.org", "nurse")

    # Assign nurse to patient
    assign_res = client.post(
        f"/api/v1/patients/{patient['patient_id']}/assign",
        headers=staff_headers,
        json={"user_id": nurse["user_id"]},
    )
    assert assign_res.status_code == 200

    # Assigned nurse can see patient in list
    list_res = client.get("/api/v1/patients", headers=nurse_headers)
    assert list_res.status_code == 200
    data = list_res.json()
    assert data["total"] == 1
    assert data["items"][0]["patient_id"] == patient["patient_id"]

    # Assigned nurse can get patient details
    detail_res = client.get(f"/api/v1/patients/{patient['patient_id']}", headers=nurse_headers)
    assert detail_res.status_code == 200
    assert detail_res.json()["patient_id"] == patient["patient_id"]


def test_unassign_revokes_access(client: TestClient) -> None:
    """Unassign revokes access (403 afterwards)."""
    _, staff_headers = _register_and_login(client, "Staff Member", "staff3@hospital.org", "records_staff")
    patient = _create_patient(client, staff_headers, "MRN-REVOKE-1")

    nurse, nurse_headers = _register_and_login(client, "Nurse Three", "nurse3@hospital.org", "nurse")

    # Assign
    client.post(
        f"/api/v1/patients/{patient['patient_id']}/assign",
        headers=staff_headers,
        json={"user_id": nurse["user_id"]},
    )
    assert client.get(f"/api/v1/patients/{patient['patient_id']}", headers=nurse_headers).status_code == 200

    # Unassign
    unassign_res = client.delete(
        f"/api/v1/patients/{patient['patient_id']}/assign/{nurse['user_id']}",
        headers=staff_headers,
    )
    assert unassign_res.status_code == 204

    # Access revoked: detail gives 403, list gives total 0
    detail_res = client.get(f"/api/v1/patients/{patient['patient_id']}", headers=nurse_headers)
    assert detail_res.status_code == 403

    list_res = client.get("/api/v1/patients", headers=nurse_headers)
    assert list_res.status_code == 200
    assert list_res.json()["total"] == 0


def test_unassign_revokes_even_after_upload(client: TestClient, monkeypatch, tmp_path: Path) -> None:
    """Nurse that uploaded a document, then is unassigned, loses access (covers step 5)."""
    _mock_ocr(monkeypatch, tmp_path)

    _, staff_headers = _register_and_login(client, "Staff Member", "staff4@hospital.org", "records_staff")
    patient = _create_patient(client, staff_headers, "MRN-DOC-REVOKE-1")

    nurse, nurse_headers = _register_and_login(client, "Nurse Four", "nurse4@hospital.org", "nurse")

    # Assign nurse
    client.post(
        f"/api/v1/patients/{patient['patient_id']}/assign",
        headers=staff_headers,
        json={"user_id": nurse["user_id"]},
    )

    # Nurse uploads a document associated with the patient
    upload_res = client.post(
        "/api/v1/documents",
        headers=nurse_headers,
        files={"file": ("nurse_note.pdf", b"%PDF-1.4 test document content", "application/pdf")},
        data={"patient_id": patient["patient_id"]},
    )
    assert upload_res.status_code == 201

    # Before unassign: nurse has access
    assert client.get(f"/api/v1/patients/{patient['patient_id']}", headers=nurse_headers).status_code == 200

    # Unassign the nurse
    unassign_res = client.delete(
        f"/api/v1/patients/{patient['patient_id']}/assign/{nurse['user_id']}",
        headers=staff_headers,
    )
    assert unassign_res.status_code == 204

    # True revoke check: nurse MUST lose access even though they uploaded a document for that patient
    detail_res = client.get(f"/api/v1/patients/{patient['patient_id']}", headers=nurse_headers)
    assert detail_res.status_code == 403

    list_res = client.get("/api/v1/patients", headers=nurse_headers)
    assert list_res.status_code == 200
    assert list_res.json()["total"] == 0


def test_nurse_forbidden_operations(client: TestClient) -> None:
    """Nurse gets 403 on POST /patients, POST /assign, PUT /patients/{id}, GET /users, etc."""
    _, staff_headers = _register_and_login(client, "Staff Member", "staff5@hospital.org", "records_staff")
    patient = _create_patient(client, staff_headers, "MRN-FORBIDDEN-1")

    nurse, nurse_headers = _register_and_login(client, "Nurse Five", "nurse5@hospital.org", "nurse")

    # 1. Nurse cannot register patient: POST /patients -> 403
    create_res = client.post(
        "/api/v1/patients",
        headers=nurse_headers,
        json={"mrn": "MRN-NURSE-BLOCKED", "first_name": "Bad", "last_name": "Attempt"},
    )
    assert create_res.status_code == 403

    # 2. Nurse cannot assign patients: POST /patients/{id}/assign -> 403
    assign_res = client.post(
        f"/api/v1/patients/{patient['patient_id']}/assign",
        headers=nurse_headers,
        json={"user_id": nurse["user_id"]},
    )
    assert assign_res.status_code == 403

    # Assign nurse so patient is accessible for next test
    client.post(
        f"/api/v1/patients/{patient['patient_id']}/assign",
        headers=staff_headers,
        json={"user_id": nurse["user_id"]},
    )

    # 3. Nurse cannot update patient demographics: PUT /patients/{id} -> 403
    update_res = client.put(
        f"/api/v1/patients/{patient['patient_id']}",
        headers=nurse_headers,
        json={"first_name": "UpdatedName"},
    )
    assert update_res.status_code == 403

    # 4. Nurse cannot list users: GET /users -> 403
    users_res = client.get("/api/v1/users", headers=nurse_headers)
    assert users_res.status_code == 403

    # 5. Nurse cannot unassign: DELETE /patients/{id}/assign/{user_id} -> 403
    del_res = client.delete(
        f"/api/v1/patients/{patient['patient_id']}/assign/{nurse['user_id']}",
        headers=nurse_headers,
    )
    assert del_res.status_code == 403

    # 6. Nurse cannot view assignments: GET /patients/{id}/assignments -> 403
    assignments_res = client.get(
        f"/api/v1/patients/{patient['patient_id']}/assignments",
        headers=nurse_headers,
    )
    assert assignments_res.status_code == 403


def test_nurse_cannot_patch_document_status(client: TestClient, monkeypatch, tmp_path: Path) -> None:
    """PATCH /documents/{id}/status is restricted to admin, records_staff, doctor (nurse gets 403)."""
    _mock_ocr(monkeypatch, tmp_path)

    _, staff_headers = _register_and_login(client, "Staff Member", "staff_doc@hospital.org", "records_staff")
    patient = _create_patient(client, staff_headers, "MRN-STATUS-1")

    nurse, nurse_headers = _register_and_login(client, "Nurse Six", "nurse6@hospital.org", "nurse")

    # Assign nurse
    client.post(
        f"/api/v1/patients/{patient['patient_id']}/assign",
        headers=staff_headers,
        json={"user_id": nurse["user_id"]},
    )

    # Nurse uploads document
    upload_res = client.post(
        "/api/v1/documents",
        headers=nurse_headers,
        files={"file": ("doc_status.pdf", b"%PDF-1.4 sample", "application/pdf")},
        data={"patient_id": patient["patient_id"]},
    )
    assert upload_res.status_code == 201
    doc_id = upload_res.json()["document_id"]

    # Nurse attempts PATCH /documents/{id}/status -> 403
    patch_res = client.patch(
        f"/api/v1/documents/{doc_id}/status?status=Completed",
        headers=nurse_headers,
    )
    assert patch_res.status_code == 403

    # Doctor/Staff can PATCH /documents/{id}/status -> 200
    staff_patch = client.patch(
        f"/api/v1/documents/{doc_id}/status?status=Completed",
        headers=staff_headers,
    )
    assert staff_patch.status_code == 200
    assert staff_patch.json()["processing_status"] == "Completed"


def test_list_users_role_filter_and_no_password_hash(client: TestClient) -> None:
    """GET /users?role=nurse returns only nurses and never includes password_hash."""
    _, staff_headers = _register_and_login(client, "Staff Member", "staff_u@hospital.org", "records_staff")
    nurse1, _ = _register_and_login(client, "Nurse Alpha", "nurse_alpha@hospital.org", "nurse")
    nurse2, _ = _register_and_login(client, "Nurse Beta", "nurse_beta@hospital.org", "nurse")
    doctor1, _ = _register_and_login(client, "Dr Gamma", "doc_gamma@hospital.org", "doctor")

    # Staff queries users with role=nurse
    res = client.get("/api/v1/users?role=nurse", headers=staff_headers)
    assert res.status_code == 200
    users = res.json()
    assert len(users) >= 2

    # Verify every returned user has role nurse and never includes password_hash
    user_ids = [u["user_id"] for u in users]
    assert nurse1["user_id"] in user_ids
    assert nurse2["user_id"] in user_ids
    assert doctor1["user_id"] not in user_ids

    for u in users:
        assert u["role"] == "nurse"
        assert "password_hash" not in u
        assert "password" not in u


def test_get_patient_assignments_returns_assigned_nurse(client: TestClient) -> None:
    """GET /patients/{id}/assignments returns the assigned nurse for staff."""
    _, staff_headers = _register_and_login(client, "Staff Member", "staff_assign@hospital.org", "records_staff")
    patient = _create_patient(client, staff_headers, "MRN-ASSIGN-LIST-1")

    nurse, _ = _register_and_login(client, "Nurse Assigned", "nurse_assigned@hospital.org", "nurse")

    # Initially empty or creator assigned
    init_res = client.get(f"/api/v1/patients/{patient['patient_id']}/assignments", headers=staff_headers)
    assert init_res.status_code == 200

    # Assign nurse
    assign_res = client.post(
        f"/api/v1/patients/{patient['patient_id']}/assign",
        headers=staff_headers,
        json={"user_id": nurse["user_id"]},
    )
    assert assign_res.status_code == 200

    # Fetch assignments list
    get_res = client.get(f"/api/v1/patients/{patient['patient_id']}/assignments", headers=staff_headers)
    assert get_res.status_code == 200
    assignments = get_res.json()

    assigned_user_ids = [a["user_id"] for a in assignments]
    assert nurse["user_id"] in assigned_user_ids

    # Find the nurse assignment
    nurse_assignment = next(a for a in assignments if a["user_id"] == nurse["user_id"])
    assert nurse_assignment["role"] == "nurse" or nurse_assignment.get("user", {}).get("role") == "nurse"


def test_unassign_nonexistent_returns_404(client: TestClient) -> None:
    """Deleting a non-existent assignment returns 404."""
    _, staff_headers = _register_and_login(client, "Staff Member", "staff_404@hospital.org", "records_staff")
    patient = _create_patient(client, staff_headers, "MRN-404-1")
    random_user_id = str(uuid4())

    res = client.delete(
        f"/api/v1/patients/{patient['patient_id']}/assign/{random_user_id}",
        headers=staff_headers,
    )
    assert res.status_code == 404


def test_unassigned_nurse_document_access_revoked(client: TestClient, monkeypatch, tmp_path: Path) -> None:
    """Nurse uploads for an assigned patient, is unassigned, then gets 403 on GET /documents/{id}, /text, /summarize, and document is absent from GET /documents."""
    _mock_ocr(monkeypatch, tmp_path)

    _, staff_headers = _register_and_login(client, "Staff Member", "staff_doc_revoke@hospital.org", "records_staff")
    patient = _create_patient(client, staff_headers, "MRN-DOC-REVOKE-2")

    nurse, nurse_headers = _register_and_login(client, "Nurse Seven", "nurse7@hospital.org", "nurse")

    # Assign nurse
    client.post(
        f"/api/v1/patients/{patient['patient_id']}/assign",
        headers=staff_headers,
        json={"user_id": nurse["user_id"]},
    )

    # Nurse uploads document for assigned patient
    upload_res = client.post(
        "/api/v1/documents",
        headers=nurse_headers,
        files={"file": ("vital_signs.pdf", b"%PDF-1.4 vital signs content", "application/pdf")},
        data={"patient_id": patient["patient_id"]},
    )
    assert upload_res.status_code == 201
    doc_id = upload_res.json()["document_id"]

    # While assigned, nurse can access document details, text, and list
    assert client.get(f"/api/v1/documents/{doc_id}", headers=nurse_headers).status_code == 200
    assert client.get(f"/api/v1/documents/{doc_id}/text", headers=nurse_headers).status_code == 200
    list_before = client.get("/api/v1/documents", headers=nurse_headers)
    assert list_before.status_code == 200
    assert any(d["document_id"] == doc_id for d in list_before.json()["items"])

    # Unassign nurse
    unassign_res = client.delete(
        f"/api/v1/patients/{patient['patient_id']}/assign/{nurse['user_id']}",
        headers=staff_headers,
    )
    assert unassign_res.status_code == 204

    # Now unassigned: nurse gets 403 on GET /documents/{id}
    assert client.get(f"/api/v1/documents/{doc_id}", headers=nurse_headers).status_code == 403

    # Nurse gets 403 on GET /documents/{id}/text
    assert client.get(f"/api/v1/documents/{doc_id}/text", headers=nurse_headers).status_code == 403

    # Nurse gets 403 on POST /documents/{id}/summarize
    assert client.post(f"/api/v1/documents/{doc_id}/summarize", headers=nurse_headers).status_code == 403

    # Document is absent from GET /documents listing
    list_after = client.get("/api/v1/documents", headers=nurse_headers)
    assert list_after.status_code == 200
    assert not any(d["document_id"] == doc_id for d in list_after.json()["items"])


def test_nurse_standalone_document_access_preserved(client: TestClient, monkeypatch, tmp_path: Path) -> None:
    """Nurse retains access to documents uploaded with no patient_id."""
    _mock_ocr(monkeypatch, tmp_path)

    _, nurse_headers = _register_and_login(client, "Nurse Standalone", "nurse_standalone@hospital.org", "nurse")

    # Upload document with no patient_id
    upload_res = client.post(
        "/api/v1/documents",
        headers=nurse_headers,
        files={"file": ("standalone.pdf", b"%PDF-1.4 standalone", "application/pdf")},
    )
    assert upload_res.status_code == 201
    doc_id = upload_res.json()["document_id"]

    # Nurse can access standalone document
    assert client.get(f"/api/v1/documents/{doc_id}", headers=nurse_headers).status_code == 200
    assert client.get(f"/api/v1/documents/{doc_id}/text", headers=nurse_headers).status_code == 200

    # Standalone document is present in listing
    list_res = client.get("/api/v1/documents", headers=nurse_headers)
    assert list_res.status_code == 200
    assert any(d["document_id"] == doc_id for d in list_res.json()["items"])
