from uuid import uuid4
from unittest.mock import patch
import pytest
from fastapi.testclient import TestClient

from app.models.document import Document


def test_list_documents_ownership_isolation(
    client: TestClient,
    auth_headers: dict[str, str],
) -> None:
    """Test that users can only view their own uploaded documents."""
    # Register second user
    reg_res = client.post(
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

    # Upload document as User 1
    with patch("app.services.document_service.extract_text") as mock_extract:
        mock_extract.return_value = (
            "User 1 document content",
            {"pages": 1, "tables_detected": 0, "page_confidences": [], "word_confidences": []},
        )
        upload1 = client.post(
            "/api/v1/documents",
            headers=auth_headers,
            files={"file": ("doc_user1.pdf", b"pdf content", "application/pdf")},
        )
        assert upload1.status_code == 201
        doc1_id = upload1.json()["document_id"]

        # Upload document as User 2
        upload2 = client.post(
            "/api/v1/documents",
            headers=user2_headers,
            files={"file": ("doc_user2.pdf", b"pdf content", "application/pdf")},
        )
        assert upload2.status_code == 201
        doc2_id = upload2.json()["document_id"]

    # User 1 lists documents
    res1 = client.get("/api/v1/documents", headers=auth_headers)
    assert res1.status_code == 200
    data1 = res1.json()
    ids1 = [doc["document_id"] for doc in data1["items"]]
    assert doc1_id in ids1
    assert doc2_id not in ids1

    # User 2 lists documents
    res2 = client.get("/api/v1/documents", headers=user2_headers)
    assert res2.status_code == 200
    data2 = res2.json()
    ids2 = [doc["document_id"] for doc in data2["items"]]
    assert doc2_id in ids2
    assert doc1_id not in ids2


def test_list_documents_pagination_and_search(
    client: TestClient,
    auth_headers: dict[str, str],
) -> None:
    """Test pagination metadata and filename search filtering."""
    with patch("app.services.document_service.extract_text") as mock_extract:
        mock_extract.return_value = ("Sample text", {"pages": 1})
        client.post(
            "/api/v1/documents",
            headers=auth_headers,
            files={"file": ("blood_test_results.pdf", b"content", "application/pdf")},
        )
        client.post(
            "/api/v1/documents",
            headers=auth_headers,
            files={"file": ("discharge_summary.pdf", b"content", "application/pdf")},
        )

    # Search by filename
    search_res = client.get("/api/v1/documents?search=blood", headers=auth_headers)
    assert search_res.status_code == 200
    search_data = search_res.json()
    assert search_data["total"] == 1
    assert "blood_test_results.pdf" in search_data["items"][0]["file_name"]

    # Pagination page_size=1
    page1_res = client.get("/api/v1/documents?page=1&page_size=1", headers=auth_headers)
    assert page1_res.status_code == 200
    page1_data = page1_res.json()
    assert len(page1_data["items"]) == 1
    assert page1_data["page"] == 1
    assert page1_data["page_size"] == 1
    assert page1_data["total_pages"] >= 2


def test_delete_document_success_and_cleanup(
    client: TestClient,
    auth_headers: dict[str, str],
) -> None:
    """Test deleting owned document removes record and safely handles file cleanup."""
    with patch("app.services.document_service.extract_text") as mock_extract:
        mock_extract.return_value = ("Sample text", {"pages": 1})
        upload = client.post(
            "/api/v1/documents",
            headers=auth_headers,
            files={"file": ("to_delete.pdf", b"content", "application/pdf")},
        )
        doc_id = upload.json()["document_id"]

    # Verify exists
    get_res = client.get(f"/api/v1/documents/{doc_id}", headers=auth_headers)
    assert get_res.status_code == 200

    # Delete
    del_res = client.delete(f"/api/v1/documents/{doc_id}", headers=auth_headers)
    assert del_res.status_code == 204

    # Verify no longer exists
    get_res2 = client.get(f"/api/v1/documents/{doc_id}", headers=auth_headers)
    assert get_res2.status_code == 404


def test_delete_document_unauthorized(
    client: TestClient,
    auth_headers: dict[str, str],
) -> None:
    """Test that a user cannot delete another user's document."""
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

    # User 1 uploads doc
    with patch("app.services.document_service.extract_text") as mock_extract:
        mock_extract.return_value = ("Sample text", {"pages": 1})
        upload = client.post(
            "/api/v1/documents",
            headers=auth_headers,
            files={"file": ("user1_protected.pdf", b"content", "application/pdf")},
        )
        doc_id = upload.json()["document_id"]

    # User 2 attempts to delete User 1's document
    del_res = client.delete(f"/api/v1/documents/{doc_id}", headers=user2_headers)
    assert del_res.status_code == 403
    assert del_res.json()["detail"] == "Access denied"

    # User 2 attempts to view detail of User 1's document
    detail_res = client.get(f"/api/v1/documents/{doc_id}", headers=user2_headers)
    assert detail_res.status_code == 403
