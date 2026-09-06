from unittest.mock import MagicMock
from uuid import UUID, uuid4
import pytest
from fastapi.testclient import TestClient
from sqlalchemy.orm import Session, sessionmaker

from app.models.document import Document
from app.models.document_chunk import DocumentChunk
from app.models.document_text import DocumentText
from app.services.chunking_service import ChunkingService, chunking_service
from app.services.embedding_service import EmbeddingService
from app.services.llm_service import (
    LLMConfigurationError,
    LLMResponseParsingError,
    LLMService,
    LLMServiceError,
    LLMTimeoutError,
)
from app.services import rag_service as rag_service_module


def _create_test_document(
    db: Session,
    user_id: UUID,
    privacy_status: str = "completed",
    raw_text: str = "Patient Name: Rahul Kumar\nMRN: MRN12345\nHbA1c: 8.2%\nCreatinine: 1.41 mg/dL\nPrescription: Metformin 500mg twice daily.",
    sanitized_text: str | None = "Patient Name: [PATIENT]\nMRN: [MRN]\nHbA1c: 8.2%\nCreatinine: 1.41 mg/dL\nPrescription: Metformin 500mg twice daily.",
) -> Document:
    doc = Document(
        uploaded_by=user_id,
        file_name="discharge_summary.pdf",
        file_type="application/pdf",
        file_url="/tmp/discharge_summary.pdf",
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


# ==========================================
# 1. ChunkingService Unit Tests
# ==========================================

def test_chunking_empty_and_whitespace() -> None:
    chunker = ChunkingService()
    assert chunker.chunk_text("") == []
    assert chunker.chunk_text("   \n\t  ") == []


def test_chunking_short_text() -> None:
    chunker = ChunkingService(target_words=100, overlap_words=20)
    text = "Patient presents with persistent cough and mild fever. Prescribed amoxicillin 500mg."
    chunks = chunker.chunk_text(text)
    assert len(chunks) == 1
    assert chunks[0].chunk_index == 0
    assert chunks[0].chunk_text == text
    assert chunks[0].start_offset == 0
    assert chunks[0].end_offset == len(text)


def test_chunking_long_text_and_overlap() -> None:
    chunker = ChunkingService(target_words=10, overlap_words=3)
    words = [f"word{i}" for i in range(25)]
    text = " ".join(words)
    chunks = chunker.chunk_text(text)

    assert len(chunks) > 1
    # Check indices are sequential
    for i, ch in enumerate(chunks):
        assert ch.chunk_index == i
        assert ch.start_offset >= 0
        assert ch.end_offset <= len(text)
        assert ch.start_offset < ch.end_offset
        # Check text slice matches offsets
        assert text[ch.start_offset:ch.end_offset].strip() == ch.chunk_text

    # Full text coverage verification: start at word0, end at word24
    assert "word0" in chunks[0].chunk_text
    assert "word24" in chunks[-1].chunk_text
    assert chunks[-1].end_offset == len(text)



def test_chunking_min_words_merges_trailing_fragment() -> None:
    # 22 words total: first chunk takes 20 words, leaving 2 words (less than min_words=5), which gets merged
    words = [f"item{i}" for i in range(22)]
    text = " ".join(words)
    chunker = ChunkingService(target_words=20, overlap_words=5, min_words=5)
    chunks = chunker.chunk_text(text)

    # Should merge the trailing 2 words into the first chunk rather than creating a tiny 2-word chunk
    assert len(chunks) == 1
    assert "item21" in chunks[0].chunk_text


def test_chunking_overlap_zero() -> None:
    words = [f"token{i}" for i in range(25)]
    text = " ".join(words)
    chunker = ChunkingService(target_words=10, overlap_words=0, min_words=1)
    chunks = chunker.chunk_text(text)

    # 25 words with target=10, overlap=0: chunk0 (0..9), chunk1 (10..19), chunk2 (20..24)
    assert len(chunks) == 3
    assert chunks[0].chunk_text == " ".join(words[0:10])
    assert chunks[1].chunk_text == " ".join(words[10:20])
    assert chunks[2].chunk_text == " ".join(words[20:25])
    assert chunks[0].start_offset == 0
    assert chunks[-1].end_offset == len(text)


def test_chunking_target_plus_one_boundary() -> None:
    # Exactly target_words + 1 words (11 words for target=10)
    words = [f"elem{i}" for i in range(11)]
    text = " ".join(words)

    # Case A: min_words=1 allows creating the second chunk
    chunker_strict = ChunkingService(target_words=10, overlap_words=3, min_words=1)
    chunks_strict = chunker_strict.chunk_text(text)
    assert len(chunks_strict) == 2
    assert "elem0" in chunks_strict[0].chunk_text
    assert "elem10" in chunks_strict[1].chunk_text
    assert chunks_strict[-1].end_offset == len(text)

    # Case B: min_words=5 absorbs the 1 trailing word into chunk 0
    chunker_absorb = ChunkingService(target_words=10, overlap_words=3, min_words=5)
    chunks_absorb = chunker_absorb.chunk_text(text)
    assert len(chunks_absorb) == 1
    assert "elem0" in chunks_absorb[0].chunk_text
    assert "elem10" in chunks_absorb[0].chunk_text
    assert chunks_absorb[0].end_offset == len(text)




def test_chunking_deterministic() -> None:
    chunker = ChunkingService(target_words=15, overlap_words=5)
    text = "The quick brown fox jumps over the lazy dog. Clinical evaluation shows normal heart sounds and clear lungs." * 5
    run1 = chunker.chunk_text(text)
    run2 = chunker.chunk_text(text)

    assert len(run1) == len(run2)
    for c1, c2 in zip(run1, run2):
        assert c1.chunk_index == c2.chunk_index
        assert c1.chunk_text == c2.chunk_text
        assert c1.start_offset == c2.start_offset
        assert c1.end_offset == c2.end_offset


def test_chunking_preserves_numbers_and_punctuation() -> None:
    chunker = ChunkingService(target_words=50, overlap_words=10)
    text = "BP: 140/90 mmHg, HbA1c: 8.2%, SpO2: 98%, Dosage: 500 mg q12h."
    chunks = chunker.chunk_text(text)
    assert len(chunks) == 1
    assert "140/90 mmHg" in chunks[0].chunk_text
    assert "8.2%" in chunks[0].chunk_text
    assert "500 mg q12h" in chunks[0].chunk_text


# ==========================================
# 2. EmbeddingService Unit Tests
# ==========================================

def test_embedding_service_dimension() -> None:
    service = EmbeddingService()
    # Mock underlying SentenceTransformer to avoid loading weights in unit tests
    mock_model = MagicMock()
    mock_model.get_embedding_dimension.return_value = 384
    service._model = mock_model

    assert service.dimension == 384


def test_embedding_service_encode_mocked() -> None:
    service = EmbeddingService()
    mock_model = MagicMock()
    import numpy as np
    mock_vec = np.ones(384, dtype=float)
    mock_model.encode.return_value = mock_vec
    service._model = mock_model

    vec = service.encode("test clinical text")
    assert len(vec) == 384
    assert vec[0] == 1.0


def test_embedding_service_empty_text() -> None:
    service = EmbeddingService()
    vec = service.encode("")
    assert len(vec) == 384
    assert all(v == 0.0 for v in vec)


# ==========================================
# 3. LLMService Unit Tests
# ==========================================

def test_llm_service_generate_text_success() -> None:
    mock_client = MagicMock()
    mock_choice = MagicMock()
    mock_choice.message.content = "Metformin 500mg twice daily was prescribed for elevated blood glucose."
    mock_client.chat.completions.create.return_value = MagicMock(choices=[mock_choice])

    service = LLMService(client=mock_client)
    answer = service.generate_text("What medication was prescribed?")
    assert "Metformin 500mg" in answer


def test_llm_service_generate_text_timeout() -> None:
    from groq import APITimeoutError
    mock_client = MagicMock()
    mock_client.chat.completions.create.side_effect = APITimeoutError("timeout")

    service = LLMService(client=mock_client)
    with pytest.raises(LLMTimeoutError):
        service.generate_text("test prompt")


def test_llm_service_generate_text_connection_error() -> None:
    from groq import APIConnectionError
    mock_client = MagicMock()
    mock_client.chat.completions.create.side_effect = APIConnectionError(request=MagicMock())

    service = LLMService(client=mock_client)
    with pytest.raises(LLMServiceError):
        service.generate_text("test prompt")


# ==========================================
# 4. RAG Indexing API Tests
# ==========================================

def test_index_document_happy_path(
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

    # Mock embedder to return 384-dim dummy vectors
    mock_encode_batch = MagicMock(return_value=[[0.1] * 384])
    monkeypatch.setattr(rag_service_module.rag_service.embedder, "encode_batch", mock_encode_batch)

    response = client.post(f"/api/v1/documents/{doc_id}/index", headers=auth_headers)
    assert response.status_code == 200
    data = response.json()
    assert data["document_id"] == doc_id
    assert data["chunks_created"] >= 1
    assert data["status"] == "indexed"

    # Verify chunks exist in DB
    check_db = database()
    try:
        chunks = check_db.query(DocumentChunk).filter(DocumentChunk.document_id == UUID(doc_id)).all()
        assert len(chunks) >= 1
        assert "Metformin 500mg" in chunks[0].chunk_text
    finally:
        check_db.close()


def test_index_document_reindexing_idempotency(
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

    mock_encode_batch = MagicMock(return_value=[[0.1] * 384])
    monkeypatch.setattr(rag_service_module.rag_service.embedder, "encode_batch", mock_encode_batch)

    # First index call
    res1 = client.post(f"/api/v1/documents/{doc_id}/index", headers=auth_headers)
    assert res1.status_code == 200
    count1 = res1.json()["chunks_created"]

    # Second index call (should cleanly replace old chunks)
    res2 = client.post(f"/api/v1/documents/{doc_id}/index", headers=auth_headers)
    assert res2.status_code == 200
    count2 = res2.json()["chunks_created"]
    assert count1 == count2

    check_db = database()
    try:
        chunks = check_db.query(DocumentChunk).filter(DocumentChunk.document_id == UUID(doc_id)).all()
        assert len(chunks) == count1
    finally:
        check_db.close()


def test_index_document_privacy_gate(
    client: TestClient,
    auth_headers: dict[str, str],
    database: sessionmaker[Session],
    registered_user: dict,
) -> None:
    db = database()
    try:
        user_id = UUID(registered_user["user_id"])
        doc = _create_test_document(db, user_id, privacy_status="pending")
        doc_id = str(doc.document_id)
    finally:
        db.close()

    response = client.post(f"/api/v1/documents/{doc_id}/index", headers=auth_headers)
    assert response.status_code == 400
    assert "Sanitized text unavailable" in response.json()["detail"]


def test_index_document_empty_sanitized_text(
    client: TestClient,
    auth_headers: dict[str, str],
    database: sessionmaker[Session],
    registered_user: dict,
) -> None:
    db = database()
    try:
        user_id = UUID(registered_user["user_id"])
        doc = _create_test_document(db, user_id, privacy_status="completed", sanitized_text="")
        doc_id = str(doc.document_id)
    finally:
        db.close()

    response = client.post(f"/api/v1/documents/{doc_id}/index", headers=auth_headers)
    assert response.status_code == 400
    assert "Sanitized OCR text is empty" in response.json()["detail"]


def test_index_document_unauthorized_access(
    client: TestClient,
    auth_headers: dict[str, str],
    database: sessionmaker[Session],
) -> None:
    # Document owned by someone else
    db = database()
    try:
        other_user_id = uuid4()
        doc = _create_test_document(db, other_user_id)
        doc_id = str(doc.document_id)
    finally:
        db.close()

    response = client.post(f"/api/v1/documents/{doc_id}/index", headers=auth_headers)
    assert response.status_code == 403
    assert response.json()["detail"] == "Access denied"


def test_index_document_unauthenticated(client: TestClient) -> None:
    response = client.post(f"/api/v1/documents/{uuid4()}/index")
    assert response.status_code == 401


# ==========================================
# 5. RAG Search API Tests
# ==========================================

def test_search_document_happy_path(
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

    # Mock embeddings and LLM text generation
    monkeypatch.setattr(rag_service_module.rag_service.embedder, "encode", MagicMock(return_value=[0.1] * 384))
    monkeypatch.setattr(rag_service_module.rag_service.embedder, "encode_batch", MagicMock(return_value=[[0.1] * 384]))

    mock_generate = MagicMock(return_value="The patient was prescribed Metformin 500mg twice daily.")
    monkeypatch.setattr(rag_service_module.rag_service.llm, "generate_text", mock_generate)

    response = client.post(
        f"/api/v1/documents/{doc_id}/search",
        json={"query": "What medication was prescribed?", "top_k": 3},
        headers=auth_headers,
    )

    assert response.status_code == 200
    data = response.json()
    assert "Metformin 500mg" in data["answer"]
    assert len(data["sources"]) >= 1
    assert data["sources"][0]["chunk_index"] == 0
    assert "Metformin 500mg" in data["sources"][0]["content_preview"]
    mock_generate.assert_called_once()


def test_search_document_privacy_boundary_strictly_enforced(
    client: TestClient,
    auth_headers: dict[str, str],
    database: sessionmaker[Session],
    registered_user: dict,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """
    CRITICAL PRIVACY TEST:
    Verify raw OCR text containing real PHI (e.g. 'Rahul Kumar', 'MRN12345')
    NEVER reaches the prompt passed to the LLM.
    """
    db = database()
    try:
        user_id = UUID(registered_user["user_id"])
        raw_phi_text = "CONFIDENTIAL PATIENT: Rahul Kumar, SSN: 999-00-1234, Diagnosis: Type 2 Diabetes"
        sanitized_clean_text = "CONFIDENTIAL PATIENT: [PATIENT], SSN: [SSN], Diagnosis: Type 2 Diabetes"
        doc = _create_test_document(db, user_id, raw_text=raw_phi_text, sanitized_text=sanitized_clean_text)
        doc_id = str(doc.document_id)
    finally:
        db.close()

    monkeypatch.setattr(rag_service_module.rag_service.embedder, "encode", MagicMock(return_value=[0.1] * 384))
    monkeypatch.setattr(rag_service_module.rag_service.embedder, "encode_batch", MagicMock(return_value=[[0.1] * 384]))

    captured_prompt = None

    def mock_generate_text(prompt: str, system_prompt: str | None = None, temperature: float = 0.1) -> str:
        nonlocal captured_prompt
        captured_prompt = prompt
        return "The patient is diagnosed with Type 2 Diabetes."

    monkeypatch.setattr(rag_service_module.rag_service.llm, "generate_text", mock_generate_text)

    response = client.post(
        f"/api/v1/documents/{doc_id}/search",
        json={"query": "What is the diagnosis?", "top_k": 2},
        headers=auth_headers,
    )

    assert response.status_code == 200
    assert captured_prompt is not None
    # PHI checks
    assert "Rahul Kumar" not in captured_prompt
    assert "999-00-1234" not in captured_prompt
    # Sanitized tokens must be present
    assert "[PATIENT]" in captured_prompt
    assert "Type 2 Diabetes" in captured_prompt


def test_search_document_unauthorized_access(
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

    response = client.post(
        f"/api/v1/documents/{doc_id}/search",
        json={"query": "What are the lab results?"},
        headers=auth_headers,
    )
    assert response.status_code == 403
    assert response.json()["detail"] == "Access denied"


def test_search_document_timeout_error(
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

    monkeypatch.setattr(rag_service_module.rag_service.embedder, "encode", MagicMock(return_value=[0.1] * 384))
    monkeypatch.setattr(rag_service_module.rag_service.embedder, "encode_batch", MagicMock(return_value=[[0.1] * 384]))
    monkeypatch.setattr(
        rag_service_module.rag_service.llm,
        "generate_text",
        MagicMock(side_effect=LLMTimeoutError("Groq request timed out")),
    )

    response = client.post(
        f"/api/v1/documents/{doc_id}/search",
        json={"query": "What is the diagnosis?"},
        headers=auth_headers,
    )
    assert response.status_code == 504
    assert "timed out" in response.json()["detail"]


def test_search_document_missing_api_key(
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

    monkeypatch.setattr(rag_service_module.rag_service.embedder, "encode", MagicMock(return_value=[0.1] * 384))
    monkeypatch.setattr(rag_service_module.rag_service.embedder, "encode_batch", MagicMock(return_value=[[0.1] * 384]))
    monkeypatch.setattr(
        rag_service_module.rag_service.llm,
        "generate_text",
        MagicMock(side_effect=LLMConfigurationError("Missing API key")),
    )

    response = client.post(
        f"/api/v1/documents/{doc_id}/search",
        json={"query": "What is the diagnosis?"},
        headers=auth_headers,
    )
    assert response.status_code == 503
    assert "not configured" in response.json()["detail"]


def test_search_document_fallback_when_answer_not_in_context(
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

    monkeypatch.setattr(rag_service_module.rag_service.embedder, "encode", MagicMock(return_value=[0.1] * 384))
    monkeypatch.setattr(rag_service_module.rag_service.embedder, "encode_batch", MagicMock(return_value=[[0.1] * 384]))

    fallback_answer = "The provided document excerpts do not contain sufficient information to answer this question."
    mock_generate = MagicMock(return_value=fallback_answer)
    monkeypatch.setattr(rag_service_module.rag_service.llm, "generate_text", mock_generate)

    response = client.post(
        f"/api/v1/documents/{doc_id}/search",
        json={"query": "What is the patient's favorite hobby?", "top_k": 3},
        headers=auth_headers,
    )

    assert response.status_code == 200
    data = response.json()
    assert data["answer"] == fallback_answer
    assert isinstance(data["sources"], list)
    assert len(data["sources"]) >= 1
    mock_generate.assert_called_once()

