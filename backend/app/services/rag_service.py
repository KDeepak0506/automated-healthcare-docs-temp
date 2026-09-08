import logging
from uuid import UUID
import numpy as np
from fastapi import HTTPException, status
from sqlalchemy import func
from sqlalchemy.orm import Session

from app.core.config import settings
from app.models.document import Document
from app.models.document_chunk import DocumentChunk
from app.models.document_text import DocumentText
from app.schemas.rag import IndexResponse, SearchResponse, SourceReference
from app.services.chunking_service import ChunkingService, chunking_service as default_chunking_service
from app.services.embedding_service import EmbeddingService, embedding_service as default_embedding_service
from app.services.llm_service import (
    LLMConfigurationError,
    LLMResponseParsingError,
    LLMService,
    LLMServiceError,
    LLMTimeoutError,
    llm_service as default_llm_service,
)

logger = logging.getLogger(__name__)

SYSTEM_PROMPT = (
    "You are an expert clinical AI assistant working within a strict healthcare privacy boundary. "
    "Answer questions accurately and factually based ONLY on the provided de-identified document excerpts. "
    "Preserve all numerical values, test results, dosages, and units exactly without alteration. "
    "Never diagnose, invent facts, or recommend clinical treatments. "
    "If the information is not contained in the excerpts, state clearly that the document does not contain that information."
)


class RagService:
    """
    Service for M6 RAG / AI Search.
    Indexes sanitized document chunks into vector embeddings and performs semantic search + Q&A.
    STRICTLY consumes sanitized_text ONLY (never raw OCR).
    """

    def __init__(
        self,
        llm: LLMService | None = None,
        chunker: ChunkingService | None = None,
        embedder: EmbeddingService | None = None,
    ) -> None:
        self.llm = llm or default_llm_service
        self.chunker = chunker or default_chunking_service
        self.embedder = embedder or default_embedding_service

    def index_document(
        self,
        db: Session,
        document: Document,
    ) -> IndexResponse:
        """
        Index a document for semantic search:
        1. Verifies privacy status is completed.
        2. Chunks sanitized OCR text.
        3. Generates dense embeddings.
        4. Persists chunks into document_chunks table (idempotent).
        """
        # 1. Verify privacy status
        if document.privacy_status != "completed":
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail="Sanitized text unavailable. Privacy processing has not completed or failed.",
            )

        # 2. Retrieve document text record
        doc_text = (
            db.query(DocumentText)
            .filter(DocumentText.document_id == document.document_id)
            .first()
        )

        if doc_text is None or not doc_text.sanitized_text or not doc_text.sanitized_text.strip():
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail="Sanitized OCR text is empty or missing for this document.",
            )

        sanitized_text = doc_text.sanitized_text.strip()

        # 3. Clear any existing chunks for idempotency / re-indexing
        db.query(DocumentChunk).filter(
            DocumentChunk.document_id == document.document_id
        ).delete(synchronize_session=False)

        # 4. Chunk sanitized text
        chunks_info = self.chunker.chunk_text(sanitized_text)
        if not chunks_info:
            db.commit()
            return IndexResponse(
                document_id=document.document_id,
                chunks_created=0,
                status="empty_text",
            )

        # 5. Compute vector embeddings in batch
        chunk_texts = [c.chunk_text for c in chunks_info]
        embeddings = self.embedder.encode_batch(chunk_texts)

        # 6. Save chunks to database
        for info, emb in zip(chunks_info, embeddings):
            chunk_record = DocumentChunk(
                document_id=document.document_id,
                chunk_index=info.chunk_index,
                chunk_text=info.chunk_text,
                start_offset=info.start_offset,
                end_offset=info.end_offset,
                page_number=info.page_number,
                embedding=emb,
            )
            db.add(chunk_record)

        db.commit()

        logger.info(
            f"Successfully indexed document {document.document_id} with {len(chunks_info)} chunks"
        )

        return IndexResponse(
            document_id=document.document_id,
            chunks_created=len(chunks_info),
            status="indexed",
        )

    def search_document(
        self,
        db: Session,
        document: Document,
        query: str,
        top_k: int = 5,
    ) -> SearchResponse:
        """
        Perform RAG Q&A on a document:
        1. Verifies privacy status is completed.
        2. Retrieves top-k most relevant chunks using semantic vector search.
        3. Synthesizes a grounded answer via LLM.
        """
        # 1. Verify privacy status
        if document.privacy_status != "completed":
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail="Sanitized text unavailable. Privacy processing has not completed or failed.",
            )

        # 2. Check if chunks exist; auto-index if necessary
        chunk_count = (
            db.query(func.count(DocumentChunk.chunk_id))
            .filter(DocumentChunk.document_id == document.document_id)
            .scalar()
        )

        if not chunk_count:
            self.index_document(db, document)

        # 3. Generate query embedding
        query_embedding = self.embedder.encode(query)

        # 4. Retrieve top_k chunks by cosine similarity
        is_postgres = db.bind is not None and db.bind.dialect.name == "postgresql"

        if is_postgres:
            distance_expr = DocumentChunk.embedding.cosine_distance(query_embedding)
            query_results = (
                db.query(DocumentChunk, distance_expr.label("distance"))
                .filter(DocumentChunk.document_id == document.document_id)
                .order_by("distance")
                .limit(top_k)
                .all()
            )
            top_chunks = [r[0] for r in query_results]
            # Convert cosine distance to similarity: similarity = 1 - distance
            scores = [max(0.0, 1.0 - float(r[1])) if r[1] is not None else 1.0 for r in query_results]
        else:
            # SQLite fallback for test environment
            all_chunks = (
                db.query(DocumentChunk)
                .filter(DocumentChunk.document_id == document.document_id)
                .order_by(DocumentChunk.chunk_index)
                .all()
            )
            q_vec = np.array(query_embedding, dtype=float)
            q_norm = np.linalg.norm(q_vec)

            scored_chunks = []
            for c in all_chunks:
                if c.embedding is not None and len(c.embedding) > 0:
                    c_vec = np.array(c.embedding, dtype=float)
                    c_norm = np.linalg.norm(c_vec)
                    sim = float(np.dot(q_vec, c_vec) / (q_norm * c_norm)) if (q_norm > 0 and c_norm > 0) else 0.0
                else:
                    sim = 0.0
                scored_chunks.append((c, sim))

            scored_chunks.sort(key=lambda x: x[1], reverse=True)
            top_selected = scored_chunks[:top_k]
            top_chunks = [x[0] for x in top_selected]
            scores = [x[1] for x in top_selected]

        if not top_chunks:
            return SearchResponse(
                answer="No relevant text could be found in the document to answer your query.",
                sources=[],
            )

        # 5. Build prompt and source references
        sources: list[SourceReference] = []
        context_blocks: list[str] = []

        for idx, (chunk, score) in enumerate(zip(top_chunks, scores), start=1):
            preview = chunk.chunk_text[:150] + ("..." if len(chunk.chunk_text) > 150 else "")
            sources.append(
                SourceReference(
                    chunk_id=chunk.chunk_id,
                    chunk_index=chunk.chunk_index,
                    page_number=chunk.page_number,
                    similarity_score=round(score, 4) if score is not None else None,
                    content_preview=preview,
                )
            )
            context_blocks.append(f"[Excerpt {idx} (Chunk #{chunk.chunk_index})]:\n{chunk.chunk_text}")

        context_text = "\n\n".join(context_blocks)
        prompt = (
            f"You are answering a question based strictly on excerpts from a de-identified clinical document.\n\n"
            f"Document Excerpts:\n\"\"\"\n{context_text}\n\"\"\"\n\n"
            f"User Question: {query}\n\n"
            f"Strict Instructions:\n"
            f"1. Answer using ONLY the facts and values directly stated in the excerpts above.\n"
            f"2. Preserve all numerical measurements, lab values, percentages, and units exactly without alteration or rounding.\n"
            f"3. If the excerpts do not contain the answer, state: 'The provided document excerpts do not contain sufficient information to answer this question.'\n"
            f"4. Do NOT diagnose, recommend clinical treatments, or speculate beyond the provided text.\n"
            f"5. Provide a clear, concise, and direct response."
        )

        # 6. Execute LLM completion
        try:
            answer = self.llm.generate_text(
                prompt=prompt,
                system_prompt=SYSTEM_PROMPT,
                temperature=0.1,
            )
        except LLMConfigurationError as exc:
            logger.error(f"RAG search failed: {exc}")
            raise HTTPException(
                status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
                detail="Document search service is not configured (missing API key).",
            ) from exc
        except LLMTimeoutError as exc:
            logger.error(f"RAG search timed out: {exc}")
            raise HTTPException(
                status_code=status.HTTP_504_GATEWAY_TIMEOUT,
                detail="Document search request timed out.",
            ) from exc
        except LLMServiceError as exc:
            logger.error(f"RAG provider error: {exc}")
            raise HTTPException(
                status_code=status.HTTP_502_BAD_GATEWAY,
                detail="Failed to perform AI search on document.",
            ) from exc

        return SearchResponse(
            answer=answer,
            sources=sources,
        )

    def search_patient(
        self,
        db: Session,
        patient_id: UUID,
        query: str,
        top_k: int = 5,
    ) -> SearchResponse:
        """
        Perform patient-scoped RAG Q&A across ALL authorized documents belonging to patient_id.
        1. Retrieves all documents for patient_id with privacy_status == 'completed'.
        2. Auto-indexes any documents that haven't been chunked/indexed.
        3. Performs vector similarity search across all patient document chunks.
        4. Synthesizes a grounded clinical answer with multi-document source references.
        """
        # 1. Retrieve all completed documents for patient
        docs = (
            db.query(Document)
            .filter(
                Document.patient_id == patient_id,
                Document.privacy_status == "completed",
            )
            .all()
        )

        if not docs:
            return SearchResponse(
                answer="No de-identified, processed documents are currently available for this patient.",
                sources=[],
            )

        # 2. Auto-index any document lacking chunks
        doc_ids = [d.document_id for d in docs]
        doc_map = {d.document_id: d for d in docs}

        for doc in docs:
            chunk_count = (
                db.query(func.count(DocumentChunk.chunk_id))
                .filter(DocumentChunk.document_id == doc.document_id)
                .scalar()
            )
            if not chunk_count:
                try:
                    self.index_document(db, doc)
                except Exception as exc:
                    logger.warning(f"Could not auto-index document {doc.document_id} for patient RAG: {exc}")

        # 3. Generate query embedding
        query_embedding = self.embedder.encode(query)

        # 4. Search across all patient document chunks
        is_postgres = db.bind is not None and db.bind.dialect.name == "postgresql"

        if is_postgres:
            distance_expr = DocumentChunk.embedding.cosine_distance(query_embedding)
            query_results = (
                db.query(DocumentChunk, distance_expr.label("distance"))
                .filter(DocumentChunk.document_id.in_(doc_ids))
                .order_by("distance")
                .limit(top_k)
                .all()
            )
            top_chunks = [r[0] for r in query_results]
            scores = [max(0.0, 1.0 - float(r[1])) if r[1] is not None else 1.0 for r in query_results]
        else:
            all_chunks = (
                db.query(DocumentChunk)
                .filter(DocumentChunk.document_id.in_(doc_ids))
                .order_by(DocumentChunk.chunk_index)
                .all()
            )
            q_vec = np.array(query_embedding, dtype=float)
            q_norm = np.linalg.norm(q_vec)

            scored_chunks = []
            for c in all_chunks:
                if c.embedding is not None and len(c.embedding) > 0:
                    c_vec = np.array(c.embedding, dtype=float)
                    c_norm = np.linalg.norm(c_vec)
                    sim = float(np.dot(q_vec, c_vec) / (q_norm * c_norm)) if (q_norm > 0 and c_norm > 0) else 0.0
                else:
                    sim = 0.0
                scored_chunks.append((c, sim))

            scored_chunks.sort(key=lambda x: x[1], reverse=True)
            top_selected = scored_chunks[:top_k]
            top_chunks = [x[0] for x in top_selected]
            scores = [x[1] for x in top_selected]

        if not top_chunks:
            return SearchResponse(
                answer="No relevant text could be found across the patient's records to answer your query.",
                sources=[],
            )

        # 5. Build context text and source references
        sources: list[SourceReference] = []
        context_blocks: list[str] = []

        for idx, (chunk, score) in enumerate(zip(top_chunks, scores), start=1):
            doc = doc_map.get(chunk.document_id)
            doc_name = doc.file_name if doc else str(chunk.document_id)
            preview = chunk.chunk_text[:150] + ("..." if len(chunk.chunk_text) > 150 else "")

            sources.append(
                SourceReference(
                    document_id=chunk.document_id,
                    chunk_id=chunk.chunk_id,
                    chunk_index=chunk.chunk_index,
                    page_number=chunk.page_number,
                    similarity_score=round(score, 4) if score is not None else None,
                    content_preview=f"[{doc_name}] {preview}",
                )
            )
            page_str = f", Page {chunk.page_number}" if chunk.page_number else ""
            context_blocks.append(
                f"[Excerpt {idx} - Document: {doc_name}{page_str} (Chunk #{chunk.chunk_index})]:\n{chunk.chunk_text}"
            )

        context_text = "\n\n".join(context_blocks)
        prompt = (
            f"You are answering a question based strictly on excerpts from a patient's de-identified clinical records.\n\n"
            f"Patient Record Excerpts:\n\"\"\"\n{context_text}\n\"\"\"\n\n"
            f"User Question: {query}\n\n"
            f"Strict Instructions:\n"
            f"1. Answer using ONLY the facts and values directly stated in the excerpts above.\n"
            f"2. Preserve all numerical measurements, lab values, percentages, and units exactly without alteration or rounding.\n"
            f"3. If the excerpts do not contain the answer, state: 'The provided patient record excerpts do not contain sufficient information to answer this question.'\n"
            f"4. Do NOT diagnose, recommend clinical treatments, or speculate beyond the provided text.\n"
            f"5. Provide a clear, concise, and direct response."
        )

        # 6. Generate LLM completion
        try:
            answer = self.llm.generate_text(
                prompt=prompt,
                system_prompt=SYSTEM_PROMPT,
                temperature=0.1,
            )
        except LLMConfigurationError as exc:
            logger.error(f"Patient RAG search failed: {exc}")
            raise HTTPException(
                status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
                detail="Patient AI search service is not configured (missing API key).",
            ) from exc
        except LLMTimeoutError as exc:
            logger.error(f"Patient RAG search timed out: {exc}")
            raise HTTPException(
                status_code=status.HTTP_504_GATEWAY_TIMEOUT,
                detail="Patient AI search request timed out.",
            ) from exc
        except LLMServiceError as exc:
            logger.error(f"Patient RAG provider error: {exc}")
            raise HTTPException(
                status_code=status.HTTP_502_BAD_GATEWAY,
                detail="Failed to perform patient AI search.",
            ) from exc

        return SearchResponse(
            answer=answer,
            sources=sources,
        )


# Singleton instance
rag_service = RagService()
