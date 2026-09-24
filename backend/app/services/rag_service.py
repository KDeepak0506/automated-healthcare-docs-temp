import logging
import re
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
            f"3. In laboratory reports, note that an isolated 'H' or 'High' flag indicates an abnormally elevated value above reference range, and 'L' or 'Low' indicates a subnormal value below reference range. Qualitative results such as '1+' or 'Present (+)' where normal is 'Negative' or 'Absent' are abnormal.\n"
            f"4. If the excerpts do not contain the answer, state: 'The provided document excerpts do not contain sufficient information to answer this question.'\n"
            f"5. Do NOT diagnose, recommend clinical treatments, or speculate beyond the provided text.\n"
            f"6. Provide a clear, concise, and direct response."
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

    @staticmethod
    def classify_query(query: str) -> str:
        """
        Classifies clinical queries into:
        - IMAGING_DIAGNOSTIC: queries on radiology, imaging, scans, X-rays, ultrasound, ECG.
        - COMPREHENSIVE_ABNORMAL: queries asking for all abnormal findings, flagged values, or comprehensive summaries.
        - NARROW_FACTUAL: specific parameter, test, or factual clinical questions.
        """
        q_lower = query.lower()
        if re.search(
            r"\b(imaging|x-?ray|radiograph|radiology|ultrasound|usg|sonograph|ecg|ekg|electrocardiogram|echo|echocardiogram|ct(\s+scan)?|mri|chest\s+(pa|view)?)\b",
            q_lower,
        ):
            return "IMAGING_DIAGNOSTIC"
        if re.search(
            r"\b(abnormal|abnormalities|findings|flagged|flags|h-?flagged|l-?flagged|out\s+of\s+range|high\s+or\s+low|all\s+lab\s+results|summarize\s+all|summary\s+of\s+all|list\s+all|every\s+test|all\s+tests|all\s+findings)\b",
            q_lower,
        ):
            return "COMPREHENSIVE_ABNORMAL"
        return "NARROW_FACTUAL"

    def _extract_query_keywords(self, query: str) -> list[str]:
        """Extract key clinical terms from query for keyword-augmented retrieval."""
        stopwords = {
            "what", "is", "the", "patient", "patient's", "level", "levels", "value",
            "values", "test", "tests", "result", "results", "was", "were", "for",
            "with", "does", "have", "any", "report", "reported", "show", "showed",
            "please", "tell", "about", "find", "check"
        }
        words = [w.lower().strip("?,.!;:\'\"()[]{}") for w in query.split()]
        cleaned = [w for w in words if len(w) >= 2 and w not in stopwords]

        terms = set(cleaned)
        if "hemoglobin" in terms or "hb" in terms:
            terms.update(["hemoglobin", "hb", "haemoglobin"])
        if "sugar" in terms or "glucose" in terms:
            terms.update(["glucose", "sugar", "fbs"])
        if "platelet" in terms or "platelets" in terms:
            terms.update(["platelet", "platelets"])
        if "wbc" in terms or "leukocyte" in terms:
            terms.update(["wbc", "leukocyte"])

        return list(terms)

    def _score_chunk_for_narrow_query(
        self,
        chunk_text: str,
        query_terms: list[str],
        base_sim: float,
    ) -> float:
        """Scores a chunk considering dense cosine similarity and direct test result proximity."""
        text_lower = chunk_text.lower()
        score = base_sim

        has_direct_result = False
        for term in query_terms:
            pattern_fwd = rf"(?i)\b{re.escape(term)}\b[\s\S]{{0,120}}?\b\d+(?:\.\d+)?\s*(?:g/dl|gm/dl|mg/dl|mcg/dl|mmol/l|umol/l|µmol/l|mEq/l|u/l|iu/l|uiu/ml|miu/ml|ng/ml|pg/ml|bpm|°f|°c|mmhg|g%|vol%|cells/ul|/ul|/hpf|ml/min|lakhs/cumm|cumm|/cmm|%|fl|pg)\b"
            pattern_rev = rf"(?i)\b\d+(?:\.\d+)?\s*(?:g/dl|gm/dl|mg/dl|mcg/dl|mmol/l|umol/l|µmol/l|mEq/l|u/l|iu/l|uiu/ml|miu/ml|ng/ml|pg/ml|bpm|°f|°c|mmhg|g%|vol%|cells/ul|/ul|/hpf|ml/min|lakhs/cumm|cumm|/cmm|%|fl|pg)\b[\s\S]{{0,120}}?\b{re.escape(term)}\b"
            if re.search(pattern_fwd, chunk_text) or re.search(pattern_rev, chunk_text):
                has_direct_result = True
                break

        if has_direct_result:
            score += 0.40
        elif any(re.search(r"\b" + re.escape(t) + r"\b", text_lower) for t in query_terms):
            score += 0.15

        if not has_direct_result and any(
            w in text_lower
            for w in [
                "factors that interfere",
                "assay interferences",
                "further dna studies",
                "denatured froms of hemoglobins",
                "clinically correlated",
            ]
        ):
            score -= 0.15

        return score

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
        3. Routes query:
           - IMAGING_DIAGNOSTIC: aggregates all authorized imaging & diagnostic sections.
           - COMPREHENSIVE_ABNORMAL: aggregates lab flags, qualitative abnormalities, and clinical findings across all docs.
           - NARROW_FACTUAL: balanced cross-document retrieval ensuring multi-document diversity.
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

        # 3. Classify query intent
        q_type = self.classify_query(query)
        selected_chunks_with_scores: list[tuple[DocumentChunk, float]] = []

        if q_type == "IMAGING_DIAGNOSTIC":
            pattern = re.compile(
                r"(?i)\b(x-?ray|radiograph|radiology|ultrasound|usg|sonograph|ecg|ekg|electrocardiogram|echo|echocardiogram|ct\s+scan|mri|chest|impression|findings)\b"
            )
            for doc in docs:
                chunks = (
                    db.query(DocumentChunk)
                    .filter(DocumentChunk.document_id == doc.document_id)
                    .order_by(DocumentChunk.chunk_index)
                    .all()
                )
                for c in chunks:
                    if pattern.search(c.chunk_text):
                        selected_chunks_with_scores.append((c, 0.95))

        elif q_type == "COMPREHENSIVE_ABNORMAL":
            flag_pattern = re.compile(
                r"(?i)(?:\b[HL]\b|\b(High|Low|Borderline High|Very High)\b|\b(1\+|2\+|3\+|Present\s*\(\+\)|Present|Positive|Reactive|Abnormal)\b|\b(IMPRESSION|FINDINGS|Clinical Notes)\b)"
            )
            for doc in docs:
                chunks = (
                    db.query(DocumentChunk)
                    .filter(DocumentChunk.document_id == doc.document_id)
                    .order_by(DocumentChunk.chunk_index)
                    .all()
                )
                # If document has <= 2 chunks (e.g. CBC or Prescription), include all chunks to prevent missing concise findings
                if len(chunks) <= 2:
                    for c in chunks:
                        selected_chunks_with_scores.append((c, 0.90))
                else:
                    for c in chunks:
                        if flag_pattern.search(c.chunk_text):
                            selected_chunks_with_scores.append((c, 0.85))

        else:
            # NARROW_FACTUAL: balanced cross-document retrieval
            query_embedding = self.embedder.encode(query)
            q_vec = np.array(query_embedding, dtype=float)
            q_norm = np.linalg.norm(q_vec)
            query_terms = self._extract_query_keywords(query)

            doc_candidates: dict[UUID, list[tuple[DocumentChunk, float, float]]] = {}
            for doc in docs:
                chunks = (
                    db.query(DocumentChunk)
                    .filter(DocumentChunk.document_id == doc.document_id)
                    .order_by(DocumentChunk.chunk_index)
                    .all()
                )
                scored = []
                for c in chunks:
                    if c.embedding is not None and len(c.embedding) > 0:
                        c_vec = np.array(c.embedding, dtype=float)
                        c_norm = np.linalg.norm(c_vec)
                        sim = float(np.dot(q_vec, c_vec) / (q_norm * c_norm)) if (q_norm > 0 and c_norm > 0) else 0.0
                    else:
                        sim = 0.0

                    final_score = self._score_chunk_for_narrow_query(c.chunk_text, query_terms, sim)
                    scored.append((c, final_score, sim))

                scored.sort(key=lambda x: x[1], reverse=True)
                if scored:
                    doc_candidates[doc.document_id] = scored

            # Guarantee representation from each document with matching terms or notable similarity
            chosen: list[tuple[DocumentChunk, float]] = []
            for doc_id, scored in doc_candidates.items():
                best_c, best_score, best_sim = scored[0]
                if best_score > 0.25 or any(kw in best_c.chunk_text.lower() for kw in query_terms):
                    chosen.append((best_c, best_sim))

            # Fill remaining slots up to top_k with top runner-ups across documents
            all_runner_ups = []
            for doc_id, scored in doc_candidates.items():
                for c, sc, sim in scored[1:3]:
                    all_runner_ups.append((c, sc, sim))
            all_runner_ups.sort(key=lambda x: x[1], reverse=True)

            for c, sc, sim in all_runner_ups:
                if len(chosen) >= max(top_k, len(chosen)):
                    break
                if not any(x[0].chunk_id == c.chunk_id for x in chosen):
                    chosen.append((c, sim))

            selected_chunks_with_scores = chosen

        # Deduplicate chunks
        seen_ids = set()
        unique_chunks_with_scores: list[tuple[DocumentChunk, float]] = []
        for c, sc in selected_chunks_with_scores:
            if c.chunk_id not in seen_ids:
                seen_ids.add(c.chunk_id)
                unique_chunks_with_scores.append((c, sc))

        if not unique_chunks_with_scores:
            return SearchResponse(
                answer="No relevant text could be found across the patient's records to answer your query.",
                sources=[],
            )

        # 4. Build context text and source references
        sources: list[SourceReference] = []
        context_blocks: list[str] = []

        for idx, (chunk, score) in enumerate(unique_chunks_with_scores, start=1):
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
            f"You are an expert clinical AI assistant answering a question based strictly on excerpts from a patient's de-identified medical records.\n\n"
            f"Patient Record Excerpts:\n\"\"\"\n{context_text}\n\"\"\"\n\n"
            f"User Question: {query}\n\n"
            f"Clinical Instructions:\n"
            f"1. Answer factually using ONLY the facts and values directly stated in the excerpts above.\n"
            f"2. Cite the specific document name(s) when presenting findings or test values.\n"
            f"3. In laboratory reports, note that an isolated 'H' or 'High' flag indicates an abnormally elevated value above reference range, and 'L' or 'Low' indicates a subnormal value below reference range. Qualitative results such as '1+' or 'Present (+)' where normal is 'Negative' or 'Absent' are abnormal.\n"
            f"4. If relevant values or findings appear across multiple documents, report all reported values with their respective document sources and units (do not omit values from other documents).\n"
            f"5. Preserve all numerical measurements, reference ranges, and units exactly as stated.\n"
            f"6. If the provided excerpts do not mention or contain any information regarding the question, clearly state that no records or information were found for that question.\n"
            f"7. Do NOT diagnose, recommend clinical treatments, or speculate beyond the provided text.\n"
            f"8. Provide a clear, comprehensive, and well-structured response."
        )

        # 5. Generate LLM completion
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
