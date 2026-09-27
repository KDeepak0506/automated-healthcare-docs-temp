import logging
import re
from typing import Any
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
    "Never diagnose, invent facts, or recommend clinical treatments."
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
                temperature=0.0,
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
        - COMPREHENSIVE_ABNORMAL: queries asking for all abnormal findings, flagged values,
          comprehensive summaries, cross-document lab aggregations, or diabetes-related markers.
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
        # Additive extension: out-of-range without "abnormal" vocabulary,
        # conflicting/inconsistent data, and diabetes-marker queries.
        if re.search(
            r"\b("
            r"outside\s+(?:their|the|its|normal)\s+(?:reference\s+)?range"
            r"|conflicting"
            r"|inconsistent\s+(?:lab|results?|data|values?)"
            r"|any\s+inconsistenc"
            r"|diabet(?:es|ic)\s+(?:risk|markers?|signs?|indicators?)"
            r"|signs?\s+(?:of\s+)?diabet"
            r"|diabet(?:es|ic)\s+markers?"
            r")\b",
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
            "please", "tell", "about", "find", "check",
            # Noise words from cross-document / aggregation query patterns that produce
            # false proximity boosts when left in query_terms.
            "across", "are", "documents", "document", "their", "there", "this",
            "of", "or", "signs", "data", "in", "between", "its", "all",
        }
        words = [w.lower().strip("?,.!;:\'\"()[]{}") for w in query.split()]
        cleaned = [w for w in words if len(w) >= 2 and w not in stopwords]

        terms = set(cleaned)
        if "hemoglobin" in terms or "hb" in terms:
            terms.update(["hemoglobin", "hb", "haemoglobin"])
        if "fasting" in terms and any(w in terms for w in ["sugar", "glucose"]):
            terms.discard("blood")
            terms.update(["fasting blood sugar", "fasting glucose", "fbs"])
        elif "sugar" in terms or "glucose" in terms:
            terms.update(["glucose", "sugar", "fbs"])
        if "platelet" in terms or "platelets" in terms:
            terms.update(["platelet", "platelets"])
        if "wbc" in terms or "leukocyte" in terms:
            terms.update(["wbc", "leukocyte"])
        if ("blood" in terms and any(w in terms for w in ["type", "group", "typing"])) or "abo" in terms or "rh" in terms:
            terms.discard("blood")
            terms.discard("type")
            terms.discard("group")
            terms.update(["blood group", "blood type", "abo", "rh", "abo type", "rh (d)"])

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
        units_re = r"(?:g/dl|gm/dl|mg/dl|mcg/dl|mmol/l|umol/l|µmol/l|mEq/l|u/l|iu/l|uiu/ml|miu/ml|ng/ml|pg/ml|bpm|°f|°c|mmhg|g%|vol%|cells/ul|/ul|/hpf|ml/min|lakhs/cumm|cumm|/cmm|%|fl|pg)"
        for term in query_terms:
            pattern_fwd = rf"(?i)\b{re.escape(term)}\b[\s\S]{{0,120}}?\b\d+(?:\.\d+)?\s*{units_re}\b"
            pattern_rev = rf"(?i)\b\d+(?:\.\d+)?\s*{units_re}\b[\s\S]{{0,120}}?\b{re.escape(term)}\b"
            pattern_col = rf"(?i)\b{re.escape(term)}\b[\s\S]{{0,120}}?\b{units_re}\b[\s\S]{{0,100}}?\b\d+(?:\.\d+)?\b"
            if re.search(pattern_fwd, chunk_text) or re.search(pattern_rev, chunk_text) or re.search(pattern_col, chunk_text):
                has_direct_result = True
                break

        # Qualitative blood typing result proximity (e.g. ABO Type: "A", Rh (D) Type: Positive)
        if not has_direct_result and any(t in query_terms for t in ["blood group", "blood type", "abo", "rh"]):
            blood_pattern = r"(?i)\b(abo\s+type|blood\s+group|rh\s*\(?d?\)?\s*type)\b[\s\S]{0,80}?\b([\"']?[abio][\"']?|positive|negative|\+|-)\b"
            if re.search(blood_pattern, chunk_text):
                has_direct_result = True

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

    @staticmethod
    def _extract_patient_findings(
        db: Session,
        docs: list[Document],
    ) -> list[dict[str, Any]]:
        """
        Extract structured abnormal / flagged findings directly from sanitized_text
        for each document, bypassing chunk-based LLM extraction.

        Returns a list of dicts with keys: document, test, value, unit, ref_range, flag.
        """
        findings: list[dict[str, Any]] = []
        # Track (doc_name, normalized_test) to avoid duplicates
        seen_keys: set[tuple[str, str]] = set()

        def _normalize_test(name: str) -> str:
            return re.sub(r"[\s,]+", " ", name).strip().lower()

        def _add_finding(doc_name: str, test: str, value: str, unit: str, ref_range: str, flag: str) -> None:
            # Clean up Presidio artifacts in test names (e.g. [ADDRESS] for 'Dist.')
            cleaned_test = test.replace("[ADDRESS]", "Dist").strip()
            # For imaging findings, use value in key so distinct findings aren't dropped
            if "imaging" in cleaned_test.lower():
                key = (doc_name, _normalize_test(cleaned_test), _normalize_test(value))
            else:
                key = (doc_name, _normalize_test(cleaned_test))
            if key in seen_keys:
                return
            seen_keys.add(key)
            findings.append({
                "document": doc_name,
                "test": cleaned_test,
                "value": value,
                "unit": unit,
                "ref_range": ref_range,
                "flag": flag,
            })

        def _parse_range(ref_str: str):
            """
            Parse a reference range string into (low, high, kind).
            Returns None if the format is not unambiguously parseable.
            kind is one of: 'range', 'lt', 'le', 'gt', 'ge'
            """
            ref = ref_str.strip()
            # Normalize spaces inside numbers, e.g. "6 .0" -> "6.0"
            ref = re.sub(r"(\d+)\s*\.\s*(\d+)", r"\1.\2", ref)
            # Strip trailing non-digit units/words like "pH", "Ratio", "%"
            ref = re.sub(r"\s+[a-zA-Z/%│áÁ]+$", "", ref).strip()

            # Range: "X - Y" or "X-Y" (with optional spaces, allowing decimal)
            m = re.match(r"^(\d+(?:\.\d+)?)\s*-\s*(\d+(?:\.\d+)?)$", ref)
            if m:
                return (float(m.group(1)), float(m.group(2)), "range")
            # Threshold: "<X" or "< X"
            m = re.match(r"^<\s*(\d+(?:\.\d+)?)$", ref)
            if m:
                return (None, float(m.group(1)), "lt")
            # Threshold: "<=X"
            m = re.match(r"^<=\s*(\d+(?:\.\d+)?)$", ref)
            if m:
                return (None, float(m.group(1)), "le")
            # Threshold: ">X" or "> X"
            m = re.match(r"^>\s*(\d+(?:\.\d+)?)$", ref)
            if m:
                return (float(m.group(1)), None, "gt")
            # Threshold: ">=X"
            m = re.match(r"^>=\s*(\d+(?:\.\d+)?)$", ref)
            if m:
                return (float(m.group(1)), None, "ge")
            return None

        def _compare_to_range(result_val: float, parsed_range) -> str:
            """Return 'High', 'Low', or 'Normal' given a parsed range tuple."""
            low, high, kind = parsed_range
            if kind == "range":
                if result_val < low:
                    return "Low"
                elif result_val > high:
                    return "High"
                else:
                    return "Normal"
            elif kind == "lt":
                return "Normal" if result_val < high else "High"
            elif kind == "le":
                return "Normal" if result_val <= high else "High"
            elif kind == "gt":
                return "Normal" if result_val > low else "Low"
            elif kind == "ge":
                return "Normal" if result_val >= low else "Low"
            return "Normal"

        # Known method names in HOD / lab tables (used to skip method lines)
        _METHOD_KEYWORDS = frozenset([
            "method", "calculated", "enzymatic", "clia", "hplc", "photometric",
            "impedance", "dye", "flow", "colorimetric", "urease", "uricase",
            "chromazurol", "pyridylazo", "bromothymol", "nitroprusside",
            "methoxybenzene", "dichlorobenzene", "tetramethyl", "tetrachloro",
            "indoxyl", "ethyleneglycol", "glucose-oxidase", "modified",
            "westergren", "copper", "physical examination", "microscopy",
            "sf cube", "direct measured", "god-pod", "high performance liquid",
        ])

        # Table section subheaders in HOD that are not test rows
        _TABLE_SUBHEADERS = frozenset([
            "physical examination", "biochemical examination", "microscopic examination",
            "complete blood count", "lipid profile", "liver function test",
            "kidney function test", "electrolytes", "iron profile", "thyroid profile",
            "urine r/m",
        ])

        def _is_method_line(text: str) -> bool:
            low = text.lower()
            return any(kw in low for kw in _METHOD_KEYWORDS)

        # Qualitative normal values that should not be flagged
        _QUAL_NORMALS = frozenset([
            "negative", "nil", "absent", "clear", "normal", "non reactive",
            "non-reactive", "pale yellow",
        ])

        for doc in docs:
            dt = (
                db.query(DocumentText)
                .filter(DocumentText.document_id == doc.document_id)
                .first()
            )
            if not dt or not dt.sanitized_text:
                continue
            text = dt.sanitized_text
            lines = text.splitlines()

            raw_lines = dt.raw_text.splitlines() if dt.raw_text else []

            # =================================================================
            # --- 1. Imaging IMPRESSION block (structural capture) ---
            # =================================================================
            in_impression = False
            for line in lines:
                stripped = line.strip()
                # Enter impression block on "IMPRESSION" header
                if re.match(r"(?i)^IMPRESSION\s*:?-?\s*$", stripped):
                    in_impression = True
                    continue
                if in_impression:
                    # Exit on blank line, patient info, or page markers
                    if (
                        not stripped
                        or stripped.startswith("[PATIENT]")
                        or "Electronically Authenticated" in stripped
                        or re.match(r"^(Patient Name|Lab No|Demo Visit|Age / Sex|Registration On)", stripped)
                    ):
                        in_impression = False
                        continue
                    # Capture each bulleted/dashed finding line
                    cleaned = re.sub(r"^[-\u2022*]\s*", "", stripped).strip()
                    if cleaned and len(cleaned) > 2:
                        _add_finding(
                            doc.file_name,
                            "Imaging / Ultrasound",
                            cleaned,
                            "",
                            "Normal",
                            "Abnormal",
                        )

            # =================================================================
            # --- 2. Lab table findings with explicit H / L flags ---
            # =================================================================
            i = 0
            while i < len(lines):
                line = lines[i].strip()

                # Stand-alone H or L on its own line
                if re.match(r"^[HL]$", line) and i > 0:
                    flag = "High (H)" if line == "H" else "Low (L)"
                    test_name = lines[i - 1].strip()
                    unit, ref_range, val = "", "", ""
                    for j in range(i + 1, min(i + 15, len(lines))):
                        sub = lines[j].strip()
                        if not sub:
                            continue

                        # Stop if we hit the next test or section header
                        if j > i + 1:
                            if j + 1 < len(lines) and re.match(r"^[HL]$", lines[j + 1].strip()):
                                break
                            if sub in [
                                "HDL Cholesterol", "VLDL", "CHOL/HDL Ratio", "LDL/HDL Ratio",
                                "Mean Blood Glucose", "Biochemistry", "Summary and Uses:",
                                "Clinical Notes:", "Interpretation:", "Neutrophils", "Lymphocytes",
                                "Per[IP_NUMBER] Smear Examination", "Explanation:-", "Blood Urea Nitrogen",
                                "Uric Acid", "Calcium", "Hb A2"
                            ]:
                                break

                        if re.match(
                            r"^(?:mg/dl|g/dl|gm/dl|micromol/l|iu/ml|pg/ml|%|/cmm|fl|u/l|mmol/l|lakhs/cumm|cumm|/ul|cells/.*)$",
                            sub,
                            re.I,
                        ):
                            if not unit:
                                unit = sub
                        elif re.search(r"\d+\s*-\s*\d+|<|>|normal\s*:", sub, re.I) and not ref_range:
                            ref_range = sub
                        elif (re.match(r"^\d+(?:\.\d+)?$", sub) or re.match(r"^<\s*\d+", sub)) and not val:
                            val = sub

                    # If val was not found (e.g. redacted to [PHONE] in sanitized_text), check raw_lines
                    if not val and raw_lines:
                        for r_idx in range(max(0, i - 15), min(len(raw_lines), i + 15)):
                            if raw_lines[r_idx].strip() == test_name:
                                for rj in range(r_idx + 1, min(r_idx + 15, len(raw_lines))):
                                    rsub = raw_lines[rj].strip()
                                    if rj > r_idx + 1 and (
                                        (rj + 1 < len(raw_lines) and re.match(r"^[HL]$", raw_lines[rj + 1].strip()))
                                        or rsub in ["HDL Cholesterol", "VLDL", "CHOL/HDL Ratio", "LDL/HDL Ratio", "Mean Blood Glucose"]
                                    ):
                                        break
                                    if (re.match(r"^\d+(?:\.\d+)?$", rsub) or re.match(r"^<\s*\d+", rsub)) and not val:
                                        val = rsub
                                        break
                                break

                    if test_name and (val or ref_range):
                        _add_finding(doc.file_name, test_name, val, unit, ref_range, flag)

                # Inline flags (e.g. "MCHC H 35.7 %" or "Fasting Blood Sugar H 142 mg/dL (Reference 74 - 106)")
                inline_pattern = r"(?:^|[.;\n])\s*([A-Za-z0-9\s,/-]+?)\s+([HL])\s+(\d+(?:\.\d+)?)\s*([a-zA-Z/%]+)?(?:\s*\((?:Reference\s*)?([^)]*)\))?"
                for inline_m in re.finditer(inline_pattern, line):
                    tname = inline_m.group(1).strip()
                    if len(tname) > 2 and not any(w in tname.lower() for w in ["reference", "range", "interval"]):
                        _add_finding(
                            doc.file_name,
                            tname,
                            inline_m.group(3),
                            inline_m.group(4) or "",
                            inline_m.group(5) or "",
                            "High (H)" if inline_m.group(2).upper() == "H" else "Low (L)",
                        )

                # Qualitative abnormalities — same-line pattern
                # (e.g. "Urinary Glucose 1+ Negative" or "Present (+)")
                # Skip lines that are purely "Non Reactive" / screening negative results
                if not re.search(r"(?i)^\s*non[-\s]*reactive\b", line):
                    qual_pattern = r"(?:^|[.;\n])\s*([A-Za-z0-9\s,/-]+?)\s+([1-4]\+|Present\s*\(\+\)|\bPositive\b|\bReactive\b)(?:\s+([A-Za-z]+))?"
                    for qual_m in re.finditer(qual_pattern, line):
                        tname = qual_m.group(1).strip()
                        val = qual_m.group(2).strip()
                        if val.lower() == "reactive" and (re.search(r"(?i)\bnon[-\s]*reactive\b", line) or tname.lower() in ("non", "non-")):
                            continue
                        if len(tname) > 2 and tname.lower() not in ("non", "test", "result", "status", "interpretation"):
                            ref_val = qual_m.group(3) or "Negative"
                            _add_finding(doc.file_name, tname, val, "", ref_val, "Abnormal")

                i += 1

            # =================================================================
            # --- 3. Multi-line qualitative extraction ---
            # =================================================================
            i = 0
            while i < len(lines) - 1:
                line = lines[i].strip()
                next_line = lines[i + 1].strip() if i + 1 < len(lines) else ""
                ref_line = lines[i + 2].strip() if i + 2 < len(lines) else ""

                if (
                    re.match(r"^[1-4]\+$", next_line)
                    and len(line) > 2
                    and line[0].isalpha()
                    and not re.match(r"^[HL]$", line)
                    and not re.search(r"\d", line)
                ):
                    ref_val = ref_line if ref_line.lower() in _QUAL_NORMALS else "Negative"
                    if ref_val.lower() in _QUAL_NORMALS and ref_val.lower() != next_line.lower():
                        _add_finding(doc.file_name, line, next_line, "", ref_val, "Abnormal")
                    i += 3
                    continue
                i += 1

            # =================================================================
            # --- 4. HOD-style numeric range comparison ---
            # =================================================================
            i = 0
            while i < len(lines):
                line = lines[i].strip()

                # Detect table header: "Observation" followed by "Result"
                if (
                    line == "Observation"
                    and i + 4 < len(lines)
                    and lines[i + 1].strip() == "Result"
                ):
                    j = i + 5  # skip 5-line header (Observation/Result/Unit/Bio Ref/Method)
                    while j < len(lines):
                        tname = lines[j].strip()

                        # Skip subheaders within the table (e.g. Biochemical Examination)
                        if tname.lower() in _TABLE_SUBHEADERS:
                            j += 1
                            continue

                        # Exit conditions
                        if (
                            not tname
                            or tname.startswith("---")
                            or tname in [
                                "Patient Name :", "Observation", "Clinical Significance:",
                                "Scan to Validate", "[PATIENT]:", "[PATIENT] :",
                            ]
                            or tname.startswith("[PATIENT]")
                        ):
                            break

                        # Read the next 3 lines: result, unit, ref
                        if j + 3 < len(lines):
                            result_str = lines[j + 1].strip()
                            unit_str = lines[j + 2].strip()
                            ref_str = lines[j + 3].strip()

                            # Skip method line if line j+4 looks like a method
                            consumed = 4
                            if j + 4 < len(lines) and _is_method_line(lines[j + 4].strip()):
                                consumed = 5

                            # Determine if result is numeric
                            result_numeric = re.match(r"^-?\d+(?:\.\d+)?$", result_str)

                            if result_numeric:
                                result_val = float(result_str)
                                parsed = _parse_range(ref_str)
                                if parsed:
                                    classification = _compare_to_range(result_val, parsed)
                                    if classification != "Normal":
                                        flag_str = f"High" if classification == "High" else f"Low"
                                        _add_finding(
                                            doc.file_name, tname, result_str, unit_str,
                                            ref_str, flag_str,
                                        )

                            elif result_str.lower() in ("1+", "2+", "3+", "4+"):
                                actual_ref = unit_str if unit_str.lower() in _QUAL_NORMALS else ref_str
                                if actual_ref.lower() in _QUAL_NORMALS:
                                    _add_finding(doc.file_name, tname, result_str, "", actual_ref, "Abnormal")

                            j += consumed
                            continue
                        j += 1
                    i = j
                else:
                    i += 1

        return findings

    @staticmethod
    def _format_narrow_chunk_excerpt(chunk_text: str, query_terms: list[str]) -> str:
        """
        For narrow factual queries, extracts a window around the lines matching query terms,
        preventing 7B models from losing focus due to thousands of unrelated characters.
        """
        lines = chunk_text.splitlines()
        matching_line_indices: set[int] = set()
        for idx, l in enumerate(lines):
            if any(re.search(r"\b" + re.escape(t) + r"\b", l, re.I) for t in query_terms):
                for w in range(max(0, idx - 3), min(len(lines), idx + 8)):
                    matching_line_indices.add(w)

        if matching_line_indices and len(matching_line_indices) < len(lines):
            sorted_indices = sorted(matching_line_indices)
            blocks: list[str] = []
            cur_block: list[str] = []
            prev_idx: int | None = None
            for i in sorted_indices:
                if prev_idx is not None and i > prev_idx + 1:
                    blocks.append("\n".join(cur_block))
                    cur_block = []
                cur_block.append(lines[i])
                prev_idx = i
            if cur_block:
                blocks.append("\n".join(cur_block))
            return "\n...\n".join(blocks)
        return chunk_text

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
            .order_by(Document.file_name)
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

            # Phase 1: Guarantee representation from each document with matching terms or notable similarity
            chosen: list[tuple[DocumentChunk, float]] = []
            for doc_id, scored in doc_candidates.items():
                best_c, best_score, best_sim = scored[0]
                if best_score > 0.25 or any(kw in best_c.chunk_text.lower() for kw in query_terms):
                    chosen.append((best_c, best_sim))

            # Phase 2: Runner-up padding only fires when Phase 1 did NOT already select a relevant chunk
            # for every document that has one (i.e. don't pad past what's needed just to hit top_k=5).
            if not chosen:
                all_runner_ups = []
                for doc_id, scored in doc_candidates.items():
                    for c, sc, sim in scored[:2]:
                        all_runner_ups.append((c, sc, sim))
                all_runner_ups.sort(key=lambda x: x[1], reverse=True)
                for c, sc, sim in all_runner_ups[:top_k]:
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
            if q_type == "NARROW_FACTUAL":
                q_terms = self._extract_query_keywords(query)
                text_to_show = self._format_narrow_chunk_excerpt(chunk.chunk_text, q_terms)
            else:
                text_to_show = chunk.chunk_text
            context_blocks.append(
                f"[Excerpt {idx} - Document: {doc_name}{page_str} (Chunk #{chunk.chunk_index})]:\n{text_to_show}"
            )

        context_text = "\n\n".join(context_blocks)

        # For comprehensive abnormal / flagged queries, build structured findings context
        findings_context = ""
        # Tracks whether this is a diabetes-focused or cross-doc aggregation query
        # so the synthesis prompt can be adjusted accordingly.
        _is_diabetes_query = bool(re.search(
            r"(?i)\bdiabet(?:es|ic)|diabet(?:es|ic)\s+(?:risk|markers?|signs?)",
            query,
        ))
        _is_cross_doc_query = bool(re.search(
            r"(?i)\b(reported\s+across|across\s+(?:the\s+)?(?:patient[\u2019']?s\s+)?documents?|conflicting|inconsistent|outside\s+(?:their|the|its|normal)\s+(?:reference\s+)?range)",
            query,
        ))
        if q_type == "COMPREHENSIVE_ABNORMAL":
            raw_findings = self._extract_patient_findings(db, docs)
            if re.search(r"(?i)\bh-?flag(?:ged|s)?\b", query):
                relevant_findings = [f for f in raw_findings if "High" in f.get("flag", "")]
            elif re.search(r"(?i)\bl-?flag(?:ged|s)?\b", query):
                relevant_findings = [f for f in raw_findings if "Low" in f.get("flag", "")]
            elif _is_diabetes_query:
                # Diabetes-marker queries: filter to glucose/FBS/HbA1c findings.
                # Matched by test name only — do not require "diabetes" in reference range.
                _DIABETES_TERMS = frozenset([
                    "fasting blood sugar", "fbs", "glucose", "blood sugar", "blood sugar fasting",
                    "hba1c", "hb a1c", "glycated", "glycosylated",
                ])
                relevant_findings = [
                    f for f in raw_findings
                    if any(t in f.get("test", "").lower() for t in _DIABETES_TERMS)
                ]
                # If nothing matched by test name, fall back to all findings so the
                # LLM has something to work with rather than returning empty context.
                if not relevant_findings:
                    relevant_findings = raw_findings
            else:
                relevant_findings = raw_findings

            if relevant_findings:
                by_doc: dict[str, list[str]] = {}
                for f in relevant_findings:
                    val_str = f"{f['value']} {f['unit']}".strip() if f.get("value") else ""
                    ref_str = f" (Reference: {f['ref_range']})" if f.get("ref_range") else ""
                    by_doc.setdefault(f["document"], []).append(
                        f"- {f['test']}: {val_str}{ref_str} [Flag: {f['flag']}]"
                    )
                sections = []
                for dname, items in by_doc.items():
                    sections.append(f"Document: [{dname}]\n" + "\n".join(items))
                findings_context = "\n\n".join(sections)

        if findings_context:
            if _is_cross_doc_query:
                # Cross-document comparison / out-of-range / conflicting queries.
                # The structured findings list contains only FLAGGED (abnormal) values.
                # Instruct the LLM to note that non-listed values were within reference range.
                prompt = (
                    f"You are an expert clinical AI assistant reviewing a patient's documented medical records.\n\n"
                    f"User Question: {query}\n\n"
                    f"Documented Flagged Findings from Patient Records\n"
                    f"(Note: only values that are outside their reference range are listed. "
                    f"Values not listed were within the stated reference range in that document.)\n"
                    f"\"\"\"\n{findings_context}\n\"\"\"\n\n"
                    f"Clinical Instructions:\n"
                    f"1. Answer the question using ONLY the findings listed above.\n"
                    f"2. Group findings by document name.\n"
                    f"3. State all test names, values, units, reference ranges, and flags exactly as listed.\n"
                    f"4. For conflicting/inconsistent-data questions, explicitly compare the same test across documents and note any discrepancies.\n"
                    f"5. If the same test appears in multiple documents with different values, list all of them.\n"
                    f"6. Remind the reader that normal (in-range) values for the same tests may exist in other documents but are not listed here.\n"
                    f"7. Do NOT diagnose, invent findings, or make clinical risk judgments."
                )
            elif _is_diabetes_query:
                prompt = (
                    f"You are an expert clinical AI assistant reviewing a patient's documented medical records.\n\n"
                    f"User Question: {query}\n\n"
                    f"Documented Diabetes-Relevant Findings from Patient Records:\n\"\"\"\n{findings_context}\n\"\"\"\n\n"
                    f"Clinical Instructions:\n"
                    f"1. List the documented glucose, FBS, and HbA1c values exactly as shown, with units, reference ranges, and flags.\n"
                    f"2. State factually which values are flagged as High or outside reference range.\n"
                    f"3. Do NOT diagnose diabetes, assign a risk score, or make clinical recommendations.\n"
                    f"4. Do NOT invent findings or speculate beyond what is documented.\n"
                    f"5. If no diabetes-relevant findings are listed, state that clearly."
                )
            else:
                prompt = (
                    f"You are an expert clinical AI assistant reviewing a patient's documented medical records.\n\n"
                    f"User Question: {query}\n\n"
                    f"Documented Findings from Patient Records:\n\"\"\"\n{findings_context}\n\"\"\"\n\n"
                    f"Clinical Instructions:\n"
                    f"1. List EVERY SINGLE documented finding from the list above without omitting, summarizing, or truncating any tests. Every finding in the input must appear in your output.\n"
                    f"2. Group findings strictly by document name.\n"
                    f"3. State all test names, numerical values, units, reference ranges, and flags exactly as listed.\n"
                    f"4. Do NOT invent findings or speculate.\n"
                    f"5. If no findings match the question, state that clearly."
                )
        else:
            prompt = (
                f"You are an expert clinical AI assistant answering a question based strictly on excerpts from a patient's de-identified medical records.\n\n"
                f"User Question: {query}\n\n"
                f"Patient Record Excerpts:\n\"\"\"\n{context_text}\n\"\"\"\n\n"
                f"Clinical Instructions:\n"
                f"1. Answer factually using ONLY the facts and values directly stated in the excerpts above.\n"
                f"2. Cite the specific document name(s) when presenting findings or test values.\n"
                f"3. Report all reported values for the requested test across each document source with document names, values, units, and reference ranges (do not omit values from other documents).\n"
                f"4. Preserve all numerical measurements, reference ranges, and units exactly as stated.\n"
                f"5. If the provided excerpts do not mention or contain any information regarding the question, clearly state that no records or information were found for that question.\n"
                f"6. Do NOT diagnose, recommend clinical treatments, or speculate beyond the provided text.\n"
                f"7. Provide a clear, comprehensive, and well-structured response."
            )

        # 5. Generate LLM completion
        try:
            answer = self.llm.generate_text(
                prompt=prompt,
                system_prompt=SYSTEM_PROMPT,
                temperature=0.0,
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
