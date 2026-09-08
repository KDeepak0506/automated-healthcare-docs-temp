import logging
from uuid import UUID
from fastapi import HTTPException, status
from sqlalchemy.orm import Session

from app.core.config import settings
from app.models.document import Document
from app.models.document_text import DocumentText
from app.schemas.summary import (
    SummaryResponse,
    SummaryResult,
)
from app.services.llm_service import (
    LLMConfigurationError,
    LLMResponseParsingError,
    LLMService,
    LLMServiceError,
    LLMTimeoutError,
    llm_service as default_llm_service,
)

logger = logging.getLogger(__name__)

# Hard cap on sanitized text characters sent to Groq.
# 50,000 chars (the settings default) exceeds Groq free-tier per-request
# token limits for the gpt-oss-20b model, producing HTTP 413 errors.
# 6,000 chars (~1,500 tokens) leaves ample headroom for the system/user
# prompt overhead while still capturing the most clinically relevant content.
_MAX_SUMMARY_INPUT_CHARS: int = 6000

SYSTEM_PROMPT = (
    "You are an expert healthcare documentation summarizer working within a strict clinical privacy boundary. "
    "Summarize the supplied de-identified document factually and accurately. "
    "Preserve all numerical values, test results, and units exactly without alteration or rounding. "
    "Never diagnose, invent facts, or recommend clinical treatments."
)


class SummaryService:
    """
    Service for M5 Healthcare Document Summarization.
    STRICTLY consumes sanitized_text ONLY (never raw OCR).
    """

    def __init__(self, llm: LLMService | None = None) -> None:
        self.llm = llm or default_llm_service

    def build_summary_prompt(self, sanitized_text: str) -> str:
        return (
            f"Generate a concise, factual clinical summary and extract key findings from the de-identified text below.\n\n"
            f"Strict Instructions:\n"
            f"1. Factual Grounding: Base all summary statements strictly on the provided text. Do not speculate or invent facts.\n"
            f"2. Numerical Faithfulness: Preserve exact numerical measurements, lab values, percentages, and units "
            f"(e.g., '8.2%', '1.41 mg/dL', '139 mmol/L', '140/90 mmHg', '500 mg') without altering, rounding, or converting.\n"
            f"3. Key Findings: Extract distinct key clinical findings, stated diagnoses, vital signs, lab results, and medications directly from the text.\n"
            f"4. Clinical Safety: Do NOT provide medical diagnoses, treatment recommendations, or clinical advice.\n"
            f"5. Return ONLY a JSON object matching this schema:\n"
            f'{{\n  "summary": "<concise paragraph summarizing the clinical document>",\n  "key_findings": ["<key finding 1 with units/values>", "<key finding 2>", ...]\n}}\n\n'
            f"Document Text:\n\"\"\"\n{sanitized_text}\n\"\"\""
        )

    def summarize_document(
        self,
        db: Session,
        document: Document,
    ) -> SummaryResponse:
        """
        Generate a factual summary and key findings from the document's sanitized OCR text.
        Returns cached result if already summarized.
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

        # 3. Idempotency / cache check
        if document.summary is not None and document.key_findings is not None:
            logger.info(f"Returning cached summary for document {document.document_id}")
            return SummaryResponse(
                document_id=document.document_id,
                summary=document.summary,
                key_findings=document.key_findings,
                cached=True,
                truncated=False,
            )

        # 4. Oversized text guard
        # Use _MAX_SUMMARY_INPUT_CHARS (not the larger settings value) to stay
        # within Groq free-tier per-request limits and avoid HTTP 413 errors.
        sanitized_text = doc_text.sanitized_text.strip()
        original_char_count = len(sanitized_text)
        is_truncated = False
        if original_char_count > _MAX_SUMMARY_INPUT_CHARS:
            sanitized_text = sanitized_text[:_MAX_SUMMARY_INPUT_CHARS]
            is_truncated = True
            logger.warning(
                f"document_id={document.document_id} sanitized text truncated for summarization: "
                f"original={original_char_count} chars, bounded={_MAX_SUMMARY_INPUT_CHARS} chars, "
                f"truncated=True"
            )
        else:
            logger.info(
                f"document_id={document.document_id} sanitized text within limit: "
                f"{original_char_count} chars, truncated=False"
            )

        # 5. Build prompt and call LLM
        prompt = self.build_summary_prompt(sanitized_text)

        try:
            result = self.llm.generate_structured(
                prompt=prompt,
                response_schema=SummaryResult,
                system_prompt=SYSTEM_PROMPT,
            )
        except LLMConfigurationError as exc:
            logger.error(f"Summarization failed: {exc}")
            raise HTTPException(
                status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
                detail="Document summarization service is not configured (missing API key).",
            ) from exc
        except LLMTimeoutError as exc:
            logger.error(f"Summarization timed out: {exc}")
            raise HTTPException(
                status_code=status.HTTP_504_GATEWAY_TIMEOUT,
                detail="Document summarization request timed out.",
            ) from exc
        except LLMResponseParsingError as exc:
            logger.error(f"Summarization parsing failed: {exc}")
            raise HTTPException(
                status_code=status.HTTP_502_BAD_GATEWAY,
                detail="Invalid response received from summarization model.",
            ) from exc
        except LLMServiceError as exc:
            logger.error(f"Summarization provider error: {exc}")
            raise HTTPException(
                status_code=status.HTTP_502_BAD_GATEWAY,
                detail="Failed to summarize document with AI service.",
            ) from exc

        # 6. Persist summary to database
        document.summary = result.summary
        document.key_findings = result.key_findings
        db.commit()
        db.refresh(document)

        logger.info(f"Successfully summarized document {document.document_id}")

        return SummaryResponse(
            document_id=document.document_id,
            summary=result.summary,
            key_findings=result.key_findings,
            cached=False,
            truncated=is_truncated,
        )


# Singleton instance for service usage
summary_service = SummaryService()
