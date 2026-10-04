import logging
from uuid import UUID
from fastapi import HTTPException, status
from sqlalchemy.orm import Session

from app.core.config import settings
from app.models.document import Document
from app.models.document_text import DocumentText
from app.schemas.classification import (
    ALLOWED_DOCUMENT_TYPES,
    ClassificationResponse,
    ClassificationResult,
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

MAX_CLASSIFICATION_CHARS = 4000

SYSTEM_PROMPT = (
    "You are an expert medical document classifier working within a strict clinical privacy boundary. "
    "Classify the supplied de-identified document content accurately. "
    "Never invent medical facts, diagnoses, or clinical recommendations."
)


class ClassificationService:
    """
    Service for M3 Document Type Identification.
    STRICTLY consumes sanitized_text ONLY (never raw OCR).
    """

    def __init__(self, llm: LLMService | None = None) -> None:
        self.llm = llm or default_llm_service

    def build_classification_prompt(
        self,
        sanitized_text: str,
        validation_error: str | None = None,
    ) -> str:
        types_formatted = "\n".join(f"- {t}" for t in ALLOWED_DOCUMENT_TYPES)
        retry_section = ""
        if validation_error:
            retry_section = (
                f"\nIMPORTANT: Your previous output failed schema validation with error:\n"
                f"{validation_error}\n"
                f"You must strictly correct this and output ONLY the requested JSON structure.\n\n"
            )

        return (
            f"Analyze the following de-identified clinical document text and determine its document type.\n\n"
            f"Supported document types:\n"
            f"{types_formatted}\n\n"
            f"Rules:\n"
            f"1. Classify ONLY based on the text inside the <document> tags.\n"
            f"2. Do not use outside knowledge or make assumptions.\n"
            f"3. If the document does not clearly fit into Laboratory Report, Prescription, Discharge Summary, "
            f"Radiology Report, or Insurance Form, choose 'Other'.\n"
            f"4. Provide a classification confidence score as a float between 0.0 and 1.0.\n"
            f"5. Do NOT make medical diagnoses or clinical treatment recommendations.\n"
            f"{retry_section}"
            f"<document>\n"
            f"{sanitized_text}\n"
            f"</document>\n\n"
            f"CRITICAL: Return ONLY a JSON object with exactly these two keys:\n"
            f'- "document_type": Must be one of: {", ".join(ALLOWED_DOCUMENT_TYPES)}\n'
            f'- "classification_confidence": A float between 0.0 and 1.0\n\n'
            f'Do not include any other keys (such as "Patient", "Diagnosis", etc.). '
            f'Return ONLY the JSON object with "document_type" and "classification_confidence".'
        )

    def classify_document(
        self,
        db: Session,
        document: Document,
    ) -> ClassificationResponse:
        """
        Classify document type from its sanitized OCR text.
        Returns cached result if already classified.
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
        if document.document_type is not None and document.classification_confidence is not None:
            logger.info(f"Returning cached classification for document {document.document_id}")
            return ClassificationResponse(
                document_id=document.document_id,
                document_type=document.document_type,
                classification_confidence=float(document.classification_confidence),
                cached=True,
                truncated=False,
            )

        # 4. Truncate classification input to first ~4000 chars of sanitized text
        sanitized_text = doc_text.sanitized_text.strip()
        is_truncated = False
        if len(sanitized_text) > MAX_CLASSIFICATION_CHARS:
            logger.warning(
                f"Sanitized text for document {document.document_id} exceeded "
                f"{MAX_CLASSIFICATION_CHARS} chars ({len(sanitized_text)} chars) and was truncated for classification."
            )
            sanitized_text = sanitized_text[:MAX_CLASSIFICATION_CHARS]
            is_truncated = True

        # 5. Build prompt and call LLM (retry once on schema failure)
        result: ClassificationResult | None = None
        current_prompt = self.build_classification_prompt(sanitized_text)

        for attempt in (1, 2):
            try:
                result = self.llm.generate_structured(
                    prompt=current_prompt,
                    response_schema=ClassificationResult,
                    system_prompt=SYSTEM_PROMPT,
                )
                break
            except LLMResponseParsingError as parse_err:
                raw_out = getattr(parse_err, "raw_response", None) or str(parse_err)
                raw_truncated = raw_out[:300]
                logger.warning(
                    f"Classification schema validation failed on attempt {attempt} for document {document.document_id}. "
                    f"Validation error: {parse_err}. Raw LLM output (truncated): {raw_truncated}"
                )
                if attempt == 1:
                    # Retry once including the validation error in prompt
                    current_prompt = self.build_classification_prompt(
                        sanitized_text,
                        validation_error=str(parse_err),
                    )
                else:
                    # Retry also failed: save fallback and continue without raising 502
                    logger.warning(
                        f"Classification schema validation retry failed for document {document.document_id}. "
                        "Fallback applied: document_type='unknown', classification_confidence=0.0"
                    )
                    document.document_type = "unknown"
                    document.classification_confidence = 0.0
                    db.commit()
                    db.refresh(document)
                    return ClassificationResponse(
                        document_id=document.document_id,
                        document_type="unknown",
                        classification_confidence=0.0,
                        cached=False,
                        truncated=is_truncated,
                    )
            except LLMConfigurationError as exc:
                logger.error(f"Classification failed: {exc}")
                raise HTTPException(
                    status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
                    detail="Document classification service is not configured (missing API key).",
                ) from exc
            except LLMTimeoutError as exc:
                logger.error(f"Classification timed out: {exc}")
                raise HTTPException(
                    status_code=status.HTTP_504_GATEWAY_TIMEOUT,
                    detail="Document classification request timed out.",
                ) from exc
            except LLMServiceError as exc:
                logger.error(f"Classification provider error: {exc}")
                raise HTTPException(
                    status_code=status.HTTP_502_BAD_GATEWAY,
                    detail="Failed to classify document with AI service.",
                ) from exc

        if result is None:
            document.document_type = "unknown"
            document.classification_confidence = 0.0
            db.commit()
            db.refresh(document)
            return ClassificationResponse(
                document_id=document.document_id,
                document_type="unknown",
                classification_confidence=0.0,
                cached=False,
                truncated=is_truncated,
            )

        # 6. Persist result to database
        document.document_type = result.document_type
        document.classification_confidence = result.classification_confidence
        db.commit()
        db.refresh(document)

        logger.info(
            f"Successfully classified document {document.document_id} as "
            f"'{result.document_type}' (conf={result.classification_confidence:.2f})"
        )

        return ClassificationResponse(
            document_id=document.document_id,
            document_type=result.document_type,
            classification_confidence=result.classification_confidence,
            cached=False,
            truncated=is_truncated,
        )


# Singleton instance for service usage
classification_service = ClassificationService()
