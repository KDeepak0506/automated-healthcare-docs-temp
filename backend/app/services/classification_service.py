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

    def build_classification_prompt(self, sanitized_text: str) -> str:
        types_formatted = "\n".join(f"- {t}" for t in ALLOWED_DOCUMENT_TYPES)
        return (
            f"Analyze the following de-identified clinical document text and determine its document type.\n\n"
            f"Supported document types:\n"
            f"{types_formatted}\n\n"
            f"Rules:\n"
            f"1. Classify ONLY based on the provided document content.\n"
            f"2. Do not use outside knowledge or assumptions.\n"
            f"3. If the document does not clearly fit into Laboratory Report, Prescription, Discharge Summary, "
            f"Radiology Report, or Insurance Form, choose 'Other'.\n"
            f"4. Provide a classification confidence score as a float between 0.0 and 1.0.\n"
            f"5. Do NOT make medical diagnoses or clinical treatment recommendations.\n"
            f"6. Return ONLY a JSON object matching this schema:\n"
            f'{{\n  "document_type": "<one of the supported types>",\n  "classification_confidence": <float between 0.0 and 1.0>\n}}\n\n'
            f"Document Text:\n\"\"\"\n{sanitized_text}\n\"\"\""
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

        # 4. Oversized text guard
        sanitized_text = doc_text.sanitized_text.strip()
        is_truncated = False
        if len(sanitized_text) > settings.max_sanitized_text_chars:
            logger.warning(
                f"Sanitized text for document {document.document_id} exceeded "
                f"{settings.max_sanitized_text_chars} chars and was truncated for classification."
            )
            sanitized_text = sanitized_text[: settings.max_sanitized_text_chars]
            is_truncated = True

        # 5. Build prompt and call LLM
        prompt = self.build_classification_prompt(sanitized_text)

        try:
            result = self.llm.generate_structured(
                prompt=prompt,
                response_schema=ClassificationResult,
                system_prompt=SYSTEM_PROMPT,
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
        except LLMResponseParsingError as exc:
            logger.error(f"Classification parsing failed: {exc}")
            raise HTTPException(
                status_code=status.HTTP_502_BAD_GATEWAY,
                detail="Invalid response received from classification model.",
            ) from exc
        except LLMServiceError as exc:
            logger.error(f"Classification provider error: {exc}")
            raise HTTPException(
                status_code=status.HTTP_502_BAD_GATEWAY,
                detail="Failed to classify document with AI service.",
            ) from exc

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
