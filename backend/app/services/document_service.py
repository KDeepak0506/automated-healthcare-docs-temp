import logging
from datetime import datetime, timezone
from pathlib import Path
from uuid import UUID

from fastapi import BackgroundTasks, HTTPException, UploadFile
from sqlalchemy.orm import Session

from app.db.session import SessionLocal
from app.models.document import Document
from app.models.document_text import DocumentText
from app.models.document_entity import DocumentEntity
from app.schemas.document import DocumentProcessingStatus
from app.services.ocr_service import OCRResult, extract_text
from app.services.privacy_service import privacy_service
from app.services.ner_service import clinical_ner_service

logger = logging.getLogger(__name__)

UPLOAD_DIR = Path("uploads/documents")

ALLOWED_TYPES = {
    "application/pdf",
    "image/jpeg",
    "image/png",
}


def _run_ocr_and_store(
    document_id: UUID,
    file_path: Path,
    file_type: str,
) -> None:
    """Internal background task to run OCR, Privacy, and M4 Clinical NER using a dedicated DB session."""
    db = SessionLocal()
    try:
        ocr_result = extract_text(file_path, file_type)
        doc_text = store_ocr_result(db, document_id, ocr_result)

        # M4 Clinical NER Orchestration
        # Check that privacy_status == "completed" and sanitized_text is present before running M4
        document = db.query(Document).filter(Document.document_id == document_id).first()
        if document and document.privacy_status == "completed" and doc_text.sanitized_text:
            logger.info(f"Triggering M4 Clinical NER for document {document_id}")
            clinical_ner_service.extract_and_store_entities(
                db=db,
                document_id=document_id,
                sanitized_text=doc_text.sanitized_text,
            )
        else:
            status_str = document.privacy_status if document else "None"
            logger.warning(
                f"Skipping M4 Clinical NER for document {document_id}: "
                f"privacy_status='{status_str}'"
            )
    except Exception as exc:
        logger.error(f"Background processing failed for document {document_id}: {exc}")
        try:
            db.rollback()
            _mark_document_failed(db, document_id)
        except Exception as inner_exc:
            logger.error(f"Failed to mark document {document_id} as failed: {inner_exc}")
    finally:
        db.close()


def upload_document(
    db: Session,
    file: UploadFile,
    patient_id: UUID | None,
    uploaded_by: UUID,
    background_tasks: BackgroundTasks,
) -> Document:

    # 1. Validate file type
    if file.content_type not in ALLOWED_TYPES:
        raise HTTPException(
            status_code=400,
            detail="Unsupported file type",
        )

    # 2. Make sure upload directory exists
    UPLOAD_DIR.mkdir(parents=True, exist_ok=True)

    # 3. Create database record
    document = Document(
        patient_id=patient_id,
        uploaded_by=uploaded_by,
        file_name=file.filename,
        file_type=file.content_type,
        file_url="",
    )

    db.add(document)
    db.commit()
    db.refresh(document)

    # 4. Create file path using document ID
    file_extension = Path(file.filename).suffix
    file_path = UPLOAD_DIR / f"{document.document_id}{file_extension}"

    # 5. Save uploaded file
    try:
        with file_path.open("wb") as buffer:
            while chunk := file.file.read(1024 * 1024):
                buffer.write(chunk)

    except Exception:
        db.delete(document)
        db.commit()

        raise HTTPException(
            status_code=500,
            detail="Failed to save uploaded file",
        )

    # 6. Store file path in database and set status to Processing
    document.file_url = str(file_path)
    document.processing_status = DocumentProcessingStatus.PROCESSING.value

    db.commit()
    db.refresh(document)

    # 7. Schedule OCR as a background task
    background_tasks.add_task(
        _run_ocr_and_store,
        document_id=document.document_id,
        file_path=file_path,
        file_type=document.file_type,
    )

    return document


def _mark_document_failed(
    db: Session,
    document_id: UUID,
) -> None:
    """Set the document's processing status to Failed without raising HTTP errors."""
    document = (
        db.query(Document)
        .filter(Document.document_id == document_id)
        .first()
    )
    if document is not None:
        document.processing_status = DocumentProcessingStatus.FAILED.value
        document.privacy_status = "failed"
        db.commit()


def update_document_status(
    db: Session,
    document_id: UUID,
    status: DocumentProcessingStatus,
) -> Document:

    document = (
        db.query(Document)
        .filter(Document.document_id == document_id)
        .first()
    )

    if document is None:
        raise HTTPException(
            status_code=404,
            detail="Document not found",
        )

    document.processing_status = status.value

    if status == DocumentProcessingStatus.COMPLETED:
        document.processed_at = datetime.now(timezone.utc)

    db.commit()
    db.refresh(document)

    return document

def get_all_documents(
    db: Session,
) -> list[Document]:

    documents = (
        db.query(Document)
        .order_by(Document.uploaded_at.desc())
        .all()
    )

    return documents

def get_document_by_id(
    db: Session,
    document_id: UUID,
) -> Document:

    document = (
        db.query(Document)
        .filter(Document.document_id == document_id)
        .first()
    )

    if document is None:
        raise HTTPException(
            status_code=404,
            detail="Document not found",
        )

    return document


def store_ocr_result(
    db: Session,
    document_id: UUID,
    ocr_result: OCRResult,
) -> DocumentText:
    document = (
        db.query(Document)
        .filter(Document.document_id == document_id)
        .first()
    )

    if document is None:
        raise HTTPException(
            status_code=404,
            detail="Document not found",
        )

    doc_text = (
        db.query(DocumentText)
        .filter(DocumentText.document_id == document_id)
        .first()
    )

    if doc_text is None:
        doc_text = DocumentText(
            document_id=document_id,
            raw_text=ocr_result.raw_text,
            page_count=ocr_result.page_count,
            confidence=ocr_result.confidence,
            layout=ocr_result.layout,
            ocr_engine=ocr_result.ocr_engine,
            processing_time_ms=ocr_result.processing_time_ms,
        )
        db.add(doc_text)
    else:
        doc_text.raw_text = ocr_result.raw_text
        doc_text.page_count = ocr_result.page_count
        doc_text.confidence = ocr_result.confidence
        doc_text.layout = ocr_result.layout
        doc_text.ocr_engine = ocr_result.ocr_engine
        doc_text.processing_time_ms = ocr_result.processing_time_ms

    # Explicit branch for native vs OCR confidence score
    if ocr_result.ocr_engine == "pymupdf-native":
        document.ocr_quality_score = 1.0
    else:
        document.ocr_quality_score = ocr_result.confidence

    # Run Privacy / De-identification Layer
    document.privacy_status = "processing"
    try:
        sanitization_result = privacy_service.sanitize(ocr_result.raw_text)
        doc_text.sanitized_text = sanitization_result.sanitized_text
        doc_text.privacy_metadata = {
            "entities_detected": sanitization_result.entities_detected,
            "entity_counts": sanitization_result.entity_counts,
        }
        document.privacy_status = "completed"
        document.processing_status = DocumentProcessingStatus.COMPLETED.value
        document.processed_at = datetime.now(timezone.utc)
        logger.info(
            f"Privacy processing completed for document {document_id}: "
            f"entities_detected={sanitization_result.entities_detected}"
        )
    except Exception as exc:
        logger.error(f"Privacy processing failed for document {document_id}: {exc}")
        doc_text.sanitized_text = None
        document.privacy_status = "failed"
        document.processing_status = DocumentProcessingStatus.FAILED.value

    db.commit()
    db.refresh(doc_text)
    db.refresh(document)

    return doc_text


def get_document_text(
    db: Session,
    document_id: UUID,
) -> DocumentText:
    doc_text = (
        db.query(DocumentText)
        .filter(DocumentText.document_id == document_id)
        .first()
    )

    if doc_text is None:
        raise HTTPException(
            status_code=404,
            detail="OCR text not found for this document",
        )

    return doc_text


def get_document_entities(
    db: Session,
    document_id: UUID,
) -> list[DocumentEntity]:
    # Ensure document exists
    get_document_by_id(db, document_id)
    return (
        db.query(DocumentEntity)
        .filter(DocumentEntity.document_id == document_id)
        .order_by(DocumentEntity.confidence.desc())
        .all()
    )
