import logging
import math
from datetime import datetime, timezone
from pathlib import Path
from uuid import UUID

from fastapi import BackgroundTasks, HTTPException, UploadFile
from sqlalchemy.orm import Session

from app.db.session import SessionLocal
from app.models.user import User
from app.models.document import Document
from app.models.document_chunk import DocumentChunk
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
    """Internal background task to run OCR, Privacy, M4 NER, M3 Classification, M5 Summary, and M6 Indexing."""
    db = SessionLocal()
    try:
        ocr_result = extract_text(file_path, file_type)
        doc_text = store_ocr_result(db, document_id, ocr_result)

        document = db.query(Document).filter(Document.document_id == document_id).first()
        if document and document.privacy_status == "completed" and doc_text.sanitized_text:
            # 1. M4 Clinical NER
            try:
                logger.info(f"Triggering M4 Clinical NER for document {document_id}")
                clinical_ner_service.extract_and_store_entities(
                    db=db,
                    document_id=document_id,
                    sanitized_text=doc_text.sanitized_text,
                )
            except Exception as ner_exc:
                logger.error(f"M4 Clinical NER failed for document {document_id}: {ner_exc}")

            # 2. M3 Classification
            try:
                from app.services.classification_service import classification_service
                logger.info(f"Triggering M3 Classification for document {document_id}")
                classification_service.classify_document(db=db, document=document)
            except Exception as class_exc:
                logger.error(f"M3 Classification failed for document {document_id}: {class_exc}")

            # 3. M5 Summarization
            try:
                from app.services.summary_service import summary_service
                logger.info(f"Triggering M5 Summarization for document {document_id}")
                summary_service.summarize_document(db=db, document=document)
            except Exception as sum_exc:
                logger.error(f"M5 Summarization failed for document {document_id}: {sum_exc}")

            # 4. M6 Vector Indexing
            try:
                from app.services.rag_service import rag_service
                logger.info(f"Triggering M6 Indexing for document {document_id}")
                rag_service.index_document(db=db, document=document)
            except Exception as idx_exc:
                logger.error(f"M6 Indexing failed for document {document_id}: {idx_exc}")

        else:
            status_str = document.privacy_status if document else "None"
            logger.warning(
                f"Skipping downstream AI processing for document {document_id}: "
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
    """Internal helper: returns all documents (no ownership filter). Not exposed via API."""
    documents = (
        db.query(Document)
        .order_by(Document.uploaded_at.desc())
        .all()
    )

    return documents


def list_documents(
    db: Session,
    user_or_id: User | UUID,
    page: int = 1,
    page_size: int = 20,
    search: str | None = None,
    processing_status: str | None = None,
    privacy_status: str | None = None,
    document_type: str | None = None,
    patient_id: UUID | None = None,
) -> dict:
    """Ownership and patient-access scoped document listing with pagination, search, and filters."""
    from app.models.user import User
    from app.schemas.user import UserRole
    from app.models.patient import PatientAssignment

    if isinstance(user_or_id, User):
        user = user_or_id
        user_id = user.user_id
    else:
        user_id = user_or_id
        user = db.query(User).filter(User.user_id == user_id).first()

    query = db.query(Document)

    # Filtering logic based on user role and patient assignments
    if user and user.role not in (UserRole.ADMIN.value, UserRole.RECORDS_STAFF.value):
        assigned_patient_ids = (
            db.query(PatientAssignment.patient_id)
            .filter(PatientAssignment.user_id == user_id)
            .scalar_subquery()
        )
        query = query.filter(
            (Document.uploaded_by == user_id)
            | (Document.patient_id.in_(assigned_patient_ids))
        )
    elif not user:
        query = query.filter(Document.uploaded_by == user_id)

    # Filename search (case-insensitive)
    if search:
        query = query.filter(Document.file_name.ilike(f"%{search}%"))

    # Status filters
    if processing_status:
        query = query.filter(Document.processing_status == processing_status)
    if privacy_status:
        query = query.filter(Document.privacy_status == privacy_status)
    if document_type:
        query = query.filter(Document.document_type == document_type)
    if patient_id:
        query = query.filter(Document.patient_id == patient_id)

    total = query.count()
    total_pages = math.ceil(total / page_size) if page_size > 0 else 1

    items = (
        query
        .order_by(Document.uploaded_at.desc())
        .offset((page - 1) * page_size)
        .limit(page_size)
        .all()
    )

    return {
        "items": items,
        "page": page,
        "page_size": page_size,
        "total": total,
        "total_pages": max(total_pages, 1),
    }


def delete_document(
    db: Session,
    document_id: UUID,
    user_or_id: User | UUID,
) -> None:
    """Delete a document. Uploader or Admin / Records Staff can delete."""
    from app.models.user import User
    from app.schemas.user import UserRole

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

    if isinstance(user_or_id, User):
        user = user_or_id
        user_id = user.user_id
    else:
        user_id = user_or_id
        user = db.query(User).filter(User.user_id == user_id).first()

    is_admin_or_staff = user and user.role in (UserRole.ADMIN.value, UserRole.RECORDS_STAFF.value)
    if document.uploaded_by != user_id and not is_admin_or_staff:
        raise HTTPException(
            status_code=403,
            detail="Access denied",
        )


    # Attempt to remove the uploaded file from disk safely
    if document.file_url:
        file_path = Path(document.file_url)
        try:
            if file_path.exists():
                file_path.unlink()
        except Exception as exc:  # noqa: BLE001
            logger.warning(
                f"Could not delete file {file_path} for document {document_id}: {exc}"
            )

    # Database delete — cascades to document_text, document_entities, document_chunks
    db.delete(document)
    db.commit()


def get_chunk_source(
    db: Session,
    document_id: UUID,
    chunk_id: UUID,
    user_or_id: User | UUID,
) -> DocumentChunk:
    """Retrieve a specific chunk for M8 source verification. Enforces patient access permission."""
    from app.models.user import User
    from app.services.patient_service import has_document_access

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

    if isinstance(user_or_id, User):
        user = user_or_id
    else:
        user = db.query(User).filter(User.user_id == user_or_id).first()

    if user and not has_document_access(db, user, document):
        raise HTTPException(
            status_code=403,
            detail="Access denied",
        )

    chunk = (
        db.query(DocumentChunk)
        .filter(
            DocumentChunk.chunk_id == chunk_id,
            DocumentChunk.document_id == document_id,
        )
        .first()
    )

    if chunk is None:
        raise HTTPException(
            status_code=404,
            detail="Source chunk not found",
        )

    return chunk


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
