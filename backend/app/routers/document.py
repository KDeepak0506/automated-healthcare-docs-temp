from uuid import UUID

from fastapi import APIRouter, BackgroundTasks, Depends, File, HTTPException, Query, UploadFile, status as http_status
from sqlalchemy.orm import Session

from app.core.dependencies import get_current_user
from app.db.session import get_db
from app.models.user import User
from app.schemas.classification import ClassificationResponse
from app.schemas.document import (
    DocumentProcessingStatus,
    DocumentResponse,
    PaginatedDocumentResponse,
)
from app.schemas.entity import DocumentEntitiesResponse
from app.schemas.ocr import DocumentTextResponse
from app.schemas.rag import ChunkSourceDetail, IndexResponse, SearchRequest, SearchResponse
from app.schemas.summary import SummaryResponse
from app.services.classification_service import classification_service
from app.services.document_service import (
    delete_document,
    get_all_documents,
    get_chunk_source,
    get_document_by_id,
    get_document_entities,
    get_document_text,
    list_documents,
    update_document_status,
    upload_document,
)
from app.services.rag_service import rag_service
from app.services.summary_service import summary_service



router = APIRouter(
    prefix="/documents",
    tags=["Documents"],
)


@router.post(
    "",
    response_model=DocumentResponse,
    status_code=201,
)
def upload_document_endpoint(
    background_tasks: BackgroundTasks,
    file: UploadFile = File(...),
    patient_id: UUID | None = None,
    current_user: User = Depends(get_current_user),
    db: Session = Depends(get_db),
):
    return upload_document(
        db=db,
        file=file,
        patient_id=patient_id,
        uploaded_by=current_user.user_id,
        background_tasks=background_tasks,
    )


# M7: Ownership-scoped listing with pagination, search, and filters
@router.get(
    "",
    response_model=PaginatedDocumentResponse,
)
def get_all_documents_endpoint(
    page: int = Query(default=1, ge=1, description="Page number (1-indexed)"),
    page_size: int = Query(default=20, ge=1, le=100, description="Number of items per page"),
    search: str | None = Query(default=None, description="Search by filename"),
    processing_status: str | None = Query(default=None, description="Filter by OCR processing status"),
    privacy_status: str | None = Query(default=None, description="Filter by privacy processing status"),
    document_type: str | None = Query(default=None, description="Filter by document type"),
    patient_id: UUID | None = Query(default=None, description="Filter by patient ID"),
    current_user: User = Depends(get_current_user),
    db: Session = Depends(get_db),
):
    """List documents owned by the authenticated user with pagination, search, and filters."""
    return list_documents(
        db=db,
        user_id=current_user.user_id,
        page=page,
        page_size=page_size,
        search=search,
        processing_status=processing_status,
        privacy_status=privacy_status,
        document_type=document_type,
        patient_id=patient_id,
    )


@router.get(
    "/{document_id}",
    response_model=DocumentResponse,
)
def get_document_by_id_endpoint(
    document_id: UUID,
    current_user: User = Depends(get_current_user),
    db: Session = Depends(get_db),
):
    document = get_document_by_id(
        db=db,
        document_id=document_id,
    )

    if document.uploaded_by != current_user.user_id:
        raise HTTPException(
            status_code=http_status.HTTP_403_FORBIDDEN,
            detail="Access denied",
        )

    return document


@router.patch(
    "/{document_id}/status",
    response_model=DocumentResponse,
)
def update_document_processing_status(
    document_id: UUID,
    processing_status: DocumentProcessingStatus = Query(..., alias="status"),
    current_user: User = Depends(get_current_user),
    db: Session = Depends(get_db),
):
    document = get_document_by_id(db=db, document_id=document_id)

    if document.uploaded_by != current_user.user_id:
        raise HTTPException(
            status_code=http_status.HTTP_403_FORBIDDEN,
            detail="Access denied",
        )

    return update_document_status(
        db=db,
        document_id=document_id,
        status=processing_status,
    )


# M7: Document deletion
@router.delete(
    "/{document_id}",
    status_code=204,
)
def delete_document_endpoint(
    document_id: UUID,
    current_user: User = Depends(get_current_user),
    db: Session = Depends(get_db),
):
    """Delete a document owned by the authenticated user and clean up associated files."""
    delete_document(
        db=db,
        document_id=document_id,
        user_id=current_user.user_id,
    )


@router.get(
    "/{document_id}/text",
    response_model=DocumentTextResponse,
)
def get_document_ocr_text(
    document_id: UUID,
    current_user: User = Depends(get_current_user),
    db: Session = Depends(get_db),
):
    """Return the stored OCR text for a document owned by the authenticated user."""
    document = get_document_by_id(db=db, document_id=document_id)

    if document.uploaded_by != current_user.user_id:
        raise HTTPException(
            status_code=http_status.HTTP_403_FORBIDDEN,
            detail="Access denied",
        )

    return get_document_text(db=db, document_id=document_id)


@router.get(
    "/{document_id}/sanitized-text",
)
def get_sanitized_document_text(
    document_id: UUID,
    current_user: User = Depends(get_current_user),
    db: Session = Depends(get_db),
):
    """Return the sanitized OCR text ready for downstream AI analysis."""
    document = get_document_by_id(db=db, document_id=document_id)

    if document.uploaded_by != current_user.user_id:
        raise HTTPException(
            status_code=http_status.HTTP_403_FORBIDDEN,
            detail="Access denied",
        )

    if document.privacy_status != "completed":
        raise HTTPException(
            status_code=http_status.HTTP_400_BAD_REQUEST,
            detail="Sanitized text unavailable. Privacy processing has not completed or failed.",
        )

    doc_text = get_document_text(db=db, document_id=document_id)
    return {
        "document_id": document_id,
        "privacy_status": document.privacy_status,
        "sanitized_text": doc_text.sanitized_text,
    }


@router.get(
    "/{document_id}/entities",
    response_model=DocumentEntitiesResponse,
)
def get_document_clinical_entities(
    document_id: UUID,
    current_user: User = Depends(get_current_user),
    db: Session = Depends(get_db),
):
    """Return clinical entities extracted via M4 Biomedical NER from sanitized text."""
    document = get_document_by_id(db=db, document_id=document_id)

    if document.uploaded_by != current_user.user_id:
        raise HTTPException(
            status_code=http_status.HTTP_403_FORBIDDEN,
            detail="Access denied",
        )

    entities = get_document_entities(db=db, document_id=document_id)
    return {
        "document_id": document_id,
        "entities": entities,
        "total_count": len(entities),
    }


@router.post(
    "/{document_id}/classify",
    response_model=ClassificationResponse,
)
def classify_document_endpoint(
    document_id: UUID,
    current_user: User = Depends(get_current_user),
    db: Session = Depends(get_db),
):
    """Classify healthcare document type using LLM strictly from sanitized OCR text."""
    document = get_document_by_id(db=db, document_id=document_id)

    if document.uploaded_by != current_user.user_id:
        raise HTTPException(
            status_code=http_status.HTTP_403_FORBIDDEN,
            detail="Access denied",
        )

    return classification_service.classify_document(
        db=db,
        document=document,
    )


@router.post(
    "/{document_id}/summarize",
    response_model=SummaryResponse,
)
def summarize_document_endpoint(
    document_id: UUID,
    current_user: User = Depends(get_current_user),
    db: Session = Depends(get_db),
):
    """Generate factual clinical summary and key findings using LLM strictly from sanitized OCR text."""
    document = get_document_by_id(db=db, document_id=document_id)

    if document.uploaded_by != current_user.user_id:
        raise HTTPException(
            status_code=http_status.HTTP_403_FORBIDDEN,
            detail="Access denied",
        )

    return summary_service.summarize_document(
        db=db,
        document=document,
    )


@router.post(
    "/{document_id}/index",
    response_model=IndexResponse,
)
def index_document_endpoint(
    document_id: UUID,
    current_user: User = Depends(get_current_user),
    db: Session = Depends(get_db),
):
    """Index document text chunks and vector embeddings for semantic search."""
    document = get_document_by_id(db=db, document_id=document_id)

    if document.uploaded_by != current_user.user_id:
        raise HTTPException(
            status_code=http_status.HTTP_403_FORBIDDEN,
            detail="Access denied",
        )

    return rag_service.index_document(
        db=db,
        document=document,
    )


@router.post(
    "/{document_id}/search",
    response_model=SearchResponse,
)
def search_document_endpoint(
    document_id: UUID,
    request: SearchRequest,
    current_user: User = Depends(get_current_user),
    db: Session = Depends(get_db),
):
    """Perform grounded semantic search / RAG Q&A on a document."""
    document = get_document_by_id(db=db, document_id=document_id)

    if document.uploaded_by != current_user.user_id:
        raise HTTPException(
            status_code=http_status.HTTP_403_FORBIDDEN,
            detail="Access denied",
        )

    return rag_service.search_document(
        db=db,
        document=document,
        query=request.query,
        top_k=request.top_k,
    )


# M8: Source chunk retrieval for evidence inspection
@router.get(
    "/{document_id}/sources/{chunk_id}",
    response_model=ChunkSourceDetail,
)
def get_chunk_source_endpoint(
    document_id: UUID,
    chunk_id: UUID,
    current_user: User = Depends(get_current_user),
    db: Session = Depends(get_db),
):
    """Retrieve the exact RAG chunk used as source evidence (M8).
    Returns sanitized chunk text only. Never exposes raw OCR.
    Enforces document ownership before returning chunk data."""
    chunk = get_chunk_source(
        db=db,
        document_id=document_id,
        chunk_id=chunk_id,
        user_id=current_user.user_id,
    )

    return ChunkSourceDetail(
        document_id=chunk.document_id,
        chunk_id=chunk.chunk_id,
        chunk_index=chunk.chunk_index,
        page_number=chunk.page_number,
        similarity_score=None,  # Not stored on the chunk itself; provided by search response
        text=chunk.chunk_text,  # Sanitized text (chunks are created from sanitized_text only)
        start_offset=chunk.start_offset,
        end_offset=chunk.end_offset,
    )