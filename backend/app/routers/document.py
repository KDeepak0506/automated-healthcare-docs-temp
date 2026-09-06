from uuid import UUID

from fastapi import APIRouter, BackgroundTasks, Depends, File, HTTPException, UploadFile, status
from sqlalchemy.orm import Session

from app.core.dependencies import get_current_user
from app.db.session import get_db
from app.models.user import User
from app.schemas.classification import ClassificationResponse
from app.schemas.document import (
    DocumentProcessingStatus,
    DocumentResponse,
)
from app.schemas.entity import DocumentEntitiesResponse
from app.schemas.ocr import DocumentTextResponse
from app.schemas.rag import IndexResponse, SearchRequest, SearchResponse
from app.schemas.summary import SummaryResponse
from app.services.classification_service import classification_service
from app.services.document_service import (
    get_all_documents,
    get_document_by_id,
    get_document_entities,
    get_document_text,
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

@router.get(
    "",
    response_model=list[DocumentResponse],
)
def get_all_documents_endpoint(
    current_user: User = Depends(get_current_user),
    db: Session = Depends(get_db),
):
    return get_all_documents(db=db)

@router.get(
    "/{document_id}",
    response_model=DocumentResponse,
)
def get_document_by_id_endpoint(
    document_id: UUID,
    current_user: User = Depends(get_current_user),
    db: Session = Depends(get_db),
):
    return get_document_by_id(
        db=db,
        document_id=document_id,
    )

@router.patch(
    "/{document_id}/status",
    response_model=DocumentResponse,
)
def update_document_processing_status(
    document_id: UUID,
    status: DocumentProcessingStatus,
    current_user: User = Depends(get_current_user),
    db: Session = Depends(get_db),
):
    return update_document_status(
        db=db,
        document_id=document_id,
        status=status,
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
            status_code=status.HTTP_403_FORBIDDEN,
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
            status_code=status.HTTP_403_FORBIDDEN,
            detail="Access denied",
        )

    if document.privacy_status != "completed":
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
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
            status_code=status.HTTP_403_FORBIDDEN,
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
            status_code=status.HTTP_403_FORBIDDEN,
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
            status_code=status.HTTP_403_FORBIDDEN,
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
            status_code=status.HTTP_403_FORBIDDEN,
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
            status_code=status.HTTP_403_FORBIDDEN,
            detail="Access denied",
        )

    return rag_service.search_document(
        db=db,
        document=document,
        query=request.query,
        top_k=request.top_k,
    )
