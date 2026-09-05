from datetime import datetime
from uuid import UUID, uuid4

from sqlalchemy import DateTime, Float, ForeignKey, Numeric, String, Text, func
from sqlalchemy.dialects.postgresql import JSONB
from sqlalchemy.orm import Mapped, mapped_column, relationship
from sqlalchemy.types import JSON

from app.db.base import Base


class Document(Base):
    __tablename__ = "documents"

    document_id: Mapped[UUID] = mapped_column(
        primary_key=True,
        default=uuid4,
    )

    patient_id: Mapped[UUID | None] = mapped_column(
        nullable=True,
    )

    uploaded_by: Mapped[UUID] = mapped_column(
        ForeignKey("users.user_id"),
        nullable=False,
    )

    file_name: Mapped[str] = mapped_column(
        String,
        nullable=False,
    )

    file_type: Mapped[str] = mapped_column(
        String,
        nullable=False,
    )

    file_url: Mapped[str] = mapped_column(
        String,
        nullable=False,
    )

    document_type: Mapped[str | None] = mapped_column(
        String,
        nullable=True,
    )

    classification_confidence: Mapped[float | None] = mapped_column(
        Float,
        nullable=True,
    )

    summary: Mapped[str | None] = mapped_column(
        Text,
        nullable=True,
    )

    key_findings: Mapped[list[str] | None] = mapped_column(
        JSONB().with_variant(JSON(), "sqlite"),
        nullable=True,
    )

    processing_status: Mapped[str] = mapped_column(
        String,
        nullable=False,
        default="Pending",
    )

    privacy_status: Mapped[str] = mapped_column(
        String,
        nullable=False,
        default="pending",
    )

    ocr_quality_score: Mapped[float | None] = mapped_column(
        Numeric,
        nullable=True,
    )

    uploaded_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True),
        nullable=False,
        server_default=func.now(),
    )

    processed_at: Mapped[datetime | None] = mapped_column(
        DateTime(timezone=True),
        nullable=True,
    )

    uploader: Mapped["User"] = relationship(  # noqa: F821
        "User",
        back_populates="documents",
    )

    text_record: Mapped["DocumentText | None"] = relationship(  # noqa: F821
        "DocumentText",
        back_populates="document",
        uselist=False,
    )

    entities: Mapped[list["DocumentEntity"]] = relationship(  # noqa: F821
        "DocumentEntity",
        back_populates="document",
        cascade="all, delete-orphan",
    )