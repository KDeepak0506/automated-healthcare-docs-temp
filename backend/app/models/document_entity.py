from datetime import datetime
from uuid import UUID, uuid4

from sqlalchemy import DateTime, Float, ForeignKey, Integer, String, func
from sqlalchemy.orm import Mapped, mapped_column, relationship

from app.db.base import Base


class DocumentEntity(Base):
    __tablename__ = "document_entities"

    entity_id: Mapped[UUID] = mapped_column(
        primary_key=True,
        default=uuid4,
    )

    document_id: Mapped[UUID] = mapped_column(
        ForeignKey("documents.document_id", ondelete="CASCADE"),
        nullable=False,
        index=True,
    )

    text: Mapped[str] = mapped_column(
        String,
        nullable=False,
    )

    label: Mapped[str] = mapped_column(
        String,
        nullable=False,
    )

    confidence: Mapped[float] = mapped_column(
        Float,
        nullable=False,
    )

    start_char: Mapped[int | None] = mapped_column(
        Integer,
        nullable=True,
    )

    end_char: Mapped[int | None] = mapped_column(
        Integer,
        nullable=True,
    )

    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True),
        nullable=False,
        server_default=func.now(),
    )

    document: Mapped["Document"] = relationship(  # noqa: F821
        "Document",
        back_populates="entities",
    )
