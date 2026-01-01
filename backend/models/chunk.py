"""
Chunk Model
Database model for text chunks
"""

import uuid
from datetime import datetime
from typing import TYPE_CHECKING

from sqlalchemy import String, Integer, Float, Boolean, DateTime, Text, ForeignKey
from sqlalchemy.dialects.postgresql import UUID
from sqlalchemy.orm import Mapped, mapped_column, relationship

from backend.db.base import Base, TimestampMixin, UUIDMixin

if TYPE_CHECKING:
    from backend.db.models import Document


class Chunk(Base, TimestampMixin, UUIDMixin):
    """Text chunk model"""

    __tablename__ = "chunks"

    # References
    document_id: Mapped[uuid.UUID] = mapped_column(
        UUID(as_uuid=True),
        ForeignKey("documents.id", ondelete="CASCADE"),
        nullable=False,
        index=True,
    )
    milvus_id: Mapped[str] = mapped_column(String(64), nullable=False, unique=True, index=True)

    # Position
    page_number: Mapped[int] = mapped_column(Integer, nullable=False, index=True)
    chunk_index: Mapped[int] = mapped_column(Integer, nullable=False)

    # Content
    text_content: Mapped[str] = mapped_column(Text, nullable=False)
    token_count: Mapped[int | None] = mapped_column(Integer)

    # Metadata
    chunk_type: Mapped[str] = mapped_column(String(20), default="text", index=True)
    section_header: Mapped[str | None] = mapped_column(String(512))
    paragraph_index: Mapped[int | None] = mapped_column(Integer)

    # Quality
    confidence: Mapped[float | None] = mapped_column(Float)
    is_table: Mapped[bool] = mapped_column(Boolean, default=False)
    table_row_count: Mapped[int | None] = mapped_column(Integer)
    table_col_count: Mapped[int | None] = mapped_column(Integer)

    # Embedding info
    embedding_model: Mapped[str | None] = mapped_column(String(100))
    embedding_dimension: Mapped[int] = mapped_column(Integer, default=768)
    embedding_created_at: Mapped[datetime | None] = mapped_column(DateTime(timezone=True))

    # Relationships
    document: Mapped["Document"] = relationship(
        "Document",
        back_populates="chunks",
    )
