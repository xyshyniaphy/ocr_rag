"""
SQLAlchemy Database Models
"""

import uuid
from datetime import datetime
from typing import Optional, TYPE_CHECKING

from sqlalchemy import Boolean, DateTime, String, Text, Integer, Float, ForeignKey
from sqlalchemy.dialects.postgresql import UUID
from sqlalchemy.orm import Mapped, mapped_column, relationship

from backend.db.base import Base, TimestampMixin, UUIDMixin

if TYPE_CHECKING:
    from backend.models.chunk import Chunk


class User(UUIDMixin, TimestampMixin, Base):
    """User SQLAlchemy model"""

    __tablename__ = "users"

    email: Mapped[str] = mapped_column(String(255), unique=True, nullable=False, index=True)
    full_name: Mapped[str] = mapped_column(String(255), nullable=False)
    display_name: Mapped[Optional[str]] = mapped_column(String(100), nullable=True)
    hashed_password: Mapped[str] = mapped_column(String(255), nullable=False)
    role: Mapped[str] = mapped_column(String(50), nullable=False, default="user")
    is_active: Mapped[bool] = mapped_column(Boolean, nullable=False, default=True)
    is_verified: Mapped[bool] = mapped_column(Boolean, nullable=False, default=False)
    last_login_at: Mapped[Optional[datetime]] = mapped_column(
        DateTime(timezone=True), nullable=True
    )

    @property
    def password(self) -> str:
        """Password property getter (for backward compatibility)"""
        return self.hashed_password

    @password.setter
    def password(self, value: str):
        """Password property setter"""
        self.hashed_password = value


class Document(UUIDMixin, TimestampMixin, Base):
    """Document SQLAlchemy model"""

    __tablename__ = "documents"

    filename: Mapped[str] = mapped_column(String(512), nullable=False)
    file_size_bytes: Mapped[int] = mapped_column(Integer, nullable=False)
    file_hash: Mapped[str] = mapped_column(String(64), nullable=False, index=True)
    content_type: Mapped[str] = mapped_column(String(100), nullable=False)
    title: Mapped[Optional[str]] = mapped_column(String(512), nullable=True)
    author: Mapped[Optional[str]] = mapped_column(String(255), nullable=True)
    subject: Mapped[Optional[str]] = mapped_column(String(255), nullable=True)
    keywords: Mapped[Optional[str]] = mapped_column(Text, nullable=True)  # JSON array
    language: Mapped[str] = mapped_column(String(10), nullable=False, default="ja")
    category: Mapped[Optional[str]] = mapped_column(String(100), nullable=True)
    doc_metadata: Mapped[Optional[str]] = mapped_column(Text, nullable=True)  # JSON object (renamed to avoid conflict)
    status: Mapped[str] = mapped_column(String(50), nullable=False, default="pending")
    page_count: Mapped[Optional[int]] = mapped_column(Integer, nullable=True)
    chunk_count: Mapped[int] = mapped_column(Integer, nullable=False, default=0)
    ocr_confidence: Mapped[Optional[float]] = mapped_column(Float, nullable=True)
    thumbnail_url: Mapped[Optional[str]] = mapped_column(String(512), nullable=True)
    owner_id: Mapped[uuid.UUID] = mapped_column(
        UUID(as_uuid=True), nullable=False, index=True
    )
    processing_completed_at: Mapped[Optional[datetime]] = mapped_column(
        DateTime(timezone=True), nullable=True
    )
    deleted_at: Mapped[Optional[datetime]] = mapped_column(
        DateTime(timezone=True), nullable=True, index=True
    )

    # Relationships
    chunks: Mapped[list["Chunk"]] = relationship(
        "Chunk",
        back_populates="document",
        cascade="all, delete-orphan",
    )

    @property
    def uploaded_at(self) -> datetime:
        """Alias for created_at for backward compatibility"""
        return self.created_at


class Query(UUIDMixin, TimestampMixin, Base):
    """Query SQLAlchemy model for RAG queries"""

    __tablename__ = "queries"

    user_id: Mapped[Optional[uuid.UUID]] = mapped_column(
        UUID(as_uuid=True), nullable=True, index=True
    )
    query_text: Mapped[str] = mapped_column(Text, nullable=False)
    query_language: Mapped[str] = mapped_column(String(10), nullable=False, default="ja")
    query_type: Mapped[str] = mapped_column(String(50), nullable=False, default="hybrid")
    top_k: Mapped[int] = mapped_column(Integer, nullable=False, default=5)
    retrieved_count: Mapped[int] = mapped_column(Integer, nullable=False, default=0)
    answer: Mapped[Optional[str]] = mapped_column(Text, nullable=True)
    confidence: Mapped[Optional[float]] = mapped_column(Float, nullable=True)
    sources: Mapped[Optional[str]] = mapped_column(Text, nullable=True)  # JSON array of source references
    processing_time_ms: Mapped[Optional[int]] = mapped_column(Integer, nullable=True)
    stage_timings_ms: Mapped[Optional[str]] = mapped_column(Text, nullable=True)  # JSON object
    llm_model: Mapped[Optional[str]] = mapped_column(String(100), nullable=True)
    embedding_model: Mapped[Optional[str]] = mapped_column(String(100), nullable=True)
