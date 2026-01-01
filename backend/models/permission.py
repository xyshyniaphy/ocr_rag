"""
Permission Model
Database model for document-level access control
"""

import uuid
from datetime import datetime

from sqlalchemy import String, Boolean, DateTime, func
from sqlalchemy.dialects.postgresql import UUID
from sqlalchemy.orm import Mapped, mapped_column

from backend.db.base import Base, TimestampMixin, UUIDMixin


class Permission(Base, TimestampMixin, UUIDMixin):
    """Document-level permission model (ACL)"""

    __tablename__ = "permissions"

    # Resource
    document_id: Mapped[uuid.UUID] = mapped_column(
        UUID(as_uuid=True),
        nullable=False,
        index=True,
    )

    # Subject (who has permission)
    user_id: Mapped[uuid.UUID | None] = mapped_column(
        UUID(as_uuid=True),
        nullable=True,
        index=True,
    )
    role: Mapped[str | None] = mapped_column(String(20), index=True)

    # Permissions
    can_view: Mapped[bool] = mapped_column(Boolean, default=False)
    can_download: Mapped[bool] = mapped_column(Boolean, default=False)
    can_delete: Mapped[bool] = mapped_column(Boolean, default=False)
    can_share: Mapped[bool] = mapped_column(Boolean, default=False)

    # Timestamps
    granted_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True),
        server_default=func.now(),
    )
    granted_by: Mapped[uuid.UUID | None] = mapped_column(UUID(as_uuid=True))
    expires_at: Mapped[datetime | None] = mapped_column(DateTime(timezone=True), index=True)
