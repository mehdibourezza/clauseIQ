"""
Database tables for the contract review app (SQLAlchemy 2.0 ORM).

contracts        one row per uploaded contract, with its text and analysis status
clause_findings  one row per excerpt returned by the extraction engine, verified or not

Unverified excerpts are stored too (verified = False, no offsets): they are never
highlighted, but keeping them makes the model's invention rate measurable.
"""

import uuid
from datetime import datetime, timezone

from sqlalchemy import ForeignKey, String, Text
from sqlalchemy.orm import DeclarativeBase, Mapped, mapped_column, relationship


def _utcnow() -> datetime:
    return datetime.now(timezone.utc)


class Base(DeclarativeBase):
    pass


class Contract(Base):
    __tablename__ = "contracts"

    id: Mapped[str] = mapped_column(String(36), primary_key=True, default=lambda: str(uuid.uuid4()))
    filename: Mapped[str] = mapped_column(String(255))
    text: Mapped[str] = mapped_column(Text)
    status: Mapped[str] = mapped_column(String(16), default="processing")  # processing | done | failed
    error: Mapped[str | None] = mapped_column(Text)
    model: Mapped[str | None] = mapped_column(String(64))
    input_tokens: Mapped[int | None]
    output_tokens: Mapped[int | None]
    created_at: Mapped[datetime] = mapped_column(default=_utcnow, index=True)
    completed_at: Mapped[datetime | None]

    findings: Mapped[list["ClauseFinding"]] = relationship(
        back_populates="contract", cascade="all, delete-orphan"
    )


class ClauseFinding(Base):
    __tablename__ = "clause_findings"

    id: Mapped[int] = mapped_column(primary_key=True)
    contract_id: Mapped[str] = mapped_column(ForeignKey("contracts.id", ondelete="CASCADE"), index=True)
    clause_type: Mapped[str] = mapped_column(String(64), index=True)
    quote: Mapped[str] = mapped_column(Text)
    start_char: Mapped[int | None]
    end_char: Mapped[int | None]
    verified: Mapped[bool]

    contract: Mapped[Contract] = relationship(back_populates="findings")
