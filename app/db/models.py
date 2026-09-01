from __future__ import annotations

from datetime import datetime, timezone

from sqlalchemy import DateTime, Integer, String, Text
from sqlalchemy.orm import Mapped, mapped_column

from app.db.base import Base


def utcnow() -> datetime:
    return datetime.now(timezone.utc)


class Project(Base):
    __tablename__ = "projects"

    id: Mapped[int] = mapped_column(Integer, primary_key=True)
    name: Mapped[str] = mapped_column(String(120), unique=True, index=True)
    original_lyrics: Mapped[str] = mapped_column(Text)
    detected_sections: Mapped[str] = mapped_column(Text, default="[]")
    edited_sections: Mapped[str] = mapped_column(Text, default="[]")
    controls: Mapped[str] = mapped_column(Text, default="{}")
    analysis_features: Mapped[str] = mapped_column(Text, default="{}")
    raw_model_output: Mapped[str] = mapped_column(Text, default="")
    validated_output: Mapped[str] = mapped_column(Text, default="{}")
    repair_details: Mapped[str] = mapped_column(Text, default="[]")
    chord_alignment: Mapped[str] = mapped_column(Text, default="[]")
    manual_chord_edits: Mapped[str] = mapped_column(Text, default="[]")
    random_seed: Mapped[int] = mapped_column(Integer, default=42)
    decoding_settings: Mapped[str] = mapped_column(Text, default="{}")
    generation_source: Mapped[str] = mapped_column(String(80), default="algorithmic_fallback")
    base_model_version: Mapped[str] = mapped_column(String(120), default="google/flan-t5-small")
    adapter_version: Mapped[str] = mapped_column(String(120), default="unavailable")
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), default=utcnow)
    updated_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), default=utcnow, onupdate=utcnow)

