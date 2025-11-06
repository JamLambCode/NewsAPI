"""SQLAlchemy models for Holocron."""

from __future__ import annotations

from datetime import datetime
from typing import List, Optional

from sqlalchemy import DateTime, ForeignKey, Integer, String, Text, UniqueConstraint
from sqlalchemy.orm import DeclarativeBase, Mapped, mapped_column, relationship


class Base(DeclarativeBase):
    """Declarative base class."""


class Article(Base):
    """News article metadata."""

    __tablename__ = "articles"

    id: Mapped[int] = mapped_column(Integer, primary_key=True, index=True)
    feed: Mapped[str] = mapped_column(String(100), index=True)
    title: Mapped[str] = mapped_column(Text)
    url: Mapped[Optional[str]] = mapped_column(String(500), unique=True, nullable=True)
    published_at: Mapped[Optional[datetime]] = mapped_column(DateTime(timezone=True), nullable=True)
    ingested_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), default=datetime.utcnow, nullable=False
    )

    entities: Mapped[List["Entity"]] = relationship("Entity", back_populates="article")
    relationships: Mapped[List["Relationship"]] = relationship(
        "Relationship", back_populates="article"
    )


class Entity(Base):
    """Extracted entity."""

    __tablename__ = "entities"
    __table_args__ = (
        UniqueConstraint("article_id", "norm", name="uq_entity_norm_per_article"),
    )

    id: Mapped[int] = mapped_column(Integer, primary_key=True)
    article_id: Mapped[int] = mapped_column(ForeignKey("articles.id", ondelete="CASCADE"))
    type: Mapped[str] = mapped_column(String(50), index=True)
    text: Mapped[str] = mapped_column(Text)
    norm: Mapped[Optional[str]] = mapped_column(String(255), index=True)
    text_en: Mapped[Optional[str]] = mapped_column(Text)

    article: Mapped[Article] = relationship("Article", back_populates="entities")
    source_relationships: Mapped[List["Relationship"]] = relationship(
        "Relationship", back_populates="source_entity", foreign_keys="Relationship.source_entity_id"
    )
    target_relationships: Mapped[List["Relationship"]] = relationship(
        "Relationship", back_populates="target_entity", foreign_keys="Relationship.target_entity_id"
    )


class Relationship(Base):
    """Typed relationship between entities."""

    __tablename__ = "relationships"

    id: Mapped[int] = mapped_column(Integer, primary_key=True)
    article_id: Mapped[int] = mapped_column(ForeignKey("articles.id", ondelete="CASCADE"))
    rel_type: Mapped[str] = mapped_column(String(50), index=True)
    source_entity_id: Mapped[int] = mapped_column(
        ForeignKey("entities.id", ondelete="CASCADE"), index=True
    )
    target_entity_id: Mapped[int] = mapped_column(
        ForeignKey("entities.id", ondelete="CASCADE"), index=True
    )
    context: Mapped[Optional[str]] = mapped_column(Text)
    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), default=datetime.utcnow, nullable=False
    )

    article: Mapped[Article] = relationship("Article", back_populates="relationships")
    source_entity: Mapped[Entity] = relationship(
        "Entity", foreign_keys=[source_entity_id], back_populates="source_relationships"
    )
    target_entity: Mapped[Entity] = relationship(
        "Entity", foreign_keys=[target_entity_id], back_populates="target_relationships"
    )


