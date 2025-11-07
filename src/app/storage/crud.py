"""CRUD helpers for database operations."""

from collections.abc import Iterable
from datetime import datetime
from typing import Optional

from sqlalchemy import delete, select, func
from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy.orm import selectinload

from .models import Article, ArticleContent, Entity, Relationship


async def get_article_by_url(session: AsyncSession, url: str) -> Optional[Article]:
    result = await session.execute(select(Article).where(Article.url == url))
    return result.scalar_one_or_none()


async def get_article(session: AsyncSession, article_id: int) -> Optional[Article]:
    result = await session.execute(select(Article).where(Article.id == article_id))
    return result.scalar_one_or_none()


async def get_or_create_article(
    session: AsyncSession,
    *,
    feed: str,
    title: str,
    url: Optional[str] = None,
    published_at: Optional[datetime] = None,
) -> Article:
    article: Optional[Article] = None
    if url:
        article = await get_article_by_url(session, url)
    if article:
        article.title = title
        article.feed = feed
        if published_at:
            article.published_at = published_at
        return article

    article = Article(feed=feed, title=title, url=url, published_at=published_at)
    session.add(article)
    await session.flush()
    return article


async def list_articles(session: AsyncSession, limit: int = 20) -> list[Article]:
    result = await session.execute(select(Article).order_by(Article.ingested_at.desc()).limit(limit))
    return list(result.scalars())


async def list_entities(
    session: AsyncSession, *, entity_type: Optional[str] = None, limit: int = 100
) -> list[Entity]:
    stmt = select(Entity).order_by(Entity.id.desc()).limit(limit)
    if entity_type:
        stmt = stmt.where(Entity.type == entity_type)
    result = await session.execute(stmt)
    return list(result.scalars())


async def list_relationships(
    session: AsyncSession, *, rel_type: Optional[str] = None, limit: int = 100
) -> list[Relationship]:
    stmt = select(Relationship).order_by(Relationship.id.desc()).limit(limit)
    if rel_type:
        stmt = stmt.where(Relationship.rel_type == rel_type)
    stmt = stmt.options(
        selectinload(Relationship.source_entity),
        selectinload(Relationship.target_entity),
    )
    result = await session.execute(stmt)
    return list(result.scalars())


async def clear_relationships(session: AsyncSession, article_id: int) -> None:
    await session.execute(delete(Relationship).where(Relationship.article_id == article_id))


async def ensure_entities(
    session: AsyncSession,
    article: Article,
    entities: Iterable[dict],
) -> list[Entity]:
    """Ensure entities exist, returning the ORM objects."""

    existing = await session.execute(select(Entity).where(Entity.article_id == article.id))
    existing_by_norm = {entity.norm: entity for entity in existing.scalars() if entity.norm}

    stored: list[Entity] = []
    for data in entities:
        norm = data.get("norm")
        entity_obj = existing_by_norm.get(norm)
        if entity_obj:
            entity_obj.text = data.get("text", entity_obj.text)
            entity_obj.text_en = data.get("text_en", entity_obj.text_en)
            stored.append(entity_obj)
            continue
        entity_obj = Entity(article_id=article.id, **data)
        session.add(entity_obj)
        stored.append(entity_obj)
    await session.flush()
    return stored


async def replace_relationships(
    session: AsyncSession,
    article: Article,
    relationships: Iterable[dict],
) -> list[Relationship]:
    await clear_relationships(session, article.id)
    created: list[Relationship] = []
    for data in relationships:
        rel = Relationship(article_id=article.id, **data)
        session.add(rel)
        created.append(rel)
    await session.flush()
    return created


async def upsert_article_content(
    session: AsyncSession,
    article: Article,
    *,
    text_fr: str | None,
) -> ArticleContent | None:
    """Persist the French text body for downstream analytics."""

    if not text_fr:
        return None
    result = await session.execute(
        select(ArticleContent).where(ArticleContent.article_id == article.id)
    )
    record = result.scalar_one_or_none()
    if record:
        record.text_fr = text_fr
        return record
    record = ArticleContent(article_id=article.id, text_fr=text_fr)
    session.add(record)
    await session.flush()
    return record


async def list_article_contents(
    session: AsyncSession,
    *,
    min_chars: int = 100,
    limit: int = 200,
) -> list[tuple[Article, ArticleContent]]:
    """Return recent articles with stored text bodies."""

    stmt = (
        select(Article, ArticleContent)
        .join(ArticleContent, ArticleContent.article_id == Article.id)
        .where(func.length(ArticleContent.text_fr) >= min_chars)
        .order_by(Article.ingested_at.desc())
        .limit(limit)
    )
    result = await session.execute(stmt)
    return [(row.Article, row.ArticleContent) for row in result.all()]



