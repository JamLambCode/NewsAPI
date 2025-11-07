"""FastAPI application entrypoint."""

from __future__ import annotations

import asyncio
import logging
from typing import Optional

from apscheduler.schedulers.asyncio import AsyncIOScheduler
from fastapi import Depends, FastAPI, HTTPException, Query, status
from fastapi.middleware.cors import CORSMiddleware
from sqlalchemy.ext.asyncio import AsyncSession

from .analytics.dejavu import ArticleMeta, collect_reused_quotes
from .config import get_settings
from .deps import get_db_session
from .ingest.rss import FEED_LANG_MAP, RSSClient
from .llm.relationships import RelationshipPrompt, build_classifier_from_settings
from .llm.patterns import PatternSummarizer
from .nlp.ner import DeterministicNER, EntitySpan, normalize_text
from .nlp.translate import Translator
from .schemas import (
    Article as ArticleSchema,
    DejaVuCluster as DejaVuClusterSchema,
    DejaVuOccurrence as DejaVuOccurrenceSchema,
    EntitiesRequest,
    EntitiesResponse,
    Entity as EntitySchema,
    GraphEdge,
    GraphNode,
    GraphResponse,
    Relationship as RelationshipSchema,
    RelationshipEndpoint,
)
from .storage import crud
from .storage.db import async_session_factory
from .storage.models import Entity


settings = get_settings()

# Configure logging
logging.basicConfig(
    level=getattr(logging, settings.log_level.upper(), logging.INFO),
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
)
logger = logging.getLogger("holocron")

app = FastAPI(title="Holocron News Intelligence API", version="0.1.0")

# Optional CORS support when serving a frontend locally.
if settings.app_env == "development":
    app.add_middleware(
        CORSMiddleware,
        allow_origins=["http://localhost:5173"],
        allow_methods=["*"],
        allow_headers=["*"],
    )

logger.info("Initializing NLP and LLM components")
ner = DeterministicNER()
translator = Translator()
classifier = build_classifier_from_settings(settings)
summarizer = PatternSummarizer(
    ollama_model=settings.ollama_model,
    ollama_base_url=settings.ollama_base_url,
    openai_model=settings.openai_model,
    openai_key=settings.openai_api_key,
)
scheduler: Optional[AsyncIOScheduler] = None


@app.on_event("startup")
async def on_startup() -> None:
    global scheduler
    logger.info("Application startup initiated")
    if settings.scheduler_enabled:
        logger.info(
            "Starting scheduler with %d feed(s), interval: %d minutes",
            len(settings.ingest_feeds),
            settings.ingest_interval_minutes,
        )
        scheduler = AsyncIOScheduler()
        for feed in settings.ingest_feeds:
            scheduler.add_job(
                ingest_feed,
                "interval",
                minutes=settings.ingest_interval_minutes,
                args=[feed],
                id=f"ingest-{feed}",
                replace_existing=True,
            )
        scheduler.start()
    else:
        logger.info("Scheduler disabled")


@app.on_event("shutdown")
async def on_shutdown() -> None:
    logger.info("Application shutdown initiated")
    if scheduler and scheduler.running:
        scheduler.shutdown()
        logger.info("Scheduler stopped")


@app.get("/health", tags=["system"])
async def health() -> dict[str, str]:
    """Simple health probe endpoint."""

    return {"status": "ok", "env": settings.app_env}


@app.post("/entities", response_model=EntitiesResponse, tags=["entities"], status_code=status.HTTP_201_CREATED)
async def extract_entities(
    payload: EntitiesRequest,
    session: AsyncSession = Depends(get_db_session),
) -> EntitiesResponse:
    """Process a single article and return extracted entities/relationships."""

    logger.info("POST /entities: title=%s, feed=%s", payload.title[:50], payload.feed)
    result = await process_payload(session, payload)
    await session.commit()
    logger.info(
        "POST /entities complete: %d entities, %d relationships",
        len(result.entities),
        len(result.relationships),
    )
    return result


@app.get("/articles", response_model=list[ArticleSchema], tags=["articles"])
async def list_articles(session: AsyncSession = Depends(get_db_session)) -> list[ArticleSchema]:
    records = await crud.list_articles(session)
    return [ArticleSchema.model_validate(record) for record in records]


@app.get("/entities", response_model=list[EntitySchema], tags=["entities"])
async def list_entities(
    entity_type: Optional[str] = Query(default=None, alias="type"),
    session: AsyncSession = Depends(get_db_session),
) -> list[EntitySchema]:
    records = await crud.list_entities(session, entity_type=entity_type)
    return [EntitySchema.model_validate(record) for record in records]


@app.get("/relationships", response_model=list[RelationshipSchema], tags=["relationships"])
async def list_relationships(
    rel_type: Optional[str] = Query(default=None),
    session: AsyncSession = Depends(get_db_session),
) -> list[RelationshipSchema]:
    records = await crud.list_relationships(session, rel_type=rel_type)
    entity_index = {
        entity.id: entity
        for record in records
        for entity in (record.source_entity, record.target_entity)
        if entity is not None
    }
    response: list[RelationshipSchema] = []
    for record in records:
        source = entity_index.get(record.source_entity_id)
        target = entity_index.get(record.target_entity_id)
        if not source or not target:
            continue
        response.append(
            RelationshipSchema(
                id=record.id,
                rel_type=record.rel_type,
                context=record.context,
                source=RelationshipEndpoint(
                    entity_id=source.id,
                    text=source.text,
                    type=source.type,
                ),
                target=RelationshipEndpoint(
                    entity_id=target.id,
                    text=target.text,
                    type=target.type,
                ),
            )
        )
    return response


@app.get("/graph", response_model=GraphResponse, tags=["relationships"])
async def graph(session: AsyncSession = Depends(get_db_session)) -> GraphResponse:
    relationships = await crud.list_relationships(session)
    nodes: dict[int, GraphNode] = {}
    edges: list[GraphEdge] = []
    for rel in relationships:
        source = rel.source_entity
        target = rel.target_entity
        if not source or not target:
            continue
        nodes.setdefault(source.id, GraphNode(id=source.id, label=source.text, type=source.type))
        nodes.setdefault(target.id, GraphNode(id=target.id, label=target.text, type=target.type))
        edges.append(
            GraphEdge(
                id=rel.id,
                source=source.id,
                target=target.id,
                rel_type=rel.rel_type,
            )
        )
    return GraphResponse(nodes=list(nodes.values()), edges=edges)


@app.get(
    "/dejavu",
    response_model=list[DejaVuClusterSchema],
    tags=["analytics"],
)
async def deja_vu_detector(
    session: AsyncSession = Depends(get_db_session),
    limit: int = Query(5, ge=1, le=20),
    shingle_size: int = Query(12, ge=5, le=20),
    min_hits: int = Query(3, ge=2, le=10),
    distinct_feeds: bool = Query(True, description="Require snippets to appear across different feeds"),
) -> list[DejaVuClusterSchema]:
    """Identify repeated snippets across outlets and summarize the narrative."""

    records = await crud.list_article_contents(session, min_chars=shingle_size * 4, limit=500)
    if not records:
        return []

    articles = [
        ArticleMeta(
            article_id=article.id,
            feed=article.feed,
            title=article.title,
            text_fr=content.text_fr,
        )
        for article, content in records
    ]
    clusters = collect_reused_quotes(
        articles,
        shingle_size=shingle_size,
        min_hits=min_hits,
        require_distinct_feeds=distinct_feeds,
    )
    top_clusters = clusters[:limit]
    response: list[DejaVuClusterSchema] = []
    for cluster in top_clusters:
        summary = await summarizer.summarize(cluster.snippet, [occ.feed for occ in cluster.occurrences])
        response.append(
            DejaVuClusterSchema(
                fingerprint=cluster.fingerprint,
                summary=summary,
                snippet=cluster.snippet,
                count=cluster.count,
                outlets=sorted(cluster.feeds),
                occurrences=[
                    DejaVuOccurrenceSchema(
                        article_id=occ.article_id,
                        feed=occ.feed,
                        title=occ.title,
                        snippet=occ.snippet,
                    )
                    for occ in cluster.occurrences
                ],
            )
        )
    return response


async def process_payload(session: AsyncSession, payload: EntitiesRequest) -> EntitiesResponse:
    """Pipeline workflow invoked by API and scheduler."""

    lang = resolve_lang(payload.feed, payload.lang)
    context_text = payload.text or payload.title
    if not context_text:
        raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail="Missing text to process")

    article = await crud.get_or_create_article(
        session,
        feed=payload.feed,
        title=payload.title,
        url=str(payload.url) if payload.url else None,
    )

    await crud.upsert_article_content(session, article, text_fr=context_text)
    logger.debug("Article created/retrieved: id=%s", article.id)

    logger.debug("Running NER on %d chars of text", len(context_text))
    spans = await asyncio.to_thread(ner.extract, context_text)
    logger.debug("NER extracted %d entity spans", len(spans))

    entities_payload = await build_entities_payload(spans, lang)
    stored_entities = await crud.ensure_entities(session, article, entities_payload)
    logger.info("Stored %d unique entities for article %s", len(stored_entities), article.id)

    logger.debug("Classifying relationships for %d entities", len(stored_entities))
    relationships_payload = await build_relationships_payload(context_text, stored_entities)
    stored_relationships = await crud.replace_relationships(session, article, relationships_payload)
    logger.info("Stored %d relationships for article %s", len(stored_relationships), article.id)

    entity_map = {entity.id: entity for entity in stored_entities if entity.id}
    relationships_response = [
        RelationshipSchema(
            id=rel.id,
            rel_type=rel.rel_type,
            context=rel.context,
            source=RelationshipEndpoint(
                entity_id=rel.source_entity_id,
                text=entity_map[rel.source_entity_id].text,
                type=entity_map[rel.source_entity_id].type,
            ),
            target=RelationshipEndpoint(
                entity_id=rel.target_entity_id,
                text=entity_map[rel.target_entity_id].text,
                type=entity_map[rel.target_entity_id].type,
            ),
        )
        for rel in stored_relationships
        if rel.source_entity_id in entity_map and rel.target_entity_id in entity_map
    ]

    return EntitiesResponse(
        lang=lang,
        article_id=article.id,
        input=payload,
        entities=[EntitySchema.model_validate(entity) for entity in stored_entities],
        relationships=relationships_response,
    )


async def build_entities_payload(spans: list[EntitySpan], lang: str) -> list[dict]:
    if not spans:
        return []
    surface = [span.text for span in spans]
    translations = await asyncio.to_thread(translator.translate, surface, lang)
    payload: list[dict] = []
    seen_norms: set[str] = set()
    for idx, span in enumerate(spans):
        norm = normalize_text(span.text)
        if norm in seen_norms:
            continue
        seen_norms.add(norm)
        text_en = translations[idx] if idx < len(translations) else None
        payload.append(
            {
                "type": span.label,
                "text": span.text,
                "norm": norm,
                "text_en": text_en,
            }
        )
    return payload


async def build_relationships_payload(context: str, entities: list[Entity]) -> list[dict]:
    if not entities or not classifier.schema:
        return []
    payload: list[dict] = []
    seen_pairs: set[tuple[int, int, str]] = set()
    for definition in classifier.schema:
        sources = [entity for entity in entities if entity.type == definition.source_type]
        targets = [entity for entity in entities if entity.type == definition.target_type]
        for source in sources:
            for target in targets:
                if source.id == target.id:
                    continue
                if (
                    definition.source_type == definition.target_type
                    and source.id > target.id
                ):
                    continue
                # Only use labels compatible with this type pairing
                allowed = classifier.labels_for_types(source.type, target.type)
                if not allowed:
                    continue
                prompt = RelationshipPrompt(
                    source_text=source.text,
                    source_type=source.type,
                    target_text=target.text,
                    target_type=target.type,
                    context=context,
                )
                label = await classifier.classify(prompt, allowed)
                if label:
                    pair_key = (source.id, target.id, label)
                    if pair_key in seen_pairs:
                        continue
                    seen_pairs.add(pair_key)
                    payload.append(
                        {
                            "rel_type": label,
                            "source_entity_id": source.id,
                            "target_entity_id": target.id,
                            "context": context,
                        }
                    )
    return payload


def resolve_lang(feed: str, explicit: Optional[str]) -> str:
    if explicit:
        return explicit
    return FEED_LANG_MAP.get(feed, "fr")


async def ingest_feed(feed: str) -> None:
    """Scheduled job to pull feed entries and process them."""

    logger.info("Ingestion job started for feed: %s", feed)
    client = RSSClient(feed)
    try:
        entries = await client.fetch_entries()
    except Exception as exc:
        logger.error("Failed to fetch feed %s: %s", feed, exc, exc_info=True)
        return

    if not entries:
        logger.warning("No entries found for feed: %s", feed)
        return

    logger.info("Processing %d entries from feed: %s", len(entries), feed)
    success_count = 0
    error_count = 0

    async with async_session_factory() as session:
        for entry in entries:
            payload = EntitiesRequest(
                title=entry.title,
                url=entry.link,
                feed=feed,
                lang=resolve_lang(feed, None),
                text=entry.summary,
            )
            try:
                await process_payload(session, payload)
                success_count += 1
            except Exception as exc:
                error_count += 1
                logger.error(
                    "Failed to process entry %s: %s",
                    entry.link,
                    exc,
                    exc_info=True,
                )
                await session.rollback()
                continue
        await session.commit()

    logger.info(
        "Ingestion job complete for %s: %d succeeded, %d failed",
        feed,
        success_count,
        error_count,
    )



