"""Pydantic schema definitions."""

from datetime import datetime
from typing import List, Optional

from pydantic import BaseModel, HttpUrl


class EntitiesRequest(BaseModel):
    title: str
    url: Optional[HttpUrl] = None
    feed: str = "fr/lemonde"
    lang: Optional[str] = None
    text: Optional[str] = None


class Entity(BaseModel):
    id: Optional[int] = None
    type: str
    text: str
    text_en: Optional[str] = None
    norm: Optional[str] = None

    class Config:
        from_attributes = True


class RelationshipEndpoint(BaseModel):
    entity_id: int
    text: str
    type: str

    class Config:
        from_attributes = True


class Relationship(BaseModel):
    id: Optional[int] = None
    rel_type: str
    source: RelationshipEndpoint
    target: RelationshipEndpoint
    context: Optional[str] = None

    class Config:
        from_attributes = True


class EntitiesResponse(BaseModel):
    lang: str
    article_id: int
    input: EntitiesRequest
    entities: List[Entity]
    relationships: List[Relationship]


class Article(BaseModel):
    id: int
    feed: str
    title: str
    url: Optional[HttpUrl] = None
    published_at: Optional[datetime] = None
    ingested_at: Optional[datetime] = None

    class Config:
        from_attributes = True


class GraphNode(BaseModel):
    id: int
    label: str
    type: str


class GraphEdge(BaseModel):
    id: int
    source: int
    target: int
    rel_type: str


class GraphResponse(BaseModel):
    nodes: List[GraphNode]
    edges: List[GraphEdge]



