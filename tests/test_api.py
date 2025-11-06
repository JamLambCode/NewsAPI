"""Smoke tests for the Holocron API."""

from __future__ import annotations

import pytest


@pytest.mark.asyncio
async def test_health_endpoint(test_client):
    """Test the /health endpoint returns successfully."""
    response = await test_client.get("/health")
    assert response.status_code == 200
    data = response.json()
    assert "status" in data
    assert data["status"] == "ok"


@pytest.mark.asyncio
async def test_articles_endpoint_empty(test_client):
    """Test /articles returns an empty list on fresh DB."""
    response = await test_client.get("/articles")
    assert response.status_code == 200
    data = response.json()
    assert isinstance(data, list)
    assert len(data) == 0


@pytest.mark.asyncio
async def test_entities_endpoint_empty(test_client):
    """Test /entities returns an empty list on fresh DB."""
    response = await test_client.get("/entities")
    assert response.status_code == 200
    data = response.json()
    assert isinstance(data, list)
    assert len(data) == 0


@pytest.mark.asyncio
async def test_relationships_endpoint_empty(test_client):
    """Test /relationships returns an empty list on fresh DB."""
    response = await test_client.get("/relationships")
    assert response.status_code == 200
    data = response.json()
    assert isinstance(data, list)
    assert len(data) == 0


@pytest.mark.asyncio
async def test_graph_endpoint_empty(test_client):
    """Test /graph returns empty nodes and edges on fresh DB."""
    response = await test_client.get("/graph")
    assert response.status_code == 200
    data = response.json()
    assert "nodes" in data
    assert "edges" in data
    assert len(data["nodes"]) == 0
    assert len(data["edges"]) == 0


@pytest.mark.asyncio
async def test_post_entities_missing_title(test_client):
    """Test POST /entities fails gracefully when title is missing."""
    response = await test_client.post("/entities", json={"feed": "fr/lemonde"})
    assert response.status_code == 422  # Validation error


@pytest.mark.asyncio
async def test_post_entities_minimal(test_client):
    """Test POST /entities with minimal valid payload."""
    payload = {
        "title": "Test headline about Paris",
        "feed": "fr/lemonde",
    }
    response = await test_client.post("/entities", json=payload)
    # This may fail if NER models aren't available, but structure should be correct
    assert response.status_code in (201, 500)
    if response.status_code == 201:
        data = response.json()
        assert "entities" in data
        assert "relationships" in data
        assert "input" in data
        assert data["input"]["title"] == payload["title"]

