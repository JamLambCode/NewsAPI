"""Unit tests for relationship schema loading."""

from __future__ import annotations

from pathlib import Path

from src.app.llm.relationships import load_relationships


def test_load_relationships_from_yaml():
    """Test loading relationships from YAML file."""
    relationships = load_relationships("relationships.yaml")
    assert len(relationships) >= 3
    assert any(rel.name == "works_for" for rel in relationships)
    assert any(rel.name == "met_with" for rel in relationships)


def test_load_relationships_missing_file():
    """Test loading relationships from non-existent file returns empty list."""
    relationships = load_relationships("nonexistent.yaml")
    assert len(relationships) == 0


def test_relationship_schema_structure():
    """Test relationship definitions have required fields."""
    relationships = load_relationships("relationships.yaml")
    if relationships:
        rel = relationships[0]
        assert hasattr(rel, "name")
        assert hasattr(rel, "source_type")
        assert hasattr(rel, "target_type")
        assert hasattr(rel, "description")

