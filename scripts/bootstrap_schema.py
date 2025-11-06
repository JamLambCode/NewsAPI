"""Generate a report to guide relationship schema evolution."""

from __future__ import annotations

import argparse
import asyncio

from sqlalchemy import func, select
from sqlalchemy.orm import aliased

from src.app.storage.db import async_session_factory
from src.app.storage.models import Entity, Relationship


async def relationship_report(limit: int) -> None:
    source_alias = aliased(Entity)
    target_alias = aliased(Entity)

    async with async_session_factory() as session:
        stats_stmt = (
            select(
                Relationship.rel_type,
                source_alias.type.label("source_type"),
                target_alias.type.label("target_type"),
                func.count().label("count"),
            )
            .join(source_alias, Relationship.source_entity)
            .join(target_alias, Relationship.target_entity)
            .group_by(Relationship.rel_type, source_alias.type, target_alias.type)
            .order_by(func.count().desc())
        )
        stats = list(await session.execute(stats_stmt))

        sample_stmt = (
            select(
                Relationship.rel_type,
                source_alias.text.label("source_text"),
                source_alias.type.label("source_type"),
                target_alias.text.label("target_text"),
                target_alias.type.label("target_type"),
                Relationship.context,
            )
            .join(source_alias, Relationship.source_entity)
            .join(target_alias, Relationship.target_entity)
            .order_by(Relationship.id.desc())
            .limit(limit)
        )
        samples = list(await session.execute(sample_stmt))

    if not stats:
        print("No relationships stored yet. Run ingestion first.")
        return

    print("Relationship distribution:")
    for row in stats:
        rel_type, source_type, target_type, count = row
        print(f"- {rel_type} ({source_type}->{target_type}): {count}")

    print("\nSample contexts for manual review:")
    for row in samples:
        rel_type, source_text, source_type, target_text, target_type, context = row
        print(
            f"[{rel_type}] {source_text} ({source_type}) -> {target_text} ({target_type})\n  Context: {context}\n"
        )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Summarise stored relationships")
    parser.add_argument("--limit", type=int, default=20, help="Number of sample relationship contexts")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    asyncio.run(relationship_report(args.limit))


if __name__ == "__main__":
    main()


