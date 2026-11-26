"""Deleting images finds the rows that point at them by index: a stopped import of many images does not hold the
database while each image's rows are looked for by reading whole tables."""

from __future__ import annotations

from sqlalchemy import text

from badamoa.hub.db import _ADDED_INDEXES, Database

ITEM_TABLES = ("item_predictions", "label_set_items", "qa_results", "qa_scores")


def _searched_by_item(conn, table: str) -> bool:
    plan = conn.execute(text(f"EXPLAIN QUERY PLAN SELECT 1 FROM {table} WHERE item_id = 1")).all()
    return any("(item_id=?)" in str(row[-1]) for row in plan)


def _indexes(conn, table: str) -> set[str]:
    return {row[1] for row in conn.execute(text(f"PRAGMA index_list('{table}')")).all()}


def test_rows_of_an_image_are_found_by_index_also_on_an_older_hub(settings):
    db = Database(settings.database_url)
    db.create_all()
    with db.engine.begin() as conn:
        assert all(_searched_by_item(conn, table) for table in ITEM_TABLES)  # a new Hub has them
        for name, _, _ in _ADDED_INDEXES[1:]:  # a Hub from before 0.120.8 had none of these
            conn.execute(text(f"DROP INDEX {name}"))
    with db.engine.begin() as conn:
        assert "ix_qa_scores_item" not in _indexes(conn, "qa_scores")

    db.create_all()  # the Hub started again on the new version
    with db.engine.begin() as conn:
        assert "ix_qa_scores_item" in _indexes(conn, "qa_scores")
        assert all(_searched_by_item(conn, table) for table in ITEM_TABLES)
