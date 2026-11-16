"""Tasks belong to datasets (chosen when importing); short English names; a task a team names; no "unlabelled"."""

from __future__ import annotations

import pytest
from sqlalchemy import text

from badamoa.core import tasks
from badamoa.hub.catalog_models import Dataset, Item, LabelRevision, LabelSpace, Project, Release, ReleaseItem
from badamoa.hub.db import Database


def test_task_names():
    assert [tasks.normalize(t) for t in ("Det", "seg", "KPT", "cls", "pose", "", "자동")] == [
        "detect", "segment", "pose", "classify", "pose", "", ""
    ]  # fmt: skip
    assert tasks.normalize(" OCR ") == "OCR" and tasks.same("ocr", "OCR")
    assert [tasks.short(t) for t in ("detect", "segment", "pose", "classify", "OCR")] == ["Det", "Seg", "Kpt", "Cls", "OCR"]
    for bad in ("문자", "1st", "a b", "x" * 25):
        with pytest.raises(ValueError):
            tasks.normalize(bad)


def test_old_hub_data_moves_to_dataset_tasks_and_background(settings):
    """A Hub from before 0.34.0: the project held the task, and images without a label file were "unlabelled"."""
    db = Database(settings.database_url)
    db.create_all()
    with db.session() as s:
        project = Project(name="old", task="segment")
        s.add(project)
        s.flush()
        dataset = Dataset(project_id=project.id, name="street")
        space = LabelSpace(project_id=project.id, name="1cls", classes=[{"id": 0, "name": "car"}])
        s.add_all([dataset, space])
        s.flush()
        stats = {"items": 2, "states": {"labelled": 1, "background": 0, "unlabelled": 1}}
        release = Release(dataset_id=dataset.id, version="v1", label_space_id=space.id, status="ready", stats=stats)
        item = Item(project_id=project.id, sha256="a" * 64, file_name="a.jpg", width=1, height=1, bytes=1)
        s.add_all([release, item])
        s.flush()
        rev = LabelRevision(
            item_id=item.id, rev=1, label_space_id=space.id, source="import", state="unlabelled", boxes=[], digest="d"
        )
        s.add(rev)
        s.flush()
        s.add(ReleaseItem(release_id=release.id, item_id=item.id, split="train", revision_id=rev.id, state="unlabelled"))
        ids = (dataset.id, release.id)
    with db.engine.begin() as conn:  # the column did not exist yet
        conn.execute(text("ALTER TABLE datasets DROP COLUMN task"))

    db.create_all()
    with db.session() as s:
        assert s.get(Dataset, ids[0]).task == "segment"
        assert s.get(Release, ids[1]).stats["states"] == {"labelled": 1, "background": 1}
        assert set(s.scalars(select_states())) == {"background"}
        assert s.scalar(text("SELECT state FROM label_revisions")) == "background"


def select_states():
    return text("SELECT state FROM release_items")
