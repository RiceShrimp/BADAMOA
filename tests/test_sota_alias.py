"""0.98.0: the alias "champion" is "SOTA". A Hub from before renames it once at start: on the model versions that
hold it and in the pipelines that name it (a gate's alias, a model to evaluate); a family that has a "SOTA" already
keeps its "champion"; after that, a "champion" someone gives is theirs to keep."""

from __future__ import annotations

from badamoa.hub.catalog_models import Project
from badamoa.hub.db import Database
from badamoa.hub.models import HubSetting
from badamoa.hub.train_models import ModelVersion, Pipeline, PipelineRun


def test_champion_becomes_sota_once(tmp_path):
    database = Database(f"sqlite:///{(tmp_path / 'hub.db').as_posix()}")
    database.create_all()  # a new Hub: nothing to rename, marked done
    with database.session() as s:
        project = Project(name="별칭")
        s.add(project)
        s.flush()
        def version(name, n, aliases):
            blob = str(n) * 64
            return ModelVersion(project_id=project.id, name=name, version=n, blob=blob, aliases=aliases, alias=aliases[0])

        made = {
            "kept": version("det", 1, ["champion", "production"]),
            "other": version("seg", 1, ["champion"]),
            "has_sota": version("seg", 2, ["SOTA"]),
        }
        s.add_all(made.values())
        steps = [
            {"kind": "train", "config": {}},
            {"kind": "eval", "config": {"model": {"name": "det", "alias": "champion"}}},
            {"kind": "promote", "config": {"alias": "champion"}},
        ]
        pipeline = Pipeline(project_id=project.id, dataset_id="d", name="p", steps=steps)
        held = [*steps[:2], {**steps[2], "refs": {"alias": "champion", "model_id": "m"}}]  # a run waiting at its gate
        run = PipelineRun(project_id=project.id, release_id="r", steps=held, status="held")
        s.add_all([pipeline, run])
        s.flush()
        ids = {key: v.id for key, v in made.items()}
        pipeline_id, run_id = pipeline.id, run.id
        s.delete(s.get(HubSetting, "migration.sota_alias"))  # as a Hub from before this version
    database._sota_alias()
    with database.session() as s:
        kept = s.get(ModelVersion, ids["kept"])
        assert (kept.aliases, kept.alias) == (["SOTA", "production"], "SOTA")
        assert s.get(ModelVersion, ids["other"]).aliases == ["champion"]  # its family has a SOTA already
        steps = s.get(Pipeline, pipeline_id).steps
        assert steps[1]["config"]["model"]["alias"] == "SOTA" and steps[2]["config"]["alias"] == "SOTA"
        gate = s.get(PipelineRun, run_id).steps[2]
        assert (gate["config"]["alias"], gate["refs"]) == ("SOTA", {"alias": "SOTA", "model_id": "m"})
        assert s.get(HubSetting, "migration.sota_alias").value["versions"] == 1
        s.get(ModelVersion, ids["kept"]).aliases = ["champion"]  # given again later: kept
    database._sota_alias()
    with database.session() as s:
        assert s.get(ModelVersion, ids["kept"]).aliases == ["champion"]
    database.engine.dispose()
