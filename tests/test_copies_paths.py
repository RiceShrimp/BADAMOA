"""Where a version's images are: the folder they were registered from, in place."""

from __future__ import annotations

from test_training import hub, release  # noqa: F401  (the fixtures)


def test_a_version_names_the_folder_it_was_registered_from(hub, release, tmp_path):  # noqa: F811
    copies = hub.get(f"/api/releases/{release['id']}/copies").json()
    origin = copies["origin"]
    assert origin["node_name"] == "train-node"
    [root] = origin["roots"]
    assert root.replace("\\", "/").rstrip("/").endswith("/ds")  # registered in place: that folder
