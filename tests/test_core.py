from __future__ import annotations

import hashlib

import pytest

from badamoa.core import sysinfo
from badamoa.core.jobs import validate_params
from badamoa.core.storage import Storage


def test_storage_blobs_are_content_addressed(tmp_path):
    storage = Storage(tmp_path / "store")
    digest = storage.put_blob(b"hello")
    assert digest == hashlib.sha256(b"hello").hexdigest()
    assert storage.put_blob(b"hello") == digest  # idempotent
    assert storage.has_blob(digest) and storage.get_blob(digest) == b"hello"
    assert list(storage.list("blobs")) == [Storage.blob_key(digest)]


def test_storage_keys_cannot_escape_root(tmp_path):
    storage = Storage(tmp_path / "store")
    storage.put_bytes("a/b.txt", b"x")
    assert storage.get_bytes("a/b.txt") == b"x"
    with pytest.raises(ValueError):
        storage.put_bytes("../outside.txt", b"x")


def test_validate_params_applies_defaults_and_bounds():
    assert validate_params("debug.sleep", {}) == {"seconds": 10, "message": "tick", "fail": False}
    with pytest.raises(ValueError):
        validate_params("debug.sleep", {"seconds": 99999})
    with pytest.raises(ValueError):
        validate_params("debug.sleep", {"bogus": 1})
    with pytest.raises(ValueError):
        validate_params("unknown.type", {})


def test_sysinfo_reports_this_machine():
    info = sysinfo.static_info()
    assert info["hostname"] and info["cpu_count"] >= 1 and info["memory_total_mb"] > 0
    assert info["os"] in {"windows", "linux", "darwin"}
    assert sysinfo.metrics()["disk_total_gb"] > 0
