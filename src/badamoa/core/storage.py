"""Storage abstraction over fsspec.

Every byte BADAMOA stores (images, manifests, weights, engines) goes through a
``Storage``. Backends are chosen by URL scheme so the same code works for a local
disk, a NAS mount or S3:

    Storage("file:///var/lib/badamoa/storage")
    Storage("s3://bucket/prefix", key=..., secret=...)

Blobs are content-addressed (sha256) so they are immutable and safe to cache on
compute nodes.
"""

from __future__ import annotations

import hashlib
import posixpath
from collections.abc import Iterator
from pathlib import Path
from typing import IO, Any

import fsspec
from fsspec.core import url_to_fs
from fsspec.implementations.local import LocalFileSystem

_CHUNK = 1024 * 1024


class Storage:
    def __init__(self, root: str | Path, **options: Any) -> None:
        url = Path(root).resolve().as_uri() if isinstance(root, Path) else root
        if "://" not in url:  # plain filesystem path given as string
            url = Path(url).resolve().as_uri()
        self.url = url.rstrip("/")
        self.fs: fsspec.AbstractFileSystem
        self.fs, self._root = url_to_fs(self.url, **options)
        if isinstance(self.fs, LocalFileSystem):
            self.fs.makedirs(self._root, exist_ok=True)

    # ---- path helpers -------------------------------------------------
    def _path(self, key: str) -> str:
        key = key.lstrip("/")
        if ".." in key.split("/"):
            raise ValueError(f"invalid storage key: {key!r}")
        return posixpath.join(self._root, key)

    @staticmethod
    def blob_key(digest: str) -> str:
        return f"blobs/{digest[:2]}/{digest[2:4]}/{digest}"

    # ---- generic key/value ---------------------------------------------
    def exists(self, key: str) -> bool:
        return self.fs.exists(self._path(key))

    def open(self, key: str, mode: str = "rb") -> IO[bytes]:
        path = self._path(key)
        if "w" in mode:
            self.fs.makedirs(posixpath.dirname(path), exist_ok=True)
        return self.fs.open(path, mode)

    def put_bytes(self, key: str, data: bytes) -> None:
        with self.open(key, "wb") as f:
            f.write(data)

    def get_bytes(self, key: str) -> bytes:
        with self.open(key, "rb") as f:
            return f.read()

    def delete(self, key: str) -> None:
        self.fs.rm(self._path(key))

    def list(self, prefix: str = "") -> Iterator[str]:
        base = self._path(prefix) if prefix else self._root
        if not self.fs.exists(base):
            return
        root = self._root.rstrip("/") + "/"
        for path in self.fs.find(base):
            yield path[len(root) :] if path.startswith(root) else path

    # ---- content-addressed blobs ----------------------------------------
    def put_blob(self, stream: IO[bytes] | bytes) -> str:
        """Store bytes by sha256 digest. Returns the digest. Idempotent."""
        data = stream if isinstance(stream, bytes) else stream.read()
        digest = hashlib.sha256(data).hexdigest()
        key = self.blob_key(digest)
        if not self.exists(key):
            self.put_bytes(key, data)
        return digest

    def get_blob(self, digest: str) -> bytes:
        return self.get_bytes(self.blob_key(digest))

    def has_blob(self, digest: str) -> bool:
        return self.exists(self.blob_key(digest))


def sha256_file(path: str | Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(_CHUNK), b""):
            h.update(chunk)
    return h.hexdigest()
