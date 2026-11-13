from __future__ import annotations

import badamoa
from badamoa.hub.release_notes import parse_release_notes

SAMPLE = """# 릴리즈 노트

머리말은 무시됩니다.

## [0.2.0] - 2026-11-13

요약 문장입니다.

### 추가
- 첫 항목
- 둘째 항목이
  다음 줄로 이어집니다

### 수정
- 고친 것

## [0.1.0] - 2026-11-01

### 추가
- 처음
"""


def test_parser_reads_versions_sections_and_wrapped_items():
    releases = parse_release_notes(SAMPLE)
    assert [(r["version"], r["date"]) for r in releases] == [("0.2.0", "2026-11-13"), ("0.1.0", "2026-11-01")]
    assert releases[0]["summary"] == "요약 문장입니다."
    assert releases[0]["sections"] == [
        {"title": "추가", "items": ["첫 항목", "둘째 항목이 다음 줄로 이어집니다"]},
        {"title": "수정", "items": ["고친 것"]},
    ]
    assert parse_release_notes("no releases here") == []


def test_changelog_has_an_entry_for_the_current_version(client, admin):
    """Bumping the version without writing release notes should fail the build."""
    assert client.get("/api/release-notes").status_code == 401
    body = client.get("/api/release-notes", headers=admin).json()
    assert body["version"] == badamoa.__version__
    latest = body["releases"][0]
    assert latest["version"] == badamoa.__version__
    assert latest["date"] and any(section["items"] for section in latest["sections"])
