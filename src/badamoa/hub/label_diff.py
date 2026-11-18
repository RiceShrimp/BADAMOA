"""What changed between two sets of labels of one image: added, removed, moved and reclassed labels.

Labels are paired by overlap: the two boxes that overlap most (IoU 0.5 or more) are taken for the
same object, then the next two, and so on. A pair of different classes is "reclassed"; a pair of the
same class whose box, outline or keypoints differ is "moved"; what is left unpaired was removed (on
the old side) or added (on the new one). Class labels without a box (classification) compare by class.
The web's item view pairs them the same way (``web/src/labelDiff.ts``).
"""

from __future__ import annotations

from typing import Any

MATCH = 0.5
SAME = 1e-4
MAX_PAIRS = 250_000  # boxes × boxes worth pairing in one image; beyond, only what was added or removed


def _iou(a: list[float], b: list[float]) -> float:
    _, ax, ay, aw, ah = a[:5]
    _, bx, by, bw, bh = b[:5]
    w = min(ax + aw / 2, bx + bw / 2) - max(ax - aw / 2, bx - bw / 2)
    h = min(ay + ah / 2, by + bh / 2) - max(ay - ah / 2, by - bh / 2)
    if w <= 0 or h <= 0:
        return 0.0
    return w * h / (aw * ah + bw * bh - w * h)


def _same_shape(a: list[float], b: list[float]) -> bool:
    return len(a) == len(b) and all(abs(x - y) < SAME for x, y in zip(a[1:], b[1:], strict=True))


def diff(before: list[list[float]], after: list[list[float]]) -> dict[str, list[Any]]:
    """{"added": [new index], "removed": [old index], "moved": [(old, new)], "reclassed": [(old, new)]}."""
    out: dict[str, list[Any]] = {"added": [], "removed": [], "moved": [], "reclassed": []}
    boxed_before = [i for i, b in enumerate(before) if len(b) >= 5]
    boxed_after = [j for j, a in enumerate(after) if len(a) >= 5]
    used_before: set[int] = set()
    used_after: set[int] = set()
    if boxed_before and boxed_after and len(boxed_before) * len(boxed_after) <= MAX_PAIRS:
        pairs = sorted(
            ((_iou(before[i], after[j]), i, j) for i in boxed_before for j in boxed_after),
            key=lambda p: -p[0],
        )
        for overlap, i, j in pairs:
            if overlap < MATCH:
                break
            if i in used_before or j in used_after:
                continue
            used_before.add(i)
            used_after.add(j)
            if int(before[i][0]) != int(after[j][0]):
                out["reclassed"].append((i, j))
            elif not _same_shape(before[i], after[j]):
                out["moved"].append((i, j))
    # labels without a box (a class for the whole image): one class for another is reclassed
    bare_before = [i for i, b in enumerate(before) if 0 < len(b) < 5]
    bare_after = [j for j, a in enumerate(after) if 0 < len(a) < 5]
    for i in bare_before:
        same = next((j for j in bare_after if j not in used_after and int(after[j][0]) == int(before[i][0])), None)
        if same is not None:
            used_before.add(i)
            used_after.add(same)
    left_before = [i for i in bare_before if i not in used_before]
    left_after = [j for j in bare_after if j not in used_after]
    for i, j in zip(left_before, left_after, strict=False):
        out["reclassed"].append((i, j))
        used_before.add(i)
        used_after.add(j)
    out["removed"] = [i for i in range(len(before)) if i not in used_before]
    out["added"] = [j for j in range(len(after)) if j not in used_after]
    return out


def counts(changes: dict[str, list[Any]]) -> dict[str, int]:
    return {kind: len(changes[kind]) for kind in ("added", "removed", "moved", "reclassed")}
