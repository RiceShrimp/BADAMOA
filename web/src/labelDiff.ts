/**
 * What changed between two sets of labels of one image, paired the way the Hub pairs them
 * (``src/badamoa/hub/label_diff.py``): the two boxes that overlap most (IoU 0.5 or more) are the same
 * object; a pair of different classes is "reclassed", of the same class but another shape "moved";
 * the rest was removed (old side) or added (new side).
 */
export interface LabelChanges {
  added: number[]; // indices in the new labels
  removed: number[]; // indices in the old labels
  moved: [number, number][]; // (old, new)
  reclassed: [number, number][];
}

const MATCH = 0.5;
const SAME = 1e-4;

function iou(a: number[], b: number[]): number {
  const [, ax, ay, aw, ah] = a;
  const [, bx, by, bw, bh] = b;
  const w = Math.min(ax + aw / 2, bx + bw / 2) - Math.max(ax - aw / 2, bx - bw / 2);
  const h = Math.min(ay + ah / 2, by + bh / 2) - Math.max(ay - ah / 2, by - bh / 2);
  if (w <= 0 || h <= 0) return 0;
  return (w * h) / (aw * ah + bw * bh - w * h);
}

const sameShape = (a: number[], b: number[]) => a.length === b.length && a.every((v, i) => i === 0 || Math.abs(v - b[i]) < SAME);

export function diffLabels(before: number[][], after: number[][]): LabelChanges {
  const out: LabelChanges = { added: [], removed: [], moved: [], reclassed: [] };
  const usedBefore = new Set<number>();
  const usedAfter = new Set<number>();
  const pairs: [number, number, number][] = [];
  before.forEach((b, i) => {
    if (b.length < 5) return;
    after.forEach((a, j) => {
      if (a.length >= 5) pairs.push([iou(b, a), i, j]);
    });
  });
  pairs.sort((p, q) => q[0] - p[0]);
  for (const [overlap, i, j] of pairs) {
    if (overlap < MATCH) break;
    if (usedBefore.has(i) || usedAfter.has(j)) continue;
    usedBefore.add(i);
    usedAfter.add(j);
    if (before[i][0] !== after[j][0]) out.reclassed.push([i, j]);
    else if (!sameShape(before[i], after[j])) out.moved.push([i, j]);
  }
  // class labels without a box: one class for another is reclassed
  const bareBefore = before.map((b, i) => (b.length > 0 && b.length < 5 ? i : -1)).filter((i) => i >= 0);
  const bareAfter = after.map((a, j) => (a.length > 0 && a.length < 5 ? j : -1)).filter((j) => j >= 0);
  for (const i of bareBefore) {
    const j = bareAfter.find((k) => !usedAfter.has(k) && after[k][0] === before[i][0]);
    if (j !== undefined) {
      usedBefore.add(i);
      usedAfter.add(j);
    }
  }
  const leftBefore = bareBefore.filter((i) => !usedBefore.has(i));
  const leftAfter = bareAfter.filter((j) => !usedAfter.has(j));
  leftBefore.forEach((i, n) => {
    const j = leftAfter[n];
    if (j === undefined) return;
    out.reclassed.push([i, j]);
    usedBefore.add(i);
    usedAfter.add(j);
  });
  out.removed = before.map((_, i) => i).filter((i) => !usedBefore.has(i));
  out.added = after.map((_, j) => j).filter((j) => !usedAfter.has(j));
  return out;
}

export const CHANGE_LABEL: Record<"added" | "removed" | "moved" | "reclassed", string> = {
  added: "추가",
  removed: "지움",
  moved: "옮김",
  reclassed: "클래스 바뀜",
};
