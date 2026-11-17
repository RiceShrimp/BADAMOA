import { useEffect, useRef, useState } from "react";
import type { LabelClass } from "./catalogTypes";

/**
 * A label as stored: [class, cx, cy, w, h, …what follows the box] in fractions of the image.
 * Segmentation: the outline follows as x1, y1, x2, y2, … (the box is its bounds).
 * Pose: the keypoints follow as x, y, visibility (0 not labelled, 1 hidden, 2 seen) for each point.
 */
export type Box = number[];
export type Pt = [number, number];

type Handle = "nw" | "n" | "ne" | "e" | "se" | "s" | "sw" | "w";
const HANDLES: Handle[] = ["nw", "n", "ne", "e", "se", "s", "sw", "w"];
const CURSOR: Record<Handle, string> = {
  nw: "nwse-resize",
  se: "nwse-resize",
  ne: "nesw-resize",
  sw: "nesw-resize",
  n: "ns-resize",
  s: "ns-resize",
  e: "ew-resize",
  w: "ew-resize",
};
const MIN = 0.002; // the smallest side a box can be dragged to, as a share of the image

const clamp = (v: number, lo = 0, hi = 1) => Math.min(hi, Math.max(lo, v));
const round = (v: number) => Math.round(v * 1e5) / 1e5;

// ---- shapes ------------------------------------------------------------------------

/** x1, y1, x2, y2 of a box. */
function corners(b: Box): [number, number, number, number] {
  const [, cx, cy, w, h] = b;
  return [cx - w / 2, cy - h / 2, cx + w / 2, cy + h / 2];
}

function fromCorners(cls: number, x1: number, y1: number, x2: number, y2: number, rest: number[] = []): Box {
  const [a, b] = [clamp(Math.min(x1, x2)), clamp(Math.max(x1, x2))];
  const [c, d] = [clamp(Math.min(y1, y2)), clamp(Math.max(y1, y2))];
  return [cls, round((a + b) / 2), round((c + d) / 2), round(b - a), round(d - c), ...rest];
}

/** The outline of a segmentation label (empty when it has only a box). */
export function outlineOf(b: Box): Pt[] {
  const out: Pt[] = [];
  for (let i = 5; i + 1 < b.length; i += 2) out.push([b[i], b[i + 1]]);
  return out.length >= 3 ? out : [];
}

/** A segmentation label from its outline: the box is the outline's bounds. */
export function polygonShape(cls: number, points: Pt[]): Box {
  const xs = points.map((p) => clamp(p[0]));
  const ys = points.map((p) => clamp(p[1]));
  return fromCorners(cls, Math.min(...xs), Math.min(...ys), Math.max(...xs), Math.max(...ys), points.flatMap(([x, y]) => [round(clamp(x)), round(clamp(y))]));
}

/** The keypoints of a pose label: [x, y, visibility] each. */
export function keypointList(b: Box): [number, number, number][] {
  const out: [number, number, number][] = [];
  for (let i = 5; i + 2 < b.length; i += 3) out.push([b[i], b[i + 1], b[i + 2]]);
  return out;
}

export function withKeypoints(b: Box, points: [number, number, number][]): Box {
  return [...b.slice(0, 5), ...points.flatMap(([x, y, v]) => (v > 0 ? [round(clamp(x)), round(clamp(y)), v] : [0, 0, 0]))];
}

/** A new pose label: its box, and the class's keypoints still to be placed. */
export function poseShape(cls: number, box: Box, count: number): Box {
  return [cls, ...box.slice(1, 5), ...Array.from({ length: count * 3 }, () => 0)];
}

/** Move every point of a label (box, outline or keypoints) through ``f``; the box follows. */
function mapShape(b: Box, task: string, f: (p: Pt) => Pt): Box {
  if (task === "segment" && outlineOf(b).length) return polygonShape(b[0], outlineOf(b).map(f));
  const [x1, y1, x2, y2] = corners(b);
  const [a, c] = f([x1, y1]);
  const [d, e] = f([x2, y2]);
  const moved = fromCorners(b[0], a, c, d, e);
  if (task === "pose") return withKeypoints(moved, keypointList(b).map(([x, y, v]) => (v > 0 ? [...f([x, y]), v] : [0, 0, 0])));
  return [...moved, ...b.slice(5)];
}

/** Shift a label by (dx, dy), kept inside the image. */
export function moveShape(b: Box, task: string, dx: number, dy: number): Box {
  const [x1, y1, x2, y2] = corners(b);
  const mx = clamp(dx, -x1, 1 - x2);
  const my = clamp(dy, -y1, 1 - y2);
  return mapShape(b, task, ([x, y]) => [x + mx, y + my]);
}

/** The points a person can grab: outline corners or placed keypoints. */
function pointsOf(b: Box, task: string): (Pt | null)[] {
  if (task === "segment") return outlineOf(b);
  if (task === "pose") return keypointList(b).map(([x, y, v]) => (v > 0 ? [x, y] : null));
  return [];
}

function handlePoint(b: Box, handle: Handle): Pt {
  const [x1, y1, x2, y2] = corners(b);
  const x = handle.includes("w") ? x1 : handle.includes("e") ? x2 : (x1 + x2) / 2;
  const y = handle.includes("n") ? y1 : handle.includes("s") ? y2 : (y1 + y2) / 2;
  return [x, y];
}

function inside(p: Pt, polygon: Pt[]): boolean {
  let hit = false;
  for (let i = 0, j = polygon.length - 1; i < polygon.length; j = i++) {
    const [xi, yi] = polygon[i];
    const [xj, yj] = polygon[j];
    if (yi > p[1] !== yj > p[1] && p[0] < ((xj - xi) * (p[1] - yi)) / (yj - yi) + xi) hit = !hit;
  }
  return hit;
}

type Drag =
  | { kind: "move"; index: number; start: Pt; before: Box[] }
  | { kind: "resize"; index: number; handle: Handle; before: Box[] }
  | { kind: "point"; index: number; point: number; before: Box[] }
  | { kind: "draw"; start: Pt; before: Box[] };

/**
 * What lies over the image while fixing labels. Picks a label with a click (the smallest one under the
 * pointer), moves it, resizes it with the handles of its box, and drags its points: the corners of an
 * outline (segmentation) or the keypoints (pose). ``drawing``: a drag draws a box (pose: then its points
 * are placed one by one); for segmentation each click adds a corner and clicking the first one closes it.
 * Labels themselves are drawn underneath by the viewer, so they look as they do when only looking.
 */
export function EditLayer({
  width,
  height,
  task,
  boxes,
  visible,
  selected,
  point,
  drawing,
  sketch,
  placing,
  drawClass,
  keypointCount,
  scale,
  ignore,
  colorOf,
  onSelect,
  onChange,
  onCommit,
  onSketch,
  onClose,
  onDrawn,
  onPlaced,
}: {
  width: number;
  height: number;
  task: string;
  boxes: Box[];
  /** Indices of the labels that are shown (classes can be hidden). */
  visible: Set<number>;
  selected: number | null;
  /** The picked point of the selected label (outline corner or keypoint). */
  point: number | null;
  drawing: boolean;
  /** Corners of an outline being drawn (segmentation). */
  sketch: Pt[];
  /** Pose: the keypoint the next click places on the selected label. */
  placing: number | null;
  drawClass: number;
  /** Pose: how many keypoints a new label gets. */
  keypointCount: number;
  /** Image pixels per screen pixel, so handles stay the same size on screen whatever the zoom. */
  scale: number;
  /** A pan is under way (space or the middle button): leave the pointer alone. */
  ignore: boolean;
  colorOf: (cls: number) => string;
  onSelect: (index: number | null, point?: number | null) => void;
  /** While dragging: the labels as they are now (not yet one step of undo). */
  onChange: (boxes: Box[]) => void;
  /** A drag ended and changed something: ``before`` is what one undo step brings back. */
  onCommit: (before: Box[]) => void;
  onSketch: (points: Pt[]) => void;
  /** Close the outline being drawn. */
  onClose: () => void;
  /** A new label was drawn (it is the last one). */
  onDrawn: (index: number) => void;
  /** A keypoint was placed: the next one to place, or null when all are. */
  onPlaced: (next: number | null) => void;
}) {
  const svgRef = useRef<SVGSVGElement>(null);
  const [drag, setDrag] = useState<Drag | null>(null);
  const [hover, setHover] = useState<string>("");
  const [cursorAt, setCursorAt] = useState<Pt | null>(null);

  const toImage = (e: React.PointerEvent | React.MouseEvent): Pt => {
    const svg = svgRef.current;
    if (!svg) return [0, 0];
    const rect = svg.getBoundingClientRect();
    return [clamp((e.clientX - rect.left) / rect.width), clamp((e.clientY - rect.top) / rect.height)];
  };
  const handleSize = 9 * scale; // in image pixels
  const near = (p: Pt, q: Pt, radius = handleSize) =>
    Math.abs((p[0] - q[0]) * width) <= radius && Math.abs((p[1] - q[1]) * height) <= radius;
  const sel = selected != null && visible.has(selected) ? boxes[selected] : null;
  const boxDrawing = drawing && task !== "segment";
  const outlineDrawing = drawing && task === "segment";

  /** What is under the pointer: a point or a handle of the selected label, else the smallest label. */
  const hit = (p: Pt): { index: number; handle?: Handle; point?: number } | null => {
    if (sel && selected != null) {
      const pts = pointsOf(sel, task);
      const at = pts.findIndex((q) => q && near(p, q, handleSize * 0.8));
      if (at >= 0) return { index: selected, point: at };
      const handle = HANDLES.find((h) => near(p, handlePoint(sel, h)));
      if (handle) return { index: selected, handle };
      // On an edge of the picked outline it stays picked (a double click there adds a corner).
      const outline = task === "segment" ? outlineOf(sel) : [];
      if (outline.length && nearestEdge(p, outline).distance <= handleSize * 1.5) return { index: selected };
    }
    let best = -1;
    let bestArea = Infinity;
    const pad = handleSize / 2 / width;
    for (let i = 0; i < boxes.length; i++) {
      const b = boxes[i];
      if (!visible.has(i) || b.length < 5) continue;
      const [x1, y1, x2, y2] = corners(b);
      const outline = task === "segment" ? outlineOf(b) : [];
      const within = outline.length
        ? inside(p, outline)
        : p[0] >= x1 - pad && p[0] <= x2 + pad && p[1] >= y1 - pad && p[1] <= y2 + pad;
      const area = (x2 - x1) * (y2 - y1);
      if (within && area < bestArea) {
        best = i;
        bestArea = area;
      }
    }
    return best >= 0 ? { index: best } : null;
  };

  const down = (e: React.PointerEvent) => {
    if (e.button !== 0 || ignore) return;
    const p = toImage(e);
    // Pose: a keypoint still to place goes where the click is.
    if (placing != null && sel && selected != null) {
      const kps = keypointList(sel);
      if (placing < kps.length) {
        kps[placing] = [p[0], p[1], 2];
        const next = boxes.slice();
        next[selected] = withKeypoints(sel, kps);
        onChange(next);
        onCommit(boxes);
        const after = kps.findIndex(([, , v], i) => i > placing && v === 0);
        onPlaced(after >= 0 ? after : null);
        onSelect(selected, placing);
        return;
      }
    }
    if (outlineDrawing) {
      if (sketch.length >= 3 && near(p, sketch[0])) onClose();
      else onSketch([...sketch, p]);
      return;
    }
    (e.target as Element).setPointerCapture?.(e.pointerId);
    if (boxDrawing) {
      onSelect(null);
      setDrag({ kind: "draw", start: p, before: boxes });
      onChange([...boxes, fromCorners(drawClass, p[0], p[1], p[0], p[1])]);
      return;
    }
    const target = hit(p);
    if (!target) {
      onSelect(null);
      return;
    }
    if (target.point != null) {
      onSelect(target.index, target.point);
      setDrag({ kind: "point", index: target.index, point: target.point, before: boxes });
      return;
    }
    onSelect(target.index, null);
    if (target.handle) setDrag({ kind: "resize", index: target.index, handle: target.handle, before: boxes });
    else setDrag({ kind: "move", index: target.index, start: p, before: boxes });
  };

  const move = (e: React.PointerEvent) => {
    const p = toImage(e);
    if (outlineDrawing) setCursorAt(p);
    if (ignore) {
      setHover("grab");
      return;
    }
    if (!drag) {
      if (drawing || placing != null) {
        setHover(outlineDrawing && sketch.length >= 3 && near(p, sketch[0]) ? "pointer" : "crosshair");
        return;
      }
      const target = hit(p);
      setHover(target?.point != null ? "grab" : target?.handle ? CURSOR[target.handle] : target ? "move" : "default");
      return;
    }
    const next = drag.before.slice();
    if (drag.kind === "draw") {
      next.push(fromCorners(drawClass, drag.start[0], drag.start[1], p[0], p[1]));
    } else if (drag.kind === "move") {
      const b = drag.before[drag.index];
      const [x1, y1, x2, y2] = corners(b);
      const dx = clamp(p[0] - drag.start[0], -x1, 1 - x2);
      const dy = clamp(p[1] - drag.start[1], -y1, 1 - y2);
      next[drag.index] = mapShape(b, task, ([x, y]) => [x + dx, y + dy]);
    } else if (drag.kind === "point") {
      const b = drag.before[drag.index];
      if (task === "segment") {
        const pts = outlineOf(b);
        pts[drag.point] = p;
        next[drag.index] = polygonShape(b[0], pts);
      } else {
        const kps = keypointList(b);
        kps[drag.point] = [p[0], p[1], kps[drag.point][2] || 2];
        next[drag.index] = withKeypoints(b, kps);
      }
    } else {
      const b = drag.before[drag.index];
      const [x1, y1, x2, y2] = corners(b);
      let [a, c, d, f] = [x1, y1, x2, y2];
      if (drag.handle.includes("w")) a = Math.min(p[0], x2 - MIN);
      if (drag.handle.includes("e")) d = Math.max(p[0], x1 + MIN);
      if (drag.handle.includes("n")) c = Math.min(p[1], y2 - MIN);
      if (drag.handle.includes("s")) f = Math.max(p[1], y1 + MIN);
      const sx = (d - a) / Math.max(x2 - x1, 1e-9);
      const sy = (f - c) / Math.max(y2 - y1, 1e-9);
      next[drag.index] =
        task === "segment" || task === "pose"
          ? mapShape(b, task, ([x, y]) => [a + (x - x1) * sx, c + (y - y1) * sy])
          : fromCorners(b[0], a, c, d, f, b.slice(5));
    }
    onChange(next);
  };

  const up = () => {
    if (!drag) return;
    const before = drag.before;
    setDrag(null);
    if (drag.kind === "draw") {
      const made = boxes[boxes.length - 1];
      if (!made || made[3] * width < 4 || made[4] * height < 4) {
        onChange(before); // a click, not a box
        return;
      }
      const index = boxes.length - 1;
      if (task === "pose" && keypointCount > 0) {
        const next = boxes.slice();
        next[index] = poseShape(made[0], made, keypointCount);
        onChange(next);
      }
      onCommit(before);
      onDrawn(index);
      return;
    }
    if (JSON.stringify(before[drag.index]) !== JSON.stringify(boxes[drag.index])) onCommit(before);
  };

  /** The edge of an outline nearest to p (in image pixels): it runs from corner ``index`` to the next. */
  const nearestEdge = (p: Pt, pts: Pt[]) => {
    let index = -1;
    let distance = Infinity;
    const [px, py] = [p[0] * width, p[1] * height];
    for (let i = 0; i < pts.length; i++) {
      const [ax, ay] = [pts[i][0] * width, pts[i][1] * height];
      const [bx, by] = [pts[(i + 1) % pts.length][0] * width, pts[(i + 1) % pts.length][1] * height];
      const t = clamp(((px - ax) * (bx - ax) + (py - ay) * (by - ay)) / Math.max((bx - ax) ** 2 + (by - ay) ** 2, 1e-9));
      const d = Math.hypot(px - (ax + t * (bx - ax)), py - (ay + t * (by - ay)));
      if (d < distance) {
        index = i;
        distance = d;
      }
    }
    return { index, distance };
  };

  /** Segmentation: a double click on an edge of the selected outline adds a corner there. */
  const doubleClick = (e: React.MouseEvent) => {
    if (task !== "segment" || drawing || !sel || selected == null) return;
    const p = toImage(e);
    const pts = outlineOf(sel);
    const { index: best, distance: bestDistance } = nearestEdge(p, pts);
    if (best < 0 || bestDistance > handleSize * 1.5) return;
    pts.splice(best + 1, 0, p);
    const next = boxes.slice();
    next[selected] = polygonShape(sel[0], pts);
    onChange(next);
    onCommit(boxes);
    onSelect(selected, best + 1);
  };

  const stroke = 2 * scale;
  const dot = 5 * scale;
  const color = sel ? colorOf(sel[0]) : colorOf(drawClass);
  return (
    <svg
      ref={svgRef}
      className="edit-layer"
      viewBox={`0 0 ${width} ${height}`}
      preserveAspectRatio="none"
      style={{ cursor: drag ? (drag.kind === "draw" ? "crosshair" : drag.kind === "point" ? "grabbing" : hover || "move") : hover }}
      onPointerDown={down}
      onPointerMove={move}
      onPointerUp={up}
      onPointerCancel={up}
      onPointerLeave={() => setCursorAt(null)}
      onDoubleClick={doubleClick}
    >
      {sel && sel.length >= 5 && (
        <g>
          <rect
            x={(sel[1] - sel[3] / 2) * width}
            y={(sel[2] - sel[4] / 2) * height}
            width={sel[3] * width}
            height={sel[4] * height}
            fill={color}
            fillOpacity={task === "segment" ? 0.04 : 0.12}
            stroke="#fff"
            strokeWidth={stroke}
            strokeDasharray={`${6 * scale} ${4 * scale}`}
          />
          {HANDLES.map((h) => {
            const [x, y] = handlePoint(sel, h);
            return (
              <rect
                key={h}
                x={x * width - handleSize / 2}
                y={y * height - handleSize / 2}
                width={handleSize}
                height={handleSize}
                fill="#fff"
                stroke={color}
                strokeWidth={stroke}
              />
            );
          })}
          {pointsOf(sel, task).map((q, i) =>
            q ? (
              <circle
                key={`p${i}`}
                cx={q[0] * width}
                cy={q[1] * height}
                r={i === point ? dot * 1.4 : dot}
                fill={i === point ? "#fff" : color}
                stroke={i === point ? color : "#fff"}
                strokeWidth={stroke}
              />
            ) : null,
          )}
        </g>
      )}
      {outlineDrawing && sketch.length > 0 && (
        <g>
          <polyline
            points={[...sketch, ...(cursorAt ? [cursorAt] : [])].map(([x, y]) => `${x * width},${y * height}`).join(" ")}
            fill={color}
            fillOpacity={0.15}
            stroke={color}
            strokeWidth={stroke}
          />
          {sketch.map(([x, y], i) => (
            <circle
              key={i}
              cx={x * width}
              cy={y * height}
              r={i === 0 && sketch.length >= 3 ? dot * 1.5 : dot}
              fill={i === 0 ? "#fff" : color}
              stroke={i === 0 ? color : "#fff"}
              strokeWidth={stroke}
            />
          ))}
        </g>
      )}
    </svg>
  );
}

/** The class list while fixing: pick the class of the selected label (or of the next one drawn); 1–9 for the first nine. */
export function ClassPicker({
  classes,
  current,
  colorOf,
  onPick,
}: {
  classes: LabelClass[];
  current: number | null;
  colorOf: (id: number) => string;
  onPick: (id: number) => void;
}) {
  const [query, setQuery] = useState("");
  const listRef = useRef<HTMLDivElement>(null);
  // The class of the label just picked may be far down a long list: bring it into view.
  useEffect(() => {
    listRef.current?.querySelector<HTMLElement>('[aria-selected="true"]')?.scrollIntoView({ block: "nearest" });
  }, [current]);
  const shown = query.trim() ? classes.filter((c) => c.name.toLowerCase().includes(query.trim().toLowerCase())) : classes;
  return (
    <div className="class-picker">
      {classes.length > 12 && (
        <input value={query} onChange={(e) => setQuery(e.target.value)} placeholder="클래스 찾기…" aria-label="클래스 찾기" />
      )}
      <div className="class-picker-list" role="listbox" aria-label="클래스" ref={listRef}>
        {shown.map((c) => {
          const n = classes.indexOf(c);
          return (
            <button
              key={c.id}
              role="option"
              aria-selected={current === c.id}
              className={current === c.id ? "active" : ""}
              onClick={() => onPick(c.id)}
            >
              <i style={{ background: colorOf(c.id) }} />
              <span>{c.name}</span>
              {n < 9 && <kbd>{n + 1}</kbd>}
            </button>
          );
        })}
        {shown.length === 0 && <p className="muted small">맞는 클래스가 없습니다.</p>}
      </div>
    </div>
  );
}

const SEEN_LABEL = ["없음", "가려짐", "보임"];

/** Pose: the keypoints of the selected label — which are placed, hidden or missing, and placing one again. */
export function KeypointList({
  names,
  points,
  picked,
  placing,
  colorOf,
  onPick,
  onSeen,
  onPlace,
}: {
  names: string[];
  points: [number, number, number][];
  picked: number | null;
  placing: number | null;
  colorOf: string;
  onPick: (index: number) => void;
  onSeen: (index: number, seen: number) => void;
  onPlace: (index: number) => void;
}) {
  return (
    <div className="keypoint-list">
      {points.map(([, , seen], i) => (
        <div key={i} className={`keypoint-row ${i === picked ? "active" : ""} ${i === placing ? "placing" : ""}`}>
          <button className="keypoint-name" onClick={() => (seen > 0 ? onPick(i) : onPlace(i))} title={seen > 0 ? "이 점 고르기" : "눌러서 이미지에 놓기"}>
            <i style={{ background: seen > 0 ? colorOf : "transparent", borderColor: colorOf }} />
            {names[i] ?? `점 ${i + 1}`}
          </button>
          {i === placing ? (
            <span className="small placing-tag">이미지를 누르세요</span>
          ) : (
            <div className="seen-switch" role="group" aria-label={`${names[i] ?? i + 1} 보임 상태`}>
              {[2, 1, 0].map((v) => (
                <button
                  key={v}
                  className={seen === v ? "active" : ""}
                  aria-pressed={seen === v}
                  onClick={() => (v > 0 && seen === 0 ? onPlace(i) : onSeen(i, v))}
                  title={v > 0 && seen === 0 ? "눌러서 이미지에 놓기" : undefined}
                >
                  {SEEN_LABEL[v]}
                </button>
              ))}
            </div>
          )}
        </div>
      ))}
    </div>
  );
}
