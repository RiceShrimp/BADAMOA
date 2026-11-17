import { useEffect, useRef, useState } from "react";
import type { LabelClass } from "./catalogTypes";

/** A label as stored: [class, cx, cy, w, h, …what follows the box] in fractions of the image. */
export type Box = number[];

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

function handlePoint(b: Box, handle: Handle): [number, number] {
  const [x1, y1, x2, y2] = corners(b);
  const x = handle.includes("w") ? x1 : handle.includes("e") ? x2 : (x1 + x2) / 2;
  const y = handle.includes("n") ? y1 : handle.includes("s") ? y2 : (y1 + y2) / 2;
  return [x, y];
}

type Drag =
  | { kind: "move"; index: number; start: [number, number]; before: Box[] }
  | { kind: "resize"; index: number; handle: Handle; before: Box[] }
  | { kind: "draw"; start: [number, number]; before: Box[] };

/**
 * What lies over the image while fixing labels: picks a box with a click (the smallest one under the
 * pointer), moves and resizes it with its handles, and draws a new one when ``drawing`` is on.
 * Boxes themselves are drawn underneath by the viewer, so they look as they do when only looking.
 * ``geometry`` false (outlines, keypoints): boxes can only be picked, not moved or drawn.
 */
export function EditLayer({
  width,
  height,
  boxes,
  visible,
  selected,
  drawing,
  geometry,
  drawClass,
  scale,
  colorOf,
  onSelect,
  onChange,
  onCommit,
}: {
  width: number;
  height: number;
  boxes: Box[];
  /** Indices of the boxes that are shown (classes can be hidden). */
  visible: Set<number>;
  selected: number | null;
  drawing: boolean;
  geometry: boolean;
  drawClass: number;
  /** Image pixels per screen pixel, so handles stay the same size on screen. */
  scale: number;
  colorOf: (cls: number) => string;
  onSelect: (index: number | null) => void;
  /** While dragging: the boxes as they are now (not yet one step of undo). */
  onChange: (boxes: Box[]) => void;
  /** A drag ended and changed something: ``before`` is what one undo step brings back. */
  onCommit: (before: Box[]) => void;
}) {
  const svgRef = useRef<SVGSVGElement>(null);
  const [drag, setDrag] = useState<Drag | null>(null);
  const [hover, setHover] = useState<string>("");

  const point = (e: React.PointerEvent): [number, number] => {
    const svg = svgRef.current;
    if (!svg) return [0, 0];
    const rect = svg.getBoundingClientRect();
    return [clamp((e.clientX - rect.left) / rect.width), clamp((e.clientY - rect.top) / rect.height)];
  };
  const handleSize = 9 * scale; // in image pixels
  const near = (p: [number, number], q: [number, number]) =>
    Math.abs((p[0] - q[0]) * width) <= handleSize && Math.abs((p[1] - q[1]) * height) <= handleSize;

  /** What is under the pointer: a handle of the selected box, else the smallest box, else nothing. */
  const hit = (p: [number, number]): { index: number; handle?: Handle } | null => {
    if (selected != null && geometry && boxes[selected] && visible.has(selected)) {
      const handle = HANDLES.find((h) => near(p, handlePoint(boxes[selected], h)));
      if (handle) return { index: selected, handle };
    }
    let best = -1;
    let bestArea = Infinity;
    const pad = handleSize / 2 / width;
    for (let i = 0; i < boxes.length; i++) {
      const b = boxes[i];
      if (!visible.has(i) || b.length < 5) continue;
      const [x1, y1, x2, y2] = corners(b);
      const area = (x2 - x1) * (y2 - y1);
      if (p[0] >= x1 - pad && p[0] <= x2 + pad && p[1] >= y1 - pad && p[1] <= y2 + pad && area < bestArea) {
        best = i;
        bestArea = area;
      }
    }
    return best >= 0 ? { index: best } : null;
  };

  const down = (e: React.PointerEvent) => {
    if (e.button !== 0) return;
    const p = point(e);
    (e.target as Element).setPointerCapture?.(e.pointerId);
    if (drawing && geometry) {
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
    onSelect(target.index);
    if (!geometry) return;
    if (target.handle) setDrag({ kind: "resize", index: target.index, handle: target.handle, before: boxes });
    else setDrag({ kind: "move", index: target.index, start: p, before: boxes });
  };

  const move = (e: React.PointerEvent) => {
    const p = point(e);
    if (!drag) {
      const target = drawing && geometry ? null : hit(p);
      setHover(target?.handle ? CURSOR[target.handle] : target ? (geometry ? "move" : "pointer") : drawing && geometry ? "crosshair" : "default");
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
      next[drag.index] = [b[0], round(b[1] + dx), round(b[2] + dy), b[3], b[4], ...b.slice(5)];
    } else {
      const b = drag.before[drag.index];
      let [x1, y1, x2, y2] = corners(b);
      if (drag.handle.includes("w")) x1 = Math.min(p[0], x2 - MIN);
      if (drag.handle.includes("e")) x2 = Math.max(p[0], x1 + MIN);
      if (drag.handle.includes("n")) y1 = Math.min(p[1], y2 - MIN);
      if (drag.handle.includes("s")) y2 = Math.max(p[1], y1 + MIN);
      next[drag.index] = fromCorners(b[0], x1, y1, x2, y2, b.slice(5));
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
      onCommit(before);
      onSelect(boxes.length - 1);
      return;
    }
    if (JSON.stringify(before[drag.index]) !== JSON.stringify(boxes[drag.index])) onCommit(before);
  };

  const sel = selected != null && visible.has(selected) ? boxes[selected] : null;
  const stroke = 2 * scale;
  return (
    <svg
      ref={svgRef}
      className="edit-layer"
      viewBox={`0 0 ${width} ${height}`}
      preserveAspectRatio="none"
      style={{ cursor: drag ? (drag.kind === "draw" ? "crosshair" : hover || "move") : hover }}
      onPointerDown={down}
      onPointerMove={move}
      onPointerUp={up}
      onPointerCancel={up}
    >
      {sel && sel.length >= 5 && (
        <g>
          <rect
            x={(sel[1] - sel[3] / 2) * width}
            y={(sel[2] - sel[4] / 2) * height}
            width={sel[3] * width}
            height={sel[4] * height}
            fill={colorOf(sel[0])}
            fillOpacity={0.12}
            stroke="#fff"
            strokeWidth={stroke}
            strokeDasharray={`${6 * scale} ${4 * scale}`}
          />
          {geometry &&
            HANDLES.map((h) => {
              const [x, y] = handlePoint(sel, h);
              return (
                <rect
                  key={h}
                  x={x * width - handleSize / 2}
                  y={y * height - handleSize / 2}
                  width={handleSize}
                  height={handleSize}
                  fill="#fff"
                  stroke={colorOf(sel[0])}
                  strokeWidth={stroke}
                />
              );
            })}
        </g>
      )}
    </svg>
  );
}

/** The class list while fixing: pick the class of the selected box (or of the next one drawn); 1–9 for the first nine. */
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
  // The class of the box just picked may be far down a long list: bring it into view.
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
