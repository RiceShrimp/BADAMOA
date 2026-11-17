import { useEffect, useMemo, useRef, useState } from "react";
import { ApiError, api } from "./api";
import {
  type ItemDetail,
  type ItemPage,
  type LabelClass,
  type Revision,
  SPLIT_LABEL,
  STATE_LABEL,
  bytesLabel,
  classColor,
  keypointsOf,
} from "./catalogTypes";
import { type Box, ClassPicker, EditLayer } from "./LabelEditor";
import { Icon } from "./ui";

interface Props {
  releaseId: string;
  /** Query string of the grid's filters, so "next" follows the same list. */
  filterQuery: string;
  start: number;
  total: number;
  onClose: () => void;
  /** Colour of a class: the one picked in the classes tab, or the automatic one. */
  colorOf?: (id: number) => string;
  /** This version's classes: labels of other versions are matched to them by name. */
  classes?: LabelClass[];
  /** The project's task: what follows a box (an outline for segmentation, keypoints for pose). */
  task?: string;
  /** Labels were saved into the work in progress: the screen behind looks again. */
  onSaved?: () => void;
}

interface Saved {
  release_id: string;
  revision_id: number;
  state: string;
  box_count: number;
  edited: number;
}

/** Where a label revision came from, in words: "import:coco v1" → "가져오기 · coco v1". */
function sourceLabel(source: string): string {
  const [kind, ...rest] = source.split(":");
  const what = rest.join(":");
  const words: Record<string, string> = { import: "가져오기", classes: "클래스 변경", human: "사람이 고침", autolabel: "자동 라벨" };
  return words[kind] ? (what ? `${words[kind]} · ${what}` : words[kind]) : source;
}

const copy = (boxes: Box[]) => boxes.map((b) => [...b]);

/** Full-screen item view: boxes over the image, label revisions (one or two side by side), where else the image is used.
 * "라벨 고치기" (E) turns the same screen into an editor; what is saved lands in the dataset's work in progress. */
export function ItemViewer({ releaseId, filterQuery, start, total, onClose, colorOf = classColor, classes = [], task = "detect", onSaved }: Props) {
  const [index, setIndex] = useState(start);
  const [item, setItem] = useState<ItemDetail | null>(null);
  const [error, setError] = useState<string | null>(null);
  const [revisionId, setRevisionId] = useState<number | null>(null);
  const [compareId, setCompareId] = useState<number | null>(null);
  const [showLabels, setShowLabels] = useState(true);
  const [showNames, setShowNames] = useState(true);
  const [fill, setFill] = useState(false);
  const [onlyHere, setOnlyHere] = useState(false);
  const [hidden, setHidden] = useState<Set<string>>(new Set()); // class names
  const [full, setFull] = useState<"loading" | "ok" | "error">("loading"); // the original image
  const stageRef = useRef<HTMLDivElement>(null);
  const [stage, setStage] = useState({ w: 0, h: 0 });

  // ---- fixing labels
  const [editing, setEditing] = useState(false);
  const editingRef = useRef(false);
  editingRef.current = editing;
  const [boxes, setBoxes] = useState<Box[]>([]);
  const [baseline, setBaseline] = useState("[]"); // the labels the fix started from, to tell unsaved changes
  const [undo, setUndo] = useState<Box[][]>([]);
  const [redo, setRedo] = useState<Box[][]>([]);
  const [selected, setSelected] = useState<number | null>(null);
  const [drawing, setDrawing] = useState(false);
  const [drawClass, setDrawClass] = useState<number>(classes[0]?.id ?? 0);
  const [saving, setSaving] = useState(false);
  const [note, setNote] = useState<{ kind: "saved" | "error" | "conflict"; text: string } | null>(null);
  const clip = useRef<Box | null>(null);
  const dirty = editing && JSON.stringify(boxes) !== baseline;
  const geometry = task === "detect" || !["segment", "pose", "classify"].includes(task); // boxes only: move, resize, draw

  useEffect(() => {
    const el = stageRef.current;
    if (!el) return;
    const measure = () => setStage({ w: el.clientWidth - 32, h: el.clientHeight - 84 }); // padding and nav bar
    measure();
    const observer = new ResizeObserver(measure);
    observer.observe(el);
    return () => observer.disconnect();
  }, []);

  /** Begin fixing this image from the labels of the work in progress (or this version's). */
  const startFrom = (detail: ItemDetail) => {
    const rev = detail.revisions.find((r) => r.id === detail.edit.revision_id);
    const begin = copy(rev?.boxes ?? []);
    setBoxes(begin);
    setBaseline(JSON.stringify(begin));
    setUndo([]);
    setRedo([]);
    setSelected(null);
    if (rev) setRevisionId(rev.id);
    setCompareId(null);
  };

  const load = async (itemId: number) => {
    const detail = await api<ItemDetail>(`/releases/${releaseId}/items/${itemId}`);
    setItem(detail);
    return detail;
  };

  useEffect(() => {
    let alive = true;
    setError(null);
    void (async () => {
      try {
        const page = await api<ItemPage>(`/releases/${releaseId}/items?${filterQuery}&offset=${index}&limit=1`);
        if (!alive) return;
        if (page.items.length === 0) return setError("항목을 찾을 수 없습니다.");
        const detail = await api<ItemDetail>(`/releases/${releaseId}/items/${page.items[0].id}`);
        if (!alive) return;
        setItem(detail);
        setFull(detail.image_url ? "loading" : "error");
        setRevisionId(detail.revision_id);
        setCompareId(null);
        setNote(null);
        if (editingRef.current) {
          if (detail.edit.allowed) startFrom(detail);
          else setEditing(false);
        }
      } catch (e) {
        if (alive) setError(e instanceof Error ? e.message : String(e));
      }
    })();
    return () => {
      alive = false;
    };
  }, [releaseId, filterQuery, index]);

  // ---- editing steps (each one an undo step)
  const commit = (before: Box[]) => {
    setUndo((u) => [...u.slice(-99), before]);
    setRedo([]);
  };
  const apply = (next: Box[]) => {
    commit(boxes);
    setBoxes(next);
  };
  const undoStep = () => {
    if (!undo.length) return;
    setRedo((r) => [...r, boxes]);
    setBoxes(undo[undo.length - 1]);
    setUndo((u) => u.slice(0, -1));
    setSelected(null);
  };
  const redoStep = () => {
    if (!redo.length) return;
    setUndo((u) => [...u, boxes]);
    setBoxes(redo[redo.length - 1]);
    setRedo((r) => r.slice(0, -1));
    setSelected(null);
  };
  const removeSelected = () => {
    if (selected == null) return;
    apply(boxes.filter((_, i) => i !== selected));
    setSelected(null);
  };
  const pickClass = (id: number) => {
    setDrawClass(id);
    if (task === "classify") {
      if (boxes[0]?.[0] !== id) apply([[id]]);
    } else if (selected != null && boxes[selected] && boxes[selected][0] !== id) {
      apply(boxes.map((b, i) => (i === selected ? [id, ...b.slice(1)] : b)));
    }
  };
  const toBackground = () => {
    if (!boxes.length) return;
    apply([]);
    setSelected(null);
  };
  const paste = () => {
    const b = clip.current;
    if (!b || !geometry) return;
    const shift = (v: number, size: number) => Math.min(1 - size / 2, v + 0.02);
    apply([...boxes, [b[0], shift(b[1], b[3]), shift(b[2], b[4]), b[3], b[4]]]);
    setSelected(boxes.length);
  };

  const beginEdit = () => {
    if (!item?.edit.allowed) return;
    startFrom(item);
    setNote(null);
    setEditing(true);
  };
  const stopEdit = () => {
    if (dirty && !window.confirm("저장하지 않은 고친 내용이 있습니다. 버리고 그만 고칠까요?")) return;
    setEditing(false);
    setDrawing(false);
    setSelected(null);
    if (item) setRevisionId(item.revision_id);
  };

  /** Save this image's labels into the work in progress. ``force``: over a fix someone made in between. */
  const save = async (force = false): Promise<boolean> => {
    if (!item) return false;
    setSaving(true);
    setNote(null);
    try {
      const body = { boxes, parent_revision_id: item.edit.revision_id, force };
      const saved = await api<Saved>(`/releases/${releaseId}/items/${item.id}/labels`, { method: "PUT", body });
      const detail = await load(item.id);
      startFrom(detail);
      setRevisionId(saved.revision_id);
      setNote({ kind: "saved", text: `작업 중에 저장했습니다 · 고친 이미지 ${saved.edited.toLocaleString()}장` });
      onSaved?.();
      return true;
    } catch (e) {
      const text = e instanceof Error ? e.message : String(e);
      setNote({ kind: e instanceof ApiError && e.status === 412 ? "conflict" : "error", text });
      return false;
    } finally {
      setSaving(false);
    }
  };
  const lookAtTheirs = async () => {
    if (!item) return;
    startFrom(await load(item.id));
    setNote(null);
  };
  const revertImage = async () => {
    if (!item?.edit.draft_id) return;
    if (!window.confirm("이 이미지의 라벨을 작업 중의 바탕 버전 라벨로 되돌릴까요?")) return;
    try {
      await api(`/releases/${item.edit.draft_id}/items/${item.id}/revert`, { method: "POST" });
      const detail = await load(item.id);
      if (editing) startFrom(detail);
      else setRevisionId(detail.revision_id);
      setNote({ kind: "saved", text: "바탕 버전의 라벨로 되돌렸습니다." });
      onSaved?.();
    } catch (e) {
      setNote({ kind: "error", text: e instanceof Error ? e.message : String(e) });
    }
  };

  const go = async (to: number) => {
    if (to < 0 || to >= total || to === index) return;
    if (dirty && !(await save())) return;
    setIndex(to);
  };
  const close = () => {
    if (dirty && !window.confirm("저장하지 않은 고친 내용이 있습니다. 버리고 닫을까요?")) return;
    onClose();
  };

  // One listener; it always calls the newest handler (the editor's state changes on every step).
  const keyRef = useRef<(e: KeyboardEvent) => void>(() => {});
  keyRef.current = (e: KeyboardEvent) => {
    if (e.target instanceof HTMLInputElement || e.target instanceof HTMLTextAreaElement || e.target instanceof HTMLSelectElement) return;
    const key = e.key.toLowerCase();
    const mod = e.ctrlKey || e.metaKey;
    if (editing) {
      if (mod && key === "s") {
        e.preventDefault();
        void save();
        return;
      }
      if (mod && (key === "z" || key === "y")) {
        e.preventDefault();
        if (key === "y" || e.shiftKey) redoStep();
        else undoStep();
        return;
      }
      if (mod && key === "c") {
        if (selected != null) clip.current = boxes[selected];
        return;
      }
      if (mod && key === "v") {
        e.preventDefault();
        paste();
        return;
      }
      if (e.key === "Delete" || e.key === "Backspace") {
        e.preventDefault();
        removeSelected();
        return;
      }
      if (key === "b" && geometry) {
        setDrawing((d) => !d);
        setSelected(null);
        return;
      }
      if (key === "v") {
        setDrawing(false);
        return;
      }
      if (/^[1-9]$/.test(e.key)) {
        const c = classes[Number(e.key) - 1];
        if (c) pickClass(c.id);
        return;
      }
      if (e.key === "Escape") {
        e.preventDefault();
        if (drawing) setDrawing(false);
        else if (selected != null) setSelected(null);
        else stopEdit();
        return;
      }
    }
    if (mod) return;
    if (e.key === "Escape") close();
    else if (e.key === "ArrowRight") void go(index + 1);
    else if (e.key === "ArrowLeft") void go(index - 1);
    else if (key === "e" && !editing) beginEdit();
    else if (key === "l") setShowLabels((v) => !v);
    else if (key === "n") setShowNames((v) => !v);
    else if (key === "f") setFill((v) => !v);
  };
  useEffect(() => {
    const onKey = (e: KeyboardEvent) => keyRef.current(e);
    window.addEventListener("keydown", onKey);
    return () => window.removeEventListener("keydown", onKey);
  }, []);
  // Leaving the page with unsaved fixes asks first.
  useEffect(() => {
    if (!dirty) return;
    const warn = (e: BeforeUnloadEvent) => e.preventDefault();
    window.addEventListener("beforeunload", warn);
    return () => window.removeEventListener("beforeunload", warn);
  }, [dirty]);

  const revision = item?.revisions.find((r) => r.id === revisionId) ?? null;
  const compare = editing ? null : (item?.revisions.find((r) => r.id === compareId && r.id !== revisionId) ?? null);
  const currentRev = item?.revisions.find((r) => r.id === item.revision_id)?.rev ?? null;
  // A label revision goes by the versions that carry it; the bare revision number means nothing to people.
  const versionsOf = (rev: number) => {
    const here = item?.releases.find((r) => r.current)?.dataset;
    return (item?.releases ?? []).filter((r) => r.rev === rev).map((r) => (r.dataset === here ? r.version : `${r.dataset} ${r.version}`));
  };
  const revisionName = (r: Revision) => versionsOf(r.rev).join(" · ") || "버전에 없는 라벨";
  const inDraft = Boolean(item && item.edit.draft_id === releaseId);

  // Labels of another label space are matched to this version's classes by name (colour, "only these classes").
  const here = useMemo(() => new Map(classes.map((c) => [c.name, c.id])), [classes]);
  const foreign = (r: Revision | null) => Boolean(r && item && r.label_space.id !== item.release_label_space_id);
  const namesOf = (r: Revision) => Object.fromEntries(r.classes.map((c) => [c.id, c.name])) as Record<number, string>;
  const colorIn = (r: Revision, cls: number) => {
    if (!foreign(r)) return colorOf(cls);
    const same = here.get(namesOf(r)[cls]);
    return same != null ? colorOf(same) : classColor(cls);
  };
  const shownBox = (r: Revision, b: number[]) => {
    const name = namesOf(r)[b[0]] ?? String(b[0]);
    return !hidden.has(name) && !(onlyHere && foreign(r) && classes.length > 0 && !here.has(name));
  };
  const visible = (r: Revision) => r.boxes.filter((b) => shownBox(r, b));

  // While fixing, the labels being fixed are drawn like a revision of this version's classes.
  const editRevision: Revision | null =
    editing && revision
      ? { ...revision, id: -1, boxes, box_count: boxes.length, classes, label_space: { ...revision.label_space, id: item?.release_label_space_id ?? "" } }
      : null;
  const shownRevision = editRevision ?? revision;

  const counts = useMemo(() => {
    const map = new Map<string, { n: number; id: number }>();
    if (!shownRevision) return [];
    const names = Object.fromEntries(shownRevision.classes.map((c) => [c.id, c.name])) as Record<number, string>;
    for (const box of shownRevision.boxes) {
      const name = names[box[0]] ?? String(box[0]);
      map.set(name, { n: (map.get(name)?.n ?? 0) + 1, id: box[0] });
    }
    return [...map.entries()].sort((a, b) => b[1].n - a[1].n);
  }, [shownRevision]);

  const toggleClass = (name: string) => {
    const next = new Set(hidden);
    if (next.has(name)) next.delete(name);
    else next.add(name);
    setHidden(next);
  };

  const w = item?.width ?? 1;
  const h = item?.height ?? 1;
  const font = Math.max(w, h) / 70;
  // Fit the image (or two of them, side by side) inside the stage whatever its aspect ratio.
  const pair = Boolean(compare);
  const room = pair ? (stage.w - 12) / 2 : stage.w;
  const fitWidth = Math.max(0, Math.min(room, ((stage.h - (pair ? 26 : 0)) * w) / h));
  const frameStyle = { width: fitWidth, height: (fitWidth * h) / w };

  const drawBoxes = (r: Revision) => {
    const names = namesOf(r);
    return (
      <svg viewBox={`0 0 ${w} ${h}`} preserveAspectRatio="none">
        {visible(r).map((b, i) => {
          // A classification label has no geometry: it is drawn as a frame around the whole image.
          const [cls, cx = 0.5, cy = 0.5, bw = 1, bh = 1] = b;
          const x = (cx - bw / 2) * w;
          const y = (cy - bh / 2) * h;
          const color = colorIn(r, cls);
          const label = names[cls] ?? String(cls);
          // A segmentation label carries its outline after the box (x1, y1, x2, y2, …);
          // a keypoint label its points (x, y, visibility, …) and the class names them and their limbs.
          const pose = task === "pose";
          const keypoints = pose ? keypointsOf(b) : [];
          const kind = r.classes.find((c) => c.id === cls);
          const outline = pose ? [] : b.slice(5);
          const points = outline.map((v, n) => (n % 2 === 0 ? v * w : v * h)).join(" ");
          const thin = outline.length >= 6 || keypoints.length > 0;
          // dots sized to the person, so a crowd of small people stays readable
          const dot = Math.min(font * 0.32, Math.max(font * 0.12, Math.min(bw * w, bh * h) * 0.045));
          return (
            <g key={i}>
              {outline.length >= 6 && (
                <polygon
                  points={points}
                  fill={color}
                  fillOpacity={fill ? 0.4 : 0.18}
                  stroke={color}
                  strokeWidth="2"
                  vectorEffect="non-scaling-stroke"
                />
              )}
              <rect
                x={x}
                y={y}
                width={bw * w}
                height={bh * h}
                strokeDasharray={thin ? "4 3" : undefined}
                fill={fill && !thin ? color : "none"}
                fillOpacity={0.25}
                stroke={color}
                strokeWidth="2"
                vectorEffect="non-scaling-stroke"
              />
              {(kind?.skeleton ?? []).map(([a, c], n) =>
                keypoints[a] && keypoints[c] && keypoints[a][2] > 0 && keypoints[c][2] > 0 ? (
                  <line
                    key={`l${n}`}
                    x1={keypoints[a][0] * w}
                    y1={keypoints[a][1] * h}
                    x2={keypoints[c][0] * w}
                    y2={keypoints[c][1] * h}
                    stroke={color}
                    strokeWidth="2.5"
                    vectorEffect="non-scaling-stroke"
                  />
                ) : null,
              )}
              {keypoints.map(([kx, ky, seen], n) =>
                seen > 0 ? (
                  <circle
                    key={`p${n}`}
                    cx={kx * w}
                    cy={ky * h}
                    r={dot}
                    fill={seen === 2 ? color : "#0008"}
                    stroke={seen === 2 ? "#fff" : color}
                    strokeWidth="1.5"
                    vectorEffect="non-scaling-stroke"
                  >
                    <title>{`${kind?.keypoints?.[n] ?? n + 1}${seen === 1 ? " (가려짐)" : ""}`}</title>
                  </circle>
                ) : null,
              )}
              {showNames && (
                <>
                  <rect x={x} y={Math.max(0, y - font * 1.35)} width={label.length * font * 0.62 + font * 0.6} height={font * 1.35} fill={color} />
                  <text x={x + font * 0.3} y={Math.max(font, y - font * 0.35)} fontSize={font} fill="#fff" fontWeight="600">
                    {label}
                  </text>
                </>
              )}
            </g>
          );
        })}
      </svg>
    );
  };

  const shown = editRevision ? [editRevision] : [revision, compare].filter((r): r is Revision => r !== null);
  const anyForeign = shown.some(foreign) && classes.length > 0;
  const editVisible = new Set(boxes.map((b, i) => (editRevision && shownBox(editRevision, b) ? i : -1)).filter((i) => i >= 0));
  const editState = saving ? "저장 중…" : dirty ? "저장 안 됨" : "저장됨";

  return (
    <div className={`viewer ${editing ? "editing" : ""}`} role="dialog" aria-modal="true" aria-label="항목 보기">
      <div className="viewer-stage" ref={stageRef}>
        {error && <div className="error">{error}</div>}
        {item && (
          <div className="viewer-frames">
            {(shown.length ? shown : [null]).map((r, n) => (
              <figure key={r?.id ?? n} className="viewer-figure">
                <div className="viewer-frame" style={frameStyle}>
                  {/* the thumbnail at once, the original over it as soon as it is here */}
                  {item.thumb_url && full !== "ok" && <img src={item.thumb_url} alt="" className="placeholder" />}
                  {item.image_url && full !== "error" && (
                    <img
                      key={item.image_url}
                      src={item.image_url}
                      alt={item.file_name}
                      className={full === "ok" ? "" : "loading"}
                      onLoad={() => setFull("ok")}
                      onError={() => setFull("error")}
                    />
                  )}
                  {(showLabels || editing) && r && drawBoxes(r)}
                  {editing && task !== "classify" && fitWidth > 0 && (
                    <EditLayer
                      width={w}
                      height={h}
                      boxes={boxes}
                      visible={editVisible}
                      selected={selected}
                      drawing={drawing}
                      geometry={geometry}
                      drawClass={drawClass}
                      scale={w / fitWidth}
                      colorOf={colorOf}
                      onSelect={setSelected}
                      onChange={setBoxes}
                      onCommit={commit}
                    />
                  )}
                </div>
                {pair && r && (
                  <figcaption>
                    <strong>{revisionName(r)}</strong> · {r.state === "labelled" ? `박스 ${visible(r).length}개` : STATE_LABEL[r.state]} ·{" "}
                    {r.label_space.name}
                    {n === 1 && (
                      <button className="link" onClick={() => setCompareId(null)}>
                        닫기
                      </button>
                    )}
                  </figcaption>
                )}
              </figure>
            ))}
          </div>
        )}
        {item && full === "loading" && item.image_local === false && (
          <div className="viewer-note">원본을 이미지가 있는 노드에서 받는 중… (받은 뒤에는 바로 열림)</div>
        )}
        {item && full === "error" && (
          <div className="viewer-note">원본을 받지 못해 썸네일로 보여 줍니다 (노드가 꺼져 있거나 파일이 바뀌었을 수 있음).</div>
        )}
        <div className="viewer-nav">
          <button className="icon-btn" onClick={() => void go(index - 1)} disabled={index === 0 || saving} aria-label="이전">
            ‹
          </button>
          <span className="num">
            {(index + 1).toLocaleString()} / {total.toLocaleString()}
          </span>
          <button className="icon-btn" onClick={() => void go(index + 1)} disabled={index >= total - 1 || saving} aria-label="다음">
            ›
          </button>
        </div>
      </div>

      <aside className="viewer-panel">
        <header>
          <strong title={item?.file_name}>{item?.file_name ?? "…"}</strong>
          <button className="icon-btn" onClick={close} aria-label="닫기">
            <Icon name="x" />
          </button>
        </header>

        {item && !editing && (
          <section className="edit-entry">
            {item.edit.allowed ? (
              <button className="primary edit-start" onClick={beginEdit}>
                <Icon name="edit" size={14} />
                라벨 고치기 <kbd>E</kbd>
              </button>
            ) : (
              <p className="muted small">
                {item.edit.reason}{" "}
                {item.edit.latest && item.edit.latest.id !== releaseId && (
                  <a href={`#/datasets/${item.edit.latest.id}`} onClick={onClose}>
                    {item.edit.latest.version} 열기
                  </a>
                )}
              </p>
            )}
            {item.edit.fixed && !inDraft && item.edit.revision_id !== revisionId && (
              <p className="notice small">
                작업 중에서 고친 라벨이 있습니다.{" "}
                <button className="link" onClick={() => setRevisionId(item.edit.revision_id)}>
                  보기
                </button>
              </p>
            )}
            {note && <div className={note.kind === "saved" ? "notice ok small" : "error small"}>{note.text}</div>}
          </section>
        )}

        {item && editing && (
          <section className="edit-panel">
            <div className="edit-head">
              <div className="eyebrow">라벨 고치기</div>
              <span className={`edit-state ${dirty ? "dirty" : ""}`}>{editState}</span>
            </div>
            <p className="muted small">
              {inDraft ? "작업 중에 저장됩니다." : "저장하면 데이터셋의 작업 중에 모입니다. 이 버전은 그대로 남습니다."}
            </p>
            {geometry && (
              <div className="segmented tool-switch" role="group" aria-label="도구">
                <button className={!drawing ? "active" : ""} onClick={() => setDrawing(false)} aria-pressed={!drawing}>
                  고르기 <kbd>V</kbd>
                </button>
                <button
                  className={drawing ? "active" : ""}
                  onClick={() => {
                    setDrawing(true);
                    setSelected(null);
                  }}
                  aria-pressed={drawing}
                >
                  박스 그리기 <kbd>B</kbd>
                </button>
              </div>
            )}
            <div className="muted small edit-hint">
              {task === "classify"
                ? "이 이미지의 클래스를 고르세요."
                : selected != null
                  ? "고른 박스의 클래스를 바꾸려면 아래에서 고르세요."
                  : drawing
                    ? "끌어서 박스를 그립니다. 아래에서 고른 클래스로 그려집니다."
                    : geometry
                      ? "박스를 눌러 고르고, 끌어서 옮기거나 모서리로 크기를 바꿉니다."
                      : "라벨을 눌러 고르면 클래스를 바꾸거나 지울 수 있습니다. 윤곽 · 점 모양은 아직 고칠 수 없습니다."}
            </div>
            <ClassPicker
              classes={classes}
              current={task === "classify" ? (boxes[0]?.[0] ?? null) : selected != null ? (boxes[selected]?.[0] ?? null) : drawClass}
              colorOf={colorOf}
              onPick={pickClass}
            />
            <div className="row-actions">
              <button onClick={undoStep} disabled={!undo.length} title="Ctrl+Z">
                되돌리기
              </button>
              <button onClick={redoStep} disabled={!redo.length} title="Ctrl+Shift+Z">
                다시 하기
              </button>
              {selected != null && (
                <button onClick={removeSelected} title="Delete">
                  지우기
                </button>
              )}
              {task !== "classify" && (
                <button onClick={toBackground} disabled={!boxes.length} title="박스를 모두 지우고 '찾을 것이 없는 이미지'로">
                  배경으로
                </button>
              )}
            </div>
            {note && (
              <div className={note.kind === "saved" ? "notice ok small" : "error small"}>
                {note.text}
                {note.kind === "conflict" && (
                  <div className="row-actions">
                    <button onClick={() => void lookAtTheirs()}>그 라벨 보기</button>
                    <button className="danger" onClick={() => void save(true)}>
                      내 것으로 저장
                    </button>
                  </div>
                )}
              </div>
            )}
            <div className="edit-actions">
              <button className="primary" onClick={() => void save()} disabled={saving || !dirty}>
                저장 <kbd>Ctrl S</kbd>
              </button>
              <button onClick={stopEdit}>그만 고치기</button>
            </div>
            {item.edit.fixed && item.edit.draft_id && (
              <button className="link small" onClick={() => void revertImage()}>
                이 이미지를 바탕 버전의 라벨로 되돌리기
              </button>
            )}
            <p className="muted small">← → 저장하고 넘기기 · Esc 고르기 풀기 · Ctrl+C / Ctrl+V 박스 복사</p>
          </section>
        )}

        <section>
          <div className="eyebrow">표시</div>
          <label className="check">
            <input type="checkbox" checked={showLabels || editing} disabled={editing} onChange={(e) => setShowLabels(e.target.checked)} /> 라벨{" "}
            <kbd>L</kbd>
          </label>
          <label className="check">
            <input type="checkbox" checked={showNames} onChange={(e) => setShowNames(e.target.checked)} /> 클래스 이름 <kbd>N</kbd>
          </label>
          <label className="check">
            <input type="checkbox" checked={fill} onChange={(e) => setFill(e.target.checked)} /> 박스 채우기 <kbd>F</kbd>
          </label>
          {anyForeign && (
            <label className="check" title="다른 클래스 목록의 라벨에서, 이 버전에 같은 이름의 클래스가 없는 것은 숨깁니다">
              <input type="checkbox" checked={onlyHere} onChange={(e) => setOnlyHere(e.target.checked)} /> 이 버전의 클래스만
            </label>
          )}
          <div className="class-chips">
            {counts.map(([name, { n, id }]) => (
              <button key={name} className={`class-chip ${hidden.has(name) ? "off" : ""}`} onClick={() => toggleClass(name)} aria-pressed={!hidden.has(name)}>
                <i style={{ background: shownRevision ? colorIn(shownRevision, id) : colorOf(id) }} />
                {name} <span>{n}</span>
              </button>
            ))}
          </div>
          {counts.length > 1 && (
            <div className="row-actions">
              <button onClick={() => setHidden(new Set())}>모두</button>
              <button onClick={() => setHidden(new Set(counts.map(([name]) => name)))}>없음</button>
            </div>
          )}
        </section>

        {item && (
          <>
            <section>
              <div className="eyebrow">버전별 라벨</div>
              {item.revisions.map((r) => (
                <div key={r.id} className={`revision ${r.id === revisionId ? "active" : ""} ${r.id === compare?.id ? "paired" : ""}`}>
                  <input
                    type="radio"
                    name="revision"
                    checked={r.id === revisionId}
                    disabled={editing}
                    onChange={() => setRevisionId(r.id)}
                    aria-label={`${revisionName(r)} 라벨 보기`}
                  />
                  <div className="revision-body" onClick={() => !editing && setRevisionId(r.id)}>
                    <div>
                      <strong title={`라벨 개정 ${r.rev}`}>{revisionName(r)}</strong> ·{r.state === "labelled" ? `박스 ${r.box_count}개` : STATE_LABEL[r.state]} ·{" "}
                      {r.label_space.name}
                      {r.id === item.revision_id && <span className="tag accent">이 버전</span>}
                    </div>
                    <div className="muted small">
                      {sourceLabel(r.source)} · {r.created_at.slice(0, 10)}
                    </div>
                    {r.id === revisionId && r.labelled_classes.length > 0 && (
                      <div className="muted small">
                        빠짐없이 라벨한 클래스 {r.labelled_classes.length}개: {r.labelled_classes.slice(0, 12).join(", ")}
                        {r.labelled_classes.length > 12 && " 외"}
                      </div>
                    )}
                  </div>
                  {!editing && item.revisions.length > 1 && r.id !== revisionId && (
                    <button
                      className={`icon-btn pair-btn ${r.id === compare?.id ? "active" : ""}`}
                      onClick={() => setCompareId(r.id === compare?.id ? null : r.id)}
                      title={r.id === compare?.id ? "나란히 보기 닫기" : "선택한 라벨과 나란히 보기"}
                      aria-pressed={r.id === compare?.id}
                    >
                      <Icon name="columns" size={14} />
                    </button>
                  )}
                </div>
              ))}
              {!editing && item.revisions.length > 1 && !compare && <p className="muted small">옆의 버튼으로 두 라벨을 나란히 볼 수 있습니다.</p>}
            </section>

            <section>
              <div className="eyebrow">이미지</div>
              <dl className="info">
                <dt>크기</dt>
                <dd>
                  {item.width}×{item.height} · {bytesLabel(item.bytes)}
                </dd>
                <dt>사이트</dt>
                <dd>{item.site || "–"}</dd>
                <dt>카메라</dt>
                <dd>{item.camera || "–"}</dd>
                <dt>split</dt>
                <dd>{SPLIT_LABEL[item.split] ?? item.split}</dd>
                <dt>sha256</dt>
                <dd className="mono" title={item.sha256}>
                  {item.sha256.slice(0, 16)}…
                </dd>
                <dt>경로</dt>
                <dd className="mono path">{item.rel_path}</dd>
              </dl>
            </section>

            <section>
              <div className="eyebrow">포함된 버전 ({item.releases.length})</div>
              <ul className="release-list">
                {item.releases.map((r) => (
                  <li key={r.id}>
                    <a href={`#/datasets/${r.id}`} className={r.current ? "current" : ""} onClick={close}>
                      {r.dataset} {r.version}
                    </a>
                    <span className="muted small">
                      {SPLIT_LABEL[r.split] ?? r.split}
                      {r.rev !== currentRev && " · 라벨이 다름"}
                    </span>
                  </li>
                ))}
              </ul>
            </section>
          </>
        )}
      </aside>
    </div>
  );
}
