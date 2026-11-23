import { useEffect, useMemo, useRef, useState } from "react";
import { ApiError, api } from "./api";
import {
  type ItemDetail,
  type ItemPage,
  type PredictionRun,
  type LabelSetView,
  SET_STATE_LABEL,
  type SetState,
  type LabelClass,
  type Revision,
  SPLIT_LABEL,
  STATE_LABEL,
  bytesLabel,
  classColor,
  keypointsOf,
  runName,
} from "./catalogTypes";
import {
  type Box,
  ClassPicker,
  EditLayer,
  KeypointList,
  type Pt,
  keypointList,
  moveShape,
  outlineOf,
  polygonShape,
  withKeypoints,
} from "./LabelEditor";
import { CHANGE_LABEL, diffLabels } from "./labelDiff";
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
  /** An evaluation the images are sorted by: its model's predictions are shown over each image. */
  checkOp?: string;
  /** A model run whose predictions the grid lays over the thumbnails: shown over the image from the start. */
  predictionRun?: string;
  /** A part of a labeling set opened to label (fixing from the start) or to review. */
  labeling?: LabelingMode;
  /** Opened from a label check (검수): its name; the labels it flagged are marked on the image. */
  qaTitle?: string;
  /** The check in play (a rule's id, or extra / missed of a model): what "문제 없음" is kept for. */
  qaCheck?: string;
  /** The images are the ones someone judged right: the viewer offers to count each again. */
  qaLetBe?: boolean;
}

export interface LabelingMode {
  setId: string;
  part: number;
  mode: "label" | "review";
  review: boolean;
  name: string;
  instructions: string;
  autolabel?: boolean; // the set's 오토라벨 is done: an image not labelled yet starts from the model's labels
}

interface Saved {
  release_id: string;
  revision_id: number;
  state: string;
  box_count: number;
  edited: number;
  set?: { item_id: number; state: SetState }; // saved as part of a labeling set
}

/** Where a label revision came from, in words: "import:coco v1" → "가져오기 · coco v1". */
function sourceLabel(source: string): string {
  const [kind, ...rest] = source.split(":");
  const what = rest.join(":");
  const words: Record<string, string> = {
    import: "가져오기",
    classes: "클래스 변경",
    human: "사람이 고침",
    autolabel: "자동 라벨",
    model: "모델이 그림 (오토라벨)",
  };
  return words[kind] ? (what ? `${words[kind]} · ${what}` : words[kind]) : source;
}

// Up to three runs' predictions over an image at once: each its letter and dash (the class colour stays the class's).
interface RunStyle {
  letter: string;
  dash: string;
}
const RUN_STYLES: RunStyle[] = [
  { letter: "A", dash: "6 4" },
  { letter: "B", dash: "2 3" },
  { letter: "C", dash: "12 4 2 4" },
];
const RUNS_KEY = "badamoa.predictionRuns"; // + the version: the runs picked to lay over its images
const SCORE_KEY = "badamoa.predictionScore";

function remember(key: string, value: string) {
  try {
    localStorage.setItem(key, value);
  } catch {
    /* not remembered */
  }
}

function rememberedRuns(releaseId: string): string[] {
  try {
    const kept = JSON.parse(localStorage.getItem(`${RUNS_KEY}.${releaseId}`) ?? "[]");
    return Array.isArray(kept) ? kept.filter((v) => typeof v === "string").slice(0, RUN_STYLES.length) : [];
  } catch {
    return [];
  }
}

function rememberedScore(): number {
  try {
    const kept = Number(localStorage.getItem(SCORE_KEY));
    return Number.isFinite(kept) && kept >= 0 && kept < 1 && localStorage.getItem(SCORE_KEY) !== null ? kept : 0.25;
  } catch {
    return 0.25;
  }
}

const copy = (boxes: Box[]) => boxes.map((b) => [...b]);

/** Overlap of two boxes [class, cx, cy, w, h] (intersection over union). */
function iou(a: number[], b: number[]): number {
  const [, ax, ay, aw, ah] = a;
  const [, bx, by, bw, bh] = b;
  const w = Math.min(ax + aw / 2, bx + bw / 2) - Math.max(ax - aw / 2, bx - bw / 2);
  const h = Math.min(ay + ah / 2, by + bh / 2) - Math.max(ay - ah / 2, by - bh / 2);
  if (w <= 0 || h <= 0) return 0;
  return (w * h) / (aw * ah + bw * bh - w * h);
}
const MAX_ZOOM = 16;

/** Full-screen item view: labels over the image, label revisions (one or two side by side), where else the image is used.
 * The mouse wheel zooms (at the pointer) and a drag moves the zoomed image. "라벨 고치기" (E) turns the same screen
 * into an editor for boxes, outlines (segmentation), keypoints (pose) or the class (classification); what is saved
 * lands in the dataset's work in progress. */
export function ItemViewer({
  releaseId,
  filterQuery,
  start,
  total,
  onClose,
  colorOf = classColor,
  classes = [],
  task = "detect",
  onSaved,
  checkOp,
  predictionRun,
  labeling,
  qaTitle,
  qaCheck,
  qaLetBe = false,
}: Props) {
  const [index, setIndex] = useState(start);
  const [reloads, setReloads] = useState(0); // the list changed under the same index: load it again
  const [item, setItem] = useState<ItemDetail | null>(null);
  const [error, setError] = useState<string | null>(null);
  const [revisionId, setRevisionId] = useState<number | null>(null);
  const [compareId, setCompareId] = useState<number | null>(null);
  const [showLabels, setShowLabels] = useState(true);
  const [showNames, setShowNames] = useState(true);
  const [fill, setFill] = useState(false);
  const [onlyHere, setOnlyHere] = useState(false);
  const [hidden, setHidden] = useState<Set<string>>(new Set()); // class names
  const [prefilled, setPrefilled] = useState<{ item: number; n: number; model: string } | null>(null); // from 오토라벨
  const [hiddenBoxes, setHiddenBoxes] = useState<Set<string>>(new Set()); // single labels turned off: "revision:index"
  const [hoverBox, setHoverBox] = useState<number | null>(null); // the label pointed at in the list
  const [focusBox, setFocusBox] = useState<number | null>(null); // the label picked in the list: the others dim
  const [full, setFull] = useState<"loading" | "ok" | "error">("loading"); // the original image
  const stageRef = useRef<HTMLDivElement>(null);
  const [stage, setStage] = useState({ w: 0, h: 0 });

  // ---- a labeling set's part: this image's state in it, and how far the part got
  const [inSet, setInSet] = useState<{ state: SetState; note: string } | null>(null);
  const [flagged, setFlagged] = useState<{ boxes: number[]; note: string } | null>(null); // what a label check found
  const [letBe, setLetBe] = useState<{ note: string; by: string | null; at: string } | null>(null); // "문제 없음"
  const [letBeNote, setLetBeNote] = useState("");
  const [judging, setJudging] = useState(false);
  const [progress, setProgress] = useState<{ size: number; states: Record<SetState, number> } | null>(null);
  const refreshProgress = async () => {
    if (!labeling) return;
    try {
      const view = await api<LabelSetView>(`/label-sets/${labeling.setId}`);
      const part = view.part_list.find((p) => p.part === labeling.part);
      if (part) setProgress({ size: part.size, states: part.states });
    } catch {
      /* the counts wait for the next action */
    }
  };
  useEffect(() => {
    void refreshProgress();
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [labeling?.setId, labeling?.part]);

  // ---- what models predicted for this image: every evaluation and 오토라벨 run that did. Up to three are laid
  // over the labels at once, each with its own letter and dash; the choice stays from image to image.
  const [runs, setRuns] = useState<PredictionRun[] | null>(null);
  const [shownRuns, setShownRuns] = useState<string[]>(() => {
    const kept = rememberedRuns(releaseId);
    const first = [checkOp, predictionRun].filter((id): id is string => Boolean(id));
    return [...first, ...kept.filter((id) => !first.includes(id))].slice(0, RUN_STYLES.length);
  });
  const [minScore, setMinScore] = useState<number>(rememberedScore);
  const [showPred, setShowPred] = useState(true);
  const [showFlagged, setShowFlagged] = useState(true); // 검수's frames round the labels it flagged (Q)
  const [showDiff, setShowDiff] = useState(true); // what the work in progress changed, over the image

  // ---- zoom: 1 fits the image; pan is where the zoomed image sits inside its frame (screen pixels)
  const [zoom, setZoomState] = useState(1);
  const [pan, setPanState] = useState<Pt>([0, 0]);
  // The newest zoom and pan at once: wheel events can come faster than the screen redraws.
  const view = useRef<{ zoom: number; pan: Pt }>({ zoom: 1, pan: [0, 0] });
  const setZoom = (z: number) => {
    view.current.zoom = z;
    setZoomState(z);
  };
  const setPan = (p: Pt) => {
    view.current.pan = p;
    setPanState(p);
  };
  const [space, setSpace] = useState(false); // held: a drag moves the image, even while fixing
  const panning = useRef<{ x: number; y: number; from: Pt } | null>(null);
  const [grabbing, setGrabbing] = useState(false);

  // ---- fixing labels
  const [editing, setEditing] = useState(labeling?.mode === "label"); // labeling: fixing from the first image
  const editingRef = useRef(false);
  editingRef.current = editing;
  const [boxes, setBoxes] = useState<Box[]>([]);
  const [baseline, setBaseline] = useState("[]"); // the labels the fix started from, to tell unsaved changes
  const [undo, setUndo] = useState<Box[][]>([]);
  const [redo, setRedo] = useState<Box[][]>([]);
  const [selected, setSelected] = useState<number | null>(null);
  const [point, setPoint] = useState<number | null>(null); // outline corner or keypoint of the selected label
  const [drawing, setDrawing] = useState(false);
  const [sketch, setSketch] = useState<Pt[]>([]); // corners of an outline being drawn
  const [placing, setPlacing] = useState<number | null>(null); // pose: the keypoint the next click places
  const [drawClass, setDrawClass] = useState<number>(classes[0]?.id ?? 0);
  const [saving, setSaving] = useState(false);
  const [note, setNote] = useState<{ kind: "saved" | "error" | "conflict"; text: string } | null>(null);
  const clip = useRef<Box | null>(null);
  const dirty = editing && JSON.stringify(boxes) !== baseline;
  const shapes = task !== "classify"; // labels with a place in the image (not a class for the whole image)
  const kind = task === "segment" ? "outline" : task === "pose" ? "pose" : "box";
  const keypointNames = (cls: number) => classes.find((c) => c.id === cls)?.keypoints ?? [];

  useEffect(() => {
    const el = stageRef.current;
    if (!el) return;
    const measure = () => setStage({ w: el.clientWidth - 32, h: el.clientHeight - 84 }); // padding and nav bar
    measure();
    const observer = new ResizeObserver(measure);
    observer.observe(el);
    return () => observer.disconnect();
  }, []);

  // 오토라벨: an image nobody labelled yet starts from what the model drew (above the set's lowest confidence).
  useEffect(() => {
    if (!labeling?.autolabel || !editing || !item || prefilled?.item === item.id) return;
    const start = item.revisions.find((r) => r.id === item.edit.revision_id);
    if ((start && start.state !== "unlabelled") || boxes.length > 0) return;
    let live = true;
    api<{ model: { name: string; version: number } | null; labels: number[][] }>(
      `/label-sets/${labeling.setId}/items/${item.id}/autolabel`,
    )
      .then((found) => {
        if (!live || found.labels.length === 0) return;
        setBoxes(found.labels as Box[]);
        setPrefilled({ item: item.id, n: found.labels.length, model: found.model ? `${found.model.name} v${found.model.version}` : "모델" });
      })
      .catch(() => {
        /* no predictions for this image: start empty */
      });
    return () => {
      live = false;
    };
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [labeling?.autolabel, labeling?.setId, editing, item?.id]);

  const select = (index: number | null, at: number | null = null) => {
    setSelected(index);
    setPoint(index == null ? null : at);
    if (index !== selected) setPlacing(null);
  };

  /** Begin fixing this image from the labels of the work in progress (or this version's). */
  const startFrom = (detail: ItemDetail) => {
    const rev = detail.revisions.find((r) => r.id === detail.edit.revision_id);
    const begin = copy(rev?.boxes ?? []);
    setBoxes(begin);
    setBaseline(JSON.stringify(begin));
    setUndo([]);
    setRedo([]);
    setSelected(null);
    setPoint(null);
    setSketch([]);
    setPlacing(null);
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
        setInSet(page.items[0].set ? { state: page.items[0].set.state, note: page.items[0].set.note } : null);
        setFlagged(page.items[0].qa ? { boxes: page.items[0].qa.boxes, note: page.items[0].qa.note } : null);
        setLetBe(page.items[0].let_be ?? null);
        setLetBeNote("");
        setFull(detail.image_url ? "loading" : "error");
        setHiddenBoxes(new Set());
        setHoverBox(null);
        setFocusBox(null);
        setRevisionId(detail.revision_id);
        setCompareId(null);
        setNote(null);
        setZoom(1);
        setPan([0, 0]);
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
  }, [releaseId, filterQuery, index, reloads]);

  useEffect(() => {
    setRuns(null);
    if (!item) return;
    let alive = true;
    api<PredictionRun[]>(`/releases/${releaseId}/items/${item.id}/prediction-runs`)
      .then((found) => alive && setRuns(found))
      .catch(() => alive && setRuns([]));
    return () => {
      alive = false;
    };
  }, [releaseId, item?.id]);
  const pickRun = (opId: string, on: boolean) => {
    const next = on ? [...shownRuns.filter((id) => id !== opId), opId].slice(-RUN_STYLES.length) : shownRuns.filter((id) => id !== opId);
    setShownRuns(next);
    remember(`${RUNS_KEY}.${releaseId}`, JSON.stringify(next));
  };
  const pickScore = (score: number) => {
    setMinScore(score);
    remember(SCORE_KEY, String(score));
  };
  // the runs laid over this image, in the order they were picked (that order gives each its letter and dash)
  const overlays = shownRuns
    .map((id) => (runs ?? []).find((r) => r.op_id === id))
    .filter((r): r is PredictionRun => Boolean(r))
    .map((run) => ({ run, style: RUN_STYLES[shownRuns.indexOf(run.op_id)] }));
  const pred = checkOp ? ((runs ?? []).find((r) => r.op_id === checkOp) ?? null) : null;

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
    select(null);
  };
  const redoStep = () => {
    if (!redo.length) return;
    setUndo((u) => [...u, boxes]);
    setBoxes(redo[redo.length - 1]);
    setRedo((r) => r.slice(0, -1));
    select(null);
  };
  const replace = (index: number, label: Box) => apply(boxes.map((b, i) => (i === index ? label : b)));
  const removeSelected = () => {
    if (selected == null) return;
    apply(boxes.filter((_, i) => i !== selected));
    select(null);
  };
  /** Delete the picked point: an outline corner goes (three stay at least), a keypoint becomes "not labelled". */
  const removePoint = () => {
    if (selected == null || point == null) return;
    const b = boxes[selected];
    if (kind === "outline") {
      const pts = outlineOf(b);
      if (pts.length <= 3) return removeSelected();
      pts.splice(point, 1);
      replace(selected, polygonShape(b[0], pts));
    } else if (kind === "pose") {
      const kps = keypointList(b);
      kps[point] = [0, 0, 0];
      replace(selected, withKeypoints(b, kps));
    }
    setPoint(null);
  };
  const setSeen = (at: number, seen: number) => {
    if (selected == null) return;
    const b = boxes[selected];
    const kps = keypointList(b);
    if (!kps[at] || kps[at][2] === seen) return;
    kps[at] = seen === 0 ? [0, 0, 0] : [kps[at][0], kps[at][1], seen];
    replace(selected, withKeypoints(b, kps));
    setPoint(seen === 0 ? null : at);
  };
  const pickClass = (id: number) => {
    setDrawClass(id);
    if (task === "classify") {
      if (boxes[0]?.[0] !== id) apply([[id]]);
    } else if (selected != null && boxes[selected] && boxes[selected][0] !== id) {
      replace(selected, [id, ...boxes[selected].slice(1)]);
    }
  };
  const toBackground = () => {
    if (!boxes.length) return;
    apply([]);
    select(null);
  };
  const paste = () => {
    const b = clip.current;
    if (!b || !shapes) return;
    apply([...boxes, moveShape(b, task, 0.02, 0.02)]);
    select(boxes.length);
  };
  const closeSketch = () => {
    if (sketch.length < 3) return;
    apply([...boxes, polygonShape(drawClass, sketch)]);
    setSketch([]);
    select(boxes.length);
  };
  const toggleDrawing = (on: boolean) => {
    setDrawing(on);
    setSketch([]);
    setPlacing(null);
    if (on) select(null);
  };

  /** Take predictions as labels: added to the labels being fixed (fixing starts if it has not). */
  const takePredictions = (labels: Box[]) => {
    if (!item?.edit.allowed || !labels.length) return;
    const add = (current: Box[]) => (task === "classify" ? copy(labels.slice(0, 1)) : [...current, ...copy(labels)]);
    if (editing) {
      apply(add(boxes));
      return;
    }
    const rev = item.revisions.find((r) => r.id === item.edit.revision_id);
    const begin = copy(rev?.boxes ?? []);
    startFrom(item);
    setBoxes(add(begin));
    setUndo([begin]); // one undo step brings the labels back as they were
    setNote(null);
    setEditing(true);
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
    toggleDrawing(false);
    select(null);
    if (item) setRevisionId(item.revision_id);
  };

  /** Save this image's labels into the work in progress. ``force``: over a fix someone made in between. */
  const save = async (force = false): Promise<boolean> => {
    if (!item) return false;
    setSaving(true);
    setNote(null);
    try {
      const body = { boxes, parent_revision_id: item.edit.revision_id, force, label_set: labeling?.setId ?? "" };
      const saved = await api<Saved>(`/releases/${releaseId}/items/${item.id}/labels`, { method: "PUT", body });
      if (saved.set) setInSet((s) => ({ note: s?.note ?? "", state: saved.set!.state }));
      if (labeling) void refreshProgress();
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

  /** Labeling: this image is done (saved if changed; looked at and left as it is otherwise), on to the next. */
  const finishAndNext = async () => {
    if (!labeling || !item) return;
    const startedAs = item.revisions.find((r) => r.id === item.edit.revision_id)?.state;
    if (dirty || (editing && startedAs === "unlabelled")) {
      // nothing drawn on an image nobody had labelled: it is background now (looked at, nothing to find)
      if (!(await save())) return;
    } else if (inSet?.state !== "approved") {
      try {
        const done = await api<{ state: SetState }>(`/label-sets/${labeling.setId}/items/${item.id}/done`, { method: "POST" });
        setInSet((s) => ({ note: s?.note ?? "", state: done.state }));
        void refreshProgress();
        onSaved?.();
      } catch (e) {
        setNote({ kind: "error", text: e instanceof Error ? e.message : String(e) });
        return;
      }
    }
    if (index < total - 1) setIndex(index + 1);
    else setNote({ kind: "saved", text: "이 일감의 마지막 이미지입니다." });
  };
  /** Review: approve this image (with the fix just made, if any), or send it back with a reason. */
  const reviewItem = async (approve: boolean) => {
    if (!labeling || !item) return;
    let reason = "";
    if (!approve) {
      reason = window.prompt("돌려보내는 이유 (라벨링한 사람에게 보입니다)", "")?.trim() ?? "";
      if (!reason) return;
    }
    if (dirty && !(await save())) return;
    try {
      const done = await api<{ state: SetState; note: string }>(`/label-sets/${labeling.setId}/items/${item.id}/review`, {
        body: { approve, note: reason },
      });
      setInSet({ state: done.state, note: done.note });
      void refreshProgress();
      onSaved?.();
    } catch (e) {
      setNote({ kind: "error", text: e instanceof Error ? e.message : String(e) });
      return;
    }
    if (index < total - 1) setIndex(index + 1);
    else setNote({ kind: "saved", text: "이 일감의 마지막 이미지입니다." });
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

  // ---- zoom and pan
  const w = item?.width ?? 1;
  const h = item?.height ?? 1;
  const revision = item?.revisions.find((r) => r.id === revisionId) ?? null;
  const compare = editing ? null : (item?.revisions.find((r) => r.id === compareId && r.id !== revisionId) ?? null);
  // Fit the image (or two of them, side by side) inside the stage whatever its aspect ratio.
  const pair = Boolean(compare);
  const room = pair ? (stage.w - 12) / 2 : stage.w;
  const fitWidth = Math.max(0, Math.min(room, ((stage.h - (pair ? 26 : 0)) * w) / h));
  const fitHeight = (fitWidth * h) / w;
  const frameStyle = { width: fitWidth, height: fitHeight };

  /** Keep the zoomed image covering its frame. */
  const clampPan = ([x, y]: Pt, z: number): Pt => [
    Math.min(0, Math.max(fitWidth - fitWidth * z, x)),
    Math.min(0, Math.max(fitHeight - fitHeight * z, y)),
  ];
  /** Zoom by ``factor`` keeping the image point under (cx, cy) where it is. */
  const zoomAt = (factor: number, cx?: number, cy?: number, frame?: Element | null) => {
    const { zoom, pan } = view.current;
    const next = Math.min(MAX_ZOOM, Math.max(1, zoom * factor));
    if (next === zoom) return;
    if (next === 1) {
      setZoom(1);
      setPan([0, 0]);
      return;
    }
    const rect = (frame ?? stageRef.current?.querySelector(".viewer-frame"))?.getBoundingClientRect();
    if (!rect) return;
    const x = (cx ?? rect.left + rect.width / 2) - rect.left;
    const y = (cy ?? rect.top + rect.height / 2) - rect.top;
    const lx = (x - pan[0]) / zoom; // the point under the pointer, in unzoomed frame pixels
    const ly = (y - pan[1]) / zoom;
    setZoom(next);
    setPan(clampPan([x - lx * next, y - ly * next], next));
  };
  const fit = () => {
    setZoom(1);
    setPan([0, 0]);
  };
  const wheelRef = useRef<(e: WheelEvent) => void>(() => {});
  wheelRef.current = (e: WheelEvent) => {
    if (!item) return;
    e.preventDefault();
    const frame = (e.target as Element).closest?.(".viewer-frame");
    zoomAt(Math.exp(-e.deltaY * (e.deltaMode === 1 ? 0.05 : 0.0015)), e.clientX, e.clientY, frame);
  };
  useEffect(() => {
    const el = stageRef.current;
    if (!el) return;
    const onWheel = (e: WheelEvent) => wheelRef.current(e);
    el.addEventListener("wheel", onWheel, { passive: false });
    return () => el.removeEventListener("wheel", onWheel);
  }, []);
  /** A drag moves the zoomed image: any drag while only looking, space + drag or the middle button while fixing. */
  const panDown = (e: React.PointerEvent) => {
    if (zoom <= 1) return;
    if (!(e.button === 1 || space || (!editing && e.button === 0))) return;
    e.preventDefault();
    panning.current = { x: e.clientX, y: e.clientY, from: pan };
    setGrabbing(true);
    (e.currentTarget as Element).setPointerCapture?.(e.pointerId);
  };
  const panMove = (e: React.PointerEvent) => {
    const start = panning.current;
    if (!start) return;
    setPan(clampPan([start.from[0] + e.clientX - start.x, start.from[1] + e.clientY - start.y], zoom));
  };
  const panUp = () => {
    panning.current = null;
    setGrabbing(false);
  };

  // One listener; it always calls the newest handler (the editor's state changes on every step).
  const keyRef = useRef<(e: KeyboardEvent) => void>(() => {});
  keyRef.current = (e: KeyboardEvent) => {
    if (e.target instanceof HTMLInputElement || e.target instanceof HTMLTextAreaElement || e.target instanceof HTMLSelectElement) return;
    const key = e.key.toLowerCase();
    const mod = e.ctrlKey || e.metaKey;
    if (e.key === " ") {
      e.preventDefault();
      setSpace(true);
      return;
    }
    if (!mod && (e.key === "0" || e.key === "+" || e.key === "=" || e.key === "-")) {
      if (e.key === "0") fit();
      else zoomAt(e.key === "-" ? 1 / 1.4 : 1.4);
      return;
    }
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
      if (e.key === "Enter" && sketch.length) {
        e.preventDefault();
        closeSketch();
        return;
      }
      if (e.key === "Enter" && labeling?.mode === "label") {
        e.preventDefault();
        void finishAndNext();
        return;
      }
      if (e.key === "Delete" || e.key === "Backspace") {
        e.preventDefault();
        if (sketch.length) setSketch(sketch.slice(0, -1));
        else if (point != null) removePoint();
        else removeSelected();
        return;
      }
      if (key === "b" && shapes) {
        toggleDrawing(!drawing);
        return;
      }
      if (key === "v") {
        toggleDrawing(false);
        return;
      }
      if (/^[1-9]$/.test(e.key)) {
        const c = classes[Number(e.key) - 1];
        if (c) pickClass(c.id);
        return;
      }
      if (e.key === "Escape") {
        e.preventDefault();
        if (sketch.length) setSketch([]);
        else if (placing != null) setPlacing(null);
        else if (drawing) toggleDrawing(false);
        else if (point != null) setPoint(null);
        else if (selected != null) select(null);
        else stopEdit();
        return;
      }
    }
    if (mod) return;
    if (labeling?.mode === "review" && (key === "a" || key === "r")) {
      void reviewItem(key === "a");
      return;
    }
    if (labeling?.mode === "label" && e.key === "Enter") {
      void finishAndNext();
      return;
    }
    if (e.key === "Escape") close();
    else if (e.key === "ArrowRight") void go(index + 1);
    else if (e.key === "ArrowLeft") void go(index - 1);
    else if (key === "e" && !editing) beginEdit();
    else if (key === "l") setShowLabels((v) => !v);
    else if (key === "n") setShowNames((v) => !v);
    else if (key === "f") setFill((v) => !v);
    else if (key === "p") setShowPred((v) => !v);
    else if (key === "q") setShowFlagged((v) => !v);
    else if (key === "d") setShowDiff((v) => !v);
    else if (key === "o" && qaCheck) void judge();
  };
  useEffect(() => {
    const onKey = (e: KeyboardEvent) => keyRef.current(e);
    const onKeyUp = (e: KeyboardEvent) => e.key === " " && setSpace(false);
    const onBlur = () => setSpace(false);
    window.addEventListener("keydown", onKey);
    window.addEventListener("keyup", onKeyUp);
    window.addEventListener("blur", onBlur);
    return () => {
      window.removeEventListener("keydown", onKey);
      window.removeEventListener("keyup", onKeyUp);
      window.removeEventListener("blur", onBlur);
    };
  }, []);
  // Leaving the page with unsaved fixes asks first.
  useEffect(() => {
    if (!dirty) return;
    const warn = (e: BeforeUnloadEvent) => e.preventDefault();
    window.addEventListener("beforeunload", warn);
    return () => window.removeEventListener("beforeunload", warn);
  }, [dirty]);

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
  const boxOff = (r: Revision, i: number) => hiddenBoxes.has(`${r.id}:${i}`);

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

  const font = Math.max(w, h) / 70 / zoom; // names stay the same size on screen when zoomed in

  const drawBoxes = (r: Revision) => {
    const names = namesOf(r);
    // the label pointed at (or picked) in the list stands out; the others step back
    const active = r.id === shownRevision?.id ? (hoverBox ?? focusBox) : null;
    return (
      <svg viewBox={`0 0 ${w} ${h}`} preserveAspectRatio="none">
        {r.boxes.map((b, i) => {
          if (!shownBox(r, b) || boxOff(r, i)) return null;
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
          const kindOf = r.classes.find((c) => c.id === cls);
          const outline = pose ? [] : b.slice(5);
          const points = outline.map((v, n) => (n % 2 === 0 ? v * w : v * h)).join(" ");
          const thin = outline.length >= 6 || keypoints.length > 0;
          // dots sized to the person, so a crowd of small people stays readable
          const dot = Math.min(font * 0.32, Math.max(font * 0.12, (Math.min(bw * w, bh * h) * 0.045) / zoom));
          const lit = active === i;
          return (
            <g key={i} opacity={active != null && !lit ? 0.35 : 1} className={lit ? "label-lit" : undefined}>
              {lit && (
                <rect
                  x={x}
                  y={y}
                  width={bw * w}
                  height={bh * h}
                  fill="none"
                  stroke="#fff"
                  strokeWidth="6"
                  vectorEffect="non-scaling-stroke"
                />
              )}
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
                strokeDasharray={thin && !lit ? "4 3" : undefined}
                fill={(fill || lit) && !thin ? color : "none"}
                fillOpacity={lit ? 0.18 : 0.25}
                stroke={color}
                strokeWidth={lit ? 3 : 2}
                vectorEffect="non-scaling-stroke"
              />
              {(kindOf?.skeleton ?? []).map(([a, c], n) =>
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
                    <title>{`${kindOf?.keypoints?.[n] ?? n + 1}${seen === 1 ? " (가려짐)" : ""}`}</title>
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

  /** Classification: the image's class as a badge over it (a frame with a name is lost on a small image). */
  const classBadge = (r: Revision) => {
    const names = namesOf(r);
    const labels = visible(r);
    if (!labels.length) return <span className="class-badge empty">{r.state === "unlabelled" ? "아직 안 함" : "클래스 없음"}</span>;
    return labels.map((b, i) => (
      <span key={i} className="class-badge" style={{ background: colorIn(r, b[0]) }}>
        {names[b[0]] ?? b[0]}
      </span>
    ));
  };

  /** "문제 없음": what the check flagged on this image is right (or, listing those, count it again). The image
   * leaves the list it was in: the same place shows the next one. */
  const judge = async () => {
    if (!item || !qaCheck || judging) return;
    setJudging(true);
    try {
      const path = `/releases/${releaseId}/items/${item.id}/let-be`;
      if (qaLetBe) await api(`${path}?check=${encodeURIComponent(qaCheck)}`, { method: "DELETE" });
      else await api(path, { method: "PUT", body: { check: qaCheck, note: letBeNote } });
      onSaved?.();
      if (total <= 1) close();
      else if (index >= total - 1) setIndex(index - 1);
      else setReloads((n) => n + 1);
    } catch (e) {
      setError(e instanceof Error ? e.message : String(e));
    } finally {
      setJudging(false);
    }
  };
  const className = (id: number) => classes.find((c) => c.id === id)?.name ?? String(id);
  /** The labels a check flagged: a thick yellow frame around each with its number and class over it (on the
   * version's own labels only); the panel lists the same numbers. */
  const drawFlagged = (labels: number[][], indices: number[]) => {
    // Flagged labels often lie on each other (that is what some checks find): their tags go side by side.
    const placed: { x: number; y: number; width: number }[] = [];
    const tall = font * 1.25;
    return (
    <svg viewBox={`0 0 ${w} ${h}`} preserveAspectRatio="none" className="flagged">
      {indices.map((i, n) => {
        const b = labels[i];
        if (!b || b.length < 5) return null;
        const box = { x: (b[1] - b[3] / 2) * w, y: (b[2] - b[4] / 2) * h, width: b[3] * w, height: b[4] * h };
        const text = `${n + 1} ${className(b[0])}`;
        const width = text.length * font * 0.6 + font * 0.7;
        // under the box (the label's own name is on its top edge); inside it at the image's bottom edge
        let top = box.y + box.height + tall <= h ? box.y + box.height : box.y + box.height - tall;
        let left = Math.min(Math.max(0, box.x), Math.max(0, w - width));
        const row = (top + tall * 3 <= h ? 1 : -1) * (tall + font * 0.1); // more rows go down, or up near the bottom
        const hits = () => placed.find((p) => Math.abs(p.y - top) < tall && left < p.x + p.width && p.x < left + width);
        for (let hit = hits(), tries = 0; hit && tries < 40; hit = hits(), tries++) {
          left = hit.x + hit.width + font * 0.2;
          if (left + width > w) {
            left = Math.min(Math.max(0, box.x), Math.max(0, w - width));
            top += row; // no room to the right: another row
          }
        }
        placed.push({ x: left, y: top, width });
        return (
          <g key={i}>
            <rect {...box} fill="none" stroke="rgb(0 0 0 / 0.6)" strokeWidth="6" vectorEffect="non-scaling-stroke" />
            <rect {...box} fill="none" stroke="#ffd400" strokeWidth="3" vectorEffect="non-scaling-stroke" />
            <rect x={left} y={top} width={width} height={tall} fill="#ffd400" stroke="rgb(0 0 0 / 0.6)" strokeWidth="1" vectorEffect="non-scaling-stroke" />
            <text x={left + font * 0.35} y={top + font * 0.95} fontSize={font * 0.85} fill="#111" fontWeight="700">
              {text}
            </text>
          </g>
        );
      })}
    </svg>
    );
  };
  /** A tag goes under its box; at the image's bottom edge, inside the box instead. */
  const tagTop = (bottom: number) => (bottom + font * 1.2 <= h ? bottom : bottom - font * 1.2);
  // What changed against the base version: the labels shown (or being fixed) next to the base's.
  const baseFound = item?.edit.base_revision_id != null ? item.revisions.find((r) => r.id === item.edit.base_revision_id) : undefined;
  // An image the base version had no labels for (labelling it from scratch): every label would read "추가".
  const baseRevision = baseFound && baseFound.boxes.length > 0 ? baseFound : undefined;
  const afterLabels = editing ? boxes : revision && revision.id !== baseRevision?.id ? revision.boxes : null;
  const changes =
    baseRevision && afterLabels && (editing || revision?.label_space.id === baseRevision.label_space.id)
      ? diffLabels(baseRevision.boxes, afterLabels)
      : null;
  const changeCount = changes ? changes.added.length + changes.removed.length + changes.moved.length + changes.reclassed.length : 0;
  /** Added in green, removed as red dashes, moved with its old place dashed, reclassed in orange with old → new. */
  const drawChanges = (before: number[][], after: number[][]) => {
    if (!changes) return null;
    const rect = (b: number[]) => ({ x: (b[1] - b[3] / 2) * w, y: (b[2] - b[4] / 2) * h, width: b[3] * w, height: b[4] * h });
    const tag = (b: number[], text: string, color: string, key: string) => {
      const r = rect(b);
      const width = text.length * font * 0.55 + font * 0.6;
      const top = tagTop(r.y + r.height); // under the box, or inside it at the image's bottom edge
      return (
        <g key={key}>
          <rect x={r.x} y={top} width={width} height={font * 1.2} fill={color} />
          <text x={r.x + font * 0.3} y={top + font * 0.9} fontSize={font * 0.8} fill="#fff" fontWeight="700">
            {text}
          </text>
        </g>
      );
    };
    const frame = (b: number[], color: string, dashed: boolean, key: string) =>
      b.length >= 5 ? (
        <g key={key}>
          <rect {...rect(b)} fill="none" stroke="rgb(0 0 0 / 0.55)" strokeWidth="5" vectorEffect="non-scaling-stroke" />
          <rect {...rect(b)} fill="none" stroke={color} strokeWidth="3" strokeDasharray={dashed ? "7 5" : undefined} vectorEffect="non-scaling-stroke" />
        </g>
      ) : null;
    return (
      <svg viewBox={`0 0 ${w} ${h}`} preserveAspectRatio="none" className="changes">
        {changes.removed.map((i) => before[i]?.length >= 5 && [frame(before[i], "#e5484d", true, `r${i}`), tag(before[i], "지움", "#e5484d", `rt${i}`)])}
        {changes.moved.map(([i, j]) => [frame(before[i], "#5fb3ff", true, `m${i}`), frame(after[j], "#2f8fff", false, `mn${j}`), tag(after[j], "옮김", "#2f8fff", `mt${j}`)])}
        {changes.reclassed.map(([i, j]) =>
          after[j]?.length >= 5
            ? [frame(after[j], "#f5a524", false, `c${j}`), tag(after[j], `${className(before[i][0])} → ${className(after[j][0])}`, "#d4860b", `ct${j}`)]
            : null,
        )}
        {changes.added.map((j) => after[j]?.length >= 5 && [frame(after[j], "#30a46c", false, `a${j}`), tag(after[j], "추가", "#30a46c", `at${j}`)])}
      </svg>
    );
  };
  /** A model's predictions over the image: dashed (each run its own dash and letter), with how sure it was. */
  const drawPredictions = (labels: number[][], scores: number[], style: RunStyle = RUN_STYLES[0]) => (
    <svg key={style.letter} viewBox={`0 0 ${w} ${h}`} preserveAspectRatio="none" className="predictions">
      {labels.map((b, i) => {
        if (b.length < 5 || (scores[i] ?? 0) < minScore) return null;
        const [cls, cx, cy, bw, bh] = b;
        const color = colorOf(cls);
        const outline = task === "segment" ? b.slice(5) : [];
        const box = { x: (cx - bw / 2) * w, y: (cy - bh / 2) * h, width: bw * w, height: bh * h };
        const tag = `${style.letter} ${className(cls)} ${Math.round((scores[i] ?? 0) * 100)}%`;
        // White dashes on a dark edge: they stand apart from the labels, which are solid lines in the class colour.
        return (
          <g key={i}>
            {outline.length >= 6 &&
              [
                { stroke: "rgb(0 0 0 / 0.6)", width: 4, dash: undefined },
                { stroke: "#fff", width: 2, dash: style.dash },
              ].map((s, n) => (
                <polygon
                  key={n}
                  points={outline.map((v, k) => (k % 2 === 0 ? v * w : v * h)).join(" ")}
                  fill="none"
                  stroke={s.stroke}
                  strokeWidth={s.width}
                  strokeDasharray={s.dash}
                  vectorEffect="non-scaling-stroke"
                />
              ))}
            <rect {...box} fill="none" stroke="rgb(0 0 0 / 0.6)" strokeWidth="4" vectorEffect="non-scaling-stroke" />
            <rect {...box} fill="none" stroke="#fff" strokeWidth="2" strokeDasharray={style.dash} vectorEffect="non-scaling-stroke" />
            <rect x={box.x} y={tagTop(box.y + box.height)} width={tag.length * font * 0.5 + font * 0.5} height={font * 1.2} fill="rgb(20 26 34 / 0.85)" />
            <rect x={box.x} y={tagTop(box.y + box.height)} width={font * 0.25} height={font * 1.2} fill={color} />
            <text x={box.x + font * 0.4} y={tagTop(box.y + box.height) + font * 0.9} fontSize={font * 0.8} fill="#fff" fontWeight="600">
              {tag}
            </text>
          </g>
        );
      })}
    </svg>
  );
  // A run's predictions the labels (as they are now) do not have yet, at least as sure as asked: worth taking.
  const current = editing ? boxes : revision && !foreign(revision) ? revision.boxes : [];
  const missingOf = (run: PredictionRun) =>
    (run.labels ?? [])
      .map((label, i) => ({ label, score: run.scores?.[i] ?? 0 }))
      .filter(({ score }) => score >= minScore)
      .filter(({ label }) =>
        task === "classify"
          ? current[0]?.[0] !== label[0]
          : !current.some((b) => b[0] === label[0] && b.length >= 5 && label.length >= 5 && iou(b, label) >= 0.5),
      )
      .sort((a, b) => b.score - a.score);

  const shown = editRevision ? [editRevision] : [revision, compare].filter((r): r is Revision => r !== null);
  const anyForeign = shown.some(foreign) && classes.length > 0;
  const editVisible = new Set(boxes.map((b, i) => (editRevision && shownBox(editRevision, b) ? i : -1)).filter((i) => i >= 0));
  const editState = saving ? "저장 중…" : dirty ? "저장 안 됨" : "저장됨";
  const picked = selected != null ? boxes[selected] : null;
  const drawTool = kind === "outline" ? "윤곽 그리기" : kind === "pose" ? "새로 그리기" : "박스 그리기";
  const canvasStyle = { transform: `translate(${pan[0]}px, ${pan[1]}px) scale(${zoom})` };
  const panCursor = grabbing ? "grabbing" : zoom > 1 && (space || !editing) ? "grab" : undefined;

  const hint = (() => {
    if (task === "classify") return "이 이미지의 클래스를 고르세요. 숫자키 1~9로도 고르고, ← →로 저장하고 넘깁니다.";
    if (placing != null && picked) return `"${keypointNames(picked[0])[placing] ?? `점 ${placing + 1}`}" 자리를 누르세요. Esc로 그만 놓습니다.`;
    if (drawing && kind === "outline")
      return sketch.length >= 3 ? "점을 더 찍거나, 첫 점을 누르거나 Enter로 닫습니다. Backspace는 마지막 점을 지웁니다." : "윤곽을 따라 눌러 점을 찍습니다.";
    if (drawing && kind === "pose") return "끌어서 박스를 그리면, 이어서 점을 차례로 놓습니다.";
    if (drawing) return "끌어서 박스를 그립니다. 아래에서 고른 클래스로 그려집니다.";
    if (picked && kind === "outline") return "점을 끌어 옮기고, 선을 두 번 눌러 점을 더합니다. 점을 고르고 Delete로 그 점을 지웁니다.";
    if (picked && kind === "pose") return "점을 끌어 옮깁니다. 아래 목록에서 점마다 보임 · 가려짐 · 없음을 정하고, 없는 점은 눌러서 놓습니다.";
    if (picked) return "끌어서 옮기거나 모서리로 크기를 바꿉니다. 클래스는 아래에서 바꿉니다.";
    return "라벨을 눌러 고릅니다. 휠로 확대하고, 확대한 뒤에는 스페이스나 휠 버튼을 누른 채 끌어 옮깁니다.";
  })();

  return (
    <div className={`viewer ${editing ? "editing" : ""}`} role="dialog" aria-modal="true" aria-label="항목 보기">
      <div className="viewer-stage" ref={stageRef}>
        {error && <div className="error">{error}</div>}
        {item && (
          <div
            className="viewer-frames"
            style={{ cursor: panCursor }}
            onPointerDown={panDown}
            onPointerMove={panMove}
            onPointerUp={panUp}
            onPointerCancel={panUp}
            onMouseDown={(e) => e.button === 1 && e.preventDefault()}
          >
            {(shown.length ? shown : [null]).map((r, n) => (
              <figure key={r?.id ?? n} className="viewer-figure">
                <div className="viewer-frame" style={frameStyle}>
                  <div className="viewer-canvas" style={canvasStyle}>
                    {/* the thumbnail at once, the original over it as soon as it is here */}
                    {item.thumb_url && full !== "ok" && <img src={item.thumb_url} alt="" className="placeholder" />}
                    {item.image_url && full !== "error" && (
                      <img
                        key={item.image_url}
                        src={item.image_url}
                        alt={item.file_name}
                        className={full === "ok" ? "" : "loading"}
                        draggable={false}
                        onLoad={() => setFull("ok")}
                        onError={() => setFull("error")}
                      />
                    )}
                    {(showLabels || editing) && r && task !== "classify" && drawBoxes(r)}
                    {showPred && n === 0 && task !== "classify" && overlays.map(({ run, style }) => drawPredictions(run.labels ?? [], run.scores ?? [], style))}
                    {flagged &&
                      showFlagged &&
                      !editing &&
                      n === 0 &&
                      r &&
                      item &&
                      r.id === item.revision_id &&
                      drawFlagged(r.boxes, flagged.boxes)}
                    {showDiff && changeCount > 0 && n === 0 && baseRevision && afterLabels && drawChanges(baseRevision.boxes, afterLabels)}
                    {editing && shapes && fitWidth > 0 && (
                      <EditLayer
                        width={w}
                        height={h}
                        task={task}
                        boxes={boxes}
                        visible={editVisible}
                        selected={selected}
                        point={point}
                        drawing={drawing}
                        sketch={sketch}
                        placing={placing}
                        drawClass={drawClass}
                        keypointCount={keypointNames(drawClass).length}
                        scale={w / (fitWidth * zoom)}
                        ignore={space || grabbing}
                        colorOf={colorOf}
                        onSelect={(i, at = null) => select(i, at)}
                        onChange={setBoxes}
                        onCommit={commit}
                        onSketch={setSketch}
                        onClose={closeSketch}
                        onDrawn={(i) => {
                          select(i);
                          if (kind === "pose" && keypointNames(drawClass).length) setPlacing(0);
                        }}
                        onPlaced={setPlacing}
                      />
                    )}
                  </div>
                  {/* outside the zoomed canvas: it keeps its size */}
                  {(showLabels || editing) && r && task === "classify" && classBadge(r)}
                  {showPred &&
                    n === 0 &&
                    task === "classify" &&
                    overlays.map(
                      ({ run, style }, k) =>
                        run.labels?.[0] &&
                        (run.scores?.[0] ?? 0) >= minScore && (
                          <span
                            key={run.op_id}
                            className="class-badge predicted"
                            style={{ color: colorOf(run.labels[0][0]), top: `${46 + k * 30}px` }}
                          >
                            {style.letter} {className(run.labels[0][0])} {Math.round((run.scores?.[0] ?? 0) * 100)}%
                          </span>
                        ),
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
          <span className="zoom-controls">
            <button className="icon-btn" onClick={() => zoomAt(1 / 1.4)} disabled={zoom <= 1} aria-label="축소" title="축소 (−, 휠)">
              −
            </button>
            <button className="zoom-level" onClick={fit} title="화면에 맞추기 (0)">
              {Math.round(zoom * 100)}%
            </button>
            <button className="icon-btn" onClick={() => zoomAt(1.4)} disabled={zoom >= MAX_ZOOM} aria-label="확대" title="확대 (+, 휠)">
              +
            </button>
          </span>
        </div>
      </div>

      <aside className="viewer-panel">
        <header>
          <strong title={item?.file_name}>{item?.file_name ?? "…"}</strong>
          <button className="icon-btn" onClick={close} aria-label="닫기">
            <Icon name="x" />
          </button>
        </header>

        {item && qaTitle && (
          <section className="qa-panel">
            <div className="eyebrow">검수</div>
            <div>
              <strong>{qaTitle}</strong>
              {flagged?.note && <span className="muted"> · {flagged.note}</span>}
            </div>
            {flagged && flagged.boxes.length > 0 && (
              <>
                <ol className="qa-labels">
                  {flagged.boxes.map((i, n) => {
                    const b = item.revisions.find((r) => r.id === item.revision_id)?.boxes[i];
                    if (!b) return null;
                    return (
                      <li key={i}>
                        <span className="qa-number">{n + 1}</span>
                        <i className="class-dot" style={{ background: colorOf(b[0]) }} />
                        <strong>{className(b[0])}</strong>
                        {b.length >= 5 && (
                          <span className="muted">
                            {" "}
                            {Math.round(b[3] * w)}×{Math.round(b[4] * h)}px
                          </span>
                        )}
                      </li>
                    );
                  })}
                </ol>
                <p className="muted small">이미지 위에 같은 번호와 클래스로 노란 테두리를 쳤습니다(이 버전의 라벨을 볼 때).</p>
              </>
            )}
            {qaCheck && !editing && (
              qaLetBe ? (
                <div className="qa-judge">
                  <span className="small">
                    문제 없음 · {letBe?.by ?? "?"}
                    {letBe?.note ? ` · ${letBe.note}` : ""}
                  </span>
                  <span className="spacer" />
                  <button className="small" onClick={() => void judge()} disabled={judging} title="이 검사에 다시 걸리게 합니다">
                    다시 검사에 넣기 <kbd>O</kbd>
                  </button>
                </div>
              ) : (
                <div className="qa-judge">
                  <input
                    value={letBeNote}
                    onChange={(e) => setLetBeNote(e.target.value)}
                    onKeyDown={(e) => e.stopPropagation()}
                    placeholder="이유(선택): 예) 실제로 겹친 두 물체"
                    aria-label="문제 없음의 이유"
                  />
                  <button className="small" onClick={() => void judge()} disabled={judging} title="라벨이 맞습니다: 이 라벨이 그대로인 동안 다음 버전에서도 세지 않습니다">
                    <Icon name="check" size={12} /> 문제 없음 <kbd>O</kbd>
                  </button>
                </div>
              )
            )}
          </section>
        )}

        {item && labeling && (
          <section className="labeling-panel">
            <div className="eyebrow">
              {labeling.mode === "review" ? "검수" : "라벨링"} · {labeling.name} · 일감 {labeling.part + 1}
            </div>
            {progress && (
              <div className="small">
                {labeling.review
                  ? `승인 ${progress.states.approved ?? 0} · 검수 대기 ${progress.states.labelled ?? 0} · 돌려받음 ${progress.states.rejected ?? 0} · 남음 ${progress.states.todo ?? 0}`
                  : `끝남 ${(progress.states.approved ?? 0) + (progress.states.labelled ?? 0)} / ${progress.size}`}
              </div>
            )}
            {inSet && (
              <div>
                이 이미지: <span className={`tag set-tag set-${inSet.state}`}>{SET_STATE_LABEL[inSet.state]}</span>
              </div>
            )}
            {inSet?.note && <div className="notice small">돌려받은 이유: {inSet.note}</div>}
            {labeling.instructions && (
              <details className="instructions" open={labeling.mode === "label"}>
                <summary>지시문</summary>
                <p className="note small">{labeling.instructions}</p>
              </details>
            )}
            {labeling.mode === "label" ? (
              <button className="primary" onClick={() => void finishAndNext()} disabled={saving}>
                완료하고 다음 <kbd>Enter</kbd>
              </button>
            ) : (
              <div className="row-actions">
                <button className="primary" onClick={() => void reviewItem(true)} disabled={saving}>
                  승인 <kbd>A</kbd>
                </button>
                <button className="danger" onClick={() => void reviewItem(false)} disabled={saving}>
                  돌려보내기 <kbd>R</kbd>
                </button>
              </div>
            )}
          </section>
        )}

        {item && !editing && revision && revision.source.startsWith("model:") && (
          <section className="model-labels">
            <div className="notice small">
              지금 라벨은 <strong>모델이 그린 것</strong>입니다 ({revision.source.slice(6)}, 오토라벨에서 확신도가 높아 바로 넣음). 아직 사람이
              확인하지 않았습니다. 고치면 사람이 고친 라벨이 됩니다.
            </div>
          </section>
        )}

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
            {shapes && (
              <div className="segmented tool-switch" role="group" aria-label="도구">
                <button className={!drawing ? "active" : ""} onClick={() => toggleDrawing(false)} aria-pressed={!drawing}>
                  고르기 <kbd>V</kbd>
                </button>
                <button className={drawing ? "active" : ""} onClick={() => toggleDrawing(true)} aria-pressed={drawing}>
                  {drawTool} <kbd>B</kbd>
                </button>
              </div>
            )}
            <div className="muted small edit-hint">{hint}</div>
            {prefilled?.item === item.id && (
              <div className="notice small autolabel-start">
                {prefilled.model}이 그린 라벨 {prefilled.n}개에서 시작합니다. 틀린 것을 고치고 저장하세요.{" "}
                <button
                  className="link small"
                  onClick={() => {
                    setBoxes([]);
                    setPrefilled({ ...prefilled, n: 0 });
                  }}
                >
                  모두 지우고 처음부터
                </button>
              </div>
            )}
            {drawing && kind === "outline" && sketch.length >= 3 && (
              <button className="small" onClick={closeSketch}>
                윤곽 닫기 <kbd>Enter</kbd>
              </button>
            )}
            <ClassPicker
              classes={classes}
              current={task === "classify" ? (boxes[0]?.[0] ?? null) : picked ? picked[0] : drawClass}
              colorOf={colorOf}
              onPick={pickClass}
            />
            {kind === "pose" && picked && keypointList(picked).length > 0 && (
              <div>
                <div className="eyebrow">점</div>
                <KeypointList
                  names={keypointNames(picked[0])}
                  points={keypointList(picked)}
                  picked={point}
                  placing={placing}
                  colorOf={colorOf(picked[0])}
                  onPick={(i) => {
                    setPoint(i);
                    setPlacing(null);
                  }}
                  onSeen={setSeen}
                  onPlace={(i) => {
                    toggleDrawing(false);
                    setPlacing(i);
                  }}
                />
              </div>
            )}
            {kind === "outline" && picked && (
              <p className="muted small">윤곽 점 {outlineOf(picked).length}개{point != null && ` · ${point + 1}번 점 고름`}</p>
            )}
            <div className="row-actions">
              <button onClick={undoStep} disabled={!undo.length} title="Ctrl+Z">
                되돌리기
              </button>
              <button onClick={redoStep} disabled={!redo.length} title="Ctrl+Shift+Z">
                다시 하기
              </button>
              {selected != null && point != null && kind === "outline" && (
                <button onClick={removePoint} title="Delete">
                  점 지우기
                </button>
              )}
              {selected != null && (
                <button onClick={removeSelected} title={point == null ? "Delete" : undefined}>
                  {task === "classify" ? "지우기" : "라벨 지우기"}
                </button>
              )}
              {task !== "classify" && (
                <button onClick={toBackground} disabled={!boxes.length} title="라벨을 모두 지우고 '찾을 것이 없는 이미지'로">
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
            <p className="muted small">← → 저장하고 넘기기 · Esc 고르기 풀기 · Ctrl+C / Ctrl+V 복사 · 휠 확대 · 0 맞추기</p>
          </section>
        )}

        {item && baseRevision && afterLabels && (
          <section className="change-panel">
            <div className="eyebrow">바뀐 것 · 바탕 버전과 비교</div>
            {changeCount === 0 ? (
              <p className="muted small">바탕 버전의 라벨과 같습니다.</p>
            ) : (
              <>
                <div className="small">
                  {(Object.keys(CHANGE_LABEL) as (keyof typeof CHANGE_LABEL)[])
                    .filter((k) => changes && changes[k].length > 0)
                    .map((k, n) => (
                      <span key={k} className={`change-${k}`}>
                        {n > 0 && " · "}
                        {CHANGE_LABEL[k]} <strong>{changes?.[k].length}</strong>
                      </span>
                    ))}
                </div>
                <label className="check">
                  <input type="checkbox" checked={showDiff} onChange={(e) => setShowDiff(e.target.checked)} /> 바뀐 것 보기 <kbd>D</kbd>
                </label>
              </>
            )}
          </section>
        )}

        {item && runs && (runs.length > 0 || checkOp) && (
          <section className="prediction-panel">
            <div className="eyebrow">모델 예측 · 이 이미지를 예측한 실행 {runs.length}개</div>
            {checkOp && !pred && <p className="muted small">이 이미지는 정렬에 쓴 평가에 들지 않았습니다.</p>}
            {runs.length > 0 && (
              <>
                <label className="check">
                  <input type="checkbox" checked={showPred} onChange={(e) => setShowPred(e.target.checked)} /> 예측 겹쳐 보기{" "}
                  <kbd>P</kbd>
                  <span className="muted small"> · 라벨은 실선, 예측은 점선 (셋까지)</span>
                </label>
                <div className="pred-runs">
                  {runs.map((run) => {
                    const at = shownRuns.indexOf(run.op_id);
                    const style = at >= 0 ? RUN_STYLES[at] : null;
                    return (
                      <label key={run.op_id} className={`check pred-run ${style ? "on" : ""}`} title={runName(run)}>
                        <input type="checkbox" checked={Boolean(style)} onChange={(e) => pickRun(run.op_id, e.target.checked)} />
                        {style ? (
                          <span className="pred-key">
                            <svg width="22" height="10" aria-hidden>
                              <line x1="1" y1="5" x2="21" y2="5" stroke="currentColor" strokeWidth="2" strokeDasharray={style.dash} />
                            </svg>
                            <b>{style.letter}</b>
                          </span>
                        ) : (
                          <span className="pred-key" />
                        )}
                        <span>
                          {run.model ? `${run.model.name} v${run.model.version}` : "모델"}
                          <span className="muted small">
                            {" "}
                            · {run.kind === "eval" ? `평가${run.split ? `(${run.split})` : ""}` : `오토라벨${run.set?.name ? ` · ${run.set.name}` : ""}`} ·{" "}
                            {run.created_at.slice(0, 10)} · 예측 {(run.scores ?? []).filter((v) => v >= minScore).length}개
                          </span>
                        </span>
                      </label>
                    );
                  })}
                </div>
                <label className="pred-score">
                  <span className="small">
                    확신도 <strong>{Math.round(minScore * 100)}%</strong> 이상만
                  </span>
                  <input type="range" min={0} max={0.95} step={0.05} value={minScore} onChange={(e) => pickScore(Number(e.target.value))} />
                </label>
                {overlays.map(({ run, style }) => {
                  const missing = missingOf(run);
                  return (
                    <div key={run.op_id} className="pred-run-detail">
                      <div className="small">
                        <b>{style.letter}</b> {runName(run, true)}
                      </div>
                      {run.kind === "eval" && (
                        <p className="muted small">
                          평가할 때 맞춘 라벨 {run.found ?? 0} · 못 찾은 라벨 {run.missed ?? 0} · 라벨에 없는 예측 {run.extra ?? 0}
                        </p>
                      )}
                      {missing.length > 0 ? (
                        <>
                          <div className="muted small">지금 라벨에 없는 예측 {missing.length}개</div>
                          <div className="prediction-list">
                            {missing.map(({ label, score }, i) => (
                              <div key={i} className="prediction-row">
                                <i style={{ borderColor: colorOf(label[0]) }} />
                                <span>
                                  {className(label[0])} · {Math.round(score * 100)}%
                                </span>
                                {item.edit.allowed && (
                                  <button onClick={() => takePredictions([label])}>{task === "classify" ? "이 클래스로" : "라벨로 넣기"}</button>
                                )}
                              </div>
                            ))}
                          </div>
                          {missing.length > 1 && task !== "classify" && item.edit.allowed && (
                            <button className="small" onClick={() => takePredictions(missing.map((m) => m.label))}>
                              모두 라벨로 넣기
                            </button>
                          )}
                        </>
                      ) : (
                        <p className="muted small">이 확신도 이상의 예측은 모두 지금 라벨에 있습니다.</p>
                      )}
                    </div>
                  );
                })}
              </>
            )}
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
          {flagged && flagged.boxes.length > 0 && (
            <label className="check" title="검수가 걸린 라벨에 두른 노란 테두리와 번호">
              <input type="checkbox" checked={showFlagged} onChange={(e) => setShowFlagged(e.target.checked)} /> 검수 표시{" "}
              <kbd>Q</kbd>
            </label>
          )}
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

        {item && shownRevision && task !== "classify" && shownRevision.boxes.length > 0 && (
          <LabelList
            revision={shownRevision}
            order={counts.map(([name]) => name)}
            width={w}
            height={h}
            colorIn={colorIn}
            classHidden={(name) => hidden.has(name)}
            off={(i) => boxOff(shownRevision, i)}
            active={hoverBox ?? focusBox}
            picked={editing ? selected : focusBox}
            editing={editing}
            onHover={setHoverBox}
            onPick={(i) => (editing ? select(i) : setFocusBox(focusBox === i ? null : i))}
            onToggle={(i) => {
              const key = `${shownRevision.id}:${i}`;
              const next = new Set(hiddenBoxes);
              if (next.has(key)) next.delete(key);
              else next.add(key);
              setHiddenBoxes(next);
              if (focusBox === i) setFocusBox(null);
            }}
            onShowAll={() => setHiddenBoxes(new Set())}
          />
        )}

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

/** Where a label sits in the image, in words: 왼쪽 위, 가운데, 오른쪽 아래 … */
function placeOf(cx: number, cy: number): string {
  const col = cx < 1 / 3 ? "왼쪽" : cx < 2 / 3 ? "가운데" : "오른쪽";
  const row = cy < 1 / 3 ? "위" : cy < 2 / 3 ? "가운데" : "아래";
  if (col === "가운데") return row === "가운데" ? "가운데" : `가운데 ${row}`;
  return row === "가운데" ? col : `${col} ${row}`;
}

/** Every label of the labels shown, one row each, in words: its class and its number within the class (top to
 * bottom, left to right), where it is and how big. Pointing at a row makes its label blink on the image; a row turns
 * its label off on its own (while viewing) or picks it (while fixing). */
function LabelList({
  revision,
  order,
  width,
  height,
  colorIn,
  classHidden,
  off,
  active,
  picked,
  editing,
  onHover,
  onPick,
  onToggle,
  onShowAll,
}: {
  revision: Revision;
  order: string[]; // classes as the chips list them
  width: number;
  height: number;
  colorIn: (r: Revision, cls: number) => string;
  classHidden: (name: string) => boolean;
  off: (index: number) => boolean;
  active: number | null;
  picked: number | null;
  editing: boolean;
  onHover: (index: number | null) => void;
  onPick: (index: number) => void;
  onToggle: (index: number) => void;
  onShowAll: () => void;
}) {
  const names = Object.fromEntries(revision.classes.map((c) => [c.id, c.name])) as Record<number, string>;
  const rows = revision.boxes
    .map((b, i) => ({ b, i, name: names[b[0]] ?? String(b[0]) }))
    .filter(({ b }) => b.length >= 5)
    .sort((a, z) => order.indexOf(a.name) - order.indexOf(z.name) || a.b[2] - z.b[2] || a.b[1] - z.b[1]);
  const seen = new Map<string, number>();
  const numbered = rows.map((row) => {
    const n = (seen.get(row.name) ?? 0) + 1;
    seen.set(row.name, n);
    return { ...row, n };
  });
  const turnedOff = rows.filter(({ i }) => off(i)).length;
  return (
    <section className="label-list-panel">
      <div className="label-list-head">
        <div className="eyebrow">라벨 {rows.length}개</div>
        {!editing && turnedOff > 0 && (
          <button className="link small" onClick={onShowAll}>
            끈 {turnedOff}개 다시 보이기
          </button>
        )}
      </div>
      <div className="label-list" onMouseLeave={() => onHover(null)}>
        {numbered.map(({ b, i, name, n }) => {
          const [, cx, cy, bw, bh] = b;
          const classOff = classHidden(name);
          const isOff = classOff || off(i);
          const color = colorIn(revision, b[0]);
          return (
            <div
              key={i}
              className={`label-row ${isOff ? "off" : ""} ${active === i ? "lit" : ""} ${picked === i ? "picked" : ""}`}
              onMouseEnter={() => onHover(i)}
            >
              <button className="label-pick" onClick={() => onPick(i)} title={editing ? "이 라벨 고르기" : "이 라벨을 계속 깜박이게 (다시 누르면 풀기)"}>
                <i style={{ background: color }} />
                <span className="label-name">
                  {name} <span className="muted">{n}</span>
                </span>
                <span className="muted small label-where">
                  {placeOf(cx, cy)} · {Math.round(bw * width)}×{Math.round(bh * height)}
                </span>
              </button>
              {!editing && (
                <button
                  className="icon-btn label-eye"
                  onClick={() => onToggle(i)}
                  disabled={classOff}
                  aria-pressed={!isOff}
                  title={classOff ? "클래스가 꺼져 있습니다 (위의 클래스 칩)" : isOff ? "이 라벨 보이기" : "이 라벨만 끄기"}
                  aria-label={`${name} ${n} ${isOff ? "보이기" : "끄기"}`}
                >
                  <Icon name={isOff ? "eye-off" : "eye"} size={15} />
                </button>
              )}
            </div>
          );
        })}
      </div>
    </section>
  );
}
