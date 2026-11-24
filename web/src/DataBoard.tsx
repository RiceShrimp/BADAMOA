import { LayoutSwitch, layoutClass, useCardLayout } from "./CardLayout";
import { type FormEvent, type KeyboardEvent as KeyEvent, type ReactNode, useEffect, useMemo, useRef, useState } from "react";
import { api } from "./api";
import {
  SPLIT_LABEL,
  TASK_HINT,
  TASK_LABEL,
  type TrashInfo,
  announceTrash,
  bytesLabel,
  classColor,
  eachOf,
  keypointsOf,
  taskLabel,
} from "./catalogTypes";
import { MoveDatasetDialog } from "./DatasetMove";
import { ImportProgress, useImports } from "./Imports";
import { OrderReset, SortHandle, SortItem, Sortable, useOrder } from "./Order";
import { Icon, Modal, Stat, timeAgo, usePolling } from "./ui";

// ---- what the Hub sends ---------------------------------------------------------------

export interface Shot {
  url: string;
  width: number;
  height: number;
  boxes: number[][]; // [class, cx, cy, w, h] as fractions
  task?: string; // its dataset's: what follows each box (an outline, keypoints)
}

export interface ProjectCardInfo {
  id: string;
  name: string;
  tasks: string[]; // its datasets' tasks, oldest dataset first
  folder: string;
  description: string;
  created_at: string;
  updated_at: string;
  version_at: string | null; // its newest version
  run_at: string | null; // its newest training
  datasets: number;
  releases: number;
  items: number;
  runs: number;
  models: number;
  best: { id: string; name: string; metric: string; value: number } | null;
  collage: Shot[];
  /** Classes with labels in the newest version of each dataset, most labels first. */
  classes: { name: string; labels: number; images: number }[];
  /** A few of its datasets, the one with the newest version first: a line each under the card. */
  recent: RecentDataset[];
  recent_total: number; // datasets with a version
}

interface RecentDataset {
  id: string;
  name: string;
  task: string;
  release: { id: string; version: string; created_at: string };
  items: number;
}

interface DatasetCardInfo {
  id: string;
  name: string;
  task: string;
  release: { id: string; version: string; created_at: string; note: string };
  releases: number;
  draft: { id: string; edited: number } | null; // fixed labels not yet made a version
  items: number;
  bytes: number;
  boxes: number;
  classes: number;
  top_classes: { id: number; name: string; color?: string | null }[];
  splits: Record<string, number>;
  states: Record<string, number>;
  collage: Shot[];
}

interface ProjectPageInfo extends ProjectCardInfo {
  metric: string;
  dataset_cards: DatasetCardInfo[];
  model_groups: ModelGroup[];
  sweeps: number;
}

/** One task's models, best first by that task's own metric (a Top-1 and an mAP do not compare). */
interface ModelGroup {
  task: string;
  metric: string;
  models: { id: string; name: string; value: number | null; aliases: string[] }[];
}

export const METRIC_NAME: Record<string, string> = { map50_95: "mAP50-95", mask_map50_95: "mask mAP50-95", top1: "Top-1" };
export const SPLIT_COLOR: Record<string, string> = { train: "var(--series-3)", val: "var(--series-1)", test: "var(--series-7)", "": "var(--idle)" };
const PROJECT_KEY = "badamoa.project"; // the project other screens open with

export function rememberProject(id: string) {
  try {
    localStorage.setItem(PROJECT_KEY, id);
  } catch {
    /* not remembered */
  }
}

// ---- a thumbnail with its boxes, cropped to fill its cell -------------------------------

/**
 * A thumbnail with its labels: boxes; outlines for segmentation; points and limbs for keypoints.
 * What follows a box ([class, cx, cy, w, h, …]) depends on the project's task.
 */
export function Shot({
  shot,
  colorOf,
  boxes = true,
  task = shot.task ?? "detect",
  skeletonOf,
  predicted,
  minScore = 0,
}: {
  shot: Shot;
  colorOf: (id: number) => string;
  boxes?: boolean;
  task?: string;
  skeletonOf?: (id: number) => number[][] | undefined;
  /** A model run's predictions over the thumbnail: white dashes on a dark edge (labels are solid). */
  predicted?: { labels: number[][]; scores: number[] };
  minScore?: number;
}) {
  const w = shot.width || 1;
  const h = shot.height || 1;
  const line = Math.max(w, h) / 140;
  return (
    <svg className="shot" viewBox={`0 0 ${w} ${h}`} preserveAspectRatio="xMidYMid slice" aria-hidden="true">
      <image href={shot.url} width={w} height={h} preserveAspectRatio="xMidYMid slice" />
      {boxes &&
        shot.boxes.map((b, i) => {
          if (b.length < 5) return null;
          const color = colorOf(b[0]);
          const points = task === "pose" ? keypointsOf(b) : [];
          const outline = task === "segment" && b.length >= 11 ? b.slice(5) : [];
          return (
            <g key={i}>
              {outline.length > 0 && (
                <polygon
                  points={outline.map((v, n) => (n % 2 === 0 ? v * w : v * h)).join(" ")}
                  fill={color}
                  fillOpacity={0.25}
                  stroke={color}
                  strokeWidth={line}
                />
              )}
              <rect
                x={(b[1] - b[3] / 2) * w}
                y={(b[2] - b[4] / 2) * h}
                width={b[3] * w}
                height={b[4] * h}
                fill="none"
                stroke={color}
                strokeWidth={outline.length || points.length ? line / 2 : line}
                strokeDasharray={outline.length || points.length ? `${line * 3} ${line * 2}` : undefined}
              />
              {(skeletonOf?.(b[0]) ?? []).map(([a, c], n) =>
                points[a] && points[c] && points[a][2] > 0 && points[c][2] > 0 ? (
                  <line
                    key={`l${n}`}
                    x1={points[a][0] * w}
                    y1={points[a][1] * h}
                    x2={points[c][0] * w}
                    y2={points[c][1] * h}
                    stroke={color}
                    strokeWidth={line}
                  />
                ) : null,
              )}
              {points.map(([x, y, seen], n) =>
                seen > 0 ? <circle key={`p${n}`} cx={x * w} cy={y * h} r={Math.min(line * 1.6, Math.max(line * 0.6, Math.min(b[3] * w, b[4] * h) * 0.045))} fill={seen === 2 ? color : "none"} stroke={color} strokeWidth={line / 2} /> : null,
              )}
            </g>
          );
        })}
      {predicted?.labels.map((b, i) => {
        if (b.length < 5 || (predicted.scores[i] ?? 0) < minScore) return null;
        const box = { x: (b[1] - b[3] / 2) * w, y: (b[2] - b[4] / 2) * h, width: b[3] * w, height: b[4] * h };
        return (
          <g key={`pred${i}`}>
            <rect {...box} fill="none" stroke="rgb(0 0 0 / 0.65)" strokeWidth={line * 1.6} />
            <rect {...box} fill="none" stroke="#fff" strokeWidth={line * 0.8} strokeDasharray={`${line * 3} ${line * 2}`} />
          </g>
        );
      })}
    </svg>
  );
}

export function Collage({
  shots,
  colorOf,
  more,
  title,
}: {
  shots: Shot[];
  colorOf: (id: number) => string;
  more?: number;
  title?: string; // what the pictures are, on hover
}) {
  return (
    <div className="collage" title={title}>
      {[0, 1, 2, 3].map((i) =>
        shots[i] ? (
          <div key={i} className="collage-cell">
            <Shot shot={shots[i]} colorOf={colorOf} />
            {i === 3 && more != null && more > 4 && <span className="collage-more">+{(more - 4).toLocaleString()}</span>}
          </div>
        ) : (
          <div key={i} className="collage-cell empty">
            {i === 0 && shots.length === 0 && <Icon name="images" size={20} />}
          </div>
        ),
      )}
    </div>
  );
}

function SplitBar({ splits }: { splits: Record<string, number> }) {
  const order = ["train", "val", "test", ""].filter((s) => splits[s]);
  const total = order.reduce((n, s) => n + splits[s], 0) || 1;
  return (
    <>
      <div className="split-bar">
        {order.map((s) => (
          <i key={s} style={{ width: `${(splits[s] / total) * 100}%`, background: SPLIT_COLOR[s] }} />
        ))}
      </div>
      <div className="split-legend">
        {order.map((s) => (
          <span key={s}>
            <i style={{ background: SPLIT_COLOR[s] }} />
            {SPLIT_LABEL[s] ?? s} {splits[s].toLocaleString()}
          </span>
        ))}
      </div>
    </>
  );
}

const SORTS = [
  { id: "updated", label: "최근 수정순" },
  { id: "name", label: "이름순" },
  { id: "items", label: "이미지 많은 순" },
];
const MINE = { id: "mine", label: "내가 끌어 놓은 순서" }; // once this person has dragged the cards
const BY: Record<string, (a: ProjectCardInfo, b: ProjectCardInfo) => number> = {
  updated: (a, b) => b.updated_at.localeCompare(a.updated_at),
  name: (a, b) => a.name.localeCompare(b.name, "ko"),
  items: (a, b) => b.items - a.items,
};
const cardId = (c: { id: string }) => c.id;

// ---- a project's folder: one there is, none, or a new one typed in --------------------------

const NEW_FOLDER = "\u0000new";

// ---- picking several cards to act on together (to the trash) ----------------------------------

/** Cards picked by id while picking is on; Esc stops. */
function usePicking() {
  const [on, setOn] = useState(false);
  const [ids, setIds] = useState<string[]>([]);
  const stop = () => {
    setOn(false);
    setIds([]);
  };
  useEffect(() => {
    if (!on) return;
    const esc = (e: KeyboardEvent) => e.key === "Escape" && stop();
    window.addEventListener("keydown", esc);
    return () => window.removeEventListener("keydown", esc);
  }, [on]);
  const toggle = (id: string) => setIds((cur) => (cur.includes(id) ? cur.filter((x) => x !== id) : [...cur, id]));
  return { on, ids, setIds, start: () => setOn(true), stop, toggle };
}

interface Picking {
  selected: boolean;
  onToggle: () => void;
}

/** A card while picking: the whole card is a checkbox. */
function pickProps(p: Picking, name: string) {
  return {
    role: "checkbox",
    "aria-checked": p.selected,
    "aria-label": `${name} 선택`,
    tabIndex: 0,
    onClick: p.onToggle,
    onKeyDown: (e: KeyEvent) => {
      if (e.key === " " || e.key === "Enter") {
        e.preventDefault();
        p.onToggle();
      }
    },
  };
}

function CardCheck({ selected }: { selected: boolean }) {
  return (
    <span className="card-check" aria-hidden="true">
      {selected && <Icon name="check" size={12} />}
    </span>
  );
}

/** Above the cards while picking: how many, all or none, what to do with them. */
function PickBar({
  count,
  total,
  onAll,
  onNone,
  onCancel,
  children,
}: {
  count: number;
  total: number;
  onAll: () => void;
  onNone: () => void;
  onCancel: () => void;
  children: ReactNode;
}) {
  return (
    <div className="pick-bar" role="toolbar" aria-label="고른 카드">
      <strong>{count}개 고름</strong>
      {count < total ? (
        <button className="link small" onClick={onAll}>
          모두 고르기 ({total})
        </button>
      ) : (
        <button className="link small" onClick={onNone}>
          모두 풀기
        </button>
      )}
      <span className="spacer" />
      {children}
      <button className="ghost-btn" onClick={onCancel}>
        그만 고르기
      </button>
    </div>
  );
}

/** A folder and how many projects (not in the trash) are in it; a folder may be empty. */
export interface FolderInfo {
  name: string;
  projects: number;
}

export function useFolders() {
  return usePolling(() => api<FolderInfo[]>("/folders"), 30000);
}

/** What a folder is, in one line: shown wherever one is made or picked. */
export const FOLDER_HINT = "폴더는 프로젝트 카드를 묶어 보는 칸입니다. 데이터 · 학습 · 권한에는 영향이 없고, 없어도 됩니다.";

export function FolderPicker({ value, folders, onChange }: { value: string; folders: string[]; onChange: (folder: string) => void }) {
  const [typing, setTyping] = useState(value !== "" && !folders.includes(value));
  return (
    <div className="folder-picker">
      <select
        value={typing ? NEW_FOLDER : value}
        aria-label="폴더"
        onChange={(e) => {
          const picked = e.target.value;
          setTyping(picked === NEW_FOLDER);
          onChange(picked === NEW_FOLDER ? "" : picked);
        }}
      >
        <option value="">폴더 없음</option>
        {folders.map((f) => (
          <option key={f} value={f}>
            {f}
          </option>
        ))}
        <option value={NEW_FOLDER}>+ 새 폴더…</option>
      </select>
      {typing && (
        <input value={value} onChange={(e) => onChange(e.target.value)} placeholder="새 폴더 이름" aria-label="새 폴더 이름" required autoFocus />
      )}
    </div>
  );
}

// ---- the first screen: folders and projects -----------------------------------------------

/**
 * Projects as cards, grouped by folder. A folder is a name projects share: making one moves the
 * chosen projects into it, and it goes away when nothing is left in it.
 */
export function ProjectBoard({ onNewProject }: { onNewProject: (folder: string) => void }) {
  const { data: cards, error, refresh } = usePolling(() => api<ProjectCardInfo[]>("/project-cards"), 30000);
  const { data: trash, refresh: refreshTrash } = usePolling(() => api<TrashInfo>("/trash"), 60000);
  const binned = (trash?.projects.length ?? 0) + (trash?.releases.length ?? 0);
  const { data: folderList, refresh: refreshFolders } = useFolders();
  const [folder, setFolder] = useState<string | null>(null); // null: all, "": no folder
  const [q, setQ] = useState("");
  const [task, setTask] = useState(""); // "": every task
  const [wanted, setWanted] = useState<string[]>([]); // class names a project must all have (any case)
  const [filtering, setFiltering] = useState(false);
  const order = useOrder("datasets.projects", cards, cardId);
  const [layout, setLayout] = useCardLayout("data");
  const [sortChoice, setSort] = useState<string | null>(null); // null: the dragged order if there is one, else the newest
  const sort = sortChoice ?? (order.custom ? MINE.id : "updated");
  const [editing, setEditing] = useState<ProjectCardInfo | null>(null);
  const [makingFolder, setMakingFolder] = useState(false);
  const [renaming, setRenaming] = useState(false);
  const [failed, setFailed] = useState<string | null>(null);
  const picking = usePicking();
  const reload = () => {
    void refresh();
    void refreshFolders();
    void refreshTrash();
  };
  const trashPicked = async () => {
    const chosen = (cards ?? []).filter((c) => picking.ids.includes(c.id));
    if (chosen.length === 0) return;
    const lines = chosen.slice(0, 8).map((c) => `· ${c.name} — 데이터셋 ${c.datasets} · 학습 ${c.runs} · 모델 ${c.models}`);
    if (chosen.length > 8) lines.push(`· 그 밖에 ${chosen.length - 8}개`);
    const text = `프로젝트 ${chosen.length}개를 휴지통으로 옮길까요?\n${lines.join("\n")}\n\n휴지통에서 되살릴 수 있고, 이미지 파일은 지워지지 않습니다.`;
    if (!window.confirm(text)) return;
    const failures = await eachOf(chosen, (c) => api(`/projects/${c.id}`, { method: "DELETE" }), (c) => c.name);
    announceTrash();
    picking.stop();
    setFailed(failures.length ? failures.join("\n") : null);
    reload();
  };

  // Chips: every folder (empty ones too), then "no folder" when there are folders and some project is in none.
  const folders = useMemo(() => {
    const named = [...(folderList ?? [])]
      .sort((a, b) => a.name.localeCompare(b.name, "ko"))
      .map((f): [string, number] => [f.name, f.projects]);
    const loose = (cards ?? []).filter((c) => !c.folder).length;
    return named.length > 0 && loose > 0 ? [...named, ["", loose] as [string, number]] : named;
  }, [folderList, cards]);
  const names = folders.map(([f]) => f).filter(Boolean);
  // A folder removed or renamed elsewhere: back to everything.
  useEffect(() => {
    if (folder && folderList && !folderList.some((f) => f.name === folder)) setFolder(null);
  }, [folder, folderList]);

  const removeFolder = async (name: string) => {
    const n = folders.find(([f]) => f === name)?.[1] ?? 0;
    const text =
      n > 0
        ? `${name} 폴더를 지울까요?\n안의 프로젝트 ${n}개는 지워지지 않고 "폴더 없음"으로 옮겨집니다.`
        : `"${name}" 폴더는 비어 있습니다. 지울까요?`;
    if (!window.confirm(text)) return;
    try {
      setFailed(null);
      await api(`/folders?name=${encodeURIComponent(name)}`, { method: "DELETE" });
      setFolder(null);
      reload();
    } catch (e) {
      setFailed(e instanceof Error ? e.message : String(e));
    }
  };
  const shown = useMemo(() => {
    const needle = q.trim().toLowerCase();
    const list = (sort === MINE.id ? order.ordered : (cards ?? [])).filter(
      (c) =>
        (folder === null || c.folder === folder) &&
        (!task || c.tasks.includes(task)) &&
        hasAll(c, wanted) &&
        (!needle || `${c.name} ${c.description} ${c.folder}`.toLowerCase().includes(needle)),
    );
    return sort === MINE.id ? list : [...list].sort(BY[sort]);
  }, [cards, folder, task, wanted, q, sort, order.ordered]);
  // Dragging while sorted another way starts from the order on show: every project, as sorted now.
  const moved = (ids: string[]) => {
    order.move(ids, sort === MINE.id ? undefined : [...(cards ?? [])].sort(BY[sort]).map(cardId));
    setSort(MINE.id);
  };
  const clearFilters = () => {
    setTask("");
    setWanted([]);
  };
  const filters = (task ? 1 : 0) + wanted.length;
  // The tasks there are, with how many projects each: the filter lists only those.
  const tasks = useMemo(() => {
    const counts = new Map<string, number>();
    for (const c of cards ?? []) for (const t of c.tasks) counts.set(t, (counts.get(t) ?? 0) + 1);
    const order = Object.keys(TASK_LABEL); // the built-in ones first, then named ones
    return [...counts.entries()].sort(
      ([a], [b]) => (order.indexOf(a) + 1 || 99) - (order.indexOf(b) + 1 || 99) || a.localeCompare(b),
    );
  }, [cards]);

  if (error && !cards) return <div className="error">{error}</div>;
  return (
    <>
      <div className="page-head">
        <div>
          <h1>데이터셋</h1>
          <p className="muted">
            프로젝트는 같은 클래스 체계를 쓰는 데이터셋 · 학습 · 모델을 묶습니다. 프로젝트가 많아지면 폴더로 나눠 두세요.
          </p>
        </div>
        <div className="head-actions">
          <button className="primary" onClick={() => onNewProject(folder ?? "")}>
            <Icon name="plus" />새 프로젝트
          </button>
        </div>
      </div>

      {cards && (cards.length > 0 || names.length > 0) && (
        <div className="board-tools">
          <div className="folder-chips" role="group" aria-label="폴더">
            <button className={folder === null ? "active" : ""} onClick={() => setFolder(null)}>
              전체 <em>{cards.length}</em>
            </button>
            {folders.map(([name, n]) => (
              <button key={name || "-"} className={folder === name ? "active" : ""} onClick={() => setFolder(name)}>
                <Icon name="folder" size={13} />
                {name || "폴더 없음"} <em>{n}</em>
              </button>
            ))}
            <button className="add-chip" onClick={() => setMakingFolder(true)} title={FOLDER_HINT}>
              <Icon name="plus" size={12} />
              폴더
            </button>
          </div>
          <span className="spacer" />
          <input type="search" value={q} onChange={(e) => setQ(e.target.value)} placeholder="프로젝트 찾기" aria-label="프로젝트 찾기" />
          <div className="filter-anchor">
            <button className={filters ? "filter-btn on" : "filter-btn"} onClick={() => setFiltering(!filtering)} aria-expanded={filtering}>
              <Icon name="filter" size={14} />
              필터{filters > 0 && <em>{filters}</em>}
            </button>
            {filtering && (
              <FilterPanel
                cards={cards}
                tasks={tasks}
                task={task}
                wanted={wanted}
                onTask={setTask}
                onWanted={setWanted}
                onClear={clearFilters}
                onClose={() => setFiltering(false)}
              />
            )}
          </div>
          <select value={sort} onChange={(e) => setSort(e.target.value)} aria-label="정렬">
            {(order.custom ? [MINE, ...SORTS] : SORTS).map((s) => (
              <option key={s.id} value={s.id}>
                {s.label}
              </option>
            ))}
          </select>
          {sort === MINE.id && <OrderReset order={order} onReset={() => setSort(null)} />}
          {!picking.on && shown.length > 0 && (
            <button className="ghost-btn" onClick={picking.start} title="여러 프로젝트를 골라 한 번에 휴지통으로">
              <Icon name="check" size={13} />
              고르기
            </button>
          )}
          <LayoutSwitch value={layout} onChange={setLayout} />
        </div>
      )}
      {filters > 0 && (
        <div className="active-filters">
          {task && (
            <button onClick={() => setTask("")} title="이 조건 빼기">
              {taskLabel(task)} <Icon name="x" size={11} />
            </button>
          )}
          {wanted.map((w) => (
            <button key={w} onClick={() => setWanted(wanted.filter((x) => x !== w))} title="이 조건 빼기">
              {w} <Icon name="x" size={11} />
            </button>
          ))}
          <button className="link small" onClick={clearFilters}>
            필터 지우기
          </button>
        </div>
      )}
      {folder && (
        <div className="folder-bar">
          <Icon name="folder" size={14} />
          <strong>{folder}</strong>
          <span className="muted small">프로젝트 {folders.find(([f]) => f === folder)?.[1] ?? 0}개</span>
          <span className="spacer" />
          <button className="ghost-btn" onClick={() => setRenaming(true)}>
            폴더 이름 바꾸기
          </button>
          <button className="ghost-btn" onClick={() => void removeFolder(folder)}>
            폴더 지우기
          </button>
        </div>
      )}
      {failed && <div className="error lines">{failed}</div>}
      {picking.on && (
        <PickBar
          count={picking.ids.length}
          total={shown.length}
          onAll={() => picking.setIds(shown.map((c) => c.id))}
          onNone={() => picking.setIds([])}
          onCancel={picking.stop}
        >
          <button className="danger" disabled={picking.ids.length === 0} onClick={() => void trashPicked()}>
            <Icon name="trash" size={14} />
            휴지통으로
          </button>
        </PickBar>
      )}
      {cards && cards.length === 0 && binned > 0 && (
        <p className="muted small">
          지운 프로젝트가 있습니다. <a href="#/trash">휴지통 {binned}</a>
        </p>
      )}

      {!cards && <p className="muted">프로젝트를 불러오는 중…</p>}
      {cards && cards.length === 0 && names.length === 0 && (
        <div className="card empty-state">
          <Icon name="database" size={28} />
          <strong>첫 프로젝트를 만들어 보세요</strong>
          <p className="muted">프로젝트를 만든 뒤, 노드에 있는 YOLO · COCO 형식 데이터셋을 복사 없이 등록할 수 있습니다.</p>
          <button className="primary" onClick={() => onNewProject("")}>
            <Icon name="plus" />
            프로젝트 만들기
          </button>
        </div>
      )}

      {cards && (cards.length > 0 || names.length > 0) && shown.length === 0 && (q.trim() || filters > 0) && (
        <p className="muted">
          조건에 맞는 프로젝트가 없습니다.{" "}
          <button
            className="link"
            onClick={() => {
              setQ("");
              clearFilters();
            }}
          >
            찾기 · 필터 지우기
          </button>
        </p>
      )}
      {cards && folder && shown.length === 0 && !q.trim() && filters === 0 && (
        <p className="muted">빈 폴더입니다. 여기서 새 프로젝트를 만들거나, 다른 프로젝트의 설정(톱니바퀴)에서 이 폴더로 옮기세요.</p>
      )}
      <Sortable ids={shown.map(cardId)} onMove={moved} disabled={picking.on}>
        <div className={`card-grid${layoutClass(layout)}`}>
          {shown.map((c) => (
            <SortItem key={c.id} id={c.id}>
              <ProjectCard
                card={c}
                onEdit={() => setEditing(c)}
                highlight={wanted}
                picking={picking.on ? { selected: picking.ids.includes(c.id), onToggle: () => picking.toggle(c.id) } : undefined}
              />
            </SortItem>
          ))}
          {cards && (cards.length > 0 || names.length > 0) && !picking.on && (
            <button className="add-tile" onClick={() => onNewProject(folder ?? "")}>
              <Icon name="plus" size={18} />새 프로젝트{folder ? ` (${folder})` : ""}
            </button>
          )}
        </div>
      </Sortable>

      {editing && (
        <EditProject
          card={editing}
          folders={names}
          onClose={() => {
            setEditing(null);
            reload();
          }}
        />
      )}
      {makingFolder && cards && (
        <NewFolder
          cards={cards}
          onClose={async (made) => {
            setMakingFolder(false);
            reload();
            await refreshFolders(); // the new folder must be in the list before it is opened
            if (made) setFolder(made);
          }}
        />
      )}
      {renaming && folder && (
        <RenameFolder
          name={folder}
          onClose={async (renamed) => {
            setRenaming(false);
            reload();
            await refreshFolders();
            if (renamed) setFolder(renamed);
          }}
        />
      )}
    </>
  );
}

/** Every class name in ``wanted`` has labels in the project (names compared without case). */
function hasAll(card: ProjectCardInfo, wanted: string[]): boolean {
  if (wanted.length === 0) return true;
  const have = new Set((card.classes ?? []).map((c) => c.name.toLowerCase()));
  return wanted.every((w) => have.has(w.toLowerCase()));
}

const COMMON_CLASSES = 12;

/**
 * The less used filters, behind one button: the task, and classes a project must hold.
 * Classes are not all listed (there can be hundreds): the common ones are one click, the rest are typed.
 */
function FilterPanel({
  cards,
  tasks,
  task,
  wanted,
  onTask,
  onWanted,
  onClear,
  onClose,
}: {
  cards: ProjectCardInfo[] | null;
  tasks: [string, number][];
  task: string;
  wanted: string[];
  onTask: (task: string) => void;
  onWanted: (names: string[]) => void;
  onClear: () => void;
  onClose: () => void;
}) {
  const panel = useRef<HTMLDivElement>(null);
  const [typed, setTyped] = useState("");
  // Every class with labels somewhere: in how many projects, how many labels in all.
  const index = useMemo(() => {
    const all = new Map<string, { name: string; projects: number; labels: number }>();
    for (const card of cards ?? []) {
      for (const c of card.classes ?? []) {
        const entry = all.get(c.name.toLowerCase()) ?? { name: c.name, projects: 0, labels: 0 };
        entry.projects += 1;
        entry.labels += c.labels;
        all.set(c.name.toLowerCase(), entry);
      }
    }
    return [...all.values()].sort((a, b) => b.projects - a.projects || b.labels - a.labels || a.name.localeCompare(b.name));
  }, [cards]);
  const picked = new Set(wanted.map((w) => w.toLowerCase()));
  const needle = typed.trim().toLowerCase();
  const found = needle ? index.filter((c) => c.name.toLowerCase().includes(needle) && !picked.has(c.name.toLowerCase())).slice(0, 8) : [];
  const toggle = (name: string) => {
    onWanted(picked.has(name.toLowerCase()) ? wanted.filter((w) => w.toLowerCase() !== name.toLowerCase()) : [...wanted, name]);
  };

  // Closes on a click outside or Escape.
  useEffect(() => {
    const away = (e: MouseEvent) => {
      if (panel.current && !panel.current.parentElement?.contains(e.target as Node)) onClose();
    };
    const key = (e: KeyboardEvent) => e.key === "Escape" && onClose();
    document.addEventListener("mousedown", away);
    window.addEventListener("keydown", key);
    return () => {
      document.removeEventListener("mousedown", away);
      window.removeEventListener("keydown", key);
    };
  }, [onClose]);

  return (
    <div className="filter-panel" ref={panel} role="dialog" aria-label="필터">
      <div className="eyebrow">태스크</div>
      <div className="segmented">
        <button className={task === "" ? "active" : ""} onClick={() => onTask("")}>
          모두
        </button>
        {tasks.map(([t, n]) => (
          <button key={t} className={task === t ? "active" : ""} onClick={() => onTask(t)}>
            {taskLabel(t)} <span className="muted">{n}</span>
          </button>
        ))}
      </div>

      <div className="eyebrow">클래스</div>
      {index.length === 0 ? (
        <p className="muted small">아직 라벨이 있는 데이터셋이 없습니다.</p>
      ) : (
        <>
          <input
            type="search"
            value={typed}
            onChange={(e) => setTyped(e.target.value)}
            onKeyDown={(e) => {
              if (e.key === "Enter" && found[0]) {
                e.preventDefault();
                toggle(found[0].name);
                setTyped("");
              }
            }}
            placeholder="클래스 이름 찾기"
            aria-label="클래스 찾기"
            autoFocus
          />
          {needle && (
            <ul className="class-suggest">
              {found.length === 0 && <li className="muted small">"{typed.trim()}"이 들어간 클래스가 없습니다.</li>}
              {found.map((c) => (
                <li key={c.name}>
                  <button
                    onClick={() => {
                      toggle(c.name);
                      setTyped("");
                    }}
                  >
                    {c.name} <span className="muted small">프로젝트 {c.projects} · 라벨 {c.labels.toLocaleString()}</span>
                  </button>
                </li>
              ))}
            </ul>
          )}
          {wanted.length > 0 && (
            <div className="class-picks">
              {wanted.map((w) => (
                <button key={w} className="on" onClick={() => toggle(w)} title="빼기">
                  {w} <Icon name="x" size={11} />
                </button>
              ))}
            </div>
          )}
          <div className="muted small">많이 쓰는 클래스</div>
          <div className="class-picks">
            {index
              .filter((c) => !picked.has(c.name.toLowerCase()))
              .slice(0, COMMON_CLASSES)
              .map((c) => (
                <button key={c.name} onClick={() => toggle(c.name)} title={`프로젝트 ${c.projects}개 · 라벨 ${c.labels.toLocaleString()}개`}>
                  {c.name} <span className="muted">{c.projects}</span>
                </button>
              ))}
          </div>
          <p className="muted small">고른 클래스가 모두 들어 있는 프로젝트만 보입니다. 라벨이 실제로 있는 클래스만 셉니다(각 데이터셋의 최신 버전 기준).</p>
        </>
      )}
      <div className="filter-panel-foot">
        <button className="link small" onClick={onClear}>
          필터 지우기
        </button>
        <button className="small" onClick={onClose}>
          닫기
        </button>
      </div>
    </div>
  );
}

export function ProjectCard({
  card,
  onEdit,
  highlight = [],
  picking,
}: {
  card: ProjectCardInfo;
  onEdit?: () => void;
  highlight?: string[];
  picking?: Picking; // set while cards are being picked: a click picks instead of opening
}) {
  const matched = highlight
    .map((w) => (card.classes ?? []).find((c) => c.name.toLowerCase() === w.toLowerCase()))
    .filter((c): c is ProjectCardInfo["classes"][number] => Boolean(c));
  const body = (
    <>
        <Collage shots={card.collage} colorOf={classColor} />
        <div className="data-card-body">
          <div className="data-card-title">
            <span className="with-grip"><SortHandle /><strong>{card.name}</strong></span>
          </div>
          {card.tasks.length > 0 && (
            <div className="task-tags">
              <TaskTags tasks={card.tasks} />
            </div>
          )}
          <div className="muted small">
            {card.folder ? `${card.folder} · ` : ""}데이터셋 {card.datasets} · 버전 {card.releases} · 이미지{" "}
            {card.items.toLocaleString()}
          </div>
          <div className="muted small">
            학습 {card.runs} · 모델 {card.models}
            {card.best ? ` · 최고 ${METRIC_NAME[card.best.metric] ?? card.best.metric} ${card.best.value.toFixed(3)}` : ""}
          </div>
          {matched.length > 0 && (
            <div className="card-matched small">
              {matched.map((c) => (
                <span key={c.name}>
                  {c.name} <strong>{c.labels.toLocaleString()}</strong>
                </span>
              ))}
            </div>
          )}
          <div className="muted small">{timeAgo(card.updated_at)} 수정</div>
        </div>
    </>
  );
  return (
    <div className={`data-card${picking ? " picking" : ""}${picking?.selected ? " selected" : ""}`}>
      {picking ? (
        <div className="data-card-link" {...pickProps(picking, card.name)}>
          <CardCheck selected={picking.selected} />
          {body}
        </div>
      ) : (
        <a href={`#/datasets/project/${card.id}`} onClick={() => rememberProject(card.id)} className="data-card-link">
          {body}
        </a>
      )}
      {card.recent?.length > 0 && <DatasetLines card={card} onPick={picking?.onToggle} />}
      {onEdit && !picking && (
        <button className="icon-btn data-card-menu" onClick={onEdit} title="이름 · 폴더 · 설명" aria-label={`${card.name} 설정`}>
          <Icon name="settings" size={14} />
        </button>
      )}
    </div>
  );
}

/**
 * Under a project's card, as 학습 lists its latest runs: its datasets, the one with the newest version first,
 * each opening that version. While cards are picked a line picks its card instead.
 */
function DatasetLines({ card, onPick }: { card: ProjectCardInfo; onPick?: () => void }) {
  const more = card.recent_total - card.recent.length;
  return (
    <div className="board-lines" onClick={onPick}>
      {card.recent.map((d) => {
        const line = (
          <>
            <span className="tag task-tag">{taskLabel(d.task)}</span>
            <span className="ellipsis">
              {d.name} <span className="muted">{d.release.version}</span>
            </span>
            <span className="num muted">{d.items.toLocaleString()}장</span>
          </>
        );
        const title = `${d.name} ${d.release.version} · 이미지 ${d.items.toLocaleString()}장 · ${timeAgo(d.release.created_at)} 등록`;
        return onPick ? (
          <div key={d.id} className="board-line" title={title}>
            {line}
          </div>
        ) : (
          <a key={d.id} href={`#/datasets/${d.release.id}`} className="board-line" title={title} onClick={() => rememberProject(card.id)}>
            {line}
          </a>
        );
      })}
      {more > 0 && <div className="muted small board-foot">데이터셋 {more}개 더</div>}
    </div>
  );
}

function EditProject({
  card,
  folders,
  onClose,
}: {
  card: ProjectCardInfo;
  folders: string[];
  onClose: (deleted?: boolean) => void;
}) {
  const [name, setName] = useState(card.name);
  const [folder, setFolder] = useState(card.folder);
  const [description, setDescription] = useState(card.description);
  const [error, setError] = useState<string | null>(null);
  const save = async (e: FormEvent) => {
    e.preventDefault();
    try {
      await api(`/projects/${card.id}`, { method: "PATCH", body: { name, folder, description } });
      onClose();
    } catch (err) {
      setError(err instanceof Error ? err.message : String(err));
    }
  };
  const remove = async () => {
    const what = `${card.name} · 데이터셋 ${card.datasets} · 학습 ${card.runs} · 모델 ${card.models}`;
    if (!window.confirm(`이 프로젝트를 휴지통으로 옮길까요?\n${what}\n\n휴지통에서 되살릴 수 있고, 이미지 파일은 지워지지 않습니다.`)) return;
    try {
      await api(`/projects/${card.id}`, { method: "DELETE" });
      announceTrash();
      onClose(true);
    } catch (err) {
      setError(err instanceof Error ? err.message : String(err));
    }
  };
  return (
    <Modal title={`${card.name} 설정`} onClose={() => onClose()}>
      <form onSubmit={save}>
        <label>
          이름
          <input value={name} onChange={(e) => setName(e.target.value)} required />
        </label>
        <div className="field-label">
          폴더
          <FolderPicker value={folder} folders={folders} onChange={setFolder} />
        </div>
        <label>
          설명
          <textarea value={description} onChange={(e) => setDescription(e.target.value)} rows={3} />
        </label>
        {error && <div className="error">{error}</div>}
        <div className="modal-actions">
          <button type="button" className="danger push-left" onClick={() => void remove()}>
            휴지통으로
          </button>
          <button type="button" onClick={() => onClose()}>
            취소
          </button>
          <button className="primary">저장</button>
        </div>
      </form>
    </Modal>
  );
}

function NewFolder({ cards, onClose }: { cards: ProjectCardInfo[]; onClose: (made: string | null) => void }) {
  const [name, setName] = useState("");
  const [picked, setPicked] = useState<string[]>([]);
  const [error, setError] = useState<string | null>(null);
  const save = async (e: FormEvent) => {
    e.preventDefault();
    try {
      await api("/folders", { body: { name: name.trim(), projects: picked } });
      onClose(name.trim());
    } catch (err) {
      setError(err instanceof Error ? err.message : String(err));
    }
  };
  return (
    <Modal title="새 폴더" onClose={() => onClose(null)}>
      <form onSubmit={save}>
        <p className="muted">{FOLDER_HINT}</p>
        <label>
          폴더 이름
          <input value={name} onChange={(e) => setName(e.target.value)} required autoFocus placeholder="예: 공장" />
        </label>
        {cards.length > 0 && (
          <div className="field-label">
            넣을 프로젝트 <span className="muted small">(선택 · 비워 두고 나중에 넣어도 됩니다)</span>
          </div>
        )}
        <div className="checks scroll-list">
          {cards.map((c) => (
            <label key={c.id}>
              <input
                type="checkbox"
                checked={picked.includes(c.id)}
                onChange={(e) => setPicked(e.target.checked ? [...picked, c.id] : picked.filter((x) => x !== c.id))}
              />
              {c.name} <span className="muted small">{c.folder ? `(지금: ${c.folder})` : ""}</span>
            </label>
          ))}
        </div>
        <p className="muted small">폴더는 비어 있어도 남습니다. 지우려면 폴더를 고른 뒤 "폴더 지우기"를 누르세요.</p>
        {error && <div className="error">{error}</div>}
        <div className="modal-actions">
          <button type="button" onClick={() => onClose(null)}>
            취소
          </button>
          <button className="primary" disabled={!name.trim()}>
            만들기
          </button>
        </div>
      </form>
    </Modal>
  );
}

function RenameFolder({ name, onClose }: { name: string; onClose: (renamed: string | null) => void }) {
  const [value, setValue] = useState(name);
  const [error, setError] = useState<string | null>(null);
  const save = async (e: FormEvent) => {
    e.preventDefault();
    try {
      await api("/folders", { method: "PATCH", body: { name, new_name: value.trim() } });
      onClose(value.trim());
    } catch (err) {
      setError(err instanceof Error ? err.message : String(err));
    }
  };
  return (
    <Modal title="폴더 이름 바꾸기" onClose={() => onClose(null)}>
      <form onSubmit={save}>
        <label>
          새 이름
          <input value={value} onChange={(e) => setValue(e.target.value)} required autoFocus />
        </label>
        <p className="muted small">안의 프로젝트는 새 이름의 폴더에 그대로 들어 있습니다.</p>
        {error && <div className="error">{error}</div>}
        <div className="modal-actions">
          <button type="button" onClick={() => onClose(null)}>
            취소
          </button>
          <button className="primary" disabled={!value.trim() || value.trim() === name}>
            바꾸기
          </button>
        </div>
      </form>
    </Modal>
  );
}

// ---- a project: its datasets as cards ------------------------------------------------------

const TASK_HINT_TEXT = (task: string) => (TASK_LABEL[task] ? `${taskLabel(task)}: ${TASK_HINT[task]}` : `${task}: 직접 정한 태스크`);

/** "Det 2 · Kpt 1": how many datasets of each task (each dataset has its own). */
const tasksCounted = (tasks: string[]) => {
  const counts = new Map<string, number>();
  for (const t of tasks) counts.set(t, (counts.get(t) ?? 0) + 1);
  return [...counts].map(([t, n]) => `${taskLabel(t)} ${n}`).join(" · ");
};

/** The tasks of a project's datasets as tags (none before its first dataset). */
export function TaskTags({ tasks }: { tasks: string[] }) {
  return (
    <>
      {tasks.map((t) => (
        <span key={t} className="tag task-tag" title={`이 프로젝트에 있는 데이터셋의 태스크 — ${TASK_HINT_TEXT(t)}`}>
          {taskLabel(t)}
        </span>
      ))}
    </>
  );
}

export function ProjectPage({ projectId, onImport, isAdmin }: { projectId: string; onImport: () => void; isAdmin: boolean }) {
  const { data: page, error, refresh } = usePolling(() => api<ProjectPageInfo>(`/projects/${projectId}/overview`), 15000, [projectId]);
  const [editing, setEditing] = useState(false);
  const [failed, setFailed] = useState<string | null>(null);
  const [moving, setMoving] = useState<DatasetCardInfo | null>(null); // 다른 프로젝트로 옮기기
  const [notice, setNotice] = useState<{ text: string; href: string } | null>(null);
  const picking = usePicking();
  const { data: folderList } = useFolders();
  const importing = useImports().filter(
    (i) => i.project.id === projectId && (i.status === "queued" || i.status === "running"),
  );
  const order = useOrder(`datasets.datasets.${projectId}`, page?.dataset_cards, cardId);
  const [layout, setLayout] = useCardLayout("datasets");
  if (error && !page) return <div className="error">{error}</div>;
  if (!page) return <div className="muted">불러오는 중…</div>;
  const colorOf = (cards: DatasetCardInfo) => (id: number) =>
    cards.top_classes.find((c) => c.id === id)?.color ?? classColor(id);
  const trashPicked = async () => {
    const chosen = page.dataset_cards.filter((d) => picking.ids.includes(d.id));
    if (chosen.length === 0) return;
    const lines = chosen.slice(0, 8).map((d) => `· ${d.name} — 버전 ${d.releases} · 이미지 ${d.items.toLocaleString()}`);
    if (chosen.length > 8) lines.push(`· 그 밖에 ${chosen.length - 8}개`);
    const text = `데이터셋 ${chosen.length}개의 모든 버전을 휴지통으로 옮길까요?\n${lines.join("\n")}\n\n휴지통에서 버전마다 되살릴 수 있고, 이미지 파일은 지워지지 않습니다.`;
    if (!window.confirm(text)) return;
    const failures = await eachOf(chosen, (d) => api(`/datasets/${d.id}`, { method: "DELETE" }), (d) => d.name);
    announceTrash();
    picking.stop();
    setFailed(failures.length ? failures.join("\n") : null);
    void refresh();
  };
  return (
    <>
      <nav className="page-crumbs">
        <a href="#/datasets">데이터셋</a>
        {page.folder && <span>{page.folder}</span>}
        <strong>{page.name}</strong>
      </nav>
      <div className="page-head">
        <div>
          <h1 className="with-badge">
            {page.name}
          </h1>
          <p className="muted note">{page.description || "카드를 누르면 그 데이터셋의 최신 버전을 엽니다. 끌어서 순서를 바꿀 수 있습니다."}</p>
        </div>
        <div className="head-actions">
          <button className="icon-btn" onClick={() => setEditing(true)} title="이름 · 폴더 · 설명 · 휴지통" aria-label={`${page.name} 설정`}>
            <Icon name="settings" />
          </button>
          {isAdmin && !picking.on && page.dataset_cards.length > 0 && (
            <button className="ghost-btn" onClick={picking.start} title="여러 데이터셋을 골라 한 번에 휴지통으로">
              <Icon name="check" size={13} />
              고르기
            </button>
          )}
          <OrderReset order={order} />
          <LayoutSwitch value={layout} onChange={setLayout} />
          <a className="btn" href={`#/training/project/${page.id}`} onClick={() => rememberProject(page.id)}>
            학습
          </a>
          <button className="primary" onClick={onImport}>
            <Icon name="plus" />
            데이터셋 가져오기
          </button>
        </div>
      </div>

      <div className="stats five">
        <Stat
          icon="database"
          label="데이터셋"
          value={page.datasets}
          sub={page.dataset_cards.length > 0 ? tasksCounted(page.dataset_cards.map((d) => d.task)) : "아직 없음"}
        />
        <Stat icon="tag" label="버전" value={page.releases} sub={page.version_at ? `마지막 버전 ${timeAgo(page.version_at)}` : "아직 없음"} />
        <Stat
          icon="images"
          label="이미지"
          value={page.items.toLocaleString()}
          sub={`라벨 ${page.dataset_cards.reduce((n, d) => n + d.boxes, 0).toLocaleString()}개`}
        />
        <Stat icon="chart" label="학습" value={page.runs} sub={page.run_at ? `마지막 학습 ${timeAgo(page.run_at)}` : "아직 없음"} />
        <Stat
          icon="box"
          label="모델"
          value={page.models}
          sub={page.best ? `최고 ${METRIC_NAME[page.best.metric] ?? page.best.metric} ${page.best.value.toFixed(3)}` : "학습이 끝나면 등록됩니다"}
        />
      </div>

      {failed && <div className="error lines">{failed}</div>}
      {notice && (
        <div className="notice ok">
          {notice.text} <a href={notice.href}>열기</a>
        </div>
      )}
      {picking.on && (
        <PickBar
          count={picking.ids.length}
          total={page.dataset_cards.length}
          onAll={() => picking.setIds(page.dataset_cards.map((d) => d.id))}
          onNone={() => picking.setIds([])}
          onCancel={picking.stop}
        >
          <button className="danger" disabled={picking.ids.length === 0} onClick={() => void trashPicked()}>
            <Icon name="trash" size={14} />
            모든 버전을 휴지통으로
          </button>
        </PickBar>
      )}
      <div className="project-layout">
        <Sortable ids={order.ordered.map(cardId)} onMove={order.move} disabled={picking.on}>
          <div className={`card-grid${layoutClass(layout)}`}>
            {importing.map((i) => (
              <a key={i.job_id} className="data-card data-card-link importing" href={i.release_id ? `#/datasets/${i.release_id}` : undefined}>
                <div className="collage importing-face">
                  <span className="spinner" />
                </div>
                <div className="data-card-body">
                  <div className="data-card-title">
                    <strong>{i.dataset}</strong>
                    <span className="tag">{i.version}</span>
                  </div>
                  <div className="muted small">
                    {i.node ? `${i.node}에서 ` : ""}
                    {i.status === "queued" ? "등록 대기" : "등록 중"} · 다른 화면에서 일해도 됩니다
                  </div>
                  <ImportProgress info={i} compact />
                </div>
              </a>
            ))}
            {order.ordered.map((d) => {
              const pick = picking.on ? { selected: picking.ids.includes(d.id), onToggle: () => picking.toggle(d.id) } : null;
              const body = (
                <>
                <Collage shots={d.collage} colorOf={colorOf(d)} more={d.items} />
                <div className="data-card-body">
                  <div className="data-card-title">
                    <span className="with-grip"><SortHandle /><strong>{d.name}</strong></span>
                    <span className="title-tags">
                      <span className="tag task-tag" title={TASK_HINT_TEXT(d.task)}>
                        {taskLabel(d.task)}
                      </span>
                      <span className="tag">{d.release.version}</span>
                      {d.draft && d.draft.edited > 0 && (
                        <span className="tag accent" title="라벨을 고쳤지만 아직 버전으로 확정하지 않은 이미지">
                          작업 중 {d.draft.edited.toLocaleString()}
                        </span>
                      )}
                    </span>
                  </div>
                  <div className="muted small">
                    버전 {d.releases} · 클래스 {d.classes} · 이미지 {d.items.toLocaleString()} · {bytesLabel(d.bytes)} ·{" "}
                    {timeAgo(d.release.created_at)}
                  </div>
                  <div className="card-classes">
                    {d.top_classes.map((c) => (
                      <span key={c.id}>
                        <i style={{ background: c.color ?? classColor(c.id) }} />
                        {c.name}
                      </span>
                    ))}
                    {d.classes > d.top_classes.length && <em>+{d.classes - d.top_classes.length}</em>}
                  </div>
                  <SplitBar splits={d.splits} />
                </div>
                </>
              );
              return (
                <SortItem key={d.id} id={d.id}>
                  {pick ? (
                    <div className={`data-card data-card-link picking${pick.selected ? " selected" : ""}`} {...pickProps(pick, d.name)}>
                      <CardCheck selected={pick.selected} />
                      {body}
                    </div>
                  ) : (
                    <div className="data-card">
                      <a className="data-card-link" href={`#/datasets/${d.release.id}`}>
                        {body}
                      </a>
                      <button
                        className="icon-btn data-card-menu"
                        onClick={() => setMoving(d)}
                        title="다른 프로젝트로 옮기기"
                        aria-label={`${d.name} 다른 프로젝트로 옮기기`}
                      >
                        <Icon name="arrow" size={14} />
                      </button>
                    </div>
                  )}
                </SortItem>
              );
            })}
            {!picking.on && (
            <button className="add-tile" onClick={onImport}>
              <Icon name="plus" size={18} />
              데이터셋 가져오기
              <span className="muted small">노드의 폴더를 복사 없이 등록 · Det · Seg · Kpt · Cls</span>
            </button>
            )}
          </div>
        </Sortable>

        <aside className="card project-models">
          <h3>이 프로젝트의 모델</h3>
          {page.model_groups.length === 0 && (
            <p className="muted small">아직 학습한 모델이 없습니다. 데이터셋을 열어 "이 버전으로 학습"을 누르세요.</p>
          )}
          {page.model_groups.map((g) => (
            <div key={g.task} className="model-group">
              {page.model_groups.length > 1 && (
                <div className="model-group-head">
                  <span className="tag task-tag">{taskLabel(g.task)}</span>
                  <span className="muted small">{METRIC_NAME[g.metric] ?? g.metric} 기준</span>
                </div>
              )}
              <ol>
                {g.models.map((m) => (
                  <li key={m.id}>
                    <a href={`#/models/${m.id}`} onClick={() => rememberProject(page.id)}>
                      {m.name}
                    </a>
                    {m.aliases.map((a) => (
                      <span key={a} className="tag">
                        {a}
                      </span>
                    ))}
                    <span className="num muted">{m.value != null ? m.value.toFixed(3) : "–"}</span>
                  </li>
                ))}
              </ol>
            </div>
          ))}
          <div className="muted small">
            {page.model_groups.length === 1 && `${METRIC_NAME[page.model_groups[0].metric] ?? page.model_groups[0].metric} 기준 · `}
            학습 {page.runs}회 · 탐색 {page.sweeps}회
          </div>
          <a className="link small" href={`#/training/project/${page.id}`} onClick={() => rememberProject(page.id)}>
            학습 화면에서 비교 <Icon name="arrow" size={12} />
          </a>
        </aside>
      </div>
      {moving && (
        <MoveDatasetDialog
          dataset={moving}
          projectId={page.id}
          onClose={() => setMoving(null)}
          onMoved={(moved) => {
            setMoving(null);
            void refresh();
            const href = moved.release_id ? `#/datasets/${moved.release_id}` : `#/datasets/project/${moved.project.id}`;
            setNotice({ text: `${moved.name} 데이터셋을 ${moved.project.name} 프로젝트로 옮겼습니다.`, href });
          }}
        />
      )}
      {editing && (
        <EditProject
          card={page}
          folders={(folderList ?? []).map((f) => f.name)}
          onClose={(deleted) => {
            setEditing(false);
            if (deleted) window.location.hash = "#/datasets";
            else void refresh();
          }}
        />
      )}
    </>
  );
}
