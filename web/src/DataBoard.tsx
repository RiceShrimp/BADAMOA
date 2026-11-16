import { type FormEvent, useEffect, useMemo, useRef, useState } from "react";
import { api } from "./api";
import { SPLIT_LABEL, type TrashInfo, bytesLabel, classColor } from "./catalogTypes";
import { Icon, Modal, timeAgo, usePolling } from "./ui";

// ---- what the Hub sends ---------------------------------------------------------------

export interface Shot {
  url: string;
  width: number;
  height: number;
  boxes: number[][]; // [class, cx, cy, w, h] as fractions
}

export interface ProjectCardInfo {
  id: string;
  name: string;
  task: string;
  folder: string;
  description: string;
  created_at: string;
  updated_at: string;
  datasets: number;
  releases: number;
  items: number;
  runs: number;
  models: number;
  best: { id: string; name: string; metric: string; value: number } | null;
  collage: Shot[];
  /** Classes with labels in the newest version of each dataset, most labels first. */
  classes: { name: string; labels: number; images: number }[];
}

interface DatasetCardInfo {
  id: string;
  name: string;
  release: { id: string; version: string; created_at: string; note: string };
  releases: number;
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
  top_models: { id: string; name: string; value: number | null; aliases: string[] }[];
  sweeps: number;
}

export const TASK_LABEL: Record<string, string> = { detect: "탐지", segment: "분할", classify: "분류" };
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

export function Shot({ shot, colorOf, boxes = true }: { shot: Shot; colorOf: (id: number) => string; boxes?: boolean }) {
  const w = shot.width || 1;
  const h = shot.height || 1;
  const line = Math.max(w, h) / 140;
  return (
    <svg className="shot" viewBox={`0 0 ${w} ${h}`} preserveAspectRatio="xMidYMid slice" aria-hidden="true">
      <image href={shot.url} width={w} height={h} preserveAspectRatio="xMidYMid slice" />
      {boxes &&
        shot.boxes.map((b, i) =>
          b.length >= 5 ? (
            <rect
              key={i}
              x={(b[1] - b[3] / 2) * w}
              y={(b[2] - b[4] / 2) * h}
              width={b[3] * w}
              height={b[4] * h}
              fill="none"
              stroke={colorOf(b[0])}
              strokeWidth={line}
            />
          ) : null,
        )}
    </svg>
  );
}

function Collage({ shots, colorOf, more }: { shots: Shot[]; colorOf: (id: number) => string; more?: number }) {
  return (
    <div className="collage">
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

// ---- a project's folder: one there is, none, or a new one typed in --------------------------

const NEW_FOLDER = "\u0000new";

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
  const [sort, setSort] = useState("updated");
  const [editing, setEditing] = useState<ProjectCardInfo | null>(null);
  const [makingFolder, setMakingFolder] = useState(false);
  const [renaming, setRenaming] = useState(false);
  const [failed, setFailed] = useState<string | null>(null);
  const reload = () => {
    void refresh();
    void refreshFolders();
    void refreshTrash();
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
    const list = (cards ?? []).filter(
      (c) =>
        (folder === null || c.folder === folder) &&
        (!task || c.task === task) &&
        hasAll(c, wanted) &&
        (!needle || `${c.name} ${c.description} ${c.folder}`.toLowerCase().includes(needle)),
    );
    const by: Record<string, (a: ProjectCardInfo, b: ProjectCardInfo) => number> = {
      updated: (a, b) => b.updated_at.localeCompare(a.updated_at),
      name: (a, b) => a.name.localeCompare(b.name, "ko"),
      items: (a, b) => b.items - a.items,
    };
    return [...list].sort(by[sort]);
  }, [cards, folder, task, wanted, q, sort]);
  const clearFilters = () => {
    setTask("");
    setWanted([]);
  };
  const filters = (task ? 1 : 0) + wanted.length;
  // The tasks there are, with how many projects each: the filter lists only those.
  const tasks = useMemo(() => {
    const counts = new Map<string, number>();
    for (const c of cards ?? []) counts.set(c.task, (counts.get(c.task) ?? 0) + 1);
    return Object.keys(TASK_LABEL)
      .filter((t) => counts.has(t))
      .map((t): [string, number] => [t, counts.get(t) ?? 0]);
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
            {SORTS.map((s) => (
              <option key={s.id} value={s.id}>
                {s.label}
              </option>
            ))}
          </select>
          {binned > 0 && (
            <a className="btn ghost-btn" href="#/datasets/trash" title="지운 프로젝트와 버전을 되살립니다">
              <Icon name="trash" size={13} />
              휴지통 {binned}
            </a>
          )}
        </div>
      )}
      {filters > 0 && (
        <div className="active-filters">
          {task && (
            <button onClick={() => setTask("")} title="이 조건 빼기">
              {TASK_LABEL[task] ?? task} <Icon name="x" size={11} />
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
      {failed && <div className="error">{failed}</div>}
      {cards && cards.length === 0 && binned > 0 && (
        <p className="muted small">
          지운 프로젝트가 있습니다. <a href="#/datasets/trash">휴지통 {binned}</a>
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
      <div className="card-grid">
        {shown.map((c) => (
          <ProjectCard key={c.id} card={c} onEdit={() => setEditing(c)} highlight={wanted} />
        ))}
        {cards && (cards.length > 0 || names.length > 0) && (
          <button className="add-tile" onClick={() => onNewProject(folder ?? "")}>
            <Icon name="plus" size={18} />새 프로젝트{folder ? ` (${folder})` : ""}
          </button>
        )}
      </div>

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
            {TASK_LABEL[t] ?? t} <span className="muted">{n}</span>
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

export function ProjectCard({ card, onEdit, highlight = [] }: { card: ProjectCardInfo; onEdit?: () => void; highlight?: string[] }) {
  const matched = highlight
    .map((w) => (card.classes ?? []).find((c) => c.name.toLowerCase() === w.toLowerCase()))
    .filter((c): c is ProjectCardInfo["classes"][number] => Boolean(c));
  return (
    <div className="data-card">
      <a href={`#/datasets/project/${card.id}`} onClick={() => rememberProject(card.id)} className="data-card-link">
        <Collage shots={card.collage} colorOf={classColor} />
        <div className="data-card-body">
          <div className="data-card-title">
            <strong>{card.name}</strong>
            <span className="tag">{TASK_LABEL[card.task] ?? card.task}</span>
          </div>
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
      </a>
      {onEdit && (
        <button className="icon-btn data-card-menu" onClick={onEdit} title="이름 · 폴더 · 설명" aria-label={`${card.name} 설정`}>
          <Icon name="settings" size={14} />
        </button>
      )}
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
        <p className="muted small">태스크({TASK_LABEL[card.task] ?? card.task})는 바꿀 수 없습니다. 라벨과 학습이 그 태스크에 맞춰져 있습니다.</p>
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

// ---- the trash: deleted projects and versions, to bring back ----------------------------------

interface Emptied {
  projects: number;
  releases: number;
  files: number;
  bytes: number;
  kept: { id: string; name: string; reason: string }[];
}

export function TrashPage({ isAdmin }: { isAdmin: boolean }) {
  const { data: trash, error, refresh } = usePolling(() => api<TrashInfo>("/trash"), 30000);
  const [failed, setFailed] = useState<string | null>(null);
  const [emptied, setEmptied] = useState<Emptied | null>(null);
  const [busy, setBusy] = useState(false);
  const restore = async (path: string) => {
    try {
      setFailed(null);
      setEmptied(null);
      await api(path, { method: "POST" });
      await refresh();
    } catch (e) {
      setFailed(e instanceof Error ? e.message : String(e));
    }
  };
  // Removing for good: the records and the Hub's files only they used. The images on the nodes stay.
  const purge = async (what: string, body: { projects?: string[]; releases?: string[] }) => {
    const text = `${what}

데이터셋 · 버전 · 라벨 기록, 학습 기록 · 모델과 그 파일이 Hub에서 지워지고 되돌릴 수 없습니다. 노드에 등록한 이미지 파일은 그대로 남습니다.`;
    if (!window.confirm(text)) return;
    setBusy(true);
    try {
      setFailed(null);
      setEmptied(await api<Emptied>("/trash/empty", { body }));
      await refresh();
    } catch (e) {
      setFailed(e instanceof Error ? e.message : String(e));
    } finally {
      setBusy(false);
    }
  };
  const count = (trash?.projects.length ?? 0) + (trash?.releases.filter((r) => r.in_use.length === 0).length ?? 0);
  if (error && !trash) return <div className="error">{error}</div>;
  return (
    <>
      <nav className="page-crumbs">
        <a href="#/datasets">데이터셋</a>
        <strong>휴지통</strong>
      </nav>
      <div className="page-head">
        <div>
          <h1>휴지통</h1>
          <p className="muted">
            지운 프로젝트와 버전입니다. 되살리면 데이터셋 · 학습 · 모델이 그대로 돌아옵니다. 이미지 파일은 처음부터 지우지 않습니다.
          </p>
        </div>
        {isAdmin && count > 0 && (
          <div className="head-actions">
            <button className="danger" disabled={busy} onClick={() => void purge(`휴지통의 ${count}개를 완전히 지울까요?`, {})}>
              <Icon name="trash" />
              휴지통 비우기
            </button>
          </div>
        )}
      </div>
      {failed && <div className="error">{failed}</div>}
      {emptied && (
        <div className="notice">
          {emptied.projects + emptied.releases > 0
            ? `프로젝트 ${emptied.projects}개 · 버전 ${emptied.releases}개를 완전히 지웠습니다. ` +
              (emptied.files > 0
                ? `Hub 파일 ${emptied.files.toLocaleString()}개 (${bytesLabel(emptied.bytes)})를 비웠습니다.`
                : "함께 비울 Hub 파일은 없었습니다 (같은 이미지를 다른 프로젝트가 쓰는 경우 등).")
            : "지운 것이 없습니다."}
          {emptied.kept.map((k) => (
            <div key={k.id} className="small">
              남김: {k.name} — {k.reason}
            </div>
          ))}
        </div>
      )}
      {!trash && <p className="muted">불러오는 중…</p>}
      {trash && trash.projects.length + trash.releases.length === 0 && <div className="card empty">휴지통이 비어 있습니다.</div>}
      {trash && trash.projects.length > 0 && (
        <section className="card home-block">
          <div className="card-head">
            <h3>프로젝트</h3>
            <span className="muted small">만든 사람이나 관리자가 되살릴 수 있습니다</span>
          </div>
          <table className="plain">
            <thead>
              <tr>
                <th>이름</th>
                <th>태스크</th>
                <th className="right">버전</th>
                <th className="right">이미지</th>
                <th>지운 때</th>
                <th />
              </tr>
            </thead>
            <tbody>
              {trash.projects.map((p) => (
                <tr key={p.id}>
                  <td>
                    <strong>{p.name}</strong>
                    {p.folder && <span className="muted small"> · {p.folder}</span>}
                  </td>
                  <td>{TASK_LABEL[p.task] ?? p.task}</td>
                  <td className="num right">{p.releases}</td>
                  <td className="num right">{p.items.toLocaleString()}</td>
                  <td>{timeAgo(p.deleted_at)}</td>
                  <td className="right row-buttons">
                    <button className="small" onClick={() => void restore(`/projects/${p.id}/restore`)}>
                      되살리기
                    </button>
                    {isAdmin && (
                      <button
                        className="small danger"
                        disabled={busy}
                        onClick={() => void purge(`${p.name} 프로젝트를 완전히 지울까요?`, { projects: [p.id] })}
                      >
                        완전히 지우기
                      </button>
                    )}
                  </td>
                </tr>
              ))}
            </tbody>
          </table>
        </section>
      )}
      {trash && trash.releases.length > 0 && (
        <section className="card home-block">
          <div className="card-head">
            <h3>버전</h3>
            {!isAdmin && <span className="muted small">버전은 관리자가 되살릴 수 있습니다</span>}
          </div>
          <table className="plain">
            <thead>
              <tr>
                <th>버전</th>
                <th>프로젝트</th>
                <th className="right">이미지</th>
                <th>지운 때</th>
                <th />
              </tr>
            </thead>
            <tbody>
              {trash.releases.map((r) => (
                <tr key={r.id}>
                  <td>
                    <strong>
                      {r.dataset} {r.version}
                    </strong>
                  </td>
                  <td>
                    <a href={`#/datasets/project/${r.project.id}`}>{r.project.name}</a>
                  </td>
                  <td className="num right">{r.items.toLocaleString()}</td>
                  <td>{timeAgo(r.deleted_at)}</td>
                  <td className="right row-buttons">
                    {isAdmin && (
                      <button className="small" onClick={() => void restore(`/releases/${r.id}/restore`)}>
                        되살리기
                      </button>
                    )}
                    {isAdmin && (
                      <button
                        className="small danger"
                        disabled={busy || r.in_use.length > 0}
                        title={r.in_use.length > 0 ? `${r.in_use.join(", ")}가 이 버전을 씁니다. 계보가 끊기지 않게 남겨 둡니다.` : undefined}
                        onClick={() => void purge(`${r.dataset} ${r.version} 버전을 완전히 지울까요?`, { releases: [r.id] })}
                      >
                        완전히 지우기
                      </button>
                    )}
                    {r.in_use.length > 0 && <div className="muted small">{r.in_use.join(" · ")}가 씀</div>}
                  </td>
                </tr>
              ))}
            </tbody>
          </table>
        </section>
      )}
    </>
  );
}

// ---- a project: its datasets as cards ------------------------------------------------------

export function ProjectPage({ projectId, onImport }: { projectId: string; onImport: () => void }) {
  const { data: page, error, refresh } = usePolling(() => api<ProjectPageInfo>(`/projects/${projectId}/overview`), 15000, [projectId]);
  const [editing, setEditing] = useState(false);
  const { data: folderList } = useFolders();
  if (error && !page) return <div className="error">{error}</div>;
  if (!page) return <div className="muted">불러오는 중…</div>;
  const colorOf = (cards: DatasetCardInfo) => (id: number) =>
    cards.top_classes.find((c) => c.id === id)?.color ?? classColor(id);
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
            {page.name} <span className="tag">{TASK_LABEL[page.task] ?? page.task}</span>
          </h1>
          <p className="muted">
            데이터셋 {page.datasets} · 버전 {page.releases} · 이미지 {page.items.toLocaleString()} · 학습 {page.runs} · 모델{" "}
            {page.models} · {timeAgo(page.updated_at)} 수정
          </p>
          {page.description && <p className="note">{page.description}</p>}
        </div>
        <div className="head-actions">
          <button className="icon-btn" onClick={() => setEditing(true)} title="이름 · 폴더 · 설명 · 휴지통" aria-label={`${page.name} 설정`}>
            <Icon name="settings" />
          </button>
          <a className="btn" href="#/training" onClick={() => rememberProject(page.id)}>
            학습
          </a>
          <button className="primary" onClick={onImport}>
            <Icon name="plus" />
            데이터셋 가져오기
          </button>
        </div>
      </div>

      <div className="project-layout">
        <div className="card-grid">
          {page.dataset_cards.map((d) => (
            <a key={d.id} className="data-card data-card-link" href={`#/datasets/${d.release.id}`}>
              <Collage shots={d.collage} colorOf={colorOf(d)} more={d.items} />
              <div className="data-card-body">
                <div className="data-card-title">
                  <strong>{d.name}</strong>
                  <span className="tag">{d.release.version}</span>
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
                {(d.states.unlabelled ?? 0) > 0 && <div className="muted small">미라벨 {d.states.unlabelled.toLocaleString()}장</div>}
              </div>
            </a>
          ))}
          <button className="add-tile" onClick={onImport}>
            <Icon name="plus" size={18} />
            데이터셋 가져오기
            <span className="muted small">노드에 있는 YOLO · COCO · 클래스별 폴더를 복사 없이 등록</span>
          </button>
        </div>

        <aside className="card project-models">
          <h3>이 프로젝트의 모델</h3>
          {page.top_models.length === 0 ? (
            <p className="muted small">아직 학습한 모델이 없습니다. 데이터셋을 열어 "이 버전으로 학습"을 누르세요.</p>
          ) : (
            <ol>
              {page.top_models.map((m) => (
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
          )}
          <div className="muted small">
            {METRIC_NAME[page.metric] ?? page.metric} 기준 · 학습 {page.runs}회 · 탐색 {page.sweeps}회
          </div>
          <a className="link small" href="#/training" onClick={() => rememberProject(page.id)}>
            학습 화면에서 비교 <Icon name="arrow" size={12} />
          </a>
        </aside>
      </div>
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
