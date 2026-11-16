import { type FormEvent, useMemo, useState } from "react";
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

// ---- the first screen: folders and projects -----------------------------------------------

/**
 * Projects as cards, grouped by folder. A folder is a name projects share: making one moves the
 * chosen projects into it, and it goes away when nothing is left in it.
 */
export function ProjectBoard({ onNewProject }: { onNewProject: (folder: string) => void }) {
  const { data: cards, error, refresh } = usePolling(() => api<ProjectCardInfo[]>("/project-cards"), 30000);
  const { data: trash, refresh: refreshTrash } = usePolling(() => api<TrashInfo>("/trash"), 60000);
  const binned = (trash?.projects.length ?? 0) + (trash?.releases.length ?? 0);
  const [folder, setFolder] = useState<string | null>(null); // null: all
  const [q, setQ] = useState("");
  const [sort, setSort] = useState("updated");
  const [editing, setEditing] = useState<ProjectCardInfo | null>(null);
  const [makingFolder, setMakingFolder] = useState(false);

  const folders = useMemo(() => {
    const counts = new Map<string, number>();
    for (const c of cards ?? []) counts.set(c.folder, (counts.get(c.folder) ?? 0) + 1);
    return [...counts.entries()].sort(([a], [b]) => (a === "" ? 1 : b === "" ? -1 : a.localeCompare(b, "ko")));
  }, [cards]);
  const shown = useMemo(() => {
    const needle = q.trim().toLowerCase();
    const list = (cards ?? []).filter(
      (c) => (folder === null || c.folder === folder) && (!needle || `${c.name} ${c.description} ${c.folder}`.toLowerCase().includes(needle)),
    );
    const by: Record<string, (a: ProjectCardInfo, b: ProjectCardInfo) => number> = {
      updated: (a, b) => b.updated_at.localeCompare(a.updated_at),
      name: (a, b) => a.name.localeCompare(b.name, "ko"),
      items: (a, b) => b.items - a.items,
    };
    return [...list].sort(by[sort]);
  }, [cards, folder, q, sort]);

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
          <button onClick={() => setMakingFolder(true)} disabled={!cards?.length}>
            <Icon name="plus" />새 폴더
          </button>
          <button className="primary" onClick={() => onNewProject(folder ?? "")}>
            <Icon name="plus" />새 프로젝트
          </button>
        </div>
      </div>

      {cards && cards.length > 0 && (
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
          </div>
          <span className="spacer" />
          <input type="search" value={q} onChange={(e) => setQ(e.target.value)} placeholder="프로젝트 찾기" aria-label="프로젝트 찾기" />
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
      {cards && cards.length === 0 && binned > 0 && (
        <p className="muted small">
          지운 프로젝트가 있습니다. <a href="#/datasets/trash">휴지통 {binned}</a>
        </p>
      )}

      {!cards && <p className="muted">프로젝트를 불러오는 중…</p>}
      {cards && cards.length === 0 && (
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

      <div className="card-grid">
        {shown.map((c) => (
          <ProjectCard key={c.id} card={c} onEdit={() => setEditing(c)} />
        ))}
        {cards && cards.length > 0 && (
          <button className="add-tile" onClick={() => onNewProject(folder ?? "")}>
            <Icon name="plus" size={18} />새 프로젝트{folder ? ` (${folder})` : ""}
          </button>
        )}
      </div>

      {editing && (
        <EditProject
          card={editing}
          folders={folders.map(([f]) => f).filter(Boolean)}
          onClose={() => {
            setEditing(null);
            void refresh();
            void refreshTrash();
          }}
        />
      )}
      {makingFolder && cards && (
        <NewFolder
          cards={cards}
          onClose={(made) => {
            setMakingFolder(false);
            if (made) setFolder(made);
            void refresh();
          }}
        />
      )}
    </>
  );
}

export function ProjectCard({ card, onEdit }: { card: ProjectCardInfo; onEdit?: () => void }) {
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
        <label>
          폴더 <span className="muted small">(새 이름을 적으면 새 폴더, 비우면 폴더에서 뺌)</span>
          <input value={folder} onChange={(e) => setFolder(e.target.value)} list="project-folders" placeholder="폴더 없음" />
          <datalist id="project-folders">
            {folders.map((f) => (
              <option key={f} value={f} />
            ))}
          </datalist>
        </label>
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
      for (const id of picked) await api(`/projects/${id}`, { method: "PATCH", body: { folder: name.trim() } });
      onClose(name.trim());
    } catch (err) {
      setError(err instanceof Error ? err.message : String(err));
    }
  };
  return (
    <Modal title="새 폴더" onClose={() => onClose(null)}>
      <form onSubmit={save}>
        <label>
          폴더 이름
          <input value={name} onChange={(e) => setName(e.target.value)} required autoFocus placeholder="예: 공장" />
        </label>
        <div className="field-label">이 폴더로 옮길 프로젝트</div>
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
        <p className="muted small">폴더는 프로젝트를 묶는 이름입니다. 들어 있는 프로젝트가 없으면 목록에서 사라집니다.</p>
        {error && <div className="error">{error}</div>}
        <div className="modal-actions">
          <button type="button" onClick={() => onClose(null)}>
            취소
          </button>
          <button className="primary" disabled={!name.trim() || picked.length === 0}>
            만들기
          </button>
        </div>
      </form>
    </Modal>
  );
}

// ---- the trash: deleted projects and versions, to bring back ----------------------------------

export function TrashPage({ isAdmin }: { isAdmin: boolean }) {
  const { data: trash, error, refresh } = usePolling(() => api<TrashInfo>("/trash"), 30000);
  const [failed, setFailed] = useState<string | null>(null);
  const restore = async (path: string) => {
    try {
      setFailed(null);
      await api(path, { method: "POST" });
      await refresh();
    } catch (e) {
      setFailed(e instanceof Error ? e.message : String(e));
    }
  };
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
      </div>
      {failed && <div className="error">{failed}</div>}
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
                  <td className="right">
                    <button className="small" onClick={() => void restore(`/projects/${p.id}/restore`)}>
                      되살리기
                    </button>
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
                  <td className="right">
                    {isAdmin && (
                      <button className="small" onClick={() => void restore(`/releases/${r.id}/restore`)}>
                        되살리기
                      </button>
                    )}
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
          folders={[]}
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
