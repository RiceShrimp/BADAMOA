import { type FormEvent, useEffect, useMemo, useState } from "react";
import {
  FOLDER_HINT,
  FolderPicker,
  ProjectBoard,
  ProjectPage,
  SPLIT_COLOR,
  Shot,
  TrashPage,
  rememberProject,
  useFolders,
} from "./DataBoard";
import { FolderBrowser, LAYOUT_NAME } from "./FolderBrowser";
import { ImportProgress, announceImport, useImports } from "./Imports";
import { ItemViewer } from "./ItemViewer";
import { ExportDialog, type ItemFilter, NamesDialog, NewReleaseDialog } from "./ReleaseMaker";
import { ChartsTab, ClassesTab, ModelsTab, ReleasesTab, colorFor } from "./ReleaseTabs";
import { type Job, type NodeInfo, api } from "./api";
import {
  type CompareResult,
  type ItemPage,
  type Project,
  type ReleaseInfo,
  SPLIT_LABEL,
  STATE_HELP,
  STATE_HINT,
  STATE_LABEL,
  bytesLabel,
} from "./catalogTypes";
import { Badge, Icon, Modal, Stat, timeAgo, usePolling } from "./ui";

const PAGE_SIZES = [30, 60, 100, 200]; // the Hub sends at most 200 at a time
const PAGE_SIZE_KEY = "badamoa.pageSize";

function rememberedPageSize(): number {
  try {
    const n = Number(localStorage.getItem(PAGE_SIZE_KEY));
    return PAGE_SIZES.includes(n) ? n : 60;
  } catch {
    return 60;
  }
}

/** The page number typed in: Enter or leaving the box goes there (kept within 1 … pages). */
function PageJump({ page, pages, onGo }: { page: number; pages: number; onGo: (page: number) => void }) {
  const [typed, setTyped] = useState(String(page + 1));
  useEffect(() => setTyped(String(page + 1)), [page]);
  const go = () => {
    const n = Math.round(Number(typed));
    if (!Number.isFinite(n) || typed.trim() === "") return setTyped(String(page + 1));
    const target = Math.min(pages, Math.max(1, n)) - 1;
    setTyped(String(target + 1));
    if (target !== page) onGo(target);
  };
  return (
    <span className="page-jump">
      <input
        type="number"
        min={1}
        max={pages}
        value={typed}
        onChange={(e) => setTyped(e.target.value)}
        onKeyDown={(e) => e.key === "Enter" && go()}
        onBlur={go}
        aria-label="페이지 번호"
        style={{ width: `${Math.max(5, String(pages).length + 3)}ch` }}
      />
      <span className="num">/ {pages.toLocaleString()}</span>
    </span>
  );
}

/**
 * The data screens: projects as cards (``#/datasets``), a project's datasets as cards
 * (``#/datasets/project/<id>``), and a release with its tabs (``#/datasets/<release>/<tab>``).
 */
export function Datasets({ sub, arg, isAdmin }: { sub?: string; arg?: string; isAdmin: boolean }) {
  const { data: projects } = usePolling(() => api<Project[]>("/projects"), 30000);
  const { data: folderList, refresh: refreshFolders } = useFolders();
  const [creating, setCreating] = useState<string | null>(null); // the folder a new project goes into
  const newProject = (folder: string) => {
    void refreshFolders(); // a folder made a moment ago must be in the list
    setCreating(folder);
  };
  const [importing, setImporting] = useState<string | null>(null); // the project a dataset is imported into

  let body;
  if (!sub) body = <ProjectBoard onNewProject={newProject} />;
  else if (sub === "trash") body = <TrashPage isAdmin={isAdmin} />;
  else if (sub === "project" && arg) body = <ProjectPage projectId={arg} onImport={() => setImporting(arg)} />;
  else {
    const [tab, query = ""] = (arg ?? "images").split("?");
    body = <ReleaseView key={sub} releaseId={sub} tab={tab || "images"} query={query} isAdmin={isAdmin} projects={projects ?? []} />;
  }
  return (
    <>
      {body}
      {creating !== null && (
        <NewProject
          folder={creating}
          folders={(folderList ?? []).map((f) => f.name)}
          onClose={() => setCreating(null)}
          onCreated={(id) => {
            setCreating(null);
            rememberProject(id);
            window.location.hash = `#/datasets/project/${id}`;
          }}
        />
      )}
      {importing && <ImportDialog projectId={importing} onClose={() => setImporting(null)} />}
    </>
  );
}

const TABS = [
  { id: "images", label: "이미지" },
  { id: "classes", label: "클래스" },
  { id: "charts", label: "통계" },
  { id: "models", label: "모델" },
  { id: "releases", label: "버전" },
];
const VIEWS = [
  { id: "large", label: "크게" },
  { id: "medium", label: "보통" },
  { id: "small", label: "촘촘히" },
  { id: "list", label: "목록" },
];
const VIEW_KEY = "badamoa.gridView";

function rememberedView(): string {
  try {
    return localStorage.getItem(VIEW_KEY) ?? "medium";
  } catch {
    return "medium";
  }
}

function ReleaseView({
  releaseId,
  tab,
  query,
  isAdmin,
  projects,
}: {
  releaseId: string;
  tab: string;
  query: string;
  isAdmin: boolean;
  projects: Project[];
}) {
  const { data: release, error, refresh } = usePolling(() => api<ReleaseInfo>(`/releases/${releaseId}`), 10000, [releaseId]);
  const [split, setSplit] = useState("all");
  const [state, setState] = useState("");
  const [classId, setClassId] = useState(() => new URLSearchParams(query).get("class") ?? "");
  // A class name clicked on the classes tab opens the images tab of the same version with that class.
  useEffect(() => {
    const wanted = new URLSearchParams(query).get("class");
    if (wanted !== null) setClassId(wanted);
  }, [query]);
  const [view, setViewState] = useState(rememberedView);
  const [showBoxes, setShowBoxes] = useState(true);
  const setView = (v: string) => {
    setViewState(v);
    try {
      localStorage.setItem(VIEW_KEY, v);
    } catch {
      /* not remembered */
    }
  };
  const [site, setSite] = useState("");
  const [camera, setCamera] = useState("");
  const [q, setQ] = useState("");
  const [page, setPage] = useState(0);
  const [pageSize, setPageSizeState] = useState(rememberedPageSize);
  const setPageSize = (n: number) => {
    setPage(Math.floor((page * pageSize) / n)); // stay near the images being looked at
    setPageSizeState(n);
    try {
      localStorage.setItem(PAGE_SIZE_KEY, String(n));
    } catch {
      /* not remembered */
    }
  };
  const [viewing, setViewing] = useState<number | null>(null);
  const [compareWith, setCompareWith] = useState("");
  const [making, setMaking] = useState<"" | "items" | "classes">("");
  const [naming, setNaming] = useState(false);
  const [exporting, setExporting] = useState(false);

  const filterQuery = useMemo(() => {
    const params = new URLSearchParams();
    if (split !== "all") params.set("split", split === "none" ? "" : split);
    if (state) params.set("state", state);
    if (classId) params.set("class_id", classId);
    if (site) params.set("site", site);
    if (camera) params.set("camera", camera);
    if (q.trim()) params.set("q", q.trim());
    return params.toString();
  }, [split, state, classId, site, camera, q]);
  useEffect(() => setPage(0), [filterQuery]);
  // The same filters as the Hub takes them for a new release.
  const filter = useMemo(() => {
    const out: ItemFilter = {};
    if (split !== "all") out.split = split === "none" ? "" : split;
    if (state) out.state = state;
    if (classId) out.class_id = classId;
    if (site) out.site = site;
    if (camera) out.camera = camera;
    if (q.trim()) out.q = q.trim();
    return out;
  }, [split, state, classId, site, camera, q]);

  const { data: items } = usePolling(
    () => api<ItemPage>(`/releases/${releaseId}/items?${filterQuery}&offset=${page * pageSize}&limit=${pageSize}`),
    release?.status === "building" ? 5000 : 60000,
    [releaseId, filterQuery, page, pageSize, release?.status],
  );

  if (error && !release) return <div className="error">{error}</div>;
  if (!release) return <div className="muted">불러오는 중…</div>;

  const project = projects.find((p) => p.id === release.project_id);
  const task = project?.task ?? "detect";
  const unit = task === "classify" ? "라벨" : "박스";
  const stats = release.stats;
  const classes = release.label_space.classes;
  const colorOf = colorFor(classes);
  const splits = Object.entries(stats.splits ?? {}).sort();
  const pages = Math.max(1, Math.ceil((items?.total ?? 0) / pageSize));
  const present = Object.keys(stats.classes ?? {}).length;

  const remove = async () => {
    if (!window.confirm(`${release.dataset.name} ${release.version} 버전을 휴지통으로 옮길까요? 휴지통에서 되살릴 수 있고, 이미지 파일은 지워지지 않습니다.`)) return;
    await api(`/releases/${releaseId}`, { method: "DELETE" });
    window.location.hash = `#/datasets/project/${release.project_id}`;
  };

  return (
    <>
      <nav className="page-crumbs">
        <a href="#/datasets">데이터셋</a>
        {project?.folder && <span>{project.folder}</span>}
        <a href={`#/datasets/project/${release.project_id}`}>{project?.name ?? "프로젝트"}</a>
        <strong>
          {release.dataset.name} {release.version}
        </strong>
      </nav>
      <div className="page-head">
        <div>
          <h1 className="with-badge">
            <span className="muted-strong">{release.dataset.name}</span> {release.version}
            {release.status === "building" && <Badge kind="running">등록 중</Badge>}
          </h1>
          <p className="muted">
            라벨 공간 {release.label_space.name} · 클래스 {classes.length}개 · {release.created_at.slice(0, 10)} 등록
            {release.base && (
              <>
                {" "}
                · base <a href={`#/datasets/${release.base.id}`}>{release.base.version}</a>
              </>
            )}
          </p>
        </div>
        <div className="head-actions">
          {release.status === "ready" && (
            <button onClick={() => setMaking("items")} title="이 버전의 항목을 골라 새 버전을 만들거나 split을 다시 나눕니다">
              새 버전
            </button>
          )}
          {release.status === "ready" && (
            <button onClick={() => setExporting(true)} title="YOLO · COCO 형식의 zip으로 받습니다">
              내려받기
            </button>
          )}
          {release.status === "ready" && (
            <a className="btn primary-link" href={`#/training/new/${release.id}`}>
              이 버전으로 학습
            </a>
          )}
          {isAdmin && (
            <button className="danger" onClick={remove}>
              삭제
            </button>
          )}
        </div>
      </div>

      <div className="stats five">
        <Stat icon="images" label="전체 항목" value={(stats.items ?? 0).toLocaleString()} sub={bytesLabel(stats.bytes)} />
        {splits.map(([name, s]) => (
          <Stat key={name} icon="database" label={SPLIT_LABEL[name] ?? name} value={s.items.toLocaleString()} sub={bytesLabel(s.bytes)} />
        ))}
        <Stat icon="review" label="배경" value={(stats.states?.background ?? 0).toLocaleString()} sub="확인됨 · 박스 없음" />
        <Stat icon="alert" label="미라벨" value={(stats.states?.unlabelled ?? 0).toLocaleString()} sub="아직 라벨하지 않음" />
      </div>

      {release.status === "building" && <BuildingBar releaseId={release.id} />}
      {release.note && (
        <details className="card fold">
          <summary>메모</summary>
          <p className="note">{release.note}</p>
        </details>
      )}

      <nav className="tabs" aria-label="버전 보기">
        {TABS.map((t) => (
          <a key={t.id} href={`#/datasets/${release.id}/${t.id}`} className={tab === t.id ? "active" : ""}>
            {t.label}
            {t.id === "images" && <em>{(stats.items ?? 0).toLocaleString()}</em>}
            {t.id === "classes" && <em>{present}</em>}
            {t.id === "releases" && release.siblings.length > 0 && <em>{release.siblings.length + 1}</em>}
          </a>
        ))}
      </nav>

      {tab === "classes" && (
        <ClassesTab release={release} onChanged={() => void refresh()} onEditClasses={release.status === "ready" ? () => setMaking("classes") : undefined} />
      )}
      {tab === "charts" && <ChartsTab release={release} />}
      {tab === "models" && <ModelsTab release={release} />}
      {tab === "releases" && <ReleasesTab release={release} onCompare={setCompareWith} />}
      {tab === "releases" && compareWith && (
        <CompareCard releaseId={releaseId} otherId={compareWith} onClose={() => setCompareWith("")} />
      )}

      {tab === "images" && (
        <>
      <div className="filters">
        <select value={split} onChange={(e) => setSplit(e.target.value)} aria-label="split">
          <option value="all">모든 split</option>
          {splits.map(([name]) => (
            <option key={name} value={name === "" ? "none" : name}>
              {SPLIT_LABEL[name] ?? name}
            </option>
          ))}
        </select>
        <select value={state} onChange={(e) => setState(e.target.value)} aria-label="라벨 상태">
          <option value="">모든 라벨 상태</option>
          {Object.entries(STATE_LABEL).map(([id, label]) => (
            <option key={id} value={id} title={STATE_HELP[id]}>
              {STATE_HINT[id] ? `${label} · ${STATE_HINT[id]}` : label}
            </option>
          ))}
        </select>
        <select value={classId} onChange={(e) => setClassId(e.target.value)} aria-label="클래스">
          <option value="">모든 클래스</option>
          {classes.map((c) => (
            <option key={c.id} value={c.id}>
              {c.name}
            </option>
          ))}
        </select>
        {(stats.sites?.length ?? 0) > 0 && (
          <select value={site} onChange={(e) => setSite(e.target.value)} aria-label="사이트">
            <option value="">모든 사이트</option>
            {stats.sites?.map((s) => <option key={s}>{s}</option>)}
          </select>
        )}
        {(stats.cameras?.length ?? 0) > 0 && (
          <select value={camera} onChange={(e) => setCamera(e.target.value)} aria-label="카메라">
            <option value="">모든 카메라</option>
            {stats.cameras?.map((c) => <option key={c}>{c}</option>)}
          </select>
        )}
        <input value={q} onChange={(e) => setQ(e.target.value)} placeholder="파일 이름에 포함…" aria-label="파일 이름" />
        <span className="muted count">{(items?.total ?? 0).toLocaleString()}개</span>
        <span className="spacer" />
        {release.status === "ready" && (
          <button className="ghost-btn" onClick={() => setNaming(true)} title="파일 이름에서 사이트 · 카메라를 읽어 채웁니다">
            사이트 · 카메라
          </button>
        )}
        {task !== "classify" && (
          <label className="check inline-check">
            <input type="checkbox" checked={showBoxes} onChange={(e) => setShowBoxes(e.target.checked)} />
            박스
          </label>
        )}
        <div className="segmented" role="group" aria-label="보기">
          {VIEWS.map((v) => (
            <button key={v.id} className={view === v.id ? "active" : ""} onClick={() => setView(v.id)} aria-pressed={view === v.id}>
              {v.label}
            </button>
          ))}
        </div>
      </div>

      <div className="pager">
        <button onClick={() => setPage(0)} disabled={page === 0} aria-label="첫 페이지">
          «
        </button>
        <button onClick={() => setPage(page - 1)} disabled={page === 0} aria-label="이전 페이지">
          ‹
        </button>
        <PageJump page={page} pages={pages} onGo={setPage} />
        <button onClick={() => setPage(page + 1)} disabled={page >= pages - 1} aria-label="다음 페이지">
          ›
        </button>
        <button onClick={() => setPage(pages - 1)} disabled={page >= pages - 1} aria-label="마지막 페이지">
          »
        </button>
        <select value={pageSize} onChange={(e) => setPageSize(Number(e.target.value))} aria-label="한 페이지에 보일 수">
          {PAGE_SIZES.map((n) => (
            <option key={n} value={n}>
              {n}개씩
            </option>
          ))}
        </select>
      </div>

      {items && items.items.length === 0 && <div className="card empty">조건에 맞는 항목이 없습니다.</div>}
      {view === "list" ? (
        <div className="card table-wrap">
          <table className="plain item-list">
            <thead>
              <tr>
                <th aria-label="미리보기" />
                <th>파일</th>
                <th>split</th>
                <th>라벨</th>
                <th className="right">크기</th>
              </tr>
            </thead>
            <tbody>
              {items?.items.map((item, i) => (
                <tr key={item.id} onClick={() => setViewing(page * pageSize + i)} className="clickable">
                  <td className="item-thumb">
                    {item.thumb_url && (
                      <Shot shot={{ url: item.thumb_url, width: item.width, height: item.height, boxes: item.boxes }} colorOf={colorOf} boxes={showBoxes} />
                    )}
                  </td>
                  <td>{item.file_name}</td>
                  <td>
                    <span className="split-dot" style={{ background: SPLIT_COLOR[item.split] }} />
                    {SPLIT_LABEL[item.split] ?? item.split}
                  </td>
                  <td>{item.state === "labelled" ? `${unit} ${item.box_count}` : STATE_LABEL[item.state]}</td>
                  <td className="num right muted">
                    {item.width}×{item.height}
                  </td>
                </tr>
              ))}
            </tbody>
          </table>
        </div>
      ) : (
      <div className={`thumb-grid ${view}`}>
        {items?.items.map((item, i) => (
          <button key={item.id} className="thumb" onClick={() => setViewing(page * pageSize + i)}>
            <span className="thumb-image">
              {item.thumb_url ? (
                <Shot shot={{ url: item.thumb_url, width: item.width, height: item.height, boxes: item.boxes }} colorOf={colorOf} boxes={showBoxes} />
              ) : (
                <Icon name="images" size={22} />
              )}
              <span className="split-dot" style={{ background: SPLIT_COLOR[item.split] }} title={SPLIT_LABEL[item.split] ?? item.split} />
            </span>
            <span className="thumb-name" title={item.file_name}>
              {item.file_name}
            </span>
            <span className="thumb-meta">
              <span className="tag">{SPLIT_LABEL[item.split] ?? item.split}</span>
              {item.state === "labelled" ? (
                unit === "박스" && <span className="boxes">박스 {item.box_count}</span>
              ) : (
                <span className="tag">{STATE_LABEL[item.state]}</span>
              )}
              <span className="muted">
                {item.width}×{item.height}
              </span>
            </span>
          </button>
        ))}
      </div>
      )}
        </>
      )}

      {exporting && <ExportDialog release={release} task={task} onClose={() => setExporting(false)} />}
      {making && (
        <NewReleaseDialog
          release={release}
          task={task}
          filter={filter}
          filtered={items?.total ?? 0}
          editClasses={making === "classes"}
          onClose={() => setMaking("")}
          onCreated={(id) => {
            setMaking("");
            window.location.hash = `#/datasets/${id}`;
          }}
        />
      )}
      {naming && (
        <NamesDialog
          release={release}
          onClose={() => setNaming(false)}
          onDone={() => {
            setNaming(false);
            void refresh();
          }}
        />
      )}
      {viewing != null && items && (
        <ItemViewer releaseId={releaseId} filterQuery={filterQuery} start={viewing} total={items.total} colorOf={colorOf} classes={classes} onClose={() => setViewing(null)} />
      )}
    </>
  );
}

function BuildingBar({ releaseId }: { releaseId: string }) {
  const info = useImports().find((i) => i.release_id === releaseId);
  if (!info) return null;
  return (
    <div className="card building-bar">
      <strong className="small">등록 중</strong>
      <ImportProgress info={info} />
      <span className="muted small">들어온 이미지부터 아래에 보입니다. 다른 화면으로 가도 등록은 계속됩니다.</span>
    </div>
  );
}

function CompareCard({ releaseId, otherId, onClose }: { releaseId: string; otherId: string; onClose: () => void }) {
  const [diff, setDiff] = useState<CompareResult | null>(null);
  const [error, setError] = useState<string | null>(null);
  useEffect(() => {
    setDiff(null);
    api<CompareResult>(`/releases/${releaseId}/compare?other=${otherId}`)
      .then(setDiff)
      .catch((e) => setError(e instanceof Error ? e.message : String(e)));
  }, [releaseId, otherId]);

  const signed = (n: number) => (n > 0 ? `+${n.toLocaleString()}` : n.toLocaleString());
  const changed = diff?.classes.filter((c) => c.delta !== 0) ?? [];

  return (
    <section className="card">
      <div className="card-head">
        <h3>
          {diff ? `${diff.other.version} → ${diff.release.version}` : "비교 중…"}
        </h3>
        <button className="icon-btn" onClick={onClose} aria-label="비교 닫기">
          <Icon name="x" />
        </button>
      </div>
      {error && <div className="error">{error}</div>}
      {diff && (
        <>
          <div className="stats">
            <Stat icon="plus" label="추가된 항목" value={diff.added.toLocaleString()} sub={`전체 ${signed(diff.release.items - diff.other.items)}`} />
            <Stat icon="x" label="제거된 항목" value={diff.removed.toLocaleString()} />
            <Stat icon="review" label="라벨이 바뀐 항목" value={diff.relabelled.toLocaleString()} sub="가리키는 개정이 다름" />
            <Stat icon="flow" label="split이 바뀐 항목" value={diff.moved_split.toLocaleString()} />
          </div>
          {diff.label_space_changed && <div className="notice">두 버전의 라벨 공간이 다릅니다. 클래스는 이름으로 맞춰 비교했습니다.</div>}
          <div className="table-wrap">
            <table>
              <thead>
                <tr>
                  <th>클래스</th>
                  <th className="right">{diff.other.version}</th>
                  <th className="right">{diff.release.version}</th>
                  <th className="right">증감</th>
                </tr>
              </thead>
              <tbody>
                {changed.length === 0 && (
                  <tr>
                    <td colSpan={4} className="empty">
                      클래스별 박스 수는 같습니다.
                    </td>
                  </tr>
                )}
                {changed.map((c) => (
                  <tr key={c.name}>
                    <td>{c.name}</td>
                    <td className="right num">{c.other.toLocaleString()}</td>
                    <td className="right num">{c.boxes.toLocaleString()}</td>
                    <td className="right num">
                      <strong>{signed(c.delta)}</strong>
                    </td>
                  </tr>
                ))}
              </tbody>
            </table>
          </div>
        </>
      )}
    </section>
  );
}

function NewProject({
  folder: initialFolder = "",
  folders,
  onClose,
  onCreated,
}: {
  folder?: string;
  folders: string[];
  onClose: () => void;
  onCreated: (id: string) => void;
}) {
  const [name, setName] = useState("");
  const [task, setTask] = useState("detect");
  const [folder, setFolder] = useState(initialFolder);
  const [error, setError] = useState<string | null>(null);

  const submit = async (e: FormEvent) => {
    e.preventDefault();
    try {
      const project = await api<Project>("/projects", { body: { name, task, folder } });
      onCreated(project.id);
    } catch (err) {
      setError(err instanceof Error ? err.message : String(err));
    }
  };

  return (
    <Modal title="새 프로젝트" onClose={onClose}>
      <form onSubmit={submit}>
        <label>
          이름
          <input value={name} onChange={(e) => setName(e.target.value)} required autoFocus placeholder="예: 한강 감시" />
        </label>
        <label>
          태스크
          <select value={task} onChange={(e) => setTask(e.target.value)}>
            <option value="detect">객체 탐지</option>
            <option value="segment">분할</option>
            <option value="classify">분류</option>
          </select>
        </label>
        <div className="field-label">
          폴더 <span className="muted small">(선택)</span>
          <FolderPicker value={folder} folders={folders} onChange={setFolder} />
          <span className="muted small field-hint">{FOLDER_HINT}</span>
        </div>
        {error && <div className="error">{error}</div>}
        <div className="modal-actions">
          <button type="button" onClick={onClose}>
            취소
          </button>
          <button className="primary">만들기</button>
        </div>
      </form>
    </Modal>
  );
}

function ImportDialog({ projectId, onClose }: { projectId: string; onClose: () => void }) {
  const [nodes, setNodes] = useState<NodeInfo[]>([]);
  const [form, setForm] = useState({
    node_id: "",
    path: "",
    dataset: "",
    version: "v1",
    format: "auto",
    site: "",
    camera: "",
    name_pattern: "",
    note: "",
  });
  const [job, setJob] = useState<Job | null>(null);
  const [lastLog, setLastLog] = useState("");
  const [error, setError] = useState<string | null>(null);

  useEffect(() => {
    void api<NodeInfo[]>("/nodes").then((list) => {
      const usable = list.filter((n) => n.enabled);
      setNodes(usable);
      setForm((f) => ({ ...f, node_id: usable.find((n) => n.status === "online")?.id ?? usable[0]?.id ?? "" }));
    });
  }, []);

  // Follow the import job until it ends.
  useEffect(() => {
    if (!job || ["succeeded", "failed", "canceled"].includes(job.status)) return;
    const id = setInterval(async () => {
      const fresh = await api<Job>(`/jobs/${job.id}`);
      setJob(fresh);
      const logs = await api<{ text: string }[]>(`/jobs/${job.id}/logs?after=${Math.max(0, fresh.log_count - 1)}`);
      if (logs.length) setLastLog(logs[logs.length - 1].text);
    }, 1000);
    return () => clearInterval(id);
  }, [job]);

  const set = (key: keyof typeof form) => (e: { target: { value: string } }) => setForm({ ...form, [key]: e.target.value });
  const [browsing, setBrowsing] = useState(false);
  const [layout, setLayout] = useState(""); // what the picked folder looks like
  const node = nodes.find((n) => n.id === form.node_id);

  const submit = async (e: FormEvent) => {
    e.preventDefault();
    try {
      setError(null);
      setJob(await api<Job>(`/projects/${projectId}/imports`, { body: form }));
      announceImport();
    } catch (err) {
      setError(err instanceof Error ? err.message : String(err));
    }
  };

  const releaseId = job?.status === "succeeded" ? (job.result as { release_id?: string } | null)?.release_id : undefined;
  const progress = useImports().find((i) => i.job_id === job?.id);

  return (
    <Modal title="데이터셋 가져오기" onClose={onClose}>
      {!job && browsing && node ? (
        <FolderBrowser
          nodeId={node.id}
          nodeName={node.name}
          start={form.path}
          onCancel={() => setBrowsing(false)}
          onPick={(path, guess) => {
            const name = path.split(/[\\/]/).filter(Boolean).pop() ?? "";
            setForm({ ...form, path, dataset: form.dataset || name.toLowerCase() });
            setLayout(guess);
            setBrowsing(false);
          }}
        />
      ) : !job ? (
        <form onSubmit={submit}>
          <p className="muted">
            노드에 있는 폴더를 그대로 등록합니다. 이미지는 복사하지 않고, 해시·크기·라벨·썸네일만 Hub에 저장합니다.
          </p>
          <label>
            데이터가 있는 노드
            <select value={form.node_id} onChange={set("node_id")} required>
              {nodes.map((n) => (
                <option key={n.id} value={n.id}>
                  {n.name} {n.status === "online" ? "" : "(오프라인)"}
                </option>
              ))}
            </select>
          </label>
          <div className="field-label">
            폴더 경로 <span className="muted small">(그 노드에서의 경로)</span>
            <div className="input-with-button">
              <input
                value={form.path}
                onChange={(e) => {
                  setLayout("");
                  set("path")(e);
                }}
                required
                placeholder="/data/street 또는 D:\data\street"
                className="mono"
                aria-label="폴더 경로"
              />
              <button type="button" className="small" disabled={!node || node.status !== "online"} onClick={() => setBrowsing(true)}>
                <Icon name="folder" size={13} /> 찾아보기
              </button>
            </div>
            {layout && <span className="muted small field-hint">{LAYOUT_NAME[layout] ?? layout} 형식으로 보입니다. 형식은 자동 감지로 두면 됩니다.</span>}
            {node && node.status !== "online" && <span className="muted small field-hint">노드가 꺼져 있어 폴더를 볼 수 없습니다. 경로를 직접 적으세요.</span>}
          </div>
          <div className="form-row">
            <label>
              데이터셋 이름
              <input value={form.dataset} onChange={set("dataset")} required placeholder="예: coco" />
            </label>
            <label>
              버전
              <input value={form.version} onChange={set("version")} required />
            </label>
            <label>
              형식
              <select value={form.format} onChange={set("format")}>
                <option value="auto">자동 감지</option>
                <option value="yolo">YOLO</option>
                <option value="coco">COCO</option>
                <option value="imagefolder">분류 (클래스별 폴더)</option>
              </select>
            </label>
          </div>
          <details className="advanced" open={Boolean(form.site || form.camera || form.name_pattern)}>
            <summary>
              사이트 · 카메라 <span className="muted">(선택 · 필터와 split 묶음에 쓰임)</span>
            </summary>
            <label>
              파일 이름 규칙
              <input value={form.name_pattern} onChange={set("name_pattern")} placeholder="{site}_{camera}_*" className="mono" />
            </label>
            <p className="muted small">
              <code>{"{site}"}</code> · <code>{"{camera}"}</code> 자리의 글자를 읽고, <code>*</code>는 아무 글자입니다. 예: seoul_cam03_0001.jpg →{" "}
              <code>{"{site}_{camera}_*"}</code>, seoul/cam03/0001.jpg → <code>{"{site}/{camera}/*"}</code>. 등록한 뒤에도 버전 화면에서 채울 수
              있습니다.
            </p>
            <div className="form-row">
              <label>
                모든 이미지의 사이트
                <input value={form.site} onChange={set("site")} placeholder="규칙에 맞지 않을 때" />
              </label>
              <label>
                모든 이미지의 카메라
                <input value={form.camera} onChange={set("camera")} placeholder="규칙에 맞지 않을 때" />
              </label>
            </div>
          </details>
          <label>
            메모 (선택)
            <input value={form.note} onChange={set("note")} placeholder="이 버전에서 달라진 점" />
          </label>
          {error && <div className="error">{error}</div>}
          <div className="modal-actions">
            <button type="button" onClick={onClose}>
              취소
            </button>
            <button className="primary" disabled={!form.node_id}>
              가져오기
            </button>
          </div>
        </form>
      ) : (
        <>
          <div className="facts">
            <div>
              <div className="eyebrow">상태</div>
              <Badge kind={job.status}>
                {{ queued: "대기", running: "등록 중", succeeded: "완료", failed: "실패", canceled: "취소됨" }[job.status]}
              </Badge>
            </div>
            <div>
              <div className="eyebrow">데이터셋</div>
              {form.dataset} {form.version}
            </div>
            <div>
              <div className="eyebrow">시작</div>
              {timeAgo(job.created_at)}
            </div>
          </div>
          {job.wait_reason && job.status === "queued" && <div className="notice">{job.wait_reason}</div>}
          {progress && <ImportProgress info={progress} />}
          {lastLog && <pre className="json">{lastLog}</pre>}
          {job.error && <div className="error">{job.error} 자세한 내용은 작업 화면의 로그를 확인하세요.</div>}
          {(job.status === "queued" || job.status === "running") && (
            <p className="muted small">창을 닫아도 등록은 계속됩니다. 진행 상황은 화면 오른쪽 위 "등록 중"에서 볼 수 있습니다.</p>
          )}
          <div className="modal-actions">
            <button onClick={onClose} className={job.status === "queued" || job.status === "running" ? "primary" : ""}>
              {job.status === "queued" || job.status === "running" ? "닫고 다른 작업 하기" : "닫기"}
            </button>
            {releaseId && (
              <a
                className="btn primary-link"
                href={`#/datasets/${releaseId}`}
                onClick={onClose}
              >
                버전 열기 <Icon name="arrow" size={14} />
              </a>
            )}
          </div>
        </>
      )}
    </Modal>
  );
}
