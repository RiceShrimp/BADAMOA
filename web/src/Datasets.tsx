import { ReviewRecordCard } from "./ReviewRecord";
import { type FormEvent, useEffect, useMemo, useState } from "react";
import {
  FOLDER_HINT,
  FolderPicker,
  ProjectBoard,
  ProjectPage,
  SPLIT_COLOR,
  Shot,
  rememberProject,
  useFolders,
} from "./DataBoard";
import { FolderBrowser, LAYOUT_NAME } from "./FolderBrowser";
import { ImportProgress, announceImport, useImports } from "./Imports";
import { ChangeSummary, DraftActions, DraftLine, DraftNotice } from "./DraftBar";
import { CHANGE_LABEL } from "./labelDiff";
import { ItemViewer } from "./ItemViewer";
import { NewSetDialog } from "./Labeling";
import { Fingerprint, ReleaseCopies } from "./NodeCopies";
import { ExportDialog, type ItemFilter, NewReleaseDialog, describeFilter } from "./ReleaseMaker";
import { ChartsTab, ClassesTab, ModelsTab, ReleasesTab, colorFor } from "./ReleaseTabs";
import { type Job, type NodeInfo, api } from "./api";
import {
  type ChangeKind,
  type CompareResult,
  type DraftChanges,
  type ItemPage,
  type Project,
  type ReleaseCheck,
  type ReleaseInfo,
  SPLIT_LABEL,
  STATE_HELP,
  STATE_HINT,
  STATE_LABEL,
  TASK_HINT,
  TASK_LABEL,
  type TreeDataset,
  announceTrash,
  bytesLabel,
  taskLabel,
  type PredictionRun,
  runName,
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
  else if (sub === "project" && arg) body = <ProjectPage projectId={arg} onImport={() => setImporting(arg)} isAdmin={isAdmin} />;
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

/** A per-browser choice (the run over the grid …): "" when nothing is kept or storage is off. */
function kept(key: string): string {
  try {
    return localStorage.getItem(key) ?? "";
  } catch {
    return "";
  }
}

function keep(key: string, value: string) {
  try {
    localStorage.setItem(key, value);
  } catch {
    /* not kept */
  }
}

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
  // A model run's predictions over the thumbnails (evaluation or 오토라벨): picked in the toolbar, kept per version.
  const [gridRun, setGridRun] = useState<string>(() => kept(`badamoa.gridPredictions.${releaseId}`));
  const { data: predictionRuns } = usePolling(() => api<PredictionRun[]>(`/releases/${releaseId}/prediction-runs`), 60000, [releaseId]);
  const pickGridRun = (opId: string) => {
    setGridRun(opId);
    keep(`badamoa.gridPredictions.${releaseId}`, opId);
  };
  const gridScore = Number(kept("badamoa.predictionScore") || 0.25);
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
  const [fixedOnly, setFixedOnly] = useState(false); // only the images the work in progress changed
  const [changeKind, setChangeKind] = useState<ChangeKind | "">(""); // and only those changed this way
  const [suspect, setSuspect] = useState(""); // an evaluation: its images, likely wrong labels first
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
  const [labelingSet, setLabelingSet] = useState(false); // making a labeling set of the filtered images
  const [exporting, setExporting] = useState(false);
  const [renaming, setRenaming] = useState(false);

  const filterQuery = useMemo(() => {
    const params = new URLSearchParams();
    if (split !== "all") params.set("split", split === "none" ? "" : split);
    if (state) params.set("state", state);
    if (classId) params.set("class_id", classId);
    if (site) params.set("site", site);
    if (camera) params.set("camera", camera);
    if (q.trim()) params.set("q", q.trim());
    if (fixedOnly) params.set("edited", "true");
    if (fixedOnly && changeKind) params.set("changed", changeKind);
    if (suspect) params.set("suspect", suspect);
    return params.toString();
  }, [split, state, classId, site, camera, q, fixedOnly, changeKind, suspect]);
  // Evaluations of this dataset's models that can sort the images by likely wrong labels.
  const { data: checks } = usePolling(() => api<ReleaseCheck[]>(`/releases/${releaseId}/checks`), 60000, [releaseId]);
  const check = checks?.find((c) => c.id === suspect);
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

  const { data: items, refresh: refreshItems } = usePolling(
    () =>
      api<ItemPage>(
        `/releases/${releaseId}/items?${filterQuery}&offset=${page * pageSize}&limit=${pageSize}` +
          (gridRun && predictionRuns?.some((r) => r.op_id === gridRun) ? `&pred=${encodeURIComponent(gridRun)}` : ""),
      ),
    release?.status === "building" ? 5000 : 60000,
    [releaseId, filterQuery, page, pageSize, release?.status, gridRun, predictionRuns?.length],
  );

  if (error && !release) return <div className="error">{error}</div>;
  if (!release) return <div className="muted">불러오는 중…</div>;

  const project = projects.find((p) => p.id === release.project_id);
  const task = release.task ?? "detect"; // the dataset's: how labels are drawn, trained and exported
  const unit = task === "classify" ? "라벨" : "박스";
  const stats = release.stats;
  const classes = release.label_space.classes;
  const colorOf = colorFor(classes);
  const skeletonOf = (id: number) => classes.find((c) => c.id === id)?.skeleton;
  const splits = Object.entries(stats.splits ?? {}).sort();
  const isDraft = release.status === "draft";
  // The work in progress that changes this version's images: this one, or the one based on it.
  const fixing = isDraft || release.draft?.base?.id === release.id ? release.draft : null;
  const pickChange = (kind: ChangeKind | "") => {
    setChangeKind(kind);
    if (kind) setFixedOnly(true);
  };
  const title = isDraft ? "작업 중" : release.version;
  const pages = Math.max(1, Math.ceil((items?.total ?? 0) / pageSize));
  const present = Object.keys(stats.classes ?? {}).length;

  const remove = async () => {
    if (!window.confirm(`${release.dataset.name} ${release.version} 버전을 휴지통으로 옮길까요? 휴지통에서 되살릴 수 있고, 이미지 파일은 지워지지 않습니다.`)) return;
    await api(`/releases/${releaseId}`, { method: "DELETE" });
    announceTrash();
    window.location.hash = `#/datasets/project/${release.project_id}`;
  };

  return (
    <>
      <nav className="page-crumbs">
        <a href="#/datasets">데이터셋</a>
        {project?.folder && <span>{project.folder}</span>}
        <a href={`#/datasets/project/${release.project_id}`}>{project?.name ?? "프로젝트"}</a>
        <strong>
          {release.dataset.name} {title}
        </strong>
      </nav>
      <div className="page-head">
        <div>
          <h1 className="with-badge">
            <span className="muted-strong">{release.dataset.name}</span> {title}
            <span className="tag task-tag" title={TASK_HINT[task] ?? "직접 정한 태스크"}>
              {taskLabel(task)}
            </span>
            {release.status === "building" && <Badge kind="running">등록 중</Badge>}
            {isDraft && <Badge kind="queued">아직 버전 아님</Badge>}
            <button className="icon-btn title-edit" onClick={() => setRenaming(true)} title="데이터셋 · 버전 이름 바꾸기" aria-label="이름 바꾸기">
              <Icon name="edit" size={15} />
            </button>
          </h1>
          {isDraft ? (
            <DraftLine release={release} />
          ) : (
          <p className="muted">
            라벨 공간 {release.label_space.name} · 클래스 {classes.length}개 · {release.created_at.slice(0, 10)} 등록
            {release.base && (
              <>
                {" "}
                · base <a href={`#/datasets/${release.base.id}`}>{release.base.version}</a>
              </>
            )}
            {release.digest && (
              <>
                {" "}
                · 지문 <Fingerprint digest={release.digest} />
              </>
            )}
            {(release.same_as ?? []).length > 0 && (
              <>
                {" "}
                · 내용이 같은 버전{" "}
                {release.same_as!.map((r, i) => (
                  <span key={r.id}>
                    {i > 0 && ", "}
                    <a href={`#/datasets/${r.id}`}>
                      {r.dataset === release.dataset.name ? r.version : `${r.dataset} ${r.version}`}
                    </a>
                  </span>
                ))}
              </>
            )}
          </p>
          )}
        </div>
        {isDraft ? (
          <DraftActions
            release={release}
            onChanged={() => {
              void refresh();
              void refreshItems();
            }}
          />
        ) : (
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
            <a className="btn primary-link" href={`#/training/new/${release.id}`} onClick={() => rememberProject(release.project_id)}>
              이 버전으로 학습
            </a>
          )}
          {isAdmin && (
            <button className="danger" onClick={remove}>
              삭제
            </button>
          )}
        </div>
        )}
      </div>
      <DraftNotice release={release} />
      {release.status === "ready" && <ReviewRecordCard releaseId={release.id} />}
      {isDraft && fixing && fixing.edited > 0 && (
        <DraftChangesCard releaseId={release.id} edited={fixing.edited} picked={changeKind} onPick={pickChange} />
      )}

      <div className="stats five">
        <Stat icon="images" label="전체 항목" value={(stats.items ?? 0).toLocaleString()} sub={bytesLabel(stats.bytes)} />
        {splits.map(([name, s]) => (
          <Stat key={name} icon="database" label={SPLIT_LABEL[name] ?? name} value={s.items.toLocaleString()} sub={bytesLabel(s.bytes)} />
        ))}
        <Stat
          icon="check"
          label="라벨 있음"
          value={(stats.states?.labelled ?? 0).toLocaleString()}
          sub={`${unit} ${(stats.boxes ?? 0).toLocaleString()}개`}
        />
        <Stat icon="review" label="배경" value={(stats.states?.background ?? 0).toLocaleString()} sub="라벨 없음 · 찾을 물체 없음" />
        {(stats.states?.unlabelled ?? 0) > 0 && (
          <Stat icon="edit" label="아직 안 함" value={(stats.states?.unlabelled ?? 0).toLocaleString()} sub="라벨할 이미지 · 학습에서 빠짐" />
        )}
      </div>

      {release.status === "building" && <BuildingBar releaseId={release.id} />}
      {release.status === "ready" && <ReleaseCopies releaseId={release.id} />}
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
        {fixing && fixing.edited > 0 && (
          <label className="check inline-check" title="작업 중에서 라벨을 고친 이미지만 봅니다">
            <input
              type="checkbox"
              checked={fixedOnly}
              onChange={(e) => {
                setFixedOnly(e.target.checked);
                if (!e.target.checked) setChangeKind("");
              }}
            />
            고친 것만 ({fixing.edited.toLocaleString()})
          </label>
        )}
        {fixing && fixedOnly && (
          <select value={changeKind} onChange={(e) => setChangeKind(e.target.value as ChangeKind | "")} aria-label="바뀐 종류">
            <option value="">모든 바뀜</option>
            {(Object.keys(CHANGE_LABEL) as ChangeKind[]).map((k) => (
              <option key={k} value={k}>
                {CHANGE_LABEL[k]}된 라벨이 있는 것
              </option>
            ))}
          </select>
        )}
        {(checks?.length ?? 0) > 0 && (
          <select
            value={suspect}
            onChange={(e) => setSuspect(e.target.value)}
            aria-label="정렬"
            title="평가한 모델의 예측과 라벨이 어긋나는 이미지부터 봅니다"
            className={suspect ? "active-select" : ""}
          >
            <option value="">기본 순서</option>
            {checks?.map((c) => (
              <option key={c.id} value={c.id}>
                라벨 의심 순 · {c.model ? `${c.model.name} v${c.model.version}` : "모델"} ({c.release?.version} {c.split} · {c.created_at.slice(0, 10)})
              </option>
            ))}
          </select>
        )}
        <span className="muted count">{(items?.total ?? 0).toLocaleString()}개</span>
        <span className="spacer" />
        {release.fixable && (
          <button className="ghost-btn" onClick={() => setLabelingSet(true)} title="지금 거른 이미지를 일감으로 나눠 사람에게 맡깁니다">
            <Icon name="edit" size={13} /> 라벨링 묶음 만들기
          </button>
        )}
        {task !== "classify" && (
          <label className="check inline-check">
            <input type="checkbox" checked={showBoxes} onChange={(e) => setShowBoxes(e.target.checked)} />
            박스
          </label>
        )}
        {task !== "classify" && predictionRuns && predictionRuns.length > 0 && (
          <label className="inline-select" title="고른 모델의 예측을 썸네일에 점선으로 겹칩니다 (라벨은 실선)">
            <span className="muted small">예측</span>
            <select value={predictionRuns.some((r) => r.op_id === gridRun) ? gridRun : ""} onChange={(e) => pickGridRun(e.target.value)}>
              <option value="">겹쳐 보지 않음</option>
              {predictionRuns.map((r) => (
                <option key={r.op_id} value={r.op_id}>
                  {runName(r)} · {(r.images ?? 0).toLocaleString()}장
                </option>
              ))}
            </select>
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

      {check && (
        <div className="notice compact check-note">
          {check.model?.name} v{check.model?.version}이(가) {check.release?.version}의 {check.split} split을 평가할 때 예측과 라벨이 어긋난 이미지부터 봅니다
          · 평가한 이미지 {check.images.toLocaleString()}장 중 어긋난 것 {check.wrong.toLocaleString()}장. 숫자는 평가할 때의 라벨 기준이고, 이미지를
          열면 예측을 겹쳐 보고 라벨로 넣을 수 있습니다.
        </div>
      )}
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
                      <Shot
                        shot={{ url: item.thumb_url, width: item.width, height: item.height, boxes: item.boxes }}
                        colorOf={colorOf}
                        boxes={showBoxes}
                        task={task}
                        skeletonOf={skeletonOf}
                        predicted={item.pred}
                        minScore={gridScore}
                      />
                    )}
                  </td>
                  <td>
                    {item.file_name}
                    {item.fixed && <span className="tag fixed-tag">고침</span>}
                  </td>
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
                <Shot
                  shot={{ url: item.thumb_url, width: item.width, height: item.height, boxes: item.boxes }}
                  colorOf={colorOf}
                  boxes={showBoxes}
                  task={task}
                  skeletonOf={skeletonOf}
                  predicted={item.pred}
                  minScore={gridScore}
                />
              ) : (
                <Icon name="images" size={22} />
              )}
              <span className="split-dot" style={{ background: SPLIT_COLOR[item.split] }} title={SPLIT_LABEL[item.split] ?? item.split} />
              {item.fixed && !item.change && (
                <span className="fixed-mark" title={isDraft ? "이 작업 중에서 고친 이미지" : "작업 중에서 고친 이미지 (이 버전은 그대로)"}>
                  <Icon name="edit" size={11} />
                </span>
              )}
              {item.fixed && item.change && <ChangeBadge change={item.change} draft={isDraft} />}
              {item.check && item.check.missed + item.check.extra + (item.check.reclass ?? 0) > 0 && (
                <span
                  className="check-mark"
                  title={`모델이 못 찾은 라벨 ${item.check.missed} · 라벨에 없는 물체 ${item.check.extra} · 클래스 다름 ${item.check.reclass ?? 0} (FiftyOne 방식, 평가한 버전의 라벨 기준)`}
                >
                  {[
                    item.check.missed > 0 && `놓침 ${item.check.missed}`,
                    item.check.extra > 0 && `더 ${item.check.extra}`,
                    (item.check.reclass ?? 0) > 0 && `클래스 ${item.check.reclass}`,
                  ]
                    .filter(Boolean)
                    .join(" · ")}
                </span>
              )}
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
      {renaming && (
        <RenameDialog
          release={release}
          onClose={() => setRenaming(false)}
          onDone={() => {
            setRenaming(false);
            void refresh();
          }}
        />
      )}
      {labelingSet && (
        <NewSetDialog
          release={release}
          filter={{ ...filter, ...(fixedOnly ? { edited: true } : {}), ...(suspect ? { suspect } : {}) }}
          filtered={suspect ? (check?.wrong ?? 0) : (items?.total ?? 0)}
          describe={[describeFilter(filter, release), fixedOnly ? "고친 것만" : "", check ? `${check.model?.name} 평가에서 어긋난 것` : ""]
            .filter(Boolean)
            .join(" · ")}
          onClose={() => setLabelingSet(false)}
        />
      )}
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
      {viewing != null && items && (
        <ItemViewer
          releaseId={releaseId}
          filterQuery={filterQuery}
          start={viewing}
          total={items.total}
          colorOf={colorOf}
          classes={classes}
          task={task}
          checkOp={suspect || undefined}
          predictionRun={predictionRuns?.some((r) => r.op_id === gridRun) ? gridRun : undefined}
          onClose={() => setViewing(null)}
          onSaved={() => {
            void refresh();
            void refreshItems();
          }}
        />
      )}
    </>
  );
}

/** On a thumbnail: how the work in progress changed this image's labels (+added −removed ⤢moved ⇄reclassed). */
function ChangeBadge({ change, draft }: { change: Record<ChangeKind, number>; draft: boolean }) {
  const parts: [ChangeKind, string][] = [
    ["added", "+"],
    ["removed", "−"],
    ["moved", "⤢"],
    ["reclassed", "⇄"],
  ];
  const title = parts.map(([k]) => `${CHANGE_LABEL[k]} ${change[k]}`).join(" · ");
  const shown = parts.filter(([k]) => change[k] > 0);
  return (
    <span className="change-badge" title={`${draft ? "이 작업 중에서" : "작업 중에서"} 고침: ${title}`}>
      <Icon name="edit" size={10} />
      {shown.length === 0 ? <span>상태</span> : shown.map(([k, sign]) => (
        <span key={k} className={`change-${k}`}>
          {sign}
          {change[k]}
        </span>
      ))}
    </span>
  );
}

/** The work in progress's change summary, above its grid: a kind of change filters the grid. */
function DraftChangesCard({
  releaseId,
  edited,
  picked,
  onPick,
}: {
  releaseId: string;
  edited: number;
  picked: ChangeKind | "";
  onPick: (kind: ChangeKind | "") => void;
}) {
  const { data: changes } = usePolling(() => api<DraftChanges>(`/releases/${releaseId}/changes`), 30000, [releaseId, edited]);
  if (!changes) return null;
  return (
    <div className="card draft-changes">
      <div className="eyebrow">바뀐 내용 · {changes.base?.version ?? "바탕"}과 비교 · 이미지 {changes.images.toLocaleString()}장</div>
      <ChangeSummary changes={changes} picked={picked} onPick={onPick} />
    </div>
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

/**
 * Rename the dataset (every version goes by the new name) and this version. A work in progress has no name of its
 * own: it gets one when it is made a version. Runs and models keep pointing at the version; their own names stay.
 */
function RenameDialog({ release, onClose, onDone }: { release: ReleaseInfo; onClose: () => void; onDone: () => void }) {
  const isDraft = release.status === "draft";
  const [name, setName] = useState(release.dataset.name);
  const [version, setVersion] = useState(isDraft ? "" : release.version);
  const [busy, setBusy] = useState(false);
  const [error, setError] = useState<string | null>(null);
  const others = release.siblings.length + (isDraft ? 0 : 1);
  const changed = name.trim() !== release.dataset.name || (!isDraft && version.trim() !== release.version);

  const submit = async (e: FormEvent) => {
    e.preventDefault();
    setBusy(true);
    setError(null);
    try {
      if (name.trim() !== release.dataset.name) await api(`/datasets/${release.dataset.id}`, { method: "PATCH", body: { name: name.trim() } });
      if (!isDraft && version.trim() !== release.version) await api(`/releases/${release.id}`, { method: "PATCH", body: { version: version.trim() } });
      onDone();
    } catch (err) {
      setError(err instanceof Error ? err.message : String(err));
      setBusy(false);
    }
  };

  return (
    <Modal title="이름 바꾸기" onClose={onClose}>
      <form onSubmit={submit}>
        <label>
          데이터셋 이름
          <input value={name} onChange={(e) => setName(e.target.value)} maxLength={128} required autoFocus />
          <span className="muted small">이 데이터셋의 버전 {others}개가 모두 이 이름으로 바뀝니다.</span>
        </label>
        {isDraft ? (
          <p className="muted small">작업 중은 아직 버전 이름이 없습니다. 버전으로 확정할 때 정합니다.</p>
        ) : (
          <label>
            버전 이름
            <input value={version} onChange={(e) => setVersion(e.target.value)} maxLength={64} required />
          </label>
        )}
        <p className="muted small">
          학습 · 모델은 이 버전을 그대로 가리킵니다. 그때 붙은 학습 · 모델 이름은 바뀌지 않습니다. 같은 이름의 데이터셋 · 버전이 있으면 바꿀
          수 없습니다.
        </p>
        {error && <div className="error">{error}</div>}
        <div className="modal-actions">
          <button type="button" onClick={onClose}>
            취소
          </button>
          <button className="primary" disabled={busy || !changed || !name.trim() || (!isDraft && !version.trim())}>
            바꾸기
          </button>
        </div>
      </form>
    </Modal>
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
  const [folder, setFolder] = useState(initialFolder);
  const [error, setError] = useState<string | null>(null);

  const submit = async (e: FormEvent) => {
    e.preventDefault();
    try {
      const project = await api<Project>("/projects", { body: { name, folder } });
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
        <div className="field-label">
          폴더 <span className="muted small">(선택)</span>
          <FolderPicker value={folder} folders={folders} onChange={setFolder} />
          <span className="muted small field-hint">{FOLDER_HINT}</span>
        </div>
        <p className="muted small">태스크(Det · Seg · Kpt · Cls 등)는 데이터셋을 가져올 때 데이터셋마다 정합니다.</p>
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

const CUSTOM_TASK = "\u0000custom"; // the "직접 입력" choice; never a task's name

/** The version after a dataset's newest: v1, v2 → v3 (names in the trash count as taken). */
function nextVersion(d: TreeDataset): string {
  const taken = new Set(d.releases.map((r) => r.version.toLowerCase()));
  const numbers = d.releases.map((r) => /^v(\d+)$/i.exec(r.version)?.[1]).filter(Boolean).map(Number);
  let n = (numbers.length ? Math.max(...numbers) : d.releases.length) + 1;
  while (taken.has(`v${n}`)) n++;
  return `v${n}`;
}

function ImportDialog({ projectId, onClose }: { projectId: string; onClose: () => void }) {
  const [nodes, setNodes] = useState<NodeInfo[]>([]);
  const [form, setForm] = useState({
    node_id: "",
    path: "",
    dataset: "",
    version: "v1",
    format: "auto",
    note: "",
    unlabelled: "auto", // an image without a label file: auto | background | unlabelled
    classes: "", // class names for a folder of images to label, "car, person"
  });
  const [job, setJob] = useState<Job | null>(null);
  const [lastLog, setLastLog] = useState("");
  const [error, setError] = useState<string | null>(null);
  // The task: "" (자동: from the folder), a built-in one, a name used before, or CUSTOM (typed).
  const [taskChoice, setTaskChoice] = useState("");
  const [customTask, setCustomTask] = useState("");
  const [named, setNamed] = useState<string[]>([]); // tasks teams named, anywhere on the Hub
  // This project's datasets (versions in the trash too: their names are taken), and the one picked ("" = a new one).
  const [datasets, setDatasets] = useState<TreeDataset[]>([]);
  const [picked, setPicked] = useState("");
  const live = datasets.filter((d) => d.releases.some((r) => !r.deleted_at));
  const datasetTasks: Record<string, string> = Object.fromEntries(datasets.map((d) => [d.name, d.task]));

  useEffect(() => {
    void api<TreeDataset[]>(`/projects/${projectId}/datasets?trash=true`).then(setDatasets);
    void api<Project[]>("/projects").then((list) =>
      setNamed([...new Set(list.flatMap((p) => p.tasks ?? []))].filter((t) => !TASK_LABEL[t]).sort()),
    );
  }, [projectId]);

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
  // A dataset picked from the list gets its next version; a new one starts at v1.
  const pickDataset = (name: string) => {
    setPicked(name);
    const d = datasets.find((x) => x.name === name);
    setForm((f) => ({ ...f, dataset: name, version: d ? nextVersion(d) : "v1" }));
  };
  const [browsing, setBrowsing] = useState(false);
  const [layout, setLayout] = useState(""); // what the picked folder looks like
  const node = nodes.find((n) => n.id === form.node_id);

  // A version of a dataset picked from the list keeps that dataset's task. A new dataset's name must be
  // new: one already taken is not quietly made its next version (with that dataset's task) but offered as such.
  const existingTask = picked ? datasetTasks[picked] : undefined;
  const clash = !picked && form.dataset.trim() !== "" ? datasets.find((d) => d.name === form.dataset.trim()) : undefined;
  const clashLive = clash ? live.some((d) => d.id === clash.id) : false;
  const task = existingTask ? "" : taskChoice === CUSTOM_TASK ? customTask.trim() : taskChoice;

  const submit = async (e: FormEvent) => {
    e.preventDefault();
    try {
      setError(null);
      const classes = form.classes
        .split(/[,\n]/)
        .map((c) => c.trim())
        .filter(Boolean);
      setJob(await api<Job>(`/projects/${projectId}/imports`, { body: { ...form, classes, task } }));
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
          <div className="field-label">
            데이터셋
            {live.length > 0 ? (
              <div className={picked ? "pick-or-type" : "pick-or-type two"}>
                <select value={picked} onChange={(e) => pickDataset(e.target.value)} aria-label="데이터셋">
                  <option value="">새 데이터셋 (이름 입력)</option>
                  {live.map((d) => (
                    <option key={d.id} value={d.name}>
                      {d.name} · {taskLabel(d.task)} · 최근 {d.releases.find((r) => !r.deleted_at)?.version}
                    </option>
                  ))}
                </select>
                {!picked && (
                  <input value={form.dataset} onChange={set("dataset")} required placeholder="예: coco" aria-label="새 데이터셋 이름" />
                )}
              </div>
            ) : (
              <input value={form.dataset} onChange={set("dataset")} required placeholder="예: coco" aria-label="데이터셋 이름" />
            )}
            {picked && <span className="muted small field-hint">{picked}의 새 버전으로 들어갑니다. 버전 이름은 바꿀 수 있습니다.</span>}
            {clash && (
              <div className="notice compact">
                {clashLive ? (
                  <>
                    이미 있는 데이터셋 이름입니다(태스크 {taskLabel(clash.task)}). 새 데이터셋이면 다른 이름을 쓰세요.{" "}
                    <button type="button" className="small" onClick={() => pickDataset(clash.name)}>
                      그 데이터셋의 새 버전으로
                    </button>
                  </>
                ) : (
                  <>휴지통에 같은 이름의 데이터셋이 있습니다. 다른 이름을 쓰거나 휴지통에서 되살리세요.</>
                )}
              </div>
            )}
          </div>
          <div className="form-row">
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
                <option value="imagefolder">클래스별 폴더</option>
              </select>
            </label>
          </div>
          <div className="field-label">
            태스크
            {existingTask ? (
              <>
                <select value={existingTask} disabled aria-label="태스크">
                  <option value={existingTask}>{taskLabel(existingTask)}</option>
                </select>
                <span className="muted small field-hint">
                  기존 데이터셋 {picked}의 새 버전이라 그 태스크({taskLabel(existingTask)}) 그대로입니다. 다른 태스크로 가져오려면
                  데이터셋에서 "새 데이터셋"을 고르세요.
                </span>
              </>
            ) : (
              <>
                <div className={taskChoice === CUSTOM_TASK ? "pick-or-type two" : "pick-or-type"}>
                  <select value={taskChoice} onChange={(e) => setTaskChoice(e.target.value)} aria-label="태스크">
                    <option value="">자동 · 폴더를 보고 정함</option>
                    {Object.entries(TASK_LABEL).map(([id, short]) => (
                      <option key={id} value={id}>
                        {short} · {TASK_HINT[id]}
                      </option>
                    ))}
                    {named.map((t) => (
                      <option key={t} value={t}>
                        {t}
                      </option>
                    ))}
                    <option value={CUSTOM_TASK}>직접 입력…</option>
                  </select>
                  {taskChoice === CUSTOM_TASK && (
                    <input
                      value={customTask}
                      onChange={(e) => setCustomTask(e.target.value)}
                      required
                      maxLength={24}
                      pattern="[A-Za-z][A-Za-z0-9_\-]*"
                      title="영문으로 시작하는 영문 · 숫자 · - · _"
                      placeholder="예: OCR, Depth"
                      aria-label="태스크 이름"
                      autoFocus
                    />
                  )}
                </div>
                <span className="muted small field-hint">
                  {taskChoice === ""
                    ? "클래스별 폴더는 Cls, kpt_shape · keypoints가 있으면 Kpt, 윤곽이 있으면 Seg, 그 밖은 Det."
                    : taskChoice === CUSTOM_TASK
                      ? "라벨 값을 그대로 둡니다. 학습은 이 태스크를 적은 모델 템플릿으로 합니다."
                      : "폴더의 라벨이 이 태스크에 맞지 않으면 가져오지 않고 이유를 알립니다."}
                </span>
              </>
            )}
          </div>
          <details className="advanced" open={form.unlabelled !== "auto" || Boolean(form.classes)}>
            <summary>
              라벨이 없는 이미지 <span className="muted">(라벨할 이미지를 가져올 때)</span>
            </summary>
            <label>
              라벨 파일이 없는 이미지
              <select value={form.unlabelled} onChange={set("unlabelled")}>
                <option value="auto">자동 · 폴더에 라벨 파일이 하나도 없으면 아직 안 함, 아니면 배경</option>
                <option value="background">배경 · 찾을 물체가 없는 이미지</option>
                <option value="unlabelled">아직 안 함 · 라벨할 이미지</option>
              </select>
            </label>
            <label>
              클래스 <span className="muted small">(클래스 이름이 없는 폴더에)</span>
              <input value={form.classes} onChange={set("classes")} placeholder="예: car, person, bike" />
            </label>
            <p className="muted small">
              이미지만 있는 폴더는 클래스 목록이 없습니다. 여기 적은 클래스로 라벨을 고칠 수 있습니다. 비우면 기존 데이터셋의 새 버전일 때 그
              데이터셋의 클래스를 씁니다. "아직 안 함" 이미지는 학습에서 빠지고, 이미지 보기의 "라벨 고치기"로 라벨하면 들어갑니다.
            </p>
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
            <button className="primary" disabled={!form.node_id || Boolean(clash)}>
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
