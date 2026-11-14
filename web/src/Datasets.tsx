import { type FormEvent, useEffect, useMemo, useState } from "react";
import { ItemViewer } from "./ItemViewer";
import { type Job, type NodeInfo, api } from "./api";
import {
  type CompareResult,
  type ItemPage,
  type Project,
  type ReleaseInfo,
  SPLIT_LABEL,
  STATE_LABEL,
  type TreeDataset,
  bytesLabel,
} from "./catalogTypes";
import { Badge, Icon, Modal, Stat, timeAgo, usePolling } from "./ui";

const PAGE_SIZE = 60;
const PROJECT_KEY = "badamoa.project";

function remembered(): string {
  try {
    return localStorage.getItem(PROJECT_KEY) ?? "";
  } catch {
    return "";
  }
}

export function Datasets({ releaseId, isAdmin }: { releaseId?: string; isAdmin: boolean }) {
  const { data: projects, refresh: refreshProjects } = usePolling(() => api<Project[]>("/projects"), 30000);
  const [projectId, setProjectId] = useState(remembered);
  const [showTrash, setShowTrash] = useState(false);
  const [creating, setCreating] = useState(false);
  const [importing, setImporting] = useState(false);

  // Fall back to the first project when the remembered one is gone.
  useEffect(() => {
    if (projects && projects.length > 0 && !projects.some((p) => p.id === projectId)) setProjectId(projects[0].id);
  }, [projects, projectId]);

  const choose = (id: string) => {
    setProjectId(id);
    try {
      localStorage.setItem(PROJECT_KEY, id);
    } catch {
      /* not remembered across reloads */
    }
    window.location.hash = "#/datasets";
  };

  const { data: tree, refresh: refreshTree } = usePolling(
    () => (projectId ? api<TreeDataset[]>(`/projects/${projectId}/datasets?trash=${showTrash}`) : Promise.resolve([])),
    8000,
    [projectId, showTrash],
  );
  const project = projects?.find((p) => p.id === projectId);

  if (projects && projects.length === 0) {
    return (
      <>
        <div className="page-head">
          <div>
            <h1>데이터셋</h1>
            <p className="muted">프로젝트는 같은 클래스 체계를 쓰는 데이터셋·학습·모델을 묶는 단위입니다.</p>
          </div>
        </div>
        <div className="card empty-state">
          <Icon name="database" size={28} />
          <strong>첫 프로젝트를 만들어 보세요</strong>
          <p className="muted">프로젝트를 만든 뒤, 노드에 있는 YOLO·COCO 형식 데이터셋을 복사 없이 등록할 수 있습니다.</p>
          <button className="primary" onClick={() => setCreating(true)}>
            <Icon name="plus" />
            프로젝트 만들기
          </button>
        </div>
        {creating && (
          <NewProject
            onClose={() => setCreating(false)}
            onCreated={(id) => {
              setCreating(false);
              void refreshProjects();
              choose(id);
            }}
          />
        )}
      </>
    );
  }

  return (
    <div className="catalog">
      <aside className="catalog-side">
        <div className="eyebrow">프로젝트</div>
        <div className="project-row">
          <select value={projectId} onChange={(e) => choose(e.target.value)} aria-label="프로젝트">
            {projects?.map((p) => (
              <option key={p.id} value={p.id}>
                {p.folder ? `${p.folder} / ` : ""}
                {p.name}
              </option>
            ))}
          </select>
          <button className="icon-btn" onClick={() => setCreating(true)} title="새 프로젝트" aria-label="새 프로젝트">
            <Icon name="plus" />
          </button>
        </div>
        <button className="primary wide" onClick={() => setImporting(true)} disabled={!projectId}>
          <Icon name="plus" />
          데이터셋 가져오기
        </button>

        <div className="eyebrow tree-head">데이터셋 · 릴리스</div>
        {tree?.length === 0 && <p className="muted small">아직 등록한 데이터셋이 없습니다.</p>}
        <nav className="tree">
          {tree?.map((dataset) => (
            <div key={dataset.id} className="tree-group">
              <div className="tree-name">
                <Icon name="database" size={14} />
                <span>{dataset.name}</span>
                <em>{dataset.releases.length}</em>
              </div>
              {dataset.releases.map((r) => (
                <a
                  key={r.id}
                  href={r.deleted_at ? undefined : `#/datasets/${r.id}`}
                  className={`tree-release ${r.id === releaseId ? "active" : ""} ${r.deleted_at ? "deleted" : ""}`}
                >
                  <span>{r.version}</span>
                  {r.deleted_at ? (
                    isAdmin && (
                      <button
                        className="mini"
                        onClick={() => void api(`/releases/${r.id}/restore`, { method: "POST" }).then(refreshTree)}
                      >
                        복원
                      </button>
                    )
                  ) : r.status === "building" ? (
                    <em>등록 중</em>
                  ) : (
                    <em>{r.items.toLocaleString()}</em>
                  )}
                </a>
              ))}
            </div>
          ))}
        </nav>
        {isAdmin && (
          <label className="check trash-toggle">
            <input type="checkbox" checked={showTrash} onChange={(e) => setShowTrash(e.target.checked)} /> 삭제한 릴리스 보기
          </label>
        )}
      </aside>

      <div className="catalog-main">
        {releaseId ? (
          <ReleaseView
            key={releaseId}
            releaseId={releaseId}
            isAdmin={isAdmin}
            unit={project?.task === "classify" ? "라벨" : "박스"}
            onDeleted={() => void refreshTree()}
          />
        ) : (
          <div className="card empty-state">
            <Icon name="images" size={28} />
            <strong>{project ? project.name : "프로젝트"}</strong>
            <p className="muted">
              왼쪽에서 릴리스를 고르면 내용을 볼 수 있습니다. 데이터가 없다면 "데이터셋 가져오기"로 노드에 있는 폴더를 등록하세요.
              이미지는 복사하지 않고 제자리에 둡니다.
            </p>
          </div>
        )}
      </div>

      {creating && (
        <NewProject
          onClose={() => setCreating(false)}
          onCreated={(id) => {
            setCreating(false);
            void refreshProjects();
            choose(id);
          }}
        />
      )}
      {importing && projectId && (
        <ImportDialog
          projectId={projectId}
          onClose={() => {
            setImporting(false);
            void refreshTree();
          }}
        />
      )}
    </div>
  );
}

function ReleaseView({
  releaseId,
  isAdmin,
  unit,
  onDeleted,
}: {
  releaseId: string;
  isAdmin: boolean;
  unit: string;
  onDeleted: () => void;
}) {
  const { data: release, error } = usePolling(() => api<ReleaseInfo>(`/releases/${releaseId}`), 10000, [releaseId]);
  const [split, setSplit] = useState("all");
  const [state, setState] = useState("");
  const [classId, setClassId] = useState("");
  const [site, setSite] = useState("");
  const [camera, setCamera] = useState("");
  const [q, setQ] = useState("");
  const [page, setPage] = useState(0);
  const [viewing, setViewing] = useState<number | null>(null);
  const [compareWith, setCompareWith] = useState("");

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

  const { data: items } = usePolling(
    () => api<ItemPage>(`/releases/${releaseId}/items?${filterQuery}&offset=${page * PAGE_SIZE}&limit=${PAGE_SIZE}`),
    release?.status === "building" ? 5000 : 60000,
    [releaseId, filterQuery, page, release?.status],
  );

  if (error && !release) return <div className="error">{error}</div>;
  if (!release) return <div className="muted">불러오는 중…</div>;

  const stats = release.stats;
  const classes = release.label_space.classes;
  const classRows = classes
    .map((c) => ({ ...c, ...(stats.classes?.[String(c.id)] ?? { boxes: 0, items: 0 }) }))
    .sort((a, b) => b.boxes - a.boxes);
  const topBoxes = Math.max(1, ...classRows.map((c) => c.boxes));
  const splits = Object.entries(stats.splits ?? {}).sort();
  const pages = Math.max(1, Math.ceil((items?.total ?? 0) / PAGE_SIZE));

  const remove = async () => {
    if (!window.confirm(`${release.dataset.name} ${release.version} 릴리스를 삭제할까요? 휴지통에서 복원할 수 있고, 이미지 파일은 지워지지 않습니다.`)) return;
    await api(`/releases/${releaseId}`, { method: "DELETE" });
    onDeleted();
    window.location.hash = "#/datasets";
  };

  return (
    <>
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
            <a className="btn primary-link" href={`#/training/new/${release.id}`}>
              이 릴리스로 학습
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

      <details className="card fold">
        <summary>
          클래스별 {unit} 수 <span className="muted">({(stats.boxes ?? 0).toLocaleString()})</span>
        </summary>
        <div className="class-bars">
          {classRows.map((c) => (
            <button key={c.id} className="class-bar" onClick={() => setClassId(String(c.id))} title="이 클래스가 있는 항목만 보기">
              <span className="class-name">{c.name}</span>
              <span className="bar">
                <i style={{ width: `${(c.boxes / topBoxes) * 100}%` }} />
              </span>
              <span className="num">{c.boxes.toLocaleString()}</span>
              <span className="num muted">{c.items.toLocaleString()}장</span>
            </button>
          ))}
        </div>
      </details>
      {release.note && (
        <details className="card fold">
          <summary>메모</summary>
          <p className="note">{release.note}</p>
        </details>
      )}

      {release.siblings.length > 0 && (
        <div className="compare-row">
          <strong>비교</strong>
          <select value={compareWith} onChange={(e) => setCompareWith(e.target.value)} aria-label="비교할 릴리스">
            <option value="">비교할 릴리스 선택…</option>
            {release.siblings.map((s) => (
              <option key={s.id} value={s.id}>
                {s.version}
              </option>
            ))}
          </select>
          {release.base && (
            <button onClick={() => setCompareWith(release.base!.id)}>base {release.base.version}와 비교</button>
          )}
        </div>
      )}
      {compareWith && <CompareCard releaseId={releaseId} otherId={compareWith} onClose={() => setCompareWith("")} />}

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
            <option key={id} value={id}>
              {label}
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
      </div>

      <div className="pager">
        <button onClick={() => setPage(0)} disabled={page === 0} aria-label="첫 페이지">
          «
        </button>
        <button onClick={() => setPage(page - 1)} disabled={page === 0} aria-label="이전 페이지">
          ‹
        </button>
        <span className="num">
          {page + 1} / {pages.toLocaleString()}
        </span>
        <button onClick={() => setPage(page + 1)} disabled={page >= pages - 1} aria-label="다음 페이지">
          ›
        </button>
        <button onClick={() => setPage(pages - 1)} disabled={page >= pages - 1} aria-label="마지막 페이지">
          »
        </button>
      </div>

      {items && items.items.length === 0 && <div className="card empty">조건에 맞는 항목이 없습니다.</div>}
      <div className="thumb-grid">
        {items?.items.map((item, i) => (
          <button key={item.id} className="thumb" onClick={() => setViewing(page * PAGE_SIZE + i)}>
            <span className="thumb-image">
              {item.thumb_url ? <img src={item.thumb_url} alt="" loading="lazy" /> : <Icon name="images" size={22} />}
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

      {viewing != null && items && (
        <ItemViewer releaseId={releaseId} filterQuery={filterQuery} start={viewing} total={items.total} onClose={() => setViewing(null)} />
      )}
    </>
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
          {diff.label_space_changed && <div className="notice">두 릴리스의 라벨 공간이 다릅니다. 클래스는 이름으로 맞춰 비교했습니다.</div>}
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

function NewProject({ onClose, onCreated }: { onClose: () => void; onCreated: (id: string) => void }) {
  const [name, setName] = useState("");
  const [task, setTask] = useState("detect");
  const [folder, setFolder] = useState("");
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
        <label>
          폴더 (선택)
          <input value={folder} onChange={(e) => setFolder(e.target.value)} placeholder="프로젝트를 묶는 이름" />
        </label>
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
  const [form, setForm] = useState({ node_id: "", path: "", dataset: "", version: "v1", format: "auto", site: "", camera: "", note: "" });
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

  const submit = async (e: FormEvent) => {
    e.preventDefault();
    try {
      setError(null);
      setJob(await api<Job>(`/projects/${projectId}/imports`, { body: form }));
    } catch (err) {
      setError(err instanceof Error ? err.message : String(err));
    }
  };

  const releaseId = job?.status === "succeeded" ? (job.result as { release_id?: string } | null)?.release_id : undefined;

  return (
    <Modal title="데이터셋 가져오기" onClose={onClose}>
      {!job ? (
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
          <label>
            폴더 경로 (그 노드에서의 경로)
            <input value={form.path} onChange={set("path")} required placeholder="/data/street 또는 D:\data\street" className="mono" />
          </label>
          <div className="form-row">
            <label>
              데이터셋 이름
              <input value={form.dataset} onChange={set("dataset")} required placeholder="예: OD_Sur_Dev" />
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
          <div className="form-row">
            <label>
              사이트 (선택)
              <input value={form.site} onChange={set("site")} />
            </label>
            <label>
              카메라 (선택)
              <input value={form.camera} onChange={set("camera")} />
            </label>
          </div>
          <label>
            메모 (선택)
            <input value={form.note} onChange={set("note")} placeholder="이 릴리스에서 달라진 점" />
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
          {lastLog && <pre className="json">{lastLog}</pre>}
          {job.error && <div className="error">{job.error} 자세한 내용은 작업 화면의 로그를 확인하세요.</div>}
          <div className="modal-actions">
            <button onClick={onClose}>닫기</button>
            {releaseId && (
              <a
                className="btn primary-link"
                href={`#/datasets/${releaseId}`}
                onClick={onClose}
              >
                릴리스 열기 <Icon name="arrow" size={14} />
              </a>
            )}
          </div>
        </>
      )}
    </Modal>
  );
}
