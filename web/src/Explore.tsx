import { type FormEvent, useEffect, useState } from "react";
import { useDebounced } from "./Annotate";
import { useBoardFilter } from "./Boards";
import { ItemViewer } from "./ItemViewer";
import { colorFor } from "./ReleaseTabs";
import { TaskTags, rememberProject } from "./DataBoard";
import { type Order, OrderReset, SortHandle, SortItem, Sortable, useOrder } from "./Order";
import { DataAfterChoice, type GpuChoice, GpuChips, ImagesNote, PlaceGrid, PlaceRow, type PlaceServer, ServerSelect, copiedTo, gpuBody, spaceShort } from "./ServerPlace";
import { ProjectPicker, useProject } from "./ProjectPicker";
import { LogTail } from "./Training";
import { api } from "./api";
import type { ReleaseInfo, TreeDataset } from "./catalogTypes";
import type { NodeGpu } from "./trainTypes";
import { Badge, Icon, Modal, STATUS_LABEL, Stat, duration, timeAgo, usePolling } from "./ui";
import "./autolabel.css";
import "./explore.css";

/** How much of a project 이미지 탐색 can find (``/projects/{id}/explore``). */
interface ExploreStatus {
  model: string;
  title: string;
  images: number;
  embedded: number;
  text: { offered: boolean; ready: boolean; problem: string };
}

/** An image found: how near (cosine) and a version it is in. */
interface Found {
  id: number;
  file_name: string;
  width: number;
  height: number;
  score: number;
  thumb_url: string | null;
  image_url: string | null;
  release: { id: string; version: string; dataset: string } | null;
}

type Query = { text: string } | { like: Found };

/** What a search sent back: the nearest images at least ``min_score`` near, and whether more are left. */
interface SearchResult {
  searched: number;
  min_score: number;
  more: boolean;
  items: Found[];
}

/** The lowest score shown. Words and images score on different scales (words 0.1~0.2 is a good match, a similar
 * image 0.7 and up), so each kind of search keeps its own, remembered in this browser. */
type ScoreKind = "text" | "like";
const SCORE_KEY = "badamoa.explore.min";
const SCORE_RANGE: Record<ScoreKind, { max: number; step: number; digits: number }> = {
  text: { max: 0.4, step: 0.005, digits: 3 },
  like: { max: 1, step: 0.01, digits: 2 },
};
const FEW = 120; // the nearest shown with no lowest score
const MANY = 300; // with one: everything at least that near, up to this many

function keptScore(kind: ScoreKind): number {
  try {
    const kept = Number(localStorage.getItem(`${SCORE_KEY}.${kind}`));
    return Number.isFinite(kept) && kept >= 0 && kept <= SCORE_RANGE[kind].max ? kept : 0;
  } catch {
    return 0;
  }
}

/** A version and how far its images have vectors (``/projects/{id}/explore/versions``). */
interface VersionRow {
  release: { id: string; version: string; dataset: string };
  total: number;
  done: number;
  todo: number;
  going: string | null;
}

/** The servers a version's vectors can be made on (``/releases/{id}/embeddings/places``). */
interface Places {
  default: string;
  holder: { id: string; name: string } | null;
  items: number;
  bytes: number;
  nodes: (PlaceServer & { gpu_list: NodeGpu[] })[];
  problem: string;
}

/** An embedding job as a record (``/embeddings/jobs``), as trainings, evaluations and checks are listed. */
export interface EmbedJob {
  id: string;
  status: string;
  error: string | null;
  wait_reason: string | null;
  model: string;
  model_title: string;
  release: { id: string; version: string; dataset: string } | null;
  project: { id: string; name: string } | null;
  node_name: string;
  gpus: number;
  gpu_indices: number[];
  images: number | null; // the images it had to make
  made: number; // vectors it sent so far
  missing: number | null;
  text: boolean; // it also makes 글 검색's text encoder
  duration_s: number | null;
  created_at: string;
  finished_at: string | null;
  log_count: number;
  by: string;
  via: string;
  run_id: string | null;
  data_after: string; // keep | delete: the copy its server received, once it is over ("": registered there, or kept)
  sent: { job_id: string; status: string; error: string | null; node_name: string; duration_s: number | null } | null;
  coverage?: { total: number; done: number; todo: number };
}

interface ProjectCard {
  project: { id: string; name: string; folder: string; tasks: string[] };
  images: number;
  embedded: number;
  versions: number;
  jobs: number;
  going: number;
  failed: number;
  recent: EmbedJob[];
  last_at: string | null;
}

interface BoardInfo {
  model: { name: string; title: string };
  text: { offered: boolean; ready: boolean; problem: string };
  going: EmbedJob[];
  projects: ProjectCard[];
  totals: { projects: number; images: number; embedded: number; jobs: number; going: number; failed: number };
}

const GOING = ["queued", "assigned", "running"];
const STATUS_KIND: Record<string, string> = {
  running: "running",
  assigned: "running",
  queued: "queued",
  succeeded: "succeeded",
  failed: "failed",
};
const VIA_LABEL: Record<string, string> = { training: "학습과 함께", pipeline: "파이프라인", ui: "웹", api: "API", mcp: "에이전트" };
const n = (v: number | null | undefined) => (v == null ? "–" : v.toLocaleString());

function StatusBadge({ status }: { status: string }) {
  return <Badge kind={STATUS_KIND[status] ?? "idle"}>{STATUS_LABEL[status] ?? status}</Badge>;
}

const versionName = (j: EmbedJob) => (j.release ? `${j.release.dataset} ${j.release.version}` : "지운 버전");

/** "120/500장" while it runs, "500장" once done: what a job made of what it had to. */
function madeText(j: EmbedJob): string {
  if (j.status === "succeeded") return `${n(j.made)}장`;
  if (j.images == null) return j.made ? `${n(j.made)}장` : "";
  return `${n(j.made)}/${n(j.images)}장`;
}

/** 이미지 탐색: the board of projects, a project's search page, or one embedding job's record. */
export function Explore({ sub, arg }: { sub?: string; arg?: string }) {
  if (sub === "project" && arg) return <ExploreProject key={arg} projectId={arg} />;
  if (sub === "job" && arg) return <EmbedJobPage key={arg} jobId={arg} />;
  return <ExploreBoard />;
}

// ---- the board ---------------------------------------------------------------------------

const cardProject = (c: ProjectCard) => c.project;
const cardId = (c: ProjectCard) => c.project.id;

function ExploreBoard() {
  const [going, setGoing] = useState(false);
  const { data: board, error, refresh } = usePolling(() => api<BoardInfo>("/explore/board"), going ? 4000 : 20000);
  useEffect(() => setGoing((board?.going.length ?? 0) > 0), [board]);
  const order = useOrder("explore.projects", board?.projects, cardId);
  const { shown, tools, layout } = useBoardFilter(board ? order.ordered : null, cardProject, "explore");
  const [starting, setStarting] = useState(false);
  if (error && !board) return <div className="error">{error}</div>;
  const t = board?.totals;
  const text = board?.text;
  return (
    <>
      <div className="page-head">
        <div>
          <h1>이미지 탐색</h1>
          <p className="muted">
            프로젝트의 이미지를 글이나 비슷한 이미지로 찾습니다. 찾는 것은 벡터가 있는 이미지뿐이라, 먼저 새 임베딩으로 데이터셋 버전의 벡터를 만듭니다(새
            학습의 &quot;학습하면서 이미지 임베딩도 만들기&quot;나 파이프라인의 임베딩 단계로도 만듭니다). 프로젝트를 누르면 찾기 화면으로 갑니다.
          </p>
        </div>
        <div className="head-actions">
          <OrderReset order={order as Order<unknown>} />
          <button className="primary" onClick={() => setStarting(true)}>
            <Icon name="plus" />새 임베딩
          </button>
        </div>
      </div>
      <div className="stats five">
        <Stat icon="folder" label="프로젝트" value={t?.projects ?? "–"} sub="데이터셋 버전이 있는 프로젝트" />
        <Stat icon="images" label="찾을 수 있는 이미지" value={n(t?.embedded)} sub={`이미지 ${n(t?.images)}장 중 벡터가 있는 것`} />
        <Stat icon="bot" label="임베딩 작업" value={t?.jobs ?? "–"} sub={t?.going ? `도는 중 ${t.going}` : "도는 것 없음"} />
        <Stat
          icon="eye"
          label="글 검색"
          value={!text ? "–" : text.ready ? "준비됨" : "아직"}
          sub={!text ? " " : text.ready ? "Hub가 검색어를 벡터로 바꿉니다" : text.problem || "첫 임베딩 작업이 끝나면 됩니다"}
        />
        <Stat icon="x" label="실패" value={t?.failed ?? "–"} sub="기록에서 로그를 봅니다" />
      </div>

      {board && board.going.length > 0 && (
        <section className="board-section">
          <h2>
            지금 도는 임베딩 <span className="muted">{board.going.length}</span>
          </h2>
          <div className="going-list">
            {board.going.map((j) => (
              <a key={j.id} className="card going-run" href={`#/explore/job/${j.id}`}>
                <div className="going-title">
                  <StatusBadge status={j.status} />
                  <strong>{versionName(j)}</strong>
                  <span className="muted small">· {j.project?.name}</span>
                </div>
                <div className="going-meta small">
                  <span>벡터 {madeText(j)}</span>
                  <span className="muted">
                    {j.node_name || "서버 ?"} · {j.gpus ? "GPU" : "CPU"}
                    {j.duration_s != null ? ` · ${duration(j.duration_s)} 지남` : ""}
                  </span>
                </div>
              </a>
            ))}
          </div>
        </section>
      )}

      <section className="board-section">
        <h2>프로젝트</h2>
        {tools}
        {!board && <p className="muted">불러오는 중…</p>}
        {board && board.projects.length === 0 && (
          <div className="card empty">데이터셋 버전이 있는 프로젝트가 없습니다. 먼저 데이터셋을 가져오세요.</div>
        )}
        <Sortable ids={shown.map(cardId)} onMove={order.move}>
          <div className={`card-grid board-grid${layout}`}>
            {shown.map((c) => (
              <SortItem key={c.project.id} id={c.project.id}>
                <ProjectBoardCard card={c} />
              </SortItem>
            ))}
          </div>
        </Sortable>
      </section>
      {starting && (
        <NewEmbedding
          onClose={() => setStarting(false)}
          onStarted={(jobId) => {
            setStarting(false);
            void refresh();
            window.location.hash = `#/explore/job/${jobId}`;
          }}
        />
      )}
    </>
  );
}

function Meter({ done, total }: { done: number; total: number }) {
  return (
    <span className="explore-meter" title={`${n(done)}/${n(total)}장`}>
      <i style={{ width: `${total ? (100 * done) / total : 0}%` }} />
    </span>
  );
}

function ProjectBoardCard({ card: c }: { card: ProjectCard }) {
  return (
    <div className={`data-card${c.embedded === 0 && c.jobs === 0 ? " board-quiet" : ""}`}>
      <a className="data-card-link" href={`#/explore/project/${c.project.id}`} onClick={() => rememberProject(c.project.id)}>
        <div className="data-card-body">
          <div className="data-card-title">
            <span className="with-grip">
              <SortHandle />
              <strong>{c.project.name}</strong>
            </span>
            {c.going > 0 && <Badge kind="running">도는 중 {c.going}</Badge>}
          </div>
          {c.project.tasks.length > 0 && (
            <div className="task-tags">
              <TaskTags tasks={c.project.tasks} />
            </div>
          )}
          <div className="explore-card-meter">
            <Meter done={c.embedded} total={c.images} />
            <span className="small nowrap">
              <strong className="num">{n(c.embedded)}</strong>
              <span className="muted">/{n(c.images)}장 찾을 수 있음</span>
            </span>
          </div>
          {c.jobs === 0 ? (
            <p className="muted small">아직 임베딩하지 않았습니다 · 버전 {c.versions}개</p>
          ) : (
            <div className="small muted">
              버전 {c.versions}개 · 임베딩 {c.jobs}번
              {c.failed > 0 && <span className="bad-text"> · 실패 {c.failed}</span>}
            </div>
          )}
        </div>
      </a>
      {c.recent.length > 0 && (
        <div className="board-lines">
          {c.recent.map((j) => (
            <a key={j.id} href={`#/explore/job/${j.id}`} className="board-line">
              <StatusBadge status={j.status} />
              <span className="ellipsis">
                {versionName(j)} <span className="muted">{VIA_LABEL[j.via] ?? ""}</span>
              </span>
              <span className="num muted" title="만든 벡터">
                {madeText(j)}
              </span>
            </a>
          ))}
          {c.last_at && <div className="muted small board-foot">마지막 임베딩 {timeAgo(c.last_at)}</div>}
        </div>
      )}
    </div>
  );
}

// ---- 새 임베딩 -------------------------------------------------------------------------------

/**
 * 새 임베딩: a version's images that have no vector yet, made on a server with the 임베딩용 role (one that has the
 * images first, else one that receives them), on its GPU or CPU. The same picture embedded for another version or
 * project is skipped.
 */
export function NewEmbedding({
  projectId: fixedProject,
  releaseId: fixedRelease,
  onClose,
  onStarted,
}: {
  projectId?: string;
  releaseId?: string;
  onClose: () => void;
  onStarted: (jobId: string) => void;
}) {
  const { projects, projectId: remembered, choose } = useProject();
  const [picked, setPicked] = useState(fixedProject ?? "");
  const projectId = fixedProject ?? (picked || remembered);
  const { data: rows } = usePolling(
    () => (projectId ? api<VersionRow[]>(`/projects/${projectId}/explore/versions`) : Promise.resolve(null)),
    0,
    [projectId],
  );
  const { data: status } = usePolling(
    () => (projectId ? api<ExploreStatus>(`/projects/${projectId}/explore`) : Promise.resolve(null)),
    0,
    [projectId],
  );
  const [releaseId, setReleaseId] = useState(fixedRelease ?? "");
  const textMissing = Boolean(status?.text.offered && !status.text.ready);
  const startable = (r: VersionRow) => !r.going && (r.todo > 0 || textMissing);
  // Where to begin: the version given, else the one with the most images still to embed.
  useEffect(() => {
    if (!rows || (releaseId && rows.some((r) => r.release.id === releaseId))) return;
    const best = [...rows].filter(startable).sort((a, b) => b.todo - a.todo)[0];
    setReleaseId(best?.release.id ?? rows[0]?.release.id ?? "");
  }, [rows, textMissing]); // eslint-disable-line react-hooks/exhaustive-deps
  const row = rows?.find((r) => r.release.id === releaseId);
  const { data: servers, error: serverError } = usePolling(
    () => (releaseId ? api<Places>(`/releases/${releaseId}/embeddings/places`) : Promise.resolve(null)),
    0,
    [releaseId],
  );
  // the server is picked, never chosen for you (a version's other servers differ)
  const [nodeId, setNodeId] = useState("");
  useEffect(() => setNodeId(""), [servers]);
  const server = servers?.nodes.find((s) => s.id === nodeId);
  const [gpu, setGpu] = useState<GpuChoice>("auto");
  useEffect(() => setGpu("auto"), [nodeId]);
  // as 새 학습: what becomes of the copy the server received (or receives) once it is over
  const [dataAfter, setDataAfter] = useState<"keep" | "delete">("keep");
  const copied = copiedTo(server);
  const short = spaceShort(server, servers?.bytes ?? 0);
  const [saving, setSaving] = useState(false);
  const [error, setError] = useState<string | null>(null);
  const blocked = !row || !startable(row) || !server?.fits || short;

  const submit = async (e: FormEvent) => {
    e.preventDefault();
    if (blocked) return;
    setSaving(true);
    setError(null);
    try {
      const body = { node_id: nodeId, ...gpuBody(gpu, server), data_after: copied ? dataAfter : "" };
      const got = await api<{ job_id: string }>(`/releases/${releaseId}/embeddings`, { body });
      onStarted(got.job_id);
    } catch (err) {
      setError(err instanceof Error ? err.message : String(err));
    } finally {
      setSaving(false);
    }
  };

  return (
    <Modal title="새 임베딩" onClose={onClose}>
      <form onSubmit={submit} className="al-start">
        {!fixedProject && (
          <ProjectPicker
            projects={projects}
            projectId={projectId}
            onChange={(id) => {
              setPicked(id);
              choose(id);
              setReleaseId("");
            }}
          />
        )}
        {status && (
          <p className="muted small">
            {status.title} · 이 프로젝트 이미지 {n(status.images)}장 중 {n(status.embedded)}장에 벡터가 있습니다
            {status.text.offered && !status.text.ready ? " · 글 검색에 쓸 텍스트 인코더도 이 작업이 함께 만듭니다" : ""}
          </p>
        )}

        <label>
          데이터셋 버전
          {rows && rows.length === 0 ? (
            <span className="muted small">이 프로젝트에는 등록이 끝난 버전이 없습니다.</span>
          ) : (
            <select value={releaseId} onChange={(e) => setReleaseId(e.target.value)} aria-label="데이터셋 버전" disabled={Boolean(fixedRelease) || !rows}>
              {!rows && <option value="">불러오는 중…</option>}
              {rows?.map((r) => (
                <option key={r.release.id} value={r.release.id} disabled={!startable(r)}>
                  {r.release.dataset} {r.release.version} · 벡터 {n(r.done)}/{n(r.total)}장
                  {r.going ? " · 만드는 중" : r.todo === 0 ? " · 모두 있음" : ` · 만들 이미지 ${n(r.todo)}장`}
                </option>
              ))}
            </select>
          )}
          {row && (
            <span className="muted small">
              {row.going ? (
                <>
                  이 버전의 벡터를 지금 만들고 있습니다 · <a href={`#/explore/job/${row.going}`}>기록 보기</a>
                </>
              ) : row.todo === 0 ? (
                textMissing ? (
                  "이미지는 모두 벡터가 있어 글 검색의 텍스트 인코더만 만듭니다."
                ) : (
                  "이 버전의 이미지는 모두 벡터가 있습니다."
                )
              ) : (
                `벡터가 없는 ${n(row.todo)}장만 만듭니다.${row.done ? ` 다른 버전이나 프로젝트에서 이미 만든 같은 이미지 ${n(row.done)}장은 건너뜁니다.` : ""}`
              )}
            </span>
          )}
        </label>

        {serverError && <div className="error">{serverError}</div>}
        {servers?.problem && <div className="error small">{servers.problem}</div>}
        {servers && !servers.problem && (
          <PlaceGrid>
            <PlaceRow label="서버">
              <ServerSelect servers={servers.nodes} value={nodeId} onChange={setNodeId} />
              {!servers.nodes.some((x) => x.fits) && (
                <span className="bad-text small">
                  벡터를 만들 서버가 없습니다. <a href="#/nodes">노드</a> 화면에서 만들 서버의 설정 → 역할에 &quot;임베딩용&quot;을 켜세요.
                </span>
              )}
            </PlaceRow>
            {server && (
              <PlaceRow label="이미지">
                <ImagesNote server={server} holder={servers.holder?.name} items={servers.items} bytes={servers.bytes} />
              </PlaceRow>
            )}
            {server?.fits && server.gpu_list.length > 0 && (
              <PlaceRow label="GPU">
                <GpuChips gpus={server.gpu_list} pick={gpu} onPick={setGpu} doing="만듭니다" />
              </PlaceRow>
            )}
            {server?.fits && copied && (
              <PlaceRow label="보낸 데이터">
                <DataAfterChoice value={dataAfter} onChange={setDataAfter} when="임베딩이 끝나면" />
              </PlaceRow>
            )}
          </PlaceGrid>
        )}
        {error && <div className="error">{error}</div>}
        <div className="modal-actions">
          <button type="button" onClick={onClose}>
            취소
          </button>
          <button className="primary" disabled={saving || blocked}>
            임베딩 시작{row && row.todo > 0 ? ` · ${n(row.todo)}장` : ""}
          </button>
        </div>
      </form>
    </Modal>
  );
}

// ---- 임베딩 작업 기록 ---------------------------------------------------------------------------

/** A project's (or a version's) embedding jobs as records, as 검사 작업 기록: status, where, how many, the log. */
export function EmbedHistory({ projectId, releaseId, nudge, fold = false }: { projectId?: string; releaseId?: string; nudge?: unknown; fold?: boolean }) {
  const [busy, setBusy] = useState(false);
  const query = releaseId ? `release_id=${releaseId}` : `project_id=${projectId ?? ""}`;
  const { data: rows, refresh } = usePolling(() => api<EmbedJob[]>(`/embeddings/jobs?${query}`), busy ? 4000 : 20000, [query]);
  useEffect(() => setBusy((rows ?? []).some((r) => GOING.includes(r.status))), [rows]);
  useEffect(() => {
    void refresh(); // a job may have started
  }, [nudge]); // eslint-disable-line react-hooks/exhaustive-deps
  const [log, setLog] = useState<string | null>(null); // the job whose log is shown
  if (!rows || rows.length === 0) return null;
  const list = (
    <ul className="check-history-list embed-history-list">
      {rows.map((r) => (
        <li key={r.id}>
          <div className="embed-row">
            <StatusBadge status={r.status} />
            <div className="embed-row-main">
              <a href={`#/explore/job/${r.id}`}>
                <strong>{versionName(r)}</strong>
              </a>
              <div className="muted small">
                {[
                  r.images != null || r.made ? `벡터 ${madeText(r)}` : "",
                  r.missing ? `못 읽은 이미지 ${n(r.missing)}장` : "",
                  r.text ? "글 검색 인코더" : "",
                  r.node_name ? `${r.node_name} · ${r.gpus ? "GPU" : "CPU"}` : "",
                  r.duration_s != null && r.status !== "queued" ? duration(r.duration_s) : "",
                  VIA_LABEL[r.via] ?? r.via,
                  r.by,
                  timeAgo(r.created_at),
                ]
                  .filter(Boolean)
                  .join(" · ")}
              </div>
              {r.sent && (
                <div className={`small ${r.sent.status === "failed" ? "bad-text" : "muted"}`}>
                  이미지를 {r.sent.node_name || "그 서버"}에 먼저 보냄: {STATUS_LABEL[r.sent.status] ?? r.sent.status}
                  {r.sent.duration_s != null ? ` · ${duration(r.sent.duration_s)}` : ""}
                  {r.sent.error ? ` · ${r.sent.error}` : ""}
                </div>
              )}
              {r.status === "queued" && r.wait_reason && <div className="muted small">{r.wait_reason}</div>}
              {r.error && <div className="bad-text small">{r.error}</div>}
            </div>
            <div className="embed-row-actions">
              <button type="button" className="link small" onClick={() => setLog(log === r.id ? null : r.id)} aria-expanded={log === r.id}>
                {log === r.id ? "로그 닫기" : "로그"}
              </button>
              <a className="small" href={`#/explore/job/${r.id}`}>
                기록
              </a>
            </div>
          </div>
          {log === r.id && (
            <div className="check-history-log">
              <LogTail jobId={r.id} live={GOING.includes(r.status)} open />
            </div>
          )}
        </li>
      ))}
    </ul>
  );
  const going = rows.filter((r) => GOING.includes(r.status)).length;
  if (fold)
    return (
      <details className="card fold embed-history" open={going > 0}>
        <summary>
          임베딩 작업 기록 <span className="muted">{rows.length}</span>
          {going > 0 && <Badge kind="running">도는 중 {going}</Badge>}
        </summary>
        {list}
      </details>
    );
  return (
    <section className="card embed-history">
      <div className="card-head">
        <h3>
          임베딩 작업 기록 <span className="muted">{rows.length}</span>
        </h3>
        <span className="muted small">이 프로젝트의 버전마다 벡터를 만든 작업입니다. 기록을 누르면 설정 · 진행 · 로그를 봅니다.</span>
      </div>
      {list}
    </section>
  );
}

// ---- one job -----------------------------------------------------------------------------

function Fact({ label, children }: { label: string; children: React.ReactNode }) {
  return (
    <div>
      <div className="eyebrow">{label}</div>
      <div>{children}</div>
    </div>
  );
}

function EmbedJobPage({ jobId }: { jobId: string }) {
  const [going, setGoing] = useState(false);
  const { data: job, error, refresh } = usePolling(() => api<EmbedJob>(`/embeddings/jobs/${jobId}`), going ? 3000 : 20000, [jobId]);
  useEffect(() => setGoing(GOING.includes(job?.status ?? "")), [job?.status]);
  const [failed, setFailed] = useState<string | null>(null);
  const [again, setAgain] = useState(false);
  if (error && !job) return <div className="error">{error}</div>;
  if (!job) return <div className="muted">불러오는 중…</div>;
  const cancel = async () => {
    if (!window.confirm("이 임베딩을 중단할까요? 이미 보낸 벡터는 그대로 남습니다.")) return;
    try {
      setFailed(null);
      await api(`/jobs/${job.id}/cancel`, { method: "POST" });
      void refresh();
    } catch (e) {
      setFailed(e instanceof Error ? e.message : String(e));
    }
  };
  const cov = job.coverage;
  return (
    <>
      <nav className="page-crumbs">
        <a href="#/explore">이미지 탐색</a>
        {job.project && <a href={`#/explore/project/${job.project.id}`}>{job.project.name}</a>}
        <strong>임베딩 기록</strong>
      </nav>
      <div className="page-head">
        <div>
          <h1 className="with-badge">
            {versionName(job)} 임베딩
            <StatusBadge status={job.status} />
          </h1>
          <p className="muted">
            {job.model_title} · {job.by || "?"} · {timeAgo(job.created_at)}
            {job.duration_s != null && ` · ${duration(job.duration_s)}`}
          </p>
        </div>
        <div className="head-actions">
          {job.release && (
            <a className="btn" href={`#/datasets/${job.release.id}`}>
              <Icon name="database" size={14} /> 버전 열기
            </a>
          )}
          {job.project && (
            <a className="btn" href={`#/explore/project/${job.project.id}`}>
              <Icon name="eye" size={14} /> 찾기
            </a>
          )}
          {going && (
            <button className="danger" onClick={() => void cancel()}>
              중단
            </button>
          )}
          {!going && job.release && cov && cov.todo > 0 && (
            <button className="primary" onClick={() => setAgain(true)} title="이 버전에서 아직 벡터가 없는 이미지를 만듭니다">
              <Icon name="refresh" size={14} /> 남은 것 만들기
            </button>
          )}
        </div>
      </div>
      {failed && <div className="error">{failed}</div>}
      {job.status === "failed" && (
        <div className="notice bad">
          <strong>실패</strong> · {job.error || "이유를 남기지 않았습니다. 아래 로그를 보세요."}
        </div>
      )}
      {job.status === "queued" && job.wait_reason && <div className="notice">{job.wait_reason}</div>}
      {job.status === "succeeded" && (job.missing ?? 0) > 0 && (
        <div className="notice bad small">
          서버에서 찾거나 열 수 없는 이미지 {n(job.missing)}장은 건너뛰었습니다. 이미지가 그 서버에 없으면 데이터셋 버전 화면의 복사본에서 확인하세요.
        </div>
      )}
      {job.sent && (
        <div className={`notice small${job.sent.status === "failed" ? " bad" : ""}`}>
          이미지를 {job.sent.node_name || "그 서버"}에 먼저 보냄: {STATUS_LABEL[job.sent.status] ?? job.sent.status}
          {job.sent.duration_s != null ? ` · ${duration(job.sent.duration_s)}` : ""}
          {job.sent.error ? ` · ${job.sent.error}` : ""}
        </div>
      )}
      <div className="stats five">
        <Stat icon="images" label="만들 이미지" value={n(job.images)} sub="시작할 때 벡터가 없던 이미지" />
        <Stat icon="upload" label="보낸 벡터" value={n(job.made)} sub={going ? "Hub의 벡터 DB에 들어간 것" : "이 작업이 만든 것"} />
        <Stat
          icon="check"
          label="버전의 벡터"
          value={cov ? `${n(cov.done)}/${n(cov.total)}` : "–"}
          sub={cov ? (cov.todo ? `아직 ${n(cov.todo)}장 없음` : "모두 있음") : "지운 버전"}
        />
        <Stat icon="alert" label="읽지 못한 이미지" value={n(job.missing)} sub="서버에서 열리지 않은 이미지" />
        <Stat icon="clock" label="걸린 시간" value={duration(job.duration_s)} sub={job.finished_at ? `${timeAgo(job.finished_at)} 끝남` : " "} />
      </div>
      <div className="card al-facts">
        <div className="facts">
          <Fact label="데이터셋 버전">{job.release ? <a href={`#/datasets/${job.release.id}`}>{versionName(job)}</a> : "지운 버전"}</Fact>
          <Fact label="모델">{job.model_title}</Fact>
          <Fact label="서버">
            {job.node_name || "–"} · {job.gpus ? `GPU${job.gpu_indices.length ? ` ${job.gpu_indices.join(", ")}` : ""}` : "CPU"}
          </Fact>
          <Fact label="글 검색 인코더">{job.text ? "이 작업이 함께 만듦" : "이미 있음"}</Fact>
          {job.data_after && <Fact label="보낸 데이터">{job.data_after === "delete" ? "끝나면 지움" : "남겨 둠"}</Fact>}
          <Fact label="시작">
            {job.via === "training" && job.run_id ? <a href={`#/training/${job.run_id}`}>학습과 함께</a> : (VIA_LABEL[job.via] ?? job.via)}
          </Fact>
        </div>
      </div>
      <LogTail key={job.id} jobId={job.id} live={going} open={going || job.status === "failed"} />
      {again && job.project && job.release && (
        <NewEmbedding
          projectId={job.project.id}
          releaseId={job.release.id}
          onClose={() => setAgain(false)}
          onStarted={(id) => {
            setAgain(false);
            window.location.hash = `#/explore/job/${id}`;
          }}
        />
      )}
    </>
  );
}

// ---- a project: finding its images -------------------------------------------------------

/** 버전별 임베딩: each version's vectors; 임베딩 만들기 opens 새 임베딩 on that version. */
function VersionEmbeddings({ projectId, nudge, onStart }: { projectId: string; nudge: unknown; onStart: (releaseId: string) => void }) {
  const [rows, setRows] = useState<VersionRow[] | null>(null);
  const load = () => void api<VersionRow[]>(`/projects/${projectId}/explore/versions`).then(setRows).catch(() => setRows([]));
  useEffect(load, [projectId, nudge]); // eslint-disable-line react-hooks/exhaustive-deps
  const going = rows?.some((r) => r.going) ?? false;
  useEffect(() => {
    if (!going) return;
    const timer = window.setInterval(load, 10000);
    return () => window.clearInterval(timer);
  }, [going]); // eslint-disable-line react-hooks/exhaustive-deps
  if (!rows) return null;
  return (
    <details className="card explore-versions" open={rows.some((r) => r.done === 0) || going}>
      <summary>
        버전별 임베딩 <span className="muted small">— 벡터가 없는 이미지만 임베딩용 서버가 만듭니다</span>
      </summary>
      {rows.length === 0 && <p className="muted small">이 프로젝트에는 등록이 끝난 버전이 없습니다.</p>}
      <ul className="explore-version-list">
        {rows.map((r) => (
          <li key={r.release.id}>
            <div className="explore-version-row">
              <a href={`#/datasets/${r.release.id}`}>
                {r.release.dataset} {r.release.version}
              </a>
              <Meter done={r.done} total={r.total} />
              <span className="muted small nowrap">
                벡터 {n(r.done)}/{n(r.total)}장
              </span>
              {r.going ? (
                <a className="tag" href={`#/explore/job/${r.going}`}>
                  만드는 중
                </a>
              ) : r.todo === 0 ? (
                <span className="muted small">모두 있음</span>
              ) : (
                <button type="button" className="ghost-btn" onClick={() => onStart(r.release.id)}>
                  임베딩 만들기 ({n(r.todo)}장)
                </button>
              )}
            </div>
          </li>
        ))}
      </ul>
    </details>
  );
}

/** A project's images found by words or by another image, its versions' vectors, and its embedding jobs. */
function ExploreProject({ projectId }: { projectId: string }) {
  const { projects, choose } = useProject();
  useEffect(() => rememberProject(projectId), [projectId]);
  const project = projects?.find((p) => p.id === projectId);
  const [status, setStatus] = useState<ExploreStatus | null>(null);
  const [tree, setTree] = useState<TreeDataset[]>([]);
  const [release, setRelease] = useState(""); // "": every version of the project
  const [words, setWords] = useState("");
  const [query, setQuery] = useState<Query | null>(null);
  const [found, setFound] = useState<SearchResult | null>(null);
  const [minScore, setMinScore] = useState<Record<ScoreKind, number>>(() => ({ text: keptScore("text"), like: keptScore("like") }));
  const kind: ScoreKind = query && "like" in query ? "like" : "text";
  const lowest = minScore[kind];
  const sentLowest = useDebounced(lowest, 300); // asked again once the slider rests
  const setLowest = (value: number) => {
    setMinScore((current) => ({ ...current, [kind]: value }));
    try {
      localStorage.setItem(`${SCORE_KEY}.${kind}`, String(value));
    } catch {
      // kept for this visit only
    }
  };
  const [busy, setBusy] = useState(false);
  const [error, setError] = useState<string | null>(null);
  const [starting, setStarting] = useState<{ releaseId?: string } | null>(null);
  const [nudge, setNudge] = useState(0); // a job started: read the versions and the records again
  // a result opened as on the version screen: the version it is in, and the results of that version in their order
  const [viewing, setViewing] = useState<{ release: ReleaseInfo; ids: number[]; start: number } | null>(null);
  const [opening, setOpening] = useState<number | null>(null);
  const open = async (item: Found) => {
    if (!item.release || !found) return;
    const ids = found.items.filter((f) => f.release?.id === item.release?.id).map((f) => f.id);
    (document.activeElement as HTMLElement | null)?.blur(); // ← → go to the viewer, not the search box
    setOpening(item.id);
    try {
      const release = await api<ReleaseInfo>(`/releases/${item.release.id}`);
      setViewing({ release, ids, start: Math.max(0, ids.indexOf(item.id)) });
    } catch (e) {
      setError(e instanceof Error ? e.message : String(e));
    } finally {
      setOpening(null);
    }
  };

  const loadStatus = () => void api<ExploreStatus>(`/projects/${projectId}/explore`).then(setStatus).catch((e) => setError(String(e)));
  useEffect(() => {
    loadStatus();
    void api<TreeDataset[]>(`/projects/${projectId}/datasets`).then(setTree).catch(() => setTree([]));
  }, [projectId, nudge]); // eslint-disable-line react-hooks/exhaustive-deps

  useEffect(() => {
    if (!query) return;
    let live = true;
    setBusy(true);
    setError(null);
    const params = new URLSearchParams({ limit: String(sentLowest > 0 ? MANY : FEW) });
    if ("text" in query) params.set("text", query.text);
    else params.set("like", String(query.like.id));
    if (release) params.set("release_id", release);
    if (sentLowest > 0) params.set("min_score", String(sentLowest));
    api<SearchResult>(`/projects/${projectId}/explore/search?${params}`)
      .then((got) => live && setFound(got))
      .catch((e) => live && setError(e instanceof Error ? e.message : String(e)))
      .finally(() => live && setBusy(false));
    return () => {
      live = false;
    };
  }, [query, release, projectId, sentLowest]);

  const submit = (e: FormEvent) => {
    e.preventDefault();
    if (words.trim()) setQuery({ text: words.trim() });
  };
  const textReady = Boolean(status?.text.ready);
  const versions = tree.flatMap((d) =>
    d.releases.filter((r) => r.status === "ready" && !r.deleted_at).map((r) => ({ id: r.id, label: `${d.name} ${r.version}` })),
  );

  return (
    <>
      <nav className="page-crumbs">
        <a href="#/explore">이미지 탐색</a>
        {project?.folder && <span>{project.folder}</span>}
        <strong>{project?.name ?? "…"}</strong>
      </nav>
      <div className="page-head">
        <div>
          <h1>이미지 탐색</h1>
          <p className="muted">
            이 프로젝트의 모든 데이터셋 이미지를 글이나 비슷한 이미지로 찾습니다. 찾는 것은 벡터가 있는 이미지뿐이며, 벡터는 새 임베딩으로 만듭니다.
          </p>
        </div>
        <div className="head-actions">
          <ProjectPicker
            projects={projects}
            projectId={projectId}
            onChange={(id) => {
              choose(id);
              window.location.hash = `#/explore/project/${id}`;
            }}
          />
          <button className="primary" onClick={() => setStarting({})}>
            <Icon name="plus" />새 임베딩
          </button>
        </div>
      </div>

      {status && (
        <div className="card explore-status">
          <span>
            이미지 <strong>{n(status.images)}</strong>장 중 <strong>{n(status.embedded)}</strong>장을 찾을 수 있습니다
            <span className="muted"> · {status.title}</span>
          </span>
          <span className={`small ${textReady ? "muted" : "warn-text"}`}>
            {textReady
              ? "글 검색 준비됨 (Hub에서 검색어를 벡터로 바꿈)"
              : status.text.problem
                ? `글 검색을 쓸 수 없음: ${status.text.problem}`
                : "글 검색은 첫 임베딩 작업이 끝나면 쓸 수 있습니다"}
          </span>
        </div>
      )}
      {status && status.embedded === 0 && (
        <div className="card empty-state">
          <Icon name="images" size={26} />
          <strong>아직 찾을 수 있는 이미지가 없습니다</strong>
          <p className="muted">
            &quot;새 임베딩&quot;으로 데이터셋 버전의 벡터를 만들거나, 새 학습에서 &quot;학습하면서 이미지 임베딩도 만들기&quot;를 켜고 학습하세요(만드는
            서버에 노드 역할 &quot;임베딩용&quot;이 있어야 함).
          </p>
          <div>
            <button className="primary" onClick={() => setStarting({})}>
              <Icon name="plus" />새 임베딩
            </button>
          </div>
        </div>
      )}
      {status && <VersionEmbeddings projectId={projectId} nudge={nudge} onStart={(releaseId) => setStarting({ releaseId })} />}

      {status && status.embedded > 0 && (
        <>
          <form className="explore-bar" onSubmit={submit}>
            <input
              type="search"
              value={words}
              onChange={(e) => setWords(e.target.value)}
              placeholder={textReady ? "찾을 것을 글로: 밤에 찍힌 트럭, a person riding a bike" : "글 검색은 첫 임베딩 작업이 끝나면"}
              disabled={!textReady}
              aria-label="글로 찾기"
            />
            <button className="primary" disabled={!textReady || !words.trim() || busy}>
              <Icon name="eye" size={14} /> 찾기
            </button>
            <select value={release} onChange={(e) => setRelease(e.target.value)} aria-label="찾을 버전">
              <option value="">프로젝트의 모든 버전</option>
              {versions.map((v) => (
                <option key={v.id} value={v.id}>
                  {v.label}
                </option>
              ))}
            </select>
            <label
              className="explore-score"
              title={
                kind === "text"
                  ? "글 검색: 이 점수 이상인 이미지만 봅니다. 0.1~0.2만 되어도 잘 맞는 편입니다."
                  : "비슷한 이미지: 이 점수 이상인 이미지만 봅니다. 0.7 이상이면 꽤 닮은 편입니다."
              }
            >
              <span className="muted">{kind === "text" ? "글" : "이미지"} 점수 ≥</span>
              <input
                type="range"
                min={0}
                max={SCORE_RANGE[kind].max}
                step={SCORE_RANGE[kind].step}
                value={lowest}
                onChange={(e) => setLowest(Number(e.target.value))}
                aria-label="점수 임계값"
              />
              <span className="num">{lowest > 0 ? lowest.toFixed(SCORE_RANGE[kind].digits) : "끔"}</span>
            </label>
          </form>
          {query && "like" in query && (
            <div className="explore-anchor">
              {query.like.thumb_url && <img src={query.like.thumb_url} alt="" />}
              <span>
                <strong>이 이미지와 비슷한 것</strong>
                <span className="muted small"> · {query.like.file_name}</span>
              </span>
              <button type="button" className="link small" onClick={() => setQuery(words.trim() ? { text: words.trim() } : null)}>
                지우기
              </button>
            </div>
          )}
          {error && <div className="error">{error}</div>}
          {busy && !found && <p className="muted">찾는 중…</p>}
          {!query && !busy && <p className="muted">글로 찾거나, 결과에서 &quot;비슷한 이미지&quot;를 눌러 그 이미지와 비슷한 것을 찾습니다.</p>}
          {found && (
            <>
              <p className="muted small">
                {n(found.searched)}장에서 가까운 순으로 {n(found.items.length)}장
                {found.min_score > 0 ? ` · 점수 ${found.min_score.toFixed(SCORE_RANGE[kind].digits)} 이상` : ""}
                {found.more ? ` · 더 있습니다(가까운 ${n(found.items.length)}장까지 보임, 점수를 올리면 좁혀짐)` : ""}
                {busy ? " · 다시 찾는 중…" : ""}
                {kind === "text" && !found.min_score ? " · 글 검색의 점수는 0.1~0.2만 되어도 잘 맞는 편입니다" : ""}
              </p>
              {found.items.length === 0 && found.min_score > 0 && (
                <p className="muted">이 점수 이상인 이미지가 없습니다. 점수를 낮춰 보세요.</p>
              )}
              <div className="thumb-grid explore-grid">
                {found.items.map((item) => (
                  <div key={item.id} className="thumb explore-thumb">
                    <button
                      type="button"
                      className={`thumb-image explore-open${opening === item.id ? " opening" : ""}`}
                      onClick={() => void open(item)}
                      disabled={!item.release}
                      title={item.release ? "크게 보기 (라벨 · 버전별 라벨 · 이미지 정보)" : "버전에 없는 이미지"}
                    >
                      {item.thumb_url ? <img src={item.thumb_url} alt={item.file_name} loading="lazy" /> : <Icon name="images" />}
                    </button>
                    <div className="explore-meta">
                      <span className="tag">{item.score.toFixed(3)}</span>
                      {item.release ? (
                        <a className="small ellipsis" href={`#/datasets/${item.release.id}`} title="그 버전 열기">
                          {item.release.dataset} {item.release.version}
                        </a>
                      ) : (
                        <span className="muted small">버전 없음</span>
                      )}
                    </div>
                    <div className="thumb-name ellipsis" title={item.file_name}>
                      {item.file_name}
                    </div>
                    <button type="button" className="ghost-btn" onClick={() => setQuery({ like: item })}>
                      비슷한 이미지
                    </button>
                  </div>
                ))}
              </div>
            </>
          )}
        </>
      )}

      <EmbedHistory projectId={projectId} nudge={nudge} />

      {viewing && (
        <ItemViewer
          releaseId={viewing.release.id}
          filterQuery={`ids=${viewing.ids.join(",")}`}
          start={viewing.start}
          total={viewing.ids.length}
          classes={viewing.release.label_space.classes}
          colorOf={colorFor(viewing.release.label_space.classes)}
          task={viewing.release.task ?? "detect"}
          onClose={() => setViewing(null)}
        />
      )}

      {starting && (
        <NewEmbedding
          projectId={projectId}
          releaseId={starting.releaseId}
          onClose={() => setStarting(null)}
          onStarted={(jobId) => {
            setStarting(null);
            setNudge((v) => v + 1);
            window.location.hash = `#/explore/job/${jobId}`;
          }}
        />
      )}
    </>
  );
}
