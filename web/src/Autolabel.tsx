import { type FormEvent, type ReactNode, useEffect, useMemo, useState } from "react";
import { AliasChip } from "./Annotate";
import { type BoardProject, useBoardFilter } from "./Boards";
import { TaskTags, rememberProject } from "./DataBoard";
import { type Order, OrderReset, SortHandle, SortItem, Sortable, useOrder } from "./Order";
import { ProjectPicker, useProject } from "./ProjectPicker";
import { LogTail } from "./Training";
import { api } from "./api";
import type { AutolabelModel, AutolabelStatus } from "./catalogTypes";
import { Badge, Icon, Modal, STATUS_LABEL, Stat, duration, timeAgo, usePolling } from "./ui";
import "./autolabel.css";

// ---- what the Hub sends ------------------------------------------------------------------

interface AutolabelResult {
  images: number;
  labels: number;
  accepted: number;
  empty: number;
  skipped: number;
}

/** One 오토라벨 run: a model predicted a labeling set's images nobody labelled yet. */
interface AutolabelRun {
  id: string;
  job_id: string | null;
  status: string;
  error: string | null;
  wait_reason: string | null;
  duration_s: number | null;
  node_name: string | null;
  gpus: number;
  created_at: string;
  finished_at: string | null;
  started_by: string | null;
  via: string | null;
  project: { id: string; name: string } | null;
  model: { id: string; name: string; version: number; aliases: string[]; deleted: boolean } | null;
  set: { id: string; name: string; review: boolean } | null; // null once the set is deleted
  set_gone: boolean;
  current: boolean; // the set's newest run: its editor starts from these predictions
  dataset: { id: string; name: string } | null;
  release: { id: string; version: string; draft: boolean } | null;
  keep: number | null;
  accept: number | null;
  items: number | null;
  result: AutolabelResult | null;
}

interface RunDetail extends AutolabelRun {
  code: { name?: string; version?: string; trained_with?: string };
  started_at: string | null;
  log_count: number;
  gpu_indices: number[];
}

interface ProjectCard {
  project: BoardProject;
  runs: number;
  going: number;
  succeeded: number;
  failed: number;
  images: number;
  labels: number;
  accepted: number;
  recent: AutolabelRun[];
  last_at: string | null;
  sets: { id: string; name: string; dataset: string; todo: number; autolabeled: boolean }[];
  sets_waiting: number;
  todo: number;
  models: number; // the project's models that can label
}

type Totals = Record<"runs" | "going" | "succeeded" | "failed" | "images" | "labels" | "accepted" | "sets_waiting" | "todo", number>;

interface BoardInfo {
  going: AutolabelRun[];
  projects: ProjectCard[];
  totals: Totals;
}

interface TargetSet {
  id: string;
  name: string;
  dataset: { id: string; name: string } | null;
  review: boolean;
  todo: number;
  items: number;
  autolabel: AutolabelStatus | null;
}

interface TargetDataset {
  id: string;
  name: string;
  task: string;
  release: { id: string; version: string; draft: boolean };
  counts: Record<string, Record<string, number>>; // split → label state → images
}

interface Targets {
  sets: TargetSet[];
  datasets: TargetDataset[];
}

interface Server {
  id: string | null;
  name: string;
  status: string;
  gpus: number;
  problems: string[];
  notes: string[];
}

interface Offer {
  todo: number | null;
  keep: number;
  models: AutolabelModel[];
  server: Server;
}

const STATUS_KIND: Record<string, string> = {
  running: "running",
  queued: "queued",
  succeeded: "succeeded",
  failed: "failed",
  canceled: "idle",
};
const GOING = ["queued", "running"];
const SPLITS = ["train", "val", "test"];

const n = (value: number | null | undefined) => (value ?? 0).toLocaleString();
const modelName = (r: AutolabelRun) => (r.model ? `${r.model.name} v${r.model.version}` : "지운 모델");
const setName = (r: AutolabelRun) => r.set?.name ?? "지운 묶음";

function StatusBadge({ status }: { status: string }) {
  return <Badge kind={STATUS_KIND[status] ?? "idle"}>{STATUS_LABEL[status] ?? status}</Badge>;
}

/** Labels it drew, as one line: predicted images · labels · accepted · without any. */
function Outcome({ run }: { run: AutolabelRun }) {
  const r = run.result;
  if (run.status === "failed") return <span className="bad-text small ellipsis">{run.error || "실패"}</span>;
  if (GOING.includes(run.status)) {
    return <span className="muted small">{run.status === "queued" ? run.wait_reason ?? "차례를 기다리는 중" : `이미지 ${n(run.items)}장 예측 중`}</span>;
  }
  if (!r) return <span className="muted small">–</span>;
  return (
    <span className="small">
      이미지 <strong>{n(r.images)}</strong> · 라벨 <strong>{n(r.labels)}</strong>
      {run.accept != null && (
        <>
          {" "}
          · 자동 승인 <strong>{n(r.accepted)}</strong>
        </>
      )}
      {r.empty > 0 && <span className="muted"> · 예측 없음 {n(r.empty)}</span>}
    </span>
  );
}

// ---- the screen --------------------------------------------------------------------------

/** 오토라벨: projects at a glance, then a project's runs and sets, then one run. */
export function Autolabel({ sub, arg }: { sub?: string; arg?: string }) {
  if (sub === "project" && arg) return <ProjectRuns key={arg} projectId={arg} />;
  if (sub === "run" && arg) return <RunPage key={arg} opId={arg} />;
  return <AutolabelBoard />;
}

/** The two ways: models trained here (Closed-Voca) now; text prompts (Open-Voca) later, shown as a trace. */
function Ways({ onStart }: { onStart: () => void }) {
  return (
    <div className="al-ways">
      <div className="card al-way">
        <div className="al-way-head">
          <Icon name="box" />
          <strong>Closed-Voca</strong>
          <span className="tag accent">학습한 모델로</span>
        </div>
        <p className="muted small">
          이 Hub에서 학습해 등록한 모델이 그 모델의 클래스로 라벨을 먼저 그립니다. 최소 확신도 아래는 버리고, 확신도가 높은 이미지는 바로 승인할
          수 있습니다. 사람은 모델이 그린 것에서 시작해 고칩니다.
        </p>
        <div>
          <button className="primary small" onClick={onStart}>
            <Icon name="plus" size={13} /> 새 오토라벨
          </button>
        </div>
      </div>
      <div className="card al-way off" aria-disabled="true" title="아직 쓸 수 없습니다">
        <div className="al-way-head">
          <Icon name="wand" />
          <strong>Open-Voca</strong>
          <span className="tag">준비 중</span>
        </div>
        <p className="muted small">
          클래스 이름을 글로 적으면 학습 없이 찾아 그립니다: 열린 어휘 검출(Grounding DINO · OWLv2 · Florence-2) → SAM으로 윤곽 → VLM이 다시
          확인.
        </p>
        <div>
          <button className="small" disabled>
            준비 중
          </button>
        </div>
      </div>
    </div>
  );
}

const cardProject = (c: ProjectCard) => c.project;
const cardId = (c: ProjectCard) => c.project.id;

function AutolabelBoard() {
  const [going, setGoing] = useState(false);
  const { data: board, error, refresh } = usePolling(() => api<BoardInfo>("/autolabel/board"), going ? 4000 : 15000);
  useEffect(() => setGoing((board?.going.length ?? 0) > 0), [board]);
  const order = useOrder("autolabel.projects", board?.projects, cardId);
  const { shown, tools } = useBoardFilter(board ? order.ordered : null, cardProject);
  const [starting, setStarting] = useState(false);
  if (error && !board) return <div className="error">{error}</div>;
  const t = board?.totals;
  return (
    <>
      <div className="page-head">
        <div>
          <h1>오토라벨</h1>
          <p className="muted">
            모델이 라벨을 먼저 그리고 사람은 고치기만 합니다. 라벨링 묶음의 아직 안 한 이미지나 데이터셋의 아직 안 한 이미지에 돌리고, 돌린 기록과
            결과를 프로젝트별로 봅니다. 프로젝트 카드는 끌어서 순서를 바꿀 수 있습니다.
          </p>
        </div>
        <div className="head-actions">
          <OrderReset order={order as Order<unknown>} />
          <button className="primary" onClick={() => setStarting(true)}>
            <Icon name="plus" />새 오토라벨
          </button>
        </div>
      </div>
      <div className="stats five">
        <Stat icon="bot" label="오토라벨" value={t?.runs ?? "–"} sub={t?.going ? `도는 중 ${t.going}` : "도는 것 없음"} />
        <Stat icon="images" label="예측한 이미지" value={n(t?.images)} sub={`라벨 ${n(t?.labels)}개`} />
        <Stat icon="check" label="자동 승인" value={n(t?.accepted)} sub="확신도가 높아 바로 저장한 이미지" />
        <Stat icon="edit" label="라벨링 기다림" value={n(t?.todo)} sub={`묶음 ${t?.sets_waiting ?? 0}개의 아직 안 한 이미지`} />
        <Stat icon="x" label="실패" value={t?.failed ?? "–"} sub="기록에서 로그를 봅니다" />
      </div>

      <section className="board-section">
        <h2>방식</h2>
        <Ways onStart={() => setStarting(true)} />
      </section>

      {board && board.going.length > 0 && (
        <section className="board-section">
          <h2>
            지금 도는 오토라벨 <span className="muted">{board.going.length}</span>
          </h2>
          <div className="going-list">
            {board.going.map((r) => (
              <a key={r.id} className="card going-run" href={`#/autolabel/run/${r.id}`}>
                <div className="going-title">
                  <StatusBadge status={r.status} />
                  <strong>{modelName(r)}</strong>
                  <span className="muted small">
                    → {setName(r)} · {r.project?.name}
                    {r.dataset ? ` · ${r.dataset.name}` : ""}
                  </span>
                </div>
                <div className="going-meta small">
                  <Outcome run={r} />
                  <span className="muted">
                    {r.node_name ?? "서버 ?"} · {r.gpus ? "GPU" : "CPU"}
                    {r.duration_s != null ? ` · ${duration(r.duration_s)} 지남` : ""}
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
          <div className="card empty">오토라벨할 데이터셋이 있는 프로젝트가 없습니다. 먼저 데이터셋을 가져오고 모델을 학습하세요.</div>
        )}
        <Sortable ids={shown.map(cardId)} onMove={order.move}>
          <div className="card-grid board-grid">
            {shown.map((c) => (
              <SortItem key={c.project.id} id={c.project.id}>
                <ProjectBoardCard card={c} />
              </SortItem>
            ))}
          </div>
        </Sortable>
      </section>
      {starting && (
        <StartAutolabel
          onClose={() => setStarting(false)}
          onStarted={(started) => {
            setStarting(false);
            void refresh();
            window.location.hash = `#/autolabel/run/${started.id}`;
          }}
        />
      )}
    </>
  );
}

function ProjectBoardCard({ card: c }: { card: ProjectCard }) {
  return (
    <div className={`data-card${c.runs === 0 && c.models === 0 ? " board-quiet" : ""}`}>
      <a className="data-card-link" href={`#/autolabel/project/${c.project.id}`} onClick={() => rememberProject(c.project.id)}>
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
          {c.runs === 0 ? (
            <p className="muted small">아직 오토라벨을 돌리지 않았습니다.</p>
          ) : (
            <>
              <div className="small">
                오토라벨 {c.runs} · 성공 {c.succeeded}
                {c.failed > 0 && <span className="bad-text"> · 실패 {c.failed}</span>}
              </div>
              <div className="al-sums">
                <span>
                  <strong className="num">{n(c.images)}</strong> 예측한 이미지
                </span>
                <span>
                  <strong className="num">{n(c.labels)}</strong> 라벨
                </span>
                <span>
                  <strong className="num">{n(c.accepted)}</strong> 자동 승인
                </span>
              </div>
            </>
          )}
          <div className={`small ${c.models ? "muted" : "bad-text"}`}>
            {c.models ? `오토라벨할 수 있는 모델 ${c.models}개` : "오토라벨할 수 있는 모델이 없습니다 — 학습해 등록하세요"}
          </div>
        </div>
      </a>
      {(c.recent.length > 0 || c.sets.length > 0) && (
        <div className="board-lines">
          {c.recent.map((r) => (
            <a key={r.id} href={`#/autolabel/run/${r.id}`} className="board-line">
              <StatusBadge status={r.status} />
              <span className="ellipsis">
                {modelName(r)} <span className="muted">→ {setName(r)}</span>
              </span>
              <span className="num muted" title="예측한 이미지 · 라벨">
                {r.result ? `${n(r.result.images)} · ${n(r.result.labels)}` : ""}
              </span>
            </a>
          ))}
          {c.sets.length > 0 && <div className="eyebrow al-lines-head">라벨링 기다리는 묶음</div>}
          {c.sets.map((s) => (
            <a key={s.id} href={`#/labeling/${s.id}`} className="board-line">
              <Icon name="edit" size={12} />
              <span className="ellipsis">
                {s.name} <span className="muted">{s.dataset}</span>
              </span>
              <span className="num" title={s.autolabeled ? "오토라벨을 돌린 묶음의 아직 안 한 이미지" : "아직 안 한 이미지"}>
                {n(s.todo)}장{s.autolabeled ? "" : " · 오토라벨 전"}
              </span>
            </a>
          ))}
          {c.sets_waiting > c.sets.length && <div className="muted small board-foot">묶음 {c.sets_waiting - c.sets.length}개 더</div>}
          {c.last_at && <div className="muted small board-foot">마지막 오토라벨 {timeAgo(c.last_at)}</div>}
        </div>
      )}
    </div>
  );
}

// ---- a project's runs and sets -----------------------------------------------------------

function ProjectRuns({ projectId }: { projectId: string }) {
  const { projects, choose } = useProject();
  useEffect(() => rememberProject(projectId), [projectId]);
  const [going, setGoing] = useState(false);
  const { data: runs, error, refresh } = usePolling(
    () => api<AutolabelRun[]>(`/autolabel/runs?project_id=${projectId}`),
    going ? 4000 : 15000,
    [projectId],
  );
  useEffect(() => setGoing(!!runs?.some((r) => GOING.includes(r.status))), [runs]);
  const { data: targets, refresh: retarget } = usePolling(
    () => api<Targets>(`/autolabel/targets?project_id=${projectId}`),
    30000,
    [projectId, runs?.length, going],
  );
  const [starting, setStarting] = useState<{ setId?: string } | null>(null);
  const project = projects?.find((p) => p.id === projectId);
  const done = (runs ?? []).filter((r) => r.status === "succeeded" && r.result);
  const sum = (key: keyof AutolabelResult) => done.reduce((total, r) => total + (r.result?.[key] ?? 0), 0);
  const waiting = (targets?.sets ?? []).filter((s) => s.todo > 0);
  const going_ = (runs ?? []).filter((r) => GOING.includes(r.status)).length;
  return (
    <>
      <nav className="page-crumbs">
        <a href="#/autolabel">오토라벨</a>
        {project?.folder && <span>{project.folder}</span>}
        <strong>{project?.name ?? "…"}</strong>
      </nav>
      <div className="page-head">
        <div>
          <h1>오토라벨</h1>
          <p className="muted">
            이 프로젝트에서 돌린 오토라벨과 라벨링을 기다리는 묶음입니다. 기록을 누르면 설정 · 결과 · 로그를 봅니다. 묶음의 이미지는 라벨링에서
            열면 모델이 그린 것에서 시작합니다.
          </p>
        </div>
        <div className="head-actions">
          <ProjectPicker
            projects={projects}
            projectId={projectId}
            onChange={(id) => {
              choose(id);
              window.location.hash = `#/autolabel/project/${id}`;
            }}
          />
          <button className="primary" onClick={() => setStarting({})}>
            <Icon name="plus" />새 오토라벨
          </button>
        </div>
      </div>
      <div className="stats five">
        <Stat icon="bot" label="오토라벨" value={runs?.length ?? "–"} sub={going_ ? `도는 중 ${going_}` : "도는 것 없음"} />
        <Stat icon="images" label="예측한 이미지" value={n(sum("images"))} sub={`라벨 ${n(sum("labels"))}개`} />
        <Stat icon="check" label="자동 승인" value={n(sum("accepted"))} sub="바로 저장한 이미지" />
        <Stat icon="alert" label="예측 없음" value={n(sum("empty"))} sub="모델이 아무것도 찾지 못한 이미지" />
        <Stat icon="edit" label="라벨링 기다림" value={n(waiting.reduce((t, s) => t + s.todo, 0))} sub={`묶음 ${waiting.length}개`} />
      </div>
      {error && !runs && <div className="error">{error}</div>}

      <section className="board-section">
        <h2>
          라벨링 묶음 <span className="muted small">아직 안 한 이미지가 있는 묶음에 돌립니다</span>
        </h2>
        {targets && targets.sets.length === 0 && (
          <div className="card empty">
            아직 라벨링 묶음이 없습니다. "새 오토라벨"에서 데이터셋의 아직 안 한 이미지를 고르면 묶음을 만들어 돌립니다.
          </div>
        )}
        {targets && targets.sets.length > 0 && (
          <div className="card flush table-wrap">
            <table className="al-table">
              <thead>
                <tr>
                  <th>묶음</th>
                  <th>데이터셋</th>
                  <th className="right">아직 안 함</th>
                  <th>마지막 오토라벨</th>
                  <th />
                </tr>
              </thead>
              <tbody>
                {targets.sets.map((s) => (
                  <tr key={s.id}>
                    <td>
                      <a href={`#/labeling/${s.id}`}>
                        <strong>{s.name}</strong>
                      </a>
                      <span className="muted small"> · {s.review ? "검수함" : "검수 없음"}</span>
                    </td>
                    <td>{s.dataset?.name ?? "–"}</td>
                    <td className="right num">
                      {n(s.todo)} <span className="muted small">/ {n(s.items)}</span>
                    </td>
                    <td>
                      {s.autolabel ? (
                        <span className="al-last">
                          <StatusBadge status={s.autolabel.status} />
                          <span className="small">{s.autolabel.model ? `${s.autolabel.model.name} v${s.autolabel.model.version}` : "모델"}</span>
                          <span className="muted small">{timeAgo(s.autolabel.started_at)}</span>
                        </span>
                      ) : (
                        <span className="muted small">아직 없음</span>
                      )}
                    </td>
                    <td className="right">
                      <button
                        className="small"
                        disabled={s.todo === 0 || GOING.includes(s.autolabel?.status ?? "")}
                        title={s.todo === 0 ? "아직 안 한 이미지가 없습니다" : undefined}
                        onClick={() => setStarting({ setId: s.id })}
                      >
                        <Icon name="bot" size={12} /> 오토라벨
                      </button>
                    </td>
                  </tr>
                ))}
              </tbody>
            </table>
          </div>
        )}
      </section>

      <section className="board-section">
        <h2>
          기록 <span className="muted">{runs?.length ?? ""}</span>
        </h2>
        <RunsTable runs={runs ?? null} />
      </section>
      {starting && (
        <StartAutolabel
          projectId={projectId}
          setId={starting.setId}
          pickTarget
          onClose={() => setStarting(null)}
          onStarted={(started) => {
            setStarting(null);
            void refresh();
            void retarget();
            window.location.hash = `#/autolabel/run/${started.id}`;
          }}
        />
      )}
    </>
  );
}

function RunsTable({ runs }: { runs: AutolabelRun[] | null }) {
  return (
    <div className="card flush table-wrap">
      <table className="al-table al-runs">
        <thead>
          <tr>
            <th>상태</th>
            <th>모델</th>
            <th>대상</th>
            <th>결과</th>
            <th>확신도</th>
            <th>서버</th>
            <th>시작</th>
          </tr>
        </thead>
        <tbody>
          {!runs && (
            <tr>
              <td colSpan={7} className="empty">
                불러오는 중…
              </td>
            </tr>
          )}
          {runs && runs.length === 0 && (
            <tr>
              <td colSpan={7} className="empty">
                아직 돌린 오토라벨이 없습니다.
              </td>
            </tr>
          )}
          {runs?.map((r) => (
            <tr key={r.id} onClick={() => (window.location.hash = `#/autolabel/run/${r.id}`)}>
              <td>
                <StatusBadge status={r.status} />
              </td>
              <td>
                <a href={`#/autolabel/run/${r.id}`} onClick={(e) => e.stopPropagation()}>
                  <strong>{modelName(r)}</strong>
                </a>
                {r.model?.aliases.map((a) => (
                  <AliasChip key={a} alias={a} />
                ))}
              </td>
              <td>
                {r.set ? (
                  <a href={`#/labeling/${r.set.id}`} onClick={(e) => e.stopPropagation()}>
                    {r.set.name}
                  </a>
                ) : (
                  <span className="muted">지운 묶음</span>
                )}
                {r.current && <span className="tag al-current" title="이 묶음의 라벨링은 이 기록의 예측에서 시작합니다">지금 씀</span>}
                <div className="muted small">
                  {r.dataset?.name ?? ""} {r.release?.version ?? ""}
                </div>
              </td>
              <td className="al-outcome">
                <Outcome run={r} />
              </td>
              <td className="small">
                {r.keep ?? "–"}
                {r.accept != null ? <span className="muted"> · 승인 {r.accept}</span> : ""}
              </td>
              <td className="small">
                {r.node_name ?? "–"} <span className="muted">{r.gpus ? "GPU" : "CPU"}</span>
              </td>
              <td className="small">
                {r.started_by ?? "?"} <span className="muted">· {timeAgo(r.created_at)}</span>
                {r.duration_s != null && <div className="muted">{duration(r.duration_s)}</div>}
              </td>
            </tr>
          ))}
        </tbody>
      </table>
    </div>
  );
}

// ---- one run -----------------------------------------------------------------------------

function RunPage({ opId }: { opId: string }) {
  const [going, setGoing] = useState(false);
  const { data: run, error, refresh } = usePolling(() => api<RunDetail>(`/autolabel/runs/${opId}`), going ? 3000 : 20000, [opId]);
  useEffect(() => setGoing(GOING.includes(run?.status ?? "")), [run?.status]);
  const [again, setAgain] = useState(false);
  const [failed, setFailed] = useState<string | null>(null);
  if (error && !run) return <div className="error">{error}</div>;
  if (!run) return <div className="muted">불러오는 중…</div>;
  const r = run.result;
  const cancel = async () => {
    if (!run.job_id || !window.confirm("이 오토라벨을 중단할까요? 이미 저장된 라벨은 그대로 남습니다.")) return;
    try {
      setFailed(null);
      await api(`/jobs/${run.job_id}/cancel`, { method: "POST" });
      void refresh();
    } catch (e) {
      setFailed(e instanceof Error ? e.message : String(e));
    }
  };
  return (
    <>
      <nav className="page-crumbs">
        <a href="#/autolabel">오토라벨</a>
        {run.project && <a href={`#/autolabel/project/${run.project.id}`}>{run.project.name}</a>}
        <strong>기록</strong>
      </nav>
      <div className="page-head">
        <div>
          <h1 className="with-badge">
            {modelName(run)} → {setName(run)}
            <StatusBadge status={run.status} />
          </h1>
          <p className="muted">
            {run.dataset && run.release ? (
              <>
                {run.dataset.name} <a href={`#/datasets/${run.release.id}`}>{run.release.version}</a>의 아직 안 한 이미지 {n(run.items)}장 ·{" "}
              </>
            ) : null}
            {run.started_by ?? "?"} · {timeAgo(run.created_at)}
            {run.duration_s != null && ` · ${duration(run.duration_s)}`}
          </p>
        </div>
        <div className="head-actions">
          {run.set && (
            <a className="btn" href={`#/labeling/${run.set.id}`}>
              <Icon name="edit" size={14} /> 묶음 열기
            </a>
          )}
          {run.model && !run.model.deleted && (
            <a className="btn" href={`#/models/${run.model.id}`}>
              <Icon name="box" size={14} /> 모델
            </a>
          )}
          {going && run.job_id && (
            <button className="danger" onClick={() => void cancel()}>
              중단
            </button>
          )}
          {!going && run.set && run.model && !run.model.deleted && (
            <button className="primary" onClick={() => setAgain(true)} title="같은 모델과 확신도로 이 묶음의 아직 안 한 이미지에 다시 돌립니다">
              <Icon name="refresh" size={14} /> 다시 돌리기
            </button>
          )}
        </div>
      </div>
      {failed && <div className="error">{failed}</div>}
      {run.status === "failed" && (
        <div className="notice bad">
          <strong>실패</strong> · {run.error || "이유를 남기지 않았습니다. 아래 로그를 보세요."}
        </div>
      )}
      {run.status === "queued" && run.wait_reason && <div className="notice">{run.wait_reason}</div>}
      <div className="stats five">
        <Stat icon="images" label="예측한 이미지" value={r ? n(r.images) : "–"} sub={`맡긴 이미지 ${n(run.items)}장`} />
        <Stat icon="box" label="라벨" value={r ? n(r.labels) : "–"} sub={`확신도 ${run.keep ?? "–"} 이상만 남김`} />
        <Stat
          icon="check"
          label="자동 승인"
          value={run.accept == null ? "안 함" : r ? n(r.accepted) : "–"}
          sub={run.accept == null ? "사람이 모두 봅니다" : `모든 예측이 ${run.accept} 이상인 이미지`}
        />
        <Stat icon="alert" label="예측 없음" value={r ? n(r.empty) : "–"} sub="사람이 배경인지 봅니다" />
        <Stat icon="clock" label="걸린 시간" value={duration(run.duration_s)} sub={run.finished_at ? `${timeAgo(run.finished_at)} 끝남` : " "} />
      </div>
      {run.status === "succeeded" && run.set && (
        <div className="notice small">
          {run.current
            ? `"${run.set.name}"에서 아직 안 한 이미지를 열면 이 기록의 예측에서 시작합니다${run.accept != null ? ` · 자동 승인한 이미지는 ${run.set.review ? "검수" : "완료"}로 갔습니다` : ""}.`
            : `"${run.set.name}"에는 이후 다른 오토라벨이 돌았습니다. 라벨링은 가장 최근 것의 예측에서 시작합니다.`}
        </div>
      )}
      <div className="card al-facts">
        <div className="facts">
          <Fact label="모델">
            {run.model ? (
              <>
                {modelName(run)}{" "}
                {run.model.aliases.map((a) => (
                  <AliasChip key={a} alias={a} />
                ))}
              </>
            ) : (
              "지운 모델"
            )}
          </Fact>
          <Fact label="라벨링 묶음">{run.set ? <a href={`#/labeling/${run.set.id}`}>{run.set.name}</a> : "지운 묶음"}</Fact>
          <Fact label="최소 확신도">{run.keep ?? "–"}</Fact>
          <Fact label="자동 승인 확신도">{run.accept ?? "안 함"}</Fact>
          <Fact label="서버">
            {run.node_name ?? "–"} · {run.gpus ? `GPU${run.gpu_indices.length ? ` ${run.gpu_indices.join(", ")}` : ""}` : "CPU"}
          </Fact>
          <Fact label="코드">
            {run.code.name ? `${run.code.name} ${run.code.version ?? ""}` : "–"}
            {run.code.trained_with && <span className="muted small"> (학습은 {run.code.trained_with})</span>}
          </Fact>
          <Fact label="시작">{run.via === "mcp" ? "에이전트" : run.via === "api" ? "API" : "웹"}</Fact>
        </div>
      </div>
      {run.job_id && <LogTail key={run.job_id} jobId={run.job_id} live={going} open={going || run.status === "failed"} />}
      {again && run.set && run.model && (
        <StartAutolabel
          projectId={run.project?.id}
          setId={run.set.id}
          modelId={run.model.id}
          keep={run.keep ?? undefined}
          accept={run.accept}
          review={run.set.review}
          onClose={() => setAgain(false)}
          onStarted={(started) => {
            setAgain(false);
            window.location.hash = `#/autolabel/run/${started.id}`;
          }}
        />
      )}
    </>
  );
}

function Fact({ label, children }: { label: string; children: ReactNode }) {
  return (
    <div>
      <div className="eyebrow">{label}</div>
      <div>{children}</div>
    </div>
  );
}

// ---- starting one ------------------------------------------------------------------------

export interface Started {
  id: string;
  job_id: string | null;
  set_id: string;
}

const unlabelled = (d: TargetDataset | undefined, split: string) => {
  if (!d) return 0;
  const counts = split ? [d.counts[split] ?? {}] : Object.values(d.counts);
  return counts.reduce((total, c) => total + (c.unlabelled ?? 0), 0);
};

/**
 * Start a 오토라벨: what it runs over (a labeling set's images not labelled yet, or a dataset's, made into a set
 * first), the model (the fitting ones first, an alias first), the server holding the images and how sure it must be.
 * ``setId`` alone (from a set's screen): that set, no choosing. ``pickTarget`` with it: that set first, others too.
 */
export function StartAutolabel({
  projectId: fixedProject,
  setId,
  pickTarget = false,
  modelId: wantedModel,
  keep: startKeep,
  accept: startAccept,
  review: setReviews,
  onClose,
  onStarted,
}: {
  projectId?: string;
  setId?: string;
  pickTarget?: boolean;
  review?: boolean; // with ``setId`` alone: whether that set is reviewed (where accepted images go)
  modelId?: string;
  keep?: number;
  accept?: number | null;
  onClose: () => void;
  onStarted: (started: Started) => void;
}) {
  const fixedSet = Boolean(setId) && !pickTarget;
  const { projects, projectId: remembered, choose } = useProject();
  const [picked, setPicked] = useState(fixedProject ?? "");
  const projectId = fixedProject ?? (picked || remembered);
  const { data: targets } = usePolling(
    () => (projectId && !fixedSet ? api<Targets>(`/autolabel/targets?project_id=${projectId}`) : Promise.resolve(null)),
    0,
    [projectId, fixedSet],
  );
  const [mode, setMode] = useState<"set" | "dataset">("set");
  const [chosenSet, setChosenSet] = useState(setId ?? "");
  const [datasetId, setDatasetId] = useState("");
  const [split, setSplit] = useState("");
  const [name, setName] = useState("");
  const [partSize, setPartSize] = useState(50);
  const [review, setReview] = useState(true);
  // Where to begin: the set given, else a set with images to label, else a dataset with some.
  useEffect(() => {
    if (!targets || fixedSet) return;
    const open = targets.sets.filter((s) => s.todo > 0);
    const raw = targets.datasets.filter((d) => unlabelled(d, "") > 0);
    if (setId && targets.sets.some((s) => s.id === setId)) {
      setMode("set");
      setChosenSet(setId);
    } else if (open.length) {
      setMode("set");
      setChosenSet(open[0].id);
    } else {
      setMode(raw.length || !targets.sets.length ? "dataset" : "set");
      setChosenSet("");
    }
    setDatasetId((raw[0] ?? targets.datasets[0])?.id ?? "");
    setSplit("");
  }, [targets, fixedSet, setId]);
  const dataset = targets?.datasets.find((d) => d.id === datasetId);
  useEffect(() => {
    const today = new Date();
    const day = `${String(today.getMonth() + 1).padStart(2, "0")}/${String(today.getDate()).padStart(2, "0")}`;
    if (dataset) setName(`${dataset.name} 오토라벨 ${day}`);
  }, [dataset?.id]); // eslint-disable-line react-hooks/exhaustive-deps

  const query = fixedSet || mode === "set" ? (chosenSet ? `set_id=${chosenSet}` : "") : dataset ? `release_id=${dataset.release.id}` : "";
  const { data: offer, error: offerError } = usePolling(
    () => (query ? api<Offer>(`/autolabel/models?${query}`) : Promise.resolve(null)),
    0,
    [query],
  );
  const [modelId, setModelId] = useState(wantedModel ?? "");
  const usable = useMemo(() => (offer?.models ?? []).filter((m) => !m.problem), [offer]);
  useEffect(() => {
    if (!offer) return;
    if (!usable.some((m) => m.id === modelId)) setModelId(usable.find((m) => m.id === wantedModel)?.id ?? usable[0]?.id ?? "");
  }, [offer]); // eslint-disable-line react-hooks/exhaustive-deps
  const [keep, setKeep] = useState(startKeep ?? 0.25);
  const [accepting, setAccepting] = useState(startAccept != null);
  const [accept, setAccept] = useState(startAccept ?? 0.8);
  const [cpu, setCpu] = useState(false);
  const [saving, setSaving] = useState(false);
  const [error, setError] = useState<string | null>(null);

  const server = offer?.server;
  const reviewOf = fixedSet
    ? (setReviews ?? true)
    : mode === "set"
      ? (targets?.sets.find((s) => s.id === chosenSet)?.review ?? true)
      : review;
  const images = fixedSet || mode === "set" ? (offer?.todo ?? targets?.sets.find((s) => s.id === chosenSet)?.todo ?? 0) : unlabelled(dataset, split);
  const blocked = !query || !modelId || images === 0 || (server?.problems.length ?? 0) > 0 || (accepting && accept < keep);

  const submit = async (e: FormEvent) => {
    e.preventDefault();
    setSaving(true);
    setError(null);
    const common = { model_id: modelId, keep, accept: accepting ? accept : null, gpus: cpu ? 0 : null };
    const body =
      fixedSet || mode === "set"
        ? { ...common, set_id: chosenSet }
        : { ...common, release_id: dataset?.release.id, split, name, part_size: partSize, review };
    try {
      onStarted(await api<Started>("/autolabel", { body }));
    } catch (err) {
      setError(err instanceof Error ? err.message : String(err));
    } finally {
      setSaving(false);
    }
  };

  return (
    <Modal title="새 오토라벨" onClose={onClose}>
      <form onSubmit={submit} className="al-start">
        <div className="segmented al-way-pick" role="radiogroup" aria-label="방식">
          <span className="eyebrow">방식</span>
          <button type="button" className="active" aria-pressed="true">
            Closed-Voca · 학습한 모델
          </button>
          <button type="button" disabled title="준비 중: 클래스 이름을 글로 적어 학습 없이 찾게 합니다 (OVD → SAM → VLM 재확인)">
            Open-Voca · 텍스트 프롬프트 <span className="tag">준비 중</span>
          </button>
        </div>

        {!fixedSet && !fixedProject && (
          <ProjectPicker
            projects={projects}
            projectId={projectId}
            onChange={(id) => {
              setPicked(id);
              choose(id);
            }}
          />
        )}

        {!fixedSet && (
          <fieldset className="al-target">
            <legend>대상</legend>
            <div className="segmented">
              <button type="button" className={mode === "set" ? "active" : ""} onClick={() => setMode("set")}>
                라벨링 묶음
              </button>
              <button type="button" className={mode === "dataset" ? "active" : ""} onClick={() => setMode("dataset")}>
                데이터셋의 아직 안 한 이미지
              </button>
            </div>
            {mode === "set" ? (
              targets && targets.sets.length === 0 ? (
                <p className="muted small">이 프로젝트에는 라벨링 묶음이 없습니다. "데이터셋의 아직 안 한 이미지"로 묶음을 만들어 돌리세요.</p>
              ) : (
                <label>
                  묶음
                  <select value={chosenSet} onChange={(e) => setChosenSet(e.target.value)} aria-label="라벨링 묶음">
                    <option value="">고르세요</option>
                    {targets?.sets.map((s) => (
                      <option key={s.id} value={s.id} disabled={s.todo === 0}>
                        {s.name} · {s.dataset?.name ?? ""} · 아직 안 함 {n(s.todo)}/{n(s.items)}
                        {s.autolabel ? ` · 오토라벨 ${STATUS_LABEL[s.autolabel.status] ?? s.autolabel.status}` : ""}
                      </option>
                    ))}
                  </select>
                  <span className="muted small">묶음의 아직 안 한 이미지에만 돌립니다. 사람이 라벨링했거나 돌려받은 이미지는 건드리지 않습니다.</span>
                </label>
              )
            ) : targets && targets.datasets.length === 0 ? (
              <p className="muted small">이 프로젝트에는 등록이 끝난 데이터셋 버전이 없습니다.</p>
            ) : (
              <>
                <div className="form-row">
                  <label>
                    데이터셋
                    <select value={datasetId} onChange={(e) => setDatasetId(e.target.value)} aria-label="데이터셋">
                      {targets?.datasets.map((d) => (
                        <option key={d.id} value={d.id}>
                          {d.name} {d.release.version} · 아직 안 함 {n(unlabelled(d, ""))}
                        </option>
                      ))}
                    </select>
                  </label>
                  <label>
                    새 묶음 이름
                    <input value={name} onChange={(e) => setName(e.target.value)} required />
                  </label>
                </div>
                <div className="segmented" role="radiogroup" aria-label="split">
                  <span className="eyebrow">split</span>
                  {["", ...SPLITS.filter((s) => dataset?.counts[s])].map((s) => (
                    <button type="button" key={s || "all"} className={split === s ? "active" : ""} onClick={() => setSplit(s)}>
                      {s || "전체"} <span className="muted">{n(unlabelled(dataset, s))}</span>
                    </button>
                  ))}
                </div>
                <div className="form-row">
                  <label>
                    일감 하나에 이미지
                    <input type="number" min={1} max={10000} value={partSize} onChange={(e) => setPartSize(Number(e.target.value) || 1)} />
                  </label>
                  <label className="check" style={{ alignSelf: "end" }}>
                    <input type="checkbox" checked={review} onChange={(e) => setReview(e.target.checked)} /> 검수하기 (승인 · 돌려보내기)
                  </label>
                </div>
                <p className="muted small">
                  {dataset?.release.draft ? "작업 중" : dataset?.release.version}의 라벨 상태 "아직 안 함" 이미지 {n(images)}장으로 라벨링 묶음을 만든 뒤
                  오토라벨을 돌립니다. 라벨은 데이터셋의 작업 중에 모입니다.
                </p>
              </>
            )}
          </fieldset>
        )}
        {fixedSet && (
          <p className="muted">
            등록한 모델이 이 묶음의 아직 안 한 이미지 {n(offer?.todo)}장에 라벨을 먼저 그립니다. 라벨링하는 사람은 이미지를 열면 모델이 그린 것에서
            시작해 고치기만 합니다. 이미 라벨링했거나 돌려받은 이미지는 건드리지 않습니다.
          </p>
        )}

        <div className="eyebrow al-label">모델</div>
        {offerError && <div className="error">{offerError}</div>}
        {!query && <p className="muted small">대상을 고르면 쓸 수 있는 모델이 나옵니다.</p>}
        {offer && offer.models.length === 0 && <div className="notice">이 프로젝트에 등록된 모델이 없습니다. 먼저 학습해 모델을 만드세요.</div>}
        {offer && offer.models.length > 0 && (
          <div className="autolabel-models" role="radiogroup" aria-label="모델">
            {offer.models.map((m) => (
              <label key={m.id} className={`autolabel-model ${m.problem ? "off" : ""}`} title={m.problem || undefined}>
                <input type="radio" name="model" value={m.id} checked={modelId === m.id} disabled={Boolean(m.problem)} onChange={() => setModelId(m.id)} />
                <span>
                  <strong>
                    {m.name} v{m.version}
                  </strong>{" "}
                  {m.aliases.map((a) => (
                    <AliasChip key={a} alias={a} />
                  ))}
                  <span className="muted small">
                    {" "}
                    · {m.metric} {m.value != null ? m.value.toFixed(3) : "–"} · 클래스 {m.matched}/{m.classes} 맞음
                    {m.missing.length > 0 && ` (없는 클래스: ${m.missing.join(", ")} — 버림)`}
                  </span>
                  {m.problem && <span className="muted small"> · {m.problem}</span>}
                </span>
              </label>
            ))}
          </div>
        )}

        {server && (
          <div className={`al-server ${server.problems.length ? "bad" : ""}`}>
            <Icon name="server" size={14} />
            <span>
              {server.id ? (
                <>
                  이미지가 있는 <strong>{server.name}</strong>에서 돕니다 ({server.gpus ? `GPU ${server.gpus}개` : "GPU 없음"}).
                </>
              ) : (
                "돌릴 서버를 찾지 못했습니다."
              )}
              {server.problems.map((p) => (
                <span key={p} className="bad-text">
                  {" "}
                  {p}
                </span>
              ))}
              {server.notes.map((p) => (
                <span key={p} className="muted">
                  {" "}
                  {p}
                </span>
              ))}
            </span>
            {server.gpus > 0 && (
              <div className="segmented" role="radiogroup" aria-label="계산">
                <button type="button" className={cpu ? "" : "active"} onClick={() => setCpu(false)}>
                  GPU
                </button>
                <button type="button" className={cpu ? "active" : ""} onClick={() => setCpu(true)}>
                  CPU
                </button>
              </div>
            )}
          </div>
        )}

        <div className="form-row">
          <label>
            최소 확신도
            <input type="number" min={0.05} max={0.95} step={0.05} value={keep} onChange={(e) => setKeep(Number(e.target.value) || 0.25)} />
            <span className="muted small">이보다 낮은 예측은 버립니다</span>
          </label>
          <label>
            <span className="check">
              <input type="checkbox" checked={accepting} onChange={(e) => setAccepting(e.target.checked)} /> 자동 승인
            </span>
            <input type="number" min={keep} max={1} step={0.05} value={accept} disabled={!accepting} onChange={(e) => setAccept(Number(e.target.value) || 0.8)} />
            <span className="muted small">예측이 모두 이 확신도 이상인 이미지는 바로 저장해 {reviewOf ? "검수로" : "완료로"} 보냅니다</span>
          </label>
        </div>
        {error && <div className="error">{error}</div>}
        <div className="modal-actions">
          <button type="button" onClick={onClose}>
            취소
          </button>
          <button className="primary" disabled={saving || blocked}>
            {fixedSet || mode === "set" ? "오토라벨 시작" : "묶음 만들고 오토라벨 시작"}
            {images > 0 ? ` · ${n(images)}장` : ""}
          </button>
        </div>
      </form>
    </Modal>
  );
}
