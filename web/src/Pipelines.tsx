import { type ReactNode, useEffect, useState } from "react";
import {
  type PipelineRunView,
  type PipelineStepView,
  RunStatusBadge,
  SPLIT_CHOICES,
  StepChain,
  type StepKind,
  TRIGGER_LABEL,
  stepHref,
  whenStarted,
} from "./FollowUps";
import { rememberProject } from "./DataBoard";
import { ProjectPicker, useProject } from "./ProjectPicker";
import { NewRun } from "./Training";
import { api } from "./api";
import type { TreeDataset } from "./catalogTypes";
import type { ModelInfo } from "./trainTypes";
import { Icon, Modal, Stat, duration, timeAgo, usePolling } from "./ui";
import "./pipelines.css";

// ---- what the Hub sends ------------------------------------------------------------------

interface PipelineView {
  id: string;
  name: string;
  project_id: string;
  dataset: { id: string; name: string; task: string } | null;
  steps: { kind: StepKind; label: string; summary: string; config: Record<string, unknown> }[];
  trigger: "manual" | "new_version";
  watch_since: string | null;
  note: string;
  created_by: string | null;
  created_at: string;
  runs: number;
  going: number;
  last: PipelineRunView | null;
  newest_release: { id: string; version: string } | null;
  history?: PipelineRunView[];
}

interface Board {
  pipelines: PipelineView[];
  runs: PipelineRunView[];
  totals: Record<"pipelines" | "watching" | "runs" | "going" | "succeeded" | "held" | "failed" | "promoted", number>;
}

const STEP_ICON: Record<PipelineStepView["status"], string> = {
  pending: "clock",
  running: "activity",
  succeeded: "check",
  failed: "x",
  skipped: "minus",
  stopped: "x",
  held: "minus",
};

// ---- the screen --------------------------------------------------------------------------

/** 파이프라인: a project's saved pipelines and their runs, one pipeline, one run. */
export function Pipelines({ sub, arg }: { sub?: string; arg?: string }) {
  if (sub === "run" && arg) return <RunPage key={arg} runId={arg} />;
  if (sub === "p" && arg) return <PipelinePage key={arg} pipelineId={arg} />;
  return <PipelineBoard />;
}

type Making = { template: "train" | "retrain" | "sota" } | null;

interface Template {
  id: "train" | "retrain" | "sota" | "labeling" | "sweep";
  title: string;
  steps: string[];
  body: string;
  ready: boolean;
}

/** The ways a pipeline can go, as the Hub can run them now (and the ones coming, shown as such). */
const TEMPLATES: Template[] = [
  {
    id: "train",
    title: "학습 → 평가 → 승격",
    steps: ["학습", "평가 · 검수", "승격", "내보내기"],
    body: "고른 버전으로 학습하고, 같은 데이터로 지금 SOTA와 비교해 더 나을 때만 SOTA로 지정합니다. 평가는 검수의 모델 검사로도 쓰입니다.",
    ready: true,
  },
  {
    id: "retrain",
    title: "새 버전마다 다시 학습",
    steps: ["새 버전", "학습", "평가", "승격"],
    body: "데이터셋에 새 버전이 생기면(가져오기 · 작업 중 확정) 같은 설정으로 다시 학습하고, 나을 때만 SOTA를 옮깁니다.",
    ready: true,
  },
  {
    id: "sota",
    title: "새 버전마다 SOTA 평가",
    steps: ["새 버전", "SOTA 평가 · 검수"],
    body: "새 버전이 생기면 지금 SOTA를 그 버전에서 평가합니다. 지난 버전보다 얼마나 달라졌는지 보고, 검수가 그 버전의 틀린 라벨 후보를 바로 찾습니다.",
    ready: true,
  },
  {
    id: "labeling",
    title: "오토라벨 → 라벨링 → 새 버전 → 학습",
    steps: ["오토라벨", "사람이 고침", "버전 확정", "학습"],
    body: "모델이 먼저 그리고 사람이 라벨링 묶음에서 고치면, 묶음이 끝날 때 새 버전을 만들어 다시 학습합니다. 사람 차례를 기다리는 단계가 필요해 다음에 넣습니다.",
    ready: false,
  },
  {
    id: "sweep",
    title: "탐색 → 최고 모델 평가 → 내보내기",
    steps: ["하이퍼파라미터 탐색", "최고 모델 평가", "ONNX"],
    body: "탐색이 끝나면 가장 좋은 실행의 모델을 고정한 테스트 세트로 평가하고 내보냅니다.",
    ready: false,
  },
];

function PipelineBoard() {
  const { projects, projectId, choose } = useProject();
  const [going, setGoing] = useState(false);
  const { data: board, error, refresh } = usePolling(
    () => (projectId ? api<Board>(`/projects/${projectId}/pipelines`) : Promise.resolve(null)),
    going ? 3000 : 15000,
    [projectId],
  );
  useEffect(() => setGoing((board?.totals.going ?? 0) > 0), [board]);
  const [making, setMaking] = useState<Making>(null);
  const [picking, setPicking] = useState(false);
  const project = projects?.find((p) => p.id === projectId);
  const t = board?.totals;
  const make = (template: Template["id"]) => {
    setPicking(false);
    if (template === "train" || template === "retrain" || template === "sota") setMaking({ template });
  };

  return (
    <>
      <div className="page-head">
        <div>
          <h1>파이프라인</h1>
          <p className="muted">
            학습 → 평가 · 검수 → 승격처럼 이어지는 일을 Hub가 차례로 합니다. 한 단계가 끝나면 다음을 시작하고, 실패하면 멈추고 이유를 남깁니다. 새
            학습 창의 "학습 끝나면"으로 만든 것도 여기에 모입니다.
          </p>
        </div>
        <div className="head-actions">
          <ProjectPicker projects={projects} projectId={projectId} onChange={choose} />
          <button className="primary" onClick={() => setPicking(true)} disabled={!projectId}>
            <Icon name="plus" />새 파이프라인
          </button>
        </div>
      </div>
      {error && !board && <div className="error">{error}</div>}
      <div className="stats five">
        <Stat icon="flow" label="파이프라인" value={t?.pipelines ?? "–"} sub={t?.watching ? `새 버전 감시 ${t.watching}` : "새 버전 감시 없음"} />
        <Stat icon="activity" label="진행 중" value={t?.going ?? "–"} sub={`실행 ${t?.runs ?? 0}번`} />
        <Stat icon="check" label="완료" value={t?.succeeded ?? "–"} sub={`승격 ${t?.promoted ?? 0}번`} />
        <Stat icon="minus" label="승격 안 함" value={t?.held ?? "–"} sub="지금 모델보다 낫지 않아 그대로" />
        <Stat icon="x" label="실패 · 멈춤" value={t?.failed ?? "–"} sub="실행에서 이유를 봅니다" />
      </div>

      <section className="board-section">
        <h2>템플릿</h2>
        <Templates onMake={make} />
      </section>

      <section className="board-section">
        <h2>
          저장한 파이프라인 <span className="muted">{board?.pipelines.length ?? ""}</span>
        </h2>
        {board && board.pipelines.length === 0 && (
          <div className="card empty">
            {project?.name ?? "이 프로젝트"}에 저장한 파이프라인이 없습니다. 위 템플릿에서 만들거나, 새 학습 창에서 "파이프라인으로 저장"을 고르세요.
          </div>
        )}
        <div className="pl-cards">
          {board?.pipelines.map((p) => (
            <PipelineCard key={p.id} pipeline={p} onRan={() => void refresh()} />
          ))}
        </div>
      </section>

      <section className="board-section">
        <h2>
          실행 기록 <span className="muted">{board?.runs.length ?? ""}</span>
        </h2>
        <RunRows runs={board?.runs ?? null} empty="아직 돌린 파이프라인이 없습니다. 새 학습 창의 &quot;학습 끝나면&quot;도 여기에 남습니다." />
      </section>

      {picking && <TemplatePicker onClose={() => setPicking(false)} onMake={make} />}
      {making && making.template !== "sota" && projectId && (
        <NewRun
          projectId={projectId}
          task={project?.tasks?.[0] ?? "detect"}
          defaultTemplate={project?.settings?.template}
          pipeline={{ trigger: making.template === "retrain" ? "new_version" : "manual" }}
          onClose={() => setMaking(null)}
          onStarted={() => setMaking(null)}
          onSweep={() => setMaking(null)}
          onPipeline={(id, runId) => {
            setMaking(null);
            void refresh();
            window.location.hash = runId ? `#/pipelines/run/${runId}` : `#/pipelines/p/${id}`;
          }}
        />
      )}
      {making?.template === "sota" && projectId && (
        <SotaCheck
          projectId={projectId}
          onClose={() => setMaking(null)}
          onMade={(id) => {
            setMaking(null);
            void refresh();
            window.location.hash = `#/pipelines/p/${id}`;
          }}
        />
      )}
    </>
  );
}

function Templates({ onMake }: { onMake: (id: Template["id"]) => void }) {
  const coming = TEMPLATES.filter((t) => !t.ready);
  return (
    <>
      <div className="pl-templates">
        {TEMPLATES.filter((t) => t.ready).map((t) => (
          <div key={t.id} className="card pl-template">
            <div className="pl-template-head">
              <Icon name="flow" />
              <strong>{t.title}</strong>
            </div>
            <ol className="pl-chain" aria-label="단계">
              {t.steps.map((s) => (
                <li key={s}>
                  <span className="pl-step plan">
                    <i />
                    {s}
                  </span>
                </li>
              ))}
            </ol>
            <p className="muted small">{t.body}</p>
            <div>
              <button className="small primary" onClick={() => onMake(t.id)}>
                만들기
              </button>
            </div>
          </div>
        ))}
      </div>
      {coming.length > 0 && (
        <div className="pl-coming muted small">
          <span className="tag">준비 중</span>
          {coming.map((t) => (
            <span key={t.id} title={t.body}>
              {t.steps.join(" → ")}
            </span>
          ))}
        </div>
      )}
    </>
  );
}

function TemplatePicker({ onClose, onMake }: { onClose: () => void; onMake: (id: Template["id"]) => void }) {
  return (
    <Modal title="새 파이프라인" onClose={onClose}>
      <p className="muted small">무엇을 이어서 할지 고르세요. 학습 설정은 새 학습 창과 같고, 학습 뒤 단계를 고르면 됩니다.</p>
      <div className="pl-picks">
        {TEMPLATES.filter((t) => t.ready).map((t) => (
          <button key={t.id} type="button" className="pl-pick" onClick={() => onMake(t.id)}>
            <strong>{t.title}</strong>
            <span className="muted small">{t.steps.join(" → ")}</span>
          </button>
        ))}
      </div>
    </Modal>
  );
}

function PipelineCard({ pipeline: p, onRan }: { pipeline: PipelineView; onRan: () => void }) {
  const [busy, setBusy] = useState(false);
  const [error, setError] = useState<string | null>(null);
  const run = async () => {
    setBusy(true);
    setError(null);
    try {
      const started = await api<PipelineRunView>(`/pipelines/${p.id}/run`, { body: {} });
      onRan();
      window.location.hash = `#/pipelines/run/${started.id}`;
    } catch (e) {
      setError(e instanceof Error ? e.message : String(e));
    } finally {
      setBusy(false);
    }
  };
  return (
    <div className="card pl-card">
      <a className="pl-card-link" href={`#/pipelines/p/${p.id}`}>
        <div className="pl-card-title">
          <strong>{p.name}</strong>
          {p.trigger === "new_version" ? <span className="tag accent">새 버전마다</span> : <span className="tag">직접 실행</span>}
        </div>
        <div className="muted small">
          {p.dataset?.name ?? "지운 데이터셋"}
          {p.newest_release ? ` · 최신 ${p.newest_release.version}` : ""} · 실행 {p.runs}번
        </div>
      </a>
      <StepChain steps={p.steps} />
      <div className="pl-card-foot">
        {p.last ? (
          <a className="pl-last small" href={`#/pipelines/run/${p.last.id}`}>
            <RunStatusBadge status={p.last.status} />
            <span className="muted">
              {p.last.release?.version} · {timeAgo(p.last.created_at)}
            </span>
          </a>
        ) : (
          <span className="muted small">아직 돌리지 않음</span>
        )}
        <button className="small" disabled={busy || !p.newest_release} onClick={() => void run()} title="데이터셋의 최신 버전으로 지금 돌립니다">
          <Icon name="play" size={12} /> 실행
        </button>
      </div>
      {error && <div className="error small">{error}</div>}
    </div>
  );
}

function RunRows({ runs, empty, showPipeline = true }: { runs: PipelineRunView[] | null; empty: ReactNode; showPipeline?: boolean }) {
  if (!runs) return <p className="muted">불러오는 중…</p>;
  if (runs.length === 0) return <div className="card empty">{empty}</div>;
  return (
    <div className="card flush table-wrap">
      <table className="pl-runs">
        <thead>
          <tr>
            <th>상태</th>
            <th>실행</th>
            <th>버전</th>
            <th>단계</th>
            <th>시작</th>
          </tr>
        </thead>
        <tbody>
          {runs.map((r) => (
            <tr key={r.id}>
              <td>
                <RunStatusBadge status={r.status} />
              </td>
              <td className="pl-run-name">
                <a href={`#/pipelines/run/${r.id}`}>{r.name || "이름 없음"}</a>
                {showPipeline && (
                  <div className="muted small">{r.pipeline ? `파이프라인 ${r.pipeline.name}` : "저장하지 않은 단계"}</div>
                )}
              </td>
              <td className="nowrap">{r.release ? `${r.release.dataset} ${r.release.version}` : "지운 버전"}</td>
              <td>
                <StepChain steps={r.steps} />
                {r.error && <div className="bad-text small ellipsis pl-run-error">{r.error}</div>}
              </td>
              <td className="muted small nowrap">
                {TRIGGER_LABEL[r.trigger]}
                <div>{timeAgo(r.created_at)}</div>
              </td>
            </tr>
          ))}
        </tbody>
      </table>
    </div>
  );
}

// ---- one pipeline -------------------------------------------------------------------------

function PipelinePage({ pipelineId }: { pipelineId: string }) {
  const [going, setGoing] = useState(false);
  const { data: p, error, refresh } = usePolling(() => api<PipelineView>(`/pipelines/${pipelineId}`), going ? 3000 : 15000, [pipelineId]);
  useEffect(() => setGoing((p?.going ?? 0) > 0), [p]);
  useEffect(() => {
    if (p) rememberProject(p.project_id);
  }, [p]);
  const [tree, setTree] = useState<TreeDataset[] | null>(null);
  const [release, setRelease] = useState("");
  const [busy, setBusy] = useState(false);
  const [failure, setFailure] = useState<string | null>(null);
  useEffect(() => {
    if (p) void api<TreeDataset[]>(`/projects/${p.project_id}/datasets`).then(setTree);
  }, [p?.project_id]); // eslint-disable-line react-hooks/exhaustive-deps
  if (error && !p) return <div className="error">{error}</div>;
  if (!p) return <div className="muted">불러오는 중…</div>;
  const versions = (tree?.find((d) => d.id === p.dataset?.id)?.releases ?? []).filter((r) => r.status === "ready" && !r.deleted_at);
  const act = async (work: () => Promise<unknown>) => {
    setBusy(true);
    setFailure(null);
    try {
      await work();
      await refresh();
    } catch (e) {
      setFailure(e instanceof Error ? e.message : String(e));
    } finally {
      setBusy(false);
    }
  };
  const runNow = () =>
    act(async () => {
      const started = await api<PipelineRunView>(`/pipelines/${p.id}/run`, { body: { release_id: release } });
      window.location.hash = `#/pipelines/run/${started.id}`;
    });
  const setTrigger = (trigger: string) => act(() => api(`/pipelines/${p.id}`, { method: "PATCH", body: { trigger } }));
  const remove = () => {
    if (!window.confirm(`파이프라인 ${p.name}을(를) 지울까요? 지금까지의 실행 기록과 만든 모델은 그대로 남습니다.`)) return;
    void act(async () => {
      await api(`/pipelines/${p.id}`, { method: "DELETE" });
      window.location.hash = "#/pipelines";
    });
  };
  return (
    <>
      <nav className="page-crumbs">
        <a href="#/pipelines">파이프라인</a>
        <strong>{p.name}</strong>
      </nav>
      <div className="page-head">
        <div>
          <h1>{p.name}</h1>
          <p className="muted">
            {p.dataset ? <a href={`#/datasets`}>{p.dataset.name}</a> : "지운 데이터셋"} · {p.created_by ?? "?"}가 {timeAgo(p.created_at)} 만듦 · 실행{" "}
            {p.runs}번
          </p>
        </div>
        <div className="head-actions">
          <label className="pl-run-version">
            <span className="eyebrow">버전</span>
            <select value={release} onChange={(e) => setRelease(e.target.value)} aria-label="실행할 버전">
              <option value="">최신{p.newest_release ? ` (${p.newest_release.version})` : ""}</option>
              {versions.map((r) => (
                <option key={r.id} value={r.id}>
                  {r.version}
                </option>
              ))}
            </select>
          </label>
          <button className="primary" disabled={busy || !p.newest_release} onClick={() => void runNow()}>
            <Icon name="play" size={14} /> 지금 실행
          </button>
          <button disabled={busy} onClick={remove}>
            삭제
          </button>
        </div>
      </div>
      {failure && <div className="error">{failure}</div>}

      <div className="card pl-spec">
        <div className="pl-spec-row">
          <span className="eyebrow">단계</span>
          <ol className="pl-spec-steps">
            {p.steps.map((s, i) => (
              <li key={s.kind}>
                <span className="pl-spec-n">{i + 1}</span>
                <strong>{s.label}</strong>
                <span className="muted">{s.summary}</span>
              </li>
            ))}
          </ol>
        </div>
        <div className="pl-spec-row">
          <span className="eyebrow">언제</span>
          <div className="pl-check">
            <input
              id="pl-watch"
              type="checkbox"
              checked={p.trigger === "new_version"}
              disabled={busy}
              onChange={(e) => void setTrigger(e.target.checked ? "new_version" : "manual")}
            />
            <label htmlFor="pl-watch">{p.dataset?.name ?? "데이터셋"}에 새 버전이 생기면 자동으로 실행</label>
            <span className="pl-hint">
              {p.trigger === "new_version" && p.watch_since ? `${timeAgo(p.watch_since)} 이후에 생긴 버전부터 · ` : ""}가져오기나 작업 중 확정으로 생긴
              버전, 가장 최신 하나만
            </span>
          </div>
        </div>
      </div>

      <section className="board-section">
        <h2>
          실행 기록 <span className="muted">{p.history?.length ?? ""}</span>
        </h2>
        <RunRows runs={p.history ?? null} empty="아직 돌리지 않았습니다. 지금 실행을 누르거나 새 버전을 기다립니다." showPipeline={false} />
      </section>
    </>
  );
}

// ---- one run ------------------------------------------------------------------------------

function RunPage({ runId }: { runId: string }) {
  const [live, setLive] = useState(true);
  const { data: run, error, refresh } = usePolling(() => api<PipelineRunView>(`/pipeline-runs/${runId}`), live ? 2500 : 20000, [runId]);
  useEffect(() => setLive(!run || run.status === "running"), [run]);
  useEffect(() => {
    if (run) rememberProject(run.project.id);
  }, [run]);
  const [busy, setBusy] = useState(false);
  const [failure, setFailure] = useState<string | null>(null);
  if (error && !run) return <div className="error">{error}</div>;
  if (!run) return <div className="muted">불러오는 중…</div>;
  const act = async (path: string, confirmText?: string) => {
    if (confirmText && !window.confirm(confirmText)) return;
    setBusy(true);
    setFailure(null);
    try {
      await api(path, { method: "POST" });
      await refresh();
    } catch (e) {
      setFailure(e instanceof Error ? e.message : String(e));
    } finally {
      setBusy(false);
    }
  };
  const current = run.steps.find((s) => s.status === "running" || s.status === "pending");
  const stoppedAt = run.steps.find((s) => s.status === "failed" || s.status === "stopped");
  return (
    <>
      <nav className="page-crumbs">
        <a href="#/pipelines">파이프라인</a>
        {run.pipeline ? <a href={`#/pipelines/p/${run.pipeline.id}`}>{run.pipeline.name}</a> : <span>저장하지 않은 단계</span>}
        <strong>{run.name || "실행"}</strong>
      </nav>
      <div className="page-head">
        <div>
          <h1 className="pl-run-title">
            {run.name || "파이프라인 실행"} <RunStatusBadge status={run.status} />
          </h1>
          <p className="muted">
            {run.release ? (
              <a href={`#/datasets/${run.release.id}`}>
                {run.release.dataset} {run.release.version}
              </a>
            ) : (
              "지운 버전"
            )}{" "}
            · {whenStarted(run)} · {run.status === "running" ? `${duration(run.duration_s)} 지남` : `${duration(run.duration_s)} 걸림`}
          </p>
        </div>
        <div className="head-actions">
          {run.status === "running" && (
            <button
              className="danger"
              disabled={busy}
              onClick={() =>
                void act(`/pipeline-runs/${run.id}/stop`, `지금 단계(${current?.label ?? "?"})를 중단하고 남은 단계를 하지 않을까요? 끝난 단계는 그대로 남습니다.`)
              }
            >
              멈추기
            </button>
          )}
          {stoppedAt && run.status !== "running" && (
            <button className="primary" disabled={busy} onClick={() => void act(`/pipeline-runs/${run.id}/retry`)} title="끝난 단계는 두고 그 단계부터 다시 합니다">
              {stoppedAt.label} 단계부터 다시
            </button>
          )}
        </div>
      </div>
      {failure && <div className="error">{failure}</div>}
      {run.error && <div className="error">{run.error}</div>}
      {run.status === "held" && (
        <div className="notice">
          {run.steps.find((s) => s.kind === "promote")?.message} — 새 모델은 등록되어 있고, 모델 화면에서 직접 지정할 수도 있습니다.
        </div>
      )}
      <ol className="pl-timeline">
        {run.steps.map((s, i) => (
          <StepCard key={`${s.kind}-${i}`} step={s} measured={measuredOn(run)} />
        ))}
      </ol>
    </>
  );
}

/** Where the evaluation measured: what the gate compared on. */
const measuredOn = (run: PipelineRunView) => {
  const e = run.steps.find((s) => s.kind === "eval");
  return e?.release ? `${e.release.dataset} ${e.release.version} · ${e.split}` : "";
};

function StepCard({ step: s, measured }: { step: PipelineStepView; measured: string }) {
  const href = stepHref(s);
  const links: ReactNode[] = [];
  if (s.kind === "train" && s.run) links.push(<a key="run" href={`#/training/${s.run.id}`}>학습 {s.run.name}</a>);
  if (s.model && s.kind !== "promote") links.push(<a key="model" href={`#/models/${s.model.id}`}>모델 {s.model.name} v{s.model.version}</a>);
  if (s.kind === "eval" && s.op_id) links.push(<a key="op" href={`#/models/eval/${s.op_id}`}>평가 결과</a>);
  if (s.kind === "eval" && s.review_release_id && s.status === "succeeded")
    links.push(
      <a key="review" href={`#/review/${s.review_release_id}`}>
        검수에서 틀린 라벨 후보 보기
      </a>,
    );
  if (s.kind === "promote" && s.model) links.push(<a key="m" href={`#/models/${s.model.id}`}>새 모델 v{s.model.version}</a>);
  if (s.kind === "promote" && s.holder) links.push(<a key="h" href={`#/models/${s.holder.id}`}>비교한 {s.alias} v{s.holder.version}</a>);
  if (s.kind === "export" && s.op_id) links.push(<a key="ex" href={href}>내려받기</a>);
  const progress =
    s.status === "running" && s.kind === "train" && s.run && s.run.epochs > 0 ? `epoch ${s.run.epoch}/${s.run.epochs}` : "";
  return (
    <li className={`pl-tl ${s.status}`}>
      <span className={`pl-dot ${s.status}`} aria-hidden="true">
        <Icon name={STEP_ICON[s.status]} size={14} />
      </span>
      <div className="card pl-tl-card">
        <div className="pl-tl-head">
          <strong>{s.label}</strong>
          <span className={`pl-tl-state ${s.status}`}>{STATE_LABEL[s.status]}</span>
          <span className="muted small">{s.summary}</span>
          {s.started_at && (
            <span className="muted small pl-tl-time">
              {timeAgo(s.started_at)}
              {s.finished_at ? ` · ${duration((Date.parse(s.finished_at) - Date.parse(s.started_at)) / 1000)}` : ""}
            </span>
          )}
        </div>
        {(s.message || progress || s.job?.wait_reason) && (
          <div className={`pl-tl-message${s.status === "failed" ? " bad-text" : ""}`}>
            {s.message || progress}
            {s.status === "running" && s.job?.status === "queued" && s.job.wait_reason && <div className="muted small">대기: {s.job.wait_reason}</div>}
          </div>
        )}
        {s.kind === "eval" && s.metric?.value != null && s.metric.key && s.before && (
          <div className="pl-tl-metrics small">
            <span>
              {s.metric.label} <strong className="num">{s.metric.value.toFixed(3)}</strong>
            </span>
            {s.before && (
              <span className="muted">
                {s.before.version}에서 {s.before.value.toFixed(3)}
              </span>
            )}
          </div>
        )}
        {s.kind === "promote" && s.mine != null && (
          <div className="pl-tl-metrics small">
            <span>
              새 모델 <strong className="num">{s.mine.toFixed(3)}</strong>
            </span>
            <span>
              {s.alias} {s.holder ? `v${s.holder.version}` : "없음"} <strong className="num">{s.theirs != null ? s.theirs.toFixed(3) : "–"}</strong>
            </span>
            <span className="muted">
              {s.metric?.label} · 둘 다 {measured}에서
            </span>
          </div>
        )}
        {links.length > 0 && <div className="pl-tl-links small">{links}</div>}
      </div>
    </li>
  );
}

const STATE_LABEL: Record<PipelineStepView["status"], string> = {
  pending: "대기",
  running: "진행 중",
  succeeded: "완료",
  failed: "실패",
  skipped: "건너뜀",
  stopped: "멈춤",
  held: "그대로 둠",
};

// ---- the SOTA, measured on each new version ----------------------------------------------

function SotaCheck({ projectId, onClose, onMade }: { projectId: string; onClose: () => void; onMade: (id: string) => void }) {
  const [tree, setTree] = useState<TreeDataset[] | null>(null);
  const [models, setModels] = useState<ModelInfo[] | null>(null);
  const [dataset, setDataset] = useState("");
  const [family, setFamily] = useState("");
  const [alias, setAlias] = useState("SOTA");
  const [split, setSplit] = useState("auto");
  const [review, setReview] = useState(true);
  const [watch, setWatch] = useState(true);
  const [now, setNow] = useState(true);
  const [name, setName] = useState("");
  const [error, setError] = useState<string | null>(null);
  const [busy, setBusy] = useState(false);
  useEffect(() => {
    void api<TreeDataset[]>(`/projects/${projectId}/datasets`).then(setTree);
    void api<ModelInfo[]>(`/projects/${projectId}/models`).then(setModels);
  }, [projectId]);
  // model families with an alias on one of their versions
  const families = new Map<string, Set<string>>();
  for (const m of models ?? []) for (const a of m.aliases) families.set(m.name, (families.get(m.name) ?? new Set()).add(a));
  useEffect(() => {
    if (!family && families.size > 0) pickFamily([...families.keys()][0]);
  }, [models]); // eslint-disable-line react-hooks/exhaustive-deps
  // the family's datasets of its task; unless one is picked, the one it was trained on
  const trained = (models ?? []).find((m) => m.name === family);
  const fits = (tree ?? []).filter((d) => !trained || d.task === trained.task);
  const home = fits.find((d) => d.releases.some((r) => r.id === trained?.release?.id)) ?? fits[0];
  const watched = fits.some((d) => d.id === dataset) ? dataset : (home?.id ?? "");
  function pickFamily(name: string) {
    const aliases = families.get(name) ?? new Set<string>();
    setFamily(name);
    setAlias(aliases.has("SOTA") ? "SOTA" : ([...aliases][0] ?? ""));
    setDataset("");
  }
  const submit = async () => {
    setBusy(true);
    setError(null);
    try {
      const made = await api<PipelineView>("/pipelines", {
        body: {
          dataset_id: watched,
          name,
          steps: [{ kind: "eval", config: { model: { name: family, alias }, split, review } }],
          trigger: watch ? "new_version" : "manual",
          run: now,
        },
      });
      onMade(made.id);
    } catch (e) {
      setError(e instanceof Error ? e.message : String(e));
      setBusy(false);
    }
  };
  return (
    <Modal title="새 버전마다 SOTA 평가" onClose={onClose}>
      {models && families.size === 0 ? (
        <p className="muted">이 프로젝트에는 alias(SOTA 등)가 붙은 모델이 없습니다. 모델 화면에서 alias를 붙이거나, "학습 → 평가 → 승격"으로 먼저 만드세요.</p>
      ) : (
        <>
          <div className="form-row">
            <label>
              모델
              <select value={family} onChange={(e) => pickFamily(e.target.value)}>
                {[...families.keys()].map((f) => (
                  <option key={f} value={f}>
                    {f}
                  </option>
                ))}
              </select>
            </label>
            <label>
              alias
              <select value={alias} onChange={(e) => setAlias(e.target.value)}>
                {[...(families.get(family) ?? [])].map((a) => (
                  <option key={a} value={a}>
                    {a}
                  </option>
                ))}
              </select>
            </label>
          </div>
          <div className="form-row">
            <label>
              새 버전을 볼 데이터셋
              <select value={watched} onChange={(e) => setDataset(e.target.value)}>
                {fits.map((d) => (
                  <option key={d.id} value={d.id}>
                    {d.name}
                  </option>
                ))}
              </select>
            </label>
            <label>
              split
              <select value={split} onChange={(e) => setSplit(e.target.value)}>
                {SPLIT_CHOICES.map(([value, label]) => (
                  <option key={value} value={value}>
                    {label}
                  </option>
                ))}
              </select>
            </label>
          </div>
          <div className="pl-checks">
            <div className="pl-check">
              <input id="cc-review" type="checkbox" checked={review} onChange={(e) => setReview(e.target.checked)} />
              <label htmlFor="cc-review">검수에도 쓰기</label>
              <span className="pl-hint">그 버전의 모든 이미지를 평가해 검수가 틀린 라벨 후보를 찾습니다</span>
            </div>
            <div className="pl-check">
              <input id="cc-watch" type="checkbox" checked={watch} onChange={(e) => setWatch(e.target.checked)} />
              <label htmlFor="cc-watch">새 버전이 생기면 자동으로</label>
            </div>
            <div className="pl-check">
              <input id="cc-now" type="checkbox" checked={now} onChange={(e) => setNow(e.target.checked)} />
              <label htmlFor="cc-now">지금 최신 버전으로 한 번 실행</label>
            </div>
          </div>
          <label>
            파이프라인 이름
            <input value={name} onChange={(e) => setName(e.target.value)} placeholder="자동: 데이터셋 평가" />
          </label>
        </>
      )}
      {error && <div className="error">{error}</div>}
      <div className="modal-actions">
        <button type="button" onClick={onClose}>
          취소
        </button>
        <button type="button" className="primary" disabled={busy || !watched || !family} onClick={() => void submit()}>
          만들기
        </button>
      </div>
    </Modal>
  );
}
