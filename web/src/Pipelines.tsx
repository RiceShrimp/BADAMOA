import { type ReactNode, useEffect, useMemo, useState } from "react";
import {
  DEFAULT_ALIAS,
  type PipelineRunView,
  type PipelineStepView,
  type PipelineTemplate,
  RunStatusBadge,
  SPLIT_CHOICES,
  StepChain,
  type StepKind,
  type StepSettings,
  TRIGGER_LABEL,
  stepHref,
  whenStarted,
} from "./FollowUps";
import { LayoutSwitch, layoutClass, useCardLayout } from "./CardLayout";
import { rememberProject } from "./DataBoard";
import { isDataServer } from "./Datasets";
import { FolderBrowser, LAYOUT_NAME } from "./FolderBrowser";
import { ProjectPicker, useProject } from "./ProjectPicker";
import { NewRun } from "./Training";
import { type NodeInfo, api } from "./api";
import type { TreeDataset } from "./catalogTypes";
import { METRIC_LABEL, type ModelInfo, keyMetrics } from "./trainTypes";
import { Icon, Modal, Stat, duration, timeAgo, usePolling } from "./ui";
import "./pipelines.css";

// ---- what the Hub sends ------------------------------------------------------------------

interface StepLine {
  kind: StepKind;
  label: string;
  summary: string;
  gate: boolean;
  config: Record<string, unknown>;
}

interface PipelineView {
  id: string;
  name: string;
  project_id: string;
  dataset: { id: string; name: string; task: string } | null;
  steps: StepLine[];
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

const STATE_LABEL: Record<PipelineStepView["status"], string> = {
  pending: "대기",
  running: "진행 중",
  succeeded: "완료",
  failed: "실패",
  skipped: "건너뜀",
  stopped: "멈춤",
  held: "기준 미달 · 멈춤",
};

// ---- the steps there are ------------------------------------------------------------------

const MAKES_MODEL: StepKind[] = ["train", "sweep"];

/** Every step the Hub can take now, by what it works on. Each has a few words on what it does. */
const STEP_KINDS: { group: string; kinds: { kind: StepKind; label: string; what: string }[] }[] = [
  {
    group: "데이터",
    kinds: [
      { kind: "import", label: "가져오기", what: "서버의 폴더를 이 데이터셋의 새 버전으로 (바뀌지 않은 파일은 다시 읽지 않음)" },
      {
        kind: "sync",
        label: "서버로 보내기",
        what: "이 버전의 이미지 파일을 고른 서버에 미리 복사 (학습 단계에서 노드를 고르면 그 노드가 알아서 받으므로 보통은 필요 없음)",
      },
    ],
  },
  {
    group: "라벨",
    kinds: [
      { kind: "autolabel", label: "오토라벨", what: "모델이 라벨 안 한 이미지에 먼저 그림 (라벨링 묶음으로)" },
      { kind: "label", label: "라벨링", what: "라벨링 묶음을 만들어 나누고, 사람이 끝낼 때까지 기다림" },
      { kind: "confirm", label: "버전 확정", what: "작업 중을 새 버전으로 (이름은 다음 번호 또는 직접)" },
      { kind: "review", label: "검수", what: "모델로 검사하기: 모델이 지금 버전의 모든 이미지를 예측해 라벨이 틀렸을 수 있는 이미지를 검수에 올림" },
    ],
  },
  {
    group: "모델",
    kinds: [
      { kind: "train", label: "학습", what: "지금 버전으로 학습해 모델 등록" },
      { kind: "sweep", label: "탐색", what: "하이퍼파라미터 탐색, 가장 좋은 실행의 모델을 다음 단계로" },
      { kind: "eval", label: "평가", what: "모델을 버전 · split에서 평가 (검수의 모델 기준으로도)" },
      { kind: "promote", label: "승격", what: "지금 SOTA보다 나을 때만 alias를 옮김 (게이트)" },
      { kind: "export", label: "내보내기", what: "모델을 ONNX 등으로" },
    ],
  },
];
const KIND_LABEL = {
  ...Object.fromEntries(STEP_KINDS.flatMap((g) => g.kinds.map((k) => [k.kind, k.label]))),
  check: "검수 확인 (예전)", // no longer offered; a saved one still runs
} as Record<StepKind, string>;
const USES_MODEL: StepKind[] = ["autolabel", "review", "eval", "export"];

/** A server a step may run on, as the Hub lists it by the rules of the step's own screen. */
interface Place {
  id: string;
  name: string;
  status: string;
  gpus: number;
  fits: boolean;
  why: string;
  transfer?: boolean; // the version's images go there first
  holds_images?: boolean;
  trained_here?: boolean;
}
interface Places {
  default: string;
  nodes: Place[];
  problem?: string;
}
/** The steps that run on a server picked (or 자동), and how their list of servers is asked for. */
const PLACE_KINDS: StepKind[] = ["import", "sync", "autolabel", "review", "eval", "export"];
const placeKey = (datasetId: string, kind: StepKind, c: Record<string, unknown>) => {
  const model = c.model as { name?: string; alias?: string } | undefined;
  return `${datasetId}|${kind}|${model?.name ?? ""}|${model?.alias ?? ""}`;
};

const DEFAULT_CONFIG: Record<StepKind, Record<string, unknown>> = {
  import: { path: "" },
  check: { max_flagged: 0, group: "any" },
  review: {},
  sync: {},
  autolabel: {},
  label: { divide: "size", part_size: 50, review: false },
  confirm: {},
  train: {},
  sweep: { space: {}, method: "random", max_runs: 6 },
  eval: { split: "auto", review: true },
  promote: { alias: DEFAULT_ALIAS },
  export: { format: "onnx" },
};

const CHECK_GROUPS: [string, string][] = [
  ["any", "문제"],
  ["missing", "라벨 빠짐"],
  ["wrong", "틀린 라벨"],
  ["boxes", "박스 문제"],
];

const isGate = (kind: StepKind, config: Record<string, unknown>) =>
  kind === "check" || kind === "promote" || (config.at_least !== undefined && config.at_least !== null && config.at_least !== "");
const makesModelBefore = (steps: { kind: StepKind }[], index: number) => steps.slice(0, index).some((s) => MAKES_MODEL.includes(s.kind));

/** The names a dataset's versions have (in the bin too): a new one may not take any. */
const versionNames = (dataset?: TreeDataset) => new Set((dataset?.releases ?? []).map((r) => r.version));
/** The next vN, as the Hub picks it when no name is given. */
function nextVersion(dataset?: TreeDataset): string {
  const names = versionNames(dataset);
  const numbers = [...names].flatMap((v) => {
    const m = /^[vV](\d+)$/.exec(v);
    return m ? [Number(m[1])] : [];
  });
  let n = (numbers.length ? Math.max(...numbers) : names.size) + 1;
  while (names.has(`v${n}`)) n += 1;
  return `v${n}`;
}
const today = () => new Date().toISOString().slice(0, 10).replaceAll("-", "");
/** What is wrong with a step's name for the version it makes ("" for nothing), as the Hub checks it. */
function versionProblem(pattern: string, dataset?: TreeDataset): string {
  if (!pattern) return "";
  const rest = pattern.replaceAll("{n}", "").replaceAll("{date}", "");
  if (rest.includes("{") || rest.includes("}") || pattern.split("{n}").length > 2) {
    return "버전 이름에는 {n}(다음 번호)과 {date}(그날 날짜)를 한 번씩만 넣을 수 있습니다.";
  }
  if (pattern.includes("{n}")) return "";
  const name = pattern.replaceAll("{date}", today());
  return versionNames(dataset).has(name)
    ? `${dataset?.name ?? "이 데이터셋"}에 ${name} 버전이 이미 있습니다(휴지통 포함). 다른 이름을 쓰거나 {n}을 넣으세요.`
    : "";
}

/** What keeps the steps from running in this order, by step (as the Hub refuses them when saved). */
function flowProblems(steps: Draft[], dataset?: TreeDataset, places: Record<string, Places> = {}): string[][] {
  let made = false; // a model made by a step before
  let measured = false; // and an evaluation of it since
  return steps.map((s) => {
    const c = s.config;
    const out: string[] = [];
    if (USES_MODEL.includes(s.kind) && !made && !c.model) out.push("앞에 학습이나 탐색이 없으면 쓸 모델을 고르세요.");
    if (s.kind === "promote" && !measured) out.push("승격은 바로 앞의 평가로 정합니다. 학습 · 탐색 뒤에 평가를 넣고 그다음에 두세요.");
    if (s.kind === "import" && (!c.node_id || !String(c.path ?? "").trim())) out.push("서버와 폴더를 고르세요.");
    if (s.kind === "sync" && !c.node_id) out.push("보낼 서버를 고르세요.");
    if (s.kind === "sweep" && Object.keys((c.space ?? {}) as object).length === 0) out.push("탐색 설정 열기에서 시험할 설정을 넣으세요.");
    if (s.kind === "import" || s.kind === "confirm") {
      const wrong = versionProblem(String(c.version ?? ""), dataset);
      if (wrong) out.push(wrong);
    }
    // the server: one the step's own screen would offer (as the Hub lists them); 자동's too, as those screens
    // will not start on a default that cannot
    const listed = dataset && PLACE_KINDS.includes(s.kind) ? places[placeKey(dataset.id, s.kind, c)] : undefined;
    if (listed && c.node_id) {
      const place = listed.nodes.find((n) => n.id === c.node_id);
      if (!place) out.push("고른 서버가 작업을 받지 않습니다. 다른 서버를 고르세요.");
      else if (!place.fits) out.push(`${place.name}에서는 할 수 없습니다: ${place.why}`);
    } else if (listed && listed.default && USES_MODEL.includes(s.kind)) {
      const auto = listed.nodes.find((n) => n.id === listed.default);
      if (auto && !auto.fits) out.push(`자동이면 ${auto.name}에서 하는데 할 수 없습니다(${auto.why}). 실행할 서버를 고르세요.`);
    }
    if (MAKES_MODEL.includes(s.kind)) [made, measured] = [true, false];
    else if (s.kind === "eval") measured = !c.model;
    return out;
  });
}

// ---- the screen --------------------------------------------------------------------------

/** 파이프라인: a project's templates, saved pipelines and runs; making one (composing its steps); one pipeline; one run. */
export function Pipelines({ sub, arg }: { sub?: string; arg?: string }) {
  if (sub === "run" && arg) return <RunPage key={arg} runId={arg} />;
  if (sub === "p" && arg) return <PipelinePage key={arg} pipelineId={arg} />;
  if (sub === "new") return <Composer key={arg ?? "blank"} templateId={arg ?? ""} />;
  if (sub === "edit" && arg) return <Composer key={`edit-${arg}`} pipelineId={arg} />;
  return <PipelineBoard />;
}

function PipelineBoard() {
  const { projects, projectId, choose } = useProject();
  const [going, setGoing] = useState(false);
  const { data: board, error, refresh } = usePolling(
    () => (projectId ? api<Board>(`/projects/${projectId}/pipelines`) : Promise.resolve(null)),
    going ? 3000 : 15000,
    [projectId],
  );
  useEffect(() => setGoing((board?.totals.going ?? 0) > 0), [board]);
  const { data: templates, refresh: refreshTemplates } = usePolling(() => api<PipelineTemplate[]>("/pipeline-templates"), 60000);
  const project = projects?.find((p) => p.id === projectId);
  const t = board?.totals;
  // templates and saved pipelines: one long card to a row unless the grid is picked
  const [layout, setLayout] = useCardLayout("pipelines", "rows");

  return (
    <>
      <div className="page-head">
        <div>
          <h1>파이프라인</h1>
          <p className="muted">
            학습 → 평가 · 검수 → 승격처럼 이어지는 일을 Hub가 차례로 합니다. 한 단계가 끝나면 다음을 시작하고, 기준에 못 미치면 멈추고, 실패하면 이유를
            남깁니다. 템플릿에서 시작하거나 단계를 직접 조합합니다.
          </p>
        </div>
        <div className="head-actions">
          <ProjectPicker projects={projects} projectId={projectId} onChange={choose} />
          <button className="primary" disabled={!projectId} onClick={() => (window.location.hash = "#/pipelines/new")}>
            <Icon name="plus" />새 파이프라인
          </button>
        </div>
      </div>
      {error && !board && <div className="error">{error}</div>}
      <div className="stats five">
        <Stat icon="flow" label="파이프라인" value={t?.pipelines ?? "–"} sub={t?.watching ? `새 버전 감시 ${t.watching}` : "새 버전 감시 없음"} />
        <Stat icon="activity" label="진행 중" value={t?.going ?? "–"} sub={`실행 ${t?.runs ?? 0}번`} />
        <Stat icon="check" label="완료" value={t?.succeeded ?? "–"} sub={`승격 ${t?.promoted ?? 0}번`} />
        <Stat icon="minus" label="기준 미달" value={t?.held ?? "–"} sub="게이트에서 멈춤 (승격 안 함 등)" />
        <Stat icon="x" label="실패 · 멈춤" value={t?.failed ?? "–"} sub="실행에서 이유를 봅니다" />
      </div>

      <section className="board-section">
        <div className="layout-head">
          <h2>템플릿</h2>
          <LayoutSwitch value={layout} onChange={setLayout} />
        </div>
        <Templates templates={templates ?? null} layout={layoutClass(layout)} onChanged={() => void refreshTemplates()} />
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
        <div className={`pl-cards${layoutClass(layout)}`}>
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
    </>
  );
}

function Templates({ templates, layout, onChanged }: { templates: PipelineTemplate[] | null; layout: string; onChanged: () => void }) {
  const [error, setError] = useState<string | null>(null);
  if (!templates) return <p className="muted">불러오는 중…</p>;
  const remove = async (t: PipelineTemplate) => {
    if (!window.confirm(`템플릿 ${t.name}을(를) 지울까요? 이미 만든 파이프라인은 그대로입니다.`)) return;
    setError(null);
    try {
      await api(`/pipeline-templates/${t.id}`, { method: "DELETE" });
      onChanged();
    } catch (e) {
      setError(e instanceof Error ? e.message : String(e));
    }
  };
  return (
    <>
      {error && <div className="error">{error}</div>}
      <div className={`pl-templates${layout}`}>
        {templates.map((t) => (
          <div key={t.id} className="card pl-template">
            <div className="pl-template-head">
              <Icon name="flow" />
              <strong>{t.name}</strong>
              {!t.builtin && <span className="tag accent">{t.created_by ?? "팀"}</span>}
              {t.trigger === "new_version" && <span className="tag">새 버전마다</span>}
            </div>
            <StepChain steps={t.steps} plain />
            {t.description && <p className="muted small">{t.description}</p>}
            <div className="pl-template-foot">
              <button type="button" className="small primary" onClick={() => (window.location.hash = `#/pipelines/new/${t.id}`)}>
                만들기
              </button>
              {!t.builtin && (
                <button type="button" className="link small" onClick={() => void remove(t)}>
                  지우기
                </button>
              )}
            </div>
          </div>
        ))}
        <a className="card pl-template pl-compose" href="#/pipelines/new">
          <div className="pl-template-head">
            <Icon name="plus" />
            <strong>직접 조합</strong>
          </div>
          <p className="muted small">가져오기 · 서버로 보내기 · 오토라벨 · 라벨링 · 버전 확정 · 검수 · 학습 · 탐색 · 평가 · 승격 · 내보내기를 골라 순서대로 잇습니다.</p>
        </a>
      </div>
    </>
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
                {showPipeline && <div className="muted small">{r.pipeline ? `파이프라인 ${r.pipeline.name}` : "저장하지 않은 단계"}</div>}
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

// ---- making one: steps put together ------------------------------------------------------

interface Draft {
  key: number;
  kind: StepKind;
  config: Record<string, unknown>;
  open: boolean;
}

interface ComposeData {
  tree: TreeDataset[];
  models: ModelInfo[];
  nodes: NodeInfo[];
  families: Map<string, Set<string>>; // model families with an alias on a version: name → aliases
}

let nextKey = 1;
const draftOf = (kind: StepKind, config: Record<string, unknown>, open = false): Draft => ({
  key: nextKey++,
  kind,
  config: structuredClone(config),
  open,
});

/** 새 파이프라인 (from a template or blank), or a saved one's steps changed. The steps are listed in order; each opens
 * to its settings; between them, whether the next one goes on when this one succeeds or only past its gate. */
function Composer({ templateId = "", pipelineId = "" }: { templateId?: string; pipelineId?: string }) {
  const { projects, projectId } = useProject();
  const [data, setData] = useState<ComposeData | null>(null);
  const [templates, setTemplates] = useState<PipelineTemplate[] | null>(null);
  const [editing, setEditing] = useState<PipelineView | null>(null);
  const [datasetId, setDatasetId] = useState("");
  const [steps, setSteps] = useState<Draft[]>([]);
  const [name, setName] = useState("");
  const [trigger, setTrigger] = useState<"manual" | "new_version">("manual");
  const [runNow, setRunNow] = useState(true);
  const [keep, setKeep] = useState(false);
  const [keepName, setKeepName] = useState("");
  const [keepNote, setKeepNote] = useState("");
  const [from, setFrom] = useState<PipelineTemplate | null>(null);
  const [error, setError] = useState<string | null>(null);
  const [busy, setBusy] = useState(false);
  const [adding, setAdding] = useState(false);
  const project = projects?.find((p) => p.id === (editing?.project_id ?? projectId));

  useEffect(() => {
    void api<PipelineTemplate[]>("/pipeline-templates").then(setTemplates);
    if (pipelineId) void api<PipelineView>(`/pipelines/${pipelineId}`).then(setEditing);
  }, [pipelineId]);
  const owner = editing?.project_id ?? projectId;
  useEffect(() => {
    if (!owner) return;
    void Promise.all([
      api<TreeDataset[]>(`/projects/${owner}/datasets`),
      api<ModelInfo[]>(`/projects/${owner}/models`),
      api<NodeInfo[]>("/nodes"),
    ]).then(([tree, models, nodes]) => {
      const families = new Map<string, Set<string>>();
      for (const m of models) for (const a of m.aliases) families.set(m.name, (families.get(m.name) ?? new Set()).add(a));
      setData({ tree, models, nodes, families });
    });
  }, [owner]);

  // Where it starts: the pipeline being changed, a template, or nothing
  useEffect(() => {
    if (!data) return;
    if (editing) {
      setDatasetId(editing.dataset?.id ?? "");
      setSteps(editing.steps.map((s) => draftOf(s.kind, s.config)));
      setName(editing.name);
      setTrigger(editing.trigger);
      return;
    }
    if (!templates) return;
    const picked = templates.find((t) => t.id === templateId) ?? null;
    setFrom(picked);
    const family = [...data.families.entries()].find(([, aliases]) => aliases.has(DEFAULT_ALIAS))?.[0] ?? [...data.families.keys()][0] ?? "";
    setSteps(
      (picked?.steps ?? []).map((s, i) => {
        const config = structuredClone(s.config);
        const model = config.model as { name?: string; alias?: string } | undefined;
        if (model && !model.name) config.model = { ...model, name: family }; // the blank a built-in leaves
        return draftOf(s.kind, config, i === 0 && (s.kind === "sweep" || s.kind === "import"));
      }),
    );
    setTrigger(picked?.trigger ?? "manual");
    const withVersions = data.tree.filter((d) => d.releases.some((r) => r.status === "ready" && !r.deleted_at));
    setDatasetId((current) => current || withVersions[0]?.id || "");
  }, [data, templates, editing, templateId]);

  const dataset = data?.tree.find((d) => d.id === datasetId);
  const newest = dataset?.releases.filter((r) => r.status === "ready" && !r.deleted_at).sort((a, b) => b.created_at.localeCompare(a.created_at))[0];
  const task = dataset?.task ?? project?.tasks?.[0] ?? "detect";
  const update = (key: number, change: Partial<Draft>) => setSteps((all) => all.map((s) => (s.key === key ? { ...s, ...change } : s)));
  const move = (index: number, by: number) =>
    setSteps((all) => {
      const next = [...all];
      const [taken] = next.splice(index, 1);
      next.splice(Math.max(0, Math.min(next.length, index + by)), 0, taken);
      return next;
    });
  const add = (kind: StepKind) => {
    const config = structuredClone(DEFAULT_CONFIG[kind]);
    const family = [...(data?.families.entries() ?? [])].find(([, a]) => a.has(DEFAULT_ALIAS))?.[0] ?? [...(data?.families.keys() ?? [])][0];
    if (USES_MODEL.includes(kind) && !makesModelBefore(steps, steps.length) && family) {
      config.model = { name: family, alias: DEFAULT_ALIAS }; // nothing makes a model before it: the one holding SOTA
    }
    if (kind === "import") {
      const stores = (data?.nodes ?? []).filter(isDataServer); // 데이터셋 가져오기's: data servers, one that answers first
      const first = stores.find((n) => n.status === "online") ?? stores[0];
      if (first) config.node_id = first.id;
    }
    setSteps((all) => [...all, draftOf(kind, config, true)]);
    setAdding(false);
  };
  const body = () => steps.map((s) => ({ kind: s.kind, config: s.config }));
  const [places, setPlaces] = useState<Record<string, Places>>({});
  const placeKeys = [...new Set(steps.filter((s) => PLACE_KINDS.includes(s.kind) && datasetId).map((s) => placeKey(datasetId, s.kind, s.config)))];
  useEffect(() => {
    for (const key of placeKeys) {
      if (places[key]) continue;
      const [dataset, kind, name, alias] = key.split("|");
      const query = new URLSearchParams({ dataset_id: dataset, kind, model_name: name, model_alias: alias });
      void api<Places>(`/pipeline-places?${query}`)
        .then((found) => setPlaces((was) => ({ ...was, [key]: found })))
        .catch((e) => setPlaces((was) => ({ ...was, [key]: { default: "", nodes: [], problem: e instanceof Error ? e.message : String(e) } })));
    }
  }, [placeKeys.join(",")]); // eslint-disable-line react-hooks/exhaustive-deps
  const problems = flowProblems(steps, data?.tree.find((d) => d.id === datasetId), places);
  const blocked = problems.some((p) => p.length > 0);

  const save = async () => {
    setBusy(true);
    setError(null);
    try {
      if (editing) {
        await api(`/pipelines/${editing.id}`, { method: "PATCH", body: { name, trigger, steps: body() } });
        window.location.hash = `#/pipelines/p/${editing.id}`;
        return;
      }
      if (keep) {
        await api("/pipeline-templates", { body: { name: keepName || name, description: keepNote, steps: body(), trigger } });
      }
      const made = await api<PipelineView & { started: PipelineRunView | null }>("/pipelines", {
        body: { dataset_id: datasetId, name, steps: body(), trigger, run: runNow },
      });
      window.location.hash = made.started ? `#/pipelines/run/${made.started.id}` : `#/pipelines/p/${made.id}`;
    } catch (e) {
      setError(e instanceof Error ? e.message : String(e));
      setBusy(false);
    }
  };
  const keepOnly = async () => {
    setBusy(true);
    setError(null);
    try {
      await api("/pipeline-templates", { body: { name: keepName || name, description: keepNote, steps: body(), trigger } });
      window.location.hash = "#/pipelines";
    } catch (e) {
      setError(e instanceof Error ? e.message : String(e));
      setBusy(false);
    }
  };

  if (!data || (!editing && pipelineId) || !templates) return <div className="muted">불러오는 중…</div>;
  return (
    <>
      <nav className="page-crumbs">
        <a href="#/pipelines">파이프라인</a>
        {editing ? <a href={`#/pipelines/p/${editing.id}`}>{editing.name}</a> : null}
        <strong>{editing ? "단계 고치기" : "새 파이프라인"}</strong>
      </nav>
      <div className="page-head">
        <div>
          <h1>{editing ? `${editing.name} 단계 고치기` : "새 파이프라인"}</h1>
          <p className="muted">
            단계를 순서대로 놓습니다. 단계는 앞 단계가 성공하면 시작하고, 게이트(승격 · 기준을 둔 단계)는 기준을 넘어야 다음으로 갑니다. 단계를 누르면
            설정이 열립니다. 돌 수 없는 순서는 그 단계 아래에 이유가 나오고, 고치기 전에는 저장되지 않습니다.
          </p>
        </div>
      </div>

      <div className="card pl-compose-card">
        <div className="form-row">
          {!editing && (
            <label>
              시작
              <select value={from?.id ?? ""} onChange={(e) => (window.location.hash = e.target.value ? `#/pipelines/new/${e.target.value}` : "#/pipelines/new")}>
                <option value="">빈 파이프라인</option>
                {templates.map((t) => (
                  <option key={t.id} value={t.id}>
                    템플릿: {t.name}
                  </option>
                ))}
              </select>
            </label>
          )}
          <label>
            데이터셋
            <select value={datasetId} onChange={(e) => setDatasetId(e.target.value)} disabled={!!editing}>
              {data.tree.map((d) => (
                <option key={d.id} value={d.id}>
                  {d.name} · {d.releases.filter((r) => r.status === "ready" && !r.deleted_at).length}개 버전
                </option>
              ))}
            </select>
          </label>
        </div>
        {from?.description && <p className="muted small pl-from">{from.description}</p>}

        <ol className="pl-compose-steps">
          {steps.map((s, i) => (
            <li key={s.key}>
              <div className={`pl-compose-step${s.open ? " open" : ""}${problems[i]?.length ? " bad" : ""}`}>
                <button type="button" className="pl-compose-head" onClick={() => update(s.key, { open: !s.open })} aria-expanded={s.open}>
                  <span className="pl-spec-n">{i + 1}</span>
                  <strong>{KIND_LABEL[s.kind]}</strong>
                  {isGate(s.kind, s.config) && <span className="tag">게이트</span>}
                  <span className="muted small ellipsis">{describe(s, steps, i, data)}</span>
                </button>
                <span className="pl-compose-tools">
                  <button type="button" className="icon-button" disabled={i === 0} onClick={() => move(i, -1)} title="위로" aria-label="위로">
                    ↑
                  </button>
                  <button type="button" className="icon-button" disabled={i === steps.length - 1} onClick={() => move(i, 1)} title="아래로" aria-label="아래로">
                    ↓
                  </button>
                  <button type="button" className="icon-button" onClick={() => setSteps(steps.filter((x) => x.key !== s.key))} title="빼기" aria-label="빼기">
                    <Icon name="x" size={13} />
                  </button>
                </span>
                {problems[i]?.map((p) => (
                  <div key={p} className="pl-problem">
                    <Icon name="alert" size={13} /> {p}
                  </div>
                ))}
                {s.open && (
                  <StepEditor
                    step={s}
                    index={i}
                    steps={steps}
                    data={data}
                    task={task}
                    projectId={owner}
                    defaultTemplate={project?.settings?.template}
                    releaseId={newest?.id}
                    dataset={dataset}
                    places={datasetId ? places[placeKey(datasetId, s.kind, s.config)] : undefined}
                    onChange={(config) => update(s.key, { config })}
                  />
                )}
              </div>
              {i < steps.length - 1 && <div className="pl-link">{isGate(s.kind, s.config) ? "기준을 넘으면" : "성공하면"} ↓</div>}
            </li>
          ))}
        </ol>
        {steps.length === 0 && <p className="muted">아직 단계가 없습니다. 아래에서 단계를 넣으세요.</p>}
        <div className="pl-add">
          <button type="button" onClick={() => setAdding(!adding)} aria-expanded={adding}>
            <Icon name="plus" size={13} /> 단계 넣기
          </button>
          {adding && (
            <div className="pl-add-menu">
              {STEP_KINDS.map((g) => (
                <div key={g.group}>
                  <div className="eyebrow">{g.group}</div>
                  {g.kinds.map((k) => (
                    <button type="button" key={k.kind} className="pl-add-item" onClick={() => add(k.kind)}>
                      <strong>{k.label}</strong>
                      <span className="muted small">{k.what}</span>
                    </button>
                  ))}
                </div>
              ))}
            </div>
          )}
        </div>
      </div>

      <div className="card pl-compose-card">
        <div className="form-row">
          <label>
            파이프라인 이름
            <input value={name} onChange={(e) => setName(e.target.value)} placeholder={`자동: ${dataset?.name ?? "데이터셋"} ${steps.map((s) => KIND_LABEL[s.kind]).join(" → ")}`} />
          </label>
          <label>
            언제 돌까
            <select value={trigger} onChange={(e) => setTrigger(e.target.value as "manual" | "new_version")}>
              <option value="manual">직접 실행할 때만</option>
              <option value="new_version">이 데이터셋에 새 버전이 생기면 자동으로</option>
            </select>
          </label>
        </div>
        {!editing && (
          <div className="pl-checks">
            <div className="pl-check">
              <input id="cp-now" type="checkbox" checked={runNow} onChange={(e) => setRunNow(e.target.checked)} />
              <label htmlFor="cp-now">만들고 최신 버전{newest ? `(${newest.version})` : ""}으로 한 번 실행</label>
            </div>
            <div className="pl-check">
              <input id="cp-keep" type="checkbox" checked={keep} onChange={(e) => setKeep(e.target.checked)} />
              <label htmlFor="cp-keep">템플릿으로도 등록</label>
              <span className="pl-hint">다른 데이터셋 · 다른 사람도 새 파이프라인과 새 학습의 "학습 끝나면"에서 고를 수 있습니다</span>
            </div>
            {keep && (
              <div className="form-row pl-sub">
                <label>
                  템플릿 이름
                  <input value={keepName} onChange={(e) => setKeepName(e.target.value)} placeholder={name || "필수"} />
                </label>
                <label>
                  설명
                  <input value={keepNote} onChange={(e) => setKeepNote(e.target.value)} placeholder="무엇을 하는지 한 줄로" />
                </label>
              </div>
            )}
          </div>
        )}
        {blocked && <div className="error">이유가 적힌 단계를 고치면 저장할 수 있습니다.</div>}
        {error && <div className="error">{error}</div>}
        <div className="modal-actions">
          <a className="btn" href={editing ? `#/pipelines/p/${editing.id}` : "#/pipelines"}>
            취소
          </a>
          {!editing && keep && (
            <button type="button" disabled={busy || blocked || steps.length === 0 || !(keepName || name)} onClick={() => void keepOnly()}>
              템플릿으로만 등록
            </button>
          )}
          <button type="button" className="primary" disabled={busy || blocked || steps.length === 0 || !datasetId} onClick={() => void save()}>
            {editing ? "저장" : runNow ? "만들고 실행" : "만들기"}
          </button>
        </div>
      </div>
    </>
  );
}

/** A step in a few words, as it is set (the Hub says it again once saved). */
function describe(s: Draft, steps: Draft[], index: number, data: ComposeData): string {
  const c = s.config;
  const model = c.model as { name: string; alias: string } | undefined;
  const who = model ? `${model.name} ${model.alias}` : makesModelBefore(steps, index) ? "앞 단계의 모델" : "모델을 고르세요";
  const node = (id: unknown) => data.nodes.find((n) => n.id === id)?.name ?? "서버를 고르세요";
  const where = c.node_id ? ` · ${node(c.node_id)}에서` : "";
  const version = `(${c.version ? String(c.version) : "다음 번호"})`;
  const floor = c.at_least !== undefined && c.at_least !== "" ? ` · ${c.at_least} 이상이면 다음` : "";
  switch (s.kind) {
    case "import":
      return `${node(c.node_id)}: ${String(c.path || "폴더를 고르세요")} → 새 버전 ${version}`;
    case "review":
      return `${who} · 지금 버전의 ${((c.splits as string[] | undefined) ?? []).join(" · ") || "모든 이미지"}${where}`;
    case "check":
      return `${CHECK_GROUPS.find(([g]) => g === (c.group ?? "any"))?.[1] ?? "문제"} 이미지 ${Number(c.max_flagged ?? 0)}장 이하면 다음`;
    case "sync":
      return `${node(c.node_id)}(으)로`;
    case "autolabel":
      return `${who} · 라벨 안 한 이미지${c.accept != null && c.accept !== "" ? ` · 확신도 ${c.accept} 이상 자동 승인` : ""}${where}`;
    case "label": {
      if (setMaker(steps, index) === "autolabel") return "오토라벨 묶음을 사람이 끝낼 때까지 기다림";
      const parts = c.divide === "one" ? "한 사람에게 모두" : `일감 하나에 ${Number(c.part_size ?? 50)}장`;
      return `${parts}${c.review ? " · 검수하기" : ""} · 사람이 끝낼 때까지 기다림`;
    }
    case "confirm":
      return `작업 중을 새 버전으로 ${version}`;
    case "train": {
      const params = (c.params ?? {}) as Record<string, unknown>;
      const variant = params.model ? String(params.model).replace(/\.pt$/, "") : "";
      const parts = [String(c.template ?? "프로젝트 기본 모델 종류"), variant, params.epochs ? `epoch ${params.epochs}` : ""];
      return parts.filter(Boolean).join(" · ") + floor;
    }
    case "sweep": {
      const space = Object.keys((c.space ?? {}) as Record<string, unknown>);
      return `${space.length ? `${space.join(", ")} · ${c.max_runs ?? 10}번` : "시험할 설정을 고르세요"}${floor}`;
    }
    case "eval": {
      const fixed = data.tree.flatMap((d) => d.releases.map((r) => ({ d, r }))).find((x) => x.r.id === c.release_id);
      const at = fixed ? `${fixed.d.name} ${fixed.r.version}` : model ? "그 버전" : "학습한 버전";
      const split = SPLIT_CHOICES.find(([v]) => v === (c.split ?? "auto"))?.[1] ?? String(c.split);
      return `${model ? `${who} → ` : ""}${at} · ${split}${c.review ? " · 검수에도" : ""}${where}${floor}`;
    }
    case "promote":
      return `더 나으면 ${String(c.alias || DEFAULT_ALIAS)}`;
    case "export":
      return String(c.format ?? "onnx").toUpperCase() + where;
  }
  return "";
}

/** One step's settings, the few that matter first. */
function StepEditor({
  step,
  index,
  steps,
  data,
  task,
  projectId,
  defaultTemplate,
  releaseId,
  dataset,
  places,
  onChange,
}: {
  step: Draft;
  index: number;
  steps: Draft[];
  data: ComposeData;
  task: string;
  projectId: string;
  defaultTemplate?: string;
  releaseId?: string;
  dataset?: TreeDataset;
  places?: Places; // the servers this step may run on (가져오기 · 보내기 · 오토라벨 · 검수 · 평가 · 내보내기)
  onChange: (config: Record<string, unknown>) => void;
}) {
  const c = step.config;
  const set = (part: Record<string, unknown>) => {
    const next = { ...c, ...part };
    for (const [k, v] of Object.entries(part)) if (v === undefined) delete next[k];
    onChange(next);
  };
  const [settings, setSettings] = useState<StepSettings | null>(null);
  const [browsing, setBrowsing] = useState(false); // 가져오기: picking the folder on the server
  const [layout, setLayout] = useState(""); // what the folder picked looks like
  const modelBefore = makesModelBefore(steps, index);
  // The server a model step runs on: 자동 is where the Hub would put it; a server without the images gets them first.
  const placeSelect = (auto: string) => (
    <PlaceSelect label="실행할 서버" places={places} value={String(c.node_id ?? "")} auto={auto} onChange={(id) => set({ node_id: id || undefined })} />
  );
  const placeHint = c.node_id ? <span className="pl-hint pl-lead">고른 서버에 이미지가 없으면 먼저 받은 뒤 합니다.</span> : null;
  // 가져오기 · 버전 확정: the new version's name (empty: the next vN)
  const versionField = (
    <VersionField
      key={step.key}
      value={String(c.version ?? "")}
      next={nextVersion(dataset)}
      onChange={(version) => set({ version: version || undefined })}
    />
  );
  const metric = METRIC_LABEL[keyMetrics(task)[0]] ?? "주 지표";
  const nodeSelect = (
    <PlaceSelect
      label={step.kind === "import" ? "데이터가 있는 서버" : "보낼 서버"}
      places={places}
      value={String(c.node_id ?? "")}
      onChange={(id) => set({ node_id: id || undefined })}
      brief
    />
  );
  const floor = (what: string) => (
    <label>
      기준 (선택)
      <input
        type="number"
        step="0.01"
        min="0"
        value={c.at_least === undefined || c.at_least === null ? "" : String(c.at_least)}
        onChange={(e) => set({ at_least: e.target.value === "" ? undefined : Number(e.target.value) })}
        placeholder={`${what}의 ${metric}이 이 값 이상이어야 다음으로`}
      />
    </label>
  );
  const modelPick = (required: boolean) => {
    const model = c.model as { name: string; alias: string } | undefined;
    const options = [...data.families.entries()].flatMap(([name, aliases]) => [...aliases].map((alias) => ({ name, alias, held: true })));
    if (model && !options.some((o) => o.name === model.name && o.alias === model.alias)) options.unshift({ ...model, held: false });
    return (
      <label>
        모델
        <select
          value={model ? `${model.name}\u0000${model.alias}` : ""}
          onChange={(e) => {
            const [name, alias] = e.target.value.split("\u0000");
            set({ model: e.target.value ? { name, alias } : undefined });
          }}
        >
          {!required && <option value="">앞 단계의 모델</option>}
          {required && !model && <option value="">고르세요</option>}
          {options.map((o) => (
            <option key={`${o.name}-${o.alias}`} value={`${o.name}\u0000${o.alias}`}>
              {o.name} · 지금 {o.alias}인 버전{o.held ? "" : " (아직 없음)"}
            </option>
          ))}
        </select>
        {required && options.length === 0 && <span className="pl-hint pl-lead">alias(SOTA 등)가 붙은 모델이 없습니다. 앞에 학습이나 탐색을 넣으세요.</span>}
      </label>
    );
  };

  let fields: ReactNode = null;
  switch (step.kind) {
    case "import": {
      const node = data.nodes.find((n) => n.id === c.node_id);
      if (browsing && node) {
        fields = (
          <FolderBrowser
            nodeId={node.id}
            nodeName={node.name}
            start={String(c.path ?? "")}
            onCancel={() => setBrowsing(false)}
            onPick={(path, guess) => {
              set({ path });
              setLayout(guess);
              setBrowsing(false);
            }}
          />
        );
        break;
      }
      fields = (
        <>
          <div className="form-row">{nodeSelect}</div>
          <div className="field-label">
            폴더 경로 <span className="muted small">(그 서버에서의 경로)</span>
            <div className="input-with-button">
              <input
                value={String(c.path ?? "")}
                onChange={(e) => {
                  setLayout("");
                  set({ path: e.target.value });
                }}
                placeholder="/data/street 또는 D:\data\street"
                className="mono"
                aria-label="폴더 경로"
              />
              <button type="button" className="small" disabled={!node || node.status !== "online"} onClick={() => setBrowsing(true)}>
                <Icon name="folder" size={13} /> 찾아보기
              </button>
            </div>
            {layout && <span className="muted small field-hint">{LAYOUT_NAME[layout] ?? layout} 형식으로 보입니다. 형식은 자동으로 두면 됩니다.</span>}
            {node && node.status !== "online" && <span className="muted small field-hint">서버가 꺼져 있어 폴더를 볼 수 없습니다. 경로를 직접 적으세요.</span>}
          </div>
          {versionField}
          <div className="form-row">
            <label>
              형식
              <select value={String(c.format ?? "auto")} onChange={(e) => set({ format: e.target.value })}>
                <option value="auto">자동 감지</option>
                <option value="yolo">YOLO</option>
                <option value="coco">COCO</option>
                <option value="imagefolder">클래스별 폴더</option>
              </select>
            </label>
          </div>
          <details className="advanced" open={(c.unlabelled ?? "auto") !== "auto" || Boolean(c.classes)}>
            <summary>
              라벨이 없는 이미지 <span className="muted">(라벨할 이미지를 가져올 때)</span>
            </summary>
            <label>
              라벨 파일이 없는 이미지
              <select value={String(c.unlabelled ?? "auto")} onChange={(e) => set({ unlabelled: e.target.value })}>
                <option value="auto">자동 · 폴더에 라벨 파일이 하나도 없으면 아직 안 함, 아니면 배경</option>
                <option value="background">배경 · 찾을 물체가 없는 이미지</option>
                <option value="unlabelled">아직 안 함 · 라벨할 이미지</option>
              </select>
            </label>
            <label>
              클래스 <span className="muted small">(클래스 이름이 없는 폴더에)</span>
              <input
                value={Array.isArray(c.classes) ? (c.classes as string[]).join(", ") : String(c.classes ?? "")}
                onChange={(e) => set({ classes: e.target.value || undefined })}
                placeholder="예: car, person, bike"
              />
            </label>
            <p className="muted small">비우면 이 데이터셋의 클래스를 씁니다. "아직 안 함" 이미지는 학습에서 빠지고, 라벨링 묶음으로 라벨하면 들어갑니다.</p>
          </details>
          <label>
            메모 (선택)
            <input value={String(c.note ?? "")} maxLength={500} onChange={(e) => set({ note: e.target.value || undefined })} placeholder="이 버전에서 달라진 점" />
          </label>
          <div className="full-read">
            <label className="check inline-check">
              <input type="checkbox" checked={!!c.full_read} onChange={(e) => set({ full_read: e.target.checked || undefined })} />
              바뀌지 않은 파일도 모두 다시 읽기
            </label>
            <p className="muted small">
              {c.full_read
                ? "폴더의 이미지를 전부 다시 읽습니다. 파일 내용이 바뀌었는데 수정 시각이 그대로일 때만 필요합니다."
                : "이 프로젝트에 예전에 등록한 것과 경로 · 크기 · 수정 시각이 같은 이미지는 다시 읽지 않습니다(라벨은 새로 읽음)."}
            </p>
          </div>
        </>
      );
      break;
    }
    case "review": {
      const chosen = (c.splits as string[] | undefined) ?? [];
      const on = (k: string) => chosen.length === 0 || chosen.includes(k);
      const toggle = (k: string) => {
        const next = ["train", "val", "test"].filter((x) => (x === k ? !on(x) : on(x)));
        if (next.length > 0) set({ splits: next.length === 3 ? undefined : next });
      };
      fields = (
        <>
          <p className="muted small">
            검수의 "모델로 검사하기"와 같습니다. 모델이 고른 부분의 이미지를 예측하고, 검수가 그것으로 라벨이 없는 물체 · 클래스가 다를 수 있는 라벨 ·
            물체가 없을 수 있는 라벨을 찾습니다. 파이프라인은 멈추지 않고 다음 단계로 갑니다.
          </p>
          <div className="form-row">
            {modelPick(!modelBefore)}
            {placeSelect("이미지가 있는 서버")}
          </div>
          {placeHint}
          <fieldset className="eval-splits">
            <legend>평가할 부분</legend>
            {["train", "val", "test"].map((k) => (
              <label key={k} className="check-line">
                <input type="checkbox" checked={on(k)} onChange={() => toggle(k)} />
                {k}
              </label>
            ))}
            <span className="muted small">그 버전에 있는 부분만 차례로 평가해 한 기준으로 봅니다</span>
          </fieldset>
        </>
      );
      break;
    }
    case "check":
      fields = (
        <div className="form-row">
          <label>
            걸린 이미지가 몇 장 이하면 다음으로
            <input type="number" min="0" value={Number(c.max_flagged ?? 0)} onChange={(e) => set({ max_flagged: Number(e.target.value) })} />
          </label>
          <label>
            무엇이 걸린 이미지
            <select value={String(c.group ?? "any")} onChange={(e) => set({ group: e.target.value })}>
              {CHECK_GROUPS.map(([g, label]) => (
                <option key={g} value={g}>
                  {label}
                </option>
              ))}
            </select>
          </label>
        </div>
      );
      break;
    case "sync":
      fields = (
        <>
          <p className="muted small">
            이 버전의 이미지 파일을 고른 서버에 미리 복사합니다(그 서버에 이미 있으면 그대로 넘어감). 학습 단계에서 "노드 고르기"를 하면 그 노드가
            학습 전에 알아서 받으므로, 보통은 넣지 않아도 됩니다.
          </p>
          <div className="form-row">{nodeSelect}</div>
        </>
      );
      break;
    case "autolabel": {
      const accepting = c.accept !== undefined && c.accept !== null && c.accept !== "";
      const keep = c.keep === undefined || c.keep === null || c.keep === "" ? 0.25 : Number(c.keep);
      fields = (
        <>
          <p className="muted small">
            새 오토라벨의 "데이터셋의 라벨 안 한 이미지"와 같습니다. 그때의 버전에서 라벨 안 한 이미지로 라벨링 묶음을 만들고, 모델이 라벨을 먼저
            그립니다. 라벨은 데이터셋의 작업 중에 모입니다.
          </p>
          <div className="form-row">
            {modelPick(!modelBefore)}
            <label>
              split
              <select value={String(c.split ?? "")} onChange={(e) => set({ split: e.target.value || undefined })}>
                <option value="">전체</option>
                <option value="train">train</option>
                <option value="val">val</option>
                <option value="test">test</option>
              </select>
            </label>
          </div>
          <SetFields c={c} set={set} id={step.key} name={`${dataset?.name ?? "데이터셋"} 오토라벨 날짜`} />
          <div className="form-row">
            {placeSelect("이미지가 있는 서버")}
            <label>
              계산
              <select value={c.gpus === 0 ? "cpu" : "gpu"} onChange={(e) => set({ gpus: e.target.value === "cpu" ? 0 : undefined })}>
                <option value="gpu">GPU (서버에 있으면)</option>
                <option value="cpu">CPU</option>
              </select>
            </label>
          </div>
          {placeHint}
          <div className="form-row">
            <label>
              최소 확신도
              <input type="number" min={0.05} max={0.95} step={0.05} value={keep} onChange={(e) => set({ keep: Number(e.target.value) || 0.25 })} />
              <span className="muted small">이보다 낮은 예측은 버립니다</span>
            </label>
            <label>
              <span className="check">
                <input type="checkbox" checked={accepting} onChange={(e) => set({ accept: e.target.checked ? 0.8 : undefined })} /> 자동 승인
              </span>
              <input
                type="number"
                min={keep}
                max={1}
                step={0.05}
                value={accepting ? Number(c.accept) : 0.8}
                disabled={!accepting}
                onChange={(e) => set({ accept: Number(e.target.value) || 0.8 })}
              />
              <span className="muted small">예측이 모두 이 확신도 이상인 이미지는 바로 저장해 {c.review ? "검수로" : "완료로"} 보냅니다</span>
            </label>
          </div>
        </>
      );
      break;
    }
    case "label": {
      const fromAutolabel = setMaker(steps, index) === "autolabel";
      fields = fromAutolabel ? (
        <p className="muted small">
          앞의 오토라벨이 만든 묶음을 사람이 끝낼 때까지 기다립니다. 묶음 이름 · 일감 · 검수하기는 오토라벨 단계에서 정합니다.
        </p>
      ) : (
        <>
          <p className="muted small">
            라벨링 묶음 만들기와 같습니다. 그때의 버전에서 라벨 안 한 이미지로 묶음을 만들어 일감으로 나누고, 사람이 끝낼 때까지 기다립니다.
          </p>
          <div className="form-row">
            <label>
              split
              <select value={String(c.split ?? "")} onChange={(e) => set({ split: e.target.value || undefined })}>
                <option value="">전체</option>
                <option value="train">train</option>
                <option value="val">val</option>
                <option value="test">test</option>
              </select>
            </label>
          </div>
          <SetFields c={c} set={set} id={step.key} name={`${dataset?.name ?? "데이터셋"} 라벨링 날짜`} />
          <label>
            지시문 <span className="muted small">(선택 · 라벨링하는 사람에게 보임)</span>
            <textarea
              rows={2}
              maxLength={2000}
              value={String(c.instructions ?? "")}
              onChange={(e) => set({ instructions: e.target.value || undefined })}
              placeholder="예: 사람과 차를 박스로, 30픽셀보다 작은 것은 빼기"
            />
          </label>
        </>
      );
      break;
    }
    case "confirm":
      fields = (
        <>
          <p className="muted small">
            버전으로 확정과 같습니다. 작업 중에 고친 이미지가 있으면 새 버전으로 확정하고, 그다음 단계는 그 버전으로 합니다. 고친 것이 없으면 그대로
            넘어갑니다.
          </p>
          {versionField}
          <label>
            메모
            <textarea
              rows={2}
              maxLength={500}
              value={String(c.note ?? "")}
              onChange={(e) => set({ note: e.target.value || undefined })}
              placeholder="비우면 &quot;파이프라인 (이름)에서 확정&quot;"
            />
          </label>
        </>
      );
      break;
    case "train":
    case "sweep":
      fields = (
        <>
          <div className="pl-check">
            <button type="button" className="small" onClick={() => setSettings({ kind: step.kind as "train" | "sweep", config: c, onApply: (next) => {
              onChange({ ...next, ...(c.at_least !== undefined ? { at_least: c.at_least } : {}) });
              setSettings(null);
            } })}>
              {step.kind === "sweep" ? "탐색 설정 열기" : "학습 설정 열기"}
            </button>
            <span className="pl-hint pl-lead">새 학습 창과 같은 설정입니다. 열지 않으면 프로젝트 기본 모델 종류의 기본값으로 합니다.</span>
          </div>
          <div className="form-row">{floor(step.kind === "sweep" ? "가장 좋은 실행" : "학습한 모델(검증)")}</div>
        </>
      );
      break;
    case "eval": {
      const fixed = data.tree.filter((d) => d.task === task);
      fields = (
        <>
          <div className="form-row">
            {modelBefore ? null : modelPick(true)}
            <label>
              버전
              <select value={String(c.release_id ?? "")} onChange={(e) => set({ release_id: e.target.value })}>
                <option value="">{modelBefore ? "학습한 버전" : "그 버전 (실행하는 버전)"}</option>
                {fixed.map((d) => (
                  <optgroup key={d.id} label={d.name}>
                    {d.releases
                      .filter((r) => r.status === "ready" && !r.deleted_at)
                      .map((r) => (
                        <option key={r.id} value={r.id}>
                          {d.name} {r.version} (고정)
                        </option>
                      ))}
                  </optgroup>
                ))}
              </select>
            </label>
            <label>
              평가할 부분
              <select value={String(c.split ?? "auto")} onChange={(e) => set({ split: e.target.value })}>
                {SPLIT_CHOICES.map(([v, label]) => (
                  <option key={v} value={v}>
                    {label}
                  </option>
                ))}
              </select>
            </label>
          </div>
          <div className="form-row">{placeSelect("이미지가 있는 서버")}</div>
          {placeHint}
          <div className="pl-check">
            <input id={`ev-${step.key}`} type="checkbox" checked={!!c.review} onChange={(e) => set({ review: e.target.checked })} />
            <label htmlFor={`ev-${step.key}`}>검수에도 쓰기</label>
            <span className="pl-hint">그 버전의 모든 split을 같은 평가에 넣어 검수가 틀린 라벨 후보를 찾습니다</span>
          </div>
          <div className="form-row">{floor("평가")}</div>
        </>
      );
      break;
    }
    case "promote":
      fields = (
        <div className="form-row">
          <label>
            alias
            <input value={String(c.alias ?? DEFAULT_ALIAS)} onChange={(e) => set({ alias: e.target.value })} list="cp-aliases" />
            <datalist id="cp-aliases">
              <option value="SOTA" />
              <option value="production" />
              <option value="staging" />
            </datalist>
          </label>
          <label>
            이만큼 더 나아야 (선택)
            <input
              type="number"
              step="0.001"
              min="0"
              value={c.min_delta ? String(c.min_delta) : ""}
              onChange={(e) => set({ min_delta: e.target.value === "" ? undefined : Number(e.target.value) })}
              placeholder="0: 조금이라도 높으면"
            />
          </label>
        </div>
      );
      break;
    case "export":
      fields = (
        <div className="form-row">
          <label>
            형식
            <select value={String(c.format ?? "onnx")} onChange={(e) => set({ format: e.target.value })}>
              <option value="onnx">ONNX</option>
              <option value="torchscript">TorchScript</option>
            </select>
          </label>
          {modelBefore ? null : modelPick(true)}
          {placeSelect("모델을 학습한 서버")}
        </div>
      );
      break;
  }
  return (
    <div className="pl-compose-body">
      {fields}
      {settings && projectId && (
        <NewRun
          projectId={projectId}
          task={task}
          defaultTemplate={defaultTemplate}
          releaseId={releaseId}
          settingsFor={settings}
          onClose={() => setSettings(null)}
          onStarted={() => setSettings(null)}
          onSweep={() => setSettings(null)}
        />
      )}
    </div>
  );
}

/** A step's server: the ones the Hub listed for it, those that cannot greyed out with why (as 실행할 서버 in the
 * evaluation and 오토라벨 windows); ``auto``: 자동 first (where the Hub would put it), else one must be picked. */
function PlaceSelect({
  label,
  places,
  value,
  auto,
  brief = false,
  onChange,
}: {
  label: string;
  places?: Places;
  value: string;
  auto?: string;
  brief?: boolean; // 가져오기 · 보내기: no GPUs to say
  onChange: (id: string) => void;
}) {
  const nodes = places?.nodes ?? [];
  const fallback = nodes.find((n) => n.id === places?.default)?.name;
  const words = (n: Place) =>
    [
      brief ? "" : n.gpus ? `GPU ${n.gpus}` : "CPU",
      n.holds_images ? "이미지 있음" : n.transfer ? "이미지를 먼저 받음" : "",
      n.trained_here ? "학습한 서버" : "",
      n.status === "online" ? "" : "지금 응답 없음",
    ]
      .filter(Boolean)
      .join(" · ");
  return (
    <label>
      {label}
      <select value={value} onChange={(e) => onChange(e.target.value)}>
        {auto !== undefined ? <option value="">자동 — {fallback ? `${fallback} (${auto})` : auto}</option> : <option value="">고르세요</option>}
        {nodes.map((n) => (
          <option key={n.id} value={n.id} disabled={!n.fits && n.id !== value}>
            {n.name}
            {words(n) ? ` · ${words(n)}` : ""}
            {n.fits ? "" : ` — ${n.why}`}
          </option>
        ))}
      </select>
      {!places && <span className="muted small">서버를 찾는 중…</span>}
      {places?.problem && <span className="bad-text small">{places.problem}</span>}
      {places && !places.problem && nodes.length > 0 && !nodes.some((n) => n.fits) && (
        <span className="bad-text small">이 단계를 할 수 있는 서버가 없습니다. 노드 설정에서 역할을 확인하세요.</span>
      )}
    </label>
  );
}

/** Which step makes the set a labeling step waits on: an 오토라벨 right before it (no 라벨링 · 버전 확정 · 가져오기 between),
 * else it makes its own (as the Hub decides). */
function setMaker(steps: { kind: StepKind }[], index: number): StepKind | undefined {
  return [...steps.slice(0, index)].reverse().find((s) => ["autolabel", "label", "confirm", "import"].includes(s.kind))?.kind;
}

/** A labeling set's settings, as 라벨링 묶음 만들기 and 새 오토라벨 ask for them. */
function SetFields({
  c,
  set,
  id,
  name,
}: {
  c: Record<string, unknown>;
  set: (part: Record<string, unknown>) => void;
  id: number;
  name: string;
}) {
  const one = c.divide === "one";
  return (
    <>
      <div className="form-row">
        <label>
          새 묶음 이름
          <input value={String(c.name ?? "")} maxLength={128} onChange={(e) => set({ name: e.target.value || undefined })} placeholder={`자동: ${name}`} />
        </label>
        <label>
          일감 하나에 이미지
          <input
            type="number"
            min={1}
            disabled={one}
            value={one ? "" : String(c.part_size ?? 50)}
            placeholder={one ? "모두" : "50"}
            onChange={(e) => set({ part_size: e.target.value === "" ? undefined : Math.max(1, Math.floor(Number(e.target.value)) || 1) })}
          />
        </label>
      </div>
      <div className="pl-check">
        <input id={`one-${id}`} type="checkbox" checked={one} onChange={(e) => set({ divide: e.target.checked ? "one" : "size" })} />
        <label htmlFor={`one-${id}`}>한 사람에게 모두 (일감 1개)</label>
      </div>
      <div className="pl-check">
        <input id={`rv-${id}`} type="checkbox" checked={!!c.review} onChange={(e) => set({ review: e.target.checked })} />
        <label htmlFor={`rv-${id}`}>검수하기 (승인 · 돌려보내기)</label>
        <span className="pl-hint">맡길 사람은 묶음이 생긴 뒤 라벨링 보드에서 정합니다</span>
      </div>
    </>
  );
}

/** The name of the version a step makes: the next number, or typed ({n}: the next number, {date}: that day). */
function VersionField({ value, next, onChange }: { value: string; next: string; onChange: (version: string) => void }) {
  const [typed, setTyped] = useState(Boolean(value));
  return (
    <>
      <div className="form-row">
        <label>
          새 버전 이름
          <select
            value={typed ? "typed" : "auto"}
            onChange={(e) => {
              setTyped(e.target.value === "typed");
              if (e.target.value === "auto") onChange("");
            }}
          >
            <option value="auto">자동 — 다음 번호 (지금이면 {next})</option>
            <option value="typed">직접 입력</option>
          </select>
        </label>
        {typed && (
          <label>
            이름
            <input value={value} maxLength={64} onChange={(e) => onChange(e.target.value)} placeholder="예: labeled-{n}" autoFocus />
          </label>
        )}
      </div>
      {typed && (
        <span className="pl-hint pl-lead">
          {"{n}"}은 다음 번호, {"{date}"}는 그날 날짜({today()})로 바뀝니다. 같은 이름의 버전은 만들 수 없으니, 여러 번 돌 파이프라인이면 {"{n}"}을 넣으세요.
        </span>
      )}
    </>
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
  const [keeping, setKeeping] = useState(false);
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
          <a className="btn" href={`#/pipelines/edit/${p.id}`}>
            단계 고치기
          </a>
          <button disabled={busy} onClick={() => setKeeping(true)}>
            템플릿으로 등록
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
              <li key={`${s.kind}-${i}`}>
                <span className="pl-spec-n">{i + 1}</span>
                <strong>{s.label}</strong>
                {s.gate && <span className="tag">게이트</span>}
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
              버전, 가장 최신 하나만 (이 파이프라인이 만든 버전은 빼고)
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
      {keeping && <KeepAsTemplate pipeline={p} onClose={() => setKeeping(false)} />}
    </>
  );
}

function KeepAsTemplate({ pipeline: p, onClose }: { pipeline: PipelineView; onClose: () => void }) {
  const [name, setName] = useState(p.name);
  const [note, setNote] = useState("");
  const [error, setError] = useState<string | null>(null);
  const [busy, setBusy] = useState(false);
  const steps = useMemo(() => p.steps.map((s) => ({ kind: s.kind, config: s.config })), [p]);
  const submit = async () => {
    setBusy(true);
    setError(null);
    try {
      await api("/pipeline-templates", { body: { name, description: note, steps, trigger: p.trigger } });
      window.location.hash = "#/pipelines";
    } catch (e) {
      setError(e instanceof Error ? e.message : String(e));
      setBusy(false);
    }
  };
  return (
    <Modal title="템플릿으로 등록" onClose={onClose}>
      <p className="muted small">이 단계들을 템플릿으로 남깁니다. 다른 데이터셋 · 다른 사람도 새 파이프라인에서, 학습으로 시작하면 새 학습의 "학습 끝나면"에서도 고릅니다. 학습할 클래스는 데이터셋마다 달라 빼고 남깁니다.</p>
      <StepChain steps={p.steps} plain />
      <label>
        템플릿 이름
        <input value={name} onChange={(e) => setName(e.target.value)} />
      </label>
      <label>
        설명
        <input value={note} onChange={(e) => setNote(e.target.value)} placeholder="무엇을 하는지 한 줄로" />
      </label>
      {error && <div className="error">{error}</div>}
      <div className="modal-actions">
        <button type="button" onClick={onClose}>
          취소
        </button>
        <button type="button" className="primary" disabled={busy || !name.trim()} onClick={() => void submit()}>
          등록
        </button>
      </div>
    </Modal>
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
  const heldAt = run.steps.find((s) => s.status === "held");
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
      {heldAt && (
        <div className="notice">
          {heldAt.label}: {heldAt.message}
          {heldAt.kind === "promote" ? " — 새 모델은 등록되어 있고, 모델 화면에서 직접 지정할 수도 있습니다." : ""}
        </div>
      )}
      <ol className="pl-timeline">
        {run.steps.map((s, i) => (
          <StepCard key={`${s.kind}-${i}`} step={s} measured={measuredBefore(run.steps, i)} />
        ))}
      </ol>
    </>
  );
}

/** Where the evaluation a promotion decided on measured. */
const measuredBefore = (steps: PipelineStepView[], index: number) => {
  const e = [...steps.slice(0, index)].reverse().find((s) => s.kind === "eval");
  return e?.release ? `${e.release.dataset} ${e.release.version} · ${e.split}` : "";
};

function StepCard({ step: s, measured }: { step: PipelineStepView; measured: string }) {
  const href = stepHref(s);
  const links: ReactNode[] = [];
  if (s.made_release) links.push(<a key="made" href={`#/datasets/${s.made_release.id}`}>새 버전 {s.made_release.version}</a>);
  if (s.kind === "check" && s.release) links.push(<a key="check" href={`#/review/${s.release.id}`}>검수에서 걸린 이미지 보기</a>);
  if (s.kind === "review" && s.release && s.status === "succeeded")
    links.push(
      <a key="review" href={`#/review/${s.release.id}`}>
        검수에서 틀렸을 수 있는 라벨 보기
      </a>,
    );
  if ((s.kind === "autolabel" || s.kind === "label") && s.set) links.push(<a key="set" href={`#/labeling/${s.set.id}`}>묶음 {s.set.name}</a>);
  if (s.kind === "sweep" && s.sweep) links.push(<a key="sweep" href={`#/training/sweep/${s.sweep.id}`}>탐색 {s.sweep.name}</a>);
  if (s.kind === "train" && s.run) links.push(<a key="run" href={`#/training/${s.run.id}`}>학습 {s.run.name}</a>);
  if (s.model && s.kind !== "promote") links.push(<a key="model" href={`#/models/${s.model.id}`}>모델 {s.model.name} v{s.model.version}</a>);
  if (s.kind === "eval" && s.op_id) links.push(<a key="op" href={`#/models/eval/${s.op_id}`}>평가 결과</a>);
  if (s.kind === "eval" && s.review_release_id && s.status !== "pending" && s.status !== "running")
    links.push(
      <a key="review" href={`#/review/${s.review_release_id}`}>
        검수에서 틀린 라벨 후보 보기
      </a>,
    );
  if (s.kind === "promote" && s.model) links.push(<a key="m" href={`#/models/${s.model.id}`}>새 모델 v{s.model.version}</a>);
  if (s.kind === "promote" && s.holder) links.push(<a key="h" href={`#/models/${s.holder.id}`}>비교한 {s.alias} v{s.holder.version}</a>);
  if (s.kind === "export" && s.op_id) links.push(<a key="ex" href={href}>내려받기</a>);
  let progress = "";
  if (s.status === "running" && s.kind === "train" && s.run && s.run.epochs > 0) progress = `epoch ${s.run.epoch}/${s.run.epochs}`;
  if (s.status === "running" && s.kind === "label" && s.set?.total) progress = `${s.set.done ?? 0} / ${s.set.total}장 끝남 · 사람 차례`;
  if (s.status === "running" && s.kind === "sweep" && s.sweep) progress = `실행 ${s.sweep.started} / ${s.sweep.total} 시작함`;
  return (
    <li className={`pl-tl ${s.status}`}>
      <span className={`pl-dot ${s.status}`} aria-hidden="true">
        <Icon name={STEP_ICON[s.status]} size={14} />
      </span>
      <div className="card pl-tl-card">
        <div className="pl-tl-head">
          <strong>{s.label}</strong>
          {s.gate && <span className="tag">게이트</span>}
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
            <span className="muted">
              {s.before.version}에서 {s.before.value.toFixed(3)}
            </span>
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
