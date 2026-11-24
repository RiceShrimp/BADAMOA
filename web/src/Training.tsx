import { LineageGraph } from "./Lineage";
import { MediaPanel } from "./Media";
import { SweepFields, SweepList, type SweepMethod, type SweepSpec, SweepView, type Trial, parseTrial } from "./Sweeps";
import { AliasChip, Annotations, ListFilter, useDebounced } from "./Annotate";
import { FileList, fileSize } from "./Files";
import {
  FollowBanner,
  FollowFields,
  type FollowSpec,
  type PipelineRunView,
  SaveFields,
  type SaveSpec,
  type StepSettings,
  followSteps,
  rememberFollow,
  rememberedFollow,
} from "./FollowUps";
import { type FormEvent, Fragment, type ReactNode, useEffect, useMemo, useRef, useState } from "react";
import { LineChart } from "./Chart";
import { EpochChart, type EpochSeries } from "./EpochChart";
import { TrainingBoard } from "./Boards";
import { rememberProject } from "./DataBoard";
import { ProjectPicker, useProject } from "./ProjectPicker";
import { RunTable } from "./RunTable";
import { type Job, type LogLine, type MetricHistory, api } from "./api";
import { type LabelClass, type ReleaseInfo, type TreeDataset, announceTrash, taskLabel } from "./catalogTypes";
import {
  ACTIVE,
  type Environment,
  type ClassMetric,
  METRIC_LABEL,
  type ModelInfo,
  type ProjectTags,
  type Run,
  type RunDetail,
  type NodeChoice,
  type NodeChoices,
  type NodeGpu,
  type RunSync,
  STAGE_LABEL,
  type Template,
  type TemplateParam,
  classScore,
  keyMetrics,
  metric,
} from "./trainTypes";
import { Badge, Icon, Modal, STATUS_LABEL, Stat, VIA_LABEL, duration, timeAgo, usePolling } from "./ui";

const MAX_COMPARED = 8; // one palette slot per run
const MAX_EXTRA_CHARTS = 12;
const COMPARE_CHARTS: Record<string, string[]> = {
  detect: ["map50_95", "map50", "precision", "recall", "train_box_loss", "val_box_loss", "train_loss"],
  classify: ["top1", "top5", "f1", "train_loss", "val_loss"],
  segment: ["mask_map50_95", "mask_map50", "map50_95", "map50", "train_seg_loss", "val_seg_loss"],
  pose: ["pose_map50_95", "pose_map50", "map50_95", "map50", "train_pose_loss", "val_pose_loss"],
};

export function Training({ runId, arg }: { runId?: string; arg?: string }) {
  const { projects, projectId, choose } = useProject();
  const routed = runId === "project" ? (arg ?? "") : ""; // #/training/project/<id>: that project's runs
  useEffect(() => {
    if (routed) rememberProject(routed); // other screens open on it too
  }, [routed]);

  if (projects && projects.length === 0) {
    return (
      <>
        <div className="page-head">
          <div>
            <h1>학습</h1>
            <p className="muted">데이터셋 버전을 골라 모델을 학습하고, 실행끼리 지표를 비교합니다.</p>
          </div>
        </div>
        <div className="card empty-state">
          <Icon name="chart" size={28} />
          <strong>먼저 데이터셋을 등록하세요</strong>
          <p className="muted">학습은 프로젝트의 데이터셋 버전에서 시작합니다. 프로젝트를 만들고 데이터셋을 가져온 뒤 다시 오세요.</p>
          <a className="btn primary-link" href="#/datasets">
            데이터셋으로 <Icon name="arrow" size={14} />
          </a>
        </div>
      </>
    );
  }
  if (runId === "sweep" && arg) return <SweepView key={arg} sweepId={arg} />;
  if (routed)
    return (
      <RunList
        projects={projects}
        projectId={projects?.some((p) => p.id === routed) ? routed : ""}
        onProject={(id) => {
          choose(id);
          window.location.hash = `#/training/project/${id}`;
        }}
        crumbs
      />
    );
  if (runId === "new") return <RunList projects={projects} projectId={projectId} onProject={choose} startWith={arg ?? ""} />;
  if (runId) return <RunView key={runId} runId={runId} />;
  return <TrainingBoard />;
}

// ---- list + comparison ------------------------------------------------------------------

function RunList({
  projects,
  projectId,
  onProject,
  startWith,
  crumbs = false,
}: {
  projects: ReturnType<typeof useProject>["projects"];
  projectId: string;
  onProject: (id: string) => void;
  startWith?: string;
  crumbs?: boolean; // opened from the first screen: the way back to it
}) {
  const [busy, setBusy] = useState(true);
  const [typed, setTyped] = useState("");
  const [tag, setTag] = useState("");
  const query = useDebounced(typed.trim());
  const filtered = query !== "" || tag !== "";
  const { data: runs, refresh } = usePolling(
    () =>
      projectId
        ? api<Run[]>(`/projects/${projectId}/runs?${new URLSearchParams({ q: query, tag }).toString()}`)
        : Promise.resolve([]),
    busy ? 3000 : 15000,
    [projectId, query, tag],
  );
  const { data: projectTags } = usePolling(
    () => (projectId ? api<ProjectTags>(`/projects/${projectId}/tags`) : Promise.resolve({ runs: [], models: [] })),
    30000,
    [projectId, runs?.length],
  );
  const [picked, setPicked] = useState<string[] | null>(null); // null: follow the newest runs
  const [creating, setCreating] = useState(startWith !== undefined);
  // Columns and charts follow the project's first dataset's task; a run of another task shows its own.
  const task = projects?.find((p) => p.id === projectId)?.tasks?.[0] ?? "detect";
  const metricKeys = keyMetrics(task);
  const primary = metricKeys[0];

  useEffect(() => {
    setPicked(null);
    setTag("");
    setTyped("");
  }, [projectId]);
  useEffect(() => setBusy(!!runs?.some((r) => ACTIVE.includes(r.status))), [runs]);

  const compared = useMemo(() => {
    const withData = (runs ?? []).filter((r) => r.epoch > 0);
    const ids = picked ?? withData.slice(0, 4).map((r) => r.id);
    return ids.filter((id) => runs?.some((r) => r.id === id)).slice(0, MAX_COMPARED);
  }, [runs, picked]);

  const toggle = (id: string) => {
    const next = compared.includes(id) ? compared.filter((x) => x !== id) : [...compared, id];
    setPicked(next.slice(-MAX_COMPARED));
  };

  const counts = {
    running: runs?.filter((r) => r.status === "running").length ?? 0,
    queued: runs?.filter((r) => r.status === "queued").length ?? 0,
    done: runs?.filter((r) => r.status === "succeeded").length ?? 0,
  };
  const best = (runs ?? []).filter((r) => r.metrics[primary] != null).sort((a, b) => b.metrics[primary] - a.metrics[primary])[0];

  const project = projects?.find((p) => p.id === projectId);
  return (
    <>
      {crumbs && (
        <nav className="page-crumbs">
          <a href="#/training">학습</a>
          <strong>{project?.name ?? "…"}</strong>
        </nav>
      )}
      <div className="page-head">
        <div>
          <h1>학습</h1>
          <p className="muted">데이터셋 버전을 골라 모델을 학습합니다. 표에서 실행을 체크하면 지표 곡선을 겹쳐 비교할 수 있습니다.</p>
        </div>
        <div className="head-actions">
          <ProjectPicker projects={projects} projectId={projectId} onChange={onProject} />
          <button className="primary" onClick={() => setCreating(true)} disabled={!projectId}>
            <Icon name="plus" />새 학습
          </button>
        </div>
      </div>

      <div className="stats metrics">
        <Stat icon="activity" label="실행 중" value={counts.running} sub={counts.queued ? `대기 ${counts.queued}` : "대기 없음"} />
        <Stat icon="check" label="완료" value={counts.done} sub={`전체 ${runs?.length ?? 0}`} />
        <Stat
          icon="chart"
          label={`최고 ${METRIC_LABEL[primary]}`}
          value={best ? metric(best.metrics[primary]) : "–"}
          sub={best ? best.name : "완료한 학습 없음"}
        />
        <Stat icon="box" label="등록된 모델" value={runs?.filter((r) => r.model).length ?? 0} sub="학습이 끝나면 자동 등록" />
      </div>

      {compared.length > 0 && <Comparison ids={compared} live={busy} task={task} />}

      <SweepList projectId={projectId} />

      <RunTable
        runs={runs ?? null}
        projectId={projectId}
        task={task}
        filtered={filtered}
        compared={compared}
        onCompare={toggle}
        tools={
          (filtered || (runs?.length ?? 0) > 5 || (projectTags?.runs.length ?? 0) > 0) && (
            <ListFilter
              query={typed}
              onQuery={setTyped}
              tags={projectTags?.runs ?? []}
              tag={tag}
              onTag={setTag}
              placeholder="이름 · 메모 · 태그로 찾기"
            />
          )
        }
      />

      {creating && (
        <NewRun
          projectId={projectId}
          task={task}
          defaultTemplate={projects?.find((p) => p.id === projectId)?.settings?.template}
          releaseId={startWith}
          onClose={() => {
            setCreating(false);
            if (startWith !== undefined) window.location.hash = projectId ? `#/training/project/${projectId}` : "#/training";
          }}
          onStarted={(runId) => {
            setCreating(false);
            void refresh();
            window.location.hash = `#/training/${runId}`;
          }}
          onSweep={(id) => {
            setCreating(false);
            window.location.hash = `#/training/sweep/${id}`;
          }}
        />
      )}
    </>
  );
}

function settingValue(value: unknown): string {
  return typeof value === "boolean" ? (value ? "예" : "아니오") : String(value);
}

/** Where a waiting or running job is: one line, in plain words. */
function Stages({ run }: { run: RunDetail }) {
  const order = [run.sync ? "sync" : "queued", "data", "env", "train", "upload"];
  const current = run.status === "queued" ? order[0] : run.stage || "data";
  const at = order.indexOf(current);
  return (
    <ol className="stages" aria-label="진행 단계">
      {order.map((key, i) => (
        <li key={key} className={i < at ? "done" : i === at ? "now" : ""} aria-current={i === at ? "step" : undefined}>
          <i />
          {STAGE_LABEL[key]}
          {key === "sync" && i === at && run.sync?.bytes ? (
            <span className="num">{Math.floor(((run.sync.done_bytes ?? 0) / run.sync.bytes) * 100)}%</span>
          ) : null}
          {key === "train" && i === at && run.epochs > 0 && (
            <span className="num">
              {run.epoch}/{run.epochs}
            </span>
          )}
        </li>
      ))}
    </ol>
  );
}

const GB = 1024 ** 3;
/** "320.4 GB", or "7.2 MB" below a gigabyte */
const gbText = (bytes: number) =>
  bytes >= GB
    ? `${(bytes / GB).toLocaleString(undefined, { maximumFractionDigits: 1 })} GB`
    : `${(bytes / 1024 ** 2).toLocaleString(undefined, { maximumFractionDigits: 1 })} MB`;

/** "약 25분", "약 3시간 10분" */
function etaText(seconds: number): string {
  if (seconds < 60) return "1분 안";
  const minutes = Math.round(seconds / 60);
  return minutes < 60 ? `약 ${minutes}분` : `약 ${Math.floor(minutes / 60)}시간 ${minutes % 60}분`;
}

/** A run waiting for its images: how far the copy from the node that holds them got. */
function SyncProgress({ sync }: { sync: RunSync }) {
  const total = sync.bytes ?? 0;
  const done = sync.done_bytes ?? 0;
  const pct = total ? Math.min(100, (done / total) * 100) : 0;
  const parts = total
    ? [
        `${Math.floor(pct)}%`,
        `${gbText(done)} / ${gbText(total)}`,
        sync.rate ? `${Math.round(sync.rate / 1024 ** 2).toLocaleString()} MB/s` : null,
        sync.eta_s ? `남은 시간 ${etaText(sync.eta_s)}` : null,
      ].filter(Boolean)
    : ["준비 중"];
  return (
    <div className="notice sync-progress">
      <div>
        <strong>{sync.node_name || "학습할 노드"}가 이미지를 받는 중</strong>{" "}
        <span className="muted small">— 이미지가 있는 노드에서 직접 받습니다. 받는 동안 GPU는 잡지 않습니다.</span>
      </div>
      <div className="sync-bar">
        <i style={{ width: `${pct}%` }} />
      </div>
      <div className="small num">{parts.join(" · ")}</div>
    </div>
  );
}

/** Where a run may train: automatically where the images are, or on the servers picked. */
/** Where a run will train, as the Hub decides it (``training.placement``): the node with the images, the picked
 * ones that have them, or the one that receives them first. GPUs can be picked by number only when it is one. */
function trainTargets(choices: NodeChoices, mode: "auto" | "pick", picked: string[]): NodeChoice[] {
  if (mode === "auto") return choices.nodes.filter((n) => n.id === choices.source);
  const chosen = choices.nodes.filter((n) => picked.includes(n.id));
  const ready = chosen.filter((n) => n.images);
  if (ready.length) return ready;
  const receiver = [...chosen].sort((a, b) => Number(b.status === "online") - Number(a.status === "online") || b.gpus - a.gpus)[0];
  return receiver ? [receiver] : [];
}

export type GpuPick = "auto" | "cpu" | number[];

const GPU_STATE: Record<NodeGpu["state"], string> = {
  free: "비어 있음",
  busy: "사용 중",
  job: "다른 학습 중",
  reserved: "사람 전용",
  offline: "오프라인",
};

/** The GPUs of the node a run trains on: 자동 (a free one), CPU only, or the ones ticked (people's GPUs cannot be). */
function GpuField({
  targets,
  pick,
  onPick,
  count,
  onCount,
}: {
  targets: NodeChoice[];
  pick: GpuPick;
  onPick: (pick: GpuPick) => void;
  count: string;
  onCount: (count: string) => void;
}) {
  const one = targets.length === 1 ? targets[0] : null;
  const gpus = one?.gpu_list ?? [];
  if (!one || gpus.length === 0) {
    // Several servers may take it (or one without GPUs): only how many, up to what one of them can give.
    const most = Math.max(0, ...targets.map((n) => (n.gpu_list ?? []).filter((g) => !g.reserved).length));
    const kept = targets.some((n) => (n.gpu_list ?? []).some((g) => g.reserved));
    return (
      <label>
        GPU 수
        <select value={count} onChange={(e) => onCount(e.target.value)}>
          <option value="">자동{most ? " (빈 GPU 1개)" : ""}</option>
          <option value="0">CPU만</option>
          {Array.from({ length: most }, (_, i) => i + 1).map((n) => (
            <option key={n} value={String(n)}>
              {n}개
            </option>
          ))}
        </select>
        {kept && <span className="muted small">사람 전용 GPU는 고를 수 없어 셈에서 뺐습니다.</span>}
      </label>
    );
  }
  const ticked = Array.isArray(pick) ? pick : [];
  const toggle = (index: number) => {
    const next = ticked.includes(index) ? ticked.filter((i) => i !== index) : [...ticked, index].sort((a, b) => a - b);
    onPick(next.length ? next : "auto");
  };
  const waiting = gpus.filter((g) => ticked.includes(g.index) && g.state !== "free");
  return (
    <div className="gpu-field">
      <div className="eyebrow">GPU — {one.name}</div>
      <div className="gpu-pick">
        <button type="button" className={`chip ${pick === "auto" ? "active" : ""}`} onClick={() => onPick("auto")}>
          자동
        </button>
        <button type="button" className={`chip ${pick === "cpu" ? "active" : ""}`} onClick={() => onPick("cpu")}>
          CPU만
        </button>
        {gpus.map((g) => {
          const label = `GPU ${g.index}${g.name ? ` · ${g.name.replace(/^NVIDIA (GeForce )?/, "")}` : ""}`;
          const who = g.users.length ? g.users.join(", ") : "";
          const why = g.reserved
            ? `사람 전용${who ? ` · ${who}` : ""}`
            : g.state === "busy" && who
              ? `사용 중: ${who}`
              : GPU_STATE[g.state];
          return (
            <button
              type="button"
              key={g.index}
              className={`chip gpu ${g.state} ${ticked.includes(g.index) ? "active" : ""}`}
              disabled={g.reserved}
              aria-pressed={ticked.includes(g.index)}
              title={g.reserved ? "사람 전용으로 빼 둔 GPU입니다 (노드 설정)" : why}
              onClick={() => toggle(g.index)}
            >
              {label} <span>{why}</span>
            </button>
          );
        })}
      </div>
      <p className="muted small">
        {pick === "auto"
          ? "비어 있는 GPU 1개를 씁니다."
          : pick === "cpu"
            ? "GPU 없이 CPU로 학습합니다."
            : `GPU ${ticked.join(", ")} (${ticked.length}개)로 학습합니다.` +
              (waiting.length ? ` 지금 쓰이고 있는 GPU ${waiting.map((g) => g.index).join(", ")}이(가) 비면 시작합니다.` : "")}
      </p>
    </div>
  );
}

function NodeField({
  choices,
  mode,
  onMode,
  picked,
  onPicked,
  dataAfter,
  onDataAfter,
  sweep,
}: {
  choices: NodeChoices;
  mode: "auto" | "pick";
  onMode: (mode: "auto" | "pick") => void;
  picked: string[];
  onPicked: (ids: string[]) => void;
  dataAfter: "keep" | "delete";
  onDataAfter: (value: "keep" | "delete") => void;
  sweep: boolean; // a search: the copy goes (or stays) once all its runs are over
}) {
  const source = choices.nodes.find((n) => n.id === choices.source);
  const chosen = choices.nodes.filter((n) => picked.includes(n.id));
  const ready = chosen.filter((n) => n.images);
  // The Hub's pick when none of them has the images: online first, then the one with the most GPUs.
  const receiver = [...chosen].sort((a, b) => Number(b.status === "online") - Number(a.status === "online") || b.gpus - a.gpus)[0];
  const toggle = (id: string) => onPicked(picked.includes(id) ? picked.filter((x) => x !== id) : [...picked, id]);
  // Where it will train holds a copy sent from elsewhere (or will receive one): what becomes of it afterwards
  const targets = ready.length ? ready : receiver ? [receiver] : [];
  const copied = targets.some((n) => n.images !== "registered");
  const when = sweep ? "탐색이 모두 끝나면" : "학습이 끝나면";
  return (
    <div className="node-field">
      <label>
        학습할 노드
        <select value={mode} onChange={(e) => onMode(e.target.value as "auto" | "pick")}>
          <option value="auto">자동 — 이미지가 있는 {source?.name ?? "노드"}</option>
          <option value="pick">노드 고르기</option>
        </select>
      </label>
      {mode === "pick" && (
        <>
          <div className="node-pick">
            {choices.nodes.map((n) => {
              const blocked = !n.enabled
                ? "작업 받지 않음"
                : !(n.roles ?? ["compute"]).includes("compute")
                  ? "학습용 아님"
                  : !n.images && !n.can_receive
                    ? "0.37.0 이상으로 업데이트 필요"
                    : "";
              return (
                <label key={n.id} className={blocked ? "off" : ""}>
                  <input type="checkbox" checked={picked.includes(n.id)} disabled={Boolean(blocked)} onChange={() => toggle(n.id)} />
                  <span className="node-pick-name">
                    <i className={`dot ${n.status}`} />
                    {n.name}
                  </span>
                  <span className="muted small">{n.gpus ? `빈 GPU ${n.gpus_free}/${n.gpus}` : "GPU 없음"}</span>
                  <span className={`tag ${n.images ? "" : "muted"}`}>
                    {blocked || (n.images === "registered" ? "이미지 있음" : n.images === "received" ? "받아 둠" : `받아야 함 ${gbText(choices.bytes)}`)}
                  </span>
                </label>
              );
            })}
          </div>
          {chosen.length > 0 && ready.length > 0 && (
            <p className="muted small">
              이미지가 있는 {ready.map((n) => n.name).join(", ")}에서 학습합니다{ready.length > 1 ? " (GPU가 먼저 비는 쪽)" : ""}.
            </p>
          )}
          {chosen.length > 0 && ready.length === 0 && receiver && (
            <p className="muted small">
              고른 노드에 이미지가 없어, {receiver.name}가 먼저 {source?.name ?? "이미지가 있는 노드"}에서 {choices.items.toLocaleString()}장 (
              {gbText(choices.bytes)})을 직접 받은 뒤 학습합니다. 받는 동안 GPU는 잡지 않습니다.
              {receiver.cache_free_gb != null && receiver.cache_free_gb * GB < choices.bytes && (
                <span className="bad-text"> 보관 위치의 남은 공간({receiver.cache_free_gb} GB)이 모자랍니다 — 노드 설정에서 바꾸세요.</span>
              )}
            </p>
          )}
          {chosen.length > 0 && copied && (
            <div className="data-after">
              <div className="segmented" role="radiogroup" aria-label="보낸 데이터">
                <span className="eyebrow">{when} 이 서버로 보낸 데이터를</span>
                <button type="button" className={dataAfter === "keep" ? "active" : ""} onClick={() => onDataAfter("keep")}>
                  남겨 두기
                </button>
                <button type="button" className={dataAfter === "delete" ? "active" : ""} onClick={() => onDataAfter("delete")}>
                  지우기
                </button>
              </div>
              <span className="muted small">
                {dataAfter === "keep"
                  ? "같은 버전으로 다시 학습하면 바로 시작합니다. 버전 화면에서 언제든 지울 수 있습니다."
                  : `${sweep ? "탐색이 끝나면" : "학습이 성공하면"} 지웁니다. 실패하면 이어서 실행할 수 있게 남깁니다.`}
              </span>
            </div>
          )}
        </>
      )}
    </div>
  );
}

/** The same metric of several runs in one small chart each; a run keeps its colour across charts. */
function Comparison({ ids, live, task }: { ids: string[]; live: boolean; task: string }) {
  const key = ids.join(",");
  const { data: details } = usePolling(
    () => Promise.all(ids.map((id) => api<RunDetail>(`/runs/${id}`).catch(() => null))),
    live ? 4000 : 30000,
    [key],
  );
  const shown = (details ?? []).filter((d): d is RunDetail => d != null && ids.includes(d.id));
  const series = (metricKey: string): EpochSeries[] =>
    shown
      .filter((d) => d.history.series[metricKey])
      .map((d) => ({
        key: d.id,
        label: d.name,
        slot: ids.indexOf(d.id) + 1,
        steps: d.history.steps,
        values: d.history.series[metricKey],
      }));
  const charts = (COMPARE_CHARTS[task] ?? COMPARE_CHARTS.detect).filter((k) => series(k).length > 0).slice(0, 6);

  return (
    <section className="compare-runs">
      <div className="legend runs-legend">
        {shown.map((d) => (
          <a key={d.id} className="legend-item" href={`#/training/${d.id}`}>
            <i style={{ background: `var(--series-${ids.indexOf(d.id) + 1})` }} />
            {d.name}
            <strong>{metric(d.metrics[keyMetrics(task)[0]])}</strong>
          </a>
        ))}
        {!details && <span className="muted small">지표를 불러오는 중…</span>}
      </div>
      <div className="mini-charts">
        {charts.map((k) => (
          <EpochChart key={k} title={METRIC_LABEL[k] ?? k} series={series(k)} />
        ))}
      </div>
    </section>
  );
}

// ---- one run ----------------------------------------------------------------------------

const DETAIL_CHARTS: { title: string; keys: string[] }[] = [
  { title: "정확도", keys: ["top1", "top5"] },
  { title: "loss", keys: ["train_loss", "val_loss"] },
  { title: "mask mAP", keys: ["mask_map50_95", "mask_map50"] },
  { title: "mAP", keys: ["map50_95", "map50"] },
  { title: "seg loss", keys: ["train_seg_loss", "val_seg_loss"] },
  { title: "Precision · Recall", keys: ["precision", "recall"] },
  { title: "box loss", keys: ["train_box_loss", "val_box_loss"] },
  { title: "cls loss", keys: ["train_cls_loss", "val_cls_loss"] },
  { title: "dfl loss", keys: ["train_dfl_loss", "val_dfl_loss"] },
];
const VAL_SOURCE: Record<string, string> = {
  release: "버전의 val split",
  test: "버전의 test split",
  holdout: "train에서 떼어 냄",
  train: "train으로 검증 (지표가 높게 나옴)",
};

/**
 * A run's name as its screen's title, with "이름 바꾸기" beside it. A name is one per Hub: one another run has, in
 * any project, is refused, and the Hub says where.
 */
function RunName({ name, badge, onSave }: { name: string; badge: ReactNode; onSave: (name: string) => Promise<unknown> }) {
  const [draft, setDraft] = useState<string | null>(null); // the name being typed, while renaming
  const [failure, setFailure] = useState<string | null>(null);
  const [busy, setBusy] = useState(false);
  const save = async (e: FormEvent) => {
    e.preventDefault();
    const wanted = (draft ?? "").trim();
    if (!wanted || wanted === name) {
      setDraft(null);
      return;
    }
    setBusy(true);
    setFailure(null);
    try {
      await onSave(wanted);
      setDraft(null);
    } catch (err) {
      setFailure(err instanceof Error ? err.message : String(err));
    } finally {
      setBusy(false);
    }
  };
  if (draft === null) {
    return (
      <h1 className="with-badge">
        {name}
        {badge}
        <button
          type="button"
          className="icon-btn run-rename"
          title="이름 바꾸기 (Hub 전체에서 겹치지 않는 이름)"
          aria-label="실행 이름 바꾸기"
          onClick={() => {
            setFailure(null);
            setDraft(name);
          }}
        >
          <Icon name="edit" size={15} />
        </button>
      </h1>
    );
  }
  return (
    <form className="run-rename-form" onSubmit={save}>
      <input
        autoFocus
        value={draft}
        maxLength={160}
        onChange={(e) => setDraft(e.target.value)}
        onKeyDown={(e) => e.key === "Escape" && setDraft(null)}
        aria-label="새 실행 이름"
      />
      <button className="primary small" disabled={busy || !draft.trim()}>
        {busy ? "저장 중…" : "저장"}
      </button>
      <button type="button" className="small" disabled={busy} onClick={() => setDraft(null)}>
        취소
      </button>
      {failure && <div className="error small">{failure}</div>}
    </form>
  );
}

function RunView({ runId }: { runId: string }) {
  const [live, setLive] = useState(true);
  const { data: run, error, refresh } = usePolling(() => api<RunDetail>(`/runs/${runId}`), live ? 2500 : 20000, [runId]);
  const [working, setWorking] = useState(false);
  const [failure, setFailure] = useState<string | null>(null);
  const projectOf = run?.project_id ?? "";
  const { data: projectTags } = usePolling(
    () => (projectOf ? api<ProjectTags>(`/projects/${projectOf}/tags`) : Promise.resolve({ runs: [], models: [] })),
    60000,
    [projectOf],
  );
  useEffect(() => setLive(!run || ACTIVE.includes(run.status)), [run]);

  if (error && !run) return <div className="error">{error}</div>;
  if (!run) return <div className="muted">불러오는 중…</div>;

  const active = ACTIVE.includes(run.status);
  const metricKeys = keyMetrics(run.task);
  const act = async (work: () => Promise<unknown>) => {
    setWorking(true);
    setFailure(null);
    try {
      await work();
      await refresh();
    } catch (e) {
      setFailure(e instanceof Error ? e.message : String(e));
    } finally {
      setWorking(false);
    }
  };
  const again = () =>
    act(async () => {
      const next = await api<Run>(`/runs/${run.id}/rerun`, { method: "POST" });
      window.location.hash = `#/training/${next.id}`;
    });
  const resume = () =>
    act(async () => {
      const next = await api<Run>(`/runs/${run.id}/resume`, { method: "POST" });
      window.location.hash = `#/training/${next.id}`;
    });
  const remove = () => {
    if (!window.confirm(`${run.name} 학습을 휴지통으로 옮길까요? 휴지통에서 되살릴 수 있고, 등록된 모델은 그대로 남습니다.`)) return;
    void act(async () => {
      await api(`/runs/${run.id}`, { method: "DELETE" });
      announceTrash();
      window.location.hash = `#/training/project/${run.project_id}`;
    });
  };

  const charts = DETAIL_CHARTS.map((c) => ({
    title: c.title,
    series: c.keys
      .filter((k) => run.history.series[k])
      .map((k, i) => ({ key: k, label: METRIC_LABEL[k] ?? k, slot: i + 1, steps: run.history.steps, values: run.history.series[k] })),
  })).filter((c) => c.series.length > 0);
  // Anything else the code logged gets its own small chart, so custom metrics show up without any setup.
  const grouped = new Set(DETAIL_CHARTS.flatMap((c) => c.keys));
  for (const key of Object.keys(run.history.series).filter((k) => !grouped.has(k)).slice(0, MAX_EXTRA_CHARTS)) {
    const series = [{ key, label: METRIC_LABEL[key] ?? key, slot: 1, steps: run.history.steps, values: run.history.series[key] }];
    charts.push({ title: METRIC_LABEL[key] ?? key, series });
  }
  const items = run.export.items ?? {};
  const excluded = run.export.excluded ?? {};
  const leftOut = (excluded.unlabelled ?? 0) + (excluded.partial ?? 0) + (excluded.missing ?? 0);
  const classRows = [...run.class_metrics].sort((a, b) => classScore(b) - classScore(a));
  // Basic settings and anything the user changed; detailed settings left at their default are folded.
  const shownSettings = run.settings.filter((s) => !s.advanced || s.changed);
  const defaultSettings = run.settings.filter((s) => s.advanced && !s.changed);
  const info = run.info as { ultralytics?: string; torch?: string; device?: string; gpus?: string[]; packages?: string[] };

  return (
    <>
      <a className="link back" href={`#/training/project/${run.project_id}`}>
        ← 학습 목록
      </a>
      <div className="page-head">
        <div>
          <RunName
            name={run.name}
            badge={<Badge kind={run.status}>{STATUS_LABEL[run.status]}</Badge>}
            onSave={async (name) => {
              await api(`/runs/${run.id}`, { method: "PATCH", body: { name } });
              await refresh();
            }}
          />
          <p className="muted">
            {run.release ? (
              <a href={`#/datasets/${run.release.id}`}>
                {run.release.dataset} {run.release.version}
              </a>
            ) : (
              "삭제된 버전"
            )}{" "}
            · {run.template.title}
            {run.template.variant && run.template.variant !== run.template.name ? ` ${run.template.variant}` : ""} ·{" "}
            {run.node_name ?? "서버 배정 대기"} ·{" "}
            {run.sweep ? (
              <a href={`#/training/sweep/${run.sweep.id}`}>
                탐색 {run.sweep.name}의 {run.sweep.index + 1}번째
              </a>
            ) : (
              `${VIA_LABEL[run.created_via] ?? run.created_via}에서 시작`
            )}{" "}
            · {timeAgo(run.created_at)}
          </p>
        </div>
        <div className="head-actions">
          {active && run.job_id && (
            <button className="danger" disabled={working} onClick={() => act(() => api(`/jobs/${run.job_id}/cancel`, { method: "POST" }))}>
              중단
            </button>
          )}
          {run.can_resume && (
            <button className="primary" disabled={working} onClick={resume} title="멈춘 곳(마지막으로 끝난 epoch)부터 이어서 학습합니다">
              이어서 실행
            </button>
          )}
          <button disabled={working} onClick={again} title="같은 버전과 설정으로 새 실행을 시작합니다">
            다시 실행
          </button>
          {!active && (
            <button disabled={working} onClick={remove}>
              삭제
            </button>
          )}
        </div>
      </div>

      <Annotations
        note={run.note}
        tags={run.tags}
        name={run.name}
        noteLabel="메모"
        known={(projectTags?.runs ?? []).map((t) => t.name)}
        onSave={async (fields) => {
          await api(`/runs/${run.id}`, { method: "PATCH", body: fields });
          await refresh();
        }}
      />

      {failure && <div className="error">{failure}</div>}
      {active && <Stages run={run} />}
      {!run.sweep && (
        <FollowBanner
          runId={run.id}
          active={active}
          succeeded={run.status === "succeeded"}
          hasModel={!!run.model}
          task={run.task}
          projectId={run.project_id}
        />
      )}
      {run.status === "queued" && run.sync && <SyncProgress sync={run.sync} />}
      {run.status === "queued" && !run.sync && run.wait_reason && <div className="notice">{run.wait_reason}</div>}
      {run.error && <div className="error">{run.error} 아래 로그에서 원인을 확인하세요.</div>}

      <div className="stats five metrics">
        {metricKeys.map((k) => (
          <Stat key={k} icon="chart" label={METRIC_LABEL[k]} value={metric(run.metrics[k])} sub={active ? "지금까지 최고" : "best 가중치"} />
        ))}
        <Stat
          icon="clock"
          label="Epoch"
          value={`${run.epoch} / ${run.epochs}`}
          sub={run.duration_s != null ? duration(run.duration_s) : "시작 전"}
        />
      </div>

      {run.model && (
        <a className="card model-link" href={`#/models/${run.model.id}`}>
          <Icon name="box" size={18} />
          <span>
            <strong>
              {run.model.name} v{run.model.version}
            </strong>
            <span className="muted"> 모델로 등록되었습니다</span>
          </span>
          {run.model.aliases.map((a) => (
            <AliasChip key={a} alias={a} />
          ))}
          <Icon name="arrow" size={14} />
        </a>
      )}

      {charts.length > 0 && run.metrics_csv_url && (
        <div className="charts-tools">
          <a className="link small" href={run.metrics_csv_url} download>
            단계별 지표 CSV로 받기
          </a>
        </div>
      )}
      {charts.length > 0 && (
        <div className="mini-charts wide">
          {charts.map((c) => (
            <EpochChart key={c.title} title={c.title} series={c.series} legend />
          ))}
        </div>
      )}

      <MediaPanel
        media={run.media}
        images={{
          ...Object.fromEntries(
            [...run.files, ...run.media.flatMap((m) => m.steps.flatMap((s) => s.files))]
              .filter((f) => f.kind === "image")
              .map((f) => [f.name, f.url]),
          ),
          ...(run.table_images ?? {}),
        }}
      />

      <div className="two-col">
        <section className="card">
          <h3>학습 데이터</h3>
          {run.export.items ? (
            <dl className="kv">
              <dt>이미지</dt>
              <dd>
                train {(items.train ?? 0).toLocaleString()} · val {(items.val ?? 0).toLocaleString()}
                {items.test ? ` · test ${items.test.toLocaleString()}` : ""}
              </dd>
              <dt>{run.task === "classify" ? "라벨" : "박스"}</dt>
              <dd>{(run.export.boxes ?? 0).toLocaleString()}</dd>
              <dt>클래스</dt>
              <dd>{run.export.classes?.length ?? 0}개</dd>
              <dt>검증 데이터</dt>
              <dd>{VAL_SOURCE[run.export.val_source ?? "release"]}</dd>
              <dt>제외한 이미지</dt>
              <dd>
                {leftOut === 0
                  ? "없음"
                  : [
                      excluded.unlabelled ? `라벨 없음 ${excluded.unlabelled}` : "",
                      excluded.partial ? `일부 클래스만 라벨 ${excluded.partial}` : "",
                      excluded.missing ? `파일 없음 ${excluded.missing}` : "",
                    ]
                      .filter(Boolean)
                      .join(" · ")}
              </dd>
              {(run.export.as_background ?? 0) > 0 && (
                <>
                  <dt>배경으로 넣음</dt>
                  <dd>{run.export.as_background}장 (라벨이 없거나 일부만 있는 이미지)</dd>
                </>
              )}
            </dl>
          ) : (
            <p className="muted">학습이 시작되면 버전에서 실제로 쓴 이미지 수가 표시됩니다.</p>
          )}
        </section>
        <section className="card">
          <h3>설정</h3>
          <dl className="kv">
            <dt>모델 종류</dt>
            <dd>
              {run.template.title} {run.template.version && <span className="muted">v{run.template.version}</span>}
            </dd>
            {shownSettings.map((item) => (
              <Fragment key={item.name}>
                <dt>{item.label}</dt>
                <dd>{settingValue(item.value)}</dd>
              </Fragment>
            ))}
            <dt>환경</dt>
            <dd>{run.environment.kind === "auto" ? "자동 설치" : run.environment.name}</dd>
            <dt>장치</dt>
            <dd>
              {info.device
                ? info.device === "cpu"
                  ? "CPU"
                  : `GPU ${run.gpu_indices.map((i) => `#${i}`).join(", ")}${info.gpus?.length ? ` (${info.gpus[0]})` : ""}`
                : run.config.gpus
                  ? `GPU ${run.config.gpus}개 요청`
                  : "CPU"}
            </dd>
            {info.ultralytics && (
              <>
                <dt>라이브러리</dt>
                <dd className="mono small">
                  ultralytics {info.ultralytics} · torch {info.torch}
                </dd>
              </>
            )}
            {run.resumed_from && (
              <>
                <dt>이어 받은 실행</dt>
                <dd>
                  <a href={`#/training/${run.resumed_from.id}`}>{run.resumed_from.name}</a>이 멈춘 곳부터
                </dd>
              </>
            )}
            {run.base_model && (
              <>
                <dt>시작 가중치</dt>
                <dd>
                  <a href={`#/models/${run.base_model.id}`}>
                    {run.base_model.name} v{run.base_model.version}
                  </a>
                  에서 이어서
                </dd>
              </>
            )}
            {run.source_run_id && (
              <>
                <dt>원본 실행</dt>
                <dd>
                  <a href={`#/training/${run.source_run_id}`}>같은 설정의 이전 실행</a>
                </dd>
              </>
            )}
          </dl>
          {defaultSettings.length > 0 && (
            <details className="defaults">
              <summary>기본값 그대로 둔 설정 {defaultSettings.length}개</summary>
              <dl className="kv">
                {defaultSettings.map((item) => (
                  <Fragment key={item.name}>
                    <dt>
                      {item.group ? `${item.group} · ` : ""}
                      {item.label}
                    </dt>
                    <dd>{settingValue(item.value)}</dd>
                  </Fragment>
                ))}
              </dl>
            </details>
          )}
        </section>
      </div>

      {run.files.length > 0 && (
        <section className="card">
          <div className="card-head">
            <div>
              <h3>
                산출물 <span className="muted">({run.files.length})</span>
              </h3>
              <div className="muted small">
                이 실행이 남긴 파일 전부입니다. 학습한 서버가 아니라 MOA가 보관합니다 ·{" "}
                {fileSize(run.files.reduce((sum, f) => sum + f.bytes, 0))}
              </div>
            </div>
            {run.files_url && (
              <a className="btn" href={run.files_url} download>
                전체 내려받기
              </a>
            )}
          </div>
          <FileList files={run.files} />
        </section>
      )}

      <details className="card fold">
        <summary>계보</summary>
        <LineageGraph path={`/runs/${run.id}/lineage`} />
      </details>

      {classRows.length > 0 && (
        <details className="card fold" open={classRows.length <= 12}>
          <summary>
            클래스별 지표 <span className="muted">({classRows.length})</span>
          </summary>
          <ClassTable rows={classRows} />
        </details>
      )}

      {info.packages && info.packages.length > 0 && (
        <details className="card fold">
          <summary>
            설치된 패키지 <span className="muted">({info.packages.length})</span>
          </summary>
          <pre className="json packages">{info.packages.join("\n")}</pre>
        </details>
      )}

      {run.status !== "queued" && <SystemLoad runId={run.id} live={active} />}

      {run.job_id && <LogTail jobId={run.job_id} live={active} />}
    </>
  );
}

/** A class table's order: by a column, high (or Z) first or low (or A) first; none is the order reported. */
type ClassSort = { key: keyof ClassMetric; down: boolean } | null;

export function ClassTable({ rows }: { rows: ClassMetric[] }) {
  // Columns follow what the template reported: mAP for detection, F1 for classification.
  const detection = rows.some((c) => c.map50_95 != null);
  const columns: [keyof ClassMetric, string][] = detection
    ? [["map50_95", "mAP50-95"], ["map50", "mAP50"], ["precision", "Precision"], ["recall", "Recall"]]
    : [["f1", "F1"], ["precision", "Precision"], ["recall", "Recall"]];
  const [sort, setSort] = useState<ClassSort>(null);
  // a click: 내림차순 → 올림차순 → 기본순 (as reported)
  const next = (key: keyof ClassMetric) =>
    setSort((was) => (was?.key !== key ? { key, down: true } : was.down ? { key, down: false } : null));
  const shown = useMemo(() => {
    if (!sort) return rows;
    const value = (c: ClassMetric) => c[sort.key];
    return [...rows].sort((a, b) => {
      const x = value(a);
      const y = value(b);
      if (x == null || y == null) return x == null ? (y == null ? 0 : 1) : -1; // no number: last either way
      const by = typeof x === "number" && typeof y === "number" ? x - y : String(x).localeCompare(String(y), "ko");
      return sort.down ? -by : by;
    });
  }, [rows, sort]);
  const head = (key: keyof ClassMetric, label: string, right: boolean) => {
    const on = sort?.key === key;
    const state = on ? (sort.down ? "내림차순" : "올림차순") : "기본순";
    return (
      <th key={key} className={right ? "right" : ""} aria-sort={on ? (sort.down ? "descending" : "ascending") : "none"}>
        <button type="button" className={`sort-head${on ? " on" : ""}`} onClick={() => next(key)} title={`${label}: ${state} (누를 때마다 내림차순 → 올림차순 → 기본순)`}>
          {label}
          <span className="sort-mark" aria-hidden>
            {on ? (sort.down ? "▼" : "▲") : "↕"}
          </span>
        </button>
      </th>
    );
  };
  return (
    <div className="table-wrap class-table">
      <table>
        <thead>
          <tr>
            {head("name", "클래스", false)}
            {columns.map(([key, label]) => head(key, label, true))}
          </tr>
        </thead>
        <tbody>
          {shown.map((c) => (
            <tr key={c.id}>
              <td>{c.name}</td>
              {columns.map(([key], i) => (
                <td key={key} className="num right">
                  {i === 0 && (
                    <span className="cell-bar">
                      <i style={{ width: `${Math.round(classScore(c) * 100)}%` }} />
                    </span>
                  )}
                  {metric(c[key] as number | undefined)}
                </td>
              ))}
            </tr>
          ))}
        </tbody>
      </table>
    </div>
  );
}

/** CPU, memory and GPU of the server while this run was on it (what W&B shows under "System"). */
function SystemLoad({ runId, live }: { runId: string; live: boolean }) {
  const [open, setOpen] = useState(false);
  const { data } = usePolling(
    () => (open ? api<MetricHistory & { gpu_indices: number[] }>(`/runs/${runId}/system`) : Promise.resolve(null)),
    live ? 30000 : 300000,
    [runId, open],
  );
  const chart = (title: string, keys: [string, string][], subtitle?: string) =>
    data &&
    keys.some(([key]) => data.series[key]) && (
      <LineChart
        title={title}
        subtitle={subtitle}
        unit="%"
        yMax={100}
        timestamps={data.timestamps}
        from={data.from}
        to={data.to}
        stepSeconds={data.step_s}
        series={keys.filter(([key]) => data.series[key]).map(([key, label]) => ({ key, label, values: data.series[key].avg }))}
      />
    );
  const gpus = data?.gpu_indices ?? [];
  return (
    <details className="card fold" onToggle={(e) => setOpen((e.target as HTMLDetailsElement).open)}>
      <summary>
        서버 자원 <span className="muted">(학습하는 동안의 CPU · 메모리 · GPU)</span>
      </summary>
      {!data ? (
        <p className="muted">불러오는 중…</p>
      ) : data.timestamps.length === 0 ? (
        <p className="muted">
          기록이 없습니다. 서버 자원은 1분 단위로 기록하므로 1분이 안 되는 학습에는 남지 않을 수 있습니다.
        </p>
      ) : (
        <div className="chart-grid">
          {chart("CPU · 메모리", [["cpu", "CPU"], ["mem", "메모리"]])}
          {gpus.length > 0 &&
            chart(
              "GPU",
              gpus.flatMap((i): [string, string][] => [[`gpu${i}_util`, `GPU ${i} 사용률`], [`gpu${i}_mem`, `GPU ${i} 메모리`]]),
              "이 학습에 배정된 GPU",
            )}
        </div>
      )}
    </details>
  );
}

/** A job's log, following it while ``live``; folded unless live or ``open`` (asked for by a click). */
export function LogTail({ jobId, live, open }: { jobId: string; live: boolean; open?: boolean }) {
  const { data: job } = usePolling(() => api<Job>(`/jobs/${jobId}`), live ? 2000 : 30000, [jobId]);
  const [lines, setLines] = useState<LogLine[]>([]);
  const lastSeq = useRef(0);
  const box = useRef<HTMLPreElement>(null);
  const count = job?.log_count ?? 0;

  useEffect(() => {
    if (count <= lastSeq.current) return;
    let alive = true;
    // Long trainings log thousands of lines; start near the end and follow from there.
    const after = lastSeq.current === 0 ? Math.max(0, count - 300) : lastSeq.current;
    void api<LogLine[]>(`/jobs/${jobId}/logs?after=${after}`).then((fresh) => {
      if (!alive || fresh.length === 0) return;
      lastSeq.current = fresh[fresh.length - 1].seq;
      setLines((prev) => [...prev, ...fresh.filter((l) => !prev.some((p) => p.seq === l.seq))].slice(-600));
    });
    return () => {
      alive = false;
    };
  }, [jobId, count]);

  useEffect(() => {
    const el = box.current;
    if (el) el.scrollTop = el.scrollHeight;
  }, [lines.length]);

  return (
    <details className="card fold" open={open ?? live}>
      <summary>
        로그 <span className="muted">({count.toLocaleString()}줄)</span>
      </summary>
      <pre className="log" ref={box}>
        {lines.length === 0 ? (
          <span className="dim">{live ? "로그를 기다리는 중…" : "로그가 없습니다."}</span>
        ) : (
          lines.map((l) => (
            <div key={l.seq} className={l.stream === "system" ? "log-system" : ""}>
              {l.text}
            </div>
          ))
        )}
      </pre>
    </details>
  );
}

// ---- start a run -------------------------------------------------------------------------

type FormValue = string | boolean;

function initialValues(template: Template | undefined): Record<string, FormValue> {
  const values: Record<string, FormValue> = {};
  for (const p of template?.params ?? []) {
    if (p.type === "object") continue;
    values[p.name] = p.type === "boolean" ? p.default === true : p.default == null ? "" : String(p.default);
  }
  return values;
}

/** One input of the form a template declares in its [params]. */
function ParamField({ spec, value, onChange }: { spec: TemplateParam; value: FormValue; onChange: (v: FormValue) => void }) {
  const label = spec.label || spec.name;
  const [typing, setTyping] = useState(false); // "직접 입력" on a list that allows other values
  if (spec.type === "boolean") {
    // a whole row: the name, then what it does in small print under it (never squeezed into a narrow cell)
    return (
      <label className="check param-check" title={spec.help}>
        <input type="checkbox" checked={value === true} onChange={(e) => onChange(e.target.checked)} />
        <span>
          {label}
          {spec.help && <span className="muted small param-help">{spec.help}</span>}
        </span>
      </label>
    );
  }
  const text = String(value);
  const listed = spec.choices.map(String);
  if (listed.length > 0 && !(typing || (spec.custom && text !== "" && !listed.includes(text)))) {
    return (
      <label title={spec.help}>
        {label}
        <select
          value={text}
          onChange={(e) => {
            if (e.target.value === "__custom__") setTyping(true);
            else onChange(e.target.value);
          }}
        >
          {listed.map((c) => (
            <option key={c} value={c}>
              {c.replace(/\.pt$/, "")}
            </option>
          ))}
          {spec.custom && <option value={"__custom__"}>직접 입력…</option>}
        </select>
      </label>
    );
  }
  const numeric = spec.type === "integer" || spec.type === "number";
  return (
    <label title={spec.help}>
      {label}
      <input
        type={numeric ? "number" : "text"}
        value={text}
        min={spec.minimum ?? undefined}
        max={spec.maximum ?? undefined}
        step={spec.type === "number" ? "any" : undefined}
        onChange={(e) => onChange(e.target.value)}
        className={numeric ? undefined : "mono"}
        autoFocus={typing}
      />
    </label>
  );
}

/** How far a version's images have their 이미지 탐색 vectors (``/releases/{id}/embeddings``). */
interface EmbedInfo {
  model: string;
  title: string;
  total: number;
  done: number;
  todo: number;
  going: string | null;
}

/** 학습하면서 이미지 임베딩도 만들기: the server it trains on makes the vectors the version's images lack, from the
 * copy it has anyway. Only on servers with the 임베딩용 role; none to make when every image has one. */
function EmbedField({
  info,
  on,
  onChange,
  roleless,
  later,
}: {
  info: EmbedInfo | null;
  on: boolean;
  onChange: (on: boolean) => void;
  roleless: NodeChoice[];
  later: boolean; // a pipeline step: on the version it runs on then
}) {
  const done = !later && info !== null && info.todo === 0;
  const blocked = roleless.length > 0 || done;
  const n = (v: number) => v.toLocaleString();
  let say: ReactNode;
  if (roleless.length > 0) {
    say = (
      <>
        {roleless.map((r) => r.name).join(", ")}에는 &quot;임베딩용&quot; 역할이 없어 고를 수 없습니다. <a href="#/nodes">노드</a> 화면에서 그
        노드의 설정 → 역할에 &quot;임베딩용&quot;을 켜세요.
      </>
    );
  } else if (later) {
    say = "파이프라인이 돌 때의 버전에서, 벡터가 없는 이미지만 학습할 서버가 받은 이미지로 만듭니다.";
  } else if (!info) {
    say = "학습할 서버가 이 버전의 이미지를 받는 김에 벡터를 만들어 Hub에 둡니다.";
  } else if (info.going) {
    say = "이 버전의 벡터를 지금 만들고 있습니다. 끝나지 않은 것만 이어서 만들려면 끝난 뒤 다시 켜세요.";
  } else if (done) {
    say = `이 버전의 이미지 ${n(info.total)}장은 모두 벡터가 있어(다른 버전 · 데이터셋에서 만든 것 포함) 건너뜁니다.`;
  } else if (info.done > 0) {
    say = `이미지 ${n(info.total)}장 중 ${n(info.done)}장은 이미 벡터가 있어(같은 이미지는 다른 버전 · 데이터셋에서 만든 것도) 나머지 ${n(info.todo)}장만 만듭니다.`;
  } else {
    say = `이미지 ${n(info.total)}장의 벡터를 학습할 서버가 받은 이미지로 만들어 Hub에 둡니다. 따로 옮기지 않습니다.`;
  }
  return (
    <div className="embed-field">
      <label className="check">
        <input type="checkbox" checked={on && !blocked} disabled={blocked || Boolean(info?.going && !later)} onChange={(e) => onChange(e.target.checked)} />
        학습하면서 이미지 임베딩도 만들기
        <span className="muted small">이미지 탐색용{info ? ` · ${info.title}` : ""}</span>
      </label>
      <p className={`small ${roleless.length > 0 ? "bad-text" : "muted"}`}>
        {say}
        {on && !blocked && !later && info && info.todo > 0 ? " GPU가 하나면 학습이 끝난 뒤 이어서 만듭니다." : ""}
      </p>
    </div>
  );
}

/** A search step's settings to try as typed in 여러 설정 시험해 보기 again: values listed, or a range ``a~b``. */
function trialsOf(space: unknown): Trial[] {
  if (!space || typeof space !== "object") return [];
  return Object.entries(space as Record<string, SweepSpec>).map(([name, spec]) => ({
    name,
    text: spec.min !== undefined && spec.max !== undefined ? `${spec.min}~${spec.max}` : (spec.values ?? []).map(String).join(", "),
  }));
}

/** 새 학습; with ``settingsFor``, the settings of a pipeline's training or search step (nothing starts). */
export function NewRun({
  projectId,
  task,
  defaultTemplate,
  releaseId,
  onClose,
  onStarted,
  onSweep,
  settingsFor,
}: {
  projectId: string;
  task: string;
  defaultTemplate?: string;
  releaseId?: string;
  onClose: () => void;
  onStarted: (runId: string) => void;
  onSweep: (id: string) => void;
  settingsFor?: StepSettings;
}) {
  // A pipeline step's settings open as they were applied: every field below starts from them.
  const given = settingsFor?.config ?? {};
  const givenNodes = Array.isArray(given.nodes) ? given.nodes.map(String) : [];
  // "학습 끝나면": the steps the Hub takes after this training (the last choice is offered again)
  const [follow, setFollow] = useState<FollowSpec>(rememberedFollow);
  const [save, setSave] = useState<SaveSpec>({ on: false, name: "", trigger: "manual" });
  const [tree, setTree] = useState<TreeDataset[] | null>(null);
  const [templates, setTemplates] = useState<Template[] | null>(null);
  const [environments, setEnvironments] = useState<Environment[]>([]);
  const [release, setRelease] = useState(releaseId ?? "");
  const [templateName, setTemplateName] = useState("");
  const [values, setValues] = useState<Record<string, FormValue>>({});
  const [classes, setClasses] = useState<LabelClass[]>([]);
  const [releaseInfo, setReleaseInfo] = useState<ReleaseInfo | null>(null);
  const [chosen, setChosen] = useState<number[]>([]); // the classes ticked (every one when the version loads)
  const [extra, setExtra] = useState({
    environment: String(given.environment ?? ""),
    gpus: given.gpus == null ? "" : String(given.gpus),
    name: "",
    model_name: String(given.model_name ?? ""),
    base: String(given.base_model_id ?? ""),
  });
  const [models, setModels] = useState<ModelInfo[]>([]);
  const [error, setError] = useState<string | null>(null);
  const [busy, setBusy] = useState(false);
  // A search: the settings to try, and how.
  const [trials, setTrials] = useState<Trial[]>(() => trialsOf(given.space));
  const [method, setMethod] = useState<SweepMethod>(() =>
    ["grid", "random", "bayes"].includes(String(given.method)) ? (given.method as SweepMethod) : "grid",
  );
  const [drawCount, setDrawCount] = useState(Number(given.max_runs) || 10);
  const [parallel, setParallel] = useState(Number(given.parallel) || 1);
  // What the chosen version's dataset is labelled for decides the templates on offer.
  const runTask = tree?.find((d) => d.releases.some((r) => r.id === release))?.task ?? task;
  const [goalMetric, setGoalMetric] = useState(keyMetrics(runTask)[0]);
  const [choices, setChoices] = useState<NodeChoices | null>(null);
  const [nodeMode, setNodeMode] = useState<"auto" | "pick">(givenNodes.length ? "pick" : "auto");
  const [pickedNodes, setPickedNodes] = useState<string[]>(givenNodes);
  const [dataAfter, setDataAfter] = useState<"keep" | "delete">(given.data_after === "delete" ? "delete" : "keep");
  const [gpuPick, setGpuPick] = useState<GpuPick>("auto");
  // What a step's settings had for the GPUs and the classes: put back once the servers and the classes are known
  // (picking a server or a version starts them over).
  const keptGpus = useRef<"cpu" | number[] | null>(
    Array.isArray(given.gpu_indices) && given.gpu_indices.length ? given.gpu_indices.map(Number) : given.gpus === 0 ? "cpu" : null,
  );
  const keptClasses = useRef<number[] | null>(Array.isArray(given.classes) && given.classes.length ? given.classes.map(Number) : null);
  const keptMetric = useRef(String(given.metric ?? ""));
  // 이미지 탐색's vectors, made on the server it trains on from the copy it has (only the images without one)
  const [embed, setEmbed] = useState(Boolean(given.embed));
  const [embedInfo, setEmbedInfo] = useState<EmbedInfo | null>(null);
  useEffect(() => {
    setEmbedInfo(null);
    if (!release) return;
    let live = true;
    void api<EmbedInfo>(`/releases/${release}/embeddings`)
      .then((found) => live && setEmbedInfo(found))
      .catch(() => live && setEmbedInfo(null));
    return () => {
      live = false;
    };
  }, [release]);

  useEffect(() => {
    void api<TreeDataset[]>(`/projects/${projectId}/datasets`).then((data) => {
      setTree(data);
      const ready = data.flatMap((d) => d.releases).filter((r) => r.status === "ready");
      // a pipeline step: the pipeline's dataset only (none yet: no classes to pick, the task's model kinds)
      if (!settingsFor) setRelease((current) => (ready.some((r) => r.id === current) ? current : (ready[0]?.id ?? "")));
    });
    void api<Environment[]>("/environments").then(setEnvironments);
    void api<ModelInfo[]>(`/projects/${projectId}/models`).then(setModels);
  }, [projectId]);

  useEffect(() => {
    setTemplates(null);
    const metrics = keyMetrics(runTask);
    setGoalMetric(metrics.includes(keptMetric.current) ? keptMetric.current : metrics[0]);
    keptMetric.current = "";
    void api<Template[]>(`/templates?task=${encodeURIComponent(runTask)}`).then((list) => {
      setTemplates(list);
      const wanted = String(given.template ?? "") || defaultTemplate;
      const first = list.find((t) => t.name === wanted || t.id === wanted) ?? list[0];
      setTemplateName(first?.name ?? "");
      // a step's settings open as they were set (the model kind's own values first)
      const kept = (given.params ?? {}) as Record<string, FormValue>;
      setValues({ ...initialValues(first), ...(first?.name === wanted ? kept : {}) });
    });
  }, [runTask, defaultTemplate]); // eslint-disable-line react-hooks/exhaustive-deps

  useEffect(() => {
    setChosen([]);
    setChoices(null);
    setReleaseInfo(null);
    if (!release) return setClasses([]);
    void api<ReleaseInfo>(`/releases/${release}`).then((info) => {
      const ids = info.label_space.classes.map((c) => c.id);
      const kept = keptClasses.current?.filter((id) => ids.includes(id)) ?? [];
      keptClasses.current = null;
      setClasses(info.label_space.classes);
      setChosen(kept.length ? kept : ids); // every class, until some are left out
      setReleaseInfo(info);
    });
    void api<NodeChoices>(`/releases/${release}/nodes`).then(setChoices);
  }, [release]);

  const template = templates?.find((t) => t.name === templateName);
  // Where it trains decides the GPUs on offer: pick again whenever that changes.
  const targets = choices ? trainTargets(choices, nodeMode, pickedNodes) : [];
  const roleless = targets.filter((n) => !(n.roles ?? []).includes("embedding")); // servers it may train on, no 임베딩용
  const later = settingsFor?.kind === "train"; // a pipeline step: the version is the one it runs on then
  const embedding =
    embed && targets.length > 0 && roleless.length === 0 && trials.length === 0 && settingsFor?.kind !== "sweep" && (later || (embedInfo?.todo ?? 1) > 0);
  const targetKey = targets.map((n) => n.id).join(",");
  useEffect(() => {
    if (keptGpus.current === null) return setGpuPick("auto");
    if (!targetKey) return; // the servers are not known yet: a step's GPUs wait for them
    const one = targets.length === 1 ? targets[0] : null;
    const have = (one?.gpu_list ?? []).map((g) => g.index);
    const kept = keptGpus.current;
    keptGpus.current = null;
    setGpuPick(kept === "cpu" || (kept.length && kept.every((i) => have.includes(i))) ? kept : "auto");
  }, [targetKey]); // eslint-disable-line react-hooks/exhaustive-deps
  const single = targets.length === 1 && (targets[0].gpu_list ?? []).length > 0;
  const gpuBody = single
    ? gpuPick === "auto"
      ? { gpus: null }
      : gpuPick === "cpu"
        ? { gpus: 0 }
        : { gpus: gpuPick.length, gpu_indices: gpuPick }
    : { gpus: extra.gpus === "" ? null : Number(extra.gpus) };
  // every class ticked is sent as "all" (also what a version with classes added later means)
  const sentClasses = chosen.length === classes.length ? [] : chosen;
  const noClass = classes.length > 0 && chosen.length === 0;
  const fields = (template?.params ?? []).filter((p) => p.type !== "object");
  // Only the model is picked up front; everything else (epochs, a curriculum's classes and switch …) is under
  // 세부 설정, the settings the kind gives no heading first. The model: the kind's variant, else its first list.
  const lead = template?.variant || fields.find((p) => !p.advanced && p.choices.length > 0)?.name || "";
  const basic = fields.filter((p) => p.name === lead);
  const advanced = fields.filter((p) => p.name !== lead);
  // Hyperparameters under the headings the template gave them, in the template's order.
  const groups: [string, TemplateParam[]][] = [];
  for (const p of advanced) {
    const entry = groups.find(([name]) => name === p.group);
    if (entry) entry[1].push(p);
    else groups.push([p.group, [p]]);
  }
  const pickTemplate = (name: string) => {
    setTrials([]);
    setTemplateName(name);
    setValues(initialValues(templates?.find((t) => t.name === name)));
    setExtra((x) => ({ ...x, environment: "", base: "" }));
  };
  const setExtraField = (key: keyof typeof extra) => (e: { target: { value: string } }) => setExtra({ ...extra, [key]: e.target.value });
  const toggleClass = (id: number) => setChosen(chosen.includes(id) ? chosen.filter((c) => c !== id) : [...chosen, id]);
  const field = (p: TemplateParam) => (
    <ParamField key={`${templateName}.${p.name}`} spec={p} value={values[p.name] ?? ""} onChange={(v) => setValues({ ...values, [p.name]: v })} />
  );
  const collectParams = () => {
    const params: Record<string, unknown> = {};
    for (const p of fields) {
      const v = values[p.name];
      if (p.type === "boolean") params[p.name] = v === true;
      else if (v !== "" && v != null) params[p.name] = p.type === "string" ? String(v).trim() : Number(v);
    }
    return params;
  };

  // Whether the version fits the model, asked as one picks: what would stop the start, and what to know first.
  const [preflight, setPreflight] = useState<{ problems: string[]; notes: string[] } | null>(null);
  const checkKey = JSON.stringify([release, template?.id, sentClasses, nodeMode, pickedNodes, trials.map((t) => t.name), values]);
  useEffect(() => {
    const nodes = nodeMode === "pick" ? pickedNodes : [];
    if (!release || !template || (nodeMode === "pick" && nodes.length === 0)) {
      setPreflight(null);
      return;
    }
    let stale = false;
    const timer = window.setTimeout(() => {
      const space = Object.fromEntries(trials.map((t) => [t.name, {}]));
      void api<{ problems: string[]; notes: string[] }>("/runs/check", {
        body: { release_id: release, template: template.id, params: collectParams(), classes: sentClasses, nodes, space },
      })
        .then((found) => !stale && setPreflight(found))
        .catch(() => !stale && setPreflight(null));
    }, 300);
    return () => {
      stale = true;
      window.clearTimeout(timer);
    };
  }, [checkKey]); // checkKey stands for everything the check reads
  const blocked = (preflight?.problems.length ?? 0) > 0;

  const submit = async (e: FormEvent) => {
    e.preventDefault();
    if (!template) return;
    setBusy(true);
    setError(null);
    const params = collectParams();
    const space: Record<string, SweepSpec> = {};
    for (const t of trials) {
      const parsed = parseTrial(fields.find((p) => p.name === t.name), t.text);
      if (typeof parsed === "string") {
        setError(parsed);
        setBusy(false);
        return;
      }
      space[t.name] = parsed;
      delete params[t.name];
    }
    const ranged = Object.values(space).some((s) => s.min !== undefined);
    if (noClass) {
      setError("학습할 클래스를 하나 이상 고르세요.");
      setBusy(false);
      return;
    }
    if (nodeMode === "pick" && pickedNodes.length === 0) {
      setError("학습할 노드를 하나 이상 고르세요.");
      setBusy(false);
      return;
    }
    const nodes = nodeMode === "pick" ? pickedNodes : [];
    if (settingsFor) {
      // a pipeline step's settings: the model kind by name, so a template keeps working as kinds get new versions
      const config: Record<string, unknown> = {
        template: template.name,
        environment: extra.environment,
        params,
        ...gpuBody,
        classes: sentClasses,
        model_name: extra.model_name,
        base_model_id: extra.base,
        nodes,
        data_after: nodes.length ? dataAfter : "",
        ...(embedding ? { embed: true } : {}),
      };
      if (settingsFor.kind === "sweep") {
        if (trials.length === 0) {
          setError("시험할 설정을 하나 이상 넣으세요 (여러 설정 시험해 보기).");
          setBusy(false);
          return;
        }
        Object.assign(config, { space, method: ranged && method === "grid" ? "random" : method, max_runs: drawCount, parallel, metric: goalMetric });
      }
      settingsFor.onApply(
        Object.fromEntries(Object.entries(config).filter(([, v]) => v !== "" && v != null && !(Array.isArray(v) && v.length === 0))),
      );
      return;
    }
    try {
      if (trials.length > 0) {
        const sweep = await api<{ id: string }>("/sweeps", {
          body: {
            release_id: release,
            template: template.id,
            environment: extra.environment,
            params,
            space,
            method: ranged && method === "grid" ? "random" : method,
            max_runs: drawCount,
            parallel,
            metric: goalMetric,
            ...gpuBody,
            classes: sentClasses,
            name: extra.name,
            base_model_id: extra.base,
            nodes,
            data_after: nodes.length ? dataAfter : "",
          },
        });
        onSweep(sweep.id);
        return;
      }
      const settings = {
        template: template.id,
        environment: extra.environment,
        params,
        ...gpuBody,
        classes: sentClasses,
        name: extra.name,
        model_name: extra.model_name,
        base_model_id: extra.base,
        nodes,
        data_after: nodes.length ? dataAfter : "",
        ...(embedding ? { embed: true } : {}),
      };
      const after = followSteps(follow);
      if (after.length > 0 || save.on) {
        // the training starts now as usual; the Hub takes the steps after it (and keeps them as a pipeline)
        const save_ = save.on ? { save: { name: save.name, trigger: save.trigger } } : {};
        const started = await api<PipelineRunView>("/pipeline-runs", { body: { release_id: release, train: settings, after, ...save_ } });
        rememberFollow(follow);
        onStarted(started.train_run_id ?? "");
        return;
      }
      const run = await api<Run>("/runs", { body: { release_id: release, ...settings } });
      rememberFollow(follow); // nothing ticked is a choice too
      onStarted(run.id);
    } catch (err) {
      setError(err instanceof Error ? err.message : String(err));
      setBusy(false);
    }
  };

  const none = tree != null && !tree.some((d) => d.releases.some((r) => r.status === "ready"));
  const defaultEnv = template?.environment.default && environments.some((e) => e.name === template.environment.default);

  return (
    <Modal title={settingsFor ? (settingsFor.kind === "sweep" ? "탐색 단계 설정" : "학습 단계 설정") : "새 학습"} onClose={onClose}>
      {none ? (
        <>
          <p className="muted">이 프로젝트에는 학습할 버전이 없습니다. 먼저 데이터셋을 가져오세요.</p>
          <div className="modal-actions">
            <a className="btn primary-link" href="#/datasets" onClick={onClose}>
              데이터셋으로 <Icon name="arrow" size={14} />
            </a>
          </div>
        </>
      ) : (
        <form onSubmit={submit}>
          {settingsFor ? (
            // The version is the pipeline's when it runs, not one picked here
            <p className="muted small step-version-note">
              학습할 버전은 고르지 않습니다. 파이프라인이 돌 때의 버전(앞 단계가 새로 만든 버전, 없으면 실행한 버전)으로 학습합니다.
              {releaseInfo ? ` 클래스와 설정은 지금 최신 ${releaseInfo.dataset.name} ${releaseInfo.version} 기준으로 보여 줍니다.` : ""}
            </p>
          ) : (
          <label>
            버전
            <select value={release} onChange={(e) => setRelease(e.target.value)} required>
              {tree?.map((d) => (
                <optgroup key={d.id} label={`${d.name} · ${taskLabel(d.task)}`}>
                  {d.releases
                    .filter((r) => r.status === "ready" && !r.deleted_at)
                    .map((r) => (
                      <option key={r.id} value={r.id}>
                        {d.name} {r.version} · {r.items.toLocaleString()}장
                      </option>
                    ))}
                </optgroup>
              ))}
            </select>
          </label>
          )}
          {templates && templates.length === 0 && (
            <div className="notice">
              이 데이터셋의 태스크({taskLabel(runTask)})를 학습하는 모델 템플릿이 없습니다.{" "}
              {runTask === "pose"
                ? "Kpt 학습은 아직 없습니다. 박스로 학습하려면 같은 폴더를 태스크 Det에 두고 새 데이터셋으로 가져오세요."
                : "이 태스크를 적은 템플릿을 등록하세요."}
            </div>
          )}
          {templates && templates.length > 1 && (
            <label>
              모델 종류
              <select value={templateName} onChange={(e) => pickTemplate(e.target.value)}>
                {templates.map((t) => (
                  <option key={t.id} value={t.name}>
                    {t.title}
                  </option>
                ))}
              </select>
            </label>
          )}
          {template?.description && templates && templates.length > 1 && <p className="muted small">{template.description}</p>}
          {basic.length > 0 && <div className="param-grid basic-params">{basic.map(field)}</div>}
          {releaseInfo?.draft && releaseInfo.draft.base?.id === releaseInfo.id && releaseInfo.draft.edited > 0 && (
            <div className="notice compact">
              작업 중에 고친 이미지 {releaseInfo.draft.edited.toLocaleString()}장은 아직 이 버전에 없습니다. 고친 라벨로 학습하려면{" "}
              <a href={`#/datasets/${releaseInfo.draft.id}`}>작업 중</a>을 버전으로 확정한 뒤 그 버전을 고르세요.
            </div>
          )}
          {(releaseInfo?.stats.states?.unlabelled ?? 0) > 0 && (
            <p className="muted small">
              아직 라벨 안 한 이미지 {(releaseInfo?.stats.states?.unlabelled ?? 0).toLocaleString()}장은 학습에서 뺍니다.
            </p>
          )}
          {choices && (
            <NodeField
              choices={choices}
              mode={nodeMode}
              onMode={setNodeMode}
              picked={pickedNodes}
              onPicked={setPickedNodes}
              dataAfter={dataAfter}
              onDataAfter={setDataAfter}
              sweep={trials.length > 0}
            />
          )}
          {choices && (
            <GpuField
              targets={targets}
              pick={gpuPick}
              onPick={setGpuPick}
              count={extra.gpus}
              onCount={(gpus) => setExtra({ ...extra, gpus })}
            />
          )}
          {choices && trials.length === 0 && settingsFor?.kind !== "sweep" && (
            <EmbedField info={embedInfo} on={embed} onChange={setEmbed} roleless={roleless} later={later} />
          )}

          {settingsFor ? (
            <p className="muted small">
              {settingsFor.kind === "sweep"
                ? "아래 \"여러 설정 시험해 보기\"에 시험할 설정을 넣으세요. 파이프라인이 돌 때 그때의 버전으로 탐색합니다."
                : "파이프라인이 돌 때 그때의 버전으로 이 설정대로 학습합니다. 여기서는 시작하지 않습니다."}
            </p>
          ) : trials.length === 0 ? (
            <>
              <FollowFields
                spec={follow}
                onChange={setFollow}
                tree={tree}
                task={runTask}
                legend="학습 끝나면"
                hint={
                  followSteps(follow).length === 0 ? (
                    <span className="pl-hint pl-lead">평가 · 검수 · 승격은 나중에 모델 화면에서도 할 수 있지만, 여기서 고르면 끝나는 대로 Hub가 합니다.</span>
                  ) : undefined
                }
              />
              <SaveFields save={save} onChange={setSave} />
            </>
          ) : (
            <p className="muted small">탐색 뒤에 이어서 할 일은 파이프라인 화면에서 "탐색 → 최고 모델 평가 → 내보내기"로 만드세요.</p>
          )}

          <details className="advanced">
            <summary>
              세부 설정{" "}
              <span className="muted">
                {advanced.length > 0 ? `(하이퍼파라미터 ${advanced.length}개 · 환경 · 클래스 — 그대로 두면 기본값)` : "(그대로 두면 자동)"}
              </span>
            </summary>
            {groups.map(([group, members]) => (
              <fieldset className="param-group" key={group}>
                {group && <legend>{group}</legend>}
                {/* values first, a few to a row; switches after them, each on a row of its own */}
                <div className="param-grid">
                  {members.filter((p) => p.type !== "boolean").map(field)}
                  {members.filter((p) => p.type === "boolean").map(field)}
                </div>
              </fieldset>
            ))}
            <fieldset className="param-group">
              {groups.some(([group]) => group) && <legend>실행</legend>}
            <div className="form-row">
              <label>
                환경
                <select value={extra.environment} onChange={setExtraField("environment")}>
                  <option value="">자동{defaultEnv ? ` (${template?.environment.default})` : ""}</option>
                  {template?.environment.declared_image && (
                    <option value="docker">Docker 이미지 ({template.environment.declared_image})</option>
                  )}
                  {environments.map((env) => (
                    <option key={env.id} value={env.id}>
                      {env.name}
                      {env.kind === "docker" ? " · Docker" : ""}
                    </option>
                  ))}
                </select>
              </label>
            </div>
            {classes.length > 0 && (
              <>
                <div className="eyebrow class-pick-head">
                  학습할 클래스 — {chosen.length === classes.length ? `전체 ${classes.length}개` : `${chosen.length}/${classes.length}개`}
                  <button type="button" className="link small" onClick={() => setChosen(classes.map((c) => c.id))}>
                    모두 선택
                  </button>
                  <button type="button" className="link small" onClick={() => setChosen([])}>
                    모두 해제
                  </button>
                </div>
                <div className="class-pick">
                  {classes.map((c) => (
                    <button
                      type="button"
                      key={c.id}
                      className={`chip ${chosen.includes(c.id) ? "active" : ""}`}
                      onClick={() => toggleClass(c.id)}
                      aria-pressed={chosen.includes(c.id)}
                    >
                      {c.name}
                    </button>
                  ))}
                </div>
              </>
            )}
            {models.some((m) => m.template.name === templateName) && (
              <label>
                시작 가중치
                <select value={extra.base} onChange={setExtraField("base")}>
                  <option value="">사전학습 가중치 (기본)</option>
                  {models
                    .filter((m) => m.template.name === templateName)
                    .map((m) => (
                      <option key={m.id} value={m.id}>
                        등록된 모델 {m.name} v{m.version}
                        {m.alias ? ` (${m.alias})` : ""}에서 이어서
                      </option>
                    ))}
                </select>
              </label>
            )}
            <div className="form-row">
              <label>
                실행 이름
                <input
                  value={extra.name}
                  onChange={setExtraField("name")}
                  placeholder="자동: 데이터셋-버전-모델-오늘 날짜"
                  title="비우면 데이터셋-버전-모델-오늘 날짜(같은 날 둘째부터 -2). 실행 이름은 Hub 전체에서 하나씩이라, 이미 있는 이름이면 번호가 붙습니다. 나중에 실행 화면에서 바꿀 수 있습니다."
                />
              </label>
              <label>
                모델 이름
                <input value={extra.model_name} onChange={setExtraField("model_name")} placeholder="자동" />
              </label>
            </div>
            </fieldset>
          </details>

          {template && (!settingsFor || settingsFor.kind === "sweep") && (
            <SweepFields
              params={fields}
              task={task}
              trials={trials}
              onTrials={setTrials}
              method={method}
              onMethod={setMethod}
              runs={drawCount}
              onRuns={setDrawCount}
              parallel={parallel}
              onParallel={setParallel}
              goalMetric={goalMetric}
              onGoalMetric={setGoalMetric}
            />
          )}

          <p className="muted small">
            어느 서버든 처음 학습할 때는 환경을 준비하느라 몇 분 더 걸릴 수 있습니다.
            {template?.license ? ` ${template.title}은 ${template.license} 라이선스입니다 — 제품에 탑재하기 전에 확인하세요.` : ""}
          </p>
          {preflight && preflight.notes.length > 0 && <p className="muted small">{preflight.notes.join(" ")}</p>}
          {blocked && (
            <div className="error">
              {preflight?.problems.map((p) => (
                <div key={p}>{p}</div>
              ))}
            </div>
          )}
          {noClass && <div className="error">학습할 클래스를 하나 이상 고르세요 (세부 설정).</div>}
          {error && <div className="error">{error}</div>}
          <div className="modal-actions">
            <button type="button" onClick={onClose}>
              취소
            </button>
            <button className="primary" disabled={busy || !release || !template || blocked || noClass}>
              {settingsFor ? "설정 적용" : trials.length > 0 ? "탐색 시작" : "학습 시작"}
            </button>
          </div>
        </form>
      )}
    </Modal>
  );
}
