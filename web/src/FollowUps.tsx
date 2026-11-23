import { type ReactNode, useEffect, useState } from "react";
import { api } from "./api";
import type { TreeDataset } from "./catalogTypes";
import { METRIC_LABEL, keyMetrics } from "./trainTypes";
import { Badge, Icon, Modal, timeAgo, usePolling } from "./ui";
import "./pipelines.css";

// ---- what the Hub sends about pipelines ----------------------------------------------------

export type StepKind = "train" | "eval" | "promote" | "export";
export interface StepSpec {
  kind: StepKind;
  config: Record<string, unknown>;
}
type Ref = { id: string; name: string; version: number; aliases?: string[]; deleted?: boolean };

export interface PipelineStepView {
  kind: StepKind;
  label: string;
  status: "pending" | "running" | "succeeded" | "failed" | "skipped" | "stopped" | "held";
  message: string;
  summary: string;
  config: Record<string, unknown>;
  started_at: string | null;
  finished_at: string | null;
  run?: { id: string; name: string; epoch: number; epochs: number } | null;
  job?: { id: string; status: string; wait_reason?: string } | null;
  model?: Ref | null;
  release?: { id: string; version: string; dataset: string } | null;
  split?: string;
  op_id?: string;
  metric?: { key: string; label: string; value?: number | null } | null;
  before?: { version: string; value: number } | null;
  review_release_id?: string;
  alias?: string;
  outcome?: "promoted" | "held" | null;
  mine?: number | null;
  theirs?: number | null;
  holder?: { id: string; name: string; version: number; op_id: string } | null;
}

export interface PipelineRunView {
  id: string;
  name: string;
  status: "running" | "succeeded" | "held" | "failed" | "stopped";
  error: string | null;
  trigger: "manual" | "new_version" | "training";
  project: { id: string; name: string };
  pipeline: { id: string; name: string; deleted: boolean } | null;
  release: { id: string; version: string; dataset: string; dataset_id: string; task: string; deleted: boolean } | null;
  train_run_id: string | null;
  created_by: string | null;
  created_at: string;
  finished_at: string | null;
  duration_s: number;
  steps: PipelineStepView[];
}

export const RUN_STATUS: Record<PipelineRunView["status"], [string, string]> = {
  running: ["진행 중", "running"],
  succeeded: ["완료", "succeeded"],
  held: ["승격 안 함", "queued"],
  failed: ["실패", "failed"],
  stopped: ["멈춤", "idle"],
};
/** 와/과 after a word: by its last sound (Hangul by its final consonant; a Latin word ending in l, m, n or ng too). */
export function wa(word: string): string {
  const last = word.trim().slice(-1);
  const code = last.charCodeAt(0) - 0xac00;
  if (code >= 0 && code < 11172) return code % 28 ? "과" : "와";
  return /(?:[lmn]|ng)$/i.test(word.trim()) ? "과" : "와";
}

export const TRIGGER_LABEL: Record<string, string> = { manual: "직접 실행", new_version: "새 버전이 생겨서", training: "새 학습에서" };
const STEP_STATUS: Record<PipelineStepView["status"], string> = {
  pending: "대기",
  running: "진행 중",
  succeeded: "완료",
  failed: "실패",
  skipped: "건너뜀",
  stopped: "멈춤",
  held: "그대로 둠",
};

export function RunStatusBadge({ status }: { status: PipelineRunView["status"] }) {
  const [label, kind] = RUN_STATUS[status] ?? [status, "idle"];
  return <Badge kind={kind}>{label}</Badge>;
}

/** Where a step's work is seen: its training run, its evaluation, the model it promoted or exported. */
export function stepHref(s: PipelineStepView): string | undefined {
  if (s.kind === "train" && s.run) return `#/training/${s.run.id}`;
  if (s.kind === "eval" && s.op_id) return `#/models/eval/${s.op_id}`;
  if ((s.kind === "promote" || s.kind === "export") && s.model) return `#/models/${s.model.id}`;
  return undefined;
}

/** A step's state in a few words: epochs while training, the number it measured, what the gate did. */
export function stepNote(s: PipelineStepView): string {
  if (s.status === "running") {
    if (s.job?.status === "queued") return "차례 기다림";
    if (s.kind === "train" && s.run && s.run.epochs > 0) return `${s.run.epoch}/${s.run.epochs}`;
    return "";
  }
  if (s.kind === "eval" && s.metric?.value != null) return s.metric.value.toFixed(3);
  if (s.kind === "promote" && s.status === "succeeded" && s.outcome === "promoted") return `→ ${s.alias}`;
  if (s.kind === "train" && s.model) return `v${s.model.version}`;
  return "";
}

/** The steps in a row: 학습 ✓ → 평가 · 검수 ● → 승격 ○. Each links to what it made. */
export function StepChain({ steps, plain = false }: { steps: (PipelineStepView | { kind: StepKind; label: string; summary?: string })[]; plain?: boolean }) {
  return (
    <ol className="pl-chain" aria-label="단계">
      {steps.map((s, i) => {
        const live = "status" in s ? s : null;
        const href = live && !plain ? stepHref(live) : undefined;
        const note = live ? stepNote(live) : "";
        const title = live
          ? [`${s.label}: ${STEP_STATUS[live.status]}`, live.message || live.summary, live.job?.wait_reason].filter(Boolean).join("\n")
          : s.summary;
        const body = (
          <>
            <i />
            {s.label}
            {note && <span className="num">{note}</span>}
          </>
        );
        return (
          <li key={`${s.kind}-${i}`}>
            {href ? (
              <a className={`pl-step ${live?.status ?? "plan"}`} href={href} title={title}>
                {body}
              </a>
            ) : (
              <span className={`pl-step ${live?.status ?? "plan"}`} title={title}>
                {body}
              </span>
            )}
          </li>
        );
      })}
    </ol>
  );
}

// ---- "학습 끝나면": the steps after a training --------------------------------------------

export interface FollowSpec {
  eval: boolean;
  release: string; // "" = the version it trained on
  split: string; // auto | val | test | train
  review: boolean;
  promote: boolean;
  alias: string;
  export: boolean;
}

const FOLLOW_KEY = "badamoa.follow-ups"; // the last choice, offered again next time (this browser only)
const NO_FOLLOW: FollowSpec = { eval: false, release: "", split: "auto", review: true, promote: false, alias: "SOTA", export: false };

export function rememberedFollow(): FollowSpec {
  try {
    const kept = JSON.parse(localStorage.getItem(FOLLOW_KEY) ?? "null");
    if (kept && typeof kept === "object") return { ...NO_FOLLOW, ...kept, release: "" };
  } catch {
    /* none kept */
  }
  return { ...NO_FOLLOW, eval: true }; // first time: evaluating is offered, already ticked
}

export function rememberFollow(spec: FollowSpec) {
  try {
    localStorage.setItem(FOLLOW_KEY, JSON.stringify({ ...spec, release: "" }));
  } catch {
    /* not kept */
  }
}

/** The steps a choice makes, in the Hub's order (evaluate → promote → export). */
export function followSteps(spec: FollowSpec): StepSpec[] {
  const steps: StepSpec[] = [];
  if (spec.eval) steps.push({ kind: "eval", config: { release_id: spec.release, split: spec.split, review: spec.review } });
  if (spec.eval && spec.promote) steps.push({ kind: "promote", config: { alias: spec.alias.trim() || "SOTA" } });
  if (spec.export) steps.push({ kind: "export", config: { format: "onnx" } });
  return steps;
}

export const SPLIT_CHOICES: [string, string][] = [
  ["auto", "test (없으면 val)"],
  ["val", "val"],
  ["test", "test"],
  ["train", "train"],
];

/** The follow-up choices: evaluate (on which version and split, and for 검수), promote if better, export. */
export function FollowFields({
  spec,
  onChange,
  tree,
  task,
  legend = "학습 끝나면",
  hint,
  optional = true,
}: {
  spec: FollowSpec;
  onChange: (spec: FollowSpec) => void;
  tree: TreeDataset[] | null; // versions the evaluation may use instead (a fixed test set)
  task: string;
  legend?: string;
  hint?: ReactNode;
  optional?: boolean; // a new training may have none; a pipeline or "이어서 하기" is about them
}) {
  const set = (part: Partial<FollowSpec>) => onChange({ ...spec, ...part });
  const metric = METRIC_LABEL[keyMetrics(task)[0]] ?? "지표";
  const others = (tree ?? []).filter((d) => d.task === task || !task);
  return (
    <fieldset className="pl-follow">
      <legend>
        {legend} <span className="muted">— Hub가 이어서 합니다{optional ? " (선택)" : ""}</span>
      </legend>
      {hint}
      <div className="pl-check">
        <input id="fu-eval" type="checkbox" checked={spec.eval} onChange={(e) => set({ eval: e.target.checked })} />
        <label htmlFor="fu-eval">평가</label>
        {spec.eval && (
          <>
            <select value={spec.release} onChange={(e) => set({ release: e.target.value })} aria-label="평가할 버전">
              <option value="">학습한 버전</option>
              {others.map((d) => (
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
            <select value={spec.split} onChange={(e) => set({ split: e.target.value })} aria-label="평가할 split">
              {SPLIT_CHOICES.map(([value, label]) => (
                <option key={value} value={value}>
                  {label}
                </option>
              ))}
            </select>
          </>
        )}
      </div>
      {spec.eval && (
        <div className="pl-check pl-sub">
          <input id="fu-review" type="checkbox" checked={spec.review} onChange={(e) => set({ review: e.target.checked })} />
          <label htmlFor="fu-review">검수에도 쓰기</label>
          <span className="pl-hint">학습한 버전의 모든 이미지를 함께 평가해, 검수에서 틀렸을 법한 라벨부터 봅니다</span>
        </div>
      )}
      <div className={`pl-check${spec.eval ? "" : " off"}`}>
        <input
          id="fu-promote"
          type="checkbox"
          checked={spec.eval && spec.promote}
          disabled={!spec.eval}
          onChange={(e) => set({ promote: e.target.checked })}
        />
        <label htmlFor="fu-promote">더 나으면</label>
        <input
          className="pl-alias"
          value={spec.alias}
          onChange={(e) => set({ alias: e.target.value })}
          disabled={!spec.eval || !spec.promote}
          aria-label="alias"
          list="fu-aliases"
        />
        <datalist id="fu-aliases">
          <option value="SOTA" />
          <option value="production" />
          <option value="staging" />
        </datalist>
        <span>(으)로 지정</span>
        <span className="pl-hint">
          {spec.eval ? `지금 ${spec.alias || "SOTA"}${wa(spec.alias || "SOTA")} 같은 데이터로 재서 ${metric}이 더 높을 때만` : "평가가 있어야 비교합니다"}
        </span>
      </div>
      <div className="pl-check">
        <input id="fu-export" type="checkbox" checked={spec.export} onChange={(e) => set({ export: e.target.checked })} />
        <label htmlFor="fu-export">ONNX로 내보내기</label>
        {spec.eval && spec.promote && spec.export && <span className="pl-hint">승격됐을 때만</span>}
      </div>
    </fieldset>
  );
}

export interface SaveSpec {
  on: boolean;
  name: string;
  trigger: "manual" | "new_version";
}

/** "파이프라인으로 저장": the same way kept, to run again or on each new version of the dataset. */
export function SaveFields({ save, onChange, forced = false }: { save: SaveSpec; onChange: (save: SaveSpec) => void; forced?: boolean }) {
  const set = (part: Partial<SaveSpec>) => onChange({ ...save, ...part });
  return (
    <div className="pl-save">
      {!forced && (
        <div className="pl-check">
          <input id="fu-save" type="checkbox" checked={save.on} onChange={(e) => set({ on: e.target.checked })} />
          <label htmlFor="fu-save">파이프라인으로 저장</label>
          <span className="pl-hint">같은 설정과 단계를 파이프라인 화면에서 다시 돌리거나, 새 버전마다 자동으로</span>
        </div>
      )}
      {(save.on || forced) && (
        <div className={`form-row${forced ? "" : " pl-sub"}`}>
          <label>
            파이프라인 이름
            <input value={save.name} onChange={(e) => set({ name: e.target.value })} placeholder="자동: 데이터셋 학습 → 평가 → …" />
          </label>
          <label>
            언제 돌까
            <select value={save.trigger} onChange={(e) => set({ trigger: e.target.value as SaveSpec["trigger"] })}>
              <option value="manual">직접 실행할 때만</option>
              <option value="new_version">이 데이터셋에 새 버전이 생기면 자동으로</option>
            </select>
          </label>
        </div>
      )}
    </div>
  );
}

// ---- on a training run's screen -------------------------------------------------------------

interface Follow {
  pipeline: PipelineRunView | null;
  evaluations: number;
}

/** What follows a training run: the pipeline it starts (its steps as they go), else a way to add them — and a
 * gentle nudge when a finished run's model was never evaluated. */
export function FollowBanner({
  runId,
  active,
  succeeded,
  hasModel,
  task,
  projectId,
}: {
  runId: string;
  active: boolean;
  succeeded: boolean;
  hasModel: boolean;
  task: string;
  projectId: string;
}) {
  const [adding, setAdding] = useState(false);
  const [going, setGoing] = useState(active);
  const { data: follow, refresh } = usePolling(() => api<Follow>(`/runs/${runId}/follow`), going ? 4000 : 30000, [runId]);
  useEffect(() => setGoing(active || follow?.pipeline?.status === "running"), [active, follow]);
  if (!follow) return null;
  const p = follow.pipeline;
  let body: ReactNode = null;
  if (p) {
    body = (
      <div className="card pl-banner">
        <div className="pl-banner-head">
          <Icon name="flow" size={15} />
          <strong>{p.pipeline ? `파이프라인 ${p.pipeline.name}` : "학습 끝나면"}</strong>
          <RunStatusBadge status={p.status} />
          <a className="link small pl-banner-link" href={`#/pipelines/run/${p.id}`}>
            파이프라인에서 보기 <Icon name="arrow" size={12} />
          </a>
        </div>
        <StepChain steps={p.steps} />
        {p.error && <div className="pl-banner-note bad-text small">{p.error}</div>}
        {!p.error && p.status !== "running" && (
          <div className="pl-banner-note muted small">
            {(p.steps.find((s) => s.kind === "promote" && s.message) ?? p.steps.filter((s) => s.message).at(-1))?.message}
          </div>
        )}
      </div>
    );
  } else if (active) {
    body = (
      <div className="pl-nudge muted small">
        <Icon name="flow" size={13} /> 학습 끝나면 할 일이 없습니다.
        <button type="button" className="link small" onClick={() => setAdding(true)}>
          평가 · 검수 · 승격 붙이기
        </button>
      </div>
    );
  } else if (succeeded && hasModel && follow.evaluations === 0) {
    body = (
      <div className="notice pl-nudge-card">
        <div>
          <strong>이 모델은 아직 평가하지 않았습니다.</strong> 평가하면 다른 모델과 같은 데이터로 비교하고, 검수에서 틀렸을 법한 라벨부터 보고, 지금
          SOTA보다 나으면 바로 지정할 수 있습니다.
        </div>
        <button type="button" className="primary small" onClick={() => setAdding(true)}>
          이어서 하기
        </button>
      </div>
    );
  }
  return (
    <>
      {body}
      {adding && (
        <FollowDialog
          runId={runId}
          task={task}
          projectId={projectId}
          onClose={() => setAdding(false)}
          onStarted={() => {
            setAdding(false);
            setGoing(true);
            void refresh();
          }}
        />
      )}
    </>
  );
}

/** The steps after a training run that is there already (going or done), optionally kept as a pipeline. */
function FollowDialog({
  runId,
  task,
  projectId,
  onClose,
  onStarted,
}: {
  runId: string;
  task: string;
  projectId: string;
  onClose: () => void;
  onStarted: (run: PipelineRunView) => void;
}) {
  const [spec, setSpec] = useState<FollowSpec>(() => ({ ...rememberedFollow(), eval: true }));
  const [save, setSave] = useState<SaveSpec>({ on: false, name: "", trigger: "manual" });
  const [tree, setTree] = useState<TreeDataset[] | null>(null);
  const [error, setError] = useState<string | null>(null);
  const [busy, setBusy] = useState(false);
  useEffect(() => {
    void api<TreeDataset[]>(`/projects/${projectId}/datasets`).then(setTree);
  }, [projectId]);
  const after = followSteps(spec);
  const submit = async () => {
    setBusy(true);
    setError(null);
    try {
      const body = { run_id: runId, after, ...(save.on ? { save: { name: save.name, trigger: save.trigger } } : {}) };
      const started = await api<PipelineRunView>("/pipeline-runs", { body });
      rememberFollow(spec);
      onStarted(started);
    } catch (e) {
      setError(e instanceof Error ? e.message : String(e));
      setBusy(false);
    }
  };
  return (
    <Modal title="이어서 하기" onClose={onClose}>
      <p className="muted small">이 학습이 끝나면(끝났으면 바로) Hub가 차례로 합니다. 한 단계가 실패하면 멈추고 이유를 남깁니다.</p>
      <FollowFields spec={spec} onChange={setSpec} tree={tree} task={task} legend="할 일" optional={false} />
      <SaveFields save={save} onChange={setSave} />
      {error && <div className="error">{error}</div>}
      <div className="modal-actions">
        <button type="button" onClick={onClose}>
          취소
        </button>
        <button type="button" className="primary" disabled={busy || after.length === 0} onClick={() => void submit()}>
          {after.length === 0 ? "할 일을 고르세요" : "이어서 하기"}
        </button>
      </div>
    </Modal>
  );
}

export const whenStarted = (r: PipelineRunView) => `${TRIGGER_LABEL[r.trigger] ?? r.trigger} · ${r.created_by ?? "?"} · ${timeAgo(r.created_at)}`;
