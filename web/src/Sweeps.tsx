import { api } from "./api";
import { RunTable } from "./RunTable";
import { METRIC_LABEL, type Run, type RunStatus, type TemplateParam, keyMetrics, metric } from "./trainTypes";

import { Badge, Icon, Stat, duration, timeAgo, usePolling } from "./ui";

export interface SweepSpec {
  values?: unknown[];
  min?: number;
  max?: number;
  log?: boolean;
}

export interface Sweep {
  id: string;
  name: string;
  project_id: string;
  task: string;
  status: "running" | "finished" | "stopped";
  error: string | null;
  method: "grid" | "random";
  metric: string;
  goal: "maximize" | "minimize";
  space: Record<string, SweepSpec>;
  fixed: Record<string, unknown>;
  parallel: number;
  total: number;
  started: number;
  active: number;
  finished: number;
  failed: number;
  best: { run_id: string; name: string; value: number; settings: Record<string, unknown> } | null;
  release: { id: string; dataset: string; version: string } | null;
  template: { name: string; title: string; version: string };
  created_at: string;
  finished_at: string | null;
  points?: { run_id: string; name: string; status: RunStatus; settings: Record<string, unknown>; value: number | null }[];
}

const STATUS_KIND: Record<Sweep["status"], string> = { running: "running", finished: "succeeded", stopped: "canceled" };
const STATUS_TEXT: Record<Sweep["status"], string> = { running: "진행 중", finished: "끝남", stopped: "멈춤" };
export const METHOD_LABEL: Record<string, string> = { grid: "모든 조합", random: "무작위" };

const show = (v: unknown) => (typeof v === "number" ? String(Number(v.toPrecision(4))) : String(v));
export const settingsText = (settings: Record<string, unknown>) =>
  Object.entries(settings)
    .map(([k, v]) => `${k} ${show(v)}`)
    .join(" · ");

// ---- building a search in the new-run dialog ---------------------------------------------

export interface Trial {
  name: string;
  text: string; // "0.01, 0.001" or "0.0001~0.01"
}

/** What a person typed for one setting: listed values, or a range (``a~b``) for random draws. */
export function parseTrial(spec: TemplateParam | undefined, text: string): SweepSpec | string {
  if (!spec) return "설정을 고르세요";
  const numeric = spec.type === "integer" || spec.type === "number";
  if (text.includes("~")) {
    if (!numeric) return `${spec.label}: 범위는 숫자 설정에만 쓸 수 있습니다`;
    const [low, high] = text.split("~").map((s) => Number(s.trim()));
    if (!Number.isFinite(low) || !Number.isFinite(high) || low >= high) return `${spec.label}: 범위는 작은값~큰값`;
    // A range spanning orders of magnitude (a learning rate) is drawn on a log scale.
    return { min: low, max: high, log: low > 0 && high / low >= 20 };
  }
  const parts = text
    .split(",")
    .map((s) => s.trim())
    .filter(Boolean);
  if (parts.length === 0) return `${spec.label}: 시험할 값을 적으세요`;
  const values: unknown[] = [];
  for (const part of parts) {
    if (numeric) {
      const n = Number(part);
      if (!Number.isFinite(n)) return `${spec.label}: ${part}는 숫자가 아닙니다`;
      values.push(spec.type === "integer" ? Math.round(n) : n);
    } else if (spec.type === "boolean") {
      values.push(["true", "예", "1", "yes"].includes(part.toLowerCase()));
    } else values.push(part);
  }
  return { values: [...new Set(values)] };
}

function hint(spec: TemplateParam | undefined): string {
  if (!spec) return "";
  if (spec.choices.length > 0) return `예: ${spec.choices.slice(0, 3).map(String).join(", ")}`;
  if (spec.type === "number") return "예: 0.01, 0.001  또는  0.0001~0.01";
  if (spec.type === "integer") return "예: 16, 32, 64  또는  8~64";
  if (spec.type === "boolean") return "true, false";
  return "쉼표로 여러 값";
}

export function SweepFields({
  params,
  task,
  trials,
  onTrials,
  method,
  onMethod,
  runs,
  onRuns,
  parallel,
  onParallel,
  goalMetric,
  onGoalMetric,
}: {
  params: TemplateParam[];
  task: string;
  trials: Trial[];
  onTrials: (t: Trial[]) => void;
  method: "grid" | "random";
  onMethod: (m: "grid" | "random") => void;
  runs: number;
  onRuns: (n: number) => void;
  parallel: number;
  onParallel: (n: number) => void;
  goalMetric: string;
  onGoalMetric: (m: string) => void;
}) {
  const tunable = params.filter((p) => p.type !== "object");
  const parsed = trials.map((t) => parseTrial(tunable.find((p) => p.name === t.name), t.text));
  const ranged = parsed.some((p) => typeof p !== "string" && p.min !== undefined);
  const combos = parsed.reduce((n, p) => n * (typeof p === "string" ? 1 : (p.values?.length ?? 1)), 1);
  const used = new Set(trials.map((t) => t.name));
  const update = (i: number, change: Partial<Trial>) => onTrials(trials.map((t, j) => (j === i ? { ...t, ...change } : t)));
  const add = () => {
    const next = tunable.find((p) => !used.has(p.name));
    if (next) onTrials([...trials, { name: next.name, text: "" }]);
  };
  return (
    <details className="advanced" open={trials.length > 0}>
      <summary>
        여러 설정 시험해 보기 <span className="muted">(탐색 — 설정마다 따로 학습해 가장 좋은 것을 찾습니다)</span>
      </summary>
      {trials.map((t, i) => (
        <div className="sweep-row" key={t.name}>
          <label>
            설정
            <select value={t.name} onChange={(e) => update(i, { name: e.target.value, text: "" })}>
              {tunable
                .filter((p) => p.name === t.name || !used.has(p.name))
                .map((p) => (
                  <option key={p.name} value={p.name}>
                    {p.label || p.name}
                  </option>
                ))}
            </select>
          </label>
          <label>
            시험할 값
            <input
              value={t.text}
              onChange={(e) => update(i, { text: e.target.value })}
              placeholder={hint(tunable.find((p) => p.name === t.name))}
              className="mono"
            />
          </label>
          <button type="button" className="icon-btn" onClick={() => onTrials(trials.filter((_, j) => j !== i))} aria-label="빼기">
            <Icon name="x" size={14} />
          </button>
          {typeof parsed[i] === "string" && t.text !== "" && <div className="field-error">{parsed[i] as string}</div>}
        </div>
      ))}
      {trials.length < tunable.length && (
        <button type="button" className="link small" onClick={add}>
          + 시험할 설정
        </button>
      )}
      {trials.length > 0 && (
        <div className="form-row">
          <label>
            방식
            <select value={ranged ? "random" : method} onChange={(e) => onMethod(e.target.value as "grid" | "random")} disabled={ranged}>
              <option value="grid">모든 조합 ({combos}회)</option>
              <option value="random">무작위로 뽑기</option>
            </select>
          </label>
          {(ranged || method === "random") && (
            <label>
              실행 수
              <input type="number" min={1} max={100} value={runs} onChange={(e) => onRuns(Number(e.target.value))} />
            </label>
          )}
          <label>
            동시에 돌릴 수
            <input type="number" min={1} max={8} value={parallel} onChange={(e) => onParallel(Number(e.target.value))} />
          </label>
          <label>
            가장 좋은 것의 기준
            <select value={goalMetric} onChange={(e) => onGoalMetric(e.target.value)}>
              {keyMetrics(task).map((k) => (
                <option key={k} value={k}>
                  {METRIC_LABEL[k] ?? k} 높은 것
                </option>
              ))}
            </select>
          </label>
        </div>
      )}
    </details>
  );
}

// ---- the searches of a project, above the run table --------------------------------------

export function SweepList({ projectId }: { projectId: string }) {
  const { data: sweeps } = usePolling(
    () => (projectId ? api<Sweep[]>(`/projects/${projectId}/sweeps`) : Promise.resolve([])),
    10000,
    [projectId],
  );
  if (!sweeps || sweeps.length === 0) return null;
  return (
    <details className="card fold" open={sweeps.some((s) => s.status === "running")}>
      <summary>
        하이퍼파라미터 탐색 <span className="muted">({sweeps.length})</span>
      </summary>
      <div className="table-wrap">
        <table className="plain">
          <thead>
            <tr>
              <th>탐색</th>
              <th>상태</th>
              <th className="right">진행</th>
              <th>가장 좋은 설정</th>
              <th className="right">시작</th>
            </tr>
          </thead>
          <tbody>
            {sweeps.map((s) => (
              <tr key={s.id} className="clickable" onClick={() => (window.location.hash = `#/training/sweep/${s.id}`)}>
                <td>
                  <strong>{s.name}</strong>
                  <div className="muted small">
                    {Object.keys(s.space).join(", ")} · {METHOD_LABEL[s.method]}
                  </div>
                </td>
                <td>
                  <Badge kind={STATUS_KIND[s.status]}>{STATUS_TEXT[s.status]}</Badge>
                </td>
                <td className="num right">
                  {s.finished + s.failed}/{s.total}
                </td>
                <td>
                  {s.best ? (
                    <>
                      <span className="num">
                        {METRIC_LABEL[s.metric] ?? s.metric} {metric(s.best.value)}
                      </span>
                      <div className="muted small mono">{settingsText(s.best.settings)}</div>
                    </>
                  ) : (
                    <span className="muted">–</span>
                  )}
                </td>
                <td className="muted right">{timeAgo(s.created_at)}</td>
              </tr>
            ))}
          </tbody>
        </table>
      </div>
    </details>
  );
}

// ---- one search ----------------------------------------------------------------------------

/** The objective against one setting: where the good runs sit. Text values are spread evenly. */
function SettingChart({
  name,
  points,
  bestId,
  label,
}: {
  name: string;
  points: NonNullable<Sweep["points"]>;
  bestId?: string;
  label: string;
}) {
  const scored = points.filter((p) => p.value != null && p.settings[name] !== undefined);
  const W = 280;
  const H = 170;
  const pad = { l: 44, r: 12, t: 12, b: 30 };
  const numeric = scored.every((p) => typeof p.settings[name] === "number");
  const categories = numeric ? [] : [...new Set(scored.map((p) => String(p.settings[name])))];
  const xs = scored.map((p) => Number(p.settings[name]));
  const logScale = numeric && xs.length > 1 && Math.min(...xs) > 0 && Math.max(...xs) / Math.min(...xs) >= 20;
  const fx = (v: number) => (logScale ? Math.log10(v) : v);
  const xMin = numeric ? Math.min(...xs.map(fx)) : 0;
  const xMax = numeric ? Math.max(...xs.map(fx)) : Math.max(1, categories.length - 1);
  const ys = scored.map((p) => p.value as number);
  const yMin = Math.min(...ys);
  const yMax = Math.max(...ys);
  const sx = (p: (typeof scored)[number]) => {
    const t = numeric ? fx(Number(p.settings[name])) : categories.indexOf(String(p.settings[name]));
    return pad.l + (xMax === xMin ? 0.5 : (t - xMin) / (xMax - xMin)) * (W - pad.l - pad.r);
  };
  const sy = (v: number) => H - pad.b - (yMax === yMin ? 0.5 : (v - yMin) / (yMax - yMin)) * (H - pad.t - pad.b);
  const ticks = numeric
    ? [xMin, xMax].map((t) => ({ x: pad.l + (xMax === xMin ? 0.5 : (t - xMin) / (xMax - xMin)) * (W - pad.l - pad.r), text: show(logScale ? 10 ** t : t) }))
    : categories.map((c, i) => ({ x: pad.l + (xMax === 0 ? 0.5 : i / xMax) * (W - pad.l - pad.r), text: c }));
  return (
    <section className="card epoch-chart">
      <div className="epoch-head">
        <h3>{name}</h3>
        <span className="muted small">{logScale ? "로그 눈금" : ""}</span>
      </div>
      {scored.length === 0 ? (
        <p className="muted small">끝난 실행이 생기면 보입니다.</p>
      ) : (
        <svg viewBox={`0 0 ${W} ${H}`} className="sweep-chart" role="img" aria-label={`${name}에 따른 ${label}`}>
          <line x1={pad.l} y1={H - pad.b} x2={W - pad.r} y2={H - pad.b} className="grid-line" />
          <line x1={pad.l} y1={pad.t} x2={pad.l} y2={H - pad.b} className="grid-line" />
          {[yMin, yMax].map((v) => (
            <text key={v} x={pad.l - 6} y={sy(v) + 4} textAnchor="end" className="axis-text">
              {metric(v)}
            </text>
          ))}
          {ticks.map((t) => (
            <text key={`${t.x}-${t.text}`} x={t.x} y={H - pad.b + 16} textAnchor="middle" className="axis-text">
              {t.text.length > 10 ? `${t.text.slice(0, 9)}…` : t.text}
            </text>
          ))}
          {scored.map((p) => (
            <a key={p.run_id} href={`#/training/${p.run_id}`}>
              <circle cx={sx(p)} cy={sy(p.value as number)} r={p.run_id === bestId ? 6 : 4} className={p.run_id === bestId ? "best" : ""}>
                <title>
                  {p.name}: {label} {metric(p.value)}
                </title>
              </circle>
            </a>
          ))}
        </svg>
      )}
    </section>
  );
}

export function SweepView({ sweepId }: { sweepId: string }) {
  const { data: sweep, error, refresh } = usePolling(() => api<Sweep>(`/sweeps/${sweepId}`), 3000, [sweepId]);
  const projectId = sweep?.project_id ?? "";
  const { data: runs } = usePolling(
    () => (projectId ? api<Run[]>(`/projects/${projectId}/runs?sweep=${sweepId}`) : Promise.resolve([])),
    sweep?.status === "running" ? 3000 : 20000,
    [projectId, sweepId],
  );
  if (error && !sweep) return <div className="error">{error}</div>;
  if (!sweep) return <div className="muted">불러오는 중…</div>;

  const label = METRIC_LABEL[sweep.metric] ?? sweep.metric;
  const stop = async () => {
    if (!window.confirm("탐색을 멈출까요? 더 시작하지 않고, 도는 실행은 중단합니다. 끝난 실행과 모델은 남습니다.")) return;
    await api(`/sweeps/${sweep.id}/stop`, { method: "POST" });
    await refresh();
  };
  const names = Object.keys(sweep.space);
  const spaceText = names
    .map((n) => {
      const s = sweep.space[n];
      return s.values ? `${n} ${s.values.map(show).join(" · ")}` : `${n} ${show(s.min)}~${show(s.max)}${s.log ? " (로그)" : ""}`;
    })
    .join("  /  ");

  return (
    <>
      <a className="link back" href="#/training">
        ← 학습 목록
      </a>
      <div className="page-head">
        <div>
          <h1 className="with-badge">
            {sweep.name}
            <Badge kind={STATUS_KIND[sweep.status]}>{STATUS_TEXT[sweep.status]}</Badge>
          </h1>
          <p className="muted">
            {sweep.release ? `${sweep.release.dataset} ${sweep.release.version}` : "삭제된 릴리스"} · {sweep.template.title} ·{" "}
            {METHOD_LABEL[sweep.method]} {sweep.total}회 · {label} {sweep.goal === "maximize" ? "높은" : "낮은"} 것이 좋음 ·{" "}
            {timeAgo(sweep.created_at)}
          </p>
        </div>
        <div className="head-actions">
          {sweep.status === "running" && (
            <button className="danger" onClick={stop}>
              탐색 멈추기
            </button>
          )}
        </div>
      </div>
      {sweep.error && <div className="error">{sweep.error}</div>}

      <div className="stats three metrics">
        <Stat
          icon="activity"
          label="진행"
          value={`${sweep.finished + sweep.failed} / ${sweep.total}`}
          sub={`완료 ${sweep.finished} · 실패 ${sweep.failed} · 실행 중 ${sweep.active}`}
        />
        <Stat
          icon="chart"
          label={`최고 ${label}`}
          value={sweep.best ? metric(sweep.best.value) : "–"}
          sub={
            sweep.best ? (
              <>
                <a href={`#/training/${sweep.best.run_id}`}>{sweep.best.name}</a> · {settingsText(sweep.best.settings)}
              </>
            ) : (
              "끝난 실행 없음"
            )
          }
        />
        <Stat
          icon="clock"
          label="걸린 시간"
          value={sweep.finished_at ? duration((Date.parse(sweep.finished_at) - Date.parse(sweep.created_at)) / 1000) : "–"}
          sub={`동시에 ${sweep.parallel}개`}
        />
      </div>

      <section className="card">
        <h3>시험한 설정</h3>
        <p className="mono small">{spaceText}</p>
        {Object.keys(sweep.fixed).length > 0 && <p className="muted small">공통: {settingsText(sweep.fixed)}</p>}
      </section>

      <div className="mini-charts">
        {names.map((n) => (
          <SettingChart key={n} name={n} points={sweep.points ?? []} bestId={sweep.best?.run_id} label={label} />
        ))}
      </div>

      <RunTable
        runs={runs ?? null}
        projectId={projectId}
        task={sweep.task}
        filtered={false}
        storageKey={`badamoa.sweepColumns.${sweep.id}`}
        defaults={[...names.map((n) => `s:${n}`), `m:${sweep.metric}`, "i:duration", "i:node"]}
      />
    </>
  );
}
