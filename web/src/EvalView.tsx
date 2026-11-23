import { useState } from "react";
import { FileList } from "./Files";
import { LoggedTable } from "./Media";
import { PlacePicker, usePlaces } from "./ModelOps";
import { ClassTable, LogTail } from "./Training";
import { api } from "./api";
import { SPLIT_LABEL } from "./catalogTypes";
import { type ClassMetric, METRIC_LABEL, type ModelOp, classScore, keyMetrics, metric } from "./trainTypes";
import { Badge, Modal, STATUS_LABEL, Stat, duration, timeAgo, usePolling } from "./ui";

const ACTIVE = ["queued", "running"];

interface ModelRef {
  id: string;
  name: string;
  version: number;
  task: string;
  aliases: string[];
  project_id: string;
  deleted: boolean;
}

/** One evaluation (export, test) as its own screen shows it. */
interface OpDetail extends ModelOp {
  model: ModelRef | null;
  splits: string[];
  split_metrics: Record<string, Record<string, number>>;
  class_metrics: ClassMetric[];
  node_id: string | null;
  node_name: string | null;
  started_at: string | null;
  finished_at: string | null;
  log_count: number;
  gpus: number | null;
}

/** An evaluation on a version's split, as the comparison board lists it. */
interface BoardRow {
  op_id: string;
  model: ModelRef | null;
  split: string;
  metrics: Record<string, number>;
  class_metrics: ClassMetric[];
  created_at: string;
  duration_s: number | null;
  latest: boolean;
}

interface EvalBoardInfo {
  release: { id: string; dataset: string; version: string };
  task: string;
  splits: string[];
  rows: BoardRow[];
}

const splitName = (s: string) => SPLIT_LABEL[s] ?? s;
const modelName = (m: ModelRef | null) => (m ? `${m.name} v${m.version}` : "지운 모델");

/**
 * An evaluation of its own, like a training run: where and how long it ran, stopping it or running it again, its
 * numbers (overall, per split, per class), what it wrote, its log, and every model evaluated on the same data.
 */
export function EvalView({ opId }: { opId: string }) {
  const { data: op, error, refresh } = usePolling(() => api<OpDetail>(`/model-ops/${opId}`), 3000, [opId]);
  const [again, setAgain] = useState(false);
  const [busy, setBusy] = useState(false);
  const [failed, setFailed] = useState<string | null>(null);
  const remove = async () => {
    if (!op) return;
    const what = op.kind === "eval" ? "평가" : op.kind === "export" ? "내보낸 파일" : "테스트";
    if (!window.confirm(`이 ${what} 기록을 지울까요? 지표 · 결과 파일 · 이미지별 예측(검수의 모델 기준, 이미지 위 예측 표시)이 함께 지워지고 되돌릴 수 없습니다.`)) return;
    try {
      await api(`/model-ops/${op.id}`, { method: "DELETE" });
      window.location.hash = op.model ? `#/models/${op.model.id}` : "#/models";
    } catch (err) {
      setFailed(err instanceof Error ? err.message : String(err));
    }
  };
  if (error && !op) return <div className="error">{error}</div>;
  if (!op) return <div className="muted">불러오는 중…</div>;
  const active = ACTIVE.includes(op.status);
  const task = op.model?.task ?? "detect";
  const what = op.kind === "eval" ? "평가" : op.kind === "export" ? "내보내기" : "테스트";
  const where = op.release ? `${op.release.dataset} ${op.release.version}` : "삭제된 버전";
  const splits = op.splits.map(splitName).join(" · ");

  const stop = async () => {
    if (!op.job_id || !window.confirm(`이 ${what}를 중단할까요? 지금까지 한 것은 남지 않습니다.`)) return;
    setBusy(true);
    setFailed(null);
    try {
      await api(`/jobs/${op.job_id}/cancel`, { method: "POST" });
      await refresh();
    } catch (err) {
      setFailed(err instanceof Error ? err.message : String(err));
    } finally {
      setBusy(false);
    }
  };

  return (
    <>
      <nav className="page-crumbs">
        <a href="#/models">모델</a>
        {op.model && <a href={`#/models/project/${op.model.project_id}`}>목록</a>}
        {op.model ? <a href={`#/models/${op.model.id}`}>{modelName(op.model)}</a> : <span>지운 모델</span>}
        <strong>{what}</strong>
      </nav>
      <div className="page-head">
        <div>
          <h1 className="with-badge">
            {op.kind === "eval" ? (
              <>
                <span className="muted-strong">{where}</span> {splits}
              </>
            ) : (
              what
            )}
            <Badge kind={op.status}>{STATUS_LABEL[op.status] ?? op.status}</Badge>
          </h1>
          <p className="muted">
            {modelName(op.model)} · {op.node_name ?? "서버 정해지기 전"} · {op.gpus ? `GPU ${op.gpus}` : "CPU"} ·{" "}
            {op.started_at ? `${timeAgo(op.started_at)} 시작` : `${timeAgo(op.created_at)} 요청`}
            {op.duration_s != null && ` · ${active ? "지난 시간" : "걸린 시간"} ${duration(op.duration_s)}`}
            {op.code?.trained_with && ` · ${op.code.version} 코드로`}
          </p>
        </div>
        <div className="head-actions">
          {active ? (
            <button className="danger" disabled={busy} onClick={() => void stop()}>
              중단
            </button>
          ) : (
            <>
              <button onClick={() => setAgain(true)} title="같은 모델 · 버전 · split · 설정으로 다시 실행합니다">
                다시 실행
              </button>
              <button className="danger" onClick={() => void remove()} title="실행한 사람이나 관리자만 지웁니다">
                삭제
              </button>
            </>
          )}
        </div>
      </div>
      {op.status === "queued" && op.wait_reason && <div className="notice">{op.wait_reason}</div>}
      {op.status === "failed" && op.error && <div className="error">실패: {op.error} (아래 로그에서 자세히 봅니다)</div>}
      {op.status === "canceled" && <div className="notice">중단했습니다. "다시 실행"으로 같은 설정으로 다시 돌립니다.</div>}
      {failed && <div className="error">{failed}</div>}

      {op.kind === "eval" && op.result.metrics && (
        <div className="stats metrics">
          {keyMetrics(task).map((k) => (
            <Stat key={k} icon="chart" label={METRIC_LABEL[k] ?? k} value={metric(op.result.metrics?.[k])} sub={op.splits[0] ? splitName(op.splits[0]) : ""} />
          ))}
        </div>
      )}
      {Object.keys(op.split_metrics).length > 1 && (
        <section className="card">
          <h3>split별</h3>
          <div className="table-wrap">
            <table className="plain">
              <thead>
                <tr>
                  <th>split</th>
                  {keyMetrics(task).map((k) => (
                    <th key={k} className="right">
                      {METRIC_LABEL[k] ?? k}
                    </th>
                  ))}
                </tr>
              </thead>
              <tbody>
                {Object.entries(op.split_metrics).map(([s, numbers]) => (
                  <tr key={s}>
                    <td>{splitName(s)}</td>
                    {keyMetrics(task).map((k) => (
                      <td key={k} className="num right">
                        {metric(numbers[k])}
                      </td>
                    ))}
                  </tr>
                ))}
              </tbody>
            </table>
          </div>
        </section>
      )}
      {op.class_metrics.length > 0 && (
        <section className="card">
          <h3>
            클래스별 <span className="muted small">{op.splits.length > 1 ? `${splitName(op.splits[0])} 기준` : ""}</span>
          </h3>
          <ClassTable rows={op.class_metrics} />
        </section>
      )}
      {(op.tables?.length ?? 0) + op.files.length > 0 && (
        <section className="card">
          <h3>
            결과 파일
            {op.kind === "eval" && op.release && (op.predicted ?? 0) > 0 && (
              <a className="small eval-to-review" href={`#/review/${op.release.id}`} title="틀린 것을 데이터셋 이미지 위에서 봅니다">
                검수에서 이미지로 보기
              </a>
            )}
          </h3>
          {[...(op.tables ?? [])]
            .sort((a, b) => (a.name.startsWith("failures") ? -1 : 1) - (b.name.startsWith("failures") ? -1 : 1))
            .map((t) => (
              <div key={t.name} className="op-table">
                <div className="eyebrow">
                  {t.name.startsWith("failures") ? "틀린 것이 많은 이미지" : t.name.startsWith("per_image") ? "이미지별 결과 (틀린 것 먼저)" : t.name}
                </div>
                <LoggedTable
                  file={t}
                  images={{
                    ...Object.fromEntries(op.files.filter((f) => f.kind === "image").map((f) => [f.name, f.url])),
                    ...(op.table_images ?? {}),
                  }}
                />
              </div>
            ))}
          {op.files.length > 0 && <FileList files={op.files} />}
        </section>
      )}
      {op.job_id && <LogTail jobId={op.job_id} live={active || op.status === "failed"} />}
      {op.kind === "eval" && op.release && (
        <EvalBoard releaseId={op.release.id} splits={op.splits} highlight={op.id} />
      )}
      {again && <RerunDialog op={op} onClose={() => setAgain(false)} />}
    </>
  );
}

/** The same evaluation again: same model, version, splits and settings; on the server it ran on unless another. */
function RerunDialog({ op, onClose }: { op: OpDetail; onClose: () => void }) {
  const kind = op.kind === "eval" ? "eval" : op.kind === "export" ? "export" : "predict";
  const { data: places, error: placesError } = usePlaces(op.model?.id ?? "", kind, op.release?.id ?? "");
  const before = places?.nodes.find((n) => n.id === op.node_id && n.fits)?.id ?? "";
  const [node, setNode] = useState<string | null>(null);
  const [busy, setBusy] = useState(false);
  const [error, setError] = useState<string | null>(null);
  const chosen = node ?? before;
  const start = async () => {
    setBusy(true);
    setError(null);
    try {
      const made = await api<{ id: string }>(`/model-ops/${op.id}/rerun`, { body: { node_id: chosen || places?.default || "" } });
      window.location.hash = `#/models/eval/${made.id}`;
      onClose();
    } catch (err) {
      setError(err instanceof Error ? err.message : String(err));
      setBusy(false);
    }
  };
  return (
    <Modal title="다시 실행" onClose={onClose}>
      <p className="muted">
        {modelName(op.model)}을(를) 같은 설정으로 다시 {op.kind === "eval" ? "평가합니다" : "실행합니다"}
        {op.kind === "eval" && op.release ? ` (${op.release.dataset} ${op.release.version} · ${op.splits.map(splitName).join(" · ")})` : ""}.
        기록은 새로 남고 이전 것도 그대로 둡니다.
      </p>
      {op.model && <PlacePicker places={places} error={placesError} value={chosen} onChange={setNode} />}
      {error && <div className="error">{error}</div>}
      <div className="modal-actions">
        <button type="button" onClick={onClose}>
          취소
        </button>
        <button className="primary" disabled={busy || !op.model} onClick={() => void start()}>
          다시 실행
        </button>
      </div>
    </Modal>
  );
}

/**
 * Every model evaluated on one version, split by split: the fair comparison (same images, same labels). Each model's
 * newest evaluation by default; tick up to three to set their classes side by side.
 */
export function EvalBoard({ releaseId, splits, highlight }: { releaseId: string; splits?: string[]; highlight?: string }) {
  const { data } = usePolling(() => api<EvalBoardInfo>(`/releases/${releaseId}/evaluations`), 30000, [releaseId]);
  const [split, setSplit] = useState<string | null>(null);
  const [all, setAll] = useState(false);
  const [picked, setPicked] = useState<string[]>([]);
  if (!data) return null;
  const current = split ?? (splits?.find((s) => data.splits.includes(s)) || data.splits[0] || "");
  const keys = keyMetrics(data.task);
  const rows = data.rows
    .filter((r) => r.split === current && (all || r.latest || r.op_id === highlight))
    .sort((a, b) => (b.metrics[keys[0]] ?? -1) - (a.metrics[keys[0]] ?? -1));
  const best = Object.fromEntries(keys.map((k) => [k, Math.max(...rows.map((r) => r.metrics[k] ?? -Infinity))]));
  const chosen = rows.filter((r) => picked.includes(r.op_id) && r.class_metrics.length > 0);
  const toggle = (id: string) =>
    setPicked((now) => (now.includes(id) ? now.filter((x) => x !== id) : [...now, id].slice(-3)));

  return (
    <section className="card eval-board">
      <div className="card-head">
        <h3>
          같은 데이터로 평가한 모델{" "}
          <span className="muted small">
            {data.release.dataset} {data.release.version}
          </span>
        </h3>
        <span className="spacer" />
        {data.splits.length > 1 && (
          <div className="segmented" role="group" aria-label="split">
            {data.splits.map((s) => (
              <button key={s} className={s === current ? "active" : ""} onClick={() => setSplit(s)}>
                {splitName(s)}
              </button>
            ))}
          </div>
        )}
        <label className="check small">
          <input type="checkbox" checked={all} onChange={(e) => setAll(e.target.checked)} />
          모델마다 최신 것만이 아니라 모두
        </label>
      </div>
      {rows.length === 0 ? (
        <p className="muted small">이 split으로 끝까지 평가한 모델이 없습니다.</p>
      ) : (
        <div className="table-wrap">
          <table className="plain">
            <thead>
              <tr>
                <th aria-label="클래스별 비교에 고르기" />
                <th>모델</th>
                {keys.map((k) => (
                  <th key={k} className="right">
                    {METRIC_LABEL[k] ?? k}
                  </th>
                ))}
                <th className="right">언제</th>
              </tr>
            </thead>
            <tbody>
              {rows.map((r) => (
                <tr key={`${r.op_id}-${r.split}`} className={r.op_id === highlight ? "selected-row" : ""}>
                  <td>
                    <input
                      type="checkbox"
                      checked={picked.includes(r.op_id)}
                      disabled={r.class_metrics.length === 0}
                      title={r.class_metrics.length ? "클래스별로 나란히 보기 (3개까지)" : "클래스별 지표가 없는 평가"}
                      onChange={() => toggle(r.op_id)}
                      aria-label={`${modelName(r.model)} 클래스별 비교`}
                    />
                  </td>
                  <td>
                    {r.model && !r.model.deleted ? <a href={`#/models/${r.model.id}`}>{modelName(r.model)}</a> : <span className="muted">{modelName(r.model)}</span>}
                    {r.model?.aliases.map((a) => (
                      <span key={a} className="tag alias-tag">
                        {a}
                      </span>
                    ))}
                  </td>
                  {keys.map((k) => (
                    <td key={k} className={`num right${r.metrics[k] != null && r.metrics[k] === best[k] && rows.length > 1 ? " best" : ""}`}>
                      {metric(r.metrics[k])}
                    </td>
                  ))}
                  <td className="muted right">
                    <a href={`#/models/eval/${r.op_id}`}>{timeAgo(r.created_at)}</a>
                  </td>
                </tr>
              ))}
            </tbody>
          </table>
        </div>
      )}
      {chosen.length > 0 && <ClassCompare rows={chosen} />}
    </section>
  );
}

/** Classes side by side for the evaluations picked on the board: the main per-class number of each. */
function ClassCompare({ rows }: { rows: BoardRow[] }) {
  const names = [...new Set(rows.flatMap((r) => r.class_metrics.map((c) => c.name)))];
  const label = rows[0].class_metrics.some((c) => c.map50_95 != null) ? "mAP50-95" : "F1";
  const value = (r: BoardRow, name: string) => {
    const found = r.class_metrics.find((c) => c.name === name);
    return found ? classScore(found) : null;
  };
  return (
    <div className="table-wrap class-compare">
      <div className="eyebrow">클래스별 {label}</div>
      <table className="plain">
        <thead>
          <tr>
            <th>클래스</th>
            {rows.map((r) => (
              <th key={r.op_id} className="right">
                {modelName(r.model)}
              </th>
            ))}
          </tr>
        </thead>
        <tbody>
          {names.map((name) => {
            const values = rows.map((r) => value(r, name));
            const top = Math.max(...values.map((v) => v ?? -Infinity));
            return (
              <tr key={name}>
                <td>{name}</td>
                {values.map((v, i) => (
                  <td key={rows[i].op_id} className={`num right${v != null && v === top && rows.length > 1 ? " best" : ""}`}>
                    {v != null ? metric(v) : "–"}
                  </td>
                ))}
              </tr>
            );
          })}
        </tbody>
      </table>
    </div>
  );
}

/** ``#/models/evals/<version>``: the comparison board on its own (from a dataset version's 모델 tab). */
export function EvalBoardPage({ releaseId }: { releaseId: string }) {
  return (
    <>
      <nav className="page-crumbs">
        <a href="#/models">모델</a>
        <a href={`#/datasets/${releaseId}/models`}>데이터셋 버전</a>
        <strong>평가 비교</strong>
      </nav>
      <div className="page-head">
        <div>
          <h1>평가 비교</h1>
          <p className="muted">같은 버전 · 같은 split으로 평가한 모델끼리 나란히 봅니다. 같은 이미지와 라벨로 잰 숫자라 바로 비교할 수 있습니다.</p>
        </div>
      </div>
      <EvalBoard releaseId={releaseId} />
    </>
  );
}
