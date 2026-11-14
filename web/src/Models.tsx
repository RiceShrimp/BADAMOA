import { useEffect, useState } from "react";
import { ModelOps } from "./ModelOps";
import { ProjectPicker, useProject } from "./ProjectPicker";
import { ClassTable } from "./Training";
import { api } from "./api";
import { bytesLabel } from "./catalogTypes";
import {
  METRIC_LABEL,
  type ModelDetail,
  type ModelInfo,
  type ModelOp,
  classScore,
  delta,
  keyMetrics,
  metric,
} from "./trainTypes";
import { Icon, Stat, timeAgo, usePolling } from "./ui";

export function Models({ modelId, isAdmin }: { modelId?: string; isAdmin: boolean }) {
  const { projects, projectId, choose } = useProject();
  if (modelId) return <ModelView key={modelId} modelId={modelId} isAdmin={isAdmin} />;
  return <ModelList projects={projects} projectId={projectId} onProject={choose} />;
}

function Alias({ alias }: { alias: string }) {
  return alias ? <span className={`alias ${alias}`}>{alias}</span> : null;
}

function ModelList({
  projects,
  projectId,
  onProject,
}: {
  projects: ReturnType<typeof useProject>["projects"];
  projectId: string;
  onProject: (id: string) => void;
}) {
  const { data: models } = usePolling(
    () => (projectId ? api<ModelInfo[]>(`/projects/${projectId}/models`) : Promise.resolve([])),
    10000,
    [projectId],
  );
  const [picked, setPicked] = useState<string[]>([]);
  const metricKeys = keyMetrics(projects?.find((p) => p.id === projectId)?.task);
  const primary = metricKeys[0];
  useEffect(() => setPicked([]), [projectId]);

  const toggle = (id: string) => setPicked(picked.includes(id) ? picked.filter((x) => x !== id) : [...picked, id].slice(-2));
  const families = new Set(models?.map((m) => m.name)).size;
  const best = [...(models ?? [])].sort((a, b) => (b.metrics[primary] ?? 0) - (a.metrics[primary] ?? 0))[0];
  const champion = models?.find((m) => m.alias === "champion");

  return (
    <>
      <div className="page-head">
        <div>
          <h1>모델</h1>
          <p className="muted">
            학습이 끝나면 가중치가 여기에 버전으로 등록됩니다. 두 버전을 체크하면 지표를 나란히 비교합니다.
          </p>
        </div>
        <div className="head-actions">
          <ProjectPicker projects={projects} projectId={projectId} onChange={onProject} />
        </div>
      </div>

      <div className="stats three metrics">
        <Stat icon="box" label="모델 버전" value={models?.length ?? 0} sub={`모델 ${families}종`} />
        <Stat
          icon="chart"
          label={`최고 ${METRIC_LABEL[primary]}`}
          value={best ? metric(best.metrics[primary]) : "–"}
          sub={best ? `${best.name} v${best.version}` : "등록된 모델 없음"}
        />
        <Stat
          icon="check"
          label="champion"
          value={champion ? `v${champion.version}` : "–"}
          sub={champion ? champion.name : "아직 지정하지 않음"}
        />
      </div>

      {picked.length === 2 && <CompareModels ids={picked} onClose={() => setPicked([])} />}

      <div className="card flush table-wrap">
        <table className="runs">
          <thead>
            <tr>
              <th aria-label="비교" />
              <th>모델</th>
              {metricKeys.map((k) => (
                <th key={k} className="right">
                  {METRIC_LABEL[k]}
                </th>
              ))}
              <th className="right">크기</th>
              <th>학습 데이터</th>
              <th className="right">등록</th>
            </tr>
          </thead>
          <tbody>
            {models?.length === 0 && (
              <tr>
                <td colSpan={9} className="empty">
                  아직 등록된 모델이 없습니다. <a href="#/training">학습</a>을 실행하면 끝날 때 자동으로 등록됩니다.
                </td>
              </tr>
            )}
            {models?.map((m) => (
              <tr key={m.id} className="clickable" onClick={() => (window.location.hash = `#/models/${m.id}`)}>
                <td onClick={(e) => e.stopPropagation()} className="pick">
                  <label title="비교할 두 버전 고르기">
                    <input
                      type="checkbox"
                      checked={picked.includes(m.id)}
                      onChange={() => toggle(m.id)}
                      aria-label={`${m.name} v${m.version} 비교에 넣기`}
                    />
                  </label>
                </td>
                <td>
                  <strong>
                    {m.name} <span className="version">v{m.version}</span>
                  </strong>{" "}
                  <Alias alias={m.alias} />
                  <div className="muted small">
                    {m.arch} · {m.class_count}개 클래스 · {m.imgsz}px
                  </div>
                </td>
                {metricKeys.map((k) => (
                  <td key={k} className="num right">
                    {metric(m.metrics[k])}
                  </td>
                ))}
                <td className="num right">{bytesLabel(m.bytes)}</td>
                <td>{m.release ? `${m.release.dataset} ${m.release.version}` : <span className="muted">–</span>}</td>
                <td className="muted right">{timeAgo(m.created_at)}</td>
              </tr>
            ))}
          </tbody>
        </table>
      </div>
    </>
  );
}

function CompareModels({ ids, onClose }: { ids: string[]; onClose: () => void }) {
  const [pair, setPair] = useState<ModelDetail[] | null>(null);
  const key = ids.join(",");
  useEffect(() => {
    let alive = true;
    setPair(null);
    void Promise.all(ids.map((id) => api<ModelDetail>(`/models/${id}`))).then((data) => {
      // Older first, so the difference reads as "what the newer one gained".
      if (alive) setPair(data.sort((a, b) => a.created_at.localeCompare(b.created_at)));
    });
    return () => {
      alive = false;
    };
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [key]);

  if (!pair) return <div className="card muted">비교할 모델을 불러오는 중…</div>;
  const [a, b] = pair;
  const metricKeys = keyMetrics(b.task);
  const label = (m: ModelDetail) => `${m.name} v${m.version}`;
  const byName = new Map(a.class_metrics.map((c) => [c.name, c]));
  const classRows = b.class_metrics
    .filter((c) => byName.has(c.name))
    .map((c) => ({ name: c.name, a: classScore(byName.get(c.name)!), b: classScore(c) }))
    .sort((x, y) => Math.abs(y.b - y.a) - Math.abs(x.b - x.a));
  const sameData = a.release?.id === b.release?.id;

  return (
    <section className="card compare-models">
      <div className="card-head">
        <h3>
          {label(a)} → {label(b)}
        </h3>
        <button className="ghost-btn" onClick={onClose}>
          닫기
        </button>
      </div>
      {!sameData && (
        <div className="notice">
          두 모델은 서로 다른 릴리스로 학습·검증했습니다. 검증 데이터가 다르면 지표를 그대로 비교할 수 없습니다.
        </div>
      )}
      <table>
        <thead>
          <tr>
            <th>지표</th>
            <th className="right">{label(a)}</th>
            <th className="right">{label(b)}</th>
            <th className="right">차이</th>
          </tr>
        </thead>
        <tbody>
          {metricKeys.map((k) => (
            <tr key={k}>
              <td>{METRIC_LABEL[k]}</td>
              <td className="num right">{metric(a.metrics[k])}</td>
              <td className="num right">{metric(b.metrics[k])}</td>
              <td className={`num right ${sign(b.metrics[k], a.metrics[k])}`}>{delta(b.metrics[k], a.metrics[k])}</td>
            </tr>
          ))}
          <tr>
            <td>학습 데이터</td>
            <td className="right">{a.release ? `${a.release.dataset} ${a.release.version}` : "–"}</td>
            <td className="right">{b.release ? `${b.release.dataset} ${b.release.version}` : "–"}</td>
            <td className="right muted">{sameData ? "같음" : "다름"}</td>
          </tr>
          <tr>
            <td>구조 · epoch · 이미지 크기</td>
            <td className="right">
              {a.arch} · {String(a.config.epochs ?? "–")} · {a.imgsz || "–"}
            </td>
            <td className="right">
              {b.arch} · {String(b.config.epochs ?? "–")} · {b.imgsz || "–"}
            </td>
            <td />
          </tr>
        </tbody>
      </table>
      <SharedEvaluations a={a} b={b} metricKeys={metricKeys} />
      {classRows.length > 0 && (
        <details className="fold inner">
          <summary>
            클래스별 {b.task === "classify" ? "F1" : "mAP50-95"} 차이 <span className="muted">(차이가 큰 순)</span>
          </summary>
          <div className="table-wrap class-table">
            <table>
              <thead>
                <tr>
                  <th>클래스</th>
                  <th className="right">{label(a)}</th>
                  <th className="right">{label(b)}</th>
                  <th className="right">차이</th>
                </tr>
              </thead>
              <tbody>
                {classRows.map((c) => (
                  <tr key={c.name}>
                    <td>{c.name}</td>
                    <td className="num right">{metric(c.a)}</td>
                    <td className="num right">{metric(c.b)}</td>
                    <td className={`num right ${sign(c.b, c.a)}`}>{delta(c.b, c.a)}</td>
                  </tr>
                ))}
              </tbody>
            </table>
          </div>
        </details>
      )}
    </section>
  );
}

/** Evaluations both models have on the same release and split: the fair comparison. */
function SharedEvaluations({ a, b, metricKeys }: { a: ModelDetail; b: ModelDetail; metricKeys: string[] }) {
  const [rows, setRows] = useState<{ key: string; title: string; a: ModelOp; b: ModelOp }[] | null>(null);
  useEffect(() => {
    let alive = true;
    const latest = (ops: ModelOp[]) => {
      const found = new Map<string, ModelOp>();
      for (const op of ops) {
        if (op.kind !== "eval" || op.status !== "succeeded" || !op.release) continue;
        const key = `${op.release.id}:${String(op.params.split)}`;
        if (!found.has(key)) found.set(key, op); // newest first
      }
      return found;
    };
    void Promise.all([a, b].map((m) => api<ModelOp[]>(`/models/${m.id}/ops`))).then(([first, second]) => {
      if (!alive) return;
      const ours = latest(first);
      const theirs = latest(second);
      setRows(
        [...ours.keys()]
          .filter((key) => theirs.has(key))
          .map((key) => {
            const op = ours.get(key)!;
            const title = `${op.release!.dataset} ${op.release!.version} · ${String(op.params.split)}`;
            return { key, title, a: op, b: theirs.get(key)! };
          }),
      );
    });
    return () => {
      alive = false;
    };
  }, [a, b]);

  if (rows == null) return null;
  if (rows.length === 0) {
    return (
      <p className="muted small">
        두 모델을 같은 릴리스로 평가한 기록이 없습니다. 각 모델 화면의 "평가"에서 같은 릴리스와 같은 부분으로 평가하면 여기에 나란히
        나옵니다.
      </p>
    );
  }
  return (
    <>
      <div className="eyebrow op-head">같은 데이터로 평가한 결과</div>
      <table>
        <thead>
          <tr>
            <th>릴리스 · 부분 / 지표</th>
            <th className="right">
              {a.name} v{a.version}
            </th>
            <th className="right">
              {b.name} v{b.version}
            </th>
            <th className="right">차이</th>
          </tr>
        </thead>
        <tbody>
          {rows.flatMap((row) =>
            metricKeys.map((k, i) => (
              <tr key={`${row.key}.${k}`}>
                <td>
                  {i === 0 && <strong>{row.title}</strong>} <span className="muted">{METRIC_LABEL[k] ?? k}</span>
                </td>
                <td className="num right">{metric(row.a.result.metrics?.[k])}</td>
                <td className="num right">{metric(row.b.result.metrics?.[k])}</td>
                <td className={`num right ${sign(row.b.result.metrics?.[k], row.a.result.metrics?.[k])}`}>
                  {delta(row.b.result.metrics?.[k], row.a.result.metrics?.[k])}
                </td>
              </tr>
            )),
          )}
        </tbody>
      </table>
    </>
  );
}

function sign(now: number | undefined, before: number | undefined): string {
  if (now == null || before == null || Math.abs(now - before) < 0.0005) return "";
  return now > before ? "up" : "down";
}

function ModelView({ modelId, isAdmin }: { modelId: string; isAdmin: boolean }) {
  const { data: model, error, refresh } = usePolling(() => api<ModelDetail>(`/models/${modelId}`), 30000, [modelId]);
  const [failure, setFailure] = useState<string | null>(null);

  if (error && !model) return <div className="error">{error}</div>;
  if (!model) return <div className="muted">불러오는 중…</div>;

  const setAlias = async (alias: string) => {
    try {
      setFailure(null);
      await api(`/models/${model.id}/alias`, { body: { alias } });
      await refresh();
    } catch (e) {
      setFailure(e instanceof Error ? e.message : String(e));
    }
  };
  const remove = async () => {
    if (!window.confirm(`${model.name} v${model.version} 모델을 삭제할까요?`)) return;
    await api(`/models/${model.id}`, { method: "DELETE" });
    window.location.hash = "#/models";
  };
  const classRows = [...model.class_metrics].sort((a, b) => classScore(b) - classScore(a));
  const info = model.info as { ultralytics?: string; torch?: string };
  const metricKeys = keyMetrics(model.task);
  const items = model.export.items ?? {};

  return (
    <>
      <a className="link back" href="#/models">
        ← 모델 목록
      </a>
      <div className="page-head">
        <div>
          <h1 className="with-badge">
            {model.name} <span className="version">v{model.version}</span>
            <Alias alias={model.alias} />
          </h1>
          <p className="muted">
            {model.arch} · {model.task === "classify" ? "분류" : "객체 검출"} · 클래스 {model.class_count}개 · 입력 {model.imgsz}px · {timeAgo(model.created_at)} 등록
          </p>
        </div>
        <div className="head-actions">
          <label className="project-picker">
            <span className="eyebrow">alias</span>
            <select value={model.alias} onChange={(e) => void setAlias(e.target.value)} aria-label="alias">
              <option value="">없음</option>
              <option value="champion">champion</option>
              <option value="challenger">challenger</option>
            </select>
          </label>
          <a className="btn" href={model.download_url} download>
            가중치 받기 ({bytesLabel(model.bytes)})
          </a>
          {isAdmin && (
            <button className="danger" onClick={remove}>
              삭제
            </button>
          )}
        </div>
      </div>
      {failure && <div className="error">{failure}</div>}

      <div className="stats metrics">
        {metricKeys.map((k) => (
          <Stat key={k} icon="chart" label={METRIC_LABEL[k]} value={metric(model.metrics[k])} sub="검증 데이터 기준" />
        ))}
      </div>

      <ModelOps model={model} />

      <section className="card">
        <h3>계보</h3>
        <div className="lineage">
          {model.release ? (
            <a className="lineage-step" href={model.release.deleted ? undefined : `#/datasets/${model.release.id}`}>
              <span className="eyebrow">
                <Icon name="database" size={14} />
                릴리스
              </span>
              <strong>
                {model.release.dataset} {model.release.version}
              </strong>
              <small>
                {model.export.items
                  ? `train ${(items.train ?? 0).toLocaleString()} · val ${(items.val ?? 0).toLocaleString()}장으로 학습`
                  : "학습 데이터"}
                {model.release.deleted ? " · 삭제됨" : ""}
              </small>
            </a>
          ) : (
            <div className="lineage-step">
              <span className="eyebrow">릴리스</span>
              <strong>알 수 없음</strong>
            </div>
          )}
          <Icon name="arrow" />
          {model.run ? (
            <a className="lineage-step" href={`#/training/${model.run.id}`}>
              <span className="eyebrow">
                <Icon name="chart" size={14} />
                학습 실행
              </span>
              <strong>{model.run.name}</strong>
              <small>
                {model.template.title}
                {model.template.version ? ` v${model.template.version}` : ""}
                {model.config.epochs != null ? ` · epoch ${String(model.config.epochs)}` : ""}
              </small>
            </a>
          ) : (
            <div className="lineage-step">
              <span className="eyebrow">학습 실행</span>
              <strong>기록이 지워짐</strong>
            </div>
          )}
          <Icon name="arrow" />
          <div className="lineage-step current">
            <span className="eyebrow">
              <Icon name="box" size={14} />
              모델
            </span>
            <strong>
              {model.name} v{model.version}
            </strong>
            <small>
              {model.file_name} · {bytesLabel(model.bytes)}
            </small>
          </div>
        </div>
        <dl className="kv compact">
          <dt>sha256</dt>
          <dd className="mono small">{model.sha256}</dd>
          {info.ultralytics && (
            <>
              <dt>환경</dt>
              <dd className="mono small">
                ultralytics {info.ultralytics} · torch {info.torch}
              </dd>
            </>
          )}
        </dl>
      </section>

      {classRows.length > 0 && (
        <details className="card fold" open={classRows.length <= 12}>
          <summary>
            클래스별 지표 <span className="muted">({classRows.length})</span>
          </summary>
          <ClassTable rows={classRows} />
        </details>
      )}
    </>
  );
}
