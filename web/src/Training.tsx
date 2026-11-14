import { type FormEvent, useEffect, useMemo, useRef, useState } from "react";
import { EpochChart, type EpochSeries } from "./EpochChart";
import { ProjectPicker, useProject } from "./ProjectPicker";
import { type Job, type LogLine, api } from "./api";
import type { LabelClass, ReleaseInfo, TreeDataset } from "./catalogTypes";
import { ACTIVE, KEY_METRICS, METRIC_LABEL, type Run, type RunDetail, metric } from "./trainTypes";
import { Badge, Icon, Modal, STATUS_LABEL, Stat, VIA_LABEL, duration, timeAgo, usePolling } from "./ui";

const MAX_COMPARED = 8; // one palette slot per run
const COMPARE_CHARTS = ["map50_95", "map50", "precision", "recall", "train_box_loss", "val_box_loss"];
const MODEL_FAMILIES: { label: string; prefix: string }[] = [
  { label: "YOLO11", prefix: "yolo11" },
  { label: "YOLO26", prefix: "yolo26" },
];
const SIZES = ["n", "s", "m", "l", "x"];
const DEFAULT_MODEL = "yolo11n.pt";

export function Training({ runId, arg }: { runId?: string; arg?: string }) {
  const { projects, projectId, choose } = useProject();

  if (projects && projects.length === 0) {
    return (
      <>
        <div className="page-head">
          <div>
            <h1>학습</h1>
            <p className="muted">릴리스를 골라 모델을 학습하고, 실행끼리 지표를 비교합니다.</p>
          </div>
        </div>
        <div className="card empty-state">
          <Icon name="chart" size={28} />
          <strong>먼저 데이터셋을 등록하세요</strong>
          <p className="muted">학습은 프로젝트의 릴리스에서 시작합니다. 프로젝트를 만들고 데이터셋을 가져온 뒤 다시 오세요.</p>
          <a className="btn primary-link" href="#/datasets">
            데이터셋으로 <Icon name="arrow" size={14} />
          </a>
        </div>
      </>
    );
  }
  if (runId && runId !== "new") return <RunView key={runId} runId={runId} />;
  return (
    <RunList
      projects={projects}
      projectId={projectId}
      onProject={choose}
      startWith={runId === "new" ? (arg ?? "") : undefined}
    />
  );
}

// ---- list + comparison ------------------------------------------------------------------

function RunList({
  projects,
  projectId,
  onProject,
  startWith,
}: {
  projects: ReturnType<typeof useProject>["projects"];
  projectId: string;
  onProject: (id: string) => void;
  startWith?: string;
}) {
  const [busy, setBusy] = useState(true);
  const { data: runs, refresh } = usePolling(
    () => (projectId ? api<Run[]>(`/projects/${projectId}/runs`) : Promise.resolve([])),
    busy ? 3000 : 15000,
    [projectId],
  );
  const [picked, setPicked] = useState<string[] | null>(null); // null: follow the newest runs
  const [creating, setCreating] = useState(startWith !== undefined);

  useEffect(() => setPicked(null), [projectId]);
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
  const best = (runs ?? [])
    .filter((r) => r.metrics.map50_95 != null)
    .sort((a, b) => b.metrics.map50_95 - a.metrics.map50_95)[0];

  return (
    <>
      <div className="page-head">
        <div>
          <h1>학습</h1>
          <p className="muted">릴리스를 골라 모델을 학습합니다. 표에서 실행을 체크하면 지표 곡선을 겹쳐 비교할 수 있습니다.</p>
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
          label="최고 mAP50-95"
          value={best ? metric(best.metrics.map50_95) : "–"}
          sub={best ? best.name : "완료한 학습 없음"}
        />
        <Stat icon="box" label="등록된 모델" value={runs?.filter((r) => r.model).length ?? 0} sub="학습이 끝나면 자동 등록" />
      </div>

      {compared.length > 0 && <Comparison ids={compared} live={busy} />}

      <div className="card flush table-wrap">
        <table className="runs">
          <thead>
            <tr>
              <th aria-label="비교" />
              <th>실행</th>
              <th>상태</th>
              {KEY_METRICS.map((k) => (
                <th key={k} className="right">
                  {METRIC_LABEL[k]}
                </th>
              ))}
              <th>모델</th>
              <th>노드</th>
              <th className="right">소요</th>
              <th className="right">시작</th>
            </tr>
          </thead>
          <tbody>
            {runs?.length === 0 && (
              <tr>
                <td colSpan={11} className="empty">
                  아직 학습한 적이 없습니다. "새 학습"으로 릴리스를 골라 시작하세요.
                </td>
              </tr>
            )}
            {runs?.map((run) => {
              const slot = compared.indexOf(run.id) + 1;
              return (
                <tr key={run.id} className="clickable" onClick={() => (window.location.hash = `#/training/${run.id}`)}>
                  <td onClick={(e) => e.stopPropagation()} className="pick">
                    <label title="지표 그래프에 겹쳐 보기">
                      <input
                        type="checkbox"
                        checked={slot > 0}
                        onChange={() => toggle(run.id)}
                        disabled={run.epoch === 0}
                        aria-label={`${run.name} 지표 겹쳐 보기`}
                      />
                      {slot > 0 && <i style={{ background: `var(--series-${slot})` }} />}
                    </label>
                  </td>
                  <td>
                    <strong>{run.name}</strong>
                    <div className="muted small">
                      {run.release ? `${run.release.dataset} ${run.release.version}` : "삭제된 릴리스"} · {run.config.model}
                    </div>
                  </td>
                  <td>
                    <Badge kind={run.status}>{STATUS_LABEL[run.status]}</Badge>
                    {run.status === "running" && <Progress epoch={run.epoch} epochs={run.epochs} />}
                    {run.status === "queued" && run.wait_reason && <div className="wait-reason">{run.wait_reason}</div>}
                  </td>
                  {KEY_METRICS.map((k) => (
                    <td key={k} className="num right">
                      {metric(run.metrics[k])}
                    </td>
                  ))}
                  <td onClick={(e) => e.stopPropagation()}>
                    {run.model ? (
                      <a href={`#/models/${run.model.id}`}>
                        {run.model.name} v{run.model.version}
                      </a>
                    ) : (
                      <span className="muted">–</span>
                    )}
                  </td>
                  <td>{run.node_name ?? <span className="muted">배정 대기</span>}</td>
                  <td className="num right">{duration(run.duration_s)}</td>
                  <td className="muted right">{timeAgo(run.created_at)}</td>
                </tr>
              );
            })}
          </tbody>
        </table>
      </div>

      {creating && (
        <NewRun
          projectId={projectId}
          releaseId={startWith}
          onClose={() => {
            setCreating(false);
            if (startWith !== undefined) window.location.hash = "#/training";
          }}
          onStarted={(run) => {
            setCreating(false);
            void refresh();
            window.location.hash = `#/training/${run.id}`;
          }}
        />
      )}
    </>
  );
}

function Progress({ epoch, epochs }: { epoch: number; epochs: number }) {
  const pct = epochs > 0 ? Math.min(100, Math.round((epoch / epochs) * 100)) : 0;
  return (
    <div className="progress" title={`epoch ${epoch} / ${epochs}`}>
      <div className="progress-bar">
        <i style={{ width: `${pct}%` }} />
      </div>
      <span>
        {epoch}/{epochs}
      </span>
    </div>
  );
}

/** The same metric of several runs in one small chart each; a run keeps its colour across charts. */
function Comparison({ ids, live }: { ids: string[]; live: boolean }) {
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
  const charts = COMPARE_CHARTS.filter((k) => series(k).length > 0);

  return (
    <section className="compare-runs">
      <div className="legend runs-legend">
        {shown.map((d) => (
          <a key={d.id} className="legend-item" href={`#/training/${d.id}`}>
            <i style={{ background: `var(--series-${ids.indexOf(d.id) + 1})` }} />
            {d.name}
            <strong>{metric(d.metrics.map50_95)}</strong>
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
  { title: "mAP", keys: ["map50_95", "map50"] },
  { title: "Precision · Recall", keys: ["precision", "recall"] },
  { title: "box loss", keys: ["train_box_loss", "val_box_loss"] },
  { title: "cls loss", keys: ["train_cls_loss", "val_cls_loss"] },
  { title: "dfl loss", keys: ["train_dfl_loss", "val_dfl_loss"] },
];
const VAL_SOURCE: Record<string, string> = {
  release: "릴리스의 val split",
  holdout: "train에서 떼어 냄",
  train: "train으로 검증 (지표가 높게 나옴)",
};

function RunView({ runId }: { runId: string }) {
  const [live, setLive] = useState(true);
  const { data: run, error, refresh } = usePolling(() => api<RunDetail>(`/runs/${runId}`), live ? 2500 : 20000, [runId]);
  const [working, setWorking] = useState(false);
  const [failure, setFailure] = useState<string | null>(null);
  useEffect(() => setLive(!run || ACTIVE.includes(run.status)), [run]);

  if (error && !run) return <div className="error">{error}</div>;
  if (!run) return <div className="muted">불러오는 중…</div>;

  const active = ACTIVE.includes(run.status);
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
  const remove = () => {
    if (!window.confirm(`${run.name} 실행 기록을 지울까요? 등록된 모델은 남습니다.`)) return;
    void act(async () => {
      await api(`/runs/${run.id}`, { method: "DELETE" });
      window.location.hash = "#/training";
    });
  };

  const charts = DETAIL_CHARTS.map((c) => ({
    title: c.title,
    series: c.keys
      .filter((k) => run.history.series[k])
      .map((k, i) => ({ key: k, label: METRIC_LABEL[k] ?? k, slot: i + 1, steps: run.history.steps, values: run.history.series[k] })),
  })).filter((c) => c.series.length > 0);
  const items = run.export.items ?? {};
  const excluded = run.export.excluded ?? {};
  const leftOut = (excluded.unlabelled ?? 0) + (excluded.partial ?? 0) + (excluded.missing ?? 0);
  const classRows = [...run.class_metrics].sort((a, b) => b.map50_95 - a.map50_95);
  const info = run.info as { ultralytics?: string; torch?: string; device?: string; gpus?: string[] };

  return (
    <>
      <a className="link back" href="#/training">
        ← 학습 목록
      </a>
      <div className="page-head">
        <div>
          <h1 className="with-badge">
            {run.name}
            <Badge kind={run.status}>{STATUS_LABEL[run.status]}</Badge>
          </h1>
          <p className="muted">
            {run.release ? (
              <a href={`#/datasets/${run.release.id}`}>
                {run.release.dataset} {run.release.version}
              </a>
            ) : (
              "삭제된 릴리스"
            )}{" "}
            · {run.trainer_title} · {run.config.model} · {run.node_name ?? "노드 배정 대기"} ·{" "}
            {VIA_LABEL[run.created_via] ?? run.created_via}에서 시작 · {timeAgo(run.created_at)}
          </p>
        </div>
        <div className="head-actions">
          {active && run.job_id && (
            <button className="danger" disabled={working} onClick={() => act(() => api(`/jobs/${run.job_id}/cancel`, { method: "POST" }))}>
              중단
            </button>
          )}
          <button disabled={working} onClick={again} title="같은 릴리스와 설정으로 새 실행을 시작합니다">
            다시 실행
          </button>
          {!active && (
            <button disabled={working} onClick={remove}>
              삭제
            </button>
          )}
        </div>
      </div>

      {failure && <div className="error">{failure}</div>}
      {run.status === "queued" && run.wait_reason && <div className="notice">{run.wait_reason}</div>}
      {run.error && <div className="error">{run.error} 아래 로그에서 원인을 확인하세요.</div>}

      <div className="stats five metrics">
        {KEY_METRICS.map((k) => (
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
          {run.model.alias && <span className={`alias ${run.model.alias}`}>{run.model.alias}</span>}
          <Icon name="arrow" size={14} />
        </a>
      )}

      {charts.length > 0 && (
        <div className="mini-charts wide">
          {charts.map((c) => (
            <EpochChart key={c.title} title={c.title} series={c.series} legend />
          ))}
        </div>
      )}

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
              <dt>박스</dt>
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
                      excluded.unlabelled ? `미라벨 ${excluded.unlabelled}` : "",
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
            <p className="muted">학습이 시작되면 릴리스에서 실제로 쓴 이미지 수가 표시됩니다.</p>
          )}
        </section>
        <section className="card">
          <h3>설정</h3>
          <dl className="kv">
            <dt>시작 가중치</dt>
            <dd>
              {run.config.model}
              {run.config.pretrained === false ? " (구조만, 처음부터 학습)" : ""}
            </dd>
            <dt>epoch · 이미지 크기 · 배치</dt>
            <dd>
              {run.config.epochs} · {run.config.imgsz} · {run.config.batch}
            </dd>
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
            <dt>라벨 없는 이미지</dt>
            <dd>{run.config.unlabelled === "background" ? "배경으로 사용" : "제외"}</dd>
            {info.ultralytics && (
              <>
                <dt>환경</dt>
                <dd className="mono small">
                  ultralytics {info.ultralytics} · torch {info.torch}
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
        </section>
      </div>

      {classRows.length > 0 && (
        <details className="card fold" open={classRows.length <= 12}>
          <summary>
            클래스별 지표 <span className="muted">({classRows.length})</span>
          </summary>
          <ClassTable rows={classRows} />
        </details>
      )}

      {run.job_id && <LogTail jobId={run.job_id} live={active} />}
    </>
  );
}

export function ClassTable({ rows }: { rows: RunDetail["class_metrics"] }) {
  return (
    <div className="table-wrap class-table">
      <table>
        <thead>
          <tr>
            <th>클래스</th>
            <th className="right">mAP50-95</th>
            <th className="right">mAP50</th>
            <th className="right">Precision</th>
            <th className="right">Recall</th>
          </tr>
        </thead>
        <tbody>
          {rows.map((c) => (
            <tr key={c.id}>
              <td>{c.name}</td>
              <td className="num right">
                <span className="cell-bar">
                  <i style={{ width: `${Math.round(c.map50_95 * 100)}%` }} />
                </span>
                {metric(c.map50_95)}
              </td>
              <td className="num right">{metric(c.map50)}</td>
              <td className="num right">{metric(c.precision)}</td>
              <td className="num right">{metric(c.recall)}</td>
            </tr>
          ))}
        </tbody>
      </table>
    </div>
  );
}

function LogTail({ jobId, live }: { jobId: string; live: boolean }) {
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
    <details className="card fold" open={live}>
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

function NewRun({
  projectId,
  releaseId,
  onClose,
  onStarted,
}: {
  projectId: string;
  releaseId?: string;
  onClose: () => void;
  onStarted: (run: Run) => void;
}) {
  const [tree, setTree] = useState<TreeDataset[] | null>(null);
  const [release, setRelease] = useState(releaseId ?? "");
  const [classes, setClasses] = useState<LabelClass[]>([]);
  const [chosen, setChosen] = useState<number[]>([]); // empty = all classes
  const [unlabelledCount, setUnlabelledCount] = useState(0);
  const [form, setForm] = useState({
    model: DEFAULT_MODEL,
    customModel: "",
    epochs: "50",
    imgsz: "640",
    batch: "16",
    gpus: "",
    unlabelled: "exclude",
    name: "",
    model_name: "",
    pretrained: true,
  });
  const [error, setError] = useState<string | null>(null);
  const [busy, setBusy] = useState(false);

  useEffect(() => {
    void api<TreeDataset[]>(`/projects/${projectId}/datasets`).then((data) => {
      setTree(data);
      const ready = data.flatMap((d) => d.releases).filter((r) => r.status === "ready");
      setRelease((current) => (ready.some((r) => r.id === current) ? current : (ready[0]?.id ?? "")));
    });
  }, [projectId]);

  useEffect(() => {
    setChosen([]);
    if (!release) return setClasses([]);
    void api<ReleaseInfo>(`/releases/${release}`).then((info) => {
      setClasses(info.label_space.classes);
      setUnlabelledCount(info.stats.states?.unlabelled ?? 0);
    });
  }, [release]);

  const set = (key: keyof typeof form) => (e: { target: { value: string } }) => setForm({ ...form, [key]: e.target.value });
  const toggleClass = (id: number) => setChosen(chosen.includes(id) ? chosen.filter((c) => c !== id) : [...chosen, id]);

  const submit = async (e: FormEvent) => {
    e.preventDefault();
    setBusy(true);
    setError(null);
    try {
      const run = await api<Run>("/runs", {
        body: {
          release_id: release,
          model: form.model === "custom" ? form.customModel.trim() : form.model,
          epochs: Number(form.epochs),
          imgsz: Number(form.imgsz),
          batch: Number(form.batch),
          gpus: form.gpus === "" ? null : Number(form.gpus),
          unlabelled: form.unlabelled,
          classes: chosen,
          name: form.name,
          model_name: form.model_name,
          pretrained: form.pretrained,
        },
      });
      onStarted(run);
    } catch (err) {
      setError(err instanceof Error ? err.message : String(err));
      setBusy(false);
    }
  };

  const none = tree != null && !tree.some((d) => d.releases.some((r) => r.status === "ready"));

  return (
    <Modal title="새 학습" onClose={onClose}>
      {none ? (
        <>
          <p className="muted">이 프로젝트에는 학습할 릴리스가 없습니다. 먼저 데이터셋을 가져오세요.</p>
          <div className="modal-actions">
            <a className="btn primary-link" href="#/datasets" onClick={onClose}>
              데이터셋으로 <Icon name="arrow" size={14} />
            </a>
          </div>
        </>
      ) : (
        <form onSubmit={submit}>
          <p className="muted">
            학습은 릴리스의 이미지가 있는 노드에서 실행됩니다. 그 노드에서 처음 학습할 때는 YOLO 환경을 설치하느라 몇 분 더
            걸립니다.
          </p>
          <label>
            릴리스
            <select value={release} onChange={(e) => setRelease(e.target.value)} required>
              {tree?.map((d) => (
                <optgroup key={d.id} label={d.name}>
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
          <div className="form-row">
            <label>
              모델
              <select value={form.model} onChange={set("model")}>
                {MODEL_FAMILIES.map((family) => (
                  <optgroup key={family.prefix} label={family.label}>
                    {SIZES.map((size) => (
                      <option key={size} value={`${family.prefix}${size}.pt`}>
                        {family.prefix}
                        {size}
                      </option>
                    ))}
                  </optgroup>
                ))}
                <option value="custom">직접 입력…</option>
              </select>
            </label>
            <label>
              epoch
              <input type="number" min={1} value={form.epochs} onChange={set("epochs")} required />
            </label>
            <label>
              이미지 크기
              <input type="number" min={32} step={32} value={form.imgsz} onChange={set("imgsz")} required />
            </label>
            <label>
              배치
              <input type="number" min={-1} value={form.batch} onChange={set("batch")} required />
            </label>
          </div>
          {form.model === "custom" && (
            <label>
              가중치 이름 (Ultralytics가 내려받을 수 있는 이름 또는 노드의 파일 경로)
              <input value={form.customModel} onChange={set("customModel")} required className="mono" placeholder="yolo11n.pt" />
            </label>
          )}
          <div className="form-row">
            <label>
              GPU 수
              <input type="number" min={0} max={16} value={form.gpus} onChange={set("gpus")} placeholder="자동" />
            </label>
            <label>
              라벨 없는 이미지{unlabelledCount > 0 ? ` (${unlabelledCount.toLocaleString()}장)` : ""}
              <select value={form.unlabelled} onChange={set("unlabelled")}>
                <option value="exclude">제외</option>
                <option value="background">배경으로 사용</option>
              </select>
            </label>
          </div>
          <p className="muted small">
            GPU 수를 비우면 노드에 GPU가 있을 때 1개를 씁니다. 0이면 CPU로 학습합니다. 빈 GPU가 없으면 생길 때까지
            기다립니다.
          </p>

          <details className="advanced">
            <summary>
              세부 설정 <span className="muted">(클래스 선택 · 이름)</span>
            </summary>
            {classes.length > 0 && (
              <>
                <div className="eyebrow">
                  학습할 클래스 — {chosen.length === 0 ? `전체 ${classes.length}개` : `${chosen.length}개 선택`}
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
                <p className="muted small">아무것도 고르지 않으면 전체 클래스로 학습합니다.</p>
              </>
            )}
            <div className="form-row">
              <label>
                실행 이름 (선택)
                <input value={form.name} onChange={set("name")} placeholder="자동: 데이터셋-버전-모델" />
              </label>
              <label>
                모델 이름 (선택)
                <input value={form.model_name} onChange={set("model_name")} placeholder="자동: 데이터셋-모델" />
              </label>
            </div>
            <label className="check">
              <input type="checkbox" checked={form.pretrained} onChange={(e) => setForm({ ...form, pretrained: e.target.checked })} />{" "}
              사전학습 가중치에서 시작 (끄면 같은 구조를 처음부터 학습)
            </label>
          </details>

          <p className="muted small">Ultralytics YOLO는 AGPL-3.0 라이선스입니다. 학습한 모델을 제품에 탑재하기 전에 라이선스를 확인하세요.</p>
          {error && <div className="error">{error}</div>}
          <div className="modal-actions">
            <button type="button" onClick={onClose}>
              취소
            </button>
            <button className="primary" disabled={busy || !release}>
              학습 시작
            </button>
          </div>
        </form>
      )}
    </Modal>
  );
}
