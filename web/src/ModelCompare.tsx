import { useState } from "react";
import { useDebounced } from "./Annotate";
import { api } from "./api";
import { SPLIT_LABEL, type TreeDataset, sameTask } from "./catalogTypes";
import { type ClassMetric, METRIC_LABEL, classScore, keyMetrics, metric } from "./trainTypes";
import { Icon, Modal, usePolling } from "./ui";

interface Picked {
  id: string;
  name: string;
  version: number;
  task: string;
  aliases: string[];
  project_id: string;
  project: { id: string; name: string };
  deleted?: boolean;
}

interface Found extends Picked {
  metric: string;
  value: number | null;
}

interface Compared extends Picked {
  metrics: Record<string, number>;
  class_metrics: ClassMetric[];
  release: { id: string; dataset: string; version: string; deleted?: boolean } | null;
}

interface SharedEval {
  release: { id: string; dataset: string; version: string; project: { id: string; name: string }; deleted: boolean };
  split: string;
  rows: Record<string, { op_id: string; metrics: Record<string, number>; class_metrics: ClassMetric[]; created_at: string }>;
}

interface Comparison {
  models: Compared[];
  evaluations: SharedEval[];
}

const MAX = 6;
const label = (m: { name: string; version: number }) => `${m.name} v${m.version}`;
const splitName = (s: string) => SPLIT_LABEL[s] ?? s;

/** ``#/models/compare/<id,id,…>``: the address keeps the models picked. */
function go(ids: string[]) {
  window.location.hash = `#/models/compare/${ids.join(",")}`;
}

/**
 * Models from any project side by side. Their own validation numbers are measured on the data each was trained
 * with (comparable only when that is the same); the fair comparison is an evaluation on the same version and split,
 * which can be started here for the models that lack it, on any project's version of the same task.
 */
export function ModelCompare({ ids }: { ids: string[] }) {
  const key = ids.join(",");
  const { data, error, refresh } = usePolling(
    () => (ids.length ? api<Comparison>(`/models-compare?ids=${encodeURIComponent(key)}`) : Promise.resolve({ models: [], evaluations: [] })),
    15000,
    [key],
  );
  const [basis, setBasis] = useState<string>("own"); // classes side by side: "own" or "<release>:<split>"
  const [evaluating, setEvaluating] = useState(false);
  const models = data?.models ?? [];
  const task = models[0]?.task ?? "detect";
  const keys = keyMetrics(task);
  const mixed = models.some((m) => !sameTask(m.task, task));
  const sameValidation = models.length > 1 && models.every((m) => m.release?.id && m.release.id === models[0].release?.id);

  const remove = (id: string) => go(ids.filter((x) => x !== id));
  const best = (values: (number | undefined)[]) => Math.max(...values.map((v) => v ?? -Infinity));
  const chosen = basis === "own" ? null : data?.evaluations.find((g) => `${g.release.id}:${g.split}` === basis);
  const classSource = (m: Compared): ClassMetric[] => (chosen ? (chosen.rows[m.id]?.class_metrics ?? []) : m.class_metrics);

  return (
    <>
      <nav className="page-crumbs">
        <a href="#/models">모델</a>
        <strong>비교</strong>
      </nav>
      <div className="page-head">
        <div>
          <h1>모델 비교</h1>
          <p className="muted">
            어느 프로젝트의 모델이든 {MAX}개까지 나란히 봅니다. 공정한 비교는 같은 버전 · 같은 split으로 평가한 결과입니다. 학습 때 검증
            지표는 모델마다 검증 데이터가 다를 수 있습니다.
          </p>
        </div>
      </div>
      {error && <div className="error">{error}</div>}

      <section className="card compare-picker">
        <div className="compare-chips">
          {models.map((m) => (
            <span key={m.id} className="compare-chip">
              <a href={`#/models/${m.id}`}>{label(m)}</a>
              <span className="muted small">{m.project.name}</span>
              <button className="icon-btn" onClick={() => remove(m.id)} aria-label={`${label(m)} 빼기`}>
                <Icon name="x" size={12} />
              </button>
            </span>
          ))}
          {ids.length < MAX && <ModelFinder taken={ids} onPick={(id) => go([...ids, id])} />}
        </div>
        {mixed && <div className="notice small">태스크가 다른 모델이 섞여 있습니다. 지표의 뜻이 달라 나란히 비교할 수 없습니다.</div>}
      </section>

      {models.length > 0 && (
        <section className="card">
          <div className="card-head">
            <h3>같은 데이터로 평가한 결과</h3>
            <span className="spacer" />
            {models.length > 1 && (
              <button className="small" onClick={() => setEvaluating(true)} title="고른 버전 · split으로 모든 모델을 평가합니다">
                같은 버전으로 모두 평가
              </button>
            )}
          </div>
          {(data?.evaluations.length ?? 0) === 0 ? (
            <p className="muted small">
              아직 같은 버전 · split으로 평가한 결과가 없습니다. "같은 버전으로 모두 평가"를 누르면 고른 버전(다른 프로젝트 것도 됨)으로 모든
              모델을 평가합니다.
            </p>
          ) : (
            data!.evaluations.map((g) => {
              const values = (k: string) => models.map((m) => g.rows[m.id]?.metrics[k]);
              return (
                <div key={`${g.release.id}:${g.split}`} className="compare-group">
                  <div className="eyebrow">
                    {g.release.project.name} · {g.release.dataset} {g.release.version} · {splitName(g.split)}{" "}
                    <a className="link small" href={`#/models/evals/${g.release.id}`}>
                      이 버전의 평가 전체
                    </a>
                  </div>
                  <div className="table-wrap">
                    <table className="plain">
                      <thead>
                        <tr>
                          <th>지표</th>
                          {models.map((m) => (
                            <th key={m.id} className="right">
                              {label(m)}
                            </th>
                          ))}
                        </tr>
                      </thead>
                      <tbody>
                        {keys.map((k) => {
                          const top = best(values(k));
                          return (
                            <tr key={k}>
                              <td>{METRIC_LABEL[k] ?? k}</td>
                              {models.map((m) => {
                                const row = g.rows[m.id];
                                const v = row?.metrics[k];
                                return (
                                  <td key={m.id} className={`num right${v != null && v === top && models.length > 1 ? " best" : ""}`}>
                                    {row ? (
                                      <a href={`#/models/eval/${row.op_id}`}>{metric(v)}</a>
                                    ) : (
                                      <span className="muted small">평가 안 함</span>
                                    )}
                                  </td>
                                );
                              })}
                            </tr>
                          );
                        })}
                      </tbody>
                    </table>
                  </div>
                </div>
              );
            })
          )}
        </section>
      )}

      {models.length > 0 && (
        <section className="card">
          <h3>
            학습 때 검증 지표{" "}
            <span className="muted small">{sameValidation ? "모두 같은 버전으로 검증" : "모델마다 검증 데이터가 다를 수 있음"}</span>
          </h3>
          <div className="table-wrap">
            <table className="plain">
              <thead>
                <tr>
                  <th>지표</th>
                  {models.map((m) => (
                    <th key={m.id} className="right">
                      {label(m)}
                      <div className="muted small">{m.release ? `${m.release.dataset} ${m.release.version}` : "검증 데이터 모름"}</div>
                    </th>
                  ))}
                </tr>
              </thead>
              <tbody>
                {keys.map((k) => {
                  const top = best(models.map((m) => m.metrics[k]));
                  return (
                    <tr key={k}>
                      <td>{METRIC_LABEL[k] ?? k}</td>
                      {models.map((m) => (
                        <td key={m.id} className={`num right${m.metrics[k] != null && m.metrics[k] === top && sameValidation ? " best" : ""}`}>
                          {metric(m.metrics[k])}
                        </td>
                      ))}
                    </tr>
                  );
                })}
              </tbody>
            </table>
          </div>
        </section>
      )}

      {models.length > 1 && (
        <section className="card">
          <div className="card-head">
            <h3>클래스별</h3>
            <span className="spacer" />
            <select value={basis} onChange={(e) => setBasis(e.target.value)} aria-label="클래스별 기준">
              <option value="own">학습 때 검증</option>
              {data?.evaluations.map((g) => (
                <option key={`${g.release.id}:${g.split}`} value={`${g.release.id}:${g.split}`}>
                  {g.release.dataset} {g.release.version} · {splitName(g.split)} 평가
                </option>
              ))}
            </select>
          </div>
          <ClassesSideBySide models={models} of={classSource} />
        </section>
      )}

      {evaluating && (
        <EvaluateAll
          models={models}
          onClose={() => setEvaluating(false)}
          onStarted={() => {
            setEvaluating(false);
            void refresh();
          }}
        />
      )}
    </>
  );
}

/** Classes as rows, the models as columns: the main per-class number of each (mAP50-95, or F1). */
function ClassesSideBySide({ models, of }: { models: Compared[]; of: (m: Compared) => ClassMetric[] }) {
  const names = [...new Set(models.flatMap((m) => of(m).map((c) => c.name)))];
  if (names.length === 0) return <p className="muted small">이 기준에는 클래스별 지표가 없습니다.</p>;
  return (
    <div className="table-wrap class-compare">
      <table className="plain">
        <thead>
          <tr>
            <th>클래스</th>
            {models.map((m) => (
              <th key={m.id} className="right">
                {label(m)}
              </th>
            ))}
          </tr>
        </thead>
        <tbody>
          {names.map((name) => {
            const values = models.map((m) => {
              const found = of(m).find((c) => c.name === name);
              return found ? classScore(found) : null;
            });
            const top = Math.max(...values.map((v) => v ?? -Infinity));
            return (
              <tr key={name}>
                <td>{name}</td>
                {values.map((v, i) => (
                  <td key={models[i].id} className={`num right${v != null && v === top && models.length > 1 ? " best" : ""}`}>
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

/** A box to find a model in any project and add it to the comparison. */
function ModelFinder({ taken, onPick }: { taken: string[]; onPick: (id: string) => void }) {
  const [typed, setTyped] = useState("");
  const [open, setOpen] = useState(false);
  const q = useDebounced(typed.trim());
  const { data } = usePolling(() => (open ? api<Found[]>(`/models-find?q=${encodeURIComponent(q)}&limit=20`) : Promise.resolve([])), 0, [q, open]);
  const shown = (data ?? []).filter((m) => !taken.includes(m.id));
  return (
    <div className="model-finder">
      <input
        type="search"
        value={typed}
        onChange={(e) => setTyped(e.target.value)}
        onFocus={() => setOpen(true)}
        onBlur={() => window.setTimeout(() => setOpen(false), 200)}
        placeholder="모델 더하기 (모든 프로젝트에서 찾기)"
        aria-label="비교할 모델 찾기"
      />
      {open && shown.length > 0 && (
        <ul className="model-finder-list">
          {shown.map((m) => (
            <li key={m.id}>
              <button
                type="button"
                onMouseDown={(e) => e.preventDefault()}
                onClick={() => {
                  onPick(m.id);
                  setTyped("");
                }}
              >
                <strong>{label(m)}</strong> <span className="muted small">{m.project.name}</span>
                <span className="num muted small">
                  {METRIC_LABEL[m.metric] ?? m.metric} {metric(m.value ?? undefined)}
                </span>
              </button>
            </li>
          ))}
        </ul>
      )}
    </div>
  );
}

/** Evaluate every model of the comparison on one version and split (any project's version of their task). */
function EvaluateAll({ models, onClose, onStarted }: { models: Compared[]; onClose: () => void; onStarted: () => void }) {
  const projects = [...new Map(models.map((m) => [m.project.id, m.project])).values()];
  const { data: allProjects } = usePolling(() => api<{ id: string; name: string }[]>("/projects"), 0);
  const [projectId, setProjectId] = useState(projects[0]?.id ?? "");
  const { data: tree } = usePolling(
    () => (projectId ? api<TreeDataset[]>(`/projects/${projectId}/datasets`) : Promise.resolve([] as TreeDataset[])),
    0,
    [projectId],
  );
  const [release, setRelease] = useState("");
  const [split, setSplit] = useState("val");
  const [busy, setBusy] = useState(false);
  const [results, setResults] = useState<{ name: string; ok: boolean; text: string }[] | null>(null);
  const fitting = (tree ?? []).filter((d) => sameTask(d.task, models[0]?.task));
  const releases = fitting.flatMap((d) => d.releases.filter((r) => r.status === "ready" && !r.deleted_at).map((r) => ({ id: r.id, text: `${d.name} ${r.version}` })));
  const current = release || releases[0]?.id || "";

  const start = async () => {
    setBusy(true);
    const done: { name: string; ok: boolean; text: string }[] = [];
    for (const m of models) {
      try {
        await api(`/models/${m.id}/eval`, { body: { release_id: current, split } });
        done.push({ name: label(m), ok: true, text: "평가를 시작했습니다" });
      } catch (err) {
        done.push({ name: label(m), ok: false, text: err instanceof Error ? err.message : String(err) });
      }
    }
    setResults(done);
    setBusy(false);
  };

  return (
    <Modal title="같은 버전으로 모두 평가" onClose={results ? onStarted : onClose}>
      {!results ? (
        <>
          <p className="muted">
            고른 버전 · split으로 비교 중인 모델 {models.length}개를 모두 평가합니다. 다른 프로젝트의 버전도 고를 수 있습니다(같은 태스크이고
            모델의 클래스가 그 버전에 있어야 함). 각 평가는 그 버전의 이미지가 있는 서버에서 돕니다.
          </p>
          <label>
            프로젝트
            <select value={projectId} onChange={(e) => (setProjectId(e.target.value), setRelease(""))}>
              {(allProjects ?? projects).map((p) => (
                <option key={p.id} value={p.id}>
                  {p.name}
                </option>
              ))}
            </select>
          </label>
          <label>
            버전
            <select value={current} onChange={(e) => setRelease(e.target.value)}>
              {releases.map((r) => (
                <option key={r.id} value={r.id}>
                  {r.text}
                </option>
              ))}
            </select>
          </label>
          {releases.length === 0 && tree && <div className="notice small">이 프로젝트에는 이 태스크의 버전이 없습니다.</div>}
          <label>
            평가할 부분
            <select value={split} onChange={(e) => setSplit(e.target.value)}>
              {["val", "test", "train"].map((s) => (
                <option key={s} value={s}>
                  {splitName(s)}
                </option>
              ))}
            </select>
          </label>
          <div className="modal-actions">
            <button type="button" onClick={onClose}>
              취소
            </button>
            <button className="primary" disabled={busy || !current} onClick={() => void start()}>
              {busy ? "시작하는 중…" : `${models.length}개 평가`}
            </button>
          </div>
        </>
      ) : (
        <>
          <ul className="plain-list">
            {results.map((r) => (
              <li key={r.name} className={r.ok ? "" : "bad-text"}>
                <strong>{r.name}</strong>: {r.text}
              </li>
            ))}
          </ul>
          <p className="muted small">끝나면 이 화면의 "같은 데이터로 평가한 결과"에 나옵니다.</p>
          <div className="modal-actions">
            <button className="primary" onClick={onStarted}>
              닫기
            </button>
          </div>
        </>
      )}
    </Modal>
  );
}
