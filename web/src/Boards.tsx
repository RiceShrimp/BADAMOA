import { type ReactNode, useMemo, useState } from "react";
import { AliasChip } from "./Annotate";
import { TaskTags, rememberProject } from "./DataBoard";
import { type Order, OrderReset, SortHandle, SortItem, Sortable, useOrder } from "./Order";
import { api } from "./api";
import { METRIC_LABEL, STAGE_LABEL, metric } from "./trainTypes";
import { Badge, Icon, STATUS_LABEL, Stat, duration, timeAgo, usePolling } from "./ui";

/** A project as the first screens show it. */
export interface BoardProject {
  id: string;
  name: string;
  folder: string;
  tasks: string[];
}

// ---- folders over the cards -------------------------------------------------------------

/**
 * "전체" and each folder the cards' projects are in (only when the cards fall in two or more of them), and a search
 * box once there are many: the dataset board's way of narrowing projects, without making folders here.
 */
export function useBoardFilter<T>(cards: T[] | null | undefined, projectOf: (card: T) => BoardProject) {
  const [folder, setFolder] = useState<string | null>(null); // null: all, "": no folder
  const [q, setQ] = useState("");
  const list = cards ?? [];
  const folders = useMemo(() => {
    const counts = new Map<string, number>();
    for (const c of list) {
      const f = projectOf(c).folder;
      counts.set(f, (counts.get(f) ?? 0) + 1);
    }
    const named = [...counts.entries()].filter(([f]) => f).sort(([a], [b]) => a.localeCompare(b, "ko"));
    return named.length > 0 && counts.get("") ? [...named, ["", counts.get("") ?? 0] as [string, number]] : named;
  }, [list, projectOf]);
  const needle = q.trim().toLowerCase();
  const shown = list.filter((c) => {
    const p = projectOf(c);
    return (folder === null || p.folder === folder) && (!needle || `${p.name} ${p.folder}`.toLowerCase().includes(needle));
  });
  const projects = new Set(list.map((c) => projectOf(c).id)).size;
  const tools =
    folders.length > 1 || projects > 6 ? (
      <div className="board-tools">
        {folders.length > 1 && (
          <div className="folder-chips" role="group" aria-label="폴더">
            <button className={folder === null ? "active" : ""} onClick={() => setFolder(null)}>
              전체 <em>{list.length}</em>
            </button>
            {folders.map(([name, n]) => (
              <button key={name || "-"} className={folder === name ? "active" : ""} onClick={() => setFolder(name)}>
                <Icon name="folder" size={13} />
                {name || "폴더 없음"} <em>{n}</em>
              </button>
            ))}
          </div>
        )}
        {projects > 6 && (
          <input type="search" value={q} onChange={(e) => setQ(e.target.value)} placeholder="프로젝트 찾기" aria-label="프로젝트 찾기" />
        )}
      </div>
    ) : null;
  return { shown, tools };
}

const cardProject = (card: { project: BoardProject }) => card.project;
const cardId = (card: { project: BoardProject }) => card.project.id;

function BoardHead({ title, order, children }: { title: string; order?: Order<unknown>; children: ReactNode }) {
  return (
    <div className="page-head">
      <div>
        <h1>{title}</h1>
        <p className="muted">{children}</p>
      </div>
      {order && (
        <div className="head-actions">
          <OrderReset order={order} />
        </div>
      )}
    </div>
  );
}

/** A small line of a project's finished runs, oldest to newest. */
function Trend({ values }: { values: number[] }) {
  if (values.length < 2) return null;
  const lo = Math.min(...values);
  const hi = Math.max(...values);
  const span = hi - lo || 1;
  const points = values.map((v, i) => `${(i / (values.length - 1)) * 100},${28 - ((v - lo) / span) * 24}`).join(" ");
  return (
    <svg className="board-trend" viewBox="0 0 100 32" preserveAspectRatio="none" aria-hidden="true">
      <polyline points={points} fill="none" stroke="var(--accent)" strokeWidth="2" vectorEffect="non-scaling-stroke" />
    </svg>
  );
}

function metricName(key: string): string {
  return METRIC_LABEL[key] ?? key;
}

function about(seconds: number | null): string {
  if (seconds == null) return "남은 시간 계산 중";
  if (seconds < 60) return "곧 끝남";
  if (seconds < 3600) return `약 ${Math.round(seconds / 60)}분 남음`;
  return `약 ${Math.floor(seconds / 3600)}시간 ${Math.round((seconds % 3600) / 60)}분 남음`;
}

// ---- 학습 -------------------------------------------------------------------------------

interface GoingRun {
  id: string;
  name: string;
  status: "queued" | "running";
  stage: string;
  task: string;
  epoch: number;
  epochs: number;
  release: { id: string; dataset: string; version: string } | null;
  node_name: string | null;
  gpu_indices: number[];
  wait_reason: string | null;
  duration_s: number | null;
  project: { id: string; name: string };
  metric: string;
  value: number | null;
  eta_s: number | null;
}

interface RunRow {
  id: string;
  name: string;
  status: string;
  epoch: number;
  epochs: number;
  value: number | null;
  created_at: string;
}

interface TrainingCard {
  project: BoardProject;
  metric: string;
  runs: number;
  running: number;
  queued: number;
  succeeded: number;
  failed: number;
  canceled: number;
  best: { id: string; name: string; value: number } | null;
  trend: number[];
  recent: RunRow[];
  last_at: string | null;
}

interface TrainingBoardInfo {
  going: GoingRun[];
  projects: TrainingCard[];
  totals: { runs: number; running: number; queued: number; succeeded: number; failed: number; gpus: number };
}

const STATUS_KIND: Record<string, string> = {
  running: "running",
  queued: "queued",
  succeeded: "succeeded",
  failed: "failed",
  canceled: "idle",
};

/** 학습's first screen: the runs going on now, then each project's runs summed up. */
export function TrainingBoard() {
  const { data: board, error } = usePolling(() => api<TrainingBoardInfo>("/boards/training"), 5000);
  const order = useOrder("training.projects", board?.projects, cardId);
  const { shown, tools } = useBoardFilter(board ? order.ordered : null, cardProject);
  if (error && !board) return <div className="error">{error}</div>;
  const t = board?.totals;
  return (
    <>
      <BoardHead title="학습" order={order as Order<unknown>}>
        지금 도는 학습과 프로젝트별 학습을 한눈에 봅니다. 프로젝트를 누르면 학습 표에서 실행을 비교하고 새 학습을 시작합니다. 프로젝트
        카드는 끌어서 순서를 바꿀 수 있습니다.
      </BoardHead>
      <div className="stats metrics">
        <Stat icon="activity" label="실행 중" value={t?.running ?? "–"} sub={t?.queued ? `대기 ${t.queued}` : "대기 없음"} />
        <Stat icon="chip" label="쓰는 GPU" value={t?.gpus ?? "–"} sub="실행 중인 학습이 잡은 GPU" />
        <Stat icon="check" label="성공" value={t?.succeeded ?? "–"} sub={`전체 ${t?.runs ?? 0}`} />
        <Stat icon="x" label="실패" value={t?.failed ?? "–"} sub="로그에서 이유를 봅니다" />
      </div>

      {board && board.going.length > 0 && (
        <section className="board-section">
          <h2>
            지금 도는 학습 <span className="muted">{board.going.length}</span>
          </h2>
          <div className="going-list">
            {board.going.map((r) => {
              const share = r.epochs ? Math.min(1, r.epoch / r.epochs) : 0;
              return (
                <a key={r.id} className="card going-run" href={`#/training/${r.id}`}>
                  <div className="going-title">
                    <Badge kind={STATUS_KIND[r.status]}>{STATUS_LABEL[r.status]}</Badge>
                    <strong>{r.name}</strong>
                    <span className="muted small">
                      {r.project.name}
                      {r.release ? ` · ${r.release.dataset} ${r.release.version}` : ""}
                    </span>
                  </div>
                  {r.status === "running" ? (
                    <>
                      <div className="going-bar" title={`${r.epoch} / ${r.epochs} epoch`}>
                        <i style={{ width: `${share * 100}%` }} />
                      </div>
                      <div className="going-meta small">
                        <span>
                          {STAGE_LABEL[r.stage] ?? r.stage} · epoch {r.epoch}/{r.epochs} ({Math.round(share * 100)}%)
                        </span>
                        <span>{r.stage === "train" || r.epoch > 0 ? about(r.eta_s) : `${duration(r.duration_s)} 지남`}</span>
                        <span className="muted">
                          {r.node_name ?? "노드 ?"}
                          {r.gpu_indices.length > 0 ? ` · GPU ${r.gpu_indices.join(", ")}` : " · CPU"}
                        </span>
                        {r.value != null && (
                          <span>
                            {metricName(r.metric)} <strong>{metric(r.value)}</strong>
                          </span>
                        )}
                      </div>
                    </>
                  ) : (
                    <div className="muted small">{r.wait_reason ?? "차례를 기다리는 중"}</div>
                  )}
                </a>
              );
            })}
          </div>
        </section>
      )}

      <section className="board-section">
        <h2>프로젝트</h2>
        {tools}
        {board && board.projects.length === 0 && <div className="card empty">아직 프로젝트가 없습니다.</div>}
        <Sortable ids={shown.map(cardId)} onMove={order.move}>
          <div className="card-grid board-grid">
            {shown.map((c) => (
              <SortItem key={c.project.id} id={c.project.id}>
                <div className="data-card">
                  <a className="data-card-link" href={`#/training/project/${c.project.id}`} onClick={() => rememberProject(c.project.id)}>
                    <div className="data-card-body">
                      <div className="data-card-title">
                        <span className="with-grip"><SortHandle /><strong>{c.project.name}</strong></span>
                        {c.running + c.queued > 0 && (
                          <Badge kind="running">{c.running ? `실행 중 ${c.running}` : `대기 ${c.queued}`}</Badge>
                        )}
                      </div>
                      {c.project.tasks.length > 0 && (
                        <div className="task-tags">
                          <TaskTags tasks={c.project.tasks} />
                        </div>
                      )}
                      {c.runs === 0 ? (
                        <p className="muted small">아직 학습이 없습니다. 눌러서 새 학습을 시작합니다.</p>
                      ) : (
                        <>
                          <div className="small">
                            학습 {c.runs} · 성공 {c.succeeded}
                            {c.failed > 0 && <span className="bad-text"> · 실패 {c.failed}</span>}
                          </div>
                          <div className="board-best">
                            <span className="muted small">최고 {metricName(c.metric)}</span>
                            <strong className="num">{c.best ? metric(c.best.value) : "–"}</strong>
                            <Trend values={c.trend} />
                          </div>
                          {c.best && <div className="muted small ellipsis">{c.best.name}</div>}
                        </>
                      )}
                    </div>
                  </a>
                  {c.recent.length > 0 && (
                    <div className="board-lines">
                      {c.recent.map((r) => (
                        <a key={r.id} href={`#/training/${r.id}`} className="board-line">
                          <Badge kind={STATUS_KIND[r.status] ?? "idle"}>{STATUS_LABEL[r.status] ?? r.status}</Badge>
                          <span className="ellipsis">{r.name}</span>
                          <span className="num muted">
                            {r.value != null ? metric(r.value) : r.status === "running" ? `${r.epoch}/${r.epochs}` : ""}
                          </span>
                        </a>
                      ))}
                      <div className="muted small board-foot">마지막 학습 {timeAgo(c.last_at)}</div>
                    </div>
                  )}
                </div>
              </SortItem>
            ))}
          </div>
        </Sortable>
      </section>
    </>
  );
}

// ---- 모델 -------------------------------------------------------------------------------

interface ModelRow {
  id: string;
  name: string;
  version: number;
  aliases: string[];
  task: string;
  arch: string;
  metric: string;
  value: number | null;
  created_at: string;
}

interface ModelsCard {
  project: BoardProject;
  metric: string;
  models: number;
  families: number;
  aliased: ModelRow[];
  aliased_count: number;
  best: ModelRow | null;
  latest: ModelRow | null;
}

interface ModelsBoardInfo {
  projects: ModelsCard[];
  totals: { models: number; families: number; aliased: number; production: number; projects: number };
}

/** 모델's first screen: each project's models, the ones with an alias (SOTA, production …) first. */
export function ModelsBoard() {
  const { data: board, error } = usePolling(() => api<ModelsBoardInfo>("/boards/models"), 30000);
  const order = useOrder("models.projects", board?.projects, cardId);
  const { shown, tools } = useBoardFilter(board ? order.ordered : null, cardProject);
  if (error && !board) return <div className="error">{error}</div>;
  const t = board?.totals;
  return (
    <>
      <BoardHead title="모델" order={order as Order<unknown>}>
        프로젝트별로 SOTA · production 같은 별칭이 붙은 모델과 가장 좋은 모델을 봅니다. 프로젝트를 누르면 모델 목록에서 비교 · 평가 ·
        내보내기를 합니다. 프로젝트 카드는 끌어서 순서를 바꿀 수 있습니다.
      </BoardHead>
      <div className="stats metrics">
        <Stat icon="box" label="모델 버전" value={t?.models ?? "–"} sub={`모델 ${t?.families ?? 0}종`} />
        <Stat icon="check" label="별칭 붙은 버전" value={t?.aliased ?? "–"} sub="SOTA · production 등" />
        <Stat icon="activity" label="production" value={t?.production ?? "–"} sub="서비스에 쓰는 버전" />
        <Stat icon="database" label="모델 있는 프로젝트" value={t?.projects ?? "–"} sub={`전체 ${board?.projects.length ?? 0}`} />
      </div>
      {tools}
      {board && board.projects.length === 0 && <div className="card empty">아직 프로젝트가 없습니다.</div>}
      <Sortable ids={shown.map(cardId)} onMove={order.move}>
        <div className="card-grid board-grid">
          {shown.map((c) => (
            <SortItem key={c.project.id} id={c.project.id}>
              <div className={`data-card${c.models === 0 ? " board-quiet" : ""}`}>
                <a className="data-card-link" href={`#/models/project/${c.project.id}`} onClick={() => rememberProject(c.project.id)}>
                  <div className="data-card-body">
                    <div className="data-card-title">
                      <span className="with-grip"><SortHandle /><strong>{c.project.name}</strong></span>
                    </div>
                    {c.project.tasks.length > 0 && (
                      <div className="task-tags">
                        <TaskTags tasks={c.project.tasks} />
                      </div>
                    )}
                    {c.models === 0 ? (
                      <p className="muted small">아직 모델이 없습니다. 학습이 끝나면 여기에 자동으로 등록됩니다.</p>
                    ) : (
                      <>
                        <div className="small">
                          모델 {c.models}개 · {c.families}종
                        </div>
                        <div className="board-best">
                          <span className="muted small">최고 {metricName(c.metric)}</span>
                          <strong className="num">{c.best ? metric(c.best.value) : "–"}</strong>
                        </div>
                        {c.best && (
                          <div className="muted small ellipsis">
                            {c.best.name} v{c.best.version}
                          </div>
                        )}
                      </>
                    )}
                  </div>
                </a>
                {c.models > 0 && (
                  <div className="board-lines">
                    {c.aliased.length === 0 && <div className="muted small">별칭 붙은 모델이 없습니다</div>}
                    {c.aliased.map((m) => (
                      <a key={m.id} href={`#/models/${m.id}`} className="board-line">
                        <span className="board-aliases">
                          {m.aliases.map((a) => (
                            <AliasChip key={a} alias={a} />
                          ))}
                        </span>
                        <span className="ellipsis">
                          {m.name} v{m.version}
                        </span>
                        <span className="num muted" title={metricName(m.metric)}>
                          {metric(m.value)}
                        </span>
                      </a>
                    ))}
                    {c.aliased_count > c.aliased.length && (
                      <div className="muted small">별칭 붙은 버전 {c.aliased_count - c.aliased.length}개 더</div>
                    )}
                    {c.latest && (
                      <div className="muted small board-foot">
                        최근 등록 {c.latest.name} v{c.latest.version} · {timeAgo(c.latest.created_at)}
                      </div>
                    )}
                  </div>
                )}
              </div>
            </SortItem>
          ))}
        </div>
      </Sortable>
    </>
  );
}
