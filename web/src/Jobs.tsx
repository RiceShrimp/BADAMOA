import { useEffect, useRef, useState } from "react";
import { type Job, type JobType, type LogLine, type NodeInfo, api } from "./api";
import { Badge, Icon, Modal, STATUS_LABEL, Stat, VIA_LABEL, duration, timeAgo, usePolling } from "./ui";

const FILTERS = ["all", "running", "queued", "succeeded", "failed", "canceled"] as const;
type Filter = (typeof FILTERS)[number];

export function Jobs() {
  const { data: jobs, error, refresh } = usePolling(() => api<Job[]>("/jobs?limit=200"), 3000);
  const [filter, setFilter] = useState<Filter>("all");
  const [selected, setSelected] = useState<string | null>(null);
  const [creating, setCreating] = useState(false);

  const count = (status: string) => jobs?.filter((j) => j.status === status).length ?? 0;
  const shown = jobs?.filter((j) => filter === "all" || j.status === filter);

  return (
    <>
      <div className="page-head">
        <div>
          <h1>작업</h1>
          <p className="muted">노드에서 실행되는 모든 작업입니다. 행을 누르면 실시간 로그를 볼 수 있습니다.</p>
        </div>
        <button className="primary" onClick={() => setCreating(true)}>
          <Icon name="plus" />
          작업 실행
        </button>
      </div>

      <div className="stats">
        <Stat icon="activity" label="실행 중" value={count("running")} sub="지금 노드에서 도는 작업" />
        <Stat icon="clock" label="대기" value={count("queued")} sub="노드 배정을 기다리는 작업" />
        <Stat icon="check" label="성공" value={count("succeeded")} sub="최근 200개 기준" />
        <Stat icon="alert" label="실패" value={count("failed")} sub="최근 200개 기준" />
      </div>

      <div className="chips" role="tablist" aria-label="상태 필터">
        {FILTERS.map((f) => (
          <button
            key={f}
            role="tab"
            aria-selected={filter === f}
            className={`chip ${filter === f ? "active" : ""}`}
            onClick={() => setFilter(f)}
          >
            {f === "all" ? "전체" : STATUS_LABEL[f]}
            <span>{f === "all" ? (jobs?.length ?? 0) : count(f)}</span>
          </button>
        ))}
      </div>

      {error && <div className="error">{error}</div>}
      <div className="card table-wrap flush">
        <table>
          <thead>
            <tr>
              <th>작업</th>
              <th>상태</th>
              <th>노드</th>
              <th>요청</th>
              <th>소요</th>
              <th className="right">생성</th>
            </tr>
          </thead>
          <tbody>
            {shown?.length === 0 && (
              <tr>
                <td colSpan={6} className="empty">
                  {filter === "all" ? "아직 실행한 작업이 없습니다." : "이 상태의 작업이 없습니다."}
                </td>
              </tr>
            )}
            {shown?.map((job) => (
              <tr key={job.id} onClick={() => setSelected(job.id)} className="clickable">
                <td>
                  <strong>{job.title}</strong>
                  <div className="muted small mono">{job.type}</div>
                </td>
                <td>
                  <Badge kind={job.status}>{STATUS_LABEL[job.status]}</Badge>
                  {job.status === "queued" && job.wait_reason && <div className="wait-reason">{job.wait_reason}</div>}
                </td>
                <td>
                  {job.node_name ?? <span className="muted">배정 대기</span>}
                  {job.gpu_indices.length > 0 ? (
                    <div className="muted small">GPU {job.gpu_indices.join(", ")}</div>
                  ) : (
                    job.gpus_requested > 0 && <div className="muted small">GPU {job.gpus_requested}개 필요</div>
                  )}
                </td>
                <td>
                  <span className="tag">{VIA_LABEL[job.created_via] ?? job.created_via}</span>
                </td>
                <td className="num">{duration(job.duration_s)}</td>
                <td className="muted right">{timeAgo(job.created_at)}</td>
              </tr>
            ))}
          </tbody>
        </table>
      </div>
      {selected && <JobDetail jobId={selected} onClose={() => setSelected(null)} />}
      {creating && (
        <NewJob
          onClose={() => setCreating(false)}
          onCreated={(id) => {
            setCreating(false);
            setSelected(id);
            void refresh();
          }}
        />
      )}
    </>
  );
}

function JobDetail({ jobId, onClose }: { jobId: string; onClose: () => void }) {
  const { data: job, refresh } = usePolling(() => api<Job>(`/jobs/${jobId}`), 1500, [jobId]);
  const [lines, setLines] = useState<LogLine[]>([]);
  const lastSeq = useRef(0);
  const logRef = useRef<HTMLPreElement>(null);
  const logCount = job?.log_count ?? 0;

  useEffect(() => {
    if (logCount <= lastSeq.current) return;
    let alive = true;
    void api<LogLine[]>(`/jobs/${jobId}/logs?after=${lastSeq.current}`).then((fresh) => {
      if (!alive || fresh.length === 0) return;
      lastSeq.current = fresh[fresh.length - 1].seq;
      setLines((prev) => [...prev, ...fresh.filter((l) => !prev.some((p) => p.seq === l.seq))]);
    });
    return () => {
      alive = false;
    };
  }, [jobId, logCount]);

  useEffect(() => {
    const el = logRef.current;
    if (el) el.scrollTop = el.scrollHeight;
  }, [lines.length]);

  const active = job && (job.status === "queued" || job.status === "running");

  return (
    <Modal title={job ? job.title : "작업"} onClose={onClose}>
      {job && (
        <>
          <div className="facts">
            <div>
              <div className="eyebrow">상태</div>
              <Badge kind={job.status}>{STATUS_LABEL[job.status]}</Badge>
            </div>
            <div>
              <div className="eyebrow">노드</div>
              {job.node_name ?? "배정 대기"}
            </div>
            <div>
              <div className="eyebrow">소요</div>
              {duration(job.duration_s)}
            </div>
            <div>
              <div className="eyebrow">요청</div>
              {VIA_LABEL[job.created_via] ?? job.created_via}
            </div>
            <div>
              <div className="eyebrow">GPU</div>
              {job.gpu_indices.length > 0
                ? job.gpu_indices.map((i) => `#${i}`).join(", ")
                : job.gpus_requested > 0
                  ? `${job.gpus_requested}개 필요`
                  : "사용 안 함"}
            </div>
            <div>
              <div className="eyebrow">ID</div>
              <span className="mono small">{job.id.slice(0, 12)}</span>
            </div>
          </div>
          {job.error && <div className="error">{job.error}</div>}
          {job.status === "queued" && job.wait_reason && <div className="notice">{job.wait_reason}</div>}
          <pre className="log" ref={logRef}>
            {lines.length === 0 ? (
              <span className="dim">{active ? "로그를 기다리는 중…" : "로그가 없습니다."}</span>
            ) : (
              lines.map((l) => (
                <div key={l.seq} className={l.stream === "system" ? "log-system" : ""}>
                  {l.text}
                </div>
              ))
            )}
          </pre>
          {job.result != null && (
            <details>
              <summary>결과 (JSON)</summary>
              <pre className="json">{JSON.stringify(job.result, null, 2)}</pre>
            </details>
          )}
          <div className="modal-actions">
            {active && (
              <button
                className="danger"
                disabled={job.cancel_requested}
                onClick={() => void api(`/jobs/${job.id}/cancel`, { method: "POST" }).then(refresh)}
              >
                {job.cancel_requested ? "취소 중…" : "작업 취소"}
              </button>
            )}
            <span className="spacer" />
            <button onClick={onClose}>닫기</button>
          </div>
        </>
      )}
    </Modal>
  );
}

function NewJob({ onClose, onCreated }: { onClose: () => void; onCreated: (id: string) => void }) {
  const [types, setTypes] = useState<JobType[]>([]);
  const [nodes, setNodes] = useState<NodeInfo[]>([]);
  const [type, setType] = useState("");
  const [nodeId, setNodeId] = useState("");
  const [params, setParams] = useState<Record<string, string>>({});
  const [gpus, setGpus] = useState("");
  const [error, setError] = useState<string | null>(null);

  useEffect(() => {
    void api<JobType[]>("/job-types").then((t) => {
      setTypes(t);
      setType(t[0]?.type ?? "");
    });
    void api<NodeInfo[]>("/nodes").then(setNodes);
  }, []);

  const spec = types.find((t) => t.type === type);
  const props = Object.entries(spec?.params_schema.properties ?? {});

  const submit = async () => {
    const body: Record<string, unknown> = {};
    for (const [name, schema] of props) {
      const raw = params[name];
      if (raw === undefined || raw === "") continue;
      body[name] = schema.type === "integer" ? Number(raw) : schema.type === "boolean" ? raw === "true" : raw;
    }
    try {
      const job = await api<Job>("/jobs", {
        body: { type, params: body, node_id: nodeId || null, gpus: gpus === "" ? null : Number(gpus) },
      });
      onCreated(job.id);
    } catch (e) {
      setError(e instanceof Error ? e.message : String(e));
    }
  };

  return (
    <Modal title="작업 실행" onClose={onClose}>
      <label>
        작업 유형
        <select
          value={type}
          onChange={(e) => {
            setType(e.target.value);
            setParams({});
          }}
        >
          {types.map((t) => (
            <option key={t.type} value={t.type}>
              {t.title}
            </option>
          ))}
        </select>
      </label>
      {spec && <p className="muted small">{spec.description}</p>}
      <label>
        실행할 노드
        <select value={nodeId} onChange={(e) => setNodeId(e.target.value)}>
          <option value="">자동 (조건에 맞는 아무 노드)</option>
          {nodes
            .filter((n) => n.enabled)
            .map((n) => (
              <option key={n.id} value={n.id}>
                {n.name} ({STATUS_LABEL[n.status]})
              </option>
            ))}
        </select>
      </label>
      <label>
        GPU 수
        <input
          type="number"
          min={0}
          max={16}
          placeholder={`기본값 ${spec?.gpus ?? 0}`}
          value={gpus}
          onChange={(e) => setGpus(e.target.value)}
        />
      </label>
      <p className="muted small">
        다른 사람이 쓰고 있는 GPU에는 배정되지 않습니다. 빈 GPU가 없으면 대기하고, 대기 사유가 작업 목록에 표시됩니다.
      </p>
      {props.map(([name, schema]) =>
        schema.type === "boolean" ? (
          <div className="checks" key={name}>
            <label>
              <input
                type="checkbox"
                checked={(params[name] ?? String(schema.default ?? false)) === "true"}
                onChange={(e) => setParams({ ...params, [name]: String(e.target.checked) })}
              />
              {name}
            </label>
          </div>
        ) : (
          <label key={name}>
            {name}
            <input
              type={schema.type === "integer" ? "number" : "text"}
              placeholder={schema.default != null ? String(schema.default) : ""}
              value={params[name] ?? ""}
              onChange={(e) => setParams({ ...params, [name]: e.target.value })}
            />
          </label>
        ),
      )}
      {error && <div className="error">{error}</div>}
      <div className="modal-actions">
        <span className="spacer" />
        <button onClick={onClose}>취소</button>
        <button className="primary" onClick={submit} disabled={!type}>
          실행
        </button>
      </div>
    </Modal>
  );
}
