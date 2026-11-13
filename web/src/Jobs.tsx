import { useEffect, useRef, useState } from "react";
import { type Job, type JobType, type LogLine, type NodeInfo, api } from "./api";
import { Badge, Modal, STATUS_LABEL, duration, timeAgo, usePolling } from "./ui";

const VIA_LABEL: Record<string, string> = { ui: "웹", mcp: "에이전트(MCP)", api: "API" };

export function Jobs() {
  const { data: jobs, error, refresh } = usePolling(() => api<Job[]>("/jobs?limit=100"), 3000);
  const [selected, setSelected] = useState<string | null>(null);
  const [creating, setCreating] = useState(false);

  return (
    <>
      <div className="page-head">
        <div>
          <h1>작업</h1>
          <p className="muted">노드에서 실행되는 모든 작업과 로그입니다.</p>
        </div>
        <button className="primary" onClick={() => setCreating(true)}>
          + 작업 실행
        </button>
      </div>
      {error && <div className="error">{error}</div>}
      <div className="card table-wrap">
        <table>
          <thead>
            <tr>
              <th>작업</th>
              <th>상태</th>
              <th>노드</th>
              <th>요청</th>
              <th>소요</th>
              <th>생성</th>
            </tr>
          </thead>
          <tbody>
            {jobs?.length === 0 && (
              <tr>
                <td colSpan={6} className="empty">
                  아직 실행한 작업이 없습니다.
                </td>
              </tr>
            )}
            {jobs?.map((job) => (
              <tr key={job.id} onClick={() => setSelected(job.id)} className="clickable">
                <td>
                  <div>{job.title}</div>
                  <div className="muted small mono">{job.type}</div>
                </td>
                <td>
                  <Badge kind={job.status}>{STATUS_LABEL[job.status]}</Badge>
                </td>
                <td>{job.node_name ?? <span className="muted">배정 대기</span>}</td>
                <td>{VIA_LABEL[job.created_via] ?? job.created_via}</td>
                <td>{duration(job.duration_s)}</td>
                <td className="muted">{timeAgo(job.created_at)}</td>
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
          <div className="job-meta">
            <Badge kind={job.status}>{STATUS_LABEL[job.status]}</Badge>
            <span>{job.node_name ?? "배정 대기"}</span>
            <span className="muted">{duration(job.duration_s)}</span>
            <span className="muted mono small">{job.id}</span>
          </div>
          {job.error && <div className="error">{job.error}</div>}
          <pre className="log" ref={logRef}>
            {lines.length === 0 ? (
              <span className="muted">{active ? "로그를 기다리는 중…" : "로그가 없습니다."}</span>
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
      const job = await api<Job>("/jobs", { body: { type, params: body, node_id: nodeId || null } });
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
