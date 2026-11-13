import { useState } from "react";
import { type JoinToken, type NodeInfo, api } from "./api";
import { Badge, CopyBlock, Meter, Modal, STATUS_LABEL, timeAgo, usePolling } from "./ui";

const ROLES = [
  { id: "compute", label: "학습·오토라벨" },
  { id: "storage", label: "데이터 저장" },
  { id: "inference", label: "추론" },
];

export function Nodes({ isAdmin }: { isAdmin: boolean }) {
  const { data: nodes, error, refresh } = usePolling(() => api<NodeInfo[]>("/nodes"), 5000);
  const [adding, setAdding] = useState(false);
  const [editing, setEditing] = useState<NodeInfo | null>(null);

  const runInfo = async (node: NodeInfo) => {
    await api("/jobs", { body: { type: "system.info", node_id: node.id } });
    window.location.hash = "#/jobs";
  };

  return (
    <>
      <div className="page-head">
        <div>
          <h1>노드</h1>
          <p className="muted">Hub에 연결된 학습·데이터·추론 서버입니다.</p>
        </div>
        {isAdmin && (
          <button className="primary" onClick={() => setAdding(true)}>
            + 노드 추가
          </button>
        )}
      </div>
      {error && <div className="error">{error}</div>}
      {nodes && nodes.length === 0 && <div className="card empty">연결된 노드가 없습니다.</div>}
      <div className="grid">
        {nodes?.map((node) => {
          const gpuMetrics = node.metrics.gpus ?? [];
          return (
            <section className="card node" key={node.id}>
              <header>
                <div>
                  <h3>{node.name}</h3>
                  <div className="muted small">
                    {node.os || "?"} · {node.info.cpu_count ?? "?"} 코어 ·{" "}
                    {node.info.memory_total_mb ? `${Math.round(node.info.memory_total_mb / 1024)} GB RAM` : "?"}
                  </div>
                </div>
                <Badge kind={node.status}>{STATUS_LABEL[node.status]}</Badge>
              </header>
              <div className="tags">
                {node.is_local && <span className="tag accent">Hub 머신</span>}
                {node.roles.map((r) => (
                  <span className="tag" key={r}>
                    {ROLES.find((x) => x.id === r)?.label ?? r}
                  </span>
                ))}
              </div>
              <dl>
                {node.info.gpus.length === 0 && (
                  <>
                    <dt>GPU</dt>
                    <dd className="muted">없음</dd>
                  </>
                )}
                {node.info.gpus.map((gpu) => {
                  const m = gpuMetrics.find((x) => x.index === gpu.index);
                  return (
                    <div className="row" key={gpu.index}>
                      <dt>GPU {gpu.index}</dt>
                      <dd>
                        <div>
                          {gpu.name}{" "}
                          <span className="muted">
                            {gpu.memory_total_mb ? `${Math.round(gpu.memory_total_mb / 1024)} GB` : ""}
                          </span>
                        </div>
                        <Meter value={m?.memory_used_mb} max={m?.memory_total_mb} unit="MB" />
                      </dd>
                    </div>
                  );
                })}
                <div className="row">
                  <dt>메모리</dt>
                  <dd>
                    <Meter value={node.metrics.memory_used_mb} max={node.metrics.memory_total_mb} unit="MB" />
                  </dd>
                </div>
                <div className="row">
                  <dt>디스크</dt>
                  <dd>
                    {node.metrics.disk_free_gb != null ? (
                      <span>
                        {node.metrics.disk_free_gb} GB 여유 <span className="muted">/ {node.metrics.disk_total_gb} GB</span>
                      </span>
                    ) : (
                      <span className="muted">–</span>
                    )}
                  </dd>
                </div>
              </dl>
              <footer>
                <span className="muted small">마지막 응답 {timeAgo(node.last_seen_at)}</span>
                <span className="actions">
                  <button onClick={() => runInfo(node)} disabled={node.status !== "online"}>
                    시스템 점검
                  </button>
                  {isAdmin && <button onClick={() => setEditing(node)}>설정</button>}
                </span>
              </footer>
            </section>
          );
        })}
      </div>
      {adding && <AddNode onClose={() => setAdding(false)} />}
      {editing && (
        <EditNode
          node={editing}
          onClose={() => {
            setEditing(null);
            void refresh();
          }}
        />
      )}
    </>
  );
}

function RolePicker({ roles, onChange }: { roles: string[]; onChange: (roles: string[]) => void }) {
  return (
    <div className="checks">
      {ROLES.map((r) => (
        <label key={r.id}>
          <input
            type="checkbox"
            checked={roles.includes(r.id)}
            onChange={(e) => onChange(e.target.checked ? [...roles, r.id] : roles.filter((x) => x !== r.id))}
          />
          {r.label}
        </label>
      ))}
    </div>
  );
}

function AddNode({ onClose }: { onClose: () => void }) {
  const [roles, setRoles] = useState(ROLES.map((r) => r.id));
  const [join, setJoin] = useState<JoinToken | null>(null);
  const [error, setError] = useState<string | null>(null);

  const create = async () => {
    try {
      setJoin(await api<JoinToken>("/nodes/join-tokens", { body: { roles, ttl_minutes: 60 } }));
      setError(null);
    } catch (e) {
      setError(e instanceof Error ? e.message : String(e));
    }
  };

  return (
    <Modal title="노드 추가" onClose={onClose}>
      {!join ? (
        <>
          <p>추가할 서버가 맡을 역할을 고르세요. 나중에 바꿀 수 있습니다.</p>
          <RolePicker roles={roles} onChange={setRoles} />
          {error && <div className="error">{error}</div>}
          <div className="modal-actions">
            <button className="primary" onClick={create} disabled={roles.length === 0}>
              설치 명령 만들기
            </button>
          </div>
        </>
      ) : (
        <>
          <p>
            추가할 서버의 터미널에서 아래 명령을 실행하세요. 명령은 <strong>1회용</strong>이고 1시간 뒤 만료됩니다.
          </p>
          <CopyBlock label="Ubuntu / Linux" text={join.commands.linux} />
          <CopyBlock label="Windows (관리자 PowerShell)" text={join.commands.windows} />
          <p className="muted small">
            이 주소({join.hub_url})로 원격 서버가 접속할 수 있어야 합니다. 설치가 끝나면 노드 목록에 자동으로 나타납니다.
          </p>
          <div className="modal-actions">
            <button onClick={onClose}>닫기</button>
          </div>
        </>
      )}
    </Modal>
  );
}

function EditNode({ node, onClose }: { node: NodeInfo; onClose: () => void }) {
  const [name, setName] = useState(node.name);
  const [roles, setRoles] = useState(node.roles);
  const [enabled, setEnabled] = useState(node.enabled);
  const [error, setError] = useState<string | null>(null);

  const run = async (action: () => Promise<unknown>) => {
    try {
      await action();
      onClose();
    } catch (e) {
      setError(e instanceof Error ? e.message : String(e));
    }
  };

  return (
    <Modal title={`${node.name} 설정`} onClose={onClose}>
      <label>
        이름
        <input value={name} onChange={(e) => setName(e.target.value)} />
      </label>
      <div className="field-label">역할</div>
      <RolePicker roles={roles} onChange={setRoles} />
      <div className="checks">
        <label>
          <input type="checkbox" checked={enabled} onChange={(e) => setEnabled(e.target.checked)} />
          작업 받기 (끄면 새 작업을 배정하지 않습니다)
        </label>
      </div>
      {error && <div className="error">{error}</div>}
      <div className="modal-actions">
        {!node.is_local && (
          <button
            className="danger"
            onClick={() => {
              if (window.confirm(`노드 "${node.name}"을(를) Hub에서 제거할까요?`))
                void run(() => api(`/nodes/${node.id}`, { method: "DELETE" }));
            }}
          >
            노드 제거
          </button>
        )}
        <span className="spacer" />
        <button onClick={onClose}>취소</button>
        <button
          className="primary"
          disabled={roles.length === 0}
          onClick={() => run(() => api(`/nodes/${node.id}`, { method: "PATCH", body: { name, roles, enabled } }))}
        >
          저장
        </button>
      </div>
    </Modal>
  );
}
