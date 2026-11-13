import { useState } from "react";
import { type Job, type JoinToken, type NodeInfo, api } from "./api";
import { Badge, CopyBlock, Icon, Meter, Modal, STATUS_LABEL, Stat, gb, timeAgo, usePolling } from "./ui";

const ROLES = [
  { id: "compute", label: "학습·오토라벨" },
  { id: "storage", label: "데이터 저장" },
  { id: "inference", label: "추론" },
];

export function Nodes({ isAdmin }: { isAdmin: boolean }) {
  const { data: nodes, error, refresh } = usePolling(() => api<NodeInfo[]>("/nodes"), 5000);
  const { data: running } = usePolling(() => api<Job[]>("/jobs?status=running&limit=100"), 5000);
  const [adding, setAdding] = useState(false);
  const [editing, setEditing] = useState<NodeInfo | null>(null);

  const online = nodes?.filter((n) => n.status === "online") ?? [];
  const gpus = online.flatMap((n) => n.info.gpus);
  const vram = gpus.reduce((sum, g) => sum + (g.memory_total_mb ?? 0), 0);
  const disk = online.reduce((sum, n) => sum + (n.metrics.disk_free_gb ?? 0), 0);

  const runInfo = async (node: NodeInfo) => {
    await api("/jobs", { body: { type: "system.info", node_id: node.id } });
    window.location.hash = "#/jobs";
  };

  return (
    <>
      <div className="page-head">
        <div>
          <h1>노드</h1>
          <p className="muted">Hub에 연결된 학습·데이터·추론 서버입니다. 서버를 추가하면 작업이 자동으로 분배됩니다.</p>
        </div>
        {isAdmin && (
          <button className="primary" onClick={() => setAdding(true)}>
            <Icon name="plus" />
            노드 추가
          </button>
        )}
      </div>

      <div className="stats">
        <Stat
          icon="server"
          label="온라인"
          value={
            <>
              {online.length}
              <small> / {nodes?.length ?? 0}</small>
            </>
          }
          sub="노드"
        />
        <Stat icon="chip" label="GPU" value={gpus.length} sub={vram ? `VRAM 합계 ${gb(vram)}` : "GPU 없음"} />
        <Stat icon="database" label="디스크 여유" value={disk ? Math.round(disk) : "–"} sub="GB (온라인 노드 합계)" />
        <Stat icon="activity" label="실행 중 작업" value={running?.length ?? 0} sub="모든 노드" />
      </div>

      {error && <div className="error">{error}</div>}
      {nodes && nodes.length === 0 && <div className="card empty">연결된 노드가 없습니다.</div>}
      <div className="grid">
        {nodes?.map((node) => {
          const gpuMetrics = node.metrics.gpus ?? [];
          const diskTotal = node.metrics.disk_total_gb;
          const diskUsed = diskTotal != null && node.metrics.disk_free_gb != null ? diskTotal - node.metrics.disk_free_gb : null;
          const busy = running?.filter((j) => j.node_id === node.id).length ?? 0;
          return (
            <section className="card node" key={node.id}>
              <header>
                <div className="node-icon">
                  <Icon name="server" size={18} />
                </div>
                <div className="node-title">
                  <h3>{node.name}</h3>
                  <div className="eyebrow">
                    {node.os || "?"} · {node.info.cpu_count ?? "?"} cores · {gb(node.info.memory_total_mb)} ram
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
                {busy > 0 && <span className="tag accent">작업 {busy}개 실행 중</span>}
              </div>

              <div className="resources">
                {node.info.gpus.length === 0 && (
                  <div className="resource">
                    <span>GPU</span>
                    <span className="muted">없음</span>
                  </div>
                )}
                {node.info.gpus.map((gpu) => {
                  const m = gpuMetrics.find((x) => x.index === gpu.index);
                  return (
                    <div className="resource" key={gpu.index}>
                      <span>GPU {gpu.index}</span>
                      <div>
                        <div className="resource-name">
                          {gpu.name} <span className="muted">{gb(gpu.memory_total_mb)}</span>
                        </div>
                        <Meter value={m?.memory_used_mb} max={m?.memory_total_mb} />
                      </div>
                    </div>
                  );
                })}
                <div className="resource">
                  <span>메모리</span>
                  <Meter value={node.metrics.memory_used_mb} max={node.metrics.memory_total_mb} />
                </div>
                <div className="resource">
                  <span>디스크</span>
                  <div>
                    <Meter value={diskUsed} max={diskTotal} />
                    {node.metrics.disk_free_gb != null && (
                      <div className="muted small">{node.metrics.disk_free_gb} GB 여유</div>
                    )}
                  </div>
                </div>
              </div>

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
        {isAdmin && nodes && (
          <button className="card add-card" onClick={() => setAdding(true)}>
            <Icon name="plus" size={20} />
            <strong>서버 추가</strong>
            <span className="muted small">Ubuntu · Windows · Jetson — 한 줄 명령으로 연결</span>
          </button>
        )}
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
            원격 서버가 이 주소({join.hub_url})로 접속할 수 있어야 합니다. 설치가 끝나면 노드 목록에 자동으로 나타납니다.
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
