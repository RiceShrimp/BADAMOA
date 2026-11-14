import { useState } from "react";
import { Sparkline } from "./Chart";
import { type GpuState, type Job, type JoinToken, type MetricHistory, type NodeInfo, api } from "./api";
import { Badge, CopyBlock, Icon, Meter, Modal, STATUS_LABEL, Stat, gb, timeAgo, usePolling } from "./ui";

const ROLES = [
  { id: "compute", label: "학습·오토라벨" },
  { id: "storage", label: "데이터 저장" },
  { id: "inference", label: "추론" },
];

function occupantLabel(state: GpuState | undefined, jobs: Job[] | null): string {
  if (!state || state.state === "free") return "비어 있음";
  if (state.state === "job") {
    const job = jobs?.find((j) => j.id === state.job_id);
    return job ? `작업: ${job.title}` : "BADAMOA 작업";
  }
  return state.users.length ? state.users.join(", ") : "다른 프로세스";
}

export function Nodes({ isAdmin }: { isAdmin: boolean }) {
  const { data: nodes, error, refresh } = usePolling(() => api<NodeInfo[]>("/nodes"), 5000);
  const { data: running } = usePolling(() => api<Job[]>("/jobs?status=running&limit=100"), 5000);
  const [adding, setAdding] = useState(false);
  const [editing, setEditing] = useState<NodeInfo | null>(null);

  const online = nodes?.filter((n) => n.status === "online") ?? [];
  const states = online.flatMap((n) => n.gpu_states);
  const inUse = states.filter((s) => s.state !== "free").length;
  const utils = online.flatMap((n) => (n.metrics.gpus ?? []).map((g) => g.utilization_pct ?? 0));
  const avgUtil = utils.length ? Math.round(utils.reduce((a, b) => a + b, 0) / utils.length) : null;
  const workers = [...new Set(online.flatMap((n) => (n.metrics.users ?? []).filter((u) => u.active).map((u) => u.user)))];

  return (
    <>
      <div className="page-head">
        <div>
          <h1>노드</h1>
          <p className="muted">
            Hub에 연결된 서버입니다. 다른 사람이 쓰고 있는 GPU에는 작업을 배정하지 않습니다. 이름을 누르면 자원 이력을 볼 수
            있습니다.
          </p>
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
          label="온라인 서버"
          value={
            <>
              {online.length}
              <small> / {nodes?.length ?? 0}</small>
            </>
          }
          sub={nodes && online.length === nodes.length ? "모두 정상" : "오프라인 서버 있음"}
        />
        <Stat
          icon="chip"
          label="사용 중인 GPU"
          value={
            <>
              {inUse}
              <small> / {states.length}</small>
            </>
          }
          sub={`${states.length - inUse}개 여유`}
        />
        <Stat
          icon="activity"
          label="평균 GPU 사용률"
          value={avgUtil == null ? "–" : <>{avgUtil}<small>%</small></>}
          sub="온라인 서버 전체 GPU 기준"
        />
        <Stat
          icon="bot"
          label="작업 중인 사용자"
          value={<>{workers.length}<small>명</small></>}
          sub={workers.length ? workers.slice(0, 4).join(", ") + (workers.length > 4 ? " 외" : "") : "없음"}
        />
      </div>

      {error && <div className="error">{error}</div>}
      {nodes && nodes.length === 0 && <div className="card empty">연결된 노드가 없습니다.</div>}
      <div className="grid">
        {nodes?.map((node) => (
          <NodeCard key={node.id} node={node} running={running} isAdmin={isAdmin} onEdit={() => setEditing(node)} />
        ))}
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

function NodeCard({
  node,
  running,
  isAdmin,
  onEdit,
}: {
  node: NodeInfo;
  running: Job[] | null;
  isAdmin: boolean;
  onEdit: () => void;
}) {
  const { data: trend } = usePolling(() => api<MetricHistory>(`/nodes/${node.id}/metrics?range=1h`), 60000, [node.id]);
  const m = node.metrics;
  const live = m.gpus ?? [];
  const disks = [...(m.disks ?? [])].sort((a, b) => b.pct - a.pct);
  const busy = running?.filter((j) => j.node_id === node.id) ?? [];
  const workers = (m.users ?? []).filter((u) => u.active);

  const runInfo = async () => {
    await api("/jobs", { body: { type: "system.info", node_id: node.id } });
    window.location.hash = "#/jobs";
  };

  return (
    <section className="card node">
      <header>
        <div className="node-icon">
          <Icon name="server" size={18} />
        </div>
        <div className="node-title">
          <h3>
            <a className="node-link" href={`#/nodes/${node.id}`}>
              {node.name}
            </a>
          </h3>
          <div className="muted small">
            {node.os || "?"} · {node.info.cpu_count ?? "?"}코어 · {gb(node.info.memory_total_mb)} · GPU {node.info.gpus.length}개
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
        {!node.enabled && <span className="tag">작업 받지 않음</span>}
      </div>

      <div className="resources">
        <div className="resource">
          <span>
            CPU
            {(m.cpu_temp_c != null || m.cpu_power_w != null) && (
              <small>
                {[m.cpu_temp_c != null ? `${m.cpu_temp_c}°C` : null, m.cpu_power_w != null ? `${m.cpu_power_w}W` : null]
                  .filter(Boolean)
                  .join(" · ")}
              </small>
            )}
          </span>
          <div>
            <Meter value={m.cpu_pct} max={100} />
            {m.load && <div className="muted small">load {m.load[0]}</div>}
          </div>
        </div>
        <div className="resource">
          <span>RAM</span>
          <div>
            <Meter value={m.memory_used_mb} max={m.memory_total_mb} />
            <div className="muted small">
              {((m.memory_used_mb ?? 0) / 1024).toFixed(1)} / {gb(m.memory_total_mb)}
            </div>
          </div>
        </div>
      </div>

      {node.info.gpus.length > 0 && (
        <div className="resources gpu-block">
          <div className="block-head">
            <span>{node.info.gpus[0].name}</span>
            <span className="muted small">사용률 · 메모리</span>
          </div>
          {node.info.gpus.map((gpu) => {
            const g = live.find((x) => x.index === gpu.index);
            const state = node.gpu_states.find((s) => s.index === gpu.index);
            return (
              <div className="resource" key={gpu.index}>
                <span>
                  GPU {gpu.index}
                  <small>
                    {[g?.temperature_c != null ? `${g.temperature_c}°C` : null, g?.power_w != null ? `${g.power_w}W` : null]
                      .filter(Boolean)
                      .join(" · ")}
                  </small>
                </span>
                <div>
                  <Meter value={g?.utilization_pct} max={100} />
                  <Meter value={g?.memory_used_mb} max={g?.memory_total_mb} />
                  <div className={`occupant ${state?.state ?? "free"}`}>
                    <i />
                    {occupantLabel(state, running)}
                    <span className="muted small">
                      {g ? `${((g.memory_used_mb ?? 0) / 1024).toFixed(1)} / ${gb(g.memory_total_mb)}` : ""}
                    </span>
                  </div>
                </div>
              </div>
            );
          })}
        </div>
      )}

      {disks.length > 0 && (
        <div className="resources gpu-block">
          <div className="block-head">
            <span>디스크</span>
            <span className="muted small">남은 용량 · 전체</span>
          </div>
          {disks.slice(0, 3).map((d) => (
            <div className="resource" key={d.mount}>
              <span className="mono small-key">{d.mount}</span>
              <div className="disk-row">
                <Meter value={d.pct} max={100} />
                <span className="muted small">
                  <strong>{Math.round(d.free_gb)} GB</strong> / {Math.round(d.total_gb)} GB
                </span>
              </div>
            </div>
          ))}
          {disks.length > 3 && <div className="muted small">외 {disks.length - 3}개 (상세 화면에서 확인)</div>}
        </div>
      )}

      <div className="trend">
        <div>
          <div className="small">
            {busy.length > 0 ? <strong>작업 {busy.length}개 실행 중</strong> : <span className="muted">작업 중 없음</span>}
          </div>
          <div className="muted small">{workers.length ? `사용자 ${workers.map((u) => u.user).join(", ")}` : "접속 사용자 작업 없음"}</div>
        </div>
        {trend && trend.timestamps.length > 1 && (
          <div className="trend-plot">
            <Sparkline lines={[trend.series.cpu?.avg ?? [], trend.series.gpu_util?.avg ?? []]} />
            <div className="trend-legend">
              <span>
                <i style={{ background: "var(--series-1)" }} />
                CPU
              </span>
              <span>
                <i style={{ background: "var(--series-2)" }} />
                GPU 평균
              </span>
              <span className="muted">최근 1시간</span>
            </div>
          </div>
        )}
      </div>

      <footer>
        <span className="muted small">마지막 응답 {timeAgo(node.last_seen_at)}</span>
        <span className="actions">
          <a className="btn" href={`#/nodes/${node.id}`}>
            자원 이력
          </a>
          <button onClick={runInfo} disabled={node.status !== "online"}>
            시스템 점검
          </button>
          {isAdmin && <button onClick={onEdit}>설정</button>}
        </span>
      </footer>
    </section>
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
