import { useState } from "react";
import "./nodes.css";
import { type HubAddress, type JoinToken, api } from "./api";
import { CopyBlock, Icon, Modal, RefreshButton, gb, timeAgo, usePolling, useUpdatedAt } from "./ui";

const ROLES = [
  { id: "compute", label: "학습·오토라벨" },
  { id: "storage", label: "데이터 저장" },
  { id: "inference", label: "추론" },
];

// ---- what the Hub sends (/api/gpu-board) ----------------------------------------------------

interface Occupant {
  kind: "job" | "person";
  job_id?: string;
  label?: string;
  href?: string;
  epoch?: number;
  epochs?: number;
  eta_s?: number;
  who?: string;
  programs?: string;
  light?: boolean; // a person's light use beside which a job can still run
}

interface BoardGpu {
  index: number;
  name: string | null;
  memory_total_mb: number | null;
  memory_used_mb: number | null;
  utilization_pct: number | null;
  temperature_c: number | null;
  power_w: number | null;
  state: "free" | "job" | "busy" | "reserved" | "offline";
  reserved: boolean;
  occupant: Occupant | null;
}

export interface BoardNode {
  id: string;
  name: string;
  status: "online" | "offline" | "disabled";
  enabled: boolean;
  is_local: boolean;
  roles: string[];
  gpu_name: string | null;
  cpu_pct: number | null;
  memory_pct: number | null;
  disk: { mount: string; pct: number; free_gb: number } | null;
  last_seen_at: string | null;
  gpus: BoardGpu[];
}

interface Board {
  nodes: BoardNode[];
  counts: { free: number; job: number; busy: number; reserved: number; offline: number; waiting: number };
  waiting: { job_id: string; label: string; href: string; gpus: number; min_vram_mb: number; reason: string }[];
  next_free: { node: string; label: string; eta_s: number; gpus: number[] } | null;
}

export function duration(seconds: number): string {
  if (seconds < 90) return "1분 안에";
  if (seconds < 3600) return `약 ${Math.round(seconds / 60)}분 뒤`;
  const hours = seconds / 3600;
  return hours < 24 ? `약 ${hours.toFixed(hours < 10 ? 1 : 0)}시간 뒤` : `약 ${Math.round(hours / 24)}일 뒤`;
}

/** "13분 남음": short enough for a GPU tile. */
function left(seconds: number): string {
  if (seconds < 90) return "곧 끝";
  if (seconds < 3600) return `${Math.round(seconds / 60)}분 남음`;
  const hours = seconds / 3600;
  return hours < 24 ? `${hours.toFixed(hours < 10 ? 1 : 0)}시간 남음` : `${Math.round(hours / 24)}일 남음`;
}

/**
 * Every GPU of every server on one board: what holds it (a BADAMOA run, a person, nobody, kept for
 * people) and how full its memory is, the height of the fill. Above it, the jobs waiting for a GPU.
 */
export function Nodes({ isAdmin }: { isAdmin: boolean }) {
  const { data: board, error, refresh } = usePolling(() => api<Board>("/gpu-board"), 5000);
  const updated = useUpdatedAt(board);
  const [adding, setAdding] = useState(false);
  const [editing, setEditing] = useState<BoardNode | null>(null);
  const c = board?.counts;
  const total = c ? c.free + c.job + c.busy + c.reserved : 0;

  return (
    <>
      <div className="page-head">
        <div>
          <h1>노드</h1>
          <p className="muted">
            연결된 서버의 GPU를 한 판에 보여 줍니다. 칸이 차오른 높이는 GPU 메모리이고, 사람이 쓰는 GPU와 사람 전용으로 둔
            GPU에는 작업을 보내지 않습니다.
          </p>
        </div>
        <div className="head-actions">
          <RefreshButton onRefresh={refresh} updated={updated} />
          {isAdmin && (
            <button className="primary" onClick={() => setAdding(true)}>
              <Icon name="plus" />
              노드 추가
            </button>
          )}
        </div>
      </div>

      {c && (
        <div className="board-counts">
          <span className="board-count">
            빈 GPU <strong>{c.free}</strong> / {total}
          </span>
          <span className="board-count">
            <i className="swatch job" />
            BADAMOA {c.job}
          </span>
          <span className="board-count">
            <i className="swatch busy" />
            사람 {c.busy}
          </span>
          {c.reserved > 0 && (
            <span className="board-count">
              <i className="swatch reserved" />
              사람 전용 {c.reserved}
            </span>
          )}
          {c.offline > 0 && <span className="board-count">연결 끊김 {c.offline}</span>}
          <span className="board-count">대기 {c.waiting}</span>
        </div>
      )}

      {board && board.waiting.length > 0 && (
        <div className="board-queue">
          {board.waiting.slice(0, 3).map((w) => (
            <div key={w.job_id} className="queue-row">
              <Icon name="clock" size={14} />
              <span>
                대기 · <a href={w.href}>{w.label}</a> · GPU {w.gpus}개
                {w.min_vram_mb ? ` · ${gb(w.min_vram_mb)} 이상` : ""}
              </span>
              <span className="muted">{w.reason}</span>
            </div>
          ))}
          {board.waiting.length > 3 && <div className="muted small">외 {board.waiting.length - 3}개 · 작업 화면에서 확인</div>}
          {board.next_free && (
            <div className="muted small">
              가장 먼저 끝날 학습: {board.next_free.label} ({board.next_free.node} GPU {board.next_free.gpus.join(", ")}){" "}
              {duration(board.next_free.eta_s)}
            </div>
          )}
        </div>
      )}

      {error && <div className="error">{error}</div>}
      {board && board.nodes.length === 0 && <div className="card empty">연결된 노드가 없습니다.</div>}
      {board && board.nodes.length > 0 && (
        <section className="card board">
          {board.nodes.map((node) => (
            <ServerRow key={node.id} node={node} isAdmin={isAdmin} onEdit={() => setEditing(node)} />
          ))}
          {isAdmin && (
            <button className="board-add" onClick={() => setAdding(true)}>
              <Icon name="plus" size={14} />
              서버 추가 <span className="muted small">Ubuntu · Windows — 한 줄 명령으로 연결</span>
            </button>
          )}
        </section>
      )}
      {board && board.nodes.length > 0 && (
        <div className="board-legend muted small">
          <span>
            <i className="swatch job" /> BADAMOA 실행
          </span>
          <span>
            <i className="swatch busy" /> 사람이 쓰는 중 (<a href="#/settings/gpu">기준</a>)
          </span>
          <span>
            <i className="swatch free" /> 비어 있음
          </span>
          <span>
            <i className="swatch reserved" /> 사람 전용 (작업을 보내지 않음)
          </span>
          <span>칸이 차오른 높이 = GPU 메모리</span>
        </div>
      )}

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

function ServerRow({ node, isAdmin, onEdit }: { node: BoardNode; isAdmin: boolean; onEdit: () => void }) {
  const online = node.status === "online";
  const facts = [
    node.cpu_pct != null ? `CPU ${Math.round(node.cpu_pct)}%` : null,
    node.memory_pct != null ? `RAM ${node.memory_pct}%` : null,
  ].filter(Boolean);
  const diskWarn = node.disk && node.disk.pct >= 90;
  return (
    <div className={`board-row ${online ? "" : "away"}`}>
      <div className="board-server">
        <a className="board-name" href={`#/nodes/${node.id}`}>
          <i className={`dot ${node.status}`} />
          {node.name}
        </a>
        <div className="muted small">
          {node.gpus.length > 0 ? `${node.gpu_name ?? "GPU"} × ${node.gpus.length}` : "GPU 없음"}
          {node.is_local ? " · Hub" : ""}
        </div>
        {online ? (
          <div className="muted small">{facts.join(" · ")}</div>
        ) : (
          <div className="small bad-text">{node.status === "disabled" ? "작업 받지 않음" : `응답 없음 · ${timeAgo(node.last_seen_at)}`}</div>
        )}
        {online && !node.enabled && <div className="small warn-text">작업 받지 않음</div>}
        {diskWarn && (
          <div className="small bad-text">
            {node.disk!.mount} {Math.round(node.disk!.pct)}% · {Math.round(node.disk!.free_gb)} GB 남음
          </div>
        )}
        {isAdmin && (
          <button className="link small" onClick={onEdit}>
            설정
          </button>
        )}
      </div>
      <div className="board-gpus">
        {node.gpus.length === 0 && <div className="muted small board-nogpu">이 서버는 CPU 작업만 받습니다.</div>}
        {node.gpus.map((gpu) => (
          <GpuTile key={gpu.index} gpu={gpu} />
        ))}
      </div>
    </div>
  );
}

function GpuTile({ gpu }: { gpu: BoardGpu }) {
  const used = gpu.memory_used_mb ?? 0;
  const level = gpu.memory_total_mb ? Math.min(100, (used / gpu.memory_total_mb) * 100) : 0;
  const o = gpu.occupant;
  const kind = gpu.state === "job" ? "job" : gpu.state === "busy" ? "busy" : gpu.state === "reserved" ? "reserved" : gpu.state === "offline" ? "offline" : "free";
  const hot = (gpu.temperature_c ?? 0) >= 83;
  const detail = [
    gpu.utilization_pct != null ? `사용률 ${Math.round(gpu.utilization_pct)}%` : null,
    gpu.memory_total_mb ? `메모리 ${(used / 1024).toFixed(1)} / ${gb(gpu.memory_total_mb)}` : null,
    gpu.temperature_c != null ? `${gpu.temperature_c}°C` : null,
    gpu.power_w != null ? `${Math.round(gpu.power_w)} W` : null,
  ]
    .filter(Boolean)
    .join(" · ");
  return (
    <div className={`gpu-tile ${kind}`} title={detail}>
      {kind !== "offline" && <div className="gpu-level" style={{ height: `${level}%` }} />}
      <div className="gpu-body">
        <div className="gpu-head">
          <span>{gpu.index}</span>
          {gpu.utilization_pct != null && kind !== "free" && <span>{Math.round(gpu.utilization_pct)}%</span>}
          {hot && <span className="bad-text">{gpu.temperature_c}°C</span>}
        </div>
        {kind === "job" && o && (
          <>
            <a className="gpu-who" href={o.href}>
              {o.label}
            </a>
            <div className="gpu-what">
              {o.epochs ? `epoch ${o.epoch ?? 0}/${o.epochs}` : "실행 중"}
              {o.eta_s != null ? ` · ${left(o.eta_s)}` : ""}
            </div>
          </>
        )}
        {kind === "busy" && (
          <>
            <div className="gpu-who">{o?.who || "알 수 없는 사용자"}</div>
            <div className="gpu-what">{o?.programs || "다른 프로세스"}</div>
          </>
        )}
        {(kind === "free" || kind === "reserved") && (
          <>
            <div className="gpu-who quiet">{kind === "reserved" ? "사람 전용" : "비어 있음"}</div>
            <div className="gpu-what">
              {o?.light ? `${o.who || "누군가"} ${(used / 1024).toFixed(1)} GB 사용 중` : gb(gpu.memory_total_mb)}
            </div>
          </>
        )}
        {kind === "offline" && <div className="gpu-who quiet">연결 끊김</div>}
      </div>
    </div>
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

  const create = async (): Promise<void> => {
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
          <AddressNote where={join.address} onChange={create} />
          <CopyBlock label="Ubuntu / Linux" text={join.commands.linux} />
          <CopyBlock label="Windows (관리자 PowerShell)" text={join.commands.windows} />
          <p className="muted small">설치가 끝나면 노드 목록에 자동으로 나타납니다.</p>
          <div className="modal-actions">
            <button onClick={onClose}>닫기</button>
          </div>
        </>
      )}
    </Modal>
  );
}

const ADDRESS_SOURCE: Record<HubAddress["source"], string> = {
  detected: "이 PC의 네트워크 주소를 자동으로 넣었습니다 (브라우저는 localhost로 열려 있음)",
  browser: "지금 브라우저가 연 주소입니다",
  saved: "설정해 둔 Hub 주소입니다",
  launch: "Hub 실행 옵션(--public-url)의 주소입니다",
  "local-only": "다른 서버가 닿을 수 있는 주소가 없습니다",
};
const TYPED = "\u0000typed";

/** Which address the commands carry and why; change it when the new server cannot reach it. */
function AddressNote({ where, onChange }: { where: HubAddress; onChange: () => Promise<void> }) {
  const [editing, setEditing] = useState(false);
  const [picked, setPicked] = useState(where.url);
  const [typed, setTyped] = useState("");
  const [error, setError] = useState<string | null>(null);
  const others = where.candidates.filter((c) => c !== where.url);
  const save = async (url: string | null) => {
    try {
      setError(null);
      await api("/settings/hub-address", { method: "PUT", body: url === null ? {} : { url } });
      await onChange(); // new commands with the new address (the earlier ones simply expire)
      setEditing(false);
    } catch (e) {
      setError(e instanceof Error ? e.message : String(e));
    }
  };
  return (
    <div className={`address-note ${where.source === "local-only" ? "warn" : ""}`}>
      <div>
        Hub 주소 <code>{where.url}</code> <span className="muted small">— {ADDRESS_SOURCE[where.source]}</span>
        {!editing && (
          <button className="link small" onClick={() => setEditing(true)}>
            바꾸기
          </button>
        )}
      </div>
      {where.listens_locally_only ? (
        <div className="small">
          이 Hub는 이 PC 안에서만 접속을 받도록 실행되어 있습니다(--host 127.0.0.1). 다른 서버를 붙이려면 Hub를 --host 0.0.0.0으로 다시
          실행하세요.
        </div>
      ) : (
        <div className="muted small">
          추가할 서버가 이 주소로 Hub에 닿아야 합니다. 닿지 않으면 그 서버에서 보이는 주소(사내 DNS 이름, VPN 주소 등)로 바꾸세요.
        </div>
      )}
      {editing && (
        <div className="address-edit">
          <select value={picked} onChange={(e) => setPicked(e.target.value)} aria-label="Hub 주소">
            <option value={where.url}>{where.url} (지금)</option>
            {others.map((c) => (
              <option key={c} value={c}>
                {c}
              </option>
            ))}
            <option value={TYPED}>직접 입력…</option>
          </select>
          {picked === TYPED && (
            <input value={typed} onChange={(e) => setTyped(e.target.value)} placeholder="http://서버이름:8470" autoFocus />
          )}
          <div className="row-buttons">
            <button
              className="small primary"
              disabled={picked === TYPED ? !typed.trim() : picked === where.url && where.source === "saved"}
              onClick={() => void save(picked === TYPED ? typed : picked)}
            >
              이 주소로 다시 만들기
            </button>
            {where.source === "saved" && (
              <button className="small" onClick={() => void save(null)}>
                자동으로
              </button>
            )}
            <button className="small" onClick={() => setEditing(false)}>
              취소
            </button>
          </div>
          <div className="muted small">고른 주소는 저장되어 다음 노드 추가에도 쓰입니다.</div>
        </div>
      )}
      {error && <div className="error">{error}</div>}
    </div>
  );
}

function EditNode({ node, onClose }: { node: BoardNode; onClose: () => void }) {
  const [name, setName] = useState(node.name);
  const [roles, setRoles] = useState(node.roles);
  const [enabled, setEnabled] = useState(node.enabled);
  const [kept, setKept] = useState<number[]>(node.gpus.filter((g) => g.reserved).map((g) => g.index));
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
      {node.gpus.length > 0 && (
        <>
          <div className="field-label">사람 전용 GPU</div>
          <p className="muted small">고른 GPU에는 BADAMOA가 작업을 보내지 않습니다. 사람이 직접 쓰는 자리로 남겨 둘 때 씁니다.</p>
          <div className="checks">
            {node.gpus.map((g) => (
              <label key={g.index}>
                <input
                  type="checkbox"
                  checked={kept.includes(g.index)}
                  onChange={(e) => setKept(e.target.checked ? [...kept, g.index] : kept.filter((i) => i !== g.index))}
                />
                GPU {g.index} · {g.name ?? ""}
              </label>
            ))}
          </div>
        </>
      )}
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
          onClick={() => run(() => api(`/nodes/${node.id}`, { method: "PATCH", body: { name, roles, enabled, reserved_gpus: kept } }))}
        >
          저장
        </button>
      </div>
    </Modal>
  );
}
