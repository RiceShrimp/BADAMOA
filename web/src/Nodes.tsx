import { useEffect, useState } from "react";
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
  agent_version: string;
  roles: string[];
  gpu_name: string | null;
  cpu_pct: number | null;
  memory_pct: number | null;
  disk: { mount: string; pct: number; free_gb: number } | null;
  cpu?: { pct: number | null; cores: number | null; model: string | null };
  memory?: { used_gb: number; total_gb: number } | null;
  disks?: { mount: string; total_gb: number; free_gb: number; pct: number }[];
  last_seen_at: string | null;
  gpus: BoardGpu[];
}

interface Board {
  hub_version: string;
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
  const [updating, setUpdating] = useState<BoardNode | null>(null);
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
            <ServerRow
              key={node.id}
              node={node}
              hubVersion={board.hub_version}
              isAdmin={isAdmin}
              onEdit={() => setEditing(node)}
              onUpdate={() => setUpdating(node)}
              onChanged={() => void refresh()}
            />
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
          <span>
            <i className="swatch cpu" /> CPU · <i className="swatch memory" /> 메모리 · <i className="swatch disk" /> 디스크
          </span>
          <span>
            <i className="swatch full" /> 90% 넘음
          </span>
          <span>칸이 차오른 높이 = 쓰는 만큼(GPU는 메모리)</span>
        </div>
      )}

      {adding && <AddNode onClose={() => setAdding(false)} />}
      {updating && <UpdateNode node={updating} onClose={() => setUpdating(null)} />}
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

/** x.y.z later than another; anything else is not compared. */
function later(a: string, b: string): boolean {
  const pa = a.split(".").map(Number);
  const pb = b.split(".").map(Number);
  if ([...pa, ...pb].some(Number.isNaN)) return false;
  for (let i = 0; i < Math.max(pa.length, pb.length); i++) {
    if ((pa[i] ?? 0) !== (pb[i] ?? 0)) return (pa[i] ?? 0) > (pb[i] ?? 0);
  }
  return false;
}
const SELF_UPDATE_SINCE = "0.31.0"; // nodes from this version update themselves

function ServerRow({
  node,
  hubVersion,
  isAdmin,
  onEdit,
  onUpdate,
  onChanged,
}: {
  node: BoardNode;
  hubVersion: string;
  isAdmin: boolean;
  onEdit: () => void;
  onUpdate: () => void;
  onChanged: () => void;
}) {
  const behind = Boolean(node.agent_version) && later(hubVersion, node.agent_version);
  const updatesItself = behind && !later(SELF_UPDATE_SINCE, node.agent_version);
  const toggle = async () => {
    const stop = node.enabled;
    if (stop && !window.confirm(`${node.name}을(를) 비활성화할까요?\n새 작업을 보내지 않습니다. 실행 중인 작업은 끝까지 갑니다. 언제든 "다시 사용"으로 되돌릴 수 있습니다.`)) return;
    await api(`/nodes/${node.id}`, { method: "PATCH", body: { enabled: !stop } });
    onChanged();
  };
  const online = node.status === "online";
  const sys = systemTiles(node);
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
        {!online && (
          <div className="small bad-text">{node.status === "disabled" ? "작업 받지 않음" : `응답 없음 · ${timeAgo(node.last_seen_at)}`}</div>
        )}
        {online && !node.enabled && <div className="small warn-text">작업 받지 않음</div>}
        {behind && (
          <div className="small warn-text">
            노드 {node.agent_version} · Hub {hubVersion}
            {updatesItself ? (
              " — 작업이 없을 때 스스로 업데이트"
            ) : isAdmin ? (
              <>
                {" — "}
                <button className="link small" onClick={onUpdate}>
                  업데이트 방법
                </button>
              </>
            ) : (
              " — 업데이트 필요"
            )}
          </div>
        )}
        {isAdmin && (
          <div className="row-links">
            <button className="link small" onClick={onEdit}>
              설정
            </button>
            <button className="link small" onClick={() => void toggle()}>
              {node.enabled ? "비활성화" : "다시 사용"}
            </button>
          </div>
        )}
      </div>
      <div className="board-gpus">
        {sys.length > 0 && (
          <div className="board-sys">
            {sys.map((tile) => (
              <SysTile key={tile.kind} tile={tile} />
            ))}
          </div>
        )}
        <div className="gpu-grid">
          {node.gpus.length === 0 && <div className="muted small board-nogpu">GPU 없음 · CPU 작업만 받습니다</div>}
          {node.gpus.map((gpu) => (
            <GpuTile key={gpu.index} gpu={gpu} />
          ))}
        </div>
      </div>
    </div>
  );
}

/** "931 GB" / "2.7 TB" */
function capacity(gbValue: number): string {
  return gbValue >= 1000 ? `${(gbValue / 1024).toFixed(1)} TB` : `${Math.round(gbValue)} GB`;
}

const FULL_PCT = 90; // from here a processor, memory or disk tile turns red

interface SysTileInfo {
  kind: "cpu" | "memory" | "disk";
  label: string;
  value: string;
  sub: string;
  pct: number | null;
  title?: string;
}

/** Processor, memory and disks as tiles beside the GPUs, each filling from the bottom as it is used. */
function systemTiles(node: BoardNode): SysTileInfo[] {
  const tiles: SysTileInfo[] = [];
  const cpu = node.cpu;
  if (cpu && (cpu.pct != null || cpu.cores)) {
    tiles.push({
      kind: "cpu",
      label: "CPU",
      value: cpu.pct != null ? `${Math.round(cpu.pct)}%` : "–",
      sub: cpu.cores ? `${cpu.cores}코어` : "",
      pct: cpu.pct,
      title: cpu.model ?? undefined,
    });
  }
  if (node.memory) {
    const { used_gb, total_gb } = node.memory;
    const pct = total_gb ? (used_gb / total_gb) * 100 : null;
    tiles.push({
      kind: "memory",
      label: "메모리",
      value: `${Math.round(used_gb)} / ${Math.round(total_gb)} GB`,
      sub: pct != null ? `${Math.round(pct)}%` : "",
      pct,
    });
  }
  const disks = node.disks ?? [];
  if (disks.length > 0) {
    const total = disks.reduce((n, d) => n + d.total_gb, 0);
    const free = disks.reduce((n, d) => n + d.free_gb, 0);
    const fullest = disks.reduce((a, d) => (d.pct > a.pct ? d : a));
    const each = disks.map((d) => `${d.mount} ${capacity(d.free_gb)} 남음 / ${capacity(d.total_gb)} (${Math.round(d.pct)}%)`).join("\n");
    const full = fullest.pct >= FULL_PCT;
    tiles.push({
      kind: "disk",
      // A disk nearly full is named on the tile; otherwise the disks are added up.
      label: full ? `${fullest.mount} ${Math.round(fullest.pct)}%` : disks.length > 1 ? `디스크 ${disks.length}개` : "디스크",
      value: `${capacity(full ? fullest.free_gb : free)} 남음`,
      sub: full ? (disks.length > 1 ? `디스크 ${disks.length}개` : `/ ${capacity(fullest.total_gb)}`) : `/ ${capacity(total)}`,
      pct: full ? fullest.pct : total ? ((total - free) / total) * 100 : null,
      title: each,
    });
  }
  return tiles;
}

function SysTile({ tile }: { tile: SysTileInfo }) {
  const level = Math.min(100, Math.max(0, tile.pct ?? 0));
  const full = tile.pct != null && tile.pct >= FULL_PCT;
  return (
    <div className={`sys-tile ${tile.kind}${full ? " full" : ""}`} title={tile.title}>
      <div className="sys-level" style={{ height: `${level}%` }} />
      <div className="gpu-body">
        <div className="gpu-head">
          <span>{tile.label}</span>
        </div>
        <div className="sys-value">{tile.value}</div>
        {tile.sub && <div className="gpu-what">{tile.sub}</div>}
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
          {!join.node_package.ready && (
            <div className="error">
              노드 설치 파일(badamoa {join.node_package.version})이 Hub에 아직 없어 설치가 중간에 멈춥니다. Hub를 소스에서 실행하면 시작할 때
              자동으로 만들어지니 잠시 뒤 다시 여세요. 설치본이라면 <code>badamoa hub publish-wheel &lt;wheel 파일&gt;</code>로 올리세요.
            </div>
          )}
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

/** For a node from before self-update: the one-time update command (the registration stays). */
function UpdateNode({ node, onClose }: { node: BoardNode; onClose: () => void }) {
  const [info, setInfo] = useState<{ version: string; commands: { linux: string; windows: string } } | null>(null);
  const [error, setError] = useState<string | null>(null);
  useEffect(() => {
    api<{ version: string; commands: { linux: string; windows: string } }>("/node-update-commands")
      .then(setInfo)
      .catch((e) => setError(e instanceof Error ? e.message : String(e)));
  }, []);
  return (
    <Modal title={`${node.name} 업데이트`} onClose={onClose}>
      <p>
        이 노드의 프로그램({node.agent_version})은 스스로 업데이트하는 기능이 생기기 전 버전입니다. 그 서버에서 아래 명령을 <strong>한 번만</strong>{" "}
        실행하세요. 등록은 그대로라 같은 노드로 이어지고, 다음부터는 Hub를 올리면 노드가 스스로 따라 올라갑니다.
      </p>
      {error && <div className="error">{error}</div>}
      {info && (
        <>
          <CopyBlock label="Ubuntu / Linux" text={info.commands.linux} />
          <CopyBlock label="Windows (관리자 PowerShell)" text={info.commands.windows} />
          <p className="muted small">설치 명령(노드 추가)을 다시 실행해도 됩니다. 이미 등록된 서버는 새로 등록하지 않고 업데이트만 합니다({info.version}부터).</p>
        </>
      )}
      <div className="modal-actions">
        <button onClick={onClose}>닫기</button>
      </div>
    </Modal>
  );
}

interface Deleted {
  removed_on_server: boolean;
  reason: string;
  commands: { linux: string; windows: string };
}

/** Deleting a node: what it means, whether BADAMOA leaves the server too, and the command when it could not. */
function DeleteNode({ node, onBack, onDone }: { node: BoardNode; onBack: () => void; onDone: () => void }) {
  const [cleanup, setCleanup] = useState(true);
  const [busy, setBusy] = useState(false);
  const [error, setError] = useState<string | null>(null);
  const [left, setLeft] = useState<Deleted | null>(null); // deleted, but BADAMOA is still on the server
  const online = node.status !== "offline";

  const remove = async () => {
    setBusy(true);
    try {
      setError(null);
      const done = await api<Deleted>(`/nodes/${node.id}?cleanup=${cleanup}`, { method: "DELETE" });
      if (cleanup && !done.removed_on_server) setLeft(done);
      else onDone();
    } catch (e) {
      setError(e instanceof Error ? e.message : String(e));
    } finally {
      setBusy(false);
    }
  };

  if (left) {
    return (
      <Modal title={`${node.name} 삭제됨`} onClose={onDone}>
        <p>Hub에서 노드를 지웠습니다. {left.reason} 그 서버에서 아래 명령을 실행하면 BADAMOA가 지워집니다(데이터 폴더는 그대로).</p>
        <CopyBlock label="Ubuntu / Linux" text={left.commands.linux} />
        <CopyBlock label="Windows (관리자 PowerShell)" text={left.commands.windows} />
        <p className="muted small">그냥 두어도 해는 없습니다. 지워진 노드는 5분에 한 번만 Hub를 확인하고 작업을 받지 않습니다.</p>
        <div className="modal-actions">
          <button onClick={onDone}>닫기</button>
        </div>
      </Modal>
    );
  }
  return (
    <Modal title={`${node.name} 삭제`} onClose={onBack}>
      <ul className="consequences">
        <li>Hub의 노드 목록에서 사라지고 이 서버로 작업을 보내지 않습니다.</li>
        <li>이 서버에서 실행 중인 작업은 실패로 끝납니다.</li>
        <li>이 서버의 이미지로 등록한 데이터셋은 학습 · 내려받기를 할 수 없게 됩니다(등록 기록은 남음).</li>
      </ul>
      <label className="check">
        <input type="checkbox" checked={cleanup} onChange={(e) => setCleanup(e.target.checked)} />
        서버에서도 BADAMOA 지우기 <span className="muted small">— 서비스 등록 · 프로그램 · 설정. 데이터 폴더는 그대로</span>
      </label>
      {cleanup && !online && (
        <p className="muted small">노드가 꺼져 있어 지금은 서버를 정리할 수 없습니다. 삭제한 뒤 그 서버에서 실행할 명령을 알려 드립니다.</p>
      )}
      <p className="muted small">잠시 쉬게 하려는 것이면 삭제 대신 비활성화하세요.</p>
      {error && <div className="error">{error}</div>}
      <div className="modal-actions">
        <button onClick={onBack} disabled={busy}>
          취소
        </button>
        <button className="danger" onClick={() => void remove()} disabled={busy}>
          {busy ? "지우는 중…" : "삭제"}
        </button>
      </div>
    </Modal>
  );
}

function EditNode({ node, onClose }: { node: BoardNode; onClose: () => void }) {
  const [name, setName] = useState(node.name);
  const [roles, setRoles] = useState(node.roles);
  const [enabled, setEnabled] = useState(node.enabled);
  const [kept, setKept] = useState<number[]>(node.gpus.filter((g) => g.reserved).map((g) => g.index));
  const [error, setError] = useState<string | null>(null);
  const [deleting, setDeleting] = useState(false);

  const run = async (action: () => Promise<unknown>) => {
    try {
      await action();
      onClose();
    } catch (e) {
      setError(e instanceof Error ? e.message : String(e));
    }
  };

  return (
    deleting ? (
      <DeleteNode node={node} onBack={() => setDeleting(false)} onDone={onClose} />
    ) : (
    <Modal title={`${node.name} 설정`} onClose={onClose}>
      <label>
        이름
        <input value={name} onChange={(e) => setName(e.target.value)} />
      </label>
      <div className="field-label">역할</div>
      <RolePicker roles={roles} onChange={setRoles} />
      <div className="checks">
        <label>
          <input type="checkbox" checked={!enabled} onChange={(e) => setEnabled(!e.target.checked)} />
          비활성화 <span className="muted small">— 새 작업을 보내지 않습니다. 실행 중인 작업은 끝까지 갑니다</span>
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
      {node.is_local && <p className="muted small">Hub가 있는 이 PC의 노드는 삭제할 수 없습니다. 쓰지 않으려면 비활성화하세요.</p>}
      {error && <div className="error">{error}</div>}
      <div className="modal-actions">
        {!node.is_local && (
          <button className="danger" onClick={() => setDeleting(true)}>
            노드 삭제
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
    )
  );
}
