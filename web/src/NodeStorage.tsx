import { useState } from "react";
import { api } from "./api";
import { bytesLabel } from "./catalogTypes";
import "./storage.css";
import { DockerMoveDialog } from "./DiskAlerts";
import { Badge, timeAgo, usePolling } from "./ui";

/** One thing BADAMOA keeps on a server: an environment, an image, a weights file, a folder. */
interface StorageItem {
  id: string;
  image?: string; // Docker images go by name
  bytes: number;
  last_used: string | null;
  label?: string;
  users?: string[]; // model types it serves (environments, images)
  environments?: string[]; // registered Docker environments with this image
  pulled?: boolean; // BADAMOA pulled the image (otherwise removing it only forgets it)
  shared?: number; // bytes in files uv's cache holds too
  model_id?: string;
  run_id?: string;
  job_status?: string;
  gone?: boolean; // what it belonged to is no longer on the Hub
  in_use: string; // the job that may use it ("" when none)
  path?: string; // still where the node kept it before its work folder (its home): its own path
  mount?: string; // that path's disk
}

interface StorageKind {
  kind: Kind;
  path: string;
  removable: boolean;
  bytes: number;
  free: number | null;
  items?: StorageItem[];
  count?: number;
  files?: number | null;
  shared?: number | null;
  in_use?: string;
  mount?: string; // the disk (mount point) the folder is on
  auto?: boolean; // the node picked the folder (the disk with the most room), not an admin
  work?: boolean; // in the node's work folder (nodes from 0.74.0)
  desktop?: boolean; // Docker images: inside Docker Desktop's virtual disk
}

interface StorageInfo {
  node_id: string;
  can_clean: boolean;
  need: string;
  cleaning: { job_id: string; kind: string }[];
  last_clean: {
    kind: string;
    ok: boolean;
    removed: number | null;
    bytes: number | null;
    error: string;
    at: string;
  } | null;
  measured_at: string | null;
  took_s?: number;
  stale?: boolean;
  home?: string;
  home_mount?: string;
  work?: string; // the node's work folder (nodes from 0.74.0)
  work_mount?: string;
  work_free?: number | null;
  free?: number | null;
  bytes?: number;
  kinds: StorageKind[];
}

type Kind = "images" | "envs" | "containers" | "models" | "templates" | "template_data" | "runs" | "uv";

const KIND: Record<Kind, { name: string; about: string; empty: string }> = {
  images: {
    name: "받은 이미지",
    about: "다른 서버에서 받아 둔 버전의 이미지. 아래 '받아 둔 버전'에서 버전별로 지웁니다",
    empty: "",
  },
  envs: {
    name: "가상환경",
    about: "모델 종류마다 설치한 Python 환경. 지우면 다음 학습 때 다시 설치합니다",
    empty: "모두 지울까요? 다음 학습 때 다시 설치합니다.",
  },
  containers: {
    name: "Docker 이미지",
    about: "학습 환경으로 쓰는 이미지. MOA가 받은 것만 지우고, 전부터 있던 이미지는 목록에서만 뺍니다",
    empty: "모두 지울까요? 필요할 때 다시 받습니다.",
  },
  models: {
    name: "모델 가중치",
    about: "평가 · 내보내기 · 이어 학습에 받아 둔 가중치. 지우면 필요할 때 다시 받습니다",
    empty: "모두 지울까요? 필요할 때 다시 받습니다.",
  },
  templates: {
    name: "모델 종류 코드",
    about: "Hub에서 받은 모델 종류의 코드. 지우면 다음 작업 때 다시 받습니다",
    empty: "모두 지울까요? 다음 작업 때 다시 받습니다.",
  },
  template_data: {
    name: "모델 종류 캐시",
    about: "모델 종류의 코드가 남겨 둔 것(대개 사전 학습 가중치). 지우면 다음 학습 때 다시 받습니다",
    empty: "모두 지울까요? 다음 학습 때 다시 받습니다.",
  },
  runs: {
    name: "작업 폴더",
    about: "멈췄거나 실패한 작업이 남긴 폴더(이어 학습용). 잘 끝난 작업은 남기지 않습니다",
    empty: "모두 지울까요? 이어 학습은 Hub에 둔 체크포인트에서 합니다.",
  },
  uv: {
    name: "uv 캐시",
    about: "가상환경을 만들 때 받은 패키지. 이 서버의 다른 uv 작업과 함께 쓰는 캐시라 정리는 쓰지 않는 것만 지웁니다",
    empty: "에서 쓰지 않는 것을 정리할까요? 가상환경과 다른 프로젝트는 그대로입니다.",
  },
};

const STATUS: Record<string, string> = {
  running: "실행 중",
  queued: "대기",
  succeeded: "끝남",
  failed: "실패",
  cancelled: "멈춤",
};

function itemTitle(kind: Kind, item: StorageItem) {
  const name = item.label || item.image || item.id;
  if (kind === "models" && item.model_id && !item.gone) return <a href={`#/models/${item.model_id}`}>{name}</a>;
  if (kind === "runs" && item.run_id) return <a href={`#/training/${item.run_id}`}>{name}</a>;
  return <span className={item.gone ? "muted" : ""}>{name}</span>;
}

function itemNote(kind: Kind, item: StorageItem): string {
  const parts: string[] = [];
  if (kind === "envs" || kind === "containers") {
    const users = [...(item.users ?? []), ...(item.environments ?? []).map((e) => `환경 ${e}`)];
    parts.push(users.length ? users.join(", ") : "쓰는 모델 종류 없음");
  }
  if (kind === "containers") parts.push(item.pulled ? "MOA가 받음" : "전부터 있던 이미지");
  if (kind === "templates" || kind === "template_data") parts.push(item.id.length > 24 ? item.id.slice(0, 12) : item.id);
  if (kind === "runs" && item.job_status) parts.push(STATUS[item.job_status] ?? item.job_status);
  parts.push(item.last_used ? `마지막 사용 ${timeAgo(item.last_used)}` : "쓴 적 없음");
  if (item.path) parts.push(`예전 위치${item.mount ? ` (디스크 ${item.mount})` : ""}`);
  return parts.join(" · ");
}

/** Who decided where a kind is kept, and the disk it is on with its room. */
function placeNote(k: StorageKind): string {
  const work = k.auto ? "작업 폴더: 노드가 고른 곳(남은 공간이 가장 많은 디스크, 노드 설정에서 바꿈)" : "작업 폴더: 관리자가 정한 곳 (노드 설정)";
  const who: Record<Kind, string> = {
    images: k.auto ? "노드가 고른 곳: 남은 공간이 가장 많은 디스크 (노드 설정에서 바꿈)" : "관리자가 정한 곳 (노드 설정)",
    envs: k.work ? work : "MOA 폴더 안",
    containers: k.desktop ? "Docker Desktop의 가상 디스크 안 (Docker Desktop 설정 → Resources)" : "Docker가 정한 곳 (Docker의 data-root)",
    models: k.work ? work : "MOA 폴더 안",
    templates: k.work ? work : "MOA 폴더 안",
    template_data: k.work ? work : "MOA 폴더 안",
    runs: k.work ? work : "MOA 폴더 안",
    uv: k.work ? "작업 폴더 안 (uv의 기본 캐시가 다른 디스크라 가상환경과 같은 디스크에 둠)" : "uv가 정한 곳 (이 계정의 uv 캐시)",
  };
  const parts = [who[k.kind]];
  if (k.mount) parts.push(`디스크 ${k.mount}`);
  if (k.free != null) parts.push(`남은 ${bytesLabel(k.free)}`);
  return parts.join(" · ");
}

/** An item's own folder or file under its kind's folder (Docker images go by name instead). */
function itemPath(k: StorageKind, item: StorageItem): string {
  if (item.path) return item.path; // still in the folder the node used before
  if (k.kind === "containers" || !k.path) return "";
  const sep = k.path.includes("\\") ? "\\" : "/";
  return `${k.path.replace(/[\\/]$/, "")}${sep}${item.id}`;
}

/** Why "비우기" is off, said on hover: what uses the kind now, or what has to happen first. */
function whyNotEmpty(k: StorageKind, notUsed: number, canClean: boolean, cleaning: boolean): string | undefined {
  if (!canClean) return "이 서버의 MOA가 오래되어 비울 수 없습니다. 노드를 업데이트하세요";
  if (cleaning) return "지우는 중입니다. 끝난 뒤에 다시 하세요";
  if (k.in_use || (k.items && k.items.length > 0 && notUsed === 0)) {
    const what = k.kind === "containers" ? "Docker 이미지" : KIND[k.kind].name;
    return `작업이 쓰는 중인 ${what}라 지금은 비울 수 없습니다. 쓰는 작업이 끝나면 비울 수 있습니다`;
  }
  if (k.items && notUsed < k.items.length) return "작업이 쓰는 것은 남기고 나머지만 비웁니다";
  return undefined;
}

/** What BADAMOA keeps on a server, by kind, with sizes against the disk's free space; admins remove items. */
export function NodeStorage({ nodeId, isAdmin }: { nodeId: string; isAdmin: boolean }) {
  const { data, refresh } = usePolling(() => api<StorageInfo>(`/nodes/${nodeId}/storage`), 15000, [nodeId]);
  const [open, setOpen] = useState<Kind | null>(null);
  const [dockerMove, setDockerMove] = useState(false);
  const [busy, setBusy] = useState(false);
  const [error, setError] = useState<string | null>(null);
  if (!data) return null;

  const post = async (body: { kind: string; items?: string[]; all?: boolean }) => {
    setBusy(true);
    setError(null);
    try {
      await api(`/nodes/${nodeId}/storage/clean`, { body });
      await refresh();
    } catch (err) {
      setError(err instanceof Error ? err.message : String(err));
    } finally {
      setBusy(false);
    }
  };
  const removeOne = (k: StorageKind, item: StorageItem) => {
    const name = item.label || item.image || item.id;
    if (!window.confirm(`${KIND[k.kind].name} '${name}'(${bytesLabel(item.bytes)})을(를) 이 서버에서 지울까요?`)) return;
    void post({ kind: k.kind, items: [k.kind === "containers" ? (item.image ?? item.id) : item.id] });
  };
  const empty = (k: StorageKind) => {
    const what = k.items ? `${k.items.length}개 · ${bytesLabel(k.bytes)}` : bytesLabel(k.bytes);
    const ask =
      k.kind === "uv"
        ? `${KIND.uv.name}(${what})${KIND.uv.empty}`
        : `${KIND[k.kind].name}(${what})을(를) ${KIND[k.kind].empty} 작업이 쓰는 것은 남깁니다.`;
    if (!window.confirm(ask)) return;
    void post({ kind: k.kind, all: true });
  };

  const cleaning = data.cleaning.length > 0;
  const measuring = data.cleaning.some((c) => !c.kind);
  const total = data.bytes ?? 0;
  const free = data.free ?? null;
  const whole = total + (free ?? 0);
  const last = data.last_clean;
  // kinds with nothing in them go on one line at the bottom: the list shows what takes room
  const shownKinds = data.kinds.filter((k) => k.bytes > 0 || (k.items?.length ?? 0) > 0);
  const emptyKinds = data.kinds.filter((k) => !shownKinds.includes(k));

  return (
    <section className="card storage">
      <div className="card-head">
        <h3>저장 공간</h3>
        <span className="muted small">
          MOA가 이 서버에 둔 것
          {data.measured_at
            ? ` · 합계 ${bytesLabel(total)}${free != null ? ` · 남은 공간 ${bytesLabel(free)}` : ""} · ${timeAgo(data.measured_at)} 잼`
            : ""}
        </span>
        <span className="spacer" />
        {data.can_clean && (
          <button
            type="button"
            className="small"
            disabled={busy || cleaning}
            title="이 서버가 지금 다시 재서 알려 줍니다 (30분마다, 그리고 작업이 끝날 때마다 스스로 잽니다)"
            onClick={() => void post({ kind: "" })}
          >
            {measuring ? "재는 중…" : "다시 재기"}
          </button>
        )}
      </div>

      {data.home && (
        <p className="muted small storage-home">
          MOA 폴더 <code className="mono">{data.home}</code>
          {data.home_mount ? ` · 디스크 ${data.home_mount}` : ""}
          {data.free != null ? ` · 남은 ${bytesLabel(data.free)}` : ""}
          {data.work && data.work !== data.home && (
            <>
              <br />
              작업 폴더 <code className="mono">{data.work}</code>
              {data.work_mount ? ` · 디스크 ${data.work_mount}` : ""}
              {data.work_free != null ? ` · 남은 ${bytesLabel(data.work_free)}` : ""}
            </>
          )}
        </p>
      )}
      {!data.measured_at ? (
        <p className="muted small storage-note">
          {data.can_clean
            ? "아직 알려 온 것이 없습니다. 서버가 켜져 있으면 곧 잽니다."
            : `이 서버의 MOA는 저장한 것을 알려 주지 않습니다. ${data.need} 이상으로 업데이트하면 보입니다.`}
        </p>
      ) : (
        <>
          {whole > 0 && (
            <div className="storage-bar" title={free != null ? `MOA ${bytesLabel(total)} · 남은 공간 ${bytesLabel(free)}` : ""}>
              {data.kinds
                .filter((k) => k.bytes > 0)
                .map((k) => (
                  <i
                    key={k.kind}
                    className={`kind-${k.kind}`}
                    style={{ width: `${(k.bytes / whole) * 100}%` }}
                    title={`${KIND[k.kind].name} ${bytesLabel(k.bytes)}`}
                  />
                ))}
            </div>
          )}
          <ul className="storage-kinds">
            {shownKinds.map((k) => {
              const items = k.items ?? [];
              const expandable = items.length > 0;
              const shown = open === k.kind;
              const free_ = k.items ? items.filter((i) => !i.in_use) : [];
              const canEmpty =
                isAdmin && data.can_clean && k.removable && k.bytes > 0 && (k.items ? free_.length > 0 : !k.in_use);
              return (
                <li key={k.kind} className={shown ? "open" : ""}>
                  <div className="storage-kind">
                    <button
                      type="button"
                      className="storage-name"
                      disabled={!expandable}
                      aria-expanded={shown}
                      onClick={() => setOpen(shown ? null : k.kind)}
                    >
                      <i className={`swatch kind-${k.kind}`} />
                      <span>{KIND[k.kind].name}</span>
                      {expandable && <span className="muted">{items.length}개</span>}
                    </button>
                    <span className="muted small storage-about" title={KIND[k.kind].about}>
                      {KIND[k.kind].about}
                    </span>
                    <strong className="storage-size">{bytesLabel(k.bytes)}</strong>
                    {k.kind === "images" ? (
                      <button
                        type="button"
                        className="small"
                        onClick={() => document.getElementById("node-copies")?.scrollIntoView({ behavior: "smooth" })}
                      >
                        버전 보기
                      </button>
                    ) : isAdmin && k.removable ? (
                      // a disabled button shows no tooltip: the reason sits on what holds it
                      <span className="btn-wrap" title={whyNotEmpty(k, free_.length, data.can_clean, cleaning)}>
                      <button
                        type="button"
                        className="small"
                        disabled={!canEmpty || busy || cleaning}
                        onClick={() => empty(k)}
                      >
                        {k.kind === "uv" ? "정리" : "비우기"}
                      </button>
                      </span>
                    ) : (
                      <span />
                    )}
                    {k.path && (
                      <div className="storage-path small">
                        <code className="mono" title={k.path}>
                          {k.path}
                        </code>
                        <span className="muted">{placeNote(k)}</span>
                        {k.kind === "containers" && !k.desktop && isAdmin && (
                          <button type="button" className="link small" onClick={() => setDockerMove(true)}>
                            다른 디스크로 옮기기
                          </button>
                        )}
                      </div>
                    )}
                  </div>
                  {shown && (
                    <ul className="storage-items">
                      {items.map((item) => (
                        <li key={item.image ?? item.id}>
                          <span className="storage-item-name">{itemTitle(k.kind, item)}</span>
                          <span className="muted small">{itemNote(k.kind, item)}</span>
                          <span className="storage-size">{bytesLabel(item.bytes)}</span>
                          {item.in_use ? (
                            <Badge kind="running">작업이 쓰는 중</Badge>
                          ) : isAdmin && data.can_clean ? (
                            <button
                              type="button"
                              className="small"
                              disabled={busy || cleaning}
                              onClick={() => removeOne(k, item)}
                            >
                              지우기
                            </button>
                          ) : (
                            <span />
                          )}
                          {itemPath(k, item) && (
                            <code className="mono storage-item-path" title={itemPath(k, item)}>
                              {itemPath(k, item)}
                            </code>
                          )}
                        </li>
                      ))}
                    </ul>
                  )}
                </li>
              );
            })}
          </ul>
          {emptyKinds.length > 0 && (
            <p className="muted small storage-rest">
              {shownKinds.length === 0 ? "MOA가 이 서버에 둔 것이 없습니다. " : ""}
              비어 있음: {emptyKinds.map((k) => KIND[k.kind].name).join(", ")}
            </p>
          )}
        </>
      )}
      {(cleaning && !measuring) || last || error ? (
        <p className="small storage-note">
          {cleaning && !measuring ? (
            <span className="muted">지우는 중… 끝나면 다시 재서 보여 줍니다.</span>
          ) : last ? (
            <span className={last.ok ? "muted" : "bad-text"}>
              마지막 정리({KIND[last.kind as Kind]?.name ?? last.kind}, {timeAgo(last.at)}):{" "}
              {last.ok ? `${last.removed ?? 0}개 · ${bytesLabel(last.bytes ?? 0)} 지움` : last.error}
            </span>
          ) : null}
          {error && <span className="bad-text"> {error}</span>}
        </p>
      ) : null}
      {dockerMove && <DockerMoveDialog nodeId={nodeId} suggest="" onClose={() => setDockerMove(false)} />}
    </section>
  );
}
