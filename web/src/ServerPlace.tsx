import type { ReactNode } from "react";
import { fileSize } from "./Files";
import type { NodeGpu } from "./trainTypes";

/**
 * Where a job over a version's images runs, as every start dialog asks it (새 학습 · 새 임베딩 · 새 오토라벨 ·
 * 모델로 검사하기): the server (picked, never chosen for you), what happens to the images (there, or received
 * first: how much, where to and the room there), the GPU by number, and what becomes of a copy sent there.
 * Laid out as rows of a label on the left and the choice on the right.
 */

/** A server as the Hub lists it for a start dialog. */
export interface PlaceServer {
  id: string;
  name: string;
  status: string;
  gpus: number;
  fits: boolean;
  why: string;
  transfer: boolean; // the images are sent there first
  holds_images?: boolean;
  trained_here?: boolean;
  gpu_list?: NodeGpu[];
  images?: "registered" | "received" | ""; // imported there · copied there before · sent there first
  cache?: { dir: string; free_gb: number | null; auto: boolean }; // where a copy goes there, and the room left
}

/** 자동 (a free one) · CPU only · GPUs by number. */
export type GpuChoice = "auto" | "cpu" | number[];

const GB = 1024 ** 3;

/** The copy would not fit where the server keeps received images (as it last said). */
export const spaceShort = (server: PlaceServer | null | undefined, bytes: number) =>
  Boolean(server?.transfer && server.cache?.free_gb != null && server.cache.free_gb * GB < bytes);

/** A server keeps a copy that came from elsewhere (or will receive one): its fate can be chosen. */
export const copiedTo = (server: PlaceServer | null | undefined) => Boolean(server && server.images !== "registered" && (server.images || server.transfer));

/** The request's GPU fields for a choice (one server). */
export function gpuBody(choice: GpuChoice, server: PlaceServer | null | undefined) {
  if (choice === "cpu" || !server?.gpus) return { gpus: 0, gpu_indices: null };
  if (choice === "auto") return { gpus: null, gpu_indices: null };
  return { gpus: choice.length, gpu_indices: choice };
}

export function PlaceGrid({ children }: { children: ReactNode }) {
  return <div className="place-grid">{children}</div>;
}

export function PlaceRow({ label, children }: { label: string; children: ReactNode }) {
  return (
    <>
      <div className="place-label">{label}</div>
      <div className="place-body">{children}</div>
    </>
  );
}

/** One server, nothing chosen until someone picks; each with its GPUs and whether it has the images. */
export function ServerSelect({
  servers,
  value,
  onChange,
  label = "서버",
}: {
  servers: PlaceServer[];
  value: string;
  onChange: (id: string) => void;
  label?: string;
}) {
  const note = (s: PlaceServer) => {
    const parts = [s.gpus ? `GPU ${s.gpus}개` : "CPU"];
    parts.push(s.images === "registered" || (s.holds_images && !s.images) ? "이미지 있음" : s.images === "received" ? "받아 둔 이미지 있음" : "이미지를 먼저 받음");
    if (s.trained_here) parts.push("학습한 서버");
    if (s.status !== "online") parts.push("지금 응답 없음");
    return parts.join(" · ");
  };
  return (
    <select value={value} onChange={(e) => onChange(e.target.value)} aria-label={label} className={value ? "" : "unpicked"}>
      <option value="">{servers.some((s) => s.fits) ? "서버를 고르세요" : "고를 수 있는 서버가 없습니다"}</option>
      {servers.map((s) => (
        <option key={s.id} value={s.id} disabled={!s.fits}>
          {s.name} · {note(s)}
          {s.fits ? "" : ` — ${s.why}`}
        </option>
      ))}
    </select>
  );
}

/** What happens to the images on the chosen server: used where they are, or received first (how much, where, room). */
export function ImagesNote({
  server,
  holder,
  items,
  bytes,
  what = "이 버전의 이미지",
}: {
  server: PlaceServer;
  holder?: string;
  items: number;
  bytes: number;
  what?: string;
}) {
  if (!server.fits) return <span className="bad-text small">{server.why}</span>;
  if (!server.transfer) {
    return (
      <span className="muted small">
        {server.images === "received" ? "받아 둔 이미지로" : "이 서버의 이미지로"} 바로 시작합니다.
        {server.status !== "online" ? " 지금 응답이 없어 켜지면 시작합니다." : ""}
      </span>
    );
  }
  const short = spaceShort(server, bytes);
  return (
    <>
      <span className="muted small">
        {holder ?? "이미지가 있는 서버"}에서 {what} {items.toLocaleString()}장({fileSize(bytes)})을 먼저 받습니다. 받는 동안 GPU는 잡지 않습니다.
      </span>
      {server.cache?.dir && (
        <span className="small place-where">
          <span className="muted">받을 곳</span> <code>{server.cache.dir}</code>
          <span className={short ? "bad-text" : "muted"}>
            {server.cache.free_gb != null ? ` · 남은 공간 ${server.cache.free_gb.toLocaleString()} GB` : " · 남은 공간 모름"}
          </span>
          <span className="muted"> · {server.cache.auto ? "처음 받을 때 가장 여유 있는 디스크로 정해진 곳" : "노드 설정에서 정한 곳"}</span>
        </span>
      )}
      {short && (
        <span className="bad-text small">
          남은 공간이 받을 양보다 적어 시작할 수 없습니다. <a href="#/nodes">노드</a> 설정에서 받은 이미지 보관 위치를 여유 있는 디스크로 바꾸거나 다른
          서버를 고르세요.
        </span>
      )}
    </>
  );
}

const GPU_STATE: Record<NodeGpu["state"], string> = {
  free: "비어 있음",
  busy: "사용 중",
  job: "다른 작업 중",
  reserved: "사람 전용",
  offline: "오프라인",
};

/** 자동 · CPU만 · each GPU of the server (several when ``multi``, else one). */
export function GpuChips({
  gpus,
  pick,
  onPick,
  multi = false,
  doing = "돌립니다",
}: {
  gpus: NodeGpu[];
  pick: GpuChoice;
  onPick: (pick: GpuChoice) => void;
  multi?: boolean;
  doing?: string;
}) {
  const ticked = Array.isArray(pick) ? pick : [];
  const toggle = (index: number) => {
    if (!multi) return onPick(ticked.includes(index) ? "auto" : [index]);
    const next = ticked.includes(index) ? ticked.filter((i) => i !== index) : [...ticked, index].sort((a, b) => a - b);
    onPick(next.length ? next : "auto");
  };
  const waiting = gpus.filter((g) => ticked.includes(g.index) && g.state !== "free");
  return (
    <>
      <div className="gpu-pick">
        <button type="button" className={`chip ${pick === "auto" ? "active" : ""}`} onClick={() => onPick("auto")}>
          자동
        </button>
        <button type="button" className={`chip ${pick === "cpu" ? "active" : ""}`} onClick={() => onPick("cpu")}>
          CPU만
        </button>
        {gpus.map((g) => {
          const label = `GPU ${g.index}${g.name ? ` · ${g.name.replace(/^NVIDIA (GeForce )?/, "")}` : ""}`;
          const who = g.users.length ? g.users.join(", ") : "";
          const why = g.reserved ? `사람 전용${who ? ` · ${who}` : ""}` : g.state === "busy" && who ? `사용 중: ${who}` : GPU_STATE[g.state];
          return (
            <button
              type="button"
              key={g.index}
              className={`chip gpu ${g.state} ${ticked.includes(g.index) ? "active" : ""}`}
              disabled={g.reserved}
              aria-pressed={ticked.includes(g.index)}
              title={g.reserved ? "사람 전용으로 빼 둔 GPU입니다 (노드 설정)" : why}
              onClick={() => toggle(g.index)}
            >
              {label} <span>{why}</span>
            </button>
          );
        })}
      </div>
      <span className="muted small">
        {pick === "auto"
          ? `비어 있는 GPU 1개로 ${doing}.`
          : pick === "cpu"
            ? `GPU 없이 CPU로 ${doing}.`
            : `GPU ${ticked.join(", ")}${multi ? ` (${ticked.length}개)` : ""}로 ${doing}.` +
              (waiting.length ? ` 지금 쓰이고 있는 GPU ${waiting.map((g) => g.index).join(", ")}이(가) 비면 시작합니다.` : "")}
      </span>
    </>
  );
}

/** 남겨 두기 · 지우기 for the copy the server received, once the job is over. */
export function DataAfterChoice({
  value,
  onChange,
  when,
  deleteNote,
}: {
  value: "keep" | "delete";
  onChange: (value: "keep" | "delete") => void;
  when: string; // "학습이 끝나면"
  deleteNote?: string;
}) {
  return (
    <>
      <div className="segmented" role="radiogroup" aria-label="보낸 데이터">
        <button type="button" className={value === "keep" ? "active" : ""} onClick={() => onChange("keep")}>
          남겨 두기
        </button>
        <button type="button" className={value === "delete" ? "active" : ""} onClick={() => onChange("delete")}>
          지우기
        </button>
      </div>
      <span className="muted small">
        {value === "keep"
          ? `${when} 이 서버로 보낸 이미지를 남겨, 같은 버전으로 다시 돌리면 바로 시작합니다. 버전 화면에서 언제든 지울 수 있습니다.`
          : (deleteNote ?? `${when} 이 서버로 보낸 이미지를 지웁니다(성공했을 때만, 그 서버에서 이 버전을 쓰는 작업이 있으면 끝난 뒤). 실패하면 다시 돌릴 수 있게 남깁니다.`)}
      </span>
    </>
  );
}
