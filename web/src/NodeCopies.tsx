import { useState } from "react";
import { api } from "./api";
import { bytesLabel } from "./catalogTypes";
import "./copies.css";
import { Badge, timeAgo, usePolling } from "./ui";

/** A copy of a version's images that a server received to train on it (the Hub's record of it). */
export interface Copy {
  node_id: string;
  node_name: string;
  node_status: string;
  items: number;
  bytes: number;
  received_at: string | null;
  state: "kept" | "remove" | "held" | "removing";
  present_items: number | null; // what the server's last check found of it
  present_bytes: number | null;
  checked_at: string | null;
  runs: number; // trainings and evaluations it served there
  last_used_at: string | null;
  in_use: string[];
  removable: boolean;
  why_not: string;
  release?: {
    id: string;
    version: string;
    digest: string;
    dataset: string;
    project_id: string;
    project: string;
    deleted: boolean;
  };
}

interface ReleaseCopiesInfo {
  release_id: string;
  digest: string;
  origin: { node_id: string; node_name: string; node_status: string } | null;
  copies: Copy[];
}

interface Sizes {
  files: number;
  bytes: number;
}

interface NodeCopiesInfo {
  node_id: string;
  copies: Copy[];
  bytes: number;
  check: { checked_at?: string; clean?: boolean; unknown?: Sizes; removed?: Sizes; error?: string | null; running: boolean };
  can_check: boolean;
}

const STATE: Record<Copy["state"], [string, string]> = {
  kept: ["남겨 둠", "succeeded"],
  remove: ["지울 예정", "queued"],
  held: ["지울 예정 · 실패해서 남김", "queued"],
  removing: ["지우는 중", "running"],
};

/** "8장 · 12 MB", or with a check that found less: "7 / 8장 · 일부 없음". */
function amount(c: Copy): string {
  const size = bytesLabel(c.bytes);
  if (c.present_items != null && c.present_items < c.items) {
    return `${c.present_items.toLocaleString()} / ${c.items.toLocaleString()}장 · ${size} · 일부 없음`;
  }
  return `${c.items.toLocaleString()}장 · ${size}`;
}

/** A version's fingerprint as screens show it: the first 12 characters, the whole one copied on a click. */
export function Fingerprint({ digest }: { digest: string }) {
  const [copied, setCopied] = useState(false);
  const copy = () => {
    void navigator.clipboard?.writeText(digest).then(() => {
      setCopied(true);
      window.setTimeout(() => setCopied(false), 1500);
    });
  };
  return (
    <button type="button" className="digest" title={`${digest}\n누르면 전체를 복사합니다`} onClick={copy}>
      {copied ? "복사됨" : digest.slice(0, 12)}
    </button>
  );
}

function used(c: Copy): string {
  if (c.in_use.length) return `쓰는 중: ${c.in_use.join(", ")}`;
  return c.runs ? `학습 · 평가 ${c.runs}번 · 마지막 ${timeAgo(c.last_used_at)}` : "아직 쓰지 않음";
}

/** The servers a version's images are on: shown on the version's screen once another server received a copy. */
export function ReleaseCopies({ releaseId }: { releaseId: string }) {
  const { data, refresh } = usePolling(() => api<ReleaseCopiesInfo>(`/releases/${releaseId}/copies`), 15000, [releaseId]);
  const [busy, setBusy] = useState("");
  const [error, setError] = useState<string | null>(null);
  if (!data || data.copies.length === 0) return null;
  const remove = async (c: Copy) => {
    const ok = window.confirm(
      `${c.node_name}에 받아 둔 이 버전의 이미지(${bytesLabel(c.bytes)})를 지울까요? 원본은 그대로이고, 다시 학습하면 다시 받습니다.`,
    );
    if (!ok) return;
    setBusy(c.node_id);
    setError(null);
    try {
      await api(`/releases/${releaseId}/copies/${c.node_id}`, { method: "DELETE" });
      await refresh();
    } catch (err) {
      setError(err instanceof Error ? err.message : String(err));
    } finally {
      setBusy("");
    }
  };
  return (
    <section className="card copies">
      <div className="card-head">
        <h3>이 버전이 있는 서버</h3>
        <span className="muted small">학습하려고 다른 서버로 보낸 이미지</span>
      </div>
      <ul className="copy-list">
        {data.origin && (
          <li>
            <span className="copy-name">
              <i className={`dot ${data.origin.node_status}`} />
              {data.origin.node_name}
            </span>
            <span className="muted small">원본 · 등록한 곳</span>
            <span />
            <span />
          </li>
        )}
        {data.copies.map((c) => (
          <li key={c.node_id}>
            <span className="copy-name">
              <i className={`dot ${c.node_status}`} />
              <a href={`#/nodes/${c.node_id}`}>{c.node_name}</a>
            </span>
            <span className="muted small">
              {amount(c)} · 받은 때 {timeAgo(c.received_at)} · {used(c)}
            </span>
            <Badge kind={STATE[c.state][1]}>{STATE[c.state][0]}</Badge>
            <button
              type="button"
              className="small"
              disabled={!c.removable || busy === c.node_id}
              title={c.why_not || "이 서버에서 이 버전의 사본을 지웁니다 (다른 버전과 함께 쓰는 이미지는 남김)"}
              onClick={() => void remove(c)}
            >
              지우기
            </button>
          </li>
        ))}
      </ul>
      {error && <p className="bad-text small">{error}</p>}
    </section>
  );
}

/** What a server keeps of versions registered elsewhere, and whether that matches the Hub's records. */
export function NodeCopies({ nodeId }: { nodeId: string }) {
  const { data, refresh } = usePolling(() => api<NodeCopiesInfo>(`/nodes/${nodeId}/copies`), 10000, [nodeId]);
  const [busy, setBusy] = useState("");
  const [error, setError] = useState<string | null>(null);
  if (!data) return null;
  const act = async (key: string, run: () => Promise<unknown>) => {
    setBusy(key);
    setError(null);
    try {
      await run();
      await refresh();
    } catch (err) {
      setError(err instanceof Error ? err.message : String(err));
    } finally {
      setBusy("");
    }
  };
  const check = (clean: boolean) => act(clean ? "clean" : "check", () => api(`/nodes/${nodeId}/copies/check`, { body: { clean } }));
  const remove = (c: Copy) => {
    const name = `${c.release?.dataset} ${c.release?.version}`;
    if (!window.confirm(`이 서버에 받아 둔 ${name}의 이미지(${bytesLabel(c.bytes)})를 지울까요? 다시 학습하면 다시 받습니다.`)) return;
    void act(c.release?.id ?? "", () => api(`/releases/${c.release?.id}/copies/${nodeId}`, { method: "DELETE" }));
  };
  const last = data.check;
  const left = (last.unknown?.files ?? 0) - (last.removed?.files ?? 0);
  const leftBytes = (last.unknown?.bytes ?? 0) - (last.removed?.bytes ?? 0);
  return (
    <section className="card copies">
      <div className="card-head">
        <h3>받아 둔 버전</h3>
        <span className="muted small">
          다른 서버에서 받아 둔 이미지{data.copies.length ? ` · ${data.copies.length}개 버전 · ${bytesLabel(data.bytes)}` : ""}
        </span>
        <span className="spacer" />
        {data.can_check && (
          <button
            type="button"
            className="small"
            disabled={last.running || busy !== ""}
            title="이 서버에 실제로 있는 이미지를 Hub의 기록과 맞춰 봅니다"
            onClick={() => void check(false)}
          >
            {last.running ? "맞춰 보는 중…" : "맞춰 보기"}
          </button>
        )}
      </div>
      {data.copies.length === 0 ? (
        <p className="muted small copy-empty">받아 둔 버전이 없습니다. 다른 서버에 있는 이미지로 여기서 학습하면 여기에 보입니다.</p>
      ) : (
        <ul className="copy-list">
          {data.copies.map((c) => (
            <li key={c.release?.id}>
              <span className="copy-name">
                {c.release?.deleted ? (
                  <span className="muted">
                    {c.release.dataset} {c.release.version} (휴지통)
                  </span>
                ) : (
                  <a href={`#/datasets/${c.release?.id}`}>
                    {c.release?.dataset} {c.release?.version}
                  </a>
                )}
                {c.release?.digest && <code className="digest">{c.release.digest.slice(0, 12)}</code>}
              </span>
              <span className="muted small">
                {c.release?.project} · {amount(c)} · 받은 때 {timeAgo(c.received_at)} · {used(c)}
              </span>
              <Badge kind={STATE[c.state][1]}>{STATE[c.state][0]}</Badge>
              <button
                type="button"
                className="small"
                disabled={!c.removable || busy !== ""}
                title={c.why_not || "이 서버에서 이 버전의 사본을 지웁니다 (다른 버전과 함께 쓰는 이미지는 남김)"}
                onClick={() => remove(c)}
              >
                지우기
              </button>
            </li>
          ))}
        </ul>
      )}
      {last.checked_at && (
        <p className="muted small copy-check">
          마지막으로 맞춰 본 때 {timeAgo(last.checked_at)}
          {last.clean && last.removed?.files ? ` · 기록에 없는 파일 ${last.removed.files.toLocaleString()}개 (${bytesLabel(last.removed.bytes)}) 지움` : ""}
          {left > 0 && (
            <>
              {" "}
              · Hub 기록에 없는 파일 {left.toLocaleString()}개 ({bytesLabel(leftBytes)}){" "}
              <button type="button" className="link" disabled={busy !== "" || last.running} onClick={() => void check(true)}>
                정리
              </button>
            </>
          )}
        </p>
      )}
      {last.error && <p className="bad-text small">마지막 확인이 실패했습니다: {last.error}</p>}
      {error && <p className="bad-text small">{error}</p>}
    </section>
  );
}
