import { type ReactNode, useEffect, useRef, useState } from "react";

/** Re-run `load` every `ms` while mounted; returns the latest value, error, and a manual refresh. */
export function usePolling<T>(load: () => Promise<T>, ms: number, deps: unknown[] = []) {
  const [data, setData] = useState<T | null>(null);
  const [error, setError] = useState<string | null>(null);
  const loadRef = useRef(load);
  loadRef.current = load;

  const refresh = async () => {
    try {
      setData(await loadRef.current());
      setError(null);
    } catch (e) {
      setError(e instanceof Error ? e.message : String(e));
    }
  };

  useEffect(() => {
    let alive = true;
    const tick = () => {
      if (alive) void refresh();
    };
    tick();
    const id = setInterval(tick, ms);
    return () => {
      alive = false;
      clearInterval(id);
    };
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [ms, ...deps]);

  return { data, error, refresh };
}

export function Badge({ kind, children }: { kind: string; children: ReactNode }) {
  return <span className={`badge badge-${kind}`}>{children}</span>;
}

export function Modal({ title, onClose, children }: { title: string; onClose: () => void; children: ReactNode }) {
  useEffect(() => {
    const onKey = (e: KeyboardEvent) => e.key === "Escape" && onClose();
    window.addEventListener("keydown", onKey);
    return () => window.removeEventListener("keydown", onKey);
  }, [onClose]);
  return (
    <div className="modal-backdrop" onMouseDown={(e) => e.target === e.currentTarget && onClose()}>
      <div className="modal" role="dialog" aria-modal="true" aria-label={title}>
        <header>
          <h2>{title}</h2>
          <button className="ghost" onClick={onClose} aria-label="닫기">
            ✕
          </button>
        </header>
        {children}
      </div>
    </div>
  );
}

export function CopyBlock({ label, text }: { label?: string; text: string }) {
  const [copied, setCopied] = useState(false);
  const copy = async () => {
    try {
      await navigator.clipboard.writeText(text);
      setCopied(true);
      setTimeout(() => setCopied(false), 1500);
    } catch {
      /* clipboard blocked: the text stays selectable */
    }
  };
  return (
    <div className="copy-block">
      {label && <div className="copy-label">{label}</div>}
      <div className="copy-row">
        <pre>{text}</pre>
        <button onClick={copy}>{copied ? "복사됨" : "복사"}</button>
      </div>
    </div>
  );
}

export function Meter({ value, max, unit }: { value?: number | null; max?: number | null; unit: string }) {
  if (value == null || !max) return <span className="muted">–</span>;
  const pct = Math.min(100, Math.round((value / max) * 100));
  return (
    <div className="meter" title={`${value} / ${max} ${unit}`}>
      <div className="meter-bar">
        <div className={`meter-fill ${pct > 90 ? "hot" : ""}`} style={{ width: `${pct}%` }} />
      </div>
      <span>{pct}%</span>
    </div>
  );
}

export function timeAgo(iso: string | null): string {
  if (!iso) return "–";
  const s = Math.max(0, (Date.now() - new Date(iso).getTime()) / 1000);
  if (s < 60) return `${Math.floor(s)}초 전`;
  if (s < 3600) return `${Math.floor(s / 60)}분 전`;
  if (s < 86400) return `${Math.floor(s / 3600)}시간 전`;
  return `${Math.floor(s / 86400)}일 전`;
}

export function duration(seconds: number | null): string {
  if (seconds == null) return "–";
  if (seconds < 60) return `${seconds.toFixed(1)}초`;
  return `${Math.floor(seconds / 60)}분 ${Math.round(seconds % 60)}초`;
}

export const STATUS_LABEL: Record<string, string> = {
  online: "온라인",
  offline: "오프라인",
  disabled: "비활성",
  queued: "대기",
  running: "실행 중",
  succeeded: "성공",
  failed: "실패",
  canceled: "취소됨",
};
