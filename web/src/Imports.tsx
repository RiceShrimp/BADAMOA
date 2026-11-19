import { useEffect, useRef, useState } from "react";
import { api } from "./api";
import { Icon, timeAgo } from "./ui";

/** A dataset import, as the Hub tracks it: it keeps going whatever screen is open. */
export interface ImportInfo {
  job_id: string;
  status: "queued" | "running" | "done" | "failed" | "canceled";
  project: { id: string; name: string };
  dataset: string;
  version: string;
  release_id: string | null;
  node: string;
  done: number;
  total: number;
  skipped: number;
  created_at: string;
  started_at: string | null;
  finished_at: string | null;
  error: string;
}

const DISMISSED_KEY = "badamoa.importsSeen";

function dismissed(): string[] {
  try {
    return JSON.parse(localStorage.getItem(DISMISSED_KEY) ?? "[]");
  } catch {
    return [];
  }
}

/** Imports under way and just ended; asks often while one runs, rarely otherwise. */
export function useImports(): ImportInfo[] {
  const [list, setList] = useState<ImportInfo[]>([]);
  useEffect(() => {
    let alive = true;
    let timer: ReturnType<typeof setTimeout>;
    const tick = async () => {
      let busy = false;
      try {
        const got = await api<ImportInfo[]>("/imports");
        if (!alive) return;
        setList(got);
        busy = got.some((i) => i.status === "queued" || i.status === "running");
      } catch {
        /* the next tick tries again */
      }
      if (alive) timer = setTimeout(tick, busy ? 2000 : 15000);
    };
    void tick();
    const kick = () => {
      clearTimeout(timer);
      void tick();
    };
    window.addEventListener("badamoa:import-started", kick);
    return () => {
      alive = false;
      clearTimeout(timer);
      window.removeEventListener("badamoa:import-started", kick);
    };
  }, []);
  return list;
}

/** Tell every screen an import was just started (they look again at once instead of waiting). */
export function announceImport() {
  window.dispatchEvent(new Event("badamoa:import-started"));
}

function left(info: ImportInfo): string {
  if (!info.started_at || info.done <= 0 || info.total <= info.done) return "";
  const elapsed = (Date.now() - new Date(info.started_at).getTime()) / 1000;
  const seconds = ((info.total - info.done) * elapsed) / info.done;
  if (seconds < 60) return "1분 안에";
  if (seconds < 3600) return `약 ${Math.round(seconds / 60)}분 남음`;
  return `약 ${(seconds / 3600).toFixed(1)}시간 남음`;
}

/** One import's bar and words: n / total, speed-based time left, or how it ended. */
export function ImportProgress({ info, compact = false }: { info: ImportInfo; compact?: boolean }) {
  const pct = info.total > 0 ? Math.min(100, (info.done / info.total) * 100) : 0;
  const words =
    info.status === "queued"
      ? "차례를 기다리는 중"
      : info.status === "failed"
        ? `실패${info.error ? ` — ${info.error}` : ""}`
        : info.status === "canceled"
          ? "중지함 · 남긴 것 없음"
          : info.status === "done"
            ? `완료 · ${info.total.toLocaleString()}장${info.skipped ? ` (읽지 못한 파일 ${info.skipped}개 건너뜀)` : ""}`
            : info.total > 0
              ? `${info.done.toLocaleString()} / ${info.total.toLocaleString()}장 · ${Math.floor(pct)}%${left(info) ? ` · ${left(info)}` : ""}`
              : "폴더를 읽는 중";
  return (
    <div className={`import-progress ${info.status}${compact ? " compact" : ""}`}>
      <div className="bar">
        <i style={{ width: `${info.status === "done" ? 100 : pct}%` }} />
      </div>
      <div className="import-progress-line">
        <span className="small">{words}</span>
        {(info.status === "queued" || info.status === "running") && <StopImport info={info} />}
      </div>
    </div>
  );
}

/** Stop an import: one waiting goes at once; one under way stops on its server, and what it read so far goes. */
export function StopImport({ info }: { info: Pick<ImportInfo, "job_id" | "status" | "dataset" | "version"> }) {
  const [stopping, setStopping] = useState(false);
  const stop = async (e: { preventDefault: () => void; stopPropagation: () => void }) => {
    e.preventDefault(); // it may sit inside a card that is a link
    e.stopPropagation();
    const name = `${info.dataset} ${info.version}`;
    const ask =
      info.status === "queued"
        ? `기다리는 ${name} 등록을 취소할까요?`
        : `${name} 등록을 멈출까요?\n지금까지 읽은 것은 남지 않고, 같은 이름으로 다시 가져올 수 있습니다.`;
    if (!window.confirm(ask)) return;
    setStopping(true);
    try {
      await api(`/jobs/${info.job_id}/cancel`, { method: "POST" });
      announceImport();
    } catch (err) {
      window.alert(err instanceof Error ? err.message : String(err));
      setStopping(false);
    }
  };
  return (
    <button className="link small import-stop" onClick={(e) => void stop(e)} disabled={stopping}>
      {stopping ? "멈추는 중…" : info.status === "queued" ? "취소" : "중지"}
    </button>
  );
}

/** Top bar: imports under way (from any screen), the ones just finished until looked at. */
export function ImportTray() {
  const all = useImports();
  const [seen, setSeen] = useState<string[]>(dismissed);
  const [open, setOpen] = useState(false);
  const box = useRef<HTMLDivElement>(null);
  const shown = all.filter((i) => i.status === "queued" || i.status === "running" || !seen.includes(i.job_id));
  const running = shown.filter((i) => i.status === "queued" || i.status === "running");
  const failed = shown.filter((i) => i.status === "failed");

  useEffect(() => {
    if (!open) return;
    const away = (e: MouseEvent) => box.current && !box.current.contains(e.target as Node) && setOpen(false);
    document.addEventListener("mousedown", away);
    return () => document.removeEventListener("mousedown", away);
  }, [open]);

  if (shown.length === 0) return null;
  const forget = (ids: string[]) => {
    const next = [...new Set([...seen, ...ids])].slice(-100);
    setSeen(next);
    try {
      localStorage.setItem(DISMISSED_KEY, JSON.stringify(next));
    } catch {
      /* not remembered */
    }
  };
  const first = running[0];
  const label = running.length
    ? `등록 중 ${running.length}${first && first.total ? ` · ${Math.floor((first.done / first.total) * 100)}%` : ""}`
    : failed.length
      ? `등록 실패 ${failed.length}`
      : shown.some((i) => i.status === "done")
        ? `등록 완료 ${shown.filter((i) => i.status === "done").length}`
        : `등록 중지 ${shown.length}`;
  return (
    <div className="import-tray" ref={box}>
      <button className={`pill ${running.length ? "busy" : failed.length ? "bad" : "ok"}`} onClick={() => setOpen(!open)}>
        {running.length > 0 ? <span className="spinner" /> : <Icon name="database" size={13} />}
        {label}
      </button>
      {open && (
        <div className="import-panel">
          <div className="import-panel-head">
            <strong>데이터셋 등록</strong>
            <span className="muted small">다른 화면에서 일해도 등록은 계속됩니다</span>
          </div>
          {shown.map((i) => (
            <div key={i.job_id} className="import-row">
              <div className="import-row-head">
                <strong>
                  {i.dataset} {i.version}
                </strong>
                <span className="muted small">
                  {i.project.name} · {i.node} · {timeAgo(i.created_at)}
                </span>
              </div>
              <ImportProgress info={i} compact />
              <div className="row-links">
                {i.release_id && (i.status === "done" || i.status === "running") && (
                  <a className="small" href={`#/datasets/${i.release_id}`} onClick={() => setOpen(false)}>
                    {i.status === "done" ? "열기" : "지금까지 본 것 보기"}
                  </a>
                )}
                {i.status === "failed" && (
                  <a className="small" href="#/jobs" onClick={() => setOpen(false)}>
                    로그 보기
                  </a>
                )}
              </div>
            </div>
          ))}
          {shown.some((i) => ["done", "failed", "canceled"].includes(i.status)) && (
            <button className="link small" onClick={() => forget(shown.filter((i) => ["done", "failed", "canceled"].includes(i.status)).map((i) => i.job_id))}>
              끝난 것 지우기
            </button>
          )}
        </div>
      )}
    </div>
  );
}
