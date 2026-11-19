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
  /** Of the step it is on: images found (find, no total), label files read (labels), images sent (read). */
  done: number;
  total: number;
  skipped: number;
  /** "" while waiting or once ended. */
  step: "" | "find" | "labels" | "read";
  step_started_at: string | null;
  /** Of the images read and sent so far. */
  bytes: number;
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

/** Seconds the step it is on has run. */
function elapsed(info: ImportInfo): number {
  const since = info.step_started_at ?? info.started_at;
  return since ? Math.max(1, (Date.now() - new Date(since).getTime()) / 1000) : 0;
}

/** What each running import's step was seen at, for its pace over the last minute (screens poll every 2 s). */
const seen = new Map<string, { step: string; points: { t: number; done: number; bytes: number }[] }>();
const PACE_WINDOW_MS = 60_000;

/** How fast the step goes now: items and bytes a second over the last minute, or since the step began
 * until a minute has been seen (0 for the first few seconds). */
function pace(info: ImportInfo): { items: number; bytes: number } {
  const now = Date.now();
  let mine = seen.get(info.job_id);
  if (!mine || mine.step !== info.step) {
    mine = { step: info.step, points: [] };
    seen.set(info.job_id, mine);
  }
  const last = mine.points[mine.points.length - 1];
  if (!last || last.done !== info.done || last.bytes !== info.bytes) mine.points.push({ t: now, done: info.done, bytes: info.bytes });
  mine.points = mine.points.filter((p) => now - p.t <= PACE_WINDOW_MS);
  const first = mine.points[0];
  const newest = mine.points[mine.points.length - 1];
  if (first && newest && newest.t - first.t >= 15_000) {
    const span = (newest.t - first.t) / 1000;
    return { items: (newest.done - first.done) / span, bytes: (newest.bytes - first.bytes) / span };
  }
  const seconds = elapsed(info);
  return seconds >= 5 ? { items: info.done / seconds, bytes: info.bytes / seconds } : { items: 0, bytes: 0 };
}

function perSecond(value: number, unit: string): string {
  return `초당 ${value.toLocaleString(undefined, { maximumFractionDigits: value < 10 ? 1 : 0 })}${unit}`;
}

/** Time left in this step, at its pace now. */
function left(info: ImportInfo): string {
  const speed = pace(info).items;
  if (speed <= 0 || info.total <= info.done || elapsed(info) < 10) return "";
  const rest = (info.total - info.done) / speed;
  if (rest < 60) return "1분 안에";
  if (rest < 3600) return `약 ${Math.round(rest / 60)}분 남음`;
  return `약 ${(rest / 3600).toFixed(1)}시간 남음`;
}

function size(bytes: number): string {
  const gb = bytes / 1024 ** 3;
  return gb >= 1 ? `${gb.toFixed(gb >= 100 ? 0 : 1)} GB` : `${Math.round(bytes / 1024 ** 2)} MB`;
}

const STEPS = { find: "이미지 찾는 중", labels: "라벨 읽는 중", read: "이미지 읽고 등록" } as const;

/** What a running import is doing and how fast, e.g. 이미지 읽고 등록 (3/3) · 12,000 / 600,000장 · 2% · 15 GB 읽음 ·
 * 초당 34장 · 21 MB/s · 약 4.7시간 남음. */
function runningWords(info: ImportInfo, pct: number): string {
  const step = info.step || (info.total > 0 ? "read" : "find");
  const head = `${STEPS[step]} (${["find", "labels", "read"].indexOf(step) + 1}/3)`;
  const speed = pace(info);
  if (step === "find") {
    const parts = [head];
    if (info.done > 0) parts.push(`${info.done.toLocaleString()}장 찾음`);
    if (speed.items > 0) parts.push(perSecond(speed.items, "장"));
    return parts.join(" · ");
  }
  const parts = [head, `${info.done.toLocaleString()} / ${info.total.toLocaleString()}${step === "read" ? "장" : ""}`];
  parts.push(`${Math.floor(pct)}%`);
  if (step === "read" && info.bytes > 0) parts.push(`${size(info.bytes)} 읽음`);
  if (speed.items > 0) parts.push(perSecond(speed.items, step === "read" ? "장" : "개"));
  if (step === "read" && speed.bytes > 0) parts.push(`${Math.round(speed.bytes / 1024 ** 2).toLocaleString()} MB/s`);
  if (left(info)) parts.push(left(info));
  return parts.join(" · ");
}

/** The tray's short word for the first running import: its step and %. */
function shortWords(info: ImportInfo): string {
  if (info.status === "queued") return "대기";
  if (info.step === "find" || (!info.step && !info.total)) return "찾는 중";
  const pct = info.total ? Math.floor((info.done / info.total) * 100) : 0;
  return info.step === "labels" ? `라벨 ${pct}%` : `${pct}%`;
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
            : runningWords(info, pct);
  // Finding images has no end known yet: the bar moves without a %.
  const searching = info.status === "running" && (info.step === "find" || (!info.step && !info.total));
  return (
    <div className={`import-progress ${info.status}${compact ? " compact" : ""}${searching ? " searching" : ""}`}>
      <div className="bar">
        <i style={{ width: searching ? undefined : `${info.status === "done" ? 100 : pct}%` }} />
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
    ? `등록 중 ${running.length}${first ? ` · ${shortWords(first)}` : ""}`
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
