import { useState } from "react";
import "./nodes.css";
import { useCrosshair } from "./Chart";
import { api } from "./api";
import { usePolling } from "./ui";

interface Span {
  kind: "job" | "person";
  start: string;
  end: string;
  // a BADAMOA job
  job_id?: string;
  status?: string;
  label?: string;
  href?: string;
  epoch?: number;
  epochs?: number;
  // a person
  who?: string;
  programs?: string;
  memory_mb?: number;
}

interface Timeline {
  range: string;
  from: string;
  to: string;
  step_s: number;
  timestamps: string[];
  gpus: { index: number; name: string | null; memory_total_mb: number | null; reserved: boolean; util: (number | null)[]; spans: Span[] }[];
}

const RANGES = [
  { id: "6h", label: "6시간" },
  { id: "24h", label: "24시간" },
  { id: "7d", label: "7일" },
];

const time = (iso: string, days: boolean) => {
  const d = new Date(iso);
  return days
    ? `${d.getMonth() + 1}/${d.getDate()}`
    : `${String(d.getHours()).padStart(2, "0")}:${String(d.getMinutes()).padStart(2, "0")}`;
};

/**
 * One lane per GPU over the chosen range: the BADAMOA runs and the people that held it as
 * labelled bars, and below them how busy it was (darker = higher utilization).
 */
export function GpuTimeline({
  nodeId,
  refreshMs = null,
  reloadKey = 0,
}: {
  nodeId: string;
  refreshMs?: number | null; // the page's chosen interval; null: its own
  reloadKey?: number; // changes when the page's refresh button is pressed
}) {
  const [range, setRange] = useState("24h");
  const crosshair = useCrosshair();
  const { data, error } = usePolling(
    () => api<Timeline>(`/nodes/${nodeId}/timeline?range=${range}`),
    refreshMs ?? (range === "7d" ? 120000 : 30000),
    [nodeId, range, reloadKey],
  );
  if (error && !data) return <div className="error">{error}</div>;
  if (!data) return <p className="muted small">GPU 타임라인을 불러오는 중…</p>;

  const from = new Date(data.from).getTime();
  const to = new Date(data.to).getTime();
  const span = Math.max(1, to - from);
  const x = (iso: string) => Math.min(100, Math.max(0, ((new Date(iso).getTime() - from) / span) * 100));
  const cell = (data.step_s * 1000 * 100) / span;
  const days = data.range === "7d";
  const ticks = [0, 0.25, 0.5, 0.75].map((f) => new Date(from + f * span).toISOString());
  // The moment under the mouse here or in a chart below: a line across every lane.
  const cursor = crosshair?.time != null && crosshair.time >= from && crosshair.time <= to ? ((crosshair.time - from) / span) * 100 : null;
  const track = {
    onMouseMove: (e: React.MouseEvent<HTMLDivElement>) => {
      const box = e.currentTarget.getBoundingClientRect();
      crosshair?.set(from + ((e.clientX - box.left) / box.width) * span, "timeline");
    },
    onMouseLeave: () => crosshair?.set(null, null),
  };

  return (
    <section className="card timeline">
      <div className="card-head">
        <div>
          <h3>GPU 타임라인</h3>
          <div className="muted small">누가 언제 어느 GPU를 썼는지. 띠를 누르면 그 실행으로 갑니다.</div>
        </div>
        <div className="segmented" role="group" aria-label="기간">
          {RANGES.map((r) => (
            <button key={r.id} className={range === r.id ? "active" : ""} onClick={() => setRange(r.id)} aria-pressed={range === r.id}>
              {r.label}
            </button>
          ))}
        </div>
      </div>
      {data.gpus.length === 0 && <p className="muted">이 서버에는 GPU가 없습니다.</p>}
      {data.gpus.map((gpu) => (
        <div className="lane" key={gpu.index}>
          <div className="lane-name">
            <strong>GPU {gpu.index}</strong>
            {gpu.reserved && <span className="tag">사람 전용</span>}
            <div className="muted small">{gpu.name ?? ""}</div>
          </div>
          <div className="lane-track" {...track}>
            {cursor != null && <div className="lane-cursor" style={{ left: `${cursor}%` }} />}
            <div className="lane-spans">
              {gpu.spans.map((s, i) => {
                const left = x(s.start);
                const width = Math.max(0.4, x(s.end) - left);
                const title =
                  s.kind === "job"
                    ? `${s.label} · ${time(s.start, false)}–${time(s.end, false)}${s.status === "running" ? " (실행 중)" : ""}`
                    : `${s.who || "알 수 없는 사용자"} · ${s.programs || "프로세스"} · 최대 ${((s.memory_mb ?? 0) / 1024).toFixed(1)} GB · ${time(s.start, false)}–${time(s.end, false)}`;
                const text = s.kind === "job" ? s.label : s.who || "사람";
                const style = { left: `${left}%`, width: `${width}%` };
                return s.kind === "job" && s.href ? (
                  <a key={i} className={`lane-span job ${s.status ?? ""}`} style={style} href={s.href} title={title}>
                    {text}
                  </a>
                ) : (
                  <span key={i} className={`lane-span ${s.kind}`} style={style} title={title}>
                    {text}
                  </span>
                );
              })}
            </div>
            <div className="lane-heat" aria-label={`GPU ${gpu.index} 사용률`}>
              {data.timestamps.map((ts, i) => {
                const util = gpu.util[i];
                if (util == null || util < 1) return null;
                return (
                  <i
                    key={ts}
                    style={{
                      left: `${x(ts)}%`,
                      width: `${cell}%`,
                      background: `color-mix(in srgb, var(--text) ${Math.round(12 + util * 0.6)}%, transparent)`,
                    }}
                    title={`${time(ts, days)} · 사용률 ${Math.round(util)}%`}
                  />
                );
              })}
            </div>
          </div>
        </div>
      ))}
      {data.gpus.length > 0 && (
        <div className="lane-axis">
          <span />
          <div>
            {ticks.map((t) => (
              <span key={t} style={{ left: `${x(t)}%` }}>
                {time(t, days)}
              </span>
            ))}
            <span className="now">지금</span>
          </div>
        </div>
      )}
      {data.gpus.length > 0 && (
        <div className="board-legend muted small">
          <span>
            <i className="swatch job" /> BADAMOA 실행
          </span>
          <span>
            <i className="swatch busy" /> 사람
          </span>
          <span>아래 띠: 사용률 (진할수록 높음)</span>
        </div>
      )}
    </section>
  );
}
