import { type ReactNode, createContext, useContext, useMemo, useState } from "react";

export interface Series {
  key: string;
  label: string;
  values: (number | null)[];
}

interface Props {
  title: string;
  subtitle?: string;
  unit: string;
  /** Fixed top of the y axis (e.g. 100 for percentages); otherwise a rounded maximum. */
  yMax?: number;
  timestamps: string[];
  from: string;
  to: string;
  stepSeconds: number;
  series: Series[];
  /** Drag across the plot: zoom into that time (milliseconds). Without it the plot does not zoom. */
  onZoom?: (from: number, to: number) => void;
  /** Double-click: go back from a zoom. */
  onReset?: () => void;
}

const W = 600;
const H = 210;
const PAD = { top: 10, right: 12, bottom: 24, left: 40 };
const MAX_SERIES = 8; // the palette has eight validated slots; more would need another form

function niceMax(value: number): number {
  if (value <= 0) return 1;
  const power = 10 ** Math.floor(Math.log10(value));
  for (const m of [1, 2, 2.5, 5, 10]) if (value <= m * power) return m * power;
  return 10 * power;
}

function fmt(value: number | null, unit: string): string {
  if (value == null) return "–";
  const digits = Math.abs(value) >= 100 || Number.isInteger(value) ? 0 : 1;
  return `${value.toFixed(digits)}${unit === "%" || unit === "°C" ? unit : ` ${unit}`}`;
}

function stamp(ms: number): string {
  const d = new Date(ms);
  const two = (n: number) => String(n).padStart(2, "0");
  return `${d.getMonth() + 1}/${d.getDate()} ${two(d.getHours())}:${two(d.getMinutes())}`;
}

function timeLabel(ms: number, spanMs: number): string {
  const d = new Date(ms);
  const hm = `${String(d.getHours()).padStart(2, "0")}:${String(d.getMinutes()).padStart(2, "0")}`;
  if (spanMs <= 36 * 3600_000) return hm;
  return `${d.getMonth() + 1}/${d.getDate()}`;
}

// ---- the moment under the mouse, shared by every chart on a page (Grafana's shared crosshair) ----

interface Crosshair {
  time: number | null;
  source: string | null; // the chart the mouse is over: only it shows the tooltip
  set: (time: number | null, source: string | null) => void;
}

const CrosshairContext = createContext<Crosshair | null>(null);

export function CrosshairProvider({ children }: { children: ReactNode }) {
  const [state, setState] = useState<{ time: number | null; source: string | null }>({ time: null, source: null });
  const value = useMemo(
    () => ({ ...state, set: (time: number | null, source: string | null) => setState({ time, source }) }),
    [state],
  );
  return <CrosshairContext.Provider value={value}>{children}</CrosshairContext.Provider>;
}

export function useCrosshair(): Crosshair | null {
  return useContext(CrosshairContext);
}

/**
 * Multi-series line chart: one y axis, crosshair tooltip, legend with latest values, table view.
 *
 * Grafana-style handling: drag across the plot to zoom into that time (``onZoom``), double-click to
 * go back (``onReset``); inside a ``CrosshairProvider`` every chart shows a line at the moment the
 * mouse is over in any of them; a legend entry shows only that series (Ctrl/Cmd adds or removes one).
 */
export function LineChart({ title, subtitle, unit, yMax, timestamps, from, to, stepSeconds, series, onZoom, onReset }: Props) {
  const shared = useCrosshair();
  const [localTime, setLocalTime] = useState<number | null>(null);
  const [asTable, setAsTable] = useState(false);
  const [only, setOnly] = useState<Set<string> | null>(null); // null: every series
  const [drag, setDrag] = useState<{ start: number; end: number } | null>(null);
  const shown = series.slice(0, MAX_SERIES);
  const visible = (s: Series) => !only || only.has(s.key);

  const times = useMemo(() => timestamps.map((t) => new Date(t).getTime()), [timestamps]);
  const t0 = new Date(from).getTime();
  const t1 = Math.max(new Date(to).getTime(), t0 + 1);
  const top =
    yMax ?? niceMax(Math.max(0, ...shown.filter(visible).flatMap((s) => s.values.filter((v): v is number => v != null))));
  const x = (t: number) => PAD.left + ((t - t0) / (t1 - t0)) * (W - PAD.left - PAD.right);
  const y = (v: number) => PAD.top + (1 - Math.min(v, top) / top) * (H - PAD.top - PAD.bottom);

  // The sample nearest the shared moment, if that moment is inside this chart's range.
  const time = shared ? shared.time : localTime;
  const hover = useMemo(() => {
    if (time == null || times.length === 0) return null;
    if (time < t0 - stepSeconds * 1000 || time > t1 + stepSeconds * 1000) return null;
    let best = 0;
    for (let i = 1; i < times.length; i++) if (Math.abs(times[i] - time) < Math.abs(times[best] - time)) best = i;
    return best;
  }, [time, times, t0, t1, stepSeconds]);
  const own = shared ? shared.source === title : localTime != null;

  const paths = shown.map((s) => {
    let d = "";
    let pen = false;
    s.values.forEach((v, i) => {
      const gap = i > 0 && times[i] - times[i - 1] > stepSeconds * 2500; // missing buckets break the line
      if (v == null) {
        pen = false;
        return;
      }
      d += `${pen && !gap ? "L" : "M"}${x(times[i]).toFixed(1)},${y(v).toFixed(1)}`;
      pen = true;
    });
    return d;
  });

  // A sample with no neighbour on either side has no line segment; mark it so it stays visible.
  const lonely = shown.map((s) =>
    s.values.flatMap((v, i) => {
      const joined = (a: number, b: number) =>
        s.values[a] != null && s.values[b] != null && times[b] - times[a] <= stepSeconds * 2500;
      const alone = v != null && !(i > 0 && joined(i - 1, i)) && !(i < times.length - 1 && joined(i, i + 1));
      return alone ? [i] : [];
    }),
  );

  const yTicks = [0, 0.25, 0.5, 0.75, 1].map((f) => f * top);
  const xTicks = [0, 0.25, 0.5, 0.75, 1].map((f) => t0 + f * (t1 - t0));
  const last = (s: Series) => [...s.values].reverse().find((v) => v != null) ?? null;

  const timeAt = (e: React.MouseEvent<SVGSVGElement>) => {
    const box = e.currentTarget.getBoundingClientRect();
    const t = t0 + ((((e.clientX - box.left) / box.width) * W - PAD.left) / (W - PAD.left - PAD.right)) * (t1 - t0);
    return Math.min(t1, Math.max(t0, t));
  };
  const point = (t: number | null) => (shared ? shared.set(t, t == null ? null : title) : setLocalTime(t));
  const onMove = (e: React.MouseEvent<SVGSVGElement>) => {
    if (times.length === 0) return;
    const t = timeAt(e);
    point(t);
    if (drag) setDrag({ ...drag, end: t });
  };
  const onDown = (e: React.MouseEvent<SVGSVGElement>) => {
    if (!onZoom || e.button !== 0) return;
    e.preventDefault();
    const t = timeAt(e);
    setDrag({ start: t, end: t });
  };
  const onUp = () => {
    if (drag && onZoom && Math.abs(x(drag.end) - x(drag.start)) > 6) {
      onZoom(Math.min(drag.start, drag.end), Math.max(drag.start, drag.end));
    }
    setDrag(null);
  };
  const pick = (key: string, add: boolean) => {
    if (add) {
      const next = new Set(only ?? shown.map((s) => s.key));
      if (next.has(key)) next.delete(key);
      else next.add(key);
      setOnly(next.size === 0 || next.size === shown.length ? null : next);
    } else setOnly(only && only.size === 1 && only.has(key) ? null : new Set([key]));
  };

  const tableRows = useMemo(() => {
    const step = Math.max(1, Math.ceil(times.length / 24));
    return times.map((_, i) => i).filter((i) => i % step === 0 || i === times.length - 1).reverse();
  }, [times]);

  return (
    <section className="card chart">
      <div className="card-head">
        <div>
          <h3>{title}</h3>
          {subtitle && <div className="muted small">{subtitle}</div>}
        </div>
        <button className="ghost-btn" onClick={() => setAsTable(!asTable)} aria-pressed={asTable}>
          {asTable ? "그래프 보기" : "표 보기"}
        </button>
      </div>

      <div className="legend">
        {shown.map((s, i) => (
          <button
            key={s.key}
            type="button"
            className={`legend-item ${visible(s) ? "" : "off"}`}
            onClick={(e) => pick(s.key, e.ctrlKey || e.metaKey)}
            title="누르면 이것만 보기 · Ctrl(⌘)과 함께 누르면 더하거나 빼기"
          >
            <i style={{ background: `var(--series-${i + 1})` }} />
            {s.label}
            <strong>{fmt(hover != null ? s.values[hover] : last(s), unit)}</strong>
          </button>
        ))}
        {series.length > MAX_SERIES && <span className="muted small">외 {series.length - MAX_SERIES}개는 표에서 확인</span>}
      </div>

      {times.length === 0 ? (
        <div className="chart-empty">이 기간의 기록이 아직 없습니다.</div>
      ) : asTable ? (
        <div className="chart-table">
          <table>
            <thead>
              <tr>
                <th>시각</th>
                {series.map((s) => (
                  <th key={s.key} className="right">
                    {s.label}
                  </th>
                ))}
              </tr>
            </thead>
            <tbody>
              {tableRows.map((i) => (
                <tr key={i}>
                  <td className="muted">{stamp(times[i])}</td>
                  {series.map((s) => (
                    <td key={s.key} className="right num">
                      {fmt(s.values[i], unit)}
                    </td>
                  ))}
                </tr>
              ))}
            </tbody>
          </table>
        </div>
      ) : (
        <div
          className={`chart-plot ${onZoom ? "zoomable" : ""}`}
          title={onZoom ? "드래그해서 확대 · 더블클릭으로 되돌리기" : undefined}
        >
          <svg
            viewBox={`0 0 ${W} ${H}`}
            role="img"
            aria-label={`${title} 그래프`}
            onMouseMove={onMove}
            onMouseDown={onDown}
            onMouseUp={onUp}
            onDoubleClick={() => onReset?.()}
            onMouseLeave={() => {
              point(null);
              setDrag(null);
            }}
          >
            {yTicks.map((v) => (
              <g key={v}>
                <line className="grid-line" x1={PAD.left} x2={W - PAD.right} y1={y(v)} y2={y(v)} />
                <text className="axis-text" x={PAD.left - 6} y={y(v) + 3.5} textAnchor="end">
                  {Number.isInteger(v) ? v : v.toFixed(1)}
                </text>
              </g>
            ))}
            {xTicks.map((t, i) => (
              <text
                key={t}
                className="axis-text"
                x={x(t)}
                y={H - 6}
                textAnchor={i === 0 ? "start" : i === xTicks.length - 1 ? "end" : "middle"}
              >
                {timeLabel(t, t1 - t0)}
              </text>
            ))}
            {drag && (
              <rect
                className="zoom-band"
                x={Math.min(x(drag.start), x(drag.end))}
                width={Math.abs(x(drag.end) - x(drag.start))}
                y={PAD.top}
                height={H - PAD.top - PAD.bottom}
              />
            )}
            {paths.map((d, i) =>
              visible(shown[i]) ? (
                <path key={shown[i].key} d={d} fill="none" stroke={`var(--series-${i + 1})`} strokeWidth="2" strokeLinejoin="round" />
              ) : null,
            )}
            {lonely.map((indices, si) =>
              visible(shown[si])
                ? indices.map((i) => (
                    <circle
                      key={`${shown[si].key}-${i}`}
                      cx={x(times[i])}
                      cy={y(shown[si].values[i] as number)}
                      r="2.5"
                      fill={`var(--series-${si + 1})`}
                    />
                  ))
                : null,
            )}
            {hover != null && (
              <g>
                <line className="crosshair" x1={x(times[hover])} x2={x(times[hover])} y1={PAD.top} y2={H - PAD.bottom} />
                {shown.map((s, i) => {
                  const v = s.values[hover];
                  return v == null || !visible(s) ? null : (
                    <circle
                      key={s.key}
                      cx={x(times[hover])}
                      cy={y(v)}
                      r="4"
                      fill={`var(--series-${i + 1})`}
                      stroke="var(--surface)"
                      strokeWidth="2"
                    />
                  );
                })}
              </g>
            )}
          </svg>
          {hover != null && own && !drag && (
            <div
              className="chart-tip"
              style={{ left: `${(x(times[hover]) / W) * 100}%` }}
              data-side={x(times[hover]) > W * 0.6 ? "left" : "right"}
            >
              <div className="muted small">{stamp(times[hover])}</div>
              {shown.map((s, i) =>
                visible(s) ? (
                  <div key={s.key} className="tip-row">
                    <i style={{ background: `var(--series-${i + 1})` }} />
                    {s.label}
                    <strong>{fmt(s.values[hover], unit)}</strong>
                  </div>
                ) : null,
              )}
            </div>
          )}
        </div>
      )}
    </section>
  );
}

/** Tiny two-line trend for node cards; the legend beside it carries the labels. */
export function Sparkline({ lines }: { lines: (number | null)[][] }) {
  const w = 160;
  const h = 36;
  const n = Math.max(1, ...lines.map((l) => l.length));
  return (
    <svg className="sparkline" viewBox={`0 0 ${w} ${h}`} preserveAspectRatio="none" aria-hidden="true">
      {lines.map((values, li) => {
        let d = "";
        let pen = false;
        values.forEach((v, i) => {
          if (v == null) {
            pen = false;
            return;
          }
          const px = n > 1 ? (i / (n - 1)) * w : w;
          d += `${pen ? "L" : "M"}${px.toFixed(1)},${(h - 2 - (Math.min(v, 100) / 100) * (h - 4)).toFixed(1)}`;
          pen = true;
        });
        return <path key={li} d={d} fill="none" stroke={`var(--series-${li + 1})`} strokeWidth="1.5" />;
      })}
    </svg>
  );
}
