import { useState } from "react";

export interface EpochSeries {
  key: string;
  label: string;
  /** Palette slot 1..8; the same run keeps the same colour in every chart. */
  slot: number;
  steps: number[];
  values: (number | null)[];
}

const W = 420;
const H = 170;
const PAD = { top: 10, right: 10, bottom: 22, left: 38 };

function niceMax(value: number): number {
  if (value <= 0) return 1;
  const power = 10 ** Math.floor(Math.log10(value));
  for (const m of [1, 2, 2.5, 5, 10]) if (value <= m * power) return m * power;
  return 10 * power;
}

function tick(value: number): string {
  if (value === 0) return "0";
  const size = Math.abs(value);
  if (size < 0.01) return value.toExponential(0); // learning rates and the like
  return size >= 10 ? value.toFixed(0) : size >= 1 ? value.toFixed(1) : value.toFixed(2);
}

/** A value in a legend or tooltip: three decimals, or scientific notation when that would show only zeros. */
function shown(value: number | null | undefined): string {
  if (value == null) return "–";
  return value !== 0 && Math.abs(value) < 0.001 ? value.toExponential(2) : value.toFixed(3);
}

/** A small line chart over epochs. Several of these side by side compare runs metric by metric. */
export function EpochChart({ title, series, legend = false }: { title: string; series: EpochSeries[]; legend?: boolean }) {
  const [hover, setHover] = useState<number | null>(null);
  const lastStep = Math.max(1, ...series.flatMap((s) => s.steps));
  const numbers = series.flatMap((s) => s.values.filter((v): v is number => v != null));
  const top = niceMax(Math.max(0, ...numbers));
  const x = (step: number) => PAD.left + (lastStep === 1 ? 0.5 : (step - 1) / (lastStep - 1)) * (W - PAD.left - PAD.right);
  const y = (v: number) => PAD.top + (1 - Math.min(v, top) / top) * (H - PAD.top - PAD.bottom);
  const at = (s: EpochSeries, step: number) => {
    const i = s.steps.indexOf(step);
    return i < 0 ? null : s.values[i];
  };

  const onMove = (e: React.MouseEvent<SVGSVGElement>) => {
    const box = e.currentTarget.getBoundingClientRect();
    const ratio = (((e.clientX - box.left) / box.width) * W - PAD.left) / (W - PAD.left - PAD.right);
    setHover(Math.min(lastStep, Math.max(1, Math.round(1 + ratio * (lastStep - 1)))));
  };

  const xTicks = lastStep <= 4 ? Array.from({ length: lastStep }, (_, i) => i + 1) : [1, Math.round(lastStep / 2), lastStep];

  return (
    <section className="card epoch-chart">
      <div className="epoch-head">
        <h3>{title}</h3>
        {hover != null && <span className="muted small">epoch {hover}</span>}
      </div>
      {legend && (
        <div className="legend">
          {series.map((s) => (
            <span key={s.key} className="legend-item">
              <i style={{ background: `var(--series-${s.slot})` }} />
              {s.label}
              <strong>{shown(at(s, hover ?? s.steps[s.steps.length - 1]))}</strong>
            </span>
          ))}
        </div>
      )}
      {numbers.length === 0 ? (
        <div className="chart-empty">아직 기록이 없습니다.</div>
      ) : (
        <div className="chart-plot">
          <svg viewBox={`0 0 ${W} ${H}`} role="img" aria-label={`${title} 그래프`} onMouseMove={onMove} onMouseLeave={() => setHover(null)}>
            {[0, 0.5, 1].map((f) => (
              <g key={f}>
                <line className="grid-line" x1={PAD.left} x2={W - PAD.right} y1={y(f * top)} y2={y(f * top)} />
                <text className="axis-text" x={PAD.left - 6} y={y(f * top) + 3.5} textAnchor="end">
                  {tick(f * top)}
                </text>
              </g>
            ))}
            {xTicks.map((step, i) => (
              <text
                key={step}
                className="axis-text"
                x={x(step)}
                y={H - 5}
                textAnchor={lastStep === 1 ? "middle" : i === 0 ? "start" : i === xTicks.length - 1 ? "end" : "middle"}
              >
                {step}
              </text>
            ))}
            {series.map((s) => {
              let d = "";
              let pen = false;
              s.values.forEach((v, i) => {
                if (v == null) {
                  pen = false;
                  return;
                }
                d += `${pen ? "L" : "M"}${x(s.steps[i]).toFixed(1)},${y(v).toFixed(1)}`;
                pen = true;
              });
              const only = s.values.filter((v) => v != null).length === 1;
              return (
                <g key={s.key}>
                  <path d={d} fill="none" stroke={`var(--series-${s.slot})`} strokeWidth="2" strokeLinejoin="round" />
                  {only && <circle cx={x(s.steps[0])} cy={y(s.values[0] as number)} r="3" fill={`var(--series-${s.slot})`} />}
                </g>
              );
            })}
            {hover != null && (
              <g>
                <line className="crosshair" x1={x(hover)} x2={x(hover)} y1={PAD.top} y2={H - PAD.bottom} />
                {series.map((s) => {
                  const v = at(s, hover);
                  return v == null ? null : (
                    <circle key={s.key} cx={x(hover)} cy={y(v)} r="3.5" fill={`var(--series-${s.slot})`} stroke="var(--surface)" strokeWidth="2" />
                  );
                })}
              </g>
            )}
          </svg>
          {hover != null && !legend && (
            <div className="chart-tip" style={{ left: `${(x(hover) / W) * 100}%` }} data-side={x(hover) > W * 0.55 ? "left" : "right"}>
              {series.map((s) => (
                <div key={s.key} className="tip-row">
                  <i style={{ background: `var(--series-${s.slot})` }} />
                  {s.label}
                  <strong>{shown(at(s, hover))}</strong>
                </div>
              ))}
            </div>
          )}
        </div>
      )}
    </section>
  );
}
