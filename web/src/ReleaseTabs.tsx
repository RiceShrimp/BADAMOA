import { useMemo, useState } from "react";
import { METRIC_NAME, SPLIT_COLOR } from "./DataBoard";
import { api } from "./api";
import { type ReleaseInfo, SPLIT_LABEL, STATE_LABEL, type TreeDataset, bytesLabel, classColor } from "./catalogTypes";
import { metric } from "./trainTypes";
import { Icon, timeAgo, usePolling } from "./ui";

interface Charts {
  splits: Record<string, number>;
  states: Record<string, number>;
  classes: { id: number; name: string; color?: string | null; boxes: number; images: number }[];
  sizes: {
    images: number;
    mean_width: number | null;
    mean_height: number | null;
    width: { edges: number[]; counts: number[] };
    height: { edges: number[]; counts: number[] };
    grid: number[][];
    max: number;
  };
  boxes: { sampled_items: number; boxes: number; centres: number[][]; shapes: number[][] };
}

export const colorFor = (classes: { id: number; color?: string | null }[]) => (id: number) =>
  classes.find((c) => c.id === id)?.color ?? classColor(id);

// ---- classes --------------------------------------------------------------------------

/** Every class of the release: colour (click to change), index, name, boxes and images, share of the boxes. */
export function ClassesTab({ release, onChanged }: { release: ReleaseInfo; onChanged: () => void }) {
  const [q, setQ] = useState("");
  const [sort, setSort] = useState<{ key: "id" | "name" | "boxes" | "items"; dir: 1 | -1 }>({ key: "boxes", dir: -1 });
  const [error, setError] = useState<string | null>(null);
  const stats = release.stats.classes ?? {};
  const total = Math.max(1, Object.values(stats).reduce((n, c) => n + c.boxes, 0));
  const rows = useMemo(() => {
    const needle = q.trim().toLowerCase();
    const list = release.label_space.classes
      .map((c) => ({ ...c, boxes: stats[String(c.id)]?.boxes ?? 0, items: stats[String(c.id)]?.items ?? 0 }))
      .filter((c) => !needle || c.name.toLowerCase().includes(needle) || String(c.id) === needle);
    return list.sort((a, b) => {
      const x = a[sort.key];
      const y = b[sort.key];
      return sort.dir * (typeof x === "number" && typeof y === "number" ? x - y : String(x).localeCompare(String(y), "ko"));
    });
  }, [release, q, sort, stats]);
  const head = (key: typeof sort.key, label: string, right = false) => (
    <th
      className={`sortable ${right ? "right" : ""}`}
      onClick={() => setSort(sort.key === key ? { key, dir: sort.dir === 1 ? -1 : 1 } : { key, dir: key === "name" || key === "id" ? 1 : -1 })}
    >
      {label}
      {sort.key === key ? (sort.dir === 1 ? " ↑" : " ↓") : ""}
    </th>
  );
  const paint = async (id: number, color: string) => {
    try {
      setError(null);
      await api(`/releases/${release.id}/classes/${id}`, { method: "PUT", body: { color } });
      onChanged();
    } catch (e) {
      setError(e instanceof Error ? e.message : String(e));
    }
  };
  return (
    <section className="card">
      <div className="card-head">
        <div>
          <h3>클래스</h3>
          <div className="muted small">
            {release.label_space.name} · 클래스 {release.label_space.classes.length}개 · {Object.keys(stats).length}개가 이 버전에 나옴
          </div>
        </div>
        <input type="search" value={q} onChange={(e) => setQ(e.target.value)} placeholder="클래스 찾기" aria-label="클래스 찾기" />
      </div>
      {error && <div className="error">{error}</div>}
      <div className="table-wrap">
        <table className="plain class-table">
          <thead>
            <tr>
              <th aria-label="색" />
              {head("id", "번호")}
              {head("name", "이름")}
              {head("boxes", "라벨", true)}
              {head("items", "이미지", true)}
              <th>비율</th>
            </tr>
          </thead>
          <tbody>
            {rows.map((c) => {
              const color = c.color ?? classColor(c.id);
              return (
                <tr key={c.id}>
                  <td>
                    <label className="swatch-pick" title="색 바꾸기 (화면에만 쓰임)">
                      <i style={{ background: color }} />
                      <input type="color" value={toHex(color)} onChange={(e) => void paint(c.id, e.target.value)} />
                    </label>
                  </td>
                  <td className="num muted">{c.id}</td>
                  <td>
                    <a href={`#/datasets/${release.id}/images?class=${c.id}`}>{c.name}</a>
                    {c.color && (
                      <button className="link small reset-color" onClick={() => void paint(c.id, "")}>
                        기본 색
                      </button>
                    )}
                  </td>
                  <td className="num right">{c.boxes.toLocaleString()}</td>
                  <td className="num right">{c.items.toLocaleString()}</td>
                  <td>
                    <span className="share-bar">
                      <i style={{ width: `${(c.boxes / total) * 100}%`, background: color }} />
                    </span>
                  </td>
                </tr>
              );
            })}
          </tbody>
        </table>
      </div>
      <p className="muted small">
        색은 화면에만 쓰이고 라벨이나 버전은 바뀌지 않습니다. 클래스를 합치거나 이름을 바꾸는 것은 새 버전을 만드는 방식으로 다음에 넣습니다.
      </p>
    </section>
  );
}

/** ``<input type="color">`` takes #rrggbb only; the automatic colours are hsl(). */
function toHex(color: string): string {
  if (color.startsWith("#")) return color;
  const m = color.match(/hsl\(([\d.]+)\s+([\d.]+)%\s+([\d.]+)%\)/);
  if (!m) return "#888888";
  const [h, s, l] = [Number(m[1]), Number(m[2]) / 100, Number(m[3]) / 100];
  const k = (n: number) => (n + h / 30) % 12;
  const a = s * Math.min(l, 1 - l);
  const f = (n: number) => l - a * Math.max(-1, Math.min(k(n) - 3, Math.min(9 - k(n), 1)));
  return `#${[f(0), f(8), f(4)].map((v) => Math.round(v * 255).toString(16).padStart(2, "0")).join("")}`;
}

// ---- charts ---------------------------------------------------------------------------

function Donut({ parts }: { parts: { label: string; value: number; color: string }[] }) {
  const total = parts.reduce((n, p) => n + p.value, 0) || 1;
  let start = 0;
  const r = 40;
  const c = 2 * Math.PI * r;
  return (
    <div className="donut">
      <svg viewBox="0 0 100 100" aria-hidden="true">
        {parts.map((p) => {
          const len = (p.value / total) * c;
          const el = (
            <circle
              key={p.label}
              cx="50"
              cy="50"
              r={r}
              fill="none"
              stroke={p.color}
              strokeWidth="14"
              strokeDasharray={`${Math.max(0, len - 1.5)} ${c}`}
              strokeDashoffset={-start}
              transform="rotate(-90 50 50)"
            />
          );
          start += len;
          return el;
        })}
      </svg>
      <ul>
        {parts.map((p) => (
          <li key={p.label}>
            <i style={{ background: p.color }} />
            {p.label}
            <strong>{p.value.toLocaleString()}</strong>
            <span className="muted">{((p.value / total) * 100).toFixed(1)}%</span>
          </li>
        ))}
      </ul>
    </div>
  );
}

function Heat({ grid, label, note, lines }: { grid: number[][]; label: (x: number, y: number) => string; note: string; lines?: boolean }) {
  const top = Math.max(1, ...grid.flat());
  const n = grid.length;
  return (
    <div className="heat">
      <svg viewBox={`0 0 ${n} ${n}`} preserveAspectRatio="none" aria-hidden="true">
        {grid.map((row, y) =>
          row.map((v, x) =>
            v ? (
              <rect key={`${x}-${y}`} x={x} y={y} width="1" height="1" fill="var(--accent)" fillOpacity={0.15 + 0.85 * (v / top)}>
                <title>{`${label(x, y)} · ${v.toLocaleString()}`}</title>
              </rect>
            ) : null,
          ),
        )}
        {lines &&
          [1, 4 / 3, 16 / 9].map((ratio) => (
            <line key={ratio} x1="0" y1="0" x2={n} y2={n / ratio} className="aspect-line" vectorEffect="non-scaling-stroke" />
          ))}
      </svg>
      <div className="muted small">{note}</div>
    </div>
  );
}

/** Split and label states, classes, image sizes, where the boxes sit and how big they are. */
export function ChartsTab({ release }: { release: ReleaseInfo }) {
  const { data, error } = usePolling(() => api<Charts>(`/releases/${release.id}/charts`), 0, [release.id]);
  if (error && !data) return <div className="error">{error}</div>;
  if (!data) return <p className="muted">통계를 계산하는 중…</p>;
  const splits = ["train", "val", "test", ""].filter((s) => data.splits[s]);
  const classes = [...data.classes].filter((c) => c.boxes > 0).sort((a, b) => b.boxes - a.boxes);
  const topBoxes = Math.max(1, ...classes.map((c) => c.boxes));
  const shown = classes.slice(0, 30);
  const s = data.sizes;
  const sizeBars = Math.max(1, ...s.width.counts, ...s.height.counts);
  const step = s.max / data.sizes.grid.length;
  return (
    <div className="chart-cards">
      <section className="card">
        <h3>split</h3>
        <div className="muted small">이미지 {Object.values(data.splits).reduce((a, b) => a + b, 0).toLocaleString()}장</div>
        <Donut parts={splits.map((k) => ({ label: SPLIT_LABEL[k] ?? k, value: data.splits[k], color: SPLIT_COLOR[k] }))} />
      </section>
      <section className="card">
        <h3>라벨 상태</h3>
        <div className="muted small">미라벨은 학습에서 빠집니다 (배경으로 넣도록 고를 수 있음)</div>
        <Donut
          parts={[
            { label: STATE_LABEL.labelled, value: data.states.labelled ?? 0, color: "var(--series-3)" },
            { label: STATE_LABEL.background, value: data.states.background ?? 0, color: "var(--series-1)" },
            { label: STATE_LABEL.unlabelled, value: data.states.unlabelled ?? 0, color: "var(--idle)" },
          ].filter((p) => p.value > 0)}
        />
      </section>
      <section className="card wide">
        <h3>클래스별 라벨 수</h3>
        <div className="muted small">
          {classes.length}개 클래스 · 라벨 {classes.reduce((n, c) => n + c.boxes, 0).toLocaleString()}개
          {classes.length > shown.length ? ` · 많은 순 ${shown.length}개` : ""}
        </div>
        <div className="hbars">
          {shown.map((c) => (
            <div key={c.id} className="hbar">
              <span className="hbar-name" title={c.name}>
                {c.name}
              </span>
              <span className="hbar-track">
                <i style={{ width: `${(c.boxes / topBoxes) * 100}%`, background: c.color ?? classColor(c.id) }} />
              </span>
              <span className="num">{c.boxes.toLocaleString()}</span>
              <span className="num muted">{c.images.toLocaleString()}장</span>
            </div>
          ))}
        </div>
      </section>
      <section className="card">
        <h3>이미지 크기</h3>
        <div className="muted small">
          {s.images.toLocaleString()}장 · 평균 {s.mean_width ?? "–"} × {s.mean_height ?? "–"} px
        </div>
        <div className="vbars">
          {s.width.counts.map((count, i) => (
            <div key={i} className="vbar-group" title={`${s.width.edges[i]}~${s.width.edges[i + 1]} px`}>
              <i className="w" style={{ height: `${(count / sizeBars) * 100}%` }} />
              <i className="h" style={{ height: `${((s.height.counts[i] ?? 0) / sizeBars) * 100}%` }} />
            </div>
          ))}
        </div>
        <div className="vbar-axis muted small">
          <span>가로 {s.width.edges[0]}~{s.width.edges.at(-1)} px</span>
          <span>세로 {s.height.edges[0]}~{s.height.edges.at(-1)} px</span>
        </div>
        <div className="split-legend">
          <span>
            <i className="w-swatch" />
            가로
          </span>
          <span>
            <i className="h-swatch" />
            세로
          </span>
        </div>
      </section>
      <section className="card">
        <h3>가로 × 세로</h3>
        <div className="muted small">진할수록 그 크기의 이미지가 많음 · 선은 1:1 · 4:3 · 16:9</div>
        <Heat
          grid={s.grid}
          lines
          label={(x, y) => `가로 ~${Math.round((x + 1) * step)} · 세로 ~${Math.round((y + 1) * step)} px`}
          note={`한 칸 ${Math.round(step)} px · 최대 ${s.max} px`}
        />
      </section>
      <section className="card">
        <h3>박스 중심 위치</h3>
        <div className="muted small">이미지 안 어디에 물체가 있는지</div>
        <Heat
          grid={data.boxes.centres}
          label={(x, y) => `가로 ${x * 5}~${x * 5 + 5}% · 세로 ${y * 5}~${y * 5 + 5}%`}
          note={`박스 ${data.boxes.boxes.toLocaleString()}개 (이미지 ${data.boxes.sampled_items.toLocaleString()}장에서)`}
        />
      </section>
      <section className="card">
        <h3>박스 크기</h3>
        <div className="muted small">이미지 대비 박스의 가로(→) · 세로(↓)</div>
        <Heat
          grid={data.boxes.shapes}
          label={(x, y) => `가로 ${x * 5}~${x * 5 + 5}% · 세로 ${y * 5}~${y * 5 + 5}%`}
          note="왼쪽 위로 몰릴수록 작은 물체"
        />
      </section>
    </div>
  );
}

// ---- models trained on it ---------------------------------------------------------------

interface TrainedOn {
  run_id: string;
  name: string;
  template: string;
  epochs: number;
  created_at: string;
  metric: string;
  value: number | null;
  model: { id: string; name: string; aliases: string[] } | null;
}

export function ModelsTab({ release }: { release: ReleaseInfo }) {
  const { data, error } = usePolling(() => api<TrainedOn[]>(`/releases/${release.id}/models`), 30000, [release.id]);
  if (error && !data) return <div className="error">{error}</div>;
  if (!data) return <p className="muted">불러오는 중…</p>;
  if (data.length === 0)
    return (
      <div className="card empty-state">
        <Icon name="flow" size={26} />
        <strong>아직 이 버전으로 학습한 모델이 없습니다</strong>
        <p className="muted">학습하면 이 버전과 모델이 이어져, 어떤 데이터로 만든 모델인지 언제든 거슬러 볼 수 있습니다.</p>
        <a className="btn primary-link" href={`#/training/new/${release.id}`}>
          이 버전으로 학습
        </a>
      </div>
    );
  return (
    <section className="card table-wrap">
      <table className="plain">
        <thead>
          <tr>
            <th>학습 실행</th>
            <th>모델 종류</th>
            <th className="right">epoch</th>
            <th className="right">{METRIC_NAME[data[0].metric] ?? data[0].metric}</th>
            <th>등록된 모델</th>
            <th className="right">언제</th>
          </tr>
        </thead>
        <tbody>
          {data.map((r) => (
            <tr key={r.run_id}>
              <td>
                <a href={`#/training/${r.run_id}`}>{r.name}</a>
              </td>
              <td className="muted">{r.template}</td>
              <td className="num right">{r.epochs}</td>
              <td className="num right">{metric(r.value)}</td>
              <td>
                {r.model ? <a href={`#/models/${r.model.id}`}>{r.model.name}</a> : <span className="muted">–</span>}
                {r.model?.aliases.map((a) => (
                  <span key={a} className="tag">
                    {a}
                  </span>
                ))}
              </td>
              <td className="muted right">{timeAgo(r.created_at)}</td>
            </tr>
          ))}
        </tbody>
      </table>
    </section>
  );
}

// ---- the family's releases ---------------------------------------------------------------

export function ReleasesTab({ release, onCompare }: { release: ReleaseInfo; onCompare: (otherId: string) => void }) {
  const { data: tree } = usePolling(() => api<TreeDataset[]>(`/projects/${release.project_id}/datasets`), 30000, [release.project_id]);
  const family = tree?.find((d) => d.id === release.dataset.id);
  if (!family) return <p className="muted">불러오는 중…</p>;
  return (
    <section className="card table-wrap">
      <table className="plain">
        <thead>
          <tr>
            <th>버전</th>
            <th className="right">항목</th>
            <th>라벨 공간</th>
            <th className="right">만든 때</th>
            <th />
          </tr>
        </thead>
        <tbody>
          {family.releases.map((r) => (
            <tr key={r.id} className={r.id === release.id ? "current-row" : ""}>
              <td>
                <a href={`#/datasets/${r.id}`}>{r.version}</a>
                {r.id === release.id && <span className="tag">지금 보는 버전</span>}
                {r.status === "building" && <span className="tag">등록 중</span>}
              </td>
              <td className="num right">{r.items.toLocaleString()}</td>
              <td className="muted">{r.label_space}</td>
              <td className="muted right">{timeAgo(r.created_at)}</td>
              <td className="right">
                {r.id !== release.id && (
                  <button className="link small" onClick={() => onCompare(r.id)}>
                    {release.version}와 비교
                  </button>
                )}
              </td>
            </tr>
          ))}
        </tbody>
      </table>
      <p className="muted small">버전은 만든 뒤 바뀌지 않습니다. 고치려면 "새 버전"으로 새로 만드세요. 용량 {bytesLabel(release.stats.bytes)}.</p>
    </section>
  );
}
