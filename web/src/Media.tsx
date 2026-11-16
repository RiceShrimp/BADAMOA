import { useEffect, useMemo, useState } from "react";
import { classColor } from "./catalogTypes";
import { metric } from "./trainTypes";

export interface MediaFile {
  name: string;
  kind: "image" | "table" | "file";
  caption: string;
  url: string;
}

export interface RunMedia {
  key: string;
  steps: { step: number; files: MediaFile[] }[];
}

/** A table cell that draws on an image (``bm.overlay``): coordinates are fractions of the image, as in YOLO labels. */
interface Overlay {
  image: string;
  boxes?: number[][]; // [class, score, cx, cy, w, h]
  truth?: number[][]; // [class, cx, cy, w, h]
  outlines?: [number, number, number[]][]; // [class, score, [x1, y1, x2, y2, …]]
  truth_outlines?: [number, number[]][];
}

interface Table {
  columns: string[];
  rows: unknown[][];
  classes?: string[];
}

interface Drawing {
  truth: boolean;
  minScore: number;
  names: string[];
}

/** How a key's tables are shown; kept while moving through the steps. */
interface ViewChoice {
  sort: { col: number; dir: 1 | -1 } | null;
  filter: string;
  asTable: boolean;
  truth: boolean;
  minScore: number;
}

const FIRST_VIEW: ViewChoice = { sort: null, filter: "", asTable: false, truth: false, minScore: 0 };

const SHOWN_ROWS = 200;

const isOverlay = (value: unknown): value is Overlay =>
  typeof value === "object" && value !== null && typeof (value as Overlay).image === "string";

/** Per-step pictures and tables of a run (``bm.log_image(…, key=)``, ``bm.log_table``): one card per key. */
export function MediaPanel({ media, images }: { media: RunMedia[]; images: Record<string, string> }) {
  if (media.length === 0) return null;
  return (
    <>
      {media.map((entry) => (
        <MediaKey key={entry.key} entry={entry} images={images} />
      ))}
    </>
  );
}

function MediaKey({ entry, images }: { entry: RunMedia; images: Record<string, string> }) {
  const [picked, setPicked] = useState<number | null>(null); // null: follow the newest step
  const [view, setView] = useState<ViewChoice>(FIRST_VIEW);
  const last = entry.steps.length - 1;
  const index = picked === null ? last : Math.min(picked, last);
  const at = entry.steps[index];
  const pictures = at.files.filter((f) => f.kind === "image");
  const tables = at.files.filter((f) => f.kind === "table");
  return (
    <section className="card media-card">
      <div className="card-head">
        <h3>{entry.key}</h3>
        <span className="muted small">
          step {at.step}
          {entry.steps.length > 1 ? ` · ${index + 1}/${entry.steps.length}` : ""}
        </span>
      </div>
      {entry.steps.length > 1 && (
        <input
          type="range"
          className="step-slider"
          min={0}
          max={last}
          value={index}
          onChange={(e) => setPicked(Number(e.target.value) === last ? null : Number(e.target.value))}
          aria-label={`${entry.key} step`}
        />
      )}
      {pictures.length > 0 && (
        <div className="run-pictures">
          {pictures.map((f) => (
            <a key={f.name} href={f.url} target="_blank" rel="noreferrer" title="새 창에서 크게 보기">
              <img src={f.url} alt={f.caption || f.name} loading="lazy" />
              <span>{f.caption || f.name.slice(f.name.lastIndexOf("/") + 1)}</span>
            </a>
          ))}
        </div>
      )}
      {tables.map((f) => (
        <TableView key={f.name} file={f} images={images} view={view} onView={(change) => setView({ ...view, ...change })} />
      ))}
    </section>
  );
}

// ---- pictures with boxes ------------------------------------------------------------------

const pointsOf = (flat: number[], w: number, h: number) => flat.map((v, n) => (n % 2 === 0 ? v * w : v * h)).join(" ");

/** An image with the predicted boxes and outlines (solid) and, when asked, the true ones (dashed) drawn on it. */
function OverlayPicture({
  cell,
  url,
  drawing,
  large = false,
  onOpen,
}: {
  cell: Overlay;
  url: string;
  drawing: Drawing;
  large?: boolean;
  onOpen?: () => void;
}) {
  const [size, setSize] = useState<{ w: number; h: number } | null>(null);
  const { w, h } = size ?? { w: 1, h: 1 };
  const font = Math.max(w, h) / (large ? 55 : 38);
  const outlines = (cell.outlines ?? []).filter((o) => o[1] >= drawing.minScore);
  const boxes = (cell.boxes ?? []).filter((b) => b[1] >= drawing.minScore);
  const name = (cls: number) => drawing.names[cls] ?? String(cls);
  const picture = (
    <>
      <img
        src={url}
        alt={cell.image}
        loading="lazy"
        onLoad={(e) => setSize({ w: e.currentTarget.naturalWidth || 1, h: e.currentTarget.naturalHeight || 1 })}
      />
      {size && (
        <svg viewBox={`0 0 ${w} ${h}`} preserveAspectRatio="none" aria-hidden="true">
          {drawing.truth &&
            (cell.truth_outlines?.length
              ? cell.truth_outlines.map(([cls, flat], i) => (
                  <polygon
                    key={`to${i}`}
                    points={pointsOf(flat, w, h)}
                    fill="none"
                    stroke={classColor(cls)}
                    strokeWidth={2}
                    strokeDasharray="6 4"
                    vectorEffect="non-scaling-stroke"
                  />
                ))
              : (cell.truth ?? []).map(([cls, cx, cy, bw, bh], i) => (
                  <rect
                    key={`t${i}`}
                    x={(cx - bw / 2) * w}
                    y={(cy - bh / 2) * h}
                    width={bw * w}
                    height={bh * h}
                    fill="none"
                    stroke={classColor(cls)}
                    strokeWidth={2}
                    strokeDasharray="6 4"
                    vectorEffect="non-scaling-stroke"
                  />
                )))}
          {outlines.map(([cls, , flat], i) => (
            <polygon
              key={`o${i}`}
              points={pointsOf(flat, w, h)}
              fill={classColor(cls)}
              fillOpacity={0.3}
              stroke={classColor(cls)}
              strokeWidth={2}
              vectorEffect="non-scaling-stroke"
            />
          ))}
          {boxes.map(([cls, score, cx, cy, bw, bh], i) => {
            const x = (cx - bw / 2) * w;
            const y = (cy - bh / 2) * h;
            const label = `${name(cls)} ${score.toFixed(2)}`;
            return (
              <g key={`b${i}`}>
                {outlines.length === 0 && (
                  <rect
                    x={x}
                    y={y}
                    width={bw * w}
                    height={bh * h}
                    fill="none"
                    stroke={classColor(cls)}
                    strokeWidth={large ? 2.5 : 2}
                    vectorEffect="non-scaling-stroke"
                  />
                )}
                {large && (
                  <>
                    <rect
                      x={x}
                      y={Math.max(0, y - font * 1.35)}
                      width={label.length * font * 0.6 + font * 0.6}
                      height={font * 1.35}
                      fill={classColor(cls)}
                    />
                    <text x={x + font * 0.3} y={Math.max(font, y - font * 0.35)} fontSize={font} fill="#fff" fontWeight="600">
                      {label}
                    </text>
                  </>
                )}
              </g>
            );
          })}
        </svg>
      )}
    </>
  );
  return onOpen ? (
    <button type="button" className={`overlay-picture${large ? " large" : ""}`} onClick={onOpen} title="크게 보기">
      {picture}
    </button>
  ) : (
    <div className={`overlay-picture${large ? " large" : ""}`}>{picture}</div>
  );
}

/** One picture at full size, with names and scores on the boxes; ← → move through the shown rows. */
function Lightbox({
  rows,
  column,
  columns,
  at,
  onAt,
  images,
  drawing,
}: {
  rows: unknown[][];
  column: number;
  columns: string[];
  at: number;
  onAt: (n: number | null) => void;
  images: Record<string, string>;
  drawing: Drawing;
}) {
  useEffect(() => {
    const keys = (e: KeyboardEvent) => {
      if (e.key === "Escape") onAt(null);
      else if (e.key === "ArrowRight") onAt(Math.min(rows.length - 1, at + 1));
      else if (e.key === "ArrowLeft") onAt(Math.max(0, at - 1));
    };
    window.addEventListener("keydown", keys);
    return () => window.removeEventListener("keydown", keys);
  }, [at, rows.length, onAt]);
  const row = rows[at];
  const cell = row?.[column];
  if (!row || !isOverlay(cell)) return null;
  return (
    <div className="overlay-lightbox" role="dialog" aria-modal="true" aria-label="예측 크게 보기" onClick={() => onAt(null)}>
      <div className="overlay-lightbox-body" onClick={(e) => e.stopPropagation()}>
        {images[cell.image] ? (
          <OverlayPicture key={cell.image} cell={cell} url={images[cell.image]} drawing={drawing} large />
        ) : (
          <p className="muted">이미지를 찾을 수 없습니다: {cell.image}</p>
        )}
        <div className="overlay-lightbox-foot">
          <button type="button" onClick={() => onAt(Math.max(0, at - 1))} disabled={at === 0} aria-label="이전">
            ‹
          </button>
          <span>
            {at + 1}/{rows.length} · {summary(row, column, columns)}
          </span>
          <button type="button" onClick={() => onAt(Math.min(rows.length - 1, at + 1))} disabled={at >= rows.length - 1} aria-label="다음">
            ›
          </button>
          <button type="button" onClick={() => onAt(null)} aria-label="닫기">
            ✕
          </button>
        </div>
      </div>
    </div>
  );
}

const shown = (value: unknown) =>
  typeof value === "number" ? (Number.isInteger(value) ? value.toLocaleString() : metric(value)) : String(value ?? "–");

/** The other cells of a row in one line: "truth 3 · predicted 2 · missed 1". */
function summary(row: unknown[], skip: number, columns: string[]) {
  return columns
    .map((c, i) => (i === skip || isOverlay(row[i]) ? null : `${c} ${shown(row[i])}`))
    .filter(Boolean)
    .join(" · ");
}

// ---- tables -------------------------------------------------------------------------------

function cell(value: unknown, images: Record<string, string>) {
  if (typeof value === "string" && images[value]) {
    return (
      <a href={images[value]} target="_blank" rel="noreferrer">
        <img className="cell-img" src={images[value]} alt={value} loading="lazy" />
      </a>
    );
  }
  if (typeof value === "number") return Number.isInteger(value) ? value.toLocaleString() : metric(value);
  if (value === null || value === undefined) return <span className="muted">–</span>;
  return String(value);
}

/** What a cell is found and sorted by: an overlay by the names of what is drawn on it and how many boxes. */
function textOf(value: unknown, names: string[]): string {
  if (!isOverlay(value)) return String(value ?? "");
  const classes = new Set([...(value.boxes ?? []), ...(value.truth ?? [])].map((b) => names[b[0]] ?? String(b[0])));
  return [value.image, ...classes].join(" ");
}

const sortValue = (value: unknown) => (isOverlay(value) ? (value.boxes?.length ?? 0) : value);

/** A logged table: sortable by any column, narrowed by a text, the first rows shown. Pictures with boxes as a gallery. */
function TableView({
  file,
  images,
  view,
  onView,
}: {
  file: MediaFile;
  images: Record<string, string>;
  view: ViewChoice;
  onView: (change: Partial<ViewChoice>) => void;
}) {
  const [table, setTable] = useState<Table | null>(null);
  const [failure, setFailure] = useState<string | null>(null);
  const [opened, setOpened] = useState<number | null>(null);
  const { sort, filter, asTable, truth, minScore } = view;
  const setSort = (next: ViewChoice["sort"]) => onView({ sort: next });
  // The link is signed afresh on every refresh; the file behind a name does not change.
  useEffect(() => {
    setTable(null);
    fetch(file.url)
      .then((r) => (r.ok ? r.json() : Promise.reject(new Error(r.statusText))))
      .then((data: Table) => setTable(data))
      .catch((e) => setFailure(e instanceof Error ? e.message : String(e)));
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [file.name]);

  const names = table?.classes ?? [];
  const overlayColumn = table ? table.columns.findIndex((_, i) => table.rows.some((r) => isOverlay(r[i]))) : -1;
  const drawing: Drawing = { truth, minScore, names };

  const rows = useMemo(() => {
    if (!table) return [];
    const needle = filter.trim().toLowerCase();
    let out = needle ? table.rows.filter((r) => r.some((v) => textOf(v, names).toLowerCase().includes(needle))) : table.rows;
    if (sort) {
      out = [...out].sort((a, b) => {
        const x = sortValue(a[sort.col]);
        const y = sortValue(b[sort.col]);
        if (x === y) return 0;
        if (x === null || x === undefined) return 1;
        if (y === null || y === undefined) return -1;
        return sort.dir * (typeof x === "number" && typeof y === "number" ? x - y : String(x).localeCompare(String(y), "ko", { numeric: true }));
      });
    }
    return out;
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [table, sort, filter]);

  // The classes drawn in the shown rows, for the legend.
  const legend = useMemo(() => {
    if (overlayColumn < 0) return [];
    const seen = new Set<number>();
    for (const row of rows.slice(0, SHOWN_ROWS)) {
      const value = row[overlayColumn];
      if (isOverlay(value)) for (const b of [...(value.boxes ?? []), ...(value.truth ?? [])]) seen.add(b[0]);
    }
    return [...seen].sort((a, b) => a - b).slice(0, 30);
  }, [rows, overlayColumn]);

  if (failure) return <div className="error">표를 읽지 못했습니다: {failure}</div>;
  if (!table) return <p className="muted small">표를 불러오는 중…</p>;
  const gallery = overlayColumn >= 0 && !asTable;
  const numeric = table.columns.map((c, i) => [c, i] as const).filter(([, i]) => table.rows.some((r) => typeof r[i] === "number"));
  return (
    <div className="logged-table">
      <div className="table-tools">
        {table.rows.length > 10 && (
          <input type="search" value={filter} onChange={(e) => onView({ filter: e.target.value })} placeholder="표에서 찾기" aria-label="표에서 찾기" />
        )}
        {overlayColumn >= 0 && (
          <>
            <div className="segmented" role="group" aria-label="보기">
              <button type="button" className={gallery ? "active" : ""} onClick={() => onView({ asTable: false })}>
                그림
              </button>
              <button type="button" className={gallery ? "" : "active"} onClick={() => onView({ asTable: true })}>
                표
              </button>
            </div>
            {gallery && numeric.length > 0 && (
              <label className="inline">
                정렬
                <select
                  value={sort ? `${sort.col}:${sort.dir}` : ""}
                  onChange={(e) => {
                    const [col, dir] = e.target.value.split(":").map(Number);
                    setSort(e.target.value ? { col, dir: dir as 1 | -1 } : null);
                  }}
                >
                  <option value="">기록한 순서</option>
                  {numeric.map(([c, i]) => (
                    <option key={c} value={`${i}:-1`}>
                      {c} 많은 순
                    </option>
                  ))}
                </select>
              </label>
            )}
            <label className="inline check">
              <input type="checkbox" checked={truth} onChange={(e) => onView({ truth: e.target.checked })} />
              정답 함께 보기 <span className="muted">(점선)</span>
            </label>
            <label className="inline">
              확신도 ≥ {minScore.toFixed(2)}
              <input type="range" min={0} max={0.95} step={0.05} value={minScore} onChange={(e) => onView({ minScore: Number(e.target.value) })} />
            </label>
          </>
        )}
      </div>
      {legend.length > 0 && (
        <div className="overlay-legend">
          {legend.map((id) => (
            <span key={id}>
              <i style={{ background: classColor(id) }} />
              {names[id] ?? id}
            </span>
          ))}
        </div>
      )}
      {gallery ? (
        <div className="overlay-gallery">
          {rows.slice(0, SHOWN_ROWS).map((row, r) => {
            const value = row[overlayColumn];
            return (
              <figure key={r}>
                {isOverlay(value) && images[value.image] ? (
                  <OverlayPicture cell={value} url={images[value.image]} drawing={drawing} onOpen={() => setOpened(r)} />
                ) : (
                  <div className="muted small">이미지 없음</div>
                )}
                <figcaption>{summary(row, overlayColumn, table.columns)}</figcaption>
              </figure>
            );
          })}
        </div>
      ) : (
        <div className="table-wrap">
          <table className="plain">
            <thead>
              <tr>
                {table.columns.map((c, i) => (
                  <th
                    key={c}
                    className="sortable"
                    onClick={() => setSort(sort?.col === i ? { col: i, dir: sort.dir === 1 ? -1 : 1 } : { col: i, dir: 1 })}
                  >
                    {c}
                    {sort?.col === i ? (sort.dir === 1 ? " ↑" : " ↓") : ""}
                  </th>
                ))}
              </tr>
            </thead>
            <tbody>
              {rows.slice(0, SHOWN_ROWS).map((row, r) => (
                <tr key={r}>
                  {table.columns.map((c, i) => {
                    const value = row[i];
                    return (
                      <td key={c} className={typeof value === "number" ? "num right" : ""}>
                        {isOverlay(value) ? (
                          images[value.image] ? (
                            <div className="cell-overlay">
                              <OverlayPicture cell={value} url={images[value.image]} drawing={drawing} onOpen={() => setOpened(r)} />
                            </div>
                          ) : (
                            <span className="muted">–</span>
                          )
                        ) : (
                          cell(value, images)
                        )}
                      </td>
                    );
                  })}
                </tr>
              ))}
            </tbody>
          </table>
        </div>
      )}
      <p className="muted small">
        {rows.length > SHOWN_ROWS
          ? `${rows.length.toLocaleString()}행 중 ${SHOWN_ROWS}행을 보여 줍니다. 전체는 파일로 받으세요.`
          : `${rows.length.toLocaleString()}행`}{" "}
        ·{" "}
        <a href={file.url} download>
          JSON으로 받기
        </a>
      </p>
      {opened !== null && overlayColumn >= 0 && (
        <Lightbox
          rows={rows.slice(0, SHOWN_ROWS)}
          column={overlayColumn}
          columns={table.columns}
          at={opened}
          onAt={setOpened}
          images={images}
          drawing={drawing}
        />
      )}
    </div>
  );
}
