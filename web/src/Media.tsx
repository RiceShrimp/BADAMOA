import { useEffect, useMemo, useState } from "react";
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

interface Table {
  columns: string[];
  rows: unknown[][];
}

const SHOWN_ROWS = 200;

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
        <TableView key={f.name} file={f} images={images} />
      ))}
    </section>
  );
}

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

/** A logged table: sortable by any column, narrowed by a text, the first rows shown. */
function TableView({ file, images }: { file: MediaFile; images: Record<string, string> }) {
  const [table, setTable] = useState<Table | null>(null);
  const [failure, setFailure] = useState<string | null>(null);
  const [sort, setSort] = useState<{ col: number; dir: 1 | -1 } | null>(null);
  const [filter, setFilter] = useState("");
  // The link is signed afresh on every refresh; the file behind a name does not change.
  useEffect(() => {
    setTable(null);
    fetch(file.url)
      .then((r) => (r.ok ? r.json() : Promise.reject(new Error(r.statusText))))
      .then((data: Table) => setTable(data))
      .catch((e) => setFailure(e instanceof Error ? e.message : String(e)));
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [file.name]);

  const rows = useMemo(() => {
    if (!table) return [];
    const needle = filter.trim().toLowerCase();
    let out = needle ? table.rows.filter((r) => r.some((v) => String(v ?? "").toLowerCase().includes(needle))) : table.rows;
    if (sort) {
      out = [...out].sort((a, b) => {
        const x = a[sort.col];
        const y = b[sort.col];
        if (x === y) return 0;
        if (x === null || x === undefined) return 1;
        if (y === null || y === undefined) return -1;
        return sort.dir * (typeof x === "number" && typeof y === "number" ? x - y : String(x).localeCompare(String(y), "ko", { numeric: true }));
      });
    }
    return out;
  }, [table, sort, filter]);

  if (failure) return <div className="error">표를 읽지 못했습니다: {failure}</div>;
  if (!table) return <p className="muted small">표를 불러오는 중…</p>;
  return (
    <div className="logged-table">
      {table.rows.length > 10 && (
        <input type="search" value={filter} onChange={(e) => setFilter(e.target.value)} placeholder="표에서 찾기" aria-label="표에서 찾기" />
      )}
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
                {table.columns.map((c, i) => (
                  <td key={c} className={typeof row[i] === "number" ? "num right" : ""}>
                    {cell(row[i], images)}
                  </td>
                ))}
              </tr>
            ))}
          </tbody>
        </table>
      </div>
      <p className="muted small">
        {rows.length > SHOWN_ROWS
          ? `${rows.length.toLocaleString()}행 중 ${SHOWN_ROWS}행을 보여 줍니다. 전체는 파일로 받으세요.`
          : `${rows.length.toLocaleString()}행`}{" "}
        ·{" "}
        <a href={file.url} download>
          JSON으로 받기
        </a>
      </p>
    </div>
  );
}
