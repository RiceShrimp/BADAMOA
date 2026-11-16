import { type ReactNode, useEffect, useMemo, useState } from "react";
import { TagList } from "./Annotate";
import { METRIC_LABEL, type Run, STAGE_LABEL, keyMetrics, metric } from "./trainTypes";
import { Badge, STATUS_LABEL, duration, timeAgo } from "./ui";

/** Settings that are bookkeeping rather than a choice worth a column. */
const HIDDEN_SETTINGS = new Set([
  "classes",
  "unlabelled",
  "val_fraction",
  "gpus",
  "min_vram_mb",
  "model_name",
  "base_model",
  "resumed_from",
  "sweep",
  "extra",
]);

type Value = string | number | null | undefined;

interface Column {
  key: string;
  label: string;
  group: "지표" | "설정" | "정보";
  value: (r: Run) => Value;
  cell?: (r: Run) => ReactNode;
  numeric?: boolean;
}

function settingValue(v: unknown): Value {
  if (v === null || v === undefined) return null;
  if (typeof v === "number" || typeof v === "string") return v;
  if (typeof v === "boolean") return v ? "true" : "false";
  return JSON.stringify(v);
}

/** Every column the runs at hand can fill: their metrics, their settings, and facts about the run. */
function columnsFor(runs: Run[], task: string): Column[] {
  const metricKeys = [...new Set([...keyMetrics(task), ...runs.flatMap((r) => Object.keys(r.metrics))])];
  const settingKeys = [
    ...new Set(
      runs.flatMap((r) =>
        Object.entries(r.config)
          .filter(([k, v]) => !HIDDEN_SETTINGS.has(k) && (v === null || typeof v !== "object"))
          .map(([k]) => k),
      ),
    ),
  ].sort();
  return [
    ...metricKeys.map(
      (k): Column => ({
        key: `m:${k}`,
        label: METRIC_LABEL[k] ?? k,
        group: "지표",
        value: (r) => r.metrics[k],
        cell: (r) => metric(r.metrics[k]),
        numeric: true,
      }),
    ),
    ...settingKeys.map(
      (k): Column => ({
        key: `s:${k}`,
        label: k,
        group: "설정",
        value: (r) => settingValue(r.config[k]),
        numeric: runs.every((r) => r.config[k] === undefined || typeof r.config[k] === "number"),
      }),
    ),
    {
      key: "i:model",
      label: "모델",
      group: "정보",
      value: (r) => (r.model ? `${r.model.name} v${r.model.version}` : null),
      cell: (r) =>
        r.model ? (
          <a href={`#/models/${r.model.id}`} onClick={(e) => e.stopPropagation()}>
            {r.model.name} v{r.model.version}
          </a>
        ) : (
          <span className="muted">–</span>
        ),
    },
    {
      key: "i:release",
      label: "버전",
      group: "정보",
      value: (r) => (r.release ? `${r.release.dataset} ${r.release.version}` : null),
    },
    { key: "i:template", label: "모델 종류", group: "정보", value: (r) => r.template.variant || r.template.title },
    {
      key: "i:node",
      label: "노드",
      group: "정보",
      value: (r) => r.node_name,
      cell: (r) => r.node_name ?? <span className="muted">배정 대기</span>,
    },
    {
      key: "i:duration",
      label: "소요",
      group: "정보",
      value: (r) => r.duration_s,
      cell: (r) => duration(r.duration_s),
      numeric: true,
    },
    { key: "i:created", label: "시작", group: "정보", value: (r) => r.created_at, cell: (r) => timeAgo(r.created_at) },
  ];
}

const defaultColumns = (task: string) => [...keyMetrics(task).map((k) => `m:${k}`), "i:model", "i:node", "i:duration", "i:created"];

/** Fixed columns can be sorted too. */
const FIXED: Record<string, (r: Run) => Value> = { name: (r) => r.name, status: (r) => STATUS_LABEL[r.status] };

function compare(a: Value, b: Value): number {
  if (a === b) return 0;
  if (a === null || a === undefined || a === "") return 1; // empty last, whichever the direction
  if (b === null || b === undefined || b === "") return -1;
  if (typeof a === "number" && typeof b === "number") return a - b;
  return String(a).localeCompare(String(b), "ko", { numeric: true });
}

/** The local date, as the file name a person expects (not UTC's). */
function today(): string {
  const d = new Date();
  return `${d.getFullYear()}-${String(d.getMonth() + 1).padStart(2, "0")}-${String(d.getDate()).padStart(2, "0")}`;
}

function csvCell(v: Value): string {
  const text = v === null || v === undefined ? "" : String(v);
  return /[",\n\r]/.test(text) ? `"${text.replace(/"/g, '""')}"` : text;
}

/** The runs as a spreadsheet: every metric and setting, not only the columns on screen. */
function downloadCsv(runs: Run[], columns: Column[], name: string) {
  const fixed: [string, (r: Run) => Value][] = [
    ["name", (r) => r.name],
    ["status", (r) => r.status],
    ["tags", (r) => r.tags.join("; ")],
    ["note", (r) => r.note],
    ["created_at", (r) => r.created_at],
    ["duration_s", (r) => r.duration_s],
    ["epoch", (r) => r.epoch],
    ["epochs", (r) => r.epochs],
  ];
  const extra = columns.filter((c) => !["i:created", "i:duration"].includes(c.key));
  const header = [...fixed.map(([k]) => k), ...extra.map((c) => c.key.replace(/^[msi]:/, (p) => ({ "m:": "", "s:": "param.", "i:": "" })[p] ?? ""))];
  const lines = [header, ...runs.map((r) => [...fixed.map(([, f]) => f(r)), ...extra.map((c) => c.value(r))])];
  const text = "﻿" + lines.map((line) => line.map(csvCell).join(",")).join("\n") + "\n"; // BOM: Excel reads Korean right
  const url = URL.createObjectURL(new Blob([text], { type: "text/csv;charset=utf-8" }));
  const link = document.createElement("a");
  link.href = url;
  link.download = name;
  link.click();
  setTimeout(() => URL.revokeObjectURL(url), 1000);
}

function ColumnPicker({
  columns,
  shown,
  onChange,
  onReset,
}: {
  columns: Column[];
  shown: string[];
  onChange: (keys: string[]) => void;
  onReset: () => void;
}) {
  return (
    <details className="column-picker">
      <summary className="btn">열 고르기</summary>
      <div className="column-menu">
        {(["지표", "설정", "정보"] as const).map((group) => {
          const cols = columns.filter((c) => c.group === group);
          if (cols.length === 0) return null;
          return (
            <div key={group} className="column-group">
              <div className="eyebrow">{group}</div>
              {cols.map((c) => (
                <label key={c.key} className="check">
                  <input
                    type="checkbox"
                    checked={shown.includes(c.key)}
                    onChange={() => onChange(shown.includes(c.key) ? shown.filter((k) => k !== c.key) : [...shown, c.key])}
                  />
                  {c.label}
                </label>
              ))}
            </div>
          );
        })}
        <button className="link small" onClick={onReset}>
          기본 열로
        </button>
      </div>
    </details>
  );
}

function readSaved(key: string): string[] | null {
  try {
    const saved = localStorage.getItem(key);
    return saved ? (JSON.parse(saved) as string[]) : null;
  } catch {
    return null;
  }
}

/**
 * The run table: pick which metrics and settings to see, sort by any column, take it all as CSV.
 * The chosen columns are remembered per project in this browser.
 */
export function RunTable({
  runs,
  projectId,
  task,
  filtered,
  compared = [],
  onCompare,
  tools,
  storageKey: ownKey,
  defaults,
}: {
  runs: Run[] | null;
  projectId: string;
  task: string;
  filtered: boolean;
  compared?: string[];
  onCompare?: (id: string) => void; // without it there is no column of checkboxes
  tools?: ReactNode; // the search box and tag chips, on the same row
  storageKey?: string; // where the chosen columns are remembered (default: per project)
  defaults?: string[];
}) {
  const storageKey = ownKey ?? `badamoa.runColumns.${projectId}`;
  const [shown, setShown] = useState<string[] | null>(() => readSaved(storageKey));
  const [sort, setSort] = useState<{ key: string; dir: 1 | -1 } | null>(null);
  useEffect(() => {
    setShown(readSaved(storageKey));
    setSort(null);
  }, [storageKey]);

  const columns = useMemo(() => columnsFor(runs ?? [], task), [runs, task]);
  const chosen = shown ?? defaults ?? defaultColumns(task);
  const visible = columns.filter((c) => chosen.includes(c.key));
  const choose = (keys: string[] | null) => {
    setShown(keys);
    try {
      if (keys) localStorage.setItem(storageKey, JSON.stringify(keys));
      else localStorage.removeItem(storageKey);
    } catch {
      /* private window: the choice lasts until the page is left */
    }
  };

  const rows = useMemo(() => {
    if (!runs || !sort) return runs ?? [];
    const read = FIXED[sort.key] ?? columns.find((c) => c.key === sort.key)?.value;
    if (!read) return runs;
    return [...runs].sort((a, b) => {
      const va = read(a);
      const vb = read(b);
      const empty = (v: Value) => v === null || v === undefined || v === "";
      if (empty(va) || empty(vb)) return compare(va, vb); // empty stays last in both directions
      return sort.dir * compare(va, vb);
    });
  }, [runs, sort, columns]);

  const sortBy = (key: string) =>
    setSort(sort?.key !== key ? { key, dir: key.startsWith("m:") ? -1 : 1 } : sort.dir === 1 ? { key, dir: -1 } : { key, dir: 1 });
  const head = (key: string, label: string, numeric?: boolean) => (
    <th
      key={key}
      className={`sortable ${numeric ? "right" : ""}`}
      onClick={() => sortBy(key)}
      aria-sort={sort?.key === key ? (sort.dir === 1 ? "ascending" : "descending") : undefined}
      title="눌러서 정렬"
    >
      {label}
      {sort?.key === key ? (sort.dir === 1 ? " ↑" : " ↓") : ""}
    </th>
  );

  return (
    <>
      <div className="table-tools">
        {tools}
        <span className="spacer" />
        {runs && runs.length > 0 && (
          <>
            <ColumnPicker columns={columns} shown={chosen} onChange={choose} onReset={() => choose(null)} />
            <button onClick={() => downloadCsv(rows, columns, `runs-${today()}.csv`)}>CSV로 받기</button>
          </>
        )}
      </div>
      <div className="card flush table-wrap">
        <table className="runs">
          <thead>
            <tr>
              {onCompare && <th aria-label="비교" />}
              {head("name", "실행")}
              {head("status", "상태")}
              {visible.map((c) => head(c.key, c.label, c.numeric))}
            </tr>
          </thead>
          <tbody>
            {runs?.length === 0 && (
              <tr>
                <td colSpan={(onCompare ? 3 : 2) + visible.length} className="empty">
                  {filtered ? "찾는 실행이 없습니다." : '아직 학습한 적이 없습니다. "새 학습"으로 데이터셋 버전을 골라 시작하세요.'}
                </td>
              </tr>
            )}
            {rows.map((run) => {
              const slot = compared.indexOf(run.id) + 1;
              return (
                <tr key={run.id} className="clickable" onClick={() => (window.location.hash = `#/training/${run.id}`)}>
                  {onCompare && (
                    <td onClick={(e) => e.stopPropagation()} className="pick">
                      <label title="지표 그래프에 겹쳐 보기">
                        <input
                          type="checkbox"
                          checked={slot > 0}
                          onChange={() => onCompare(run.id)}
                          disabled={run.epoch === 0}
                          aria-label={`${run.name} 지표 겹쳐 보기`}
                        />
                        {slot > 0 && <i style={{ background: `var(--series-${slot})` }} />}
                      </label>
                    </td>
                  )}
                  <td>
                    <strong>{run.name}</strong> <TagList tags={run.tags} />
                    <div className="muted small">
                      {run.release ? `${run.release.dataset} ${run.release.version}` : "삭제된 버전"} ·{" "}
                      {run.template.variant || run.template.title}
                      {run.sweep && (
                        <>
                          {" · "}
                          <a href={`#/training/sweep/${run.sweep.id}`} onClick={(e) => e.stopPropagation()}>
                            탐색 {run.sweep.name}
                          </a>
                        </>
                      )}
                    </div>
                    {run.note && <div className="note-line">{run.note}</div>}
                  </td>
                  <td>
                    <Badge kind={run.status}>{STATUS_LABEL[run.status]}</Badge>
                    {run.status === "running" &&
                      (run.stage === "train" && run.epochs > 0 ? (
                        <Progress epoch={run.epoch} epochs={run.epochs} />
                      ) : (
                        <div className="wait-reason">{STAGE_LABEL[run.stage] ?? "준비"} 중</div>
                      ))}
                    {run.status === "queued" && run.wait_reason && <div className="wait-reason">{run.wait_reason}</div>}
                  </td>
                  {visible.map((c) => (
                    <td key={c.key} className={c.numeric ? "num right" : ""}>
                      {c.cell ? c.cell(run) : (c.value(run) ?? <span className="muted">–</span>)}
                    </td>
                  ))}
                </tr>
              );
            })}
          </tbody>
        </table>
      </div>
    </>
  );
}

export function Progress({ epoch, epochs }: { epoch: number; epochs: number }) {
  const pct = epochs > 0 ? Math.min(100, Math.round((epoch / epochs) * 100)) : 0;
  return (
    <div className="progress" title={`epoch ${epoch} / ${epochs}`}>
      <div className="progress-bar">
        <i style={{ width: `${pct}%` }} />
      </div>
      <span>
        {epoch}/{epochs}
      </span>
    </div>
  );
}
