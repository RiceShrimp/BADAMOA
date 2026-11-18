import { type FormEvent, useEffect, useMemo, useState } from "react";
import { useDebounced } from "./Annotate";
import { type Job, api } from "./api";
import { type LabelClass, type ReleaseInfo, SPLIT_LABEL, STATE_LABEL, type TreeDataset, bytesLabel, classColor } from "./catalogTypes";
import { Modal, timeAgo, usePolling } from "./ui";

/** The release screen's filters, as the Hub takes them. */
export interface ItemFilter {
  split?: string;
  state?: string;
  class_id?: string;
  site?: string;
  camera?: string;
  q?: string;
}

interface Preview {
  items: number;
  splits: Record<string, number>;
  classes: number;
}

/** What renaming, merging and dropping classes comes to, in words. */
function classChanges(classes: LabelClass[], names: Record<number, string>, dropped: Set<number>) {
  const groups = new Map<string, LabelClass[]>();
  for (const c of classes) {
    const name = (names[c.id] ?? c.name).trim();
    if (dropped.has(c.id) || !name) continue;
    groups.set(name, [...(groups.get(name) ?? []), c]);
  }
  const merged = [...groups.entries()].filter(([, from]) => from.length > 1);
  const renamed = [...groups.entries()].filter(([name, from]) => from.length === 1 && from[0].name !== name);
  const gone = classes.filter((c) => dropped.has(c.id) || !(names[c.id] ?? c.name).trim());
  const parts = [
    ...merged.map(([name, from]) => `${from.map((c) => c.name).join(" + ")} → ${name}`),
    ...renamed.map(([name, [c]]) => `${c.name} → ${name}`),
    ...(gone.length ? [`뺌: ${gone.map((c) => c.name).join(", ")}`] : []),
  ];
  const changed = parts.length > 0;
  // Only the classes that change go to the Hub: {old id: new name}, "" for a dropped one.
  const body: Record<string, string> = {};
  for (const c of classes) {
    const name = dropped.has(c.id) ? "" : (names[c.id] ?? c.name).trim();
    if (name !== c.name) body[String(c.id)] = name;
  }
  return { count: groups.size, text: parts.join(" · "), body: changed ? body : null, names: [...groups.keys()] };
}

/** v3 → v4, 2026.1 → 2026.2, anything else → name-2. */
export function nextVersion(version: string): string {
  const match = version.match(/^(.*?)(\d+)$/);
  return match ? `${match[1]}${Number(match[2]) + 1}` : `${version}-2`;
}

export function describeFilter(filter: ItemFilter, release: ReleaseInfo): string {
  const parts: string[] = [];
  if (filter.split !== undefined) parts.push(`split ${SPLIT_LABEL[filter.split] ?? filter.split}`);
  if (filter.state) parts.push(STATE_LABEL[filter.state] ?? filter.state);
  if (filter.class_id) parts.push(release.label_space.classes.find((c) => String(c.id) === filter.class_id)?.name ?? "");
  if (filter.site) parts.push(`사이트 ${filter.site}`);
  if (filter.camera) parts.push(`카메라 ${filter.camera}`);
  if (filter.q) parts.push(`"${filter.q}" 포함`);
  return parts.filter(Boolean).join(" · ");
}

const SPLIT_ORDER = ["train", "val", "test", ""];
const splitText = (splits: Record<string, number>) =>
  Object.entries(splits)
    .sort(([a], [b]) => SPLIT_ORDER.indexOf(a) - SPLIT_ORDER.indexOf(b))
    .map(([name, n]) => `${SPLIT_LABEL[name] ?? name} ${n.toLocaleString()}`)
    .join(" · ");

/**
 * A new release from this one: all its items, those the filter shows, or all but those; their
 * splits kept or dealt out again; other releases with the same classes added. Nothing moves on disk.
 */
export function NewReleaseDialog({
  release,
  task,
  filter,
  filtered,
  editClasses = false,
  onClose,
  onCreated,
}: {
  release: ReleaseInfo;
  task: string;
  filter: ItemFilter;
  filtered: number;
  /** Opened from the classes tab: the class section starts open and the filter is not applied. */
  editClasses?: boolean;
  onClose: () => void;
  onCreated: (releaseId: string) => void;
}) {
  const filtering = Object.keys(filter).length > 0;
  const [take, setTake] = useState<"all" | "match" | "rest">(filtering && !editClasses ? "match" : "all");
  const [names, setNames] = useState<Record<number, string>>({});
  const [dropped, setDropped] = useState<Set<number>>(new Set());
  const [classQ, setClassQ] = useState("");
  const classes = release.label_space.classes;
  const plan = useMemo(() => classChanges(classes, names, dropped), [classes, names, dropped]);
  const [dataset, setDataset] = useState(release.dataset.name);
  const [version, setVersion] = useState(nextVersion(release.version));
  const [resplit, setResplit] = useState(false);
  const [ratios, setRatios] = useState({ train: 80, val: 10, test: 10 });
  const [groupBy, setGroupBy] = useState("");
  const [extra, setExtra] = useState<string[]>([]);
  const [note, setNote] = useState("");
  const [preview, setPreview] = useState<Preview | null>(null);
  const [error, setError] = useState<string | null>(null);
  const [saving, setSaving] = useState(false);
  const [tree, setTree] = useState<TreeDataset[]>([]);

  useEffect(() => {
    void api<TreeDataset[]>(`/projects/${release.project_id}/datasets`).then(setTree);
  }, [release.project_id]);
  // Releases that can be added: the same classes, ready, not this one.
  const others = tree.flatMap((d) =>
    d.releases
      .filter((r) => r.id !== release.id && r.status === "ready" && r.label_space === release.label_space.name)
      .map((r) => ({ id: r.id, label: `${d.name} ${r.version}`, items: r.items })),
  );

  const body = useMemo(
    () => ({
      sources: [release.id, ...extra],
      dataset,
      version,
      note,
      filter: take === "all" ? null : filter,
      exclude: take === "rest",
      split: resplit ? ratios : null,
      group_by: resplit ? groupBy : "",
      classes: plan.body,
    }),
    [release.id, extra, dataset, version, note, take, filter, resplit, ratios, groupBy, plan.body],
  );
  // What the release would hold, counted by the Hub as the choices change.
  const counted = useDebounced(JSON.stringify({ ...body, version: "", note: "" }), 250);
  useEffect(() => {
    let live = true;
    api<Preview>("/releases", { body: { ...JSON.parse(counted), dry_run: true } })
      .then((p) => live && (setPreview(p), setError(null)))
      .catch((e) => live && (setPreview(null), setError(e instanceof Error ? e.message : String(e))));
    return () => {
      live = false;
    };
  }, [counted]);

  const submit = async (e: FormEvent) => {
    e.preventDefault();
    setSaving(true);
    try {
      const made = await api<{ release_id: string }>("/releases", { body });
      onCreated(made.release_id);
    } catch (err) {
      setError(err instanceof Error ? err.message : String(err));
    } finally {
      setSaving(false);
    }
  };

  const total = release.stats.items ?? 0;
  const counts = release.stats.classes ?? {};
  const needle = classQ.trim().toLowerCase();
  const listed = classes.filter((c) => !needle || c.name.toLowerCase().includes(needle) || (names[c.id] ?? "").toLowerCase().includes(needle));
  const groups = [
    ...((release.stats.cameras?.length ?? 0) > 0 ? [["camera", "같은 카메라끼리"]] : []),
    ...((release.stats.sites?.length ?? 0) > 0 ? [["site", "같은 사이트끼리"]] : []),
  ];
  return (
    <Modal title="새 버전 만들기" onClose={onClose}>
      <form onSubmit={submit}>
        <p className="muted">
          {release.dataset.name} {release.version}의 항목으로 새 버전을 만듭니다. 라벨은 이 버전의 것을 그대로 쓰고, 이미지는
          옮기거나 복사하지 않습니다.
        </p>
        <label>
          담을 항목
          <select value={take} onChange={(e) => setTake(e.target.value as typeof take)}>
            <option value="all">이 버전 전체 ({total.toLocaleString()}개)</option>
            {filtering && (
              <>
                <option value="match">
                  지금 필터에 맞는 {filtered.toLocaleString()}개 ({describeFilter(filter, release)})
                </option>
                <option value="rest">지금 필터에 맞는 것을 뺀 {(total - filtered).toLocaleString()}개</option>
              </>
            )}
          </select>
        </label>
        {!filtering && <p className="muted small">일부만 담으려면 먼저 버전 화면에서 필터를 고르세요.</p>}
        <div className="form-row">
          <label>
            데이터셋
            <input value={dataset} onChange={(e) => setDataset(e.target.value)} required />
          </label>
          <label>
            버전
            <input value={version} onChange={(e) => setVersion(e.target.value)} required />
          </label>
        </div>
        {dataset.trim() !== release.dataset.name && dataset.trim() && (
          <p className="muted small">새 데이터셋 "{dataset.trim()}"으로 만듭니다.</p>
        )}
        <label>
          split
          <select value={resplit ? "again" : "keep"} onChange={(e) => setResplit(e.target.value === "again")}>
            <option value="keep">지금 split 그대로</option>
            <option value="again">비율로 다시 나누기</option>
          </select>
        </label>
        {resplit && (
          <>
            <div className="form-row">
              {(["train", "val", "test"] as const).map((name) => (
                <label key={name}>
                  {name} %
                  <input
                    type="number"
                    min={0}
                    max={100}
                    value={ratios[name]}
                    onChange={(e) => setRatios({ ...ratios, [name]: Number(e.target.value) })}
                  />
                </label>
              ))}
              {groups.length > 0 && (
                <label>
                  묶어서 나누기
                  <select value={groupBy} onChange={(e) => setGroupBy(e.target.value)}>
                    <option value="">이미지마다</option>
                    {groups.map(([value, label]) => (
                      <option key={value} value={value}>
                        {label}
                      </option>
                    ))}
                  </select>
                </label>
              )}
            </div>
            <p className="muted small">
              이미지(또는 카메라 · 사이트)마다 정해진 쪽으로 가므로, 같은 항목은 언제 다시 나눠도 같은 split에 들어갑니다.
              {groupBy && " 같은 묶음의 이미지는 모두 한 split에 들어가 train과 val에 비슷한 장면이 섞이지 않습니다."}
            </p>
          </>
        )}
        <details className="advanced" open={editClasses || plan.body !== null}>
          <summary>
            클래스 바꾸기 <span className="muted">(이름 바꾸기 · 합치기 · 빼기)</span>
          </summary>
          <p className="muted small">
            새 이름을 같게 적으면 합쳐집니다. 라벨은 새 개정으로 기록되고, 이 버전은 그대로 남습니다.{" "}
            {task === "classify" ? "뺀 클래스의 이미지는 새 버전에 담지 않습니다." : "뺀 클래스의 박스는 지우고 이미지는 남깁니다."}
          </p>
          {classes.length > 10 && (
            <input type="search" value={classQ} onChange={(e) => setClassQ(e.target.value)} placeholder="클래스 찾기" aria-label="클래스 찾기" />
          )}
          <div className="class-edit">
            <table className="plain">
              <thead>
                <tr>
                  <th>클래스</th>
                  <th className="right">라벨</th>
                  <th>새 이름</th>
                  <th className="mid">빼기</th>
                </tr>
              </thead>
              <tbody>
                {listed.map((c) => {
                  const off = dropped.has(c.id);
                  return (
                    <tr key={c.id} className={off ? "off" : ""}>
                      <td>
                        <i className="swatch-dot" style={{ background: c.color ?? classColor(c.id) }} />
                        {c.name}
                      </td>
                      <td className="num right muted">{(counts[String(c.id)]?.boxes ?? 0).toLocaleString()}</td>
                      <td>
                        <input
                          value={names[c.id] ?? c.name}
                          disabled={off}
                          list="new-class-names"
                          aria-label={`${c.name}의 새 이름`}
                          onChange={(e) => setNames({ ...names, [c.id]: e.target.value })}
                        />
                      </td>
                      <td className="mid">
                        <input
                          type="checkbox"
                          checked={off}
                          aria-label={`${c.name} 빼기`}
                          onChange={(e) => {
                            const next = new Set(dropped);
                            if (e.target.checked) next.add(c.id);
                            else next.delete(c.id);
                            setDropped(next);
                          }}
                        />
                      </td>
                    </tr>
                  );
                })}
              </tbody>
            </table>
            <datalist id="new-class-names">
              {plan.names.map((n) => (
                <option key={n} value={n} />
              ))}
            </datalist>
          </div>
          {plan.body && (
            <p className="small">
              클래스 {classes.length}개 → {plan.count}개 · {plan.text}
            </p>
          )}
        </details>
        {others.length > 0 && (
          <details className="advanced" open={extra.length > 0}>
            <summary>
              다른 버전 더하기 <span className="muted">(클래스 목록이 같은 버전)</span>
            </summary>
            {others.map((o) => (
              <label key={o.id} className="check">
                <input
                  type="checkbox"
                  checked={extra.includes(o.id)}
                  onChange={(e) => setExtra(e.target.checked ? [...extra, o.id] : extra.filter((id) => id !== o.id))}
                />
                {o.label} <span className="muted">({o.items.toLocaleString()}개, 전체)</span>
              </label>
            ))}
            <p className="muted small">같은 이미지가 여러 버전에 있으면 이 버전의 split과 라벨을 씁니다.</p>
          </details>
        )}
        <label>
          메모 (선택)
          <input value={note} onChange={(e) => setNote(e.target.value)} placeholder="이 버전에서 달라진 점" />
        </label>
        <div className="notice">
          {preview
            ? `새 버전: 항목 ${preview.items.toLocaleString()}개 · ${splitText(preview.splits)}${plan.body ? ` · 클래스 ${preview.classes}개` : ""}`
            : error
              ? null
              : "항목 수를 세는 중…"}
        </div>
        {error && <div className="error">{error}</div>}
        <div className="modal-actions">
          <button type="button" onClick={onClose}>
            취소
          </button>
          <button className="primary" disabled={saving || !preview || preview.items === 0}>
            만들기
          </button>
        </div>
      </form>
    </Modal>
  );
}

// ---- downloading a release ---------------------------------------------------------------

interface ExportEntry {
  job_id: string;
  status: Job["status"];
  format: string;
  images: boolean;
  created_at: string;
  finished_at: string | null;
  expires_at: string | null;
  file: string | null;
  bytes: number | null;
  counts: Record<string, number> | null;
  skipped: { unlabelled: number; missing: number } | null;
  error: string | null;
  download_url: string | null;
}

const FORMATS: Record<string, [string, string][]> = {
  detect: [
    ["yolo", "YOLO (images · labels · data.yaml)"],
    ["coco", "COCO (images · annotations JSON)"],
  ],
  segment: [
    ["yolo", "YOLO 분할 (윤곽 라벨)"],
    ["coco", "COCO (polygon)"],
  ],
  classify: [["imagefolder", "클래스별 폴더 (split/클래스/이미지)"]],
};
const FORMAT_NAME: Record<string, string> = { yolo: "YOLO", coco: "COCO", imagefolder: "클래스별 폴더" };

/** The release as a zip in a common format, made by the server holding its images and kept on the Hub for a week. */
export function ExportDialog({ release, task, onClose }: { release: ReleaseInfo; task: string; onClose: () => void }) {
  const formats = FORMATS[task] ?? FORMATS.detect;
  const [format, setFormat] = useState(formats[0][0]);
  const [images, setImages] = useState(true);
  const [error, setError] = useState<string | null>(null);
  const [lastLog, setLastLog] = useState("");
  const { data: exports, refresh } = usePolling(() => api<ExportEntry[]>(`/releases/${release.id}/exports`), 3000, [release.id]);
  const going = exports?.find((e) => e.status === "queued" || e.status === "running");

  // The newest log line of the export under way.
  useEffect(() => {
    if (!going) return setLastLog("");
    void api<{ text: string }[]>(`/jobs/${going.job_id}/logs?after=0`).then((lines) => setLastLog(lines.at(-1)?.text ?? ""));
  }, [going, exports]);

  const start = async (e: FormEvent) => {
    e.preventDefault();
    try {
      setError(null);
      await api(`/releases/${release.id}/exports`, { body: { format, images } });
      await refresh();
    } catch (err) {
      setError(err instanceof Error ? err.message : String(err));
    }
  };

  const size = release.stats.bytes ?? 0;
  return (
    <Modal title="버전 내려받기" onClose={onClose}>
      <form onSubmit={start}>
        <p className="muted">
          이미지가 있는 서버가 zip을 만들어 Hub에 올립니다. 라벨은 이 버전에 기록된 것을 그대로 담고, 아직 라벨하지 않은 이미지는
          뺍니다. 만든 zip은 {7}일 동안 받을 수 있습니다.
        </p>
        <label>
          형식
          <select value={format} onChange={(e) => setFormat(e.target.value)}>
            {formats.map(([value, label]) => (
              <option key={value} value={value}>
                {label}
              </option>
            ))}
          </select>
        </label>
        <label className="check">
          <input type="checkbox" checked={images} onChange={(e) => setImages(e.target.checked)} />
          이미지도 담기 <span className="muted">(약 {bytesLabel(size)} · 끄면 라벨과 파일 목록만)</span>
        </label>
        {error && <div className="error">{error}</div>}
        <div className="modal-actions">
          <button type="button" onClick={onClose}>
            닫기
          </button>
          <button className="primary" disabled={Boolean(going)}>
            {going ? "만드는 중…" : "zip 만들기"}
          </button>
        </div>
      </form>
      {going && <pre className="json">{lastLog || (going.status === "queued" ? "차례를 기다리는 중…" : "시작하는 중…")}</pre>}
      {exports && exports.length > 0 && (
        <table className="plain export-list">
          <thead>
            <tr>
              <th>형식</th>
              <th>만든 때</th>
              <th className="right">크기</th>
              <th />
            </tr>
          </thead>
          <tbody>
            {exports.map((e) => (
              <tr key={e.job_id}>
                <td>
                  {FORMAT_NAME[e.format] ?? e.format}
                  {e.images ? "" : " · 라벨만"}
                  {e.counts && (
                    <div className="muted small">
                      {Object.entries(e.counts)
                        .map(([s, n]) => `${s} ${n.toLocaleString()}`)
                        .join(" · ")}
                      {e.skipped && e.skipped.unlabelled + e.skipped.missing > 0
                        ? ` · 뺀 것 ${(e.skipped.unlabelled + e.skipped.missing).toLocaleString()}`
                        : ""}
                    </div>
                  )}
                </td>
                <td>{timeAgo(e.created_at)}</td>
                <td className="num right">{e.bytes ? bytesLabel(e.bytes) : "–"}</td>
                <td className="right">
                  {e.download_url ? (
                    <a href={e.download_url} download={e.file ?? undefined}>
                      받기
                    </a>
                  ) : e.status === "failed" || e.status === "canceled" ? (
                    <a href="#/jobs" title={e.error ?? ""}>
                      실패 · 로그
                    </a>
                  ) : e.status === "succeeded" ? (
                    <span className="muted">기간 지남</span>
                  ) : (
                    <span className="muted">만드는 중</span>
                  )}
                </td>
              </tr>
            ))}
          </tbody>
        </table>
      )}
    </Modal>
  );
}
