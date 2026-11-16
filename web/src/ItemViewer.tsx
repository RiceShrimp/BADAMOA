import { useEffect, useMemo, useRef, useState } from "react";
import { api } from "./api";
import {
  type ItemDetail,
  type ItemPage,
  type LabelClass,
  type Revision,
  SPLIT_LABEL,
  STATE_LABEL,
  bytesLabel,
  classColor,
} from "./catalogTypes";
import { Icon } from "./ui";

interface Props {
  releaseId: string;
  /** Query string of the grid's filters, so "next" follows the same list. */
  filterQuery: string;
  start: number;
  total: number;
  onClose: () => void;
  /** Colour of a class: the one picked in the classes tab, or the automatic one. */
  colorOf?: (id: number) => string;
  /** This version's classes: labels of other versions are matched to them by name. */
  classes?: LabelClass[];
}

/** Where a label revision came from, in words: "import:coco v1" → "가져오기 · coco v1". */
function sourceLabel(source: string): string {
  const [kind, ...rest] = source.split(":");
  const what = rest.join(":");
  const words: Record<string, string> = { import: "가져오기", classes: "클래스 변경", human: "사람이 고침", autolabel: "자동 라벨" };
  return words[kind] ? (what ? `${words[kind]} · ${what}` : words[kind]) : source;
}

/** Full-screen item view: boxes over the image, label revisions (one or two side by side), where else the image is used. */
export function ItemViewer({ releaseId, filterQuery, start, total, onClose, colorOf = classColor, classes = [] }: Props) {
  const [index, setIndex] = useState(start);
  const [item, setItem] = useState<ItemDetail | null>(null);
  const [error, setError] = useState<string | null>(null);
  const [revisionId, setRevisionId] = useState<number | null>(null);
  const [compareId, setCompareId] = useState<number | null>(null);
  const [showLabels, setShowLabels] = useState(true);
  const [showNames, setShowNames] = useState(true);
  const [fill, setFill] = useState(false);
  const [onlyHere, setOnlyHere] = useState(false);
  const [hidden, setHidden] = useState<Set<string>>(new Set()); // class names
  const [full, setFull] = useState<"loading" | "ok" | "error">("loading"); // the original image
  const stageRef = useRef<HTMLDivElement>(null);
  const [stage, setStage] = useState({ w: 0, h: 0 });

  useEffect(() => {
    const el = stageRef.current;
    if (!el) return;
    const measure = () => setStage({ w: el.clientWidth - 32, h: el.clientHeight - 84 }); // padding and nav bar
    measure();
    const observer = new ResizeObserver(measure);
    observer.observe(el);
    return () => observer.disconnect();
  }, []);

  useEffect(() => {
    let alive = true;
    setError(null);
    void (async () => {
      try {
        const page = await api<ItemPage>(`/releases/${releaseId}/items?${filterQuery}&offset=${index}&limit=1`);
        if (!alive) return;
        if (page.items.length === 0) return setError("항목을 찾을 수 없습니다.");
        const detail = await api<ItemDetail>(`/releases/${releaseId}/items/${page.items[0].id}`);
        if (!alive) return;
        setItem(detail);
        setFull(detail.image_url ? "loading" : "error");
        setRevisionId(detail.revision_id);
        setCompareId(null);
      } catch (e) {
        if (alive) setError(e instanceof Error ? e.message : String(e));
      }
    })();
    return () => {
      alive = false;
    };
  }, [releaseId, filterQuery, index]);

  useEffect(() => {
    const onKey = (e: KeyboardEvent) => {
      if (e.target instanceof HTMLInputElement) return;
      const key = e.key.toLowerCase();
      if (e.key === "Escape") onClose();
      else if (e.key === "ArrowRight") setIndex((i) => Math.min(total - 1, i + 1));
      else if (e.key === "ArrowLeft") setIndex((i) => Math.max(0, i - 1));
      else if (key === "l") setShowLabels((v) => !v);
      else if (key === "n") setShowNames((v) => !v);
      else if (key === "f") setFill((v) => !v);
    };
    window.addEventListener("keydown", onKey);
    return () => window.removeEventListener("keydown", onKey);
  }, [onClose, total]);

  const revision = item?.revisions.find((r) => r.id === revisionId) ?? null;
  const compare = item?.revisions.find((r) => r.id === compareId && r.id !== revisionId) ?? null;
  const currentRev = item?.revisions.find((r) => r.id === item.revision_id)?.rev ?? null;
  // A label revision goes by the versions that carry it; the bare revision number means nothing to people.
  const versionsOf = (rev: number) => {
    const here = item?.releases.find((r) => r.current)?.dataset;
    return (item?.releases ?? []).filter((r) => r.rev === rev).map((r) => (r.dataset === here ? r.version : `${r.dataset} ${r.version}`));
  };
  const revisionName = (r: Revision) => versionsOf(r.rev).join(" · ") || "버전에 없는 라벨";

  // Labels of another label space are matched to this version's classes by name (colour, "only these classes").
  const here = useMemo(() => new Map(classes.map((c) => [c.name, c.id])), [classes]);
  const foreign = (r: Revision | null) => Boolean(r && item && r.label_space.id !== item.release_label_space_id);
  const namesOf = (r: Revision) => Object.fromEntries(r.classes.map((c) => [c.id, c.name])) as Record<number, string>;
  const colorIn = (r: Revision, cls: number) => {
    if (!foreign(r)) return colorOf(cls);
    const same = here.get(namesOf(r)[cls]);
    return same != null ? colorOf(same) : classColor(cls);
  };
  const visible = (r: Revision) => {
    const names = namesOf(r);
    return r.boxes.filter((b) => {
      const name = names[b[0]] ?? String(b[0]);
      return !hidden.has(name) && !(onlyHere && foreign(r) && classes.length > 0 && !here.has(name));
    });
  };

  const counts = useMemo(() => {
    const map = new Map<string, { n: number; id: number }>();
    if (!revision) return [];
    const names = Object.fromEntries(revision.classes.map((c) => [c.id, c.name])) as Record<number, string>;
    for (const box of revision.boxes) {
      const name = names[box[0]] ?? String(box[0]);
      map.set(name, { n: (map.get(name)?.n ?? 0) + 1, id: box[0] });
    }
    return [...map.entries()].sort((a, b) => b[1].n - a[1].n);
  }, [revision]);

  const toggleClass = (name: string) => {
    const next = new Set(hidden);
    if (next.has(name)) next.delete(name);
    else next.add(name);
    setHidden(next);
  };

  const w = item?.width ?? 1;
  const h = item?.height ?? 1;
  const font = Math.max(w, h) / 70;
  // Fit the image (or two of them, side by side) inside the stage whatever its aspect ratio.
  const pair = Boolean(compare);
  const room = pair ? (stage.w - 12) / 2 : stage.w;
  const fitWidth = Math.max(0, Math.min(room, ((stage.h - (pair ? 26 : 0)) * w) / h));
  const frameStyle = { width: fitWidth, height: (fitWidth * h) / w };

  const drawBoxes = (r: Revision) => {
    const names = namesOf(r);
    return (
      <svg viewBox={`0 0 ${w} ${h}`} preserveAspectRatio="none">
        {visible(r).map((b, i) => {
          // A classification label has no geometry: it is drawn as a frame around the whole image.
          const [cls, cx = 0.5, cy = 0.5, bw = 1, bh = 1] = b;
          const x = (cx - bw / 2) * w;
          const y = (cy - bh / 2) * h;
          const color = colorIn(r, cls);
          const label = names[cls] ?? String(cls);
          // A segmentation label carries its outline after the box: x1, y1, x2, y2, …
          const outline = b.slice(5);
          const points = outline.map((v, n) => (n % 2 === 0 ? v * w : v * h)).join(" ");
          return (
            <g key={i}>
              {outline.length >= 6 && (
                <polygon
                  points={points}
                  fill={color}
                  fillOpacity={fill ? 0.4 : 0.18}
                  stroke={color}
                  strokeWidth="2"
                  vectorEffect="non-scaling-stroke"
                />
              )}
              <rect
                x={x}
                y={y}
                width={bw * w}
                height={bh * h}
                strokeDasharray={outline.length >= 6 ? "4 3" : undefined}
                fill={fill && outline.length < 6 ? color : "none"}
                fillOpacity={0.25}
                stroke={color}
                strokeWidth="2"
                vectorEffect="non-scaling-stroke"
              />
              {showNames && (
                <>
                  <rect x={x} y={Math.max(0, y - font * 1.35)} width={label.length * font * 0.62 + font * 0.6} height={font * 1.35} fill={color} />
                  <text x={x + font * 0.3} y={Math.max(font, y - font * 0.35)} fontSize={font} fill="#fff" fontWeight="600">
                    {label}
                  </text>
                </>
              )}
            </g>
          );
        })}
      </svg>
    );
  };

  const shown = [revision, compare].filter((r): r is Revision => r !== null);
  const anyForeign = shown.some(foreign) && classes.length > 0;

  return (
    <div className="viewer" role="dialog" aria-modal="true" aria-label="항목 보기">
      <div className="viewer-stage" ref={stageRef}>
        {error && <div className="error">{error}</div>}
        {item && (
          <div className="viewer-frames">
            {(shown.length ? shown : [null]).map((r, n) => (
              <figure key={r?.id ?? n} className="viewer-figure">
                <div className="viewer-frame" style={frameStyle}>
                  {/* the thumbnail at once, the original over it as soon as it is here */}
                  {item.thumb_url && full !== "ok" && <img src={item.thumb_url} alt="" className="placeholder" />}
                  {item.image_url && full !== "error" && (
                    <img
                      key={item.image_url}
                      src={item.image_url}
                      alt={item.file_name}
                      className={full === "ok" ? "" : "loading"}
                      onLoad={() => setFull("ok")}
                      onError={() => setFull("error")}
                    />
                  )}
                  {showLabels && r && drawBoxes(r)}
                </div>
                {pair && r && (
                  <figcaption>
                    <strong>{revisionName(r)}</strong> · {r.state === "labelled" ? `박스 ${visible(r).length}개` : STATE_LABEL[r.state]} ·{" "}
                    {r.label_space.name}
                    {n === 1 && (
                      <button className="link" onClick={() => setCompareId(null)}>
                        닫기
                      </button>
                    )}
                  </figcaption>
                )}
              </figure>
            ))}
          </div>
        )}
        {item && full === "loading" && item.image_local === false && (
          <div className="viewer-note">원본을 이미지가 있는 노드에서 받는 중… (받은 뒤에는 바로 열림)</div>
        )}
        {item && full === "error" && (
          <div className="viewer-note">원본을 받지 못해 썸네일로 보여 줍니다 (노드가 꺼져 있거나 파일이 바뀌었을 수 있음).</div>
        )}
        <div className="viewer-nav">
          <button className="icon-btn" onClick={() => setIndex(Math.max(0, index - 1))} disabled={index === 0} aria-label="이전">
            ‹
          </button>
          <span className="num">
            {(index + 1).toLocaleString()} / {total.toLocaleString()}
          </span>
          <button className="icon-btn" onClick={() => setIndex(Math.min(total - 1, index + 1))} disabled={index >= total - 1} aria-label="다음">
            ›
          </button>
        </div>
      </div>

      <aside className="viewer-panel">
        <header>
          <strong title={item?.file_name}>{item?.file_name ?? "…"}</strong>
          <button className="icon-btn" onClick={onClose} aria-label="닫기">
            <Icon name="x" />
          </button>
        </header>

        <section>
          <div className="eyebrow">표시</div>
          <label className="check">
            <input type="checkbox" checked={showLabels} onChange={(e) => setShowLabels(e.target.checked)} /> 라벨 <kbd>L</kbd>
          </label>
          <label className="check">
            <input type="checkbox" checked={showNames} onChange={(e) => setShowNames(e.target.checked)} /> 클래스 이름 <kbd>N</kbd>
          </label>
          <label className="check">
            <input type="checkbox" checked={fill} onChange={(e) => setFill(e.target.checked)} /> 박스 채우기 <kbd>F</kbd>
          </label>
          {anyForeign && (
            <label className="check" title="다른 클래스 목록의 라벨에서, 이 버전에 같은 이름의 클래스가 없는 것은 숨깁니다">
              <input type="checkbox" checked={onlyHere} onChange={(e) => setOnlyHere(e.target.checked)} /> 이 버전의 클래스만
            </label>
          )}
          <div className="class-chips">
            {counts.map(([name, { n, id }]) => (
              <button key={name} className={`class-chip ${hidden.has(name) ? "off" : ""}`} onClick={() => toggleClass(name)} aria-pressed={!hidden.has(name)}>
                <i style={{ background: revision ? colorIn(revision, id) : colorOf(id) }} />
                {name} <span>{n}</span>
              </button>
            ))}
          </div>
          {counts.length > 1 && (
            <div className="row-actions">
              <button onClick={() => setHidden(new Set())}>모두</button>
              <button onClick={() => setHidden(new Set(counts.map(([name]) => name)))}>없음</button>
            </div>
          )}
        </section>

        {item && (
          <>
            <section>
              <div className="eyebrow">버전별 라벨</div>
              {item.revisions.map((r) => (
                <div key={r.id} className={`revision ${r.id === revisionId ? "active" : ""} ${r.id === compare?.id ? "paired" : ""}`}>
                  <input
                    type="radio"
                    name="revision"
                    checked={r.id === revisionId}
                    onChange={() => setRevisionId(r.id)}
                    aria-label={`${revisionName(r)} 라벨 보기`}
                  />
                  <div className="revision-body" onClick={() => setRevisionId(r.id)}>
                    <div>
                      <strong title={`라벨 개정 ${r.rev}`}>{revisionName(r)}</strong> ·{r.state === "labelled" ? `박스 ${r.box_count}개` : STATE_LABEL[r.state]} ·{" "}
                      {r.label_space.name}
                      {r.id === item.revision_id && <span className="tag accent">이 버전</span>}
                    </div>
                    <div className="muted small">
                      {sourceLabel(r.source)} · {r.created_at.slice(0, 10)}
                    </div>
                    {r.id === revisionId && r.labelled_classes.length > 0 && (
                      <div className="muted small">
                        빠짐없이 라벨한 클래스 {r.labelled_classes.length}개: {r.labelled_classes.slice(0, 12).join(", ")}
                        {r.labelled_classes.length > 12 && " 외"}
                      </div>
                    )}
                  </div>
                  {item.revisions.length > 1 && r.id !== revisionId && (
                    <button
                      className={`icon-btn pair-btn ${r.id === compare?.id ? "active" : ""}`}
                      onClick={() => setCompareId(r.id === compare?.id ? null : r.id)}
                      title={r.id === compare?.id ? "나란히 보기 닫기" : "선택한 라벨과 나란히 보기"}
                      aria-pressed={r.id === compare?.id}
                    >
                      <Icon name="columns" size={14} />
                    </button>
                  )}
                </div>
              ))}
              {item.revisions.length > 1 && !compare && <p className="muted small">옆의 버튼으로 두 라벨을 나란히 볼 수 있습니다.</p>}
            </section>

            <section>
              <div className="eyebrow">이미지</div>
              <dl className="info">
                <dt>크기</dt>
                <dd>
                  {item.width}×{item.height} · {bytesLabel(item.bytes)}
                </dd>
                <dt>사이트</dt>
                <dd>{item.site || "–"}</dd>
                <dt>카메라</dt>
                <dd>{item.camera || "–"}</dd>
                <dt>split</dt>
                <dd>{SPLIT_LABEL[item.split] ?? item.split}</dd>
                <dt>sha256</dt>
                <dd className="mono" title={item.sha256}>
                  {item.sha256.slice(0, 16)}…
                </dd>
                <dt>경로</dt>
                <dd className="mono path">{item.rel_path}</dd>
              </dl>
            </section>

            <section>
              <div className="eyebrow">포함된 버전 ({item.releases.length})</div>
              <ul className="release-list">
                {item.releases.map((r) => (
                  <li key={r.id}>
                    <a href={`#/datasets/${r.id}`} className={r.current ? "current" : ""} onClick={onClose}>
                      {r.dataset} {r.version}
                    </a>
                    <span className="muted small">
                      {SPLIT_LABEL[r.split] ?? r.split}
                      {r.rev !== currentRev && " · 라벨이 다름"}
                    </span>
                  </li>
                ))}
              </ul>
            </section>
          </>
        )}
      </aside>
    </div>
  );
}
