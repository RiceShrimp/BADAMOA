import { useEffect, useMemo, useRef, useState } from "react";
import { api } from "./api";
import { type ItemDetail, type ItemPage, SPLIT_LABEL, STATE_LABEL, bytesLabel, classColor } from "./catalogTypes";
import { Icon } from "./ui";

interface Props {
  releaseId: string;
  /** Query string of the grid's filters, so "next" follows the same list. */
  filterQuery: string;
  start: number;
  total: number;
  onClose: () => void;
}

/** Full-screen item view: boxes over the image, label revisions, where else the image is used. */
export function ItemViewer({ releaseId, filterQuery, start, total, onClose }: Props) {
  const [index, setIndex] = useState(start);
  const [item, setItem] = useState<ItemDetail | null>(null);
  const [error, setError] = useState<string | null>(null);
  const [revisionId, setRevisionId] = useState<number | null>(null);
  const [showLabels, setShowLabels] = useState(true);
  const [showNames, setShowNames] = useState(true);
  const [fill, setFill] = useState(false);
  const [hidden, setHidden] = useState<Set<number>>(new Set());
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
        setRevisionId(detail.revision_id);
        setHidden(new Set());
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
  const names = useMemo(() => Object.fromEntries((revision?.classes ?? []).map((c) => [c.id, c.name])), [revision]);
  const counts = useMemo(() => {
    const map = new Map<number, number>();
    for (const box of revision?.boxes ?? []) map.set(box[0], (map.get(box[0]) ?? 0) + 1);
    return [...map.entries()].sort((a, b) => b[1] - a[1]);
  }, [revision]);

  const toggleClass = (id: number) => {
    const next = new Set(hidden);
    if (next.has(id)) next.delete(id);
    else next.add(id);
    setHidden(next);
  };

  const w = item?.width ?? 1;
  const h = item?.height ?? 1;
  const font = Math.max(w, h) / 70;
  // Fit the image inside the stage whatever its aspect ratio (small images are scaled up).
  const fitWidth = Math.max(0, Math.min(stage.w, (stage.h * w) / h));
  const frameStyle = { width: fitWidth, height: (fitWidth * h) / w };

  return (
    <div className="viewer" role="dialog" aria-modal="true" aria-label="항목 보기">
      <div className="viewer-stage" ref={stageRef}>
        {error && <div className="error">{error}</div>}
        {item && (
          <div className="viewer-frame" style={frameStyle}>
            <img src={item.image_url ?? item.thumb_url ?? ""} alt={item.file_name} />
            {showLabels && revision && (
              <svg viewBox={`0 0 ${w} ${h}`} preserveAspectRatio="none">
                {revision.boxes
                  .filter((b) => !hidden.has(b[0]))
                  .map((b, i) => {
                    // A classification label has no geometry: it is drawn as a frame around the whole image.
                    const [cls, cx = 0.5, cy = 0.5, bw = 1, bh = 1] = b;
                    const x = (cx - bw / 2) * w;
                    const y = (cy - bh / 2) * h;
                    const color = classColor(cls);
                    const label = names[cls] ?? String(cls);
                    return (
                      <g key={i}>
                        <rect
                          x={x}
                          y={y}
                          width={bw * w}
                          height={bh * h}
                          fill={fill ? color : "none"}
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
            )}
          </div>
        )}
        {item && !item.image_url && (
          <div className="viewer-note">원본 파일이 다른 노드에 있어 썸네일로 표시합니다.</div>
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
          <div className="class-chips">
            {counts.map(([id, n]) => (
              <button key={id} className={`class-chip ${hidden.has(id) ? "off" : ""}`} onClick={() => toggleClass(id)} aria-pressed={!hidden.has(id)}>
                <i style={{ background: classColor(id) }} />
                {names[id] ?? id} <span>{n}</span>
              </button>
            ))}
          </div>
          {counts.length > 1 && (
            <div className="row-actions">
              <button onClick={() => setHidden(new Set())}>모두</button>
              <button onClick={() => setHidden(new Set(counts.map(([id]) => id)))}>없음</button>
            </div>
          )}
        </section>

        {item && (
          <>
            <section>
              <div className="eyebrow">라벨 개정</div>
              {item.revisions.map((r) => (
                <label key={r.id} className={`revision ${r.id === revisionId ? "active" : ""}`}>
                  <input type="radio" name="revision" checked={r.id === revisionId} onChange={() => setRevisionId(r.id)} />
                  <div>
                    <div>
                      <strong>rev {r.rev}</strong> · {r.state === "labelled" ? `박스 ${r.box_count}개` : STATE_LABEL[r.state]} ·{" "}
                      {r.label_space.name}
                      {r.id === item.revision_id && <span className="tag accent">이 릴리스</span>}
                    </div>
                    <div className="muted small">
                      {r.source} · {r.created_at.slice(0, 10)}
                    </div>
                    {r.id === revisionId && r.labelled_classes.length > 0 && (
                      <div className="muted small">
                        빠짐없이 라벨한 클래스 {r.labelled_classes.length}개: {r.labelled_classes.slice(0, 12).join(", ")}
                        {r.labelled_classes.length > 12 && " 외"}
                      </div>
                    )}
                  </div>
                </label>
              ))}
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
              <div className="eyebrow">포함된 릴리스 ({item.releases.length})</div>
              <ul className="release-list">
                {item.releases.map((r) => (
                  <li key={r.id}>
                    <a href={`#/datasets/${r.id}`} className={r.current ? "current" : ""} onClick={onClose}>
                      {r.dataset} {r.version}
                    </a>
                    <span className="muted small">
                      {SPLIT_LABEL[r.split] ?? r.split} · rev {r.rev ?? "?"}
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
