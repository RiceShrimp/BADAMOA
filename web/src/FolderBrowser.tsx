import { useEffect, useState } from "react";
import { api } from "./api";
import { Icon } from "./ui";

/** A node's folder, as the node itself lists it: names and counts only. */
interface Listing {
  path: string; // "" is the list of places to start from
  parent: string | null;
  folders: { name: string; path: string; guess: string }[];
  images: number;
  guess: string;
  truncated: boolean;
}

export const LAYOUT_NAME: Record<string, string> = { yolo: "YOLO", coco: "COCO", imagefolder: "클래스별 폴더" };

/**
 * Pick a folder on a node instead of typing its path. The Hub asks the node for each listing,
 * so it works on remote servers too; the node answers with folder names, image counts, and the
 * dataset layout a folder looks like.
 */
export function FolderBrowser({
  nodeId,
  nodeName,
  start,
  onPick,
  onCancel,
}: {
  nodeId: string;
  nodeName: string;
  start: string;
  onPick: (path: string, guess: string) => void;
  onCancel: () => void;
}) {
  const [path, setPath] = useState(start);
  const [hidden, setHidden] = useState(false);
  const [listing, setListing] = useState<Listing | null>(null);
  const [error, setError] = useState<string | null>(null);
  const [loading, setLoading] = useState(false);

  useEffect(() => {
    let live = true;
    setLoading(true);
    setError(null);
    api<Listing>(`/nodes/${nodeId}/browse?path=${encodeURIComponent(path)}&hidden=${hidden}`)
      .then((l) => live && setListing(l))
      .catch((e) => {
        if (!live) return;
        setError(e instanceof Error ? e.message : String(e));
        if (path && start === path) setPath(""); // a typed path that is not there: from the start
      })
      .finally(() => live && setLoading(false));
    return () => {
      live = false;
    };
  }, [nodeId, path, hidden, start]);

  const atStart = (listing?.path ?? path) === "";
  return (
    <div className="folder-browser">
      <div className="browser-head">
        <strong>{nodeName}</strong>
        <span className="mono path">{atStart ? "어디서 시작할까요?" : listing?.path ?? path}</span>
      </div>
      <div className="browser-tools">
        <button type="button" className="small" disabled={atStart} onClick={() => setPath("")}>
          처음
        </button>
        <button type="button" className="small" disabled={atStart || listing?.parent == null} onClick={() => setPath(listing?.parent ?? "")}>
          ↑ 위로
        </button>
        {listing && !atStart && listing.guess && <span className="tag accent">{LAYOUT_NAME[listing.guess] ?? listing.guess}로 보임</span>}
        {listing && !atStart && listing.images > 0 && <span className="muted small">이 폴더에 이미지 {listing.images.toLocaleString()}장</span>}
        <span className="spacer" />
        <label className="check inline-check">
          <input type="checkbox" checked={hidden} onChange={(e) => setHidden(e.target.checked)} /> 숨긴 폴더
        </label>
      </div>
      {error && <div className="error">{error}</div>}
      <ul className="browser-list" aria-busy={loading}>
        {loading && !listing && <li className="muted small">노드에 묻는 중…</li>}
        {listing && listing.folders.length === 0 && <li className="muted small">하위 폴더가 없습니다.</li>}
        {listing?.folders.map((f) => (
          <li key={f.path}>
            <button type="button" onClick={() => setPath(f.path)}>
              <Icon name="folder" size={14} />
              <span className="name">{f.name}</span>
              {f.guess && <span className="tag">{LAYOUT_NAME[f.guess] ?? f.guess}</span>}
            </button>
          </li>
        ))}
        {listing?.truncated && <li className="muted small">폴더가 많아 앞의 500개만 보입니다. 경로를 직접 적어도 됩니다.</li>}
      </ul>
      <div className="modal-actions">
        <button type="button" onClick={onCancel}>
          취소
        </button>
        <button type="button" className="primary" disabled={atStart || !listing} onClick={() => listing && onPick(listing.path, listing.guess)}>
          이 폴더 선택
        </button>
      </div>
    </div>
  );
}
