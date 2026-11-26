import { useEffect, useRef, useState } from "react";
import { downloadFile } from "./api";
import { Icon } from "./ui";

/** "목록 받기": the list on screen under its conditions (every image of it, not only this page) as a file —
 * CSV (a row per image: dataset, version, split, file name, path on its server, labels) or TXT (the paths). */
export function ListDownload({ path, name, disabled = false }: { path: string; name: string; disabled?: boolean }) {
  const [open, setOpen] = useState(false);
  const [toLeft, setToLeft] = useState(false); // near the right edge: the menu opens leftwards
  const [busy, setBusy] = useState("");
  const [error, setError] = useState("");
  const box = useRef<HTMLDivElement>(null);
  useEffect(() => {
    if (!open) return;
    const away = (e: MouseEvent) => {
      if (box.current && !box.current.contains(e.target as Node)) setOpen(false);
    };
    document.addEventListener("mousedown", away);
    return () => document.removeEventListener("mousedown", away);
  }, [open]);

  const get = async (format: "csv" | "txt") => {
    setOpen(false);
    setBusy(format);
    setError("");
    try {
      const joined = path.includes("?") ? "&" : "?";
      await downloadFile(`${path}${joined}format=${format}`, `${name}.${format}`);
    } catch (e) {
      setError(e instanceof Error ? e.message : String(e));
    } finally {
      setBusy("");
    }
  };

  return (
    <div className="list-download" ref={box}>
      <button
        type="button"
        className="ghost-btn"
        onClick={(e) => {
          const at = e.currentTarget.getBoundingClientRect();
          setToLeft(at.left + 300 > window.innerWidth);
          setOpen(!open);
        }}
        disabled={disabled || Boolean(busy)}
        title="지금 조건에 맞는 이미지 전체(이 페이지만이 아님)를 파일로 받습니다"
      >
        <Icon name="download" size={13} /> {busy ? "목록 만드는 중…" : "목록 받기"}
      </button>
      {open && (
        <div className={`list-download-menu${toLeft ? " to-left" : ""}`} role="menu">
          <button type="button" role="menuitem" onClick={() => void get("csv")}>
            <strong>CSV</strong>
            <span className="muted small">데이터셋 · 버전 · split · 파일 이름 · 서버의 경로 · 클래스</span>
          </button>
          <button type="button" role="menuitem" onClick={() => void get("txt")}>
            <strong>TXT</strong>
            <span className="muted small">서버의 이미지 경로만, 한 줄에 하나</span>
          </button>
        </div>
      )}
      {error && <span className="error small">{error}</span>}
    </div>
  );
}
