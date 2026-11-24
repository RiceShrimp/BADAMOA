import { type ReactNode, useEffect, useRef, useState } from "react";
import { colorFor } from "./ReleaseTabs";

/** What the class filter reads as: 모든 클래스, one class, or the first and how many more (horse 외 2). */
export function classFilterLabel(classes: { id: number; name: string }[], picked: number[]): string {
  const names = classes.filter((c) => picked.includes(c.id)).map((c) => c.name);
  if (names.length === 0) return "모든 클래스";
  return names.length === 1 ? names[0] : `${names[0]} 외 ${names.length - 1}`;
}

/**
 * A filter row's class filter: several classes at once, ticked in a list that drops down from a control the size of
 * the row's selects; 모든 클래스 clears it. What the picked classes narrow is the caller's (the version screen: the
 * images holding any of them; 검수: the findings of those classes), told in ``hint``.
 */
export function ClassFilter({
  classes,
  picked,
  onChange,
  hint,
}: {
  classes: { id: number; name: string; color?: string | null }[];
  picked: number[];
  onChange: (ids: number[]) => void;
  hint?: ReactNode;
}) {
  const [open, setOpen] = useState(false);
  const [find, setFind] = useState("");
  const box = useRef<HTMLDivElement>(null);
  const colorOf = colorFor(classes);
  useEffect(() => {
    if (!open) return setFind(""); // opened again, it lists every class
    const away = (e: MouseEvent) => box.current && !box.current.contains(e.target as Node) && setOpen(false);
    const esc = (e: KeyboardEvent) => e.key === "Escape" && setOpen(false);
    document.addEventListener("mousedown", away);
    document.addEventListener("keydown", esc);
    return () => {
      document.removeEventListener("mousedown", away);
      document.removeEventListener("keydown", esc);
    };
  }, [open]);
  const label = classFilterLabel(classes, picked);
  const names = classes.filter((c) => picked.includes(c.id)).map((c) => c.name);
  const shown = classes.filter((c) => !find.trim() || c.name.toLowerCase().includes(find.trim().toLowerCase()));
  // kept in the classes' order, so the same classes are the same filter however they were ticked
  const toggle = (id: number) =>
    onChange(classes.map((c) => c.id).filter((x) => (x === id ? !picked.includes(id) : picked.includes(x))));
  return (
    <div className="class-filter" ref={box}>
      <button
        type="button"
        className="class-filter-button"
        onClick={() => setOpen(!open)}
        aria-expanded={open}
        aria-haspopup="listbox"
        aria-label={`클래스: ${label}`}
        title={names.length > 1 ? names.join(", ") : undefined}
      >
        <span>{label}</span>
        <svg width="12" height="12" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2.4" aria-hidden="true">
          <path d="M6 9l6 6 6-6" />
        </svg>
      </button>
      {open && (
        <div className="class-filter-panel" role="listbox" aria-multiselectable="true" aria-label="클래스">
          {hint && <div className="muted small">{hint}</div>}
          {classes.length > 12 && <input value={find} onChange={(e) => setFind(e.target.value)} placeholder="클래스 찾기…" autoFocus />}
          <div className="class-filter-list">
            {!find.trim() && (
              <label className="check all">
                <input type="checkbox" checked={picked.length === 0} onChange={() => onChange([])} />
                모든 클래스
              </label>
            )}
            {shown.map((c) => (
              <label key={c.id} className="check">
                <input type="checkbox" checked={picked.includes(c.id)} onChange={() => toggle(c.id)} />
                <i className="class-dot" style={{ background: colorOf(c.id) }} />
                {c.name}
              </label>
            ))}
            {shown.length === 0 && <span className="muted small">맞는 클래스가 없습니다.</span>}
          </div>
        </div>
      )}
    </div>
  );
}
