import { type FormEvent, useEffect, useState } from "react";
import { Icon } from "./ui";

/** A value that follows ``value`` once it has stopped changing for ``ms`` (search boxes). */
export function useDebounced<T>(value: T, ms = 300): T {
  const [settled, setSettled] = useState(value);
  useEffect(() => {
    const timer = setTimeout(() => setSettled(value), ms);
    return () => clearTimeout(timer);
  }, [value, ms]);
  return settled;
}

export function TagList({ tags }: { tags: string[] }) {
  if (tags.length === 0) return null;
  return (
    <span className="tag-list">
      {tags.map((t) => (
        <span key={t} className="tag">
          {t}
        </span>
      ))}
    </span>
  );
}

const splitTags = (text: string) =>
  text
    .split(",")
    .map((t) => t.trim())
    .filter(Boolean);

/**
 * A search box and the tags in use: what a list can be narrowed by.
 * Shown only as much as there is to show — no tags, no tag row.
 */
export function ListFilter({
  query,
  onQuery,
  tags,
  tag,
  onTag,
  placeholder,
}: {
  query: string;
  onQuery: (q: string) => void;
  tags: { name: string; count: number }[];
  tag: string;
  onTag: (t: string) => void;
  placeholder: string;
}) {
  return (
    <div className="list-filter">
      <input type="search" value={query} onChange={(e) => onQuery(e.target.value)} placeholder={placeholder} aria-label="찾기" />
      {tags.length > 0 && (
        <div className="chips">
          {tags.slice(0, 16).map((t) => (
            <button key={t.name} className={`chip ${tag === t.name ? "active" : ""}`} onClick={() => onTag(tag === t.name ? "" : t.name)}>
              {t.name} <span>{t.count}</span>
            </button>
          ))}
        </div>
      )}
    </div>
  );
}

/**
 * Note and tags of a run or a model (and the run's name): read as text, edited in place.
 * Nothing is shown but a small link until someone writes something.
 */
export function Annotations({
  note,
  tags,
  name,
  noteLabel,
  known,
  onSave,
}: {
  note: string;
  tags: string[];
  name?: string; // given: the name can be changed here too
  noteLabel: string;
  known: string[]; // tags used elsewhere in the project, offered while typing
  onSave: (fields: { name?: string; note: string; tags: string[] }) => Promise<unknown>;
}) {
  const [editing, setEditing] = useState(false);
  const [draft, setDraft] = useState({ name: name ?? "", note, tags: tags.join(", ") });
  const [failure, setFailure] = useState<string | null>(null);
  const [busy, setBusy] = useState(false);

  const open = () => {
    setDraft({ name: name ?? "", note, tags: tags.join(", ") });
    setFailure(null);
    setEditing(true);
  };
  const save = async (e: FormEvent) => {
    e.preventDefault();
    setBusy(true);
    setFailure(null);
    try {
      await onSave({ ...(name !== undefined ? { name: draft.name } : {}), note: draft.note, tags: splitTags(draft.tags) });
      setEditing(false);
    } catch (err) {
      setFailure(err instanceof Error ? err.message : String(err));
    } finally {
      setBusy(false);
    }
  };

  if (!editing) {
    return (
      <div className="annotations">
        {(note || tags.length > 0) && (
          <div className="annotations-body">
            <TagList tags={tags} />
            {note && <p className="note-text">{note}</p>}
          </div>
        )}
        <button className="link small" onClick={open}>
          {note || tags.length > 0 ? "편집" : `${noteLabel} · 태그 추가`}
        </button>
      </div>
    );
  }
  const typed = splitTags(draft.tags).map((t) => t.toLowerCase());
  const offered = known.filter((t) => !typed.includes(t.toLowerCase())).slice(0, 12);
  return (
    <form className="annotations editing card" onSubmit={save}>
      {name !== undefined && (
        <label>
          이름
          <input value={draft.name} onChange={(e) => setDraft({ ...draft, name: e.target.value })} maxLength={160} required />
        </label>
      )}
      <label>
        {noteLabel}
        <textarea
          value={draft.note}
          onChange={(e) => setDraft({ ...draft, note: e.target.value })}
          rows={3}
          placeholder="무엇을 바꿨는지, 왜 했는지, 결과가 어땠는지"
        />
      </label>
      <label>
        태그 <span className="muted small">쉼표로 구분</span>
        <input value={draft.tags} onChange={(e) => setDraft({ ...draft, tags: e.target.value })} placeholder="예: baseline, 야간" />
      </label>
      {offered.length > 0 && (
        <div className="chips">
          {offered.map((t) => (
            <button
              type="button"
              key={t}
              className="chip"
              onClick={() => setDraft({ ...draft, tags: [...splitTags(draft.tags), t].join(", ") })}
            >
              + {t}
            </button>
          ))}
        </div>
      )}
      {failure && <div className="error">{failure}</div>}
      <div className="modal-actions">
        <button type="button" onClick={() => setEditing(false)} disabled={busy}>
          취소
        </button>
        <button className="primary" disabled={busy}>
          {busy ? "저장 중…" : "저장"}
        </button>
      </div>
    </form>
  );
}

export function AliasChip({ alias, onRemove }: { alias: string; onRemove?: () => void }) {
  return (
    <span className={`alias ${alias}`}>
      {alias}
      {onRemove && (
        <button className="alias-x" onClick={onRemove} title={`${alias} 떼기`} aria-label={`${alias} 떼기`}>
          <Icon name="x" size={11} />
        </button>
      )}
    </span>
  );
}

/** Aliases on a model version, and a small field to add one (suggested names first, any name works). */
export function AliasEditor({
  aliases,
  suggestions,
  onAdd,
  onRemove,
}: {
  aliases: string[];
  suggestions: string[];
  onAdd: (alias: string) => Promise<unknown>;
  onRemove: (alias: string) => Promise<unknown>;
}) {
  const [adding, setAdding] = useState(false);
  const [text, setText] = useState("");
  const add = async (e: FormEvent) => {
    e.preventDefault();
    if (!text.trim()) return;
    await onAdd(text.trim());
    setText("");
    setAdding(false);
  };
  const offered = suggestions.filter((s) => !aliases.includes(s));
  return (
    <span className="alias-editor">
      {aliases.map((a) => (
        <AliasChip key={a} alias={a} onRemove={() => void onRemove(a)} />
      ))}
      {adding ? (
        <form onSubmit={add} className="alias-form">
          <input
            autoFocus
            value={text}
            onChange={(e) => setText(e.target.value)}
            list="alias-suggestions"
            placeholder="alias"
            maxLength={32}
            onKeyDown={(e) => e.key === "Escape" && setAdding(false)}
          />
          <datalist id="alias-suggestions">
            {offered.map((s) => (
              <option key={s} value={s} />
            ))}
          </datalist>
          <button className="primary small">붙이기</button>
        </form>
      ) : (
        <button className="link small" onClick={() => setAdding(true)} title="SOTA, production처럼 이 버전을 가리키는 이름">
          + alias
        </button>
      )}
    </span>
  );
}
