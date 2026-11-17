import { type FormEvent, useState } from "react";
import { api } from "./api";
import type { ReleaseInfo } from "./catalogTypes";
import { nextVersion } from "./ReleaseMaker";
import { Icon, Modal, timeAgo } from "./ui";

interface Rebased {
  base: string;
  kept: number;
  overlapping: number[];
  dropped: number;
}

/** On a version screen: the dataset has fixed labels that are not a version yet. */
export function DraftNotice({ release }: { release: ReleaseInfo }) {
  const draft = release.draft;
  if (!draft || release.status === "draft") return null;
  const here = draft.base?.id === release.id;
  return (
    <a className="card draft-notice" href={`#/datasets/${draft.id}`}>
      <Icon name="edit" size={15} />
      <span>
        <strong>작업 중</strong> · 고친 이미지 {draft.edited.toLocaleString()}장
        {draft.base && !here && ` (바탕 ${draft.base.version})`}
        {draft.editors.length > 0 && ` · ${draft.editors.join(", ")}`} — 아직 버전이 아닙니다
      </span>
      <span className="link">
        열기 <Icon name="arrow" size={12} />
      </span>
    </a>
  );
}

/** On the work in progress itself: what it is based on, and making it a version or throwing it away. */
export function DraftActions({ release, onChanged }: { release: ReleaseInfo; onChanged: () => void }) {
  const draft = release.draft;
  const [confirming, setConfirming] = useState(false);
  const [busy, setBusy] = useState(false);
  const [message, setMessage] = useState<{ ok: boolean; text: string } | null>(null);
  if (!draft) return null;

  const discard = async () => {
    const text = `작업 중에서 고친 이미지 ${draft.edited.toLocaleString()}장을 모두 되돌릴까요?\n고친 라벨은 각 이미지의 라벨 기록에 남지만, 새 버전에는 들어가지 않습니다.`;
    if (!window.confirm(text)) return;
    setBusy(true);
    try {
      const done = await api<{ latest_id: string | null }>(`/releases/${release.id}/discard`, { method: "POST" });
      window.location.hash = done.latest_id ? `#/datasets/${done.latest_id}` : `#/datasets/project/${release.project_id}`;
    } catch (e) {
      setMessage({ ok: false, text: e instanceof Error ? e.message : String(e) });
    } finally {
      setBusy(false);
    }
  };
  const rebase = async () => {
    if (!draft.behind) return;
    setBusy(true);
    try {
      const done = await api<Rebased>(`/releases/${release.id}/rebase`, { method: "POST" });
      const parts = [`바탕을 ${done.base}(으)로 옮겼습니다 · 고친 이미지 ${done.kept.toLocaleString()}장`];
      if (done.overlapping.length) parts.push(`${done.overlapping.length.toLocaleString()}장은 ${done.base}에서도 라벨이 바뀌어 고친 쪽을 남겼습니다`);
      if (done.dropped) parts.push(`${done.dropped.toLocaleString()}장은 ${done.base}에 없어 뺐습니다`);
      setMessage({ ok: true, text: parts.join(" · ") });
      onChanged();
    } catch (e) {
      setMessage({ ok: false, text: e instanceof Error ? e.message : String(e) });
    } finally {
      setBusy(false);
    }
  };

  return (
    <>
      {draft.behind && (
        <div className="notice draft-behind">
          작업 중을 시작한 뒤 새 버전 {draft.behind.version}이(가) 들어왔습니다. 고친 라벨을 {draft.behind.version}에 얹으려면 바탕을 옮기세요.
          <button className="small" onClick={() => void rebase()} disabled={busy}>
            바탕을 {draft.behind.version}(으)로 옮기기
          </button>
        </div>
      )}
      {message && <div className={message.ok ? "notice ok" : "error"}>{message.text}</div>}
      <div className="head-actions">
        <button className="danger" onClick={() => void discard()} disabled={busy}>
          모두 되돌리기
        </button>
        <button className="primary" onClick={() => setConfirming(true)} disabled={busy || draft.edited === 0}>
          버전으로 확정
        </button>
      </div>
      {confirming && (
        <ConfirmDraft
          release={release}
          onClose={() => setConfirming(false)}
          onDone={() => {
            setConfirming(false);
            onChanged();
          }}
        />
      )}
    </>
  );
}

/** The work in progress's line under its title. */
export function DraftLine({ release }: { release: ReleaseInfo }) {
  const draft = release.draft;
  if (!draft) return null;
  return (
    <p className="muted">
      {draft.base && (
        <>
          바탕 <a href={`#/datasets/${draft.base.id}`}>{draft.base.version}</a> ·{" "}
        </>
      )}
      고친 이미지 {draft.edited.toLocaleString()}장{draft.editors.length > 0 && ` · ${draft.editors.join(", ")}`} · {timeAgo(draft.created_at)} 시작 ·
      아직 버전이 아니라 학습 · 내려받기에는 쓸 수 없습니다
    </p>
  );
}

function ConfirmDraft({ release, onClose, onDone }: { release: ReleaseInfo; onClose: () => void; onDone: () => void }) {
  const draft = release.draft!;
  const [version, setVersion] = useState(() => {
    const taken = new Set(release.siblings.map((s) => s.version));
    let next = nextVersion(draft.base?.version ?? "v1");
    while (taken.has(next)) next = nextVersion(next);
    return next;
  });
  const [note, setNote] = useState("");
  const [saving, setSaving] = useState(false);
  const [error, setError] = useState<string | null>(null);
  const submit = async (e: FormEvent) => {
    e.preventDefault();
    setSaving(true);
    setError(null);
    try {
      await api(`/releases/${release.id}/confirm`, { body: { version, note } });
      onDone();
    } catch (err) {
      setError(err instanceof Error ? err.message : String(err));
    } finally {
      setSaving(false);
    }
  };
  return (
    <Modal title="버전으로 확정" onClose={onClose}>
      <form onSubmit={submit}>
        <p className="muted">
          {draft.base ? `${draft.base.version}에 ` : ""}고친 이미지 {draft.edited.toLocaleString()}장의 라벨을 얹어 {release.dataset.name}의 새 버전으로
          만듭니다. 확정한 버전은 바뀌지 않고, 학습 · 내려받기에 쓸 수 있습니다.
        </p>
        <label>
          버전 이름
          <input value={version} onChange={(e) => setVersion(e.target.value)} autoFocus required />
        </label>
        <label>
          메모
          <textarea value={note} onChange={(e) => setNote(e.target.value)} rows={3} placeholder={`비우면 "이미지 ${draft.edited.toLocaleString()}장의 라벨을 고침"`} />
        </label>
        <p className="muted small">split을 다시 나누거나 클래스를 바꾸려면, 확정한 뒤 그 버전에서 "새 버전"을 쓰세요.</p>
        {error && <div className="error">{error}</div>}
        <div className="modal-actions">
          <button type="button" onClick={onClose}>
            취소
          </button>
          <button className="primary" disabled={saving || !version.trim()}>
            확정
          </button>
        </div>
      </form>
    </Modal>
  );
}
