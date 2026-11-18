import { type FormEvent, useEffect, useRef, useState } from "react";
import { type BoardProject, useBoardFilter } from "./Boards";
import { Collage, type Shot } from "./DataBoard";
import { ItemViewer, type LabelingMode } from "./ItemViewer";
import type { ItemFilter } from "./ReleaseMaker";
import { colorFor } from "./ReleaseTabs";
import { api } from "./api";
import {
  type LabelSetPart,
  type LabelSetSummary,
  type LabelSetView,
  type ReleaseInfo,
  SET_STATE_LABEL,
  classColor,
  type SetColumn,
  type SetState,
} from "./catalogTypes";
import { Icon, Modal, Stat, timeAgo, usePolling } from "./ui";

const COLUMNS: { id: SetColumn; label: string; hint: string }[] = [
  { id: "unassigned", label: "미배정", hint: "아직 맡은 사람이 없는 일감" },
  { id: "labeling", label: "라벨링", hint: "라벨링하는 중이거나 돌려받은 이미지가 있는 일감" },
  { id: "review", label: "검수", hint: "다 라벨링해서 검수를 기다리는 일감" },
  { id: "done", label: "완료", hint: "모두 승인된(검수 없으면 모두 라벨링된) 일감" },
];
const STATE_ORDER: SetState[] = ["approved", "labelled", "rejected", "todo"];

/** How far the images got, as one bar: approved, labelled (waiting for review), sent back, still to do. */
function Progress({ states, size }: { states: Record<SetState, number>; size: number }) {
  return (
    <div className="set-progress" title={STATE_ORDER.map((s) => `${SET_STATE_LABEL[s]} ${states[s] ?? 0}`).join(" · ")}>
      {STATE_ORDER.map((s) => (states[s] ? <i key={s} className={`set-${s}`} style={{ width: `${(states[s] / Math.max(size, 1)) * 100}%` }} /> : null))}
    </div>
  );
}

function StateCounts({ states }: { states: Record<SetState, number> }) {
  return (
    <span className="muted small">
      {STATE_ORDER.filter((s) => states[s]).map((s, n) => (
        <span key={s}>
          {n > 0 && " · "}
          <span className={`set-dot set-${s}`} />
          {SET_STATE_LABEL[s]} {states[s].toLocaleString()}
        </span>
      ))}
    </span>
  );
}

/** Labeling: sets of images people label (and review) part by part. ``arg`` ("label-2", "review-0") opens a part. */
export function Labeling({ setId, arg }: { setId?: string; arg?: string }) {
  return setId ? <SetBoard key={setId} id={setId} arg={arg} /> : <LabelingBoard />;
}

/** A part given to the viewer, with what is left for them in it. */
interface MyPart {
  set: { id: string; name: string; review: boolean };
  project: { id: string; name: string };
  dataset: { id: string; name: string } | null;
  part: number;
  size: number;
  states: Record<SetState, number>;
  column: SetColumn;
  roles: ("label" | "review")[];
  to_label: number;
  rejected: number;
  to_review: number;
}

interface SetCard extends LabelSetSummary {
  collage: Shot[];
  open_release_id: string | null;
  updated_at: string;
}

interface LabelingBoardInfo {
  me: { id: string; name: string };
  mine: MyPart[];
  projects: { project: BoardProject; sets: SetCard[]; updated_at: string }[];
  without: BoardProject[];
  totals: { sets: number; items: number; todo: number; review: number; done: number };
}

const projectOfGroup = (g: LabelingBoardInfo["projects"][number]) => g.project;

/** 라벨링's first screen: my parts and what is left in them, then every project's sets. */
function LabelingBoard() {
  const { data: board, error } = usePolling(() => api<LabelingBoardInfo>("/boards/labeling"), 15000);
  const { shown, tools } = useBoardFilter(board?.projects, projectOfGroup);
  if (error && !board) return <div className="error">{error}</div>;
  const t = board?.totals;
  return (
    <>
      <div className="page-head">
        <div>
          <h1>라벨링</h1>
          <p className="muted">
            이미지를 일감으로 나눠 사람에게 맡기고, 라벨링 → 검수 → 완료로 따라갑니다. 라벨은 데이터셋의 작업 중에 모였다가 "버전으로
            확정"하면 새 버전이 됩니다.
          </p>
        </div>
      </div>
      <div className="stats five">
        <Stat icon="edit" label="라벨링 묶음" value={t?.sets ?? "–"} sub={`이미지 ${(t?.items ?? 0).toLocaleString()}장`} />
        <Stat icon="images" label="할 일" value={(t?.todo ?? 0).toLocaleString()} sub="라벨링할 · 돌려받은 이미지" />
        <Stat icon="review" label="검수 대기" value={(t?.review ?? 0).toLocaleString()} sub="라벨링이 끝나 검수를 기다림" />
        <Stat icon="check" label="완료" value={(t?.done ?? 0).toLocaleString()} sub="승인(검수 없으면 라벨함)" />
        <Stat icon="activity" label="내 일감" value={board?.mine.length ?? "–"} sub={board ? `${board.me.name}에게 맡겨진 일감` : " "} />
      </div>

      <section className="board-section">
        <h2>
          내 일감 <span className="muted">{board?.mine.length ?? ""}</span>
        </h2>
        {board && board.mine.length === 0 && (
          <p className="muted small">맡겨진 일감이 없습니다. 라벨링 묶음의 보드에서 일감마다 라벨링할 사람과 검수할 사람을 정합니다.</p>
        )}
        <div className="card-grid my-parts">
          {board?.mine.map((m) => (
            <div key={`${m.set.id}-${m.part}`} className={`card my-part ${m.to_label || m.to_review ? "" : "waiting"}`}>
              <div className="part-title">
                <a href={`#/labeling/${m.set.id}`}>
                  <strong>{m.set.name}</strong> · 일감 {m.part + 1}
                </a>
                <span className="muted small">{m.size}장</span>
              </div>
              <div className="muted small">
                {m.project.name}
                {m.dataset ? ` · ${m.dataset.name}` : ""} · {m.roles.map((r) => (r === "label" ? "라벨링" : "검수")).join(" · ")}
              </div>
              <Progress states={m.states} size={m.size} />
              <StateCounts states={m.states} />
              <div className="part-actions">
                {m.to_label > 0 && (
                  <a className="btn primary small" href={`#/labeling/${m.set.id}/label-${m.part}`}>
                    <Icon name="edit" size={12} /> 이어서 라벨링 · {m.to_label}장{m.rejected ? ` (돌려받음 ${m.rejected})` : ""}
                  </a>
                )}
                {m.to_review > 0 && (
                  <a className={`btn small ${m.to_label ? "" : "primary"}`} href={`#/labeling/${m.set.id}/review-${m.part}`}>
                    <Icon name="review" size={12} /> 검수 · {m.to_review}장
                  </a>
                )}
                {!m.to_label && !m.to_review && (
                  <span className="muted small">{m.roles.includes("review") ? "라벨링이 끝나기를 기다리는 중" : "검수를 기다리는 중"}</span>
                )}
              </div>
            </div>
          ))}
        </div>
      </section>

      <section className="board-section">
        <h2>프로젝트별 라벨링 묶음</h2>
        {board && board.projects.length === 0 && (
          <div className="card empty">
            아직 라벨링 묶음이 없습니다. 데이터셋의 최신 버전에서 이미지를 거른 뒤(예: 라벨 상태 "아직 안 함") "라벨링 묶음 만들기"를 누르세요.
          </div>
        )}
        {tools}
        {shown.map((g) => (
          <div key={g.project.id} className="board-group">
            <h3>
              <a href={`#/datasets/project/${g.project.id}`}>{g.project.name}</a>
              <span className="muted small">
                {g.project.folder ? `${g.project.folder} · ` : ""}묶음 {g.sets.length} · {timeAgo(g.updated_at)} 작업
              </span>
            </h3>
            <div className="card-grid set-grid">
              {g.sets.map((s) => (
                <a key={s.id} className="data-card data-card-link set-card" href={`#/labeling/${s.id}`}>
                  <Collage shots={s.collage} colorOf={classColor} more={s.items} />
                  <div className="data-card-body">
                    <div className="data-card-title">
                      <strong>{s.name}</strong>
                      {s.mine > 0 && <span className="tag accent">내 일감 {s.mine}</span>}
                    </div>
                    <div className="muted small">
                      {s.dataset?.name} {s.release?.version} · 이미지 {s.items.toLocaleString()} · 일감 {s.parts}
                      {s.review ? " · 검수함" : " · 검수 없음"}
                    </div>
                    <Progress states={s.states} size={s.items} />
                    <StateCounts states={s.states} />
                    <div className="set-columns small">
                      {COLUMNS.filter((c) => s.review || c.id !== "review").map((c) => (
                        <span key={c.id} className={s.columns[c.id] ? "" : "muted"}>
                          {c.label} {s.columns[c.id]}
                        </span>
                      ))}
                    </div>
                    <div className="muted small">
                      {s.created_by ?? "?"} · {timeAgo(s.updated_at)} 작업
                    </div>
                  </div>
                </a>
              ))}
            </div>
          </div>
        ))}
        {board && board.without.length > 0 && (
          <p className="muted small">
            묶음이 없는 프로젝트:{" "}
            {board.without.map((p, n) => (
              <span key={p.id}>
                {n > 0 && " · "}
                <a href={`#/datasets/project/${p.id}`}>{p.name}</a>
              </span>
            ))}{" "}
            — 데이터셋 버전 화면에서 이미지를 거른 뒤 "라벨링 묶음 만들기"로 만듭니다.
          </p>
        )}
      </section>
    </>
  );
}

function SetBoard({ id, arg }: { id: string; arg?: string }) {
  const { data: set, error, refresh } = usePolling(() => api<LabelSetView>(`/label-sets/${id}`), 15000, [id]);
  const openId = set?.open_release_id ?? "";
  const { data: release } = usePolling(() => (openId ? api<ReleaseInfo>(`/releases/${openId}`) : Promise.resolve(null)), 0, [openId]);
  const [mineOnly, setMineOnly] = useState(false);
  const [open, setOpen] = useState<{ part: LabelSetPart; mode: LabelingMode["mode"] } | null>(null);
  const [failed, setFailed] = useState<string | null>(null);
  // From "내 일감": #/labeling/<set>/label-2 opens part 2 to label (review-2 to review), once.
  const asked = useRef(arg ?? "");
  useEffect(() => {
    const wanted = /^(label|review)-(\d+)$/.exec(asked.current);
    if (!wanted || !set || !openId) return;
    asked.current = "";
    const part = set.part_list.find((p) => p.part === Number(wanted[2]));
    if (part) setOpen({ part, mode: wanted[1] as LabelingMode["mode"] });
    window.history.replaceState(null, "", `#/labeling/${id}`);
  }, [set, openId, id]);

  if (error && !set) return <div className="error">{error}</div>;
  if (!set) return <div className="muted">불러오는 중…</div>;
  const classes = release?.label_space.classes ?? [];
  const parts = set.part_list.filter((p) => !mineOnly || p.assignee?.id === set.me || p.reviewer?.id === set.me);

  const assign = async (part: number, field: "assignee" | "reviewer", value: string) => {
    try {
      setFailed(null);
      await api(`/label-sets/${id}/parts/${part}`, { method: "PATCH", body: { [field]: value } });
      void refresh();
    } catch (e) {
      setFailed(e instanceof Error ? e.message : String(e));
    }
  };
  const remove = async () => {
    if (!window.confirm(`라벨링 묶음 "${set.name}"을(를) 지울까요? 지금까지 저장한 라벨은 작업 중(또는 확정한 버전)에 그대로 남습니다.`)) return;
    try {
      await api(`/label-sets/${id}`, { method: "DELETE" });
      window.location.hash = "#/labeling";
    } catch (e) {
      setFailed(e instanceof Error ? e.message : String(e));
    }
  };
  const person = (value: { id: string } | null, field: "assignee" | "reviewer", part: number) => (
    <select value={value?.id ?? ""} onChange={(e) => void assign(part, field, e.target.value)} aria-label={field === "assignee" ? "라벨링할 사람" : "검수할 사람"}>
      <option value="">{field === "assignee" ? "라벨링: 없음" : "검수: 없음"}</option>
      {set.people.map((p) => (
        <option key={p.id} value={p.id}>
          {field === "assignee" ? "라벨링" : "검수"}: {p.name}
          {p.id === set.me ? " (나)" : ""}
        </option>
      ))}
    </select>
  );

  return (
    <>
      <nav className="page-crumbs">
        <a href="#/labeling">라벨링</a>
        <strong>{set.name}</strong>
      </nav>
      <div className="page-head">
        <div>
          <h1 className="with-badge">
            {set.name}
            <span className="tag">{set.review ? "검수함" : "검수 없음"}</span>
          </h1>
          <p className="muted">
            {set.dataset?.name}{" "}
            {set.release && <a href={`#/datasets/${set.release.id}`}>{set.release.version}</a>}에서 고른 이미지 {set.items.toLocaleString()}장 · 일감{" "}
            {set.parts}개({set.part_size}장씩) · {set.created_by ?? "?"} · {timeAgo(set.created_at)}
            {openId && openId !== set.release?.id && (
              <>
                {" "}
                · 라벨은 <a href={`#/datasets/${openId}`}>작업 중</a>에 모입니다
              </>
            )}
          </p>
        </div>
        <div className="head-actions">
          <label className="check inline-check">
            <input type="checkbox" checked={mineOnly} onChange={(e) => setMineOnly(e.target.checked)} /> 내 일감만
          </label>
          <button className="danger" onClick={() => void remove()}>
            지우기
          </button>
        </div>
      </div>
      {set.instructions && (
        <div className="card set-instructions">
          <div className="eyebrow">지시문</div>
          <p className="note">{set.instructions}</p>
        </div>
      )}
      <div className="stats five">
        <Stat icon="images" label="이미지" value={set.items.toLocaleString()} sub={`일감 ${set.parts}개`} />
        {STATE_ORDER.map((s) => (
          <Stat key={s} icon={s === "approved" ? "check" : s === "rejected" ? "x" : s === "todo" ? "edit" : "review"} label={SET_STATE_LABEL[s]} value={(set.states[s] ?? 0).toLocaleString()} />
        ))}
      </div>
      {failed && <div className="error">{failed}</div>}
      {!openId && <div className="notice">이 데이터셋에 등록이 끝난 버전이 없어 라벨링할 곳이 없습니다.</div>}

      <div className={`set-board ${set.review ? "" : "no-review"}`}>
        {COLUMNS.filter((c) => set.review || c.id !== "review").map((c) => {
          const here = parts.filter((p) => p.column === c.id);
          return (
            <section key={c.id} className="set-column" aria-label={c.label}>
              <header title={c.hint}>
                <strong>{c.label}</strong> <span className="muted">{here.length}</span>
              </header>
              {here.map((p) => (
                <div key={p.part} className={`part-card ${p.assignee?.id === set.me || p.reviewer?.id === set.me ? "mine" : ""}`}>
                  <div className="part-title">
                    <strong>일감 {p.part + 1}</strong>
                    <span className="muted small">{p.size}장</span>
                  </div>
                  <Progress states={p.states} size={p.size} />
                  <StateCounts states={p.states} />
                  {person(p.assignee, "assignee", p.part)}
                  {set.review && person(p.reviewer, "reviewer", p.part)}
                  <div className="part-actions">
                    {openId && c.id !== "done" && (
                      <button className={c.id !== "review" ? "primary small" : "small"} onClick={() => setOpen({ part: p, mode: "label" })}>
                        <Icon name="edit" size={12} /> 라벨링
                      </button>
                    )}
                    {openId && set.review && p.next_review != null && (
                      <button className={c.id === "review" ? "primary small" : "small"} onClick={() => setOpen({ part: p, mode: "review" })}>
                        <Icon name="review" size={12} /> 검수
                      </button>
                    )}
                    {openId && c.id === "done" && (
                      <button className="small" onClick={() => setOpen({ part: p, mode: "label" })}>
                        다시 보기
                      </button>
                    )}
                  </div>
                </div>
              ))}
              {here.length === 0 && <p className="muted small">없음</p>}
            </section>
          );
        })}
      </div>

      {open && openId && (
        <ItemViewer
          releaseId={openId}
          filterQuery={`label_set=${id}&part=${open.part.part}`}
          start={(open.mode === "review" ? open.part.next_review : open.part.next_label) ?? 0}
          total={open.part.size}
          colorOf={colorFor(classes)}
          classes={classes}
          task={release?.task ?? "detect"}
          labeling={{ setId: id, part: open.part.part, mode: open.mode, review: set.review, name: set.name, instructions: set.instructions }}
          onSaved={() => void refresh()}
          onClose={() => {
            setOpen(null);
            void refresh();
          }}
        />
      )}
    </>
  );
}

/** From a version screen: a set of the images the filters show now. */
export function NewSetDialog({
  release,
  filter,
  filtered,
  describe,
  onClose,
}: {
  release: ReleaseInfo;
  filter: ItemFilter & { edited?: boolean; suspect?: string; mistake?: string; check?: string };
  filtered: number;
  describe: string;
  onClose: () => void;
}) {
  const [name, setName] = useState(`${release.dataset.name} 라벨링`);
  const [instructions, setInstructions] = useState("");
  const [partSize, setPartSize] = useState(50);
  const [review, setReview] = useState(true);
  const [saving, setSaving] = useState(false);
  const [error, setError] = useState<string | null>(null);
  const parts = Math.ceil(filtered / Math.max(1, partSize));
  const submit = async (e: FormEvent) => {
    e.preventDefault();
    setSaving(true);
    setError(null);
    try {
      const made = await api<LabelSetView>("/label-sets", {
        body: { release_id: release.id, name, instructions, filter, part_size: partSize, review },
      });
      window.location.hash = `#/labeling/${made.id}`;
    } catch (err) {
      setError(err instanceof Error ? err.message : String(err));
    } finally {
      setSaving(false);
    }
  };
  return (
    <Modal title="라벨링 묶음 만들기" onClose={onClose}>
      <form onSubmit={submit}>
        <p className="muted">
          지금 거른 이미지 {filtered.toLocaleString()}장{describe ? ` (${describe})` : ""}을 일감으로 나눠 사람에게 맡깁니다. 라벨은 데이터셋의
          작업 중에 모이고, "버전으로 확정"하면 새 버전이 됩니다.
        </p>
        <label>
          이름
          <input value={name} onChange={(e) => setName(e.target.value)} required autoFocus />
        </label>
        <label>
          지시문 <span className="muted small">(선택 · 라벨링하는 사람에게 보임)</span>
          <textarea value={instructions} onChange={(e) => setInstructions(e.target.value)} rows={3} placeholder="예: 사람과 차를 박스로, 30픽셀보다 작은 것은 빼기" />
        </label>
        <div className="form-row">
          <label>
            일감 하나에 이미지
            <input type="number" min={1} max={10000} value={partSize} onChange={(e) => setPartSize(Number(e.target.value) || 1)} />
          </label>
          <label className="check" style={{ alignSelf: "end" }}>
            <input type="checkbox" checked={review} onChange={(e) => setReview(e.target.checked)} /> 검수하기 (승인 · 돌려보내기)
          </label>
        </div>
        <p className="muted small">일감 {parts.toLocaleString()}개가 생깁니다. 맡길 사람은 만든 뒤 보드에서 정합니다.</p>
        {error && <div className="error">{error}</div>}
        <div className="modal-actions">
          <button type="button" onClick={onClose}>
            취소
          </button>
          <button className="primary" disabled={saving || filtered === 0 || !name.trim()}>
            만들기
          </button>
        </div>
      </form>
    </Modal>
  );
}
