import { type ReactNode, useState } from "react";
import { api } from "./api";
import { type TrashInfo, announceTrash, bytesLabel, taskLabel } from "./catalogTypes";
import { Icon, timeAgo, usePolling } from "./ui";

// ---- the trash: everything deleted, to bring back or remove for good ------------------------------

type Kind = "projects" | "releases" | "runs" | "models" | "templates";

const KIND_LABEL: Record<Kind, string> = {
  projects: "프로젝트",
  releases: "데이터셋 버전",
  runs: "학습",
  models: "모델",
  templates: "모델 종류",
};

const RESTORE: Record<Kind, (id: string) => string> = {
  projects: (id) => `/projects/${id}/restore`,
  releases: (id) => `/releases/${id}/restore`,
  runs: (id) => `/runs/${id}/restore`,
  models: (id) => `/models/${id}/restore`,
  templates: (id) => `/templates/${id}/restore`,
};

// What removing for good takes along, said before it happens.
const GOES: Record<Kind, string> = {
  projects: "데이터셋 · 버전 · 라벨 기록, 학습 기록 · 모델과 그 파일이 Hub에서 지워집니다. 노드에 등록한 이미지 파일은 그대로 남습니다.",
  releases: "버전과 그 라벨 기록이 Hub에서 지워집니다. 노드에 등록한 이미지 파일은 그대로 남습니다.",
  runs: "지표 · 로그 · 파일 · 체크포인트가 Hub에서 지워집니다.",
  models: "가중치 파일과 평가 · 내보내기 · 테스트 기록이 Hub에서 지워집니다.",
  templates: "등록한 코드(모든 버전)가 Hub에서 지워집니다. 이것으로 한 학습 기록은 남지만 다시 실행할 수는 없습니다.",
};
const GOES_ALL = "기록과 Hub의 파일이 지워집니다. 노드에 등록한 이미지 파일은 그대로 남습니다.";

type Emptied = Record<Kind, number> & {
  files: number;
  bytes: number;
  kept: { id: string; name: string; reason: string }[];
};

const KINDS = Object.keys(KIND_LABEL) as Kind[];
const keyOf = (kind: Kind, id: string) => `${kind}:${id}`;

/** What a row is called in messages: "coco v2", "cars v3" … */
function nameOf(trash: TrashInfo, kind: Kind, id: string): string {
  if (kind === "projects") return trash.projects.find((x) => x.id === id)?.name ?? id;
  if (kind === "releases") {
    const r = trash.releases.find((x) => x.id === id);
    return r ? `${r.dataset} ${r.version}` : id;
  }
  if (kind === "runs") return trash.runs.find((x) => x.id === id)?.name ?? id;
  if (kind === "models") {
    const m = trash.models.find((x) => x.id === id);
    return m ? `${m.name} v${m.version}` : id;
  }
  const t = trash.templates.find((x) => x.id === id);
  return t ? t.title || t.name : id;
}

function Section({ title, hint, children }: { title: string; hint?: string; children: ReactNode }) {
  return (
    <section className="card home-block">
      <div className="card-head">
        <h3>{title}</h3>
        {hint && <span className="muted small">{hint}</span>}
      </div>
      <table className="plain trash-table">{children}</table>
    </section>
  );
}

export function TrashPage({ isAdmin }: { isAdmin: boolean }) {
  const { data: trash, error, refresh } = usePolling(() => api<TrashInfo>("/trash"), 30000);
  const [failed, setFailed] = useState<string | null>(null);
  const [note, setNote] = useState<string | null>(null);
  const [emptied, setEmptied] = useState<Emptied | null>(null);
  const [busy, setBusy] = useState(false);
  const [picked, setPicked] = useState<string[]>([]);

  // Members bring back their own projects and runs; the rest is for admins.
  const canPick = (kind: Kind) => isAdmin || kind === "projects" || kind === "runs";
  const present = new Set(trash ? KINDS.flatMap((k) => trash[k].map((x) => keyOf(k, x.id))) : []);
  const chosen = picked.filter((p) => present.has(p)); // what is still there after a refresh
  const chosenOf = (kind: Kind) => chosen.filter((p) => p.startsWith(`${kind}:`)).map((p) => p.slice(kind.length + 1));
  const toggle = (key: string) => setPicked((cur) => (cur.includes(key) ? cur.filter((x) => x !== key) : [...cur, key]));

  const start = () => {
    setFailed(null);
    setNote(null);
    setEmptied(null);
  };
  /** Bring back each of ``items``; notes (a run renamed …) and failures are shown together. */
  const restore = async (items: { kind: Kind; id: string }[]) => {
    if (!trash || items.length === 0) return;
    start();
    setBusy(true);
    const notes: string[] = [];
    const failures: string[] = [];
    for (const { kind, id } of items) {
      try {
        const done = await api<{ note?: string }>(RESTORE[kind](id), { method: "POST" });
        if (done?.note) notes.push(items.length > 1 ? `${nameOf(trash, kind, id)}: ${done.note}` : done.note);
      } catch (e) {
        failures.push(`${nameOf(trash, kind, id)}: ${e instanceof Error ? e.message : String(e)}`);
      }
    }
    const back = items.length - failures.length;
    if (items.length > 1 && back > 0) notes.unshift(`${back}개를 되살렸습니다.`);
    setNote(notes.length ? notes.join("\n") : null);
    setFailed(failures.length ? failures.join("\n") : null);
    setPicked([]);
    setBusy(false);
    announceTrash();
    await refresh();
  };
  // Removing for good: the records and the Hub's files only they used.
  const purge = async (question: string, goes: string, body: Partial<Record<Kind, string[]>>) => {
    if (!window.confirm(`${question}\n\n${goes} 되돌릴 수 없습니다.`)) return;
    start();
    setBusy(true);
    try {
      setEmptied(await api<Emptied>("/trash/empty", { body }));
      setPicked([]);
      announceTrash();
      await refresh();
    } catch (e) {
      setFailed(e instanceof Error ? e.message : String(e));
    } finally {
      setBusy(false);
    }
  };
  const purgeChosen = () => {
    const body: Partial<Record<Kind, string[]>> = {};
    for (const kind of KINDS) if (chosenOf(kind).length) body[kind] = chosenOf(kind);
    const counted = KINDS.filter((k) => body[k]).map((k) => `${KIND_LABEL[k]} ${body[k]?.length}개`);
    const goes = KINDS.filter((k) => body[k]).length === 1 ? GOES[KINDS.find((k) => body[k]) as Kind] : GOES_ALL;
    void purge(`고른 ${chosen.length}개(${counted.join(" · ")})를 완전히 지울까요?`, goes, body);
  };

  const pickCell = (kind: Kind, id: string, label: string) => (
    <td className="pick">
      {canPick(kind) && (
        <input
          type="checkbox"
          checked={chosen.includes(keyOf(kind, id))}
          onChange={() => toggle(keyOf(kind, id))}
          aria-label={`${label} 고르기`}
        />
      )}
    </td>
  );
  const pickAll = (kind: Kind) => {
    const keys = (trash?.[kind] ?? []).map((x) => keyOf(kind, x.id));
    const all = keys.length > 0 && keys.every((k) => chosen.includes(k));
    return (
      <th className="pick">
        {canPick(kind) && (
          <input
            type="checkbox"
            checked={all}
            onChange={() => setPicked((cur) => (all ? cur.filter((c) => !keys.includes(c)) : [...new Set([...cur, ...keys])]))}
            aria-label={`${KIND_LABEL[kind]} 모두 고르기`}
          />
        )}
      </th>
    );
  };

  const total = trash ? KINDS.reduce((n, k) => n + trash[k].length, 0) : 0;
  // What emptying removes: a version or run something still points at stays.
  const removable = trash
    ? total - trash.releases.filter((r) => r.in_use.length > 0).length - trash.runs.filter((r) => r.in_use.length > 0).length
    : 0;
  const removed = emptied ? KINDS.filter((k) => emptied[k] > 0) : [];
  if (error && !trash) return <div className="error">{error}</div>;
  return (
    <>
      <div className="page-head">
        <div>
          <h1>휴지통</h1>
          <p className="muted">
            지운 프로젝트 · 데이터셋 버전 · 학습 · 모델 · 모델 종류가 여기 모입니다. 되살리면 지울 때 그대로 돌아오고, 완전히
            지우기 전까지 Hub의 파일도 남아 있습니다. 여러 개를 골라 한 번에 되살리거나 지울 수 있습니다.
          </p>
        </div>
        {isAdmin && removable > 0 && (
          <div className="head-actions">
            <button
              className="danger"
              disabled={busy}
              onClick={() => void purge(`휴지통의 ${removable}개를 모두 완전히 지울까요?`, GOES_ALL, {})}
            >
              <Icon name="trash" />
              휴지통 비우기
            </button>
          </div>
        )}
      </div>
      {chosen.length > 0 && trash && (
        <div className="pick-bar" role="toolbar" aria-label="고른 항목">
          <strong>{chosen.length}개 고름</strong>
          <button className="link small" onClick={() => setPicked([])}>
            모두 풀기
          </button>
          <span className="spacer" />
          <button
            disabled={busy}
            onClick={() => void restore(chosen.map((key) => ({ kind: key.split(":")[0] as Kind, id: key.slice(key.indexOf(":") + 1) })))}
          >
            되살리기
          </button>
          {isAdmin && (
            <button className="danger" disabled={busy} onClick={purgeChosen}>
              <Icon name="trash" size={14} />
              완전히 지우기
            </button>
          )}
        </div>
      )}
      {failed && <div className="error lines">{failed}</div>}
      {note && <div className="notice lines">{note}</div>}
      {emptied && (
        <div className="notice">
          {removed.length > 0
            ? `${removed.map((k) => `${KIND_LABEL[k]} ${emptied[k]}개`).join(" · ")}를 완전히 지웠습니다. ` +
              (emptied.files > 0
                ? `Hub 파일 ${emptied.files.toLocaleString()}개 (${bytesLabel(emptied.bytes)})를 비웠습니다.`
                : "함께 비울 Hub 파일은 없었습니다 (같은 파일을 다른 곳에서 쓰는 경우 등).")
            : "지운 것이 없습니다."}
          {emptied.kept.map((k) => (
            <div key={k.id} className="small">
              남김: {k.name} — {k.reason}
            </div>
          ))}
        </div>
      )}
      {!trash && <p className="muted">불러오는 중…</p>}
      {trash && total === 0 && <div className="card empty">휴지통이 비어 있습니다.</div>}

      {trash && trash.projects.length > 0 && (
        <Section title="프로젝트" hint="만든 사람이나 관리자가 되살릴 수 있습니다. 안의 데이터셋 · 학습 · 모델도 함께 돌아옵니다">
          <thead>
            <tr>
              {pickAll("projects")}
              <th>이름</th>
              <th className="right">버전</th>
              <th className="right">이미지</th>
              <th>지운 때</th>
              <th />
            </tr>
          </thead>
          <tbody>
            {trash.projects.map((p) => (
              <tr key={p.id}>
                {pickCell("projects", p.id, p.name)}
                <td>
                  <strong>{p.name}</strong>
                  {p.folder && <span className="muted small"> · {p.folder}</span>}
                </td>
                <td className="num right">{p.releases}</td>
                <td className="num right">{p.items.toLocaleString()}</td>
                <td>{timeAgo(p.deleted_at)}</td>
                <td className="right row-buttons">
                  <button className="small" disabled={busy} onClick={() => void restore([{ kind: "projects", id: p.id }])}>
                    되살리기
                  </button>
                  {isAdmin && (
                    <button
                      className="small danger"
                      disabled={busy}
                      onClick={() => void purge(`${p.name} 프로젝트를 완전히 지울까요?`, GOES.projects, { projects: [p.id] })}
                    >
                      완전히 지우기
                    </button>
                  )}
                </td>
              </tr>
            ))}
          </tbody>
        </Section>
      )}

      {trash && trash.releases.length > 0 && (
        <Section title="데이터셋 버전" hint={isAdmin ? undefined : "버전은 관리자가 되살릴 수 있습니다"}>
          <thead>
            <tr>
              {pickAll("releases")}
              <th>버전</th>
              <th>프로젝트</th>
              <th className="right">이미지</th>
              <th>지운 때</th>
              <th />
            </tr>
          </thead>
          <tbody>
            {trash.releases.map((r) => (
              <tr key={r.id}>
                {pickCell("releases", r.id, `${r.dataset} ${r.version}`)}
                <td>
                  <strong>
                    {r.dataset} {r.version}
                  </strong>
                </td>
                <td>
                  <a href={`#/datasets/project/${r.project.id}`}>{r.project.name}</a>
                </td>
                <td className="num right">{r.items.toLocaleString()}</td>
                <td>{timeAgo(r.deleted_at)}</td>
                <td className="right row-buttons">
                  {isAdmin && (
                    <button className="small" disabled={busy} onClick={() => void restore([{ kind: "releases", id: r.id }])}>
                      되살리기
                    </button>
                  )}
                  {isAdmin && (
                    <button
                      className="small danger"
                      disabled={busy || r.in_use.length > 0}
                      title={r.in_use.length > 0 ? `${r.in_use.join(", ")}가 이 버전을 씁니다. 계보가 끊기지 않게 남겨 둡니다.` : undefined}
                      onClick={() => void purge(`${r.dataset} ${r.version} 버전을 완전히 지울까요?`, GOES.releases, { releases: [r.id] })}
                    >
                      완전히 지우기
                    </button>
                  )}
                  {r.in_use.length > 0 && <div className="muted small">{r.in_use.join(" · ")}가 씀</div>}
                </td>
              </tr>
            ))}
          </tbody>
        </Section>
      )}

      {trash && trash.runs.length > 0 && (
        <Section title="학습" hint="시작한 사람이나 관리자가 되살릴 수 있습니다. 지표 · 파일 · 체크포인트도 함께 돌아옵니다">
          <thead>
            <tr>
              {pickAll("runs")}
              <th>이름</th>
              <th>프로젝트</th>
              <th>데이터</th>
              <th>모델 종류</th>
              <th>지운 때</th>
              <th />
            </tr>
          </thead>
          <tbody>
            {trash.runs.map((r) => (
              <tr key={r.id}>
                {pickCell("runs", r.id, r.name)}
                <td>
                  <strong>{r.name}</strong>
                </td>
                <td>{r.project.name}</td>
                <td>{r.release || "–"}</td>
                <td>{r.template}</td>
                <td>{timeAgo(r.deleted_at)}</td>
                <td className="right row-buttons">
                  <button className="small" disabled={busy} onClick={() => void restore([{ kind: "runs", id: r.id }])}>
                    되살리기
                  </button>
                  {isAdmin && (
                    <button
                      className="small danger"
                      disabled={busy || r.in_use.length > 0}
                      title={r.in_use.length > 0 ? `${r.in_use.join(", ")}가 이 학습에서 나왔습니다. 계보가 끊기지 않게 남겨 둡니다.` : undefined}
                      onClick={() => void purge(`${r.name} 학습을 완전히 지울까요?`, GOES.runs, { runs: [r.id] })}
                    >
                      완전히 지우기
                    </button>
                  )}
                  {r.in_use.length > 0 && <div className="muted small">{r.in_use.join(" · ")}가 이 학습에서 나옴</div>}
                </td>
              </tr>
            ))}
          </tbody>
        </Section>
      )}

      {trash && trash.models.length > 0 && (
        <Section title="모델" hint="관리자가 되살릴 수 있습니다. alias는 지울 때 떼었으니 필요하면 다시 붙이세요">
          <thead>
            <tr>
              {pickAll("models")}
              <th>모델</th>
              <th>프로젝트</th>
              <th>학습</th>
              <th className="right">크기</th>
              <th>지운 때</th>
              <th />
            </tr>
          </thead>
          <tbody>
            {trash.models.map((m) => (
              <tr key={m.id}>
                {pickCell("models", m.id, `${m.name} v${m.version}`)}
                <td>
                  <strong>{m.name}</strong> <span className="version">v{m.version}</span>{" "}
                  <span className="task-tag">{taskLabel(m.task)}</span>
                </td>
                <td>{m.project.name}</td>
                <td>{m.run || "–"}</td>
                <td className="num right">{bytesLabel(m.bytes)}</td>
                <td>{timeAgo(m.deleted_at)}</td>
                <td className="right row-buttons">
                  {isAdmin && (
                    <button
                      className="small"
                      disabled={busy || !m.restorable}
                      title={m.restorable ? undefined : "모델 파일이 이미 지워져 되살릴 수 없습니다"}
                      onClick={() => void restore([{ kind: "models", id: m.id }])}
                    >
                      되살리기
                    </button>
                  )}
                  {isAdmin && (
                    <button
                      className="small danger"
                      disabled={busy}
                      onClick={() => void purge(`${m.name} v${m.version} 모델을 완전히 지울까요?`, GOES.models, { models: [m.id] })}
                    >
                      완전히 지우기
                    </button>
                  )}
                  {!m.restorable && <div className="muted small">파일이 이미 지워짐</div>}
                </td>
              </tr>
            ))}
          </tbody>
        </Section>
      )}

      {trash && trash.templates.length > 0 && (
        <Section title="모델 종류" hint="관리자가 되살릴 수 있습니다. 등록한 버전이 모두 함께 돌아옵니다">
          <thead>
            <tr>
              {pickAll("templates")}
              <th>이름</th>
              <th>태스크</th>
              <th>버전</th>
              <th className="right">학습</th>
              <th>지운 때</th>
              <th />
            </tr>
          </thead>
          <tbody>
            {trash.templates.map((t) => (
              <tr key={t.id}>
                {pickCell("templates", t.id, t.title || t.name)}
                <td>
                  <strong>{t.title || t.name}</strong>
                  {t.title && t.title !== t.name && <span className="muted small"> · {t.name}</span>}
                </td>
                <td>
                  <span className="task-tag">{taskLabel(t.task)}</span>
                </td>
                <td>{t.versions.join(", ")}</td>
                <td className="num right">{t.runs}</td>
                <td>{timeAgo(t.deleted_at)}</td>
                <td className="right row-buttons">
                  {isAdmin && (
                    <button className="small" disabled={busy} onClick={() => void restore([{ kind: "templates", id: t.id }])}>
                      되살리기
                    </button>
                  )}
                  {isAdmin && (
                    <button
                      className="small danger"
                      disabled={busy}
                      onClick={() => void purge(`${t.title || t.name} 모델 종류를 완전히 지울까요?`, GOES.templates, { templates: [t.id] })}
                    >
                      완전히 지우기
                    </button>
                  )}
                </td>
              </tr>
            ))}
          </tbody>
        </Section>
      )}
    </>
  );
}
