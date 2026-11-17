import { useEffect, useMemo, useState } from "react";
import { SPLIT_COLOR, Shot } from "./DataBoard";
import { ItemViewer } from "./ItemViewer";
import { NewSetDialog } from "./Labeling";
import { ProjectPicker, useProject } from "./ProjectPicker";
import { colorFor } from "./ReleaseTabs";
import { api } from "./api";
import { type ItemPage, type ReleaseCheck, type ReleaseInfo, SPLIT_LABEL, type TreeDataset } from "./catalogTypes";
import { Icon, timeAgo, usePolling } from "./ui";

/** One label check (rules on the labels) as the Hub counts it. */
interface Rule {
  id: string;
  group: "missing" | "wrong" | "boxes";
  title: string;
  hint: string;
  images: number;
}

interface Summary {
  release_id: string;
  computed_at: string;
  checks: Rule[];
  evaluations: (ReleaseCheck & { extra: number; missed: number })[];
}

/** What the grid shows: a rule, or one side of an evaluated model's view. */
type Pick = { kind: "rule"; id: string } | { kind: "model"; mistake: "extra" | "missed" };

const GROUPS: { id: Rule["group"]; title: string; hint: string }[] = [
  { id: "missing", title: "미라벨", hint: "라벨이 빠진 이미지 · 물체" },
  { id: "wrong", title: "오라벨", hint: "클래스나 박스가 틀렸을 라벨" },
  { id: "boxes", title: "박스 오류", hint: "박스 모양이 이상한 라벨" },
];
const MODEL: Record<"extra" | "missed", { group: Rule["group"]; title: string; hint: string }> = {
  extra: { group: "missing", title: "라벨에 없는 물체", hint: "모델이 찾았는데 라벨이 없음: 빠뜨린 라벨일 수 있음" },
  missed: { group: "wrong", title: "모델이 못 찾은 라벨", hint: "라벨은 있는데 모델이 찾지 못함: 틀린 라벨일 수 있음" },
};
const PAGE = 60;
const RELEASE_KEY = "badamoa.reviewRelease";

function rememberedRelease(): string {
  try {
    return localStorage.getItem(RELEASE_KEY) ?? "";
  } catch {
    return "";
  }
}

/** 검수: images whose labels are probably missing (미라벨) or wrong (오라벨), to fix or hand out as a labeling set. */
export function Review() {
  const { projects, projectId, choose } = useProject();
  const { data: tree } = usePolling(
    () => (projectId ? api<TreeDataset[]>(`/projects/${projectId}/datasets`) : Promise.resolve([] as TreeDataset[])),
    30000,
    [projectId],
  );
  // The versions to check: the work in progress (where fixes go) first, then the versions, newest first.
  const choices = useMemo(
    () =>
      (tree ?? []).flatMap((d) =>
        d.releases
          .filter((r) => !r.deleted_at && r.status !== "building")
          .map((r) => ({ id: r.id, label: `${d.name} ${r.version}${r.status === "draft" ? ` · 고친 것 ${r.edited ?? 0}` : ""}` })),
      ),
    [tree],
  );
  const [releaseId, setReleaseId] = useState(rememberedRelease);
  useEffect(() => {
    if (choices.length && !choices.some((c) => c.id === releaseId)) setReleaseId(choices[0].id);
  }, [choices, releaseId]);
  const pickRelease = (id: string) => {
    setReleaseId(id);
    try {
      localStorage.setItem(RELEASE_KEY, id);
    } catch {
      /* not remembered */
    }
  };
  const ready = choices.some((c) => c.id === releaseId);

  return (
    <>
      <div className="page-head">
        <div>
          <h1>검수</h1>
          <p className="muted">
            라벨이 빠졌거나(미라벨) 틀렸을(오라벨) 이미지를 찾습니다. 규칙 검사는 모델 없이 바로 되고, 학습한 모델로 평가했으면 모델이 본 것도
            더합니다. 찾은 이미지는 눌러서 바로 고치거나 라벨링 묶음으로 사람에게 맡깁니다.
          </p>
        </div>
      </div>
      <div className="filters review-picker">
        <ProjectPicker projects={projects ?? null} projectId={projectId} onChange={choose} />
        <label className="project-picker">
          <span className="eyebrow">버전</span>
          <select value={ready ? releaseId : ""} onChange={(e) => pickRelease(e.target.value)} aria-label="버전">
            {choices.map((c) => (
              <option key={c.id} value={c.id}>
                {c.label}
              </option>
            ))}
          </select>
        </label>
      </div>
      {tree && choices.length === 0 && <div className="card empty">이 프로젝트에는 검사할 데이터셋 버전이 없습니다.</div>}
      {ready && <ReviewRelease key={releaseId} releaseId={releaseId} />}
    </>
  );
}

function ReviewRelease({ releaseId }: { releaseId: string }) {
  const { data: release } = usePolling(() => api<ReleaseInfo>(`/releases/${releaseId}`), 0, [releaseId]);
  const { data: summary, error, refresh } = usePolling(() => api<Summary>(`/releases/${releaseId}/qa`), 0, [releaseId]);
  const [op, setOp] = useState("");
  const [pick, setPick] = useState<Pick | null>(null);
  const [page, setPage] = useState(0);
  const [viewing, setViewing] = useState<number | null>(null);
  const [making, setMaking] = useState(false);
  useEffect(() => {
    if (summary?.evaluations.length && !summary.evaluations.some((e) => e.id === op)) setOp(summary.evaluations[0].id);
  }, [summary, op]);
  // Open on the check with the most images, so there is something to look at right away.
  useEffect(() => {
    if (pick || !summary) return;
    const best = [...summary.checks].sort((a, b) => b.images - a.images)[0];
    if (best && best.images > 0) setPick({ kind: "rule", id: best.id });
  }, [summary, pick]);
  useEffect(() => setPage(0), [pick, op]);

  const query = useMemo(() => {
    if (!pick) return "";
    const params = new URLSearchParams();
    if (pick.kind === "rule") params.set("check", pick.id);
    else if (op) {
      params.set("suspect", op);
      params.set("mistake", pick.mistake);
    }
    return params.toString();
  }, [pick, op]);
  const { data: items, refresh: refreshItems } = usePolling(
    () => (query ? api<ItemPage>(`/releases/${releaseId}/items?${query}&offset=${page * PAGE}&limit=${PAGE}`) : Promise.resolve(null)),
    0,
    [releaseId, query, page],
  );

  if (error && !summary) return <div className="error">{error}</div>;
  if (!summary || !release) return <div className="muted">검사하는 중…</div>;
  const classes = release.label_space.classes;
  const colorOf = colorFor(classes);
  const task = release.task ?? "detect";
  const evaluation = summary.evaluations.find((e) => e.id === op);
  const rule = pick?.kind === "rule" ? summary.checks.find((c) => c.id === pick.id) : undefined;
  const title = rule ? rule.title : pick?.kind === "model" ? MODEL[pick.mistake].title : "";
  const total = items?.total ?? 0;
  const pages = Math.max(1, Math.ceil(total / PAGE));
  const card = (key: string, name: string, hint: string, count: number | null, active: boolean, onClick: () => void) => (
    <button key={key} className={`check-card ${active ? "active" : ""} ${count ? "" : "zero"}`} onClick={onClick} disabled={count == null} title={hint}>
      <span className="check-count">{count == null ? "–" : count.toLocaleString()}</span>
      <span className="check-title">{name}</span>
      <span className="muted small">{hint}</span>
    </button>
  );

  return (
    <>
      <p className="muted small">
        {release.dataset.name} {release.status === "draft" ? "작업 중" : release.version} · 이미지 {(release.stats.items ?? 0).toLocaleString()}장 ·
        검사 {timeAgo(summary.computed_at)}
        {!release.fixable && " · 이 버전은 고칠 수 없어 보기만 합니다(고치기는 최신 버전이나 작업 중에서)"}
        <button className="link small" onClick={() => void refresh()}>
          다시 검사
        </button>
      </p>
      <div className="model-pick">
        <span className="eyebrow">모델 기준</span>
        {summary.evaluations.length ? (
          <select value={op} onChange={(e) => setOp(e.target.value)} aria-label="모델 평가">
            {summary.evaluations.map((e) => (
              <option key={e.id} value={e.id}>
                {e.model ? `${e.model.name} v${e.model.version}` : "모델"} · {e.release?.version} {e.split} · {e.created_at.slice(0, 10)}
              </option>
            ))}
          </select>
        ) : (
          <span className="muted small">
            평가한 모델이 없습니다. 모델 화면에서 이 데이터셋으로 평가하면(학습 이미지는 train split으로) 모델이 본 미라벨 · 오라벨도 나옵니다.
          </span>
        )}
      </div>
      <div className="check-groups">
        {GROUPS.map((g) => (
          <section key={g.id} className="check-group">
            <header>
              <strong>{g.title}</strong> <span className="muted small">{g.hint}</span>
            </header>
            {summary.checks
              .filter((c) => c.group === g.id)
              .map((c) => card(c.id, c.title, c.hint, c.images, pick?.kind === "rule" && pick.id === c.id, () => setPick({ kind: "rule", id: c.id })))}
            {(Object.keys(MODEL) as ("extra" | "missed")[])
              .filter((m) => MODEL[m].group === g.id)
              .map((m) =>
                card(
                  m,
                  `${MODEL[m].title} (모델)`,
                  MODEL[m].hint,
                  evaluation ? evaluation[m] : null,
                  pick?.kind === "model" && pick.mistake === m,
                  () => setPick({ kind: "model", mistake: m }),
                ),
              )}
          </section>
        ))}
      </div>

      {pick && (
        <section className="check-results">
          <div className="check-results-head">
            <h2>
              {title} <span className="muted">{total.toLocaleString()}장</span>
            </h2>
            <span className="spacer" />
            {release.fixable && total > 0 && (
              <button onClick={() => setMaking(true)} title="찾은 이미지를 일감으로 나눠 사람에게 맡깁니다">
                <Icon name="edit" size={13} /> 라벨링 묶음으로 만들기
              </button>
            )}
          </div>
          {pages > 1 && (
            <div className="pager">
              <button onClick={() => setPage(page - 1)} disabled={page === 0} aria-label="이전 페이지">
                ‹
              </button>
              <span className="num">
                {page + 1} / {pages}
              </span>
              <button onClick={() => setPage(page + 1)} disabled={page >= pages - 1} aria-label="다음 페이지">
                ›
              </button>
            </div>
          )}
          {items && items.items.length === 0 && <div className="card empty">찾은 이미지가 없습니다.</div>}
          <div className="thumb-grid medium">
            {items?.items.map((item, i) => (
              <button key={item.id} className="thumb" onClick={() => setViewing(page * PAGE + i)}>
                <span className="thumb-image">
                  {item.thumb_url ? (
                    <Shot shot={{ url: item.thumb_url, width: item.width, height: item.height, boxes: item.boxes }} colorOf={colorOf} task={task} />
                  ) : (
                    <Icon name="images" size={22} />
                  )}
                  <span className="split-dot" style={{ background: SPLIT_COLOR[item.split] }} title={SPLIT_LABEL[item.split] ?? item.split} />
                </span>
                <span className="thumb-name" title={item.file_name}>
                  {item.file_name}
                </span>
                <span className="thumb-meta">
                  {item.qa?.note && <span className="tag">{item.qa.note}</span>}
                  {item.check && pick.kind === "model" && (
                    <span className="tag">{pick.mistake === "extra" ? `라벨에 없음 ${item.check.extra}` : `못 찾음 ${item.check.missed}`}</span>
                  )}
                </span>
              </button>
            ))}
          </div>
        </section>
      )}

      {viewing != null && items && (
        <ItemViewer
          releaseId={releaseId}
          filterQuery={query}
          start={viewing}
          total={total}
          colorOf={colorOf}
          classes={classes}
          task={task}
          checkOp={pick?.kind === "model" ? op : undefined}
          qaTitle={title}
          onSaved={() => {
            void refresh();
            void refreshItems();
          }}
          onClose={() => setViewing(null)}
        />
      )}
      {making && pick && (
        <NewSetDialog
          release={release}
          filter={pick.kind === "rule" ? { check: pick.id } : { suspect: op, mistake: pick.mistake }}
          filtered={total}
          describe={title}
          onClose={() => setMaking(false)}
        />
      )}
    </>
  );
}
