import { type FormEvent, useEffect, useMemo, useState } from "react";
import { type BoardProject, useBoardFilter } from "./Boards";
import { Collage, SPLIT_COLOR, Shot, rememberProject } from "./DataBoard";
import { ItemViewer } from "./ItemViewer";
import { PlacePicker, placeReady, usePlaces } from "./ModelOps";
import type { ItemFilter } from "./ReleaseMaker";
import { NewSetDialog } from "./Labeling";
import { OrderReset, SortHandle, SortItem, Sortable, useOrder } from "./Order";
import { useProject } from "./ProjectPicker";
import { colorFor } from "./ReleaseTabs";
import { api } from "./api";
import { AliasChip } from "./Annotate";
import {
  type CheckModel,
  type ItemPage,
  type ReleaseCheck,
  type ReleaseInfo,
  SPLIT_LABEL,
  STATE_HELP,
  STATE_HINT,
  STATE_LABEL,
  type TreeDataset,
  classColor,
  taskLabel,
} from "./catalogTypes";
import { METRIC_LABEL, metric } from "./trainTypes";
import { Icon, Modal, STATUS_LABEL, Stat, timeAgo, usePolling } from "./ui";

/** One label check (rules on the labels) as the Hub counts it. */
interface Rule {
  id: string;
  group: "missing" | "wrong" | "boxes";
  title: string;
  hint: string;
  images: number;
  let_be: number; // images someone judged right ("문제 없음"): not counted in ``images``
}

interface Summary {
  release_id: string;
  computed_at: string;
  checks: Rule[];
  evaluations: (ReleaseCheck & { extra: number; missed: number; extra_let_be: number; missed_let_be: number })[];
  pending: { id: string; job_id: string; status: string; split: string; model: CheckModel | null; created_at: string }[];
}

/** A model of the project and whether it can check this version (evaluate on it). */
interface EvalChoice extends CheckModel {
  arch: string;
  metric: string;
  value: number | null;
  created_at: string;
  trained_on: boolean; // trained on this version: its train split was learned
  fits: boolean;
  why: string; // why it cannot, when it cannot
}

interface EvalChoices {
  release_id: string;
  evaluate: { id: string; version: string } | null; // where it runs: the version, or a work in progress's base
  task: string;
  splits: Record<string, number>;
  models: EvalChoice[];
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

/** A dataset's newest labels as 검수's screens show them. */
interface ReviewCard {
  project: BoardProject;
  dataset: { id: string; name: string };
  task: string;
  release: { id: string; version: string; draft: boolean; base: string; edited: number };
  items: number;
  checked: boolean; // a big version whose checks are out of date is checked when it is opened
  computed_at: string | null;
  groups: Record<Rule["group"] | "any", number> | null;
  checks: { id: string; title: string; group: Rule["group"]; images: number }[];
  flagged_shots: boolean; // the pictures are of flagged images
  collage: Shot[];
  evaluation: { id: string; model: { id: string; name: string; version: number } | null; images: number; extra: number; missed: number } | null;
}

interface ReviewBoardInfo {
  datasets: ReviewCard[]; // the most problems first
  totals: Totals;
}

interface Totals {
  datasets: number;
  checked: number;
  flagged: number;
  missing: number;
  wrong: number;
  boxes: number;
  evaluated: number;
}

/** A project's datasets on the board, in the board's order: the project with the worst dataset comes first. */
interface ReviewProject {
  project: BoardProject;
  datasets: ReviewCard[];
}

function byProject(cards: ReviewCard[]): ReviewProject[] {
  const out = new Map<string, ReviewProject>();
  for (const c of cards) {
    const group = out.get(c.project.id) ?? { project: c.project, datasets: [] };
    group.datasets.push(c);
    out.set(c.project.id, group);
  }
  return [...out.values()];
}

/** The board's numbers over some of its datasets: images with each kind of problem, among those checked. */
function totalsOf(cards: ReviewCard[]): Totals {
  const checked = cards.filter((c) => c.checked && c.groups);
  const sum = (g: Rule["group"] | "any") => checked.reduce((n, c) => n + (c.groups?.[g] ?? 0), 0);
  return {
    datasets: cards.length,
    checked: checked.length,
    flagged: sum("any"),
    missing: sum("missing"),
    wrong: sum("wrong"),
    boxes: sum("boxes"),
    evaluated: cards.filter((c) => c.evaluation).length,
  };
}

const projectOfGroup = (p: ReviewProject) => p.project;
const groupId = (p: ReviewProject) => p.project.id;
const datasetId = (c: ReviewCard) => c.dataset.id;
const LINES = 5; // datasets listed on a project's card; the rest are a click away
const versionName = (c: ReviewCard) => (c.release.draft ? "작업 중" : c.release.version);

/** 검수: projects (in folders, like the other first screens), then a project's datasets, then a version's checks. */
export function Review({ sub, arg }: { sub?: string; arg?: string }) {
  if (sub === "project" && arg) return <ReviewProjectPage key={arg} projectId={arg} />;
  return sub ? <ReviewPage key={sub} releaseId={sub} /> : <ReviewBoard />;
}

function useReviewBoard() {
  return usePolling(() => api<ReviewBoardInfo>("/boards/review"), 30000);
}

function ReviewStats({ totals: t }: { totals: Totals | undefined }) {
  return (
    <div className="stats five">
      <Stat icon="database" label="데이터셋" value={t?.datasets ?? "–"} sub={t ? `검사함 ${t.checked}` : " "} />
      {GROUPS.map((g) => (
        <Stat key={g.id} icon={g.id === "missing" ? "images" : g.id === "wrong" ? "alert" : "box"} label={g.title} value={(t?.[g.id] ?? 0).toLocaleString()} sub={g.hint} />
      ))}
      <Stat icon="chart" label="모델로 평가함" value={t?.evaluated ?? "–"} sub="모델이 본 미라벨 · 오라벨도 셈" />
    </div>
  );
}

/** Images per kind of problem: one dataset's, or a project's datasets' together. */
function ReviewCounts({ groups }: { groups: Record<Rule["group"], number> }) {
  return (
    <div className="review-counts">
      {GROUPS.map((g) => (
        <span key={g.id} className={`review-count ${groups[g.id] ? `has-${g.id}` : "zero"}`} title={g.hint}>
          <strong>{groups[g.id].toLocaleString()}</strong>
          {g.title}
        </span>
      ))}
    </div>
  );
}

function ReviewBoard() {
  const { data: board, error } = useReviewBoard();
  const projects = useMemo(() => byProject(board?.datasets ?? []), [board]);
  const order = useOrder("review.projects", projects, groupId);
  const { shown, tools } = useBoardFilter(board ? order.ordered : null, projectOfGroup);
  if (error && !board) return <div className="error">{error}</div>;
  return (
    <>
      <div className="page-head">
        <div>
          <h1>검수</h1>
          <p className="muted">
            데이터셋마다 최신 라벨(작업 중이 있으면 작업 중)에서 라벨이 빠졌거나(미라벨) 틀렸을(오라벨) 이미지 수를 프로젝트별로 봅니다. 문제가
            많은 프로젝트가 먼저 나오고(끌어서 순서를 바꿀 수 있음), 프로젝트를 누르면 데이터셋별로, 데이터셋을 누르면 그 이미지를 보며 고치거나
            라벨링 묶음으로 맡깁니다.
          </p>
        </div>
        <div className="head-actions">
          <OrderReset order={order} />
        </div>
      </div>
      <ReviewStats totals={board?.totals} />
      {tools}
      {!board && <p className="muted">불러오는 중…</p>}
      {board && projects.length === 0 && <div className="card empty">검사할 데이터셋 버전이 없습니다.</div>}
      <Sortable ids={shown.map(groupId)} onMove={order.move}>
        <div className="card-grid review-grid">
          {shown.map((p) => (
            <SortItem key={p.project.id} id={p.project.id}>
              <ProjectReviewCard group={p} />
            </SortItem>
          ))}
        </div>
      </Sortable>
    </>
  );
}

/** A project on the board: its datasets' problems together, and each dataset a click away. */
function ProjectReviewCard({ group }: { group: ReviewProject }) {
  const { project, datasets } = group;
  const t = totalsOf(datasets);
  const waiting = t.datasets - t.checked;
  const images = datasets.reduce((n, c) => n + c.items, 0);
  const lines = useOrder(`review.datasets.${project.id}`, datasets, datasetId).ordered; // as on the project's own screen
  return (
    <div className="data-card review-card">
      <a className="data-card-link" href={`#/review/project/${project.id}`} onClick={() => rememberProject(project.id)}>
        <Collage shots={datasets[0].collage} colorOf={classColor} />
        <div className="data-card-body">
          <div className="data-card-title">
            <span className="with-grip"><SortHandle /><strong>{project.name}</strong></span>
          </div>
          <div className="muted small">
            데이터셋 {datasets.length} · 이미지 {images.toLocaleString()}
          </div>
          {t.checked > 0 ? (
            <ReviewCounts groups={t} />
          ) : (
            <div className="muted small">이미지가 많아 아직 검사하지 않았습니다. 데이터셋을 누르면 검사합니다.</div>
          )}
          <div className="muted small">
            {t.checked > 0 && waiting > 0 && `검사 전 데이터셋 ${waiting} · `}
            {t.evaluated > 0 ? `모델로 평가한 데이터셋 ${t.evaluated}` : "평가한 모델 없음"}
          </div>
        </div>
      </a>
      <div className="board-lines">
        {lines.slice(0, LINES).map((c) => (
          <a key={c.dataset.id} href={`#/review/${c.release.id}`} className="board-line" onClick={() => rememberProject(project.id)}>
            <Icon name="database" size={12} />
            <span className="ellipsis">
              {c.dataset.name} <span className="muted">{versionName(c)}</span>
            </span>
            <span className={`num${c.groups?.any ? "" : " muted"}`} title="검사에 걸린 이미지">
              {c.checked && c.groups ? c.groups.any.toLocaleString() : "검사 전"}
            </span>
          </a>
        ))}
        {datasets.length > LINES && <div className="muted small board-foot">데이터셋 {datasets.length - LINES}개 더</div>}
      </div>
    </div>
  );
}

/** A project's datasets, each with its newest labels' problems; a dataset opens its version's checks. */
function ReviewProjectPage({ projectId }: { projectId: string }) {
  const { data: board, error } = useReviewBoard();
  const { projects } = useProject();
  useEffect(() => rememberProject(projectId), [projectId]);
  const mine = useMemo(() => (board?.datasets ?? []).filter((c) => c.project.id === projectId), [board, projectId]);
  const order = useOrder(`review.datasets.${projectId}`, mine, datasetId);
  if (error && !board) return <div className="error">{error}</div>;
  const datasets = order.ordered;
  const project = datasets[0]?.project ?? projects?.find((p) => p.id === projectId);
  return (
    <>
      <nav className="page-crumbs">
        <a href="#/review">검수</a>
        {project?.folder && <span>{project.folder}</span>}
        <strong>{project?.name ?? "…"}</strong>
      </nav>
      <div className="page-head">
        <div>
          <h1>{project?.name ?? "검수"}</h1>
          <p className="muted">
            이 프로젝트의 데이터셋마다 최신 라벨(작업 중이 있으면 작업 중)의 문제 수입니다. 문제가 많은 데이터셋이 먼저 나오고(끌어서 순서를 바꿀
            수 있음), 누르면 그 버전의 검수 화면이 열립니다. 다른 버전은 거기서 고릅니다.
          </p>
        </div>
        <div className="head-actions">
          <OrderReset order={order} />
        </div>
      </div>
      <ReviewStats totals={board ? totalsOf(datasets) : undefined} />
      {!board && <p className="muted">불러오는 중…</p>}
      {board && datasets.length === 0 && <div className="card empty">이 프로젝트에는 검사할 데이터셋 버전이 없습니다.</div>}
      <Sortable ids={datasets.map(datasetId)} onMove={order.move}>
        <div className="card-grid review-grid">
          {datasets.map((c) => (
            <SortItem key={c.dataset.id} id={c.dataset.id}>
              <DatasetReviewCard card={c} />
            </SortItem>
          ))}
        </div>
      </Sortable>
    </>
  );
}

function DatasetReviewCard({ card: c }: { card: ReviewCard }) {
  return (
    <a className="data-card data-card-link review-card" href={`#/review/${c.release.id}`}>
      <Collage shots={c.collage} colorOf={classColor} />
      <div className="data-card-body">
        <div className="data-card-title">
          <span className="with-grip"><SortHandle /><strong>{c.dataset.name}</strong></span>
          {c.release.draft && <span className="tag accent">작업 중</span>}
        </div>
        <div className="muted small">
          {c.release.draft ? `${c.release.base}에서 고친 것 ${c.release.edited}` : c.release.version} · 이미지 {c.items.toLocaleString()}
        </div>
        {c.checked && c.groups ? (
          <>
            <ReviewCounts groups={c.groups} />
            {c.checks.length > 0 ? (
              <div className="muted small">
                {c.checks
                  .slice(0, 2)
                  .map((k) => `${k.title} ${k.images.toLocaleString()}`)
                  .join(" · ")}
                {c.checks.length > 2 ? ` 외 ${c.checks.length - 2}` : ""}
              </div>
            ) : (
              <div className="muted small">규칙 검사에 걸린 이미지가 없습니다</div>
            )}
          </>
        ) : (
          <div className="muted small">이미지가 많아 아직 검사하지 않았습니다. 누르면 검사합니다.</div>
        )}
        {c.evaluation ? (
          <div className="small">
            {c.evaluation.model ? `${c.evaluation.model.name} v${c.evaluation.model.version}` : "모델"}: 라벨에 없는 물체{" "}
            <strong>{c.evaluation.extra.toLocaleString()}</strong>장 · 못 찾은 라벨 <strong>{c.evaluation.missed.toLocaleString()}</strong>장
          </div>
        ) : (
          <div className="muted small">평가한 모델 없음</div>
        )}
        <div className="muted small">
          {c.flagged_shots ? "사진은 걸린 이미지 · " : ""}
          {c.computed_at ? `검사 ${timeAgo(c.computed_at)}` : "검사 전"}
        </div>
      </div>
    </a>
  );
}

/** One version's checks: the images each found, to fix or hand out. The dataset's other versions are a pick away. */
function ReviewPage({ releaseId }: { releaseId: string }) {
  const { projects } = useProject();
  const { data: release, error } = usePolling(() => api<ReleaseInfo>(`/releases/${releaseId}`), 0, [releaseId]);
  const projectId = release?.project_id ?? "";
  const datasetId = release?.dataset.id ?? "";
  const { data: tree } = usePolling(
    () => (projectId ? api<TreeDataset[]>(`/projects/${projectId}/datasets`) : Promise.resolve([] as TreeDataset[])),
    30000,
    [projectId],
  );
  useEffect(() => {
    if (projectId) rememberProject(projectId);
  }, [projectId]);
  // This dataset's versions to check: its work in progress (where fixes go) first, then its versions, newest first.
  const choices = useMemo(
    () =>
      (tree ?? [])
        .filter((d) => d.id === datasetId)
        .flatMap((d) =>
          d.releases
            .filter((r) => !r.deleted_at && r.status !== "building")
            .map((r) => ({ id: r.id, label: r.status === "draft" ? `작업 중 · 고친 것 ${r.edited ?? 0}` : r.version })),
        ),
    [tree, datasetId],
  );
  if (error && !release) return <div className="error">{error}</div>;
  const project = projects?.find((p) => p.id === projectId);
  const title = release ? (release.status === "draft" ? "작업 중" : release.version) : "";

  return (
    <>
      <nav className="page-crumbs">
        <a href="#/review">검수</a>
        {project?.folder && <span>{project.folder}</span>}
        {projectId ? <a href={`#/review/project/${projectId}`}>{project?.name ?? "프로젝트"}</a> : <span>…</span>}
        <strong>{release ? `${release.dataset.name} ${title}` : "…"}</strong>
      </nav>
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
        <label className="project-picker">
          <span className="eyebrow">{release ? `${release.dataset.name} 버전` : "버전"}</span>
          <select value={releaseId} onChange={(e) => (window.location.hash = `#/review/${e.target.value}`)} aria-label="버전">
            {!choices.some((c) => c.id === releaseId) && <option value={releaseId}>{title || "…"}</option>}
            {choices.map((c) => (
              <option key={c.id} value={c.id}>
                {c.label}
              </option>
            ))}
          </select>
        </label>
      </div>
      <ReviewRelease releaseId={releaseId} />
    </>
  );
}

function ReviewRelease({ releaseId }: { releaseId: string }) {
  const { data: release } = usePolling(() => api<ReleaseInfo>(`/releases/${releaseId}`), 0, [releaseId]);
  const [watching, setWatching] = useState(false); // an evaluation is going: look again until it ends
  const { data: summary, error, refresh } = usePolling(() => api<Summary>(`/releases/${releaseId}/qa`), watching ? 4000 : 0, [releaseId]);
  const [op, setOp] = useState("");
  const [choosing, setChoosing] = useState(false);
  // The evaluation started here; ``ended`` once its job is over and the summary has been read again since.
  const [started, setStarted] = useState<{ id: string; job_id: string; ended?: boolean } | null>(null);
  const [failed, setFailed] = useState<string | null>(null);
  useEffect(() => setWatching(!!started || (summary?.pending.length ?? 0) > 0), [started, summary]);
  // The evaluation started here ended: show its view, or why it gave none.
  useEffect(() => {
    if (!started || !summary) return;
    if (summary.evaluations.some((e) => e.id === started.id)) {
      setOp(started.id);
      setStarted(null);
      return;
    }
    if (summary.pending.some((p) => p.id === started.id)) return;
    // Listed neither way: the summary may be older than the start. Ask the job itself.
    void api<{ status: string; error: string | null }>(`/jobs/${started.job_id}`).then((job) => {
      if (job.status === "queued" || job.status === "running") return; // still going: the next look shows it
      if (!started.ended) {
        setStarted({ ...started, ended: true }); // over: read the summary once more before saying anything
        void refresh();
        return;
      }
      setStarted(null);
      setFailed(
        job.status === "succeeded"
          ? "평가는 끝났지만 이미지별 예측을 남기지 않아 모델 기준으로 쓸 수 없습니다(템플릿의 평가 코드가 bm.log_predictions를 부르지 않음)."
          : `평가가 끝나지 못했습니다: ${job.error ?? STATUS_LABEL[job.status] ?? job.status}`,
      );
    });
  }, [started, summary]);
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
  const [showLetBe, setShowLetBe] = useState(false); // the images someone judged right, instead
  useEffect(() => setShowLetBe(false), [pick, op]);
  // The found images narrowed like the dataset screen's: split, label state, class, site, camera, file name.
  const [filter, setFilter] = useState<ItemFilter>({});
  const narrowed = Object.values(filter).filter(Boolean).length;
  const setOne = (key: keyof ItemFilter, value: string) => setFilter((f) => ({ ...f, [key]: value || undefined }));
  const query = useMemo(() => {
    if (!pick) return "";
    const params = new URLSearchParams();
    if (pick.kind === "rule") params.set("check", pick.id);
    else if (op) {
      params.set("suspect", op);
      params.set("mistake", pick.mistake);
    }
    if (showLetBe) params.set("let_be", "true");
    for (const [key, value] of Object.entries(filter)) {
      if (value) params.set(key, key === "split" && value === "none" ? "" : value);
    }
    return params.toString();
  }, [pick, op, showLetBe, filter]);
  useEffect(() => setPage(0), [query]);
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
  // what someone judged right for the check in view: listed apart
  const letBeCount = rule ? rule.let_be : pick?.kind === "model" && evaluation ? evaluation[`${pick.mistake}_let_be`] : 0;
  const total = items?.total ?? 0;
  const pages = Math.max(1, Math.ceil(total / PAGE));
  const card = (key: string, name: string, hint: string, count: number, active: boolean, onClick: () => void) => (
    <button key={key} className={`check-card ${active ? "active" : ""} ${count ? "" : "zero"}`} onClick={onClick} title={hint}>
      <span className="check-count">{count.toLocaleString()}</span>
      <span className="check-title">{name}</span>
      <span className="muted small">{hint}</span>
    </button>
  );
  // A model's check is not worked out on the spot like the rules: it needs an evaluation, started from here.
  const going = summary.pending[0];
  const modelCard = (m: "extra" | "missed") => {
    const count = evaluation ? evaluation[m] : null;
    const active = pick?.kind === "model" && pick.mistake === m;
    return (
      <button
        key={m}
        className={`check-card model-check ${active ? "active" : ""} ${count ? "" : "zero"} ${count == null ? "waiting" : ""}`}
        onClick={() => (count == null ? !going && setChoosing(true) : setPick({ kind: "model", mistake: m }))}
        title={MODEL[m].hint}
      >
        <span className="check-count">{count != null ? count.toLocaleString() : <Icon name={going ? "clock" : "box"} size={18} />}</span>
        <span className="check-title">
          {MODEL[m].title} <span className="tag model-tag">모델</span>
        </span>
        <span className="muted small">
          {count != null
            ? MODEL[m].hint
            : going
              ? `${going.model ? `${going.model.name} v${going.model.version}` : "모델"} 평가 중 — 끝나면 셉니다`
              : "모델이 있어야 셉니다. 눌러서 모델로 검사"}
        </span>
      </button>
    );
  };
  // Every group's cards share rows: the columns are as tall as each other and their cards line up.
  const slots = Math.max(
    ...GROUPS.map((g) => summary.checks.filter((c) => c.group === g.id).length + (Object.values(MODEL).some((m) => m.group === g.id) ? 1 : 0)),
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
      <section className="card model-basis">
        <div className="model-basis-head">
          <span className="eyebrow">모델 기준</span>
          {summary.evaluations.length > 1 && (
            <select value={op} onChange={(e) => setOp(e.target.value)} aria-label="기준으로 쓸 모델 평가">
              {summary.evaluations.map((e) => (
                <option key={e.id} value={e.id}>
                  {e.model ? `${e.model.name} v${e.model.version}` : "모델"} · {e.release?.version} {e.split} · {e.created_at.slice(0, 10)}
                </option>
              ))}
            </select>
          )}
          <span className="spacer" />
          <button className="small" onClick={() => setChoosing(true)}>
            <Icon name="box" size={13} /> {summary.evaluations.length ? "다른 모델로 검사" : "모델로 검사하기"}
          </button>
        </div>
        {evaluation ? (
          <div className="model-basis-line">
            <ModelName model={evaluation.model} />
            <span className="muted small">
              {evaluation.release?.version} {evaluation.split} 이미지 {evaluation.images.toLocaleString()}장을 평가 · {timeAgo(evaluation.created_at)}
            </span>
          </div>
        ) : (
          summary.pending.length === 0 && (
            <p className="muted small">
              아직 모델로 검사하지 않았습니다. "모델로 검사하기"에서 이 데이터셋에 맞는 모델(같은 태스크 · 클래스)을 고르면, 모델이 찾았는데 라벨이
              없는 물체와 라벨은 있는데 모델이 못 찾은 것도 셉니다.
            </p>
          )
        )}
        {summary.pending.map((p) => (
          <div key={p.id} className="model-basis-line pending">
            <span className="badge badge-running">
              <i />
              {p.status === "running" ? "평가 중" : "대기"}
            </span>
            <ModelName model={p.model} />
            <span className="muted small">
              {p.split} · {timeAgo(p.created_at)} 시작 · 끝나면 여기에 기준으로 나옵니다
            </span>
          </div>
        ))}
        {failed && <div className="error small">{failed}</div>}
      </section>
      <div className="check-groups" style={{ gridTemplateRows: `auto repeat(${slots}, auto)` }}>
        {GROUPS.map((g) => (
          <section key={g.id} className="check-group" style={{ gridRow: `1 / span ${slots + 1}` }}>
            <header>
              <strong>{g.title}</strong> <span className="muted small">{g.hint}</span>
            </header>
            {summary.checks
              .filter((c) => c.group === g.id)
              .map((c) => card(c.id, c.title, c.hint, c.images, pick?.kind === "rule" && pick.id === c.id, () => setPick({ kind: "rule", id: c.id })))}
            {(Object.keys(MODEL) as ("extra" | "missed")[]).filter((m) => MODEL[m].group === g.id).map(modelCard)}
          </section>
        ))}
      </div>

      {pick && (
        <section className="check-results">
          <div className="check-results-head">
            <h2>
              {title}
              {showLetBe && " · 문제 없음으로 둔 것"} <span className="muted">{total.toLocaleString()}장{narrowed ? " (거른 것)" : ""}</span>
              {pick.kind === "model" && evaluation?.model && (
                <span className="muted small model-of">
                  {" "}
                  · {evaluation.model.name} v{evaluation.model.version} 기준
                </span>
              )}
            </h2>
            <span className="spacer" />
            {(letBeCount > 0 || showLetBe) && (
              <button className="link small" onClick={() => setShowLetBe(!showLetBe)}>
                {showLetBe ? "걸린 것 보기" : `문제 없음으로 둔 것 ${letBeCount.toLocaleString()}장`}
              </button>
            )}
            {release.fixable && total > 0 && !showLetBe && (
              <button onClick={() => setMaking(true)} title="찾은 이미지를 일감으로 나눠 사람에게 맡깁니다">
                <Icon name="edit" size={13} /> 라벨링 묶음으로 만들기
              </button>
            )}
          </div>
          <ReviewFilters release={release} filter={filter} onChange={setOne} onClear={() => setFilter({})} />
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
          qaCheck={pick?.kind === "rule" ? pick.id : pick?.mistake}
          qaLetBe={showLetBe}
          onSaved={() => {
            void refresh();
            void refreshItems();
          }}
          onClose={() => setViewing(null)}
        />
      )}
      {choosing && (
        <CheckWithModel
          release={release}
          onClose={() => setChoosing(false)}
          onStarted={(started) => {
            setChoosing(false);
            setFailed(null);
            setStarted(started);
            void refresh();
          }}
        />
      )}
      {making && pick && (
        <NewSetDialog
          release={release}
          filter={{
            ...Object.fromEntries(Object.entries(filter).map(([k, v]) => [k, k === "split" && v === "none" ? "" : v])),
            ...(pick.kind === "rule" ? { check: pick.id } : { suspect: op, mistake: pick.mistake }),
          }}
          filtered={total}
          describe={title}
          onClose={() => setMaking(false)}
        />
      )}
    </>
  );
}

/** The dataset screen's filters over the images a check found: split, label state, class, site, camera, name. */
function ReviewFilters({
  release,
  filter,
  onChange,
  onClear,
}: {
  release: ReleaseInfo;
  filter: ItemFilter;
  onChange: (key: keyof ItemFilter, value: string) => void;
  onClear: () => void;
}) {
  const stats = release.stats;
  const splits = Object.keys(stats.splits ?? {}).sort();
  const narrowed = Object.values(filter).some(Boolean);
  return (
    <div className="filters review-filters">
      <select value={filter.split ?? ""} onChange={(e) => onChange("split", e.target.value)} aria-label="split">
        <option value="">모든 split</option>
        {splits.map((name) => (
          <option key={name || "none"} value={name === "" ? "none" : name}>
            {SPLIT_LABEL[name] ?? name}
          </option>
        ))}
      </select>
      <select value={filter.state ?? ""} onChange={(e) => onChange("state", e.target.value)} aria-label="라벨 상태">
        <option value="">모든 라벨 상태</option>
        {Object.entries(STATE_LABEL).map(([id, label]) => (
          <option key={id} value={id} title={STATE_HELP[id]}>
            {STATE_HINT[id] ? `${label} · ${STATE_HINT[id]}` : label}
          </option>
        ))}
      </select>
      <select value={filter.class_id ?? ""} onChange={(e) => onChange("class_id", e.target.value)} aria-label="클래스">
        <option value="">모든 클래스</option>
        {release.label_space.classes.map((c) => (
          <option key={c.id} value={c.id}>
            {c.name}
          </option>
        ))}
      </select>
      {(stats.sites?.length ?? 0) > 0 && (
        <select value={filter.site ?? ""} onChange={(e) => onChange("site", e.target.value)} aria-label="사이트">
          <option value="">모든 사이트</option>
          {stats.sites?.map((s) => <option key={s}>{s}</option>)}
        </select>
      )}
      {(stats.cameras?.length ?? 0) > 0 && (
        <select value={filter.camera ?? ""} onChange={(e) => onChange("camera", e.target.value)} aria-label="카메라">
          <option value="">모든 카메라</option>
          {stats.cameras?.map((c) => <option key={c}>{c}</option>)}
        </select>
      )}
      <input value={filter.q ?? ""} onChange={(e) => onChange("q", e.target.value)} placeholder="파일 이름에 포함…" aria-label="파일 이름" />
      {narrowed && (
        <button className="link small" onClick={onClear}>
          필터 지우기
        </button>
      )}
    </div>
  );
}

/** A model by name and version, with its task and aliases; a link to it. */
function ModelName({ model }: { model: CheckModel | null }) {
  if (!model) return <strong>지운 모델</strong>;
  return (
    <span className="model-name">
      <a href={`#/models/${model.id}`}>
        <strong>
          {model.name} v{model.version}
        </strong>
      </a>
      <span className="tag task-tag">{taskLabel(model.task)}</span>
      {model.aliases.map((a) => (
        <AliasChip key={a} alias={a} />
      ))}
    </span>
  );
}

/**
 * Pick a model to check this version's labels with: only the project's models of the dataset's task whose
 * classes the version has can be picked; the others say why not. The evaluation runs on the version (a work
 * in progress: on its base), over the splits ticked (one basis for all of them), on the server holding the images
 * or one picked (which receives them first).
 */
function CheckWithModel({
  release,
  onClose,
  onStarted,
}: {
  release: ReleaseInfo;
  onClose: () => void;
  onStarted: (op: { id: string; job_id: string }) => void;
}) {
  const { data: choices, error: loadError } = usePolling(() => api<EvalChoices>(`/releases/${release.id}/eval-models`), 0, [release.id]);
  const [model, setModel] = useState("");
  const [picked, setPicked] = useState<string[] | null>(null); // null: not touched yet (every split)
  const [nodeId, setNodeId] = useState(""); // "": the server holding the version's images
  const [busy, setBusy] = useState(false);
  const [error, setError] = useState<string | null>(null);
  const fitting = choices?.models.filter((m) => m.fits) ?? [];
  const others = choices?.models.filter((m) => !m.fits) ?? [];
  const splits = ["train", "val", "test"].filter((k) => choices?.splits[k]);
  // Every split by default: a label check wants to see as many images as it can.
  const chosenSplits = picked ?? splits;
  const toggleSplit = (k: string) =>
    setPicked(splits.filter((s) => (s === k ? !chosenSplits.includes(k) : chosenSplits.includes(s))));
  const images = chosenSplits.reduce((n, k) => n + (choices?.splits[k] ?? 0), 0);
  useEffect(() => {
    if (!choices) return;
    setModel((current) => current || (choices.models.find((m) => m.fits)?.id ?? ""));
  }, [choices]);
  useEffect(() => setNodeId(""), [model]); // another model: its own servers, from the default
  const chosen = choices?.models.find((m) => m.id === model);
  const { data: places, error: placesError } = usePlaces(model, "eval", chosen?.fits ? (choices?.evaluate?.id ?? "") : "");

  const submit = async (e: FormEvent) => {
    e.preventDefault();
    if (!choices?.evaluate || !chosen) return;
    setBusy(true);
    setError(null);
    try {
      const op = await api<{ id: string; job_id: string }>(`/models/${chosen.id}/eval`, {
        body: { release_id: choices.evaluate.id, split: chosenSplits[0], splits: chosenSplits, node_id: nodeId },
      });
      onStarted(op);
    } catch (err) {
      setError(err instanceof Error ? err.message : String(err));
      setBusy(false);
    }
  };

  return (
    <Modal title="모델로 검사하기" onClose={onClose}>
      <form onSubmit={submit}>
        <p className="muted">
          고른 모델로 이 데이터셋을 평가해, 모델이 찾았는데 라벨이 없는 물체(미라벨일 수 있음)와 라벨은 있는데 모델이 못 찾은 것(오라벨일 수
          있음)을 셉니다. {taskLabel(choices?.task)} 데이터셋이라 {taskLabel(choices?.task)} 모델 가운데 이 버전의 클래스로 학습한 것만 고를 수
          있습니다.
          {release.status === "draft" && choices?.evaluate && ` 작업 중은 바탕 버전(${choices.evaluate.version})으로 평가합니다.`}
        </p>
        {loadError && <div className="error">{loadError}</div>}
        {choices && choices.models.length === 0 && (
          <div className="notice">
            이 프로젝트에는 아직 모델이 없습니다. <a href={`#/training/project/${release.project_id}`}>학습</a>을 하면 끝날 때 모델이 등록됩니다.
          </div>
        )}
        {choices && choices.models.length > 0 && fitting.length === 0 && (
          <div className="notice">이 데이터셋을 검사할 수 있는 모델이 없습니다. 아래에서 모델마다 이유를 볼 수 있습니다.</div>
        )}
        {fitting.length > 0 && (
          <div className="eval-choices" role="radiogroup" aria-label="검사할 모델">
            {fitting.map((m) => (
              <label key={m.id} className={`eval-choice ${model === m.id ? "active" : ""}`}>
                <input type="radio" name="model" value={m.id} checked={model === m.id} onChange={() => setModel(m.id)} />
                <ModelName model={m} />
                <span className="muted small">
                  {m.arch ? `${m.arch} · ` : ""}
                  {METRIC_LABEL[m.metric] ?? m.metric} {metric(m.value)} · {timeAgo(m.created_at)}
                </span>
              </label>
            ))}
          </div>
        )}
        {others.length > 0 && (
          <details className="eval-others">
            <summary className="muted small">쓸 수 없는 모델 {others.length}개</summary>
            {others.map((m) => (
              <div key={m.id} className="eval-choice disabled">
                <ModelName model={m} />
                <span className="muted small">{m.why}</span>
              </div>
            ))}
          </details>
        )}
        {fitting.length > 0 && (
          <fieldset className="eval-splits">
            <legend>평가할 부분</legend>
            {splits.map((k) => (
              <label key={k} className="check-line">
                <input type="checkbox" checked={chosenSplits.includes(k)} onChange={() => toggleSplit(k)} />
                {k} <span className="muted small">{(choices?.splits[k] ?? 0).toLocaleString()}장</span>
              </label>
            ))}
            <span className="muted small">
              {chosenSplits.length > 1
                ? `${chosenSplits.join(" · ")} ${images.toLocaleString()}장을 차례로 평가해 한 기준으로 봅니다`
                : chosenSplits.length === 0
                  ? "하나 이상 고르세요"
                  : ""}
            </span>
          </fieldset>
        )}
        {chosen?.fits && <PlacePicker places={places} error={placesError} value={nodeId} onChange={setNodeId} />}
        {chosen?.trained_on && chosenSplits.includes("train") && (
          <div className="notice small">
            이 모델이 학습한 이미지입니다. 모델이 라벨을 그대로 외웠을 수 있어 틀린 라벨을 덜 찾습니다. 다른 데이터로 학습한 모델이나 val로 보면 더
            잘 드러납니다.
          </div>
        )}
        {error && <div className="error">{error}</div>}
        <div className="modal-actions">
          <button type="button" onClick={onClose}>
            취소
          </button>
          <button
            className="primary"
            disabled={busy || !chosen?.fits || chosenSplits.length === 0 || !choices?.evaluate || !placeReady(places, nodeId)}
          >
            검사 시작
          </button>
        </div>
      </form>
    </Modal>
  );
}
