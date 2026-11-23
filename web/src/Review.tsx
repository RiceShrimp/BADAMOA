import { type FormEvent, useEffect, useMemo, useRef, useState } from "react";
import { type BoardProject, useBoardFilter } from "./Boards";
import { Collage, SPLIT_COLOR, Shot, rememberProject } from "./DataBoard";
import { ItemViewer } from "./ItemViewer";
import { PlacePicker, placeReady, usePlaces } from "./ModelOps";
import type { ItemFilter } from "./ReleaseMaker";
import { NewSetDialog } from "./Labeling";
import { OrderReset, SortHandle, SortItem, Sortable, useOrder } from "./Order";
import { Pager, usePaging } from "./Pager";
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
import { LogTail } from "./Training";
import { METRIC_LABEL, metric } from "./trainTypes";
import { Badge, Icon, Modal, STATUS_LABEL, Stat, duration, timeAgo, usePolling } from "./ui";

/** One label check (rules on the labels) as the Hub counts it. */
interface Rule {
  id: string;
  group: "missing" | "wrong" | "boxes";
  title: string;
  hint: string;
  images: number;
  let_be: number; // images someone judged right ("문제 없음"): not counted in ``images``
  // a box check a threshold decides: the one counted at, its default, the count at each level
  limit?: number;
  default?: number;
  levels?: [number, number][];
}

/** How 검수 counts a model check (FiftyOne's way): how sure the model must be of what it finds. */
interface ModelRule {
  conf: number;
  defaults: { conf: number }; // FiftyOne's
}

type Mistake = "extra" | "missed" | "reclass";

interface Summary {
  release_id: string;
  computed_at: string;
  checks: Rule[];
  evaluations: (ReleaseCheck &
    Record<Mistake | `${Mistake}_let_be`, number> & {
      levels: Record<Mistake, [number, number][]>; // the count at each threshold level beside the chosen one
    })[];
  rule?: ModelRule;
  pending: { id: string; job_id: string; status: string; split: string; model: CheckModel | null; created_at: string }[];
}

/** One model check of this dataset as 검수 keeps it: like a training's record, there after it ends too. */
interface CheckRecord {
  id: string;
  job_id: string | null;
  status: string;
  error: string | null;
  wait_reason: string | null;
  node_name: string | null;
  duration_s: number | null;
  created_at: string;
  finished_at: string | null;
  log_count: number;
  by: string;
  model: CheckModel | null;
  release: { id: string; dataset: string; version: string } | null;
  splits: string[];
  code: { version?: string; trained_with?: string };
  images: number; // images it kept predictions of
  basis: boolean; // usable as the model basis here
  sent: { job_id: string; status: string; error: string | null; node_name: string | null; duration_s: number | null } | null;
}

/** A model (this project's, or another's with the same task and classes) and whether it can check this version. */
interface EvalChoice extends CheckModel {
  arch: string;
  metric: string;
  value: number | null;
  created_at: string;
  trained_on: boolean; // trained on this version: its train split was learned
  fits: boolean;
  why: string; // why it cannot, when it cannot
  relation: "same_version" | "same_dataset" | "same_project" | "other_project"; // what it learned from
  trained: { project: string; dataset: string; version: string };
  warning: string; // when it learned from other data
  classes: number; // how many classes it knows
}

const RELATION: Record<EvalChoice["relation"], string> = {
  same_version: "이 버전으로 학습",
  same_dataset: "다른 버전으로 학습",
  same_project: "다른 데이터셋",
  other_project: "다른 프로젝트",
};

interface EvalChoices {
  release_id: string;
  evaluate: { id: string; version: string } | null; // where it runs: the version, or a work in progress's base
  task: string;
  splits: Record<string, number>;
  models: EvalChoice[];
}

/** What the grid shows: a rule, or one side of an evaluated model's view. */
type Pick = { kind: "rule"; id: string } | { kind: "model"; mistake: Mistake };

const GROUPS: { id: Rule["group"]; title: string; hint: string }[] = [
  { id: "missing", title: "미라벨", hint: "라벨이 빠진 이미지 · 물체" },
  { id: "wrong", title: "오라벨", hint: "클래스나 박스가 틀렸을 라벨" },
  { id: "boxes", title: "박스 오류", hint: "박스 모양이 이상한 라벨" },
];
// A model's view, counted FiftyOne's way (the Hub's ``checks.judge``): only what the model is sure of
const MODEL: Record<Mistake, { group: Rule["group"]; title: string; hint: string }> = {
  extra: { group: "missing", title: "라벨에 없는 물체", hint: "라벨이 없는 곳에 모델이 확신하는 물체: 빠뜨린 라벨일 수 있음" },
  reclass: { group: "wrong", title: "클래스가 다를 수 있는 라벨", hint: "같은 자리에서 모델은 다른 클래스라고 확신함" },
  missed: { group: "wrong", title: "모델이 못 찾은 라벨", hint: "어떤 예측도 닿지 않은 라벨: 없는 물체를 그렸을 수 있음" },
};

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
  evaluation: {
    id: string;
    model: { id: string; name: string; version: number } | null;
    images: number;
    extra: number;
    missed: number;
    reclass?: number;
  } | null;
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
            <strong>{c.evaluation.extra.toLocaleString()}</strong>장 · 클래스 다름 <strong>{(c.evaluation.reclass ?? 0).toLocaleString()}</strong>장 ·
            못 찾은 라벨 <strong>{c.evaluation.missed.toLocaleString()}</strong>장
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
  // how sure the model must be: FiftyOne's until moved from a model card's gear (the counts, images, frames follow)
  const [conf, setConf] = useState<number | null>(null);
  const [confTuning, setConfTuning] = useState(false);
  // the box checks' thresholds moved here (the rest at their defaults): duplicate:0.9,tiny:5
  const [ruleLimits, setRuleLimits] = useState<Record<string, number>>({});
  const rulesParam = Object.entries(ruleLimits)
    .map(([check, value]) => `${check}:${value}`)
    .join(",");
  const limitQuery = (() => {
    const params = new URLSearchParams();
    if (conf != null) params.set("conf", String(conf));
    if (rulesParam) params.set("rules", rulesParam);
    const text = params.toString();
    return text ? `?${text}` : "";
  })();
  const [tuning, setTuning] = useState<string | null>(null); // the box check whose threshold dialog is open
  const { data: summary, error, refresh } = usePolling(
    () => api<Summary>(`/releases/${releaseId}/qa${limitQuery}`),
    watching ? 4000 : 0,
    [releaseId, limitQuery],
  );
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
  const { page, setPage, pageSize, setPageSize } = usePaging();
  const listTop = useRef<HTMLDivElement>(null); // the pager above the images: the one under them brings it back
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
  const [showBoxes, setShowBoxes] = useState(true); // the thumbnails' label boxes
  const narrowed = Object.values(filter).filter(Boolean).length;
  const setOne = (key: keyof ItemFilter, value: string) => setFilter((f) => ({ ...f, [key]: value || undefined }));
  const query = useMemo(() => {
    if (!pick) return "";
    const params = new URLSearchParams();
    if (pick.kind === "rule") {
      params.set("check", pick.id);
      if (ruleLimits[pick.id] != null) params.set("rules", rulesParam);
    }
    else if (op) {
      params.set("suspect", op);
      params.set("mistake", pick.mistake);
      if (conf != null) params.set("conf", String(conf));
    }
    if (showLetBe) params.set("let_be", "true");
    for (const [key, value] of Object.entries(filter)) {
      if (value) params.set(key, key === "split" && value === "none" ? "" : value);
    }
    return params.toString();
  }, [pick, op, showLetBe, filter, conf, ruleLimits]);
  useEffect(() => setPage(0), [query]);
  const { data: items, refresh: refreshItems } = usePolling(
    () => (query ? api<ItemPage>(`/releases/${releaseId}/items?${query}&offset=${page * pageSize}&limit=${pageSize}`) : Promise.resolve(null)),
    0,
    [releaseId, query, page, pageSize],
  );

  if (error && !summary) return <div className="error">{error}</div>;
  if (!summary || !release) return <CheckProgress releaseId={releaseId} />;
  const classes = release.label_space.classes;
  const colorOf = colorFor(classes);
  const task = release.task ?? "detect";
  const evaluation = summary.evaluations.find((e) => e.id === op);
  const rule = pick?.kind === "rule" ? summary.checks.find((c) => c.id === pick.id) : undefined;
  const title = rule ? rule.title : pick?.kind === "model" ? MODEL[pick.mistake].title : "";
  // what someone judged right for the check in view: listed apart
  const letBeCount = rule ? rule.let_be : pick?.kind === "model" && evaluation ? evaluation[`${pick.mistake}_let_be`] : 0;
  const total = items?.total ?? 0;
  const pages = Math.max(1, Math.ceil(total / pageSize));
  const card = (key: string, name: string, hint: string, count: number, active: boolean, onClick: () => void) => (
    <button key={key} className={`check-card ${active ? "active" : ""} ${count ? "" : "zero"}`} onClick={onClick} title={hint}>
      <span className="check-count">{count.toLocaleString()}</span>
      <span className="check-title">{name}</span>
      <span className="muted small">{hint}</span>
    </button>
  );
  // A model's check is not worked out on the spot like the rules: it needs an evaluation, started from here.
  const going = summary.pending[0];
  const modelCard = (m: Mistake) => {
    const count = evaluation ? evaluation[m] : null;
    const active = pick?.kind === "model" && pick.mistake === m;
    const rule = summary.rule;
    if (count != null && rule && m !== "missed") {
      const moved = rule.conf !== rule.defaults.conf;
      return (
        <div key={m} className="check-card-wrap">
          {modelButton(m, count, active)}
          <button
            className={`icon-btn card-gear ${moved ? "moved" : ""}`}
            onClick={() => setConfTuning(true)}
            title={`확신도 설정 (지금 ${rule.conf} 초과${moved ? ` · 기본 ${rule.defaults.conf}` : ""})`}
            aria-label={`${MODEL[m].title} 확신도 설정`}
          >
            <Icon name="settings" size={14} />
          </button>
        </div>
      );
    }
    return modelButton(m, count, active);
  };
  const modelButton = (m: Mistake, count: number | null, active: boolean) => {
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
    ...GROUPS.map((g) => summary.checks.filter((c) => c.group === g.id).length + Object.values(MODEL).filter((m) => m.group === g.id).length),
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
        <a className="link small review-to-dataset" href={`#/datasets/${release.id}`} title="이 버전을 데이터셋 탭에서 엽니다">
          데이터셋에서 보기
        </a>
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
        <CheckHistory releaseId={releaseId} nudge={summary} />
      </section>
      <div className="check-groups" style={{ gridTemplateRows: `auto repeat(${slots}, auto)` }}>
        {GROUPS.map((g) => (
          <section key={g.id} className="check-group" style={{ gridRow: `1 / span ${slots + 1}` }}>
            <header>
              <strong>{g.title}</strong> <span className="muted small">{g.hint}</span>
            </header>
            {summary.checks
              .filter((c) => c.group === g.id)
              .map((c) =>
                c.levels ? (
                  <div key={c.id} className="check-card-wrap">
                    {card(c.id, c.title, c.hint, c.images, pick?.kind === "rule" && pick.id === c.id, () => setPick({ kind: "rule", id: c.id }))}
                    <button
                      className={`icon-btn card-gear ${c.limit !== c.default ? "moved" : ""}`}
                      onClick={() => setTuning(c.id)}
                      title={`임계값 설정 (지금 ${c.limit}${c.limit !== c.default ? ` · 기본 ${c.default}` : ""})`}
                      aria-label={`${c.title} 임계값 설정`}
                    >
                      <Icon name="settings" size={14} />
                    </button>
                  </div>
                ) : (
                  card(c.id, c.title, c.hint, c.images, pick?.kind === "rule" && pick.id === c.id, () => setPick({ kind: "rule", id: c.id }))
                ),
              )}
            {(Object.keys(MODEL) as Mistake[]).filter((m) => MODEL[m].group === g.id).map(modelCard)}
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
          <div className="review-grid-bar">
            <ReviewFilters release={release} filter={filter} onChange={setOne} onClear={() => setFilter({})} />
            {task !== "classify" && (
              <label className="check inline-check" title="썸네일에 라벨 박스를 그릴지">
                <input type="checkbox" checked={showBoxes} onChange={(e) => setShowBoxes(e.target.checked)} />
                라벨
              </label>
            )}
          </div>
          {total > 0 && (
            <Pager page={page} pages={pages} onGo={setPage} pageSize={pageSize} onPageSize={setPageSize} innerRef={listTop} />
          )}
          {items && items.items.length === 0 && <div className="card empty">찾은 이미지가 없습니다.</div>}
          <div className="thumb-grid medium">
            {items?.items.map((item, i) => (
              <button key={item.id} className="thumb" onClick={() => setViewing(page * pageSize + i)}>
                <span className="thumb-image">
                  {item.thumb_url ? (
                    <Shot
                      shot={{ url: item.thumb_url, width: item.width, height: item.height, boxes: item.boxes }}
                      colorOf={colorOf}
                      task={task}
                      boxes={showBoxes}
                    />
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
                    <span className="tag">
                      {pick.mistake === "extra"
                        ? `라벨에 없음 ${item.check.extra}`
                        : pick.mistake === "reclass"
                          ? `클래스 다름 ${item.check.reclass ?? 0}`
                          : `못 찾음 ${item.check.missed}`}
                    </span>
                  )}
                </span>
              </button>
            ))}
          </div>
          {pages > 1 && (items?.items.length ?? 0) > 0 && (
            <Pager page={page} pages={pages} onGo={setPage} pageSize={pageSize} onPageSize={setPageSize} bottom top={listTop} />
          )}
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
      {tuning &&
        (() => {
          const rule = summary.checks.find((c) => c.id === tuning);
          return rule ? (
            <RuleThreshold
              rule={rule}
              onChange={(value) => {
                setRuleLimits((was) => {
                  const next = { ...was };
                  if (value == null || value === rule.default) delete next[rule.id];
                  else next[rule.id] = value;
                  return next;
                });
              }}
              onClose={() => setTuning(null)}
            />
          ) : null;
        })()}
      {confTuning && summary.rule && evaluation && (
        <ConfThreshold
          rule={summary.rule}
          evaluation={evaluation}
          onChange={(value) => setConf(value == null || value === summary.rule?.defaults.conf ? null : value)}
          onClose={() => setConfTuning(false)}
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
      <ClassFilter
        classes={release.label_space.classes}
        picked={(filter.classes ?? "").split(",").filter(Boolean).map(Number)}
        onChange={(ids) => onChange("classes", ids.join(","))}
      />
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

/**
 * 검수's class filter: several classes at once, and what it narrows is the findings of those classes (a flagged
 * label of one of them, the model's missed or extra object of one of them), not the images that hold them.
 */
function ClassFilter({
  classes,
  picked,
  onChange,
}: {
  classes: { id: number; name: string }[];
  picked: number[];
  onChange: (ids: number[]) => void;
}) {
  const [open, setOpen] = useState(false);
  const [find, setFind] = useState("");
  const box = useRef<HTMLDivElement>(null);
  const colorOf = colorFor(classes);
  useEffect(() => {
    if (!open) return;
    const away = (e: MouseEvent) => box.current && !box.current.contains(e.target as Node) && setOpen(false);
    const esc = (e: KeyboardEvent) => e.key === "Escape" && setOpen(false);
    document.addEventListener("mousedown", away);
    document.addEventListener("keydown", esc);
    return () => {
      document.removeEventListener("mousedown", away);
      document.removeEventListener("keydown", esc);
    };
  }, [open]);
  const names = classes.filter((c) => picked.includes(c.id)).map((c) => c.name);
  const label = names.length === 0 ? "모든 클래스" : names.length <= 2 ? names.join(", ") : `${names.slice(0, 2).join(", ")} 외 ${names.length - 2}`;
  const shown = classes.filter((c) => !find.trim() || c.name.toLowerCase().includes(find.trim().toLowerCase()));
  const toggle = (id: number) => onChange(picked.includes(id) ? picked.filter((x) => x !== id) : [...picked, id]);
  return (
    <div className="class-filter" ref={box}>
      <button type="button" className={`class-filter-button ${picked.length ? "on" : ""}`} onClick={() => setOpen(!open)} aria-expanded={open}>
        클래스: {label} ▾
      </button>
      {open && (
        <div className="class-filter-panel" role="listbox" aria-multiselectable="true">
          <div className="muted small">고른 클래스의 라벨이 걸린 것만 보입니다(그 클래스가 들어 있는 이미지 전부가 아니라).</div>
          {classes.length > 12 && <input value={find} onChange={(e) => setFind(e.target.value)} placeholder="클래스 찾기…" autoFocus />}
          <div className="class-filter-list">
            {shown.map((c) => (
              <label key={c.id} className="check">
                <input type="checkbox" checked={picked.includes(c.id)} onChange={() => toggle(c.id)} />
                <i className="class-dot" style={{ background: colorOf(c.id) }} />
                {c.name}
              </label>
            ))}
          </div>
          {picked.length > 0 && (
            <button type="button" className="link small" onClick={() => onChange([])}>
              모두 해제
            </button>
          )}
        </div>
      )}
    </div>
  );
}

/** How a box check's threshold reads: IoU as is, pixels, a share of the image as a percentage. */
function thresholdText(check: string, value: number): string {
  if (check === "tiny") return `${value}px`;
  if (check === "huge") return `${+(value * 100).toFixed(1)}%`;
  return value.toFixed(2);
}

const RULE_RANGES: Record<string, { min: number; max: number; step: number; what: string }> = {
  duplicate: { min: 0.5, max: 0.99, step: 0.01, what: "같은 클래스 두 박스의 IoU가 이 값 이상" },
  class_overlap: { min: 0.5, max: 0.99, step: 0.01, what: "다른 클래스 두 박스의 IoU가 이 값 이상" },
  tiny: { min: 1, max: 16, step: 1, what: "박스의 가로나 세로가 이 픽셀 미만" },
  huge: { min: 0.8, max: 0.999, step: 0.001, what: "박스가 이미지 넓이의 이 비율 이상" },
};

/**
 * A box check's threshold, set from its card's gear: a slider, the count at each level (a level is clicked to take
 * it), and back to the default. The card, the images and their frames count at it on the spot; it lasts while this
 * version is open here (nothing is saved for anyone else).
 */
function RuleThreshold({ rule, onChange, onClose }: { rule: Rule; onChange: (value: number | null) => void; onClose: () => void }) {
  const range = RULE_RANGES[rule.id];
  const [value, setValue] = useState(rule.limit ?? rule.default ?? range.min);
  useEffect(() => {
    if (value === rule.limit) return;
    const timer = window.setTimeout(() => onChange(value), 250);
    return () => window.clearTimeout(timer);
  }, [value]);
  if (!range || rule.levels == null) return null;
  return (
    <Modal title={`${rule.title} · 임계값`} onClose={onClose}>
      <p className="muted small">{range.what}이면 걸립니다. 바꾸면 카드 · 이미지 목록 · 노란 테두리가 바로 이 값으로 다시 셉니다(이 화면에서만).</p>
      <label className="rule-threshold">
        <input
          type="range"
          min={range.min}
          max={range.max}
          step={range.step}
          value={value}
          onChange={(e) => setValue(Number(e.target.value))}
          aria-label={`${rule.title} 임계값`}
        />
        <strong className="num">{thresholdText(rule.id, value)}</strong>
        <span className="muted small">걸린 이미지 {rule.images.toLocaleString()}장</span>
      </label>
      <table className="threshold-levels small">
        <tbody>
          <tr>
            <th>임계값별 이미지 수</th>
            {rule.levels.map(([level, count]) => (
              <td key={level}>
                <button className={`level ${level === rule.limit ? "on" : ""}`} onClick={() => setValue(level)}>
                  <span className="muted">{thresholdText(rule.id, level)}</span>
                  <strong className="num">{count.toLocaleString()}</strong>
                </button>
              </td>
            ))}
          </tr>
        </tbody>
      </table>
      <div className="modal-actions">
        {rule.default != null && rule.limit !== rule.default && (
          <button
            type="button"
            className="push-left"
            onClick={() => {
              setValue(rule.default as number);
              onChange(null);
            }}
          >
            기본값으로 ({thresholdText(rule.id, rule.default)})
          </button>
        )}
        <button type="button" className="primary" onClick={onClose}>
          닫기
        </button>
      </div>
    </Modal>
  );
}

const CONF_ROWS: { kind: Mistake; title: string }[] = [
  { kind: "extra", title: "라벨에 없는 물체" },
  { kind: "reclass", title: "클래스가 다를 수 있는 라벨" },
];

/**
 * How sure the model must be (over ``conf``) of an object no label has or of another class, set from those two
 * cards' gear: one value for both. Moving it counts again on the spot — the cards, the images and their frames — and
 * the counts at every level are laid out, so what it changes is seen before choosing it (a level is clicked to take
 * it). It lasts while this version is open here.
 */
function ConfThreshold({
  rule,
  evaluation,
  onChange,
  onClose,
}: {
  rule: ModelRule;
  evaluation: Summary["evaluations"][number];
  onChange: (conf: number | null) => void;
  onClose: () => void;
}) {
  const [value, setValue] = useState(rule.conf);
  // the slider moves freely; the count follows once it rests
  useEffect(() => {
    if (value === rule.conf) return;
    const timer = window.setTimeout(() => onChange(value), 300);
    return () => window.clearTimeout(timer);
  }, [value]);
  return (
    <Modal title="모델 확신도 · 임계값" onClose={onClose}>
      <p className="muted small">
        모델이 이 값보다 확신한 것만 셉니다. <b>라벨에 없는 물체</b>와 <b>클래스가 다를 수 있는 라벨</b>에 함께 쓰입니다. 바꾸면 카드 · 이미지 목록 ·
        노란 테두리가 바로 이 값으로 다시 셉니다(이 화면에서만).
      </p>
      <label className="rule-threshold">
        <input
          type="range"
          min={0.25}
          max={0.99}
          step={0.01}
          value={value}
          onChange={(e) => setValue(Number(e.target.value))}
          aria-label="모델 확신도 임계값"
        />
        <strong className="num">{value.toFixed(2)} 초과</strong>
      </label>
      <table className="threshold-levels small">
        <tbody>
          {CONF_ROWS.map((row) => (
            <tr key={row.kind}>
              <th>{row.title}</th>
              {evaluation.levels[row.kind].map(([level, count]) => (
                <td key={level}>
                  <button className={`level ${level === rule.conf ? "on" : ""}`} onClick={() => setValue(level)} title={`확신도 ${level} 초과로 세기`}>
                    <span className="muted">{level.toFixed(2)}</span>
                    <strong className="num">{count.toLocaleString()}</strong>
                  </button>
                </td>
              ))}
            </tr>
          ))}
        </tbody>
      </table>
      <div className="modal-actions">
        {rule.conf !== rule.defaults.conf && (
          <button
            type="button"
            className="push-left"
            onClick={() => {
              setValue(rule.defaults.conf);
              onChange(null);
            }}
          >
            기본값으로 ({rule.defaults.conf})
          </button>
        )}
        <button type="button" className="primary" onClick={onClose}>
          닫기
        </button>
      </div>
    </Modal>
  );
}

interface QaProgress {
  running: boolean;
  step?: "prepare" | "read" | "write";
  done?: number;
  total?: number;
  seconds?: number;
}

/**
 * While a version's label checks run for the first time (a big one takes a while): how far they got, read every
 * second — looking at the labels, reading and judging N / M images with the time left, keeping the results. The
 * checks go on on the Hub whatever this screen does, and are kept.
 */
function CheckProgress({ releaseId }: { releaseId: string }) {
  const { data } = usePolling(() => api<QaProgress>(`/releases/${releaseId}/qa/progress`), 1000, [releaseId]);
  const done = data?.done ?? 0;
  const total = data?.total ?? 0;
  const reading = data?.running && data.step === "read" && total > 0;
  const share = !data?.running ? 0 : data.step === "write" ? 1 : reading ? done / total : 0;
  let words = "검사 준비하는 중…";
  if (data?.running && data.step === "prepare") words = "라벨 살펴보는 중…";
  if (reading) words = `라벨 검사하는 중 · ${done.toLocaleString()} / ${total.toLocaleString()}장 · ${Math.floor(share * 100)}%`;
  if (data?.running && data.step === "write") words = "검사 결과 저장하는 중…";
  if (data && !data.running && total === 0) words = "검사 결과 모으는 중…";
  let left = "";
  if (reading && done > 0 && (data?.seconds ?? 0) >= 2) {
    const seconds = ((total - done) * (data?.seconds ?? 0)) / done;
    left = seconds < 60 ? "1분 안에 끝남" : `약 ${Math.ceil(seconds / 60)}분 남음`;
  }
  return (
    <section className="card check-progress">
      <div className="import-progress">
        <div className="bar">
          <i style={{ width: `${Math.round(share * 100)}%` }} />
        </div>
        <div className="import-progress-line">
          <span className="small">{words}</span>
          {left && <span className="muted small">{left}</span>}
        </div>
      </div>
      <p className="muted small">처음 여는 버전만 이렇게 한 번 검사합니다. 다른 탭으로 가도 검사는 Hub에서 계속되고 결과는 저장됩니다.</p>
    </section>
  );
}

const ENDED_BADLY = ["failed", "canceled", "cancelled"];

/**
 * The model checks of this dataset (its versions with the same classes), newest first: each with how it went, on
 * which server, how long, who started it, the images it had sent there first, and its log — kept after it ends,
 * as a training's are. The newest one failing stays said until another starts.
 */
function CheckHistory({ releaseId, nudge }: { releaseId: string; nudge: unknown }) {
  const [busy, setBusy] = useState(false);
  const { data: rows, refresh } = usePolling(
    () => api<CheckRecord[]>(`/releases/${releaseId}/qa/history`),
    busy ? 4000 : 20000,
    [releaseId],
  );
  useEffect(() => setBusy((rows ?? []).some((r) => r.status === "queued" || r.status === "running")), [rows]);
  useEffect(() => {
    void refresh(); // the summary was read again: a check may have started or ended
  }, [nudge]);
  const [open, setOpen] = useState(false);
  const [log, setLog] = useState<string | null>(null); // the job whose log is shown
  if (!rows || rows.length === 0) return null;
  const last = rows[0];
  const said = (r: CheckRecord) => STATUS_LABEL[r.status] ?? r.status;
  const showLog = (jobId: string | null) => {
    setOpen(true);
    setLog(jobId);
  };
  return (
    <div className="check-history">
      {ENDED_BADLY.includes(last.status) && !open && (
        <div className="check-history-failed small">
          <Badge kind={last.status}>{said(last)}</Badge>
          <span>
            마지막 검사({last.model ? `${last.model.name} v${last.model.version}` : "모델"} · {timeAgo(last.created_at)}):{" "}
            {last.error || last.sent?.error || "끝나지 못했습니다"}
          </span>
          {last.job_id && (
            <button className="link small" onClick={() => showLog(last.job_id)}>
              로그 보기
            </button>
          )}
        </div>
      )}
      <button className="link small check-history-toggle" onClick={() => setOpen(!open)} aria-expanded={open}>
        {open ? "▾" : "▸"} 검사 작업 기록 {rows.length}개
      </button>
      {open && (
        <ul className="check-history-list">
          {rows.map((r) => (
            <li key={r.id}>
              <div className="check-history-row">
                <Badge kind={r.status}>{said(r)}</Badge>
                <ModelName model={r.model} />
                <span className="muted small check-history-what">
                  {[
                    r.release?.version,
                    r.splits.join("·"),
                    r.node_name,
                    r.duration_s != null && r.status !== "queued" ? duration(r.duration_s) : "",
                    r.by,
                    timeAgo(r.created_at),
                  ]
                    .filter(Boolean)
                    .join(" · ")}
                  {r.code.trained_with ? ` · ${r.code.version} 코드로` : ""}
                  {r.basis ? ` · 기준 이미지 ${r.images.toLocaleString()}장` : ""}
                </span>
                <span className="spacer" />
                {r.job_id && (
                  <button className="link small" onClick={() => setLog(log === r.job_id ? null : r.job_id)} aria-expanded={log === r.job_id}>
                    {log === r.job_id ? "로그 닫기" : "로그"}
                  </button>
                )}
                <a className="small" href={`#/models/eval/${r.id}`} title="평가 화면: 클래스별 · 다시 실행 · 같은 데이터로 비교">
                  평가 화면
                </a>
              </div>
              {r.sent && (
                <div className={`small check-history-sub ${ENDED_BADLY.includes(r.sent.status) ? "bad-text" : "muted"}`}>
                  이미지를 {r.sent.node_name || "그 서버"}에 먼저 보냄: {STATUS_LABEL[r.sent.status] ?? r.sent.status}
                  {r.sent.duration_s != null ? ` · ${duration(r.sent.duration_s)}` : ""}
                  {r.sent.error ? ` · ${r.sent.error}` : ""}
                </div>
              )}
              {r.status === "queued" && r.wait_reason && <div className="muted small check-history-sub">{r.wait_reason}</div>}
              {r.error && <div className="bad-text small check-history-sub">{r.error}</div>}
              {log === r.job_id && r.job_id && (
                <div className="check-history-log">
                  <LogTail jobId={r.job_id} live={r.status === "queued" || r.status === "running"} open />
                </div>
              )}
            </li>
          ))}
        </ul>
      )}
    </div>
  );
}

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
          있음)을 셉니다. {taskLabel(choices?.task)} 모델이고 이 버전에 모델의 클래스가 모두 있으면(이름으로 맞춤), 이 데이터셋의 다른
          버전이나 다른 데이터셋 · 다른 프로젝트에서 학습한 모델도 고를 수 있습니다.
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
                <span className={`tag relation-${m.relation}`}>{RELATION[m.relation]}</span>
                <span className="muted small">
                  {m.arch ? `${m.arch} · ` : ""}
                  {METRIC_LABEL[m.metric] ?? m.metric} {metric(m.value)} · {timeAgo(m.created_at)}
                  {m.relation !== "same_version" && m.trained.dataset
                    ? ` · 학습: ${m.relation === "other_project" ? `${m.trained.project} · ` : ""}${m.trained.dataset} ${m.trained.version}`
                    : ""}
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
        {chosen?.fits && chosen.warning && (
          <div className={`notice small ${chosen.relation === "same_dataset" ? "" : "warn"}`} role="alert">
            {chosen.warning}
          </div>
        )}
        {chosen?.fits && chosen.classes > 0 && chosen.classes < release.label_space.classes.length && (
          <div className="notice small">
            이 버전의 클래스 {release.label_space.classes.length}개 가운데 모델이 아는 {chosen.classes}개만 검사합니다. 나머지 클래스의 라벨은 이
            검사에 나오지 않습니다.
          </div>
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
