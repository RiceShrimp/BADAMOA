export interface Project {
  id: string;
  name: string;
  /** The tasks of its datasets, oldest dataset first (a project has none of its own). */
  tasks: string[];
  task: string; // the first of them
  folder: string;
  settings: { template?: string };
  releases: number;
  items: number;
}

export interface TreeRelease {
  id: string;
  version: string;
  status: "building" | "ready" | "draft";
  items: number;
  label_space: string;
  created_at: string;
  deleted_at: string | null;
  edited?: number; // the work in progress: how many images it changed
}

export interface TreeDataset {
  id: string;
  name: string;
  task: string;
  releases: TreeRelease[];
}

export interface LabelClass {
  id: number;
  name: string;
  color?: string | null; // set on the classes tab; otherwise an automatic colour
  keypoints?: string[]; // keypoint (pose) projects: the points' names …
  skeleton?: number[][]; // … and the limbs between them (0-based pairs)
}

/** The keypoints after a pose label's box: [x, y, visibility] each (0 not labelled, 1 hidden, 2 seen). */
export function keypointsOf(box: number[]): [number, number, number][] {
  const out: [number, number, number][] = [];
  for (let i = 5; i + 2 < box.length; i += 3) out.push([box[i], box[i + 1], box[i + 2]]);
  return out;
}

export interface ReleaseStats {
  items?: number;
  bytes?: number;
  boxes?: number;
  splits?: Record<string, { items: number; bytes: number }>;
  states?: Record<string, number>;
  classes?: Record<string, { boxes: number; items: number }>;
  sites?: string[];
  cameras?: string[];
}

/** A dataset's work in progress ("작업 중"): fixed labels not yet made a version. */
export interface DraftInfo {
  id: string;
  base: { id: string; version: string } | null;
  edited: number;
  editors: string[];
  created_at: string;
  /** A newer version came in after it was started: it can be moved onto that one. */
  behind: { id: string; version: string } | null;
}

export interface ReleaseInfo {
  id: string;
  version: string;
  status: "building" | "ready" | "draft";
  note: string;
  created_at: string;
  project_id: string;
  dataset: { id: string; name: string };
  task: string; // the dataset's
  label_space: { id: string; name: string; classes: LabelClass[] };
  base: { id: string; version: string } | null;
  stats: ReleaseStats;
  siblings: { id: string; version: string }[];
  draft: DraftInfo | null; // the dataset's work in progress, whichever version is open
}

export interface GridItem {
  id: number;
  file_name: string;
  width: number;
  height: number;
  split: string;
  state: "labelled" | "background" | "unlabelled";
  box_count: number;
  thumb_url: string | null;
  boxes: number[][]; // [class, cx, cy, w, h] as fractions, for drawing over the thumbnail
  fixed?: boolean; // changed in the work in progress
  check?: { found: number; missed: number; extra: number }; // sorted by an evaluation: its mistakes on this image
}

/** An evaluation that can sort a version's images by likely wrong labels. */
export interface ReleaseCheck {
  id: string;
  model: { id: string; name: string; version: number } | null;
  release: { id: string; version: string } | null;
  split: string;
  created_at: string;
  images: number; // images of this version it measured
  wrong: number; // of those, the ones with a mistake
}

/** What an evaluated model predicted for one image (labels in the version's label form). */
export interface ItemPredictions {
  op_id: string;
  model: { id: string; name: string; version: number } | null;
  split: string;
  evaluated: boolean;
  labels: number[][];
  scores: number[];
  found: number;
  missed: number;
  extra: number;
}

export interface ItemPage {
  total: number;
  offset: number;
  limit: number;
  items: GridItem[];
}

export interface Revision {
  id: number;
  rev: number;
  state: string;
  box_count: number;
  source: string;
  created_at: string;
  label_space: { id: string; name: string; classes: number };
  labelled_classes: string[];
  boxes: number[][];
  classes: LabelClass[];
}

export interface ItemDetail {
  id: number;
  file_name: string;
  width: number;
  height: number;
  bytes: number;
  sha256: string;
  site: string;
  camera: string;
  rel_path: string;
  split: string;
  revision_id: number;
  release_label_space_id: string;
  thumb_url: string | null;
  image_url: string | null;
  image_local?: boolean; // false: the original comes from the node that has it
  edit: ItemEdit;
  revisions: Revision[];
  releases: { id: string; dataset: string; version: string; split: string; rev: number | null; current: boolean; draft: boolean }[];
}

/** Whether an image may be fixed where it is open, and the labels a fix starts from. */
export interface ItemEdit {
  allowed: boolean;
  reason: string;
  latest: { id: string; version: string } | null;
  draft_id: string | null;
  revision_id: number | null; // the work in progress's labels, else this version's
  fixed: boolean; // fixed in the work in progress (looking at its base version)
}

/** Everything deleted, to bring back or remove for good. */
export interface TrashInfo {
  projects: { id: string; name: string; folder: string; deleted_at: string; releases: number; items: number }[];
  releases: {
    id: string;
    dataset: string;
    version: string;
    project: { id: string; name: string };
    deleted_at: string;
    items: number;
    in_use: string[]; // what still points at it, e.g. "학습 1개": it cannot be removed for good
  }[];
  runs: {
    id: string;
    name: string;
    project: { id: string; name: string };
    release: string; // "coco v2"
    template: string;
    epoch: number;
    created_by: string | null;
    deleted_at: string;
    in_use: string[]; // "모델 1개": the models it trained keep it
  }[];
  models: {
    id: string;
    name: string;
    version: number;
    task: string;
    bytes: number;
    project: { id: string; name: string };
    run: string;
    deleted_at: string;
    restorable: boolean; // false: its weights were removed before the trash kept files
  }[];
  templates: {
    id: string;
    name: string;
    title: string;
    task: string;
    versions: string[];
    deleted_at: string;
    runs: number; // runs made with it: they cannot be started again once it is removed for good
  }[];
}

export interface CompareResult {
  release: { id: string; version: string; items: number; boxes: number };
  other: { id: string; version: string; items: number; boxes: number };
  added: number;
  removed: number;
  relabelled: number;
  moved_split: number;
  label_space_changed: boolean;
  classes: { name: string; boxes: number; other: number; delta: number }[];
}

export const SPLIT_LABEL: Record<string, string> = { train: "train", val: "val", test: "test", "": "미분할" };
export const STATE_LABEL: Record<string, string> = { labelled: "라벨 있음", background: "배경", unlabelled: "아직 안 함" };
/** What each label state means, where they are picked. */
export const STATE_HINT: Record<string, string> = { labelled: "", background: "찾을 것 없음", unlabelled: "라벨할 이미지" };
export const STATE_HELP: Record<string, string> = {
  labelled: "박스나 클래스 라벨이 있는 이미지",
  background: "다 봤는데 찾을 물체가 없는 이미지. 찾을 물체가 없는 예로 학습에 들어갑니다.",
  unlabelled: "아직 아무도 라벨하지 않은 이미지. 학습에서는 빠집니다.",
};

/** Something went into or out of the trash: the sidebar's count looks again. */
export const TRASH_EVENT = "badamoa:trash";
export const announceTrash = () => window.dispatchEvent(new Event(TRASH_EVENT));

/** Do ``work`` for each item, one after another; what failed comes back as "name: reason" lines. */
export async function eachOf<T>(items: T[], work: (item: T) => Promise<unknown>, name: (item: T) => string): Promise<string[]> {
  const failed: string[] = [];
  for (const item of items) {
    try {
      await work(item);
    } catch (e) {
      failed.push(`${name(item)}: ${e instanceof Error ? e.message : String(e)}`);
    }
  }
  return failed;
}

/** Tasks by their short English names; a task a team named is shown as typed. */
export const TASK_LABEL: Record<string, string> = { detect: "Det", segment: "Seg", pose: "Kpt", classify: "Cls" };
export const TASK_HINT: Record<string, string> = {
  detect: "박스",
  segment: "윤곽 · 마스크",
  pose: "키포인트 · 관절",
  classify: "이미지 분류",
};
export const taskLabel = (task: string | undefined | null) => (task ? (TASK_LABEL[task] ?? task) : "Det");

export function classColor(id: number): string {
  return `hsl(${(id * 137.5) % 360} 72% 52%)`;
}

export function bytesLabel(bytes?: number): string {
  if (!bytes) return "0 MB";
  const gbValue = bytes / 1024 ** 3;
  if (gbValue >= 1) return `${gbValue.toFixed(gbValue >= 100 ? 0 : 1)} GB`;
  if (bytes < 1024 ** 2) return `${Math.max(1, Math.round(bytes / 1024))} KB`;
  return `${(bytes / 1024 ** 2).toFixed(bytes < 10 * 1024 ** 2 ? 1 : 0)} MB`;
}
