export interface Project {
  id: string;
  name: string;
  task: string;
  folder: string;
  settings: { template?: string };
  releases: number;
  items: number;
}

export interface TreeRelease {
  id: string;
  version: string;
  status: "building" | "ready";
  items: number;
  label_space: string;
  created_at: string;
  deleted_at: string | null;
}

export interface TreeDataset {
  id: string;
  name: string;
  releases: TreeRelease[];
}

export interface LabelClass {
  id: number;
  name: string;
  color?: string | null; // set on the classes tab; otherwise an automatic colour
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

export interface ReleaseInfo {
  id: string;
  version: string;
  status: "building" | "ready";
  note: string;
  created_at: string;
  project_id: string;
  dataset: { id: string; name: string };
  label_space: { id: string; name: string; classes: LabelClass[] };
  base: { id: string; version: string } | null;
  stats: ReleaseStats;
  siblings: { id: string; version: string }[];
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
  revisions: Revision[];
  releases: { id: string; dataset: string; version: string; split: string; rev: number | null; current: boolean }[];
}

export interface TrashInfo {
  projects: { id: string; name: string; task: string; folder: string; deleted_at: string; releases: number; items: number }[];
  releases: {
    id: string;
    dataset: string;
    version: string;
    project: { id: string; name: string };
    deleted_at: string;
    items: number;
    in_use: string[]; // what still points at it, e.g. "학습 1개": it cannot be removed for good
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
export const STATE_LABEL: Record<string, string> = { labelled: "라벨 있음", background: "배경", unlabelled: "미라벨" };
/** What each label state means, where they are picked (background and unlabelled are easy to mix up). */
export const STATE_HINT: Record<string, string> = {
  labelled: "",
  background: "물체 없음 확인",
  unlabelled: "아직 라벨 안 함",
};
export const STATE_HELP: Record<string, string> = {
  labelled: "박스나 클래스 라벨이 있는 이미지",
  background: "누가 보고 찾을 물체가 없다고 확인한 이미지(빈 라벨 파일). 학습에 '없음'의 예로 들어갑니다.",
  unlabelled: "라벨 파일이 없는, 아직 아무도 라벨하지 않은 이미지. 물체가 있을 수 있어 학습에서는 기본으로 뺍니다.",
};

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
