import type { RunMedia } from "./Media";

export type RunStatus = "queued" | "running" | "succeeded" | "failed" | "canceled";

export interface TemplateParam {
  name: string;
  type: "string" | "integer" | "number" | "boolean" | "object";
  default: unknown;
  label: string;
  help: string;
  choices: unknown[];
  custom: boolean;
  minimum: number | null;
  maximum: number | null;
  advanced: boolean;
  group: string;
}

export interface Template {
  id: string;
  name: string;
  title: string;
  version: string;
  versions: number;
  task: string;
  /** What the model kind is: written on the Hub (설정 → 모델 종류), else what its bundle says. */
  description: string;
  declared_description?: string; // what the bundle says
  memo?: string; // this version's memo, written on the Hub
  license: string;
  builtin: boolean;
  data_format: string;
  entrypoints: string[];
  params: TemplateParam[];
  environment: {
    default: string;
    requirements: string[];
    installs: boolean; // an automatically installed environment, which can be prepared ahead
    ready: string[]; // servers where it is installed
    preparing: string[];
    nodes?: number; // servers that can have it (they train; with Docker for an image)
    kind?: string; // auto (a virtual environment) or docker
    image?: string;
    declared_image?: string; // an image declared next to the requirements: the environment "docker" of a run
    // every server switched on: why it cannot have the environment ("" when it can) and whether it is there
    servers?: { id: string; name: string; status: string; trains: boolean; why: string; state: "" | "ready" | "preparing" }[];
  };
  bytes: number;
  models: number;
  created_at: string;
  check: TemplateCheck;
  offered_version: string; // admins: the version everyone else gets when this one is not offered ("" otherwise)
  code_url?: string; // admins: the version's code as uploaded
  reads_weights_from?: string; // earlier versions whose models this version works with, e.g. "1.*"
  draft: boolean; // not offered to everyone (yet): a member sees it only when it is their own
  uploaded_by: string;
  // the uploader's own ``badamoa template check --run`` of this bundle
  local_check: { ok?: boolean; steps?: TemplateCheck["steps"]; sdk?: string; python?: string; os?: string; gpus?: string[]; device?: string; took_s?: number; at?: string; by?: string; packages?: number };
}

/** The trial run before a template version is offered to everyone. */
/** One version of a model kind (설정 → 모델 종류 → 버전). */
export interface TemplateVersion {
  id: string;
  version: string;
  title: string;
  description: string; // what that bundle says
  memo: string;
  created_at: string;
  uploaded_by: string;
  builtin: boolean;
  offered: boolean;
  check: TemplateCheck;
}

export interface TemplateCheck {
  state: "pending" | "checking" | "passed" | "ok" | "failed" | "allowed" | "legacy";
  offered: boolean;
  reason?: string;
  failed?: string; // the step that failed: data · env · train · predict · eval · export
  steps?: { step: string; ok: boolean; took_s: number; detail: string; skipped?: boolean }[];
  notes?: string[];
  node?: string;
  device?: string; // GPU or CPU: what the trial ran on
  job_id?: string;
  checked_at?: string;
  started_at?: string;
  took_s?: number;
  packages?: number;
  pinned?: number; // packages other servers install at the versions the trial installed
  sources?: Record<string, { kind: string; url: string; commit?: string }>; // packages not from an index
  contract?: number;
  legacy?: boolean;
  via?: string; // "run": a real training with it succeeded
  by?: string; // who started the trial
  approved_by?: string; // who offered it to everyone, and when
  approved_at?: string;
  why?: string; // allowed without a pass: the reason
}

export interface EnvironmentNode {
  node_id: string;
  node_name: string;
  path?: string;
  ok: boolean | null;
  detail: { python?: string; packages?: Record<string, string>; cuda?: boolean; gpus?: string[]; error?: string };
  checked_at: string | null;
}

export interface Environment {
  id: string;
  name: string;
  kind: string; // python: a server's Python · docker: a Docker image
  image?: string;
  description: string;
  nodes: EnvironmentNode[];
}

/** A run's settings: the template's own params plus what BADAMOA adds. */
export interface RunConfig {
  [param: string]: unknown;
  model?: string;
  epochs?: number;
  imgsz?: number;
  batch?: number;
  classes?: number[];
  unlabelled?: string;
  val_fraction?: number;
  gpus?: number;
  model_name?: string;
  nodes?: string[]; // servers it may train on; none: where the images are
}

/** Images on their way to the node a run will train on (from the node that holds them, not through the Hub). */
export interface RunSync {
  job_id: string;
  status: string;
  node_name: string;
  items: number | null;
  bytes: number | null;
  done_items: number | null;
  done_bytes: number | null;
  cached_items: number | null;
  rate: number | null;
  eta_s: number | null;
}

/** A server a run on some version could train on. */
export interface NodeChoice {
  id: string;
  name: string;
  status: "online" | "offline" | "disabled";
  enabled: boolean;
  gpus: number;
  gpus_free: number;
  /** Each GPU: free · busy (someone uses it) · job (a BADAMOA job) · reserved (kept for people) · offline */
  gpu_list?: NodeGpu[];
  images: "" | "registered" | "received"; // "": they would be copied there first
  cache_free_gb: number | null;
  can_receive: boolean;
}

export interface NodeGpu {
  index: number;
  name: string;
  memory_total_mb: number | null;
  state: "free" | "busy" | "job" | "reserved" | "offline";
  users: string[];
  reserved: boolean;
}

export interface NodeChoices {
  items: number;
  bytes: number;
  source: string;
  nodes: NodeChoice[];
}

export interface Run {
  id: string;
  name: string;
  project_id: string;
  task: string;
  template: { id: string; name: string; title: string; version: string; variant: string };
  environment: { kind: string; name: string; id?: string };
  status: RunStatus;
  stage: "" | "data" | "env" | "train" | "upload";
  sync?: RunSync | null;
  job_id: string | null;
  node_name: string | null;
  wait_reason: string | null;
  error: string | null;
  duration_s: number | null;
  gpu_indices: number[];
  created_via: string;
  created_at: string;
  epochs: number;
  epoch: number;
  config: RunConfig;
  release: { id: string; dataset: string; version: string } | null;
  metrics: Record<string, number>;
  model: { id: string; name: string; version: number; alias: string; aliases: string[] } | null;
  base_model: { id: string; name: string; version: number } | null;
  resumed_from: { id: string; name: string } | null;
  sweep: { id: string; name: string; index: number } | null;
  can_resume: boolean;
  source_run_id: string | null;
  note: string;
  tags: string[];
}

export interface ProjectTags {
  runs: { name: string; count: number }[];
  models: { name: string; count: number }[];
}

/** Detection templates report mAP per class; classification templates report precision, recall and F1. */
export interface ClassMetric {
  id: number;
  name: string;
  precision?: number;
  recall?: number;
  map50?: number;
  map50_95?: number;
  f1?: number;
}

export interface RunFile {
  id: string;
  name: string;
  kind: "image" | "file";
  bytes: number;
  url: string;
}

export interface ExportSummary {
  items?: Record<string, number>;
  boxes?: number;
  excluded?: Record<string, number>;
  as_background?: number;
  classes?: string[];
  val_source?: "release" | "holdout" | "test" | "train";
  files?: Record<string, number>;
}

export interface RunDetail extends Run {
  history: { steps: number[]; series: Record<string, (number | null)[]> };
  export: ExportSummary;
  info: Record<string, unknown>;
  class_metrics: ClassMetric[];
  files: RunFile[];
  files_url: string | null; // everything below as one zip
  metrics_csv_url: string | null; // every logged step, not thinned out
  media: RunMedia[]; // per-step pictures and tables, by key
  table_images?: Record<string, string>; // pictures table cells draw boxes on (bm.overlay), by name
  settings: { name: string; label: string; value: unknown; group: string; advanced: boolean; changed: boolean }[];
}

export interface ModelInfo {
  id: string;
  project_id: string;
  name: string;
  version: number;
  alias: string;
  aliases: string[];
  tags: string[];
  note: string;
  task: string;
  arch: string;
  template: { name: string; title: string; version: string };
  bytes: number;
  imgsz: number;
  class_count: number;
  metrics: Record<string, number>;
  created_at: string;
  run: { id: string; name: string } | null;
  release: { id: string; dataset: string; version: string; deleted: boolean } | null;
}

/** An evaluation, an export or a test run of a model. */
export interface ModelOp {
  id: string;
  kind: "eval" | "export" | "predict";
  status: RunStatus;
  error: string | null;
  wait_reason: string | null;
  duration_s: number | null;
  job_id: string | null;
  created_at: string;
  release: { id: string; dataset: string; version: string } | null;
  params: Record<string, unknown>;
  // the template version whose code ran it: a newer one than the model's own when that one lacks the code
  code?: { name?: string; version?: string; trained_with?: string };
  result: {
    metrics?: Record<string, number>;
    classes?: ClassMetric[];
    file?: string;
    predictions?: { image: string; boxes?: number[][]; top?: [number, number][] }[];
  };
  files: { name: string; kind: "image" | "file"; bytes: number; url: string }[];
  tables?: { name: string; kind: "table"; caption: string; url: string }[]; // per-image results the code recorded
  table_images?: Record<string, string>; // pictures those tables draw on, by name
}

export interface ModelDetail extends ModelInfo {
  actions: string[];
  classes: string[];
  class_metrics: ClassMetric[];
  file_name: string;
  sha256: string;
  note: string;
  config: Partial<RunConfig>;
  info: Record<string, unknown>;
  export: ExportSummary;
  download_url: string;
}

export const METRIC_LABEL: Record<string, string> = {
  map50_95: "mAP50-95",
  map50: "mAP50",
  precision: "Precision",
  recall: "Recall",
  train_box_loss: "box loss (train)",
  val_box_loss: "box loss (val)",
  train_cls_loss: "cls loss (train)",
  val_cls_loss: "cls loss (val)",
  train_dfl_loss: "dfl loss (train)",
  val_dfl_loss: "dfl loss (val)",
  top1: "Top-1",
  top5: "Top-5",
  f1: "F1",
  train_loss: "loss (train)",
  val_loss: "loss (val)",
  lr: "학습률",
  mask_map50_95: "mask mAP50-95",
  mask_map50: "mask mAP50",
  mask_precision: "mask Precision",
  mask_recall: "mask Recall",
  train_seg_loss: "seg loss (train)",
  val_seg_loss: "seg loss (val)",
};

const TASK_METRICS: Record<string, string[]> = {
  detect: ["map50_95", "map50", "precision", "recall"],
  classify: ["top1", "top5", "f1", "val_loss"],
  segment: ["mask_map50_95", "mask_map50", "map50_95", "map50"],
};

/** The four numbers shown in tables and tiles for a task; the first is the one runs are ranked by. */
export function keyMetrics(task: string | undefined): string[] {
  return TASK_METRICS[task ?? "detect"] ?? TASK_METRICS.detect;
}

/** How good a class is, whichever number its template reports. */
export function classScore(c: ClassMetric): number {
  return c.map50_95 ?? c.f1 ?? 0;
}

export const KEY_METRICS = ["map50_95", "map50", "precision", "recall"] as const;

export const ALIAS_LABEL: Record<string, string> = { champion: "champion", challenger: "challenger" };

export function metric(value: number | null | undefined, digits = 3): string {
  return value == null || Number.isNaN(value) ? "–" : value.toFixed(digits);
}

/** Signed difference, e.g. "+0.012"; "0.000" when equal. */
export function delta(a: number | undefined, b: number | undefined, digits = 3): string {
  if (a == null || b == null) return "–";
  const d = a - b;
  return `${d > 0 ? "+" : ""}${d.toFixed(digits)}`;
}

export const ACTIVE: RunStatus[] = ["queued", "running"];

export const STAGE_LABEL: Record<string, string> = {
  queued: "대기",
  sync: "이미지 받기",
  data: "데이터 준비",
  env: "환경 준비",
  train: "학습",
  upload: "모델 등록",
};
