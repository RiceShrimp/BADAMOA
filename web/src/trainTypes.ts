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
  description: string;
  license: string;
  builtin: boolean;
  data_format: string;
  entrypoints: string[];
  params: TemplateParam[];
  environment: { default: string; requirements: string[] };
  bytes: number;
  models: number;
  created_at: string;
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
  kind: string;
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
}

export interface Run {
  id: string;
  name: string;
  project_id: string;
  template: { id: string; name: string; title: string; version: string; variant: string };
  environment: { kind: string; name: string; id?: string };
  status: RunStatus;
  stage: "" | "data" | "env" | "train" | "upload";
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
  model: { id: string; name: string; version: number; alias: string } | null;
  source_run_id: string | null;
}

export interface ClassMetric {
  id: number;
  name: string;
  precision: number;
  recall: number;
  map50: number;
  map50_95: number;
}

export interface ExportSummary {
  items?: Record<string, number>;
  boxes?: number;
  excluded?: Record<string, number>;
  as_background?: number;
  classes?: string[];
  val_source?: "release" | "holdout" | "train";
  files?: Record<string, number>;
}

export interface RunDetail extends Run {
  history: { steps: number[]; series: Record<string, (number | null)[]> };
  export: ExportSummary;
  info: Record<string, unknown>;
  class_metrics: ClassMetric[];
  settings: { name: string; label: string; value: unknown; group: string; advanced: boolean; changed: boolean }[];
}

export interface ModelInfo {
  id: string;
  project_id: string;
  name: string;
  version: number;
  alias: string;
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

export interface ModelDetail extends ModelInfo {
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
};

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
  data: "데이터 준비",
  env: "환경 준비",
  train: "학습",
  upload: "모델 등록",
};
