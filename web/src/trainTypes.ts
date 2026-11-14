export type RunStatus = "queued" | "running" | "succeeded" | "failed" | "canceled";

export interface RunConfig {
  model: string;
  epochs: number;
  imgsz: number;
  batch: number;
  workers?: number;
  pretrained?: boolean;
  classes?: number[];
  unlabelled?: string;
  val_fraction?: number;
  gpus?: number;
  model_name?: string;
  extra?: Record<string, unknown>;
}

export interface Run {
  id: string;
  name: string;
  project_id: string;
  trainer: string;
  trainer_title: string;
  status: RunStatus;
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
}

export interface ModelInfo {
  id: string;
  project_id: string;
  name: string;
  version: number;
  alias: string;
  task: string;
  arch: string;
  trainer: string;
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
