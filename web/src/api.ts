import type { DiskAlert } from "./DiskAlerts";

const TOKEN_KEY = "badamoa.token";

export function getToken(): string | null {
  try {
    return localStorage.getItem(TOKEN_KEY);
  } catch {
    return null;
  }
}

export function setToken(token: string | null): void {
  try {
    if (token) localStorage.setItem(TOKEN_KEY, token);
    else localStorage.removeItem(TOKEN_KEY);
  } catch {
    /* storage unavailable: session lasts until reload */
  }
}

export class ApiError extends Error {
  constructor(
    public status: number,
    message: string,
  ) {
    super(message);
  }
}

export async function api<T>(path: string, init: { method?: string; body?: unknown } = {}): Promise<T> {
  const headers: Record<string, string> = {};
  const token = getToken();
  if (token) headers.Authorization = `Bearer ${token}`;
  if (init.body !== undefined) headers["Content-Type"] = "application/json";
  const resp = await fetch(`/api${path}`, {
    method: init.method ?? (init.body !== undefined ? "POST" : "GET"),
    headers,
    body: init.body !== undefined ? JSON.stringify(init.body) : undefined,
  });
  if (!resp.ok) {
    let detail = resp.statusText;
    try {
      const data = await resp.json();
      if (typeof data.detail === "string") detail = data.detail;
    } catch {
      /* non-JSON error body */
    }
    throw new ApiError(resp.status, detail);
  }
  return (await resp.json()) as T;
}

/** Send a file as the raw request body (template bundles, sample images). */
export async function apiUpload<T>(path: string, file: Blob, contentType = "application/zip"): Promise<T> {
  const headers: Record<string, string> = { "Content-Type": contentType || "application/octet-stream" };
  const token = getToken();
  if (token) headers.Authorization = `Bearer ${token}`;
  const resp = await fetch(`/api${path}`, { method: "POST", headers, body: file });
  if (!resp.ok) {
    let detail = resp.statusText;
    try {
      const data = await resp.json();
      if (typeof data.detail === "string") detail = data.detail;
    } catch {
      /* non-JSON error body */
    }
    throw new ApiError(resp.status, detail);
  }
  return (await resp.json()) as T;
}

export interface User {
  id: string;
  username: string;
  role: "admin" | "member";
  template_upload?: boolean; // a member an admin lets upload model kinds (as drafts)
  can_upload_templates?: boolean;
}

export interface Gpu {
  index: number;
  name: string;
  memory_total_mb: number | null;
  driver_version?: string;
}

export interface GpuMetric {
  index: number;
  memory_used_mb: number | null;
  memory_total_mb: number | null;
  utilization_pct: number | null;
  temperature_c: number | null;
  power_w?: number | null;
  power_limit_w?: number | null;
  users?: string[];
}

export interface GpuState {
  index: number;
  state: "job" | "busy" | "free" | "reserved";
  job_id: string | null;
  users: string[];
  reserved?: boolean; // kept for people: no job is sent to it
}

export interface DiskMetric {
  mount: string;
  device: string;
  fstype: string;
  total_gb: number;
  used_gb: number;
  free_gb: number;
  pct: number;
}

export interface PhysicalDisk {
  name: string;
  model: string | null;
  kind: string | null;
  read_mbps: number | null;
  write_mbps: number | null;
  busy_pct: number | null;
}

export interface UserUsage {
  user: string;
  cpu_cores: number;
  memory_mb: number;
  processes: number;
  gpus: number[];
  gpu_memory_mb: number;
  active: boolean;
}

export interface LoginSession {
  user: string | null;
  terminal: string | null;
  host: string | null;
  started: number;
}

export interface DiskForecast {
  mount: string;
  history_days: number;
  delta_7d_gb: number | null;
  days_until_full: number | null;
}

export interface MetricHistory {
  range: string;
  resolution: string;
  step_s: number;
  from: string;
  to: string;
  timestamps: string[];
  series: Record<string, { avg: (number | null)[]; max: (number | null)[] }>;
}

export interface NodeInfo {
  id: string;
  name: string;
  status: "online" | "offline" | "disabled";
  os: string;
  roles: string[];
  is_local: boolean;
  enabled: boolean;
  agent_version: string;
  last_seen_at: string | null;
  info: {
    hostname?: string;
    os_release?: string;
    cpu_model?: string | null;
    cpu_count?: number;
    memory_total_mb?: number;
    gpus: Gpu[];
  };
  metrics: {
    cpu_pct?: number;
    load?: number[] | null;
    cpu_temp_c?: number | null;
    cpu_power_w?: number | null;
    memory_used_mb?: number;
    memory_total_mb?: number;
    uptime_s?: number;
    disk_free_gb?: number;
    disk_total_gb?: number;
    gpus?: GpuMetric[];
    disks?: DiskMetric[];
    physical_disks?: PhysicalDisk[];
    users?: UserUsage[];
    sessions?: LoginSession[];
  };
  gpu_states: GpuState[];
}

export interface NodeDetailInfo extends NodeInfo {
  disk_forecast: DiskForecast[];
  recent_jobs: Job[];
  disk_alerts?: DiskAlert[]; // disks BADAMOA writes to there that are nearly full
  docker_target?: string; // where Docker's data-root could go, when Docker's disk is nearly full
}

export interface Job {
  id: string;
  type: string;
  title: string;
  params: Record<string, unknown>;
  status: "queued" | "running" | "succeeded" | "failed" | "canceled";
  node_id: string | null;
  node_name: string | null;
  created_via: string;
  created_at: string;
  duration_s: number | null;
  cancel_requested: boolean;
  result: unknown;
  error: string | null;
  log_count: number;
  gpus_requested: number;
  gpu_indices: number[];
  wait_reason?: string | null;
}

export interface JobType {
  type: string;
  title: string;
  description: string;
  gpus: number;
  params_schema: { properties?: Record<string, { type: string; default?: unknown }> };
}

export interface LogLine {
  seq: number;
  ts: string;
  stream: string;
  text: string;
}

/** The address nodes are told to reach the Hub at, and where it came from. */
export interface HubAddress {
  url: string;
  source: "saved" | "launch" | "browser" | "detected" | "local-only";
  candidates: string[];
  launch: string | null;
  listens_locally_only: boolean;
}

export interface JoinToken {
  token: string;
  roles: string[];
  expires_at: string;
  hub_url: string;
  address: HubAddress;
  node_package: { version: string; ready: boolean };
  commands: { linux: string; linux_user: string; windows: string; dev: string };
}

export interface ApiTokenInfo {
  id: string;
  name: string;
  prefix: string;
  created_at: string;
  last_used_at: string | null;
}
