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

export interface User {
  id: string;
  username: string;
  role: "admin" | "member";
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
    cpu_count?: number;
    memory_total_mb?: number;
    gpus: Gpu[];
  };
  metrics: {
    cpu_pct?: number;
    memory_used_mb?: number;
    memory_total_mb?: number;
    disk_free_gb?: number;
    disk_total_gb?: number;
    gpus?: GpuMetric[];
  };
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
}

export interface JobType {
  type: string;
  title: string;
  description: string;
  params_schema: { properties?: Record<string, { type: string; default?: unknown }> };
}

export interface LogLine {
  seq: number;
  ts: string;
  stream: string;
  text: string;
}

export interface JoinToken {
  token: string;
  roles: string[];
  expires_at: string;
  hub_url: string;
  commands: { linux: string; windows: string; dev: string };
}

export interface ApiTokenInfo {
  id: string;
  name: string;
  prefix: string;
  created_at: string;
  last_used_at: string | null;
}
