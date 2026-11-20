import { type FormEvent, type ReactNode, useEffect, useRef, useState } from "react";
import { GpuBusyPanel, HubAutostartPanel, MyAccount, UsersPanel } from "./AdminSettings";
import { DockerRegistry, ImageInRegistry, useRegistryImages } from "./DockerRegistry";
import { fileSize } from "./Files";
import { type NodeInfo, type User, api, apiUpload } from "./api";
import { type Project, announceTrash, taskLabel } from "./catalogTypes";
import type { Environment, EnvironmentNode, Template, TemplateCheck, TemplateVersion } from "./trainTypes";
import { Badge, CopyBlock, Icon, Modal, timeAgo, useHubUrl, usePolling } from "./ui";

const ENTRY_LABEL: Record<string, string> = { train: "학습", eval: "평가", predict: "추론", export: "내보내기" };
const STEP_LABEL: Record<string, string> = {
  data: "시험 데이터",
  env: "환경",
  train: "학습",
  predict: "테스트",
  autolabel: "오토라벨",
  eval: "평가",
  export: "내보내기",
};

/** Packages that did not come from an index, as " · mylib (git 1a2b3c4)". */
function sourceNote(c: TemplateCheck): string {
  const odd = Object.entries(c.sources ?? {});
  if (!odd.length) return "";
  return ` · ${odd
    .slice(0, 3)
    .map(([name, s]) => `${name} (${s.kind}${s.commit ? ` ${s.commit.slice(0, 7)}` : ""})`)
    .join(", ")}${odd.length > 3 ? ` 외 ${odd.length - 3}개` : ""}`;
}

/** A trial step's name; a kind of several tasks is tried on each in turn, so its steps say which (Seg 학습). */
function stepName(s: { step: string; task?: string }): string {
  return `${s.task ? `${taskLabel(s.task)} ` : ""}${STEP_LABEL[s.step] ?? s.step}`;
}

/** A template version's trial as an admin reads it: one line, the details on hover, the log a click away. */
function TrialLine({ t }: { t: Template }) {
  const c: TemplateCheck = t.check;
  const steps = (c.steps ?? [])
    .map((s) => (s.skipped ? `${stepName(s)}: ${s.detail}` : `${stepName(s)} ${s.took_s}초 · ${s.detail}`))
    .join("\n");
  const log = c.job_id ? (
    <a className="link" href={`#/jobs/${c.job_id}`}>
      로그
    </a>
  ) : null;
  let line: ReactNode;
  if (c.state === "checking") {
    line = (
      <>
        <Badge kind="running">시험 실행 중</Badge>{" "}
        {c.node && (
          <span className="muted">
            {c.node}
            {c.device ? ` · ${c.device}` : ""}
          </span>
        )}{" "}
        {log}
      </>
    );
  } else if (c.state === "passed") {
    line = (
      <span title={steps}>
        <Badge kind="queued">통과 · 허용 대기</Badge>{" "}
        <span className="muted">
          {c.node ?? ""}
          {c.device ? ` · ${c.device}` : ""}
          {c.checked_at ? ` · ${timeAgo(c.checked_at)}` : ""}
        </span>{" "}
        {log}
      </span>
    );
  } else if (c.state === "ok") {
    line = (
      <span className="muted" title={steps}>
        {c.via === "run"
          ? "시험 실행 전에 등록 · 학습이 성공한 버전"
          : `허용됨${c.approved_by ? ` (${c.approved_by}${c.approved_at ? `, ${timeAgo(c.approved_at)}` : ""})` : ""} · 시험 실행 통과${c.node ? ` · ${c.node}` : ""}`}{" "}
        {log}
      </span>
    );
  } else if (c.state === "failed") {
    line = (
      <span className="bad-text" title={steps}>
        실패 — {STEP_LABEL[c.failed ?? ""] ?? "시험 실행"}: {c.reason} {log}
      </span>
    );
  } else if (c.state === "pending") {
    line = (
      <span className="warn-text">
        시험 전 — "시험 실행"으로 학습 서버에서 짧게 돌려 본 뒤 허용합니다{c.reason ? ` (지난 시도: ${c.reason})` : ""}
      </span>
    );
  } else if (c.state === "allowed") {
    line = (
      <span className="muted" title={c.reason ? `시험 실행: ${c.reason}` : undefined}>
        통과 없이 허용 — {c.why} ({c.approved_by ?? c.by}
        {c.approved_at ? `, ${timeAgo(c.approved_at)}` : ""})
      </span>
    );
  } else {
    line = <span className="muted">시험 실행 전에 등록됨 · 지금처럼 모두 쓸 수 있음{c.reason ? ` (지난 시도: ${c.reason})` : ""}</span>;
  }
  return (
    <div className="small trial-line">
      {line}
      {(c.notes ?? []).map((n) => (
        <div key={n} className="muted">
          알림: {n}
        </div>
      ))}
      {c.legacy && c.state !== "legacy" && <div className="muted">시험 실행 전에 등록된 버전이라 결과와 상관없이 모두 쓸 수 있습니다</div>}
      {!c.offered && (
        <div className="muted">{t.offered_version ? `다른 사람에게는 ${t.offered_version} 버전이 보입니다` : "다른 사람에게는 보이지 않습니다"}</div>
      )}
      {(c.pinned ?? 0) > 0 && (
        <div className="muted" title="다른 서버도 이 버전들로 설치합니다 (PyTorch는 버전만, 빌드는 서버 드라이버에 맞춤)">
          패키지 {c.pinned}개 버전 고정{sourceNote(c)}
        </div>
      )}
      {t.local_check?.ok && (
        <div className="muted" title={(t.local_check.steps ?? []).map((s) => `${stepName(s)}: ${s.detail}`).join("\n")}>
          올린 사람 PC에서 통과{t.local_check.by ? ` (${t.local_check.by})` : ""}: {(t.local_check.steps ?? []).filter((s) => !s.skipped).length}단계 · Python{" "}
          {t.local_check.python} · {(t.local_check.os ?? "").split("-")[0]}
          {t.local_check.gpus?.length ? ` · ${t.local_check.gpus[0]}` : ""}
          {t.local_check.device && t.local_check.device !== "cpu" ? " (GPU로)" : " (CPU로)"}
          {t.local_check.sdk ? ` · BADAMOA ${t.local_check.sdk}` : ""}
          {t.local_check.at ? ` · ${timeAgo(t.local_check.at)}` : ""}
        </div>
      )}
      {t.uploaded_by && <div className="muted">올린 사람: {t.uploaded_by}</div>}
      {t.reads_weights_from && <div className="muted">이전 버전({t.reads_weights_from})의 모델에도 이 버전의 코드를 씀</div>}
      {t.code_url && (
        <a className="link" href={t.code_url} download>
          이 버전의 코드 받기
        </a>
      )}
    </div>
  );
}

/** The settings menu; ``admin`` entries are shown to admins only. */
const SECTIONS: { id: string; label: string; admin?: boolean }[] = [
  { id: "templates", label: "모델 종류" },
  { id: "environments", label: "환경" },
  { id: "projects", label: "프로젝트 기본값", admin: true },
  { id: "gpu", label: "GPU 배정" },
  { id: "storage", label: "저장 공간" },
  { id: "users", label: "사용자", admin: true },
  { id: "hub", label: "Hub 실행", admin: true },
  { id: "account", label: "내 계정" },
  { id: "monitoring", label: "외부 모니터링" },
];

/**
 * What is set up once so that everyone else only picks by name: model templates,
 * environments, project defaults, GPU rules, accounts. Members can look; only admins change things.
 * Each part has its own address (``#/settings/<part>``).
 */
export function Settings({ isAdmin, nodes, user, sub }: { isAdmin: boolean; nodes: NodeInfo[] | null; user: User; sub?: string }) {
  const sections = SECTIONS.filter((s) => isAdmin || !s.admin);
  const part = sections.find((s) => s.id === sub)?.id ?? "templates";
  return (
    <>
      <div className="page-head">
        <div>
          <h1>설정</h1>
          <p className="muted">
            {isAdmin
              ? "여기서 한 번 등록해 두면, 학습을 시작하는 사람은 이름만 고르면 됩니다."
              : "학습할 때 고를 수 있는 모델 종류와 환경입니다. 추가나 변경은 관리자에게 요청하세요."}
          </p>
        </div>
      </div>
      <nav className="tabs" aria-label="설정">
        {sections.map((s) => (
          <a key={s.id} href={`#/settings/${s.id}`} className={part === s.id ? "active" : ""}>
            {s.label}
          </a>
        ))}
      </nav>
      {part === "templates" && <Templates isAdmin={isAdmin} user={user} />}
      {part === "environments" && isAdmin && <DockerRegistry />}
      {part === "environments" && <Environments isAdmin={isAdmin} nodes={nodes ?? []} />}
      {part === "environments" && <Packages isAdmin={isAdmin} />}
      {part === "projects" && <ProjectDefaults />}
      {part === "gpu" && <GpuBusyPanel isAdmin={isAdmin} />}
      {part === "storage" && <StorageUsage isAdmin={isAdmin} />}
      {part === "users" && <UsersPanel me={user} />}
      {part === "hub" && <HubAutostartPanel />}
      {part === "account" && <MyAccount me={user} />}
      {part === "monitoring" && <ExternalMonitoring />}
    </>
  );
}

// ---- for teams that already run Prometheus and Grafana ---------------------------------

function ExternalMonitoring() {
  const hub = new URL(useHubUrl()); // Prometheus runs on another machine: not the browser's localhost
  const target = hub.host;
  const scheme = hub.protocol.replace(":", "");
  const config = [
    "scrape_configs:",
    "  - job_name: badamoa",
    `    scheme: ${scheme}`,
    "    metrics_path: /metrics",
    "    authorization:",
    "      credentials: <토큰>",
    "    static_configs:",
    `      - targets: ["${target}"]`,
  ].join("\n");
  return (
    <section className="card">
      <div className="card-head">
        <h3>
          Prometheus · Grafana로 보내기 <span className="muted small">(사내에 이미 있다면)</span>
        </h3>
      </div>
      <p className="muted small">
        Hub의 <code>/metrics</code>가 노드 · GPU · 디스크 · 작업 대기열을 Prometheus 형식으로 내줍니다. 가져갈 때만 계산하고 따로
        저장하지 않으므로 BADAMOA 쪽 부담은 거의 없습니다. 토큰은 <a href="#/agents">에이전트 연결</a>에서 만듭니다.
      </p>
      <CopyBlock label="prometheus.yml" text={config} />
      <p className="muted small">
        지표 이름: <code>badamoa_node_up</code>, <code>badamoa_node_cpu_percent</code>, <code>badamoa_gpu_utilization_percent</code>,{" "}
        <code>badamoa_gpu_memory_used_bytes</code>, <code>badamoa_gpu_state</code>(free · job · busy · reserved · offline),{" "}
        <code>badamoa_disk_used_bytes</code>, <code>badamoa_jobs</code> 등
      </p>
    </section>
  );
}

// ---- where everything is kept ----------------------------------------------------------

interface Usage {
  path: string | null; // admins only
  bytes: number;
  files: number;
  /** When the Hub last counted its folder (epoch seconds; it counts in the background every 10 minutes). */
  counted_at: number | null;
  free: number;
  disk: number;
  /** Always left free: an import that would need more is refused before it reads. */
  keep_free: number;
  /** What one more registered image takes on the Hub (thumbnail and records). */
  per_image: number;
  /** Images whose thumbnail is still the older, larger kind (320 px JPEG); the Hub shrinks them between imports. */
  old_thumbs: number;
  shrinking: boolean;
  parts: {
    datasets: number;
    models: number;
    outputs: number;
    checkpoints: number;
    trash: number;
    database: number;
    other: number;
  };
  database: { bytes: number; empty: number; compacting: boolean; last_compact: { before: number; after: number; at: number } | null };
  policy: Policy;
  projects: {
    id: string;
    name: string;
    images: number;
    datasets: number;
    models: number;
    outputs: number;
    checkpoints: number;
    total: number;
  }[];
}

interface Policy {
  outputs_days: number;
  checkpoint_days: number;
  trash_days: number;
  workdir_days: number;
}

interface CleanupReport {
  policy: Policy;
  outputs: { files: number; bytes: number };
  checkpoints: { runs: number; bytes: number };
  removed: { files: number; bytes: number; partial_uploads: number };
}

const PART_LABEL: [keyof Usage["parts"], string, string][] = [
  ["datasets", "데이터셋", "이미지마다 256px 썸네일 하나(WebP). 원본은 노드의 폴더에 그대로 있습니다"],
  ["models", "모델", "등록된 모델의 가중치"],
  ["outputs", "산출물", "학습 · 평가 · 내보내기 · 테스트가 남긴 파일"],
  ["checkpoints", "체크포인트", "끝나지 않은 실행을 이어 가기 위한 것"],
  ["trash", "휴지통", "지운 학습 · 모델 · 프로젝트의 파일. 휴지통을 비우면 지워집니다"],
  ["database", "기록(DB)", "이미지 · 라벨 · 버전 · 학습의 기록"],
  ["other", "그 밖", "모델 종류의 코드, 올린 파일 등"],
];

function StorageUsage({ isAdmin }: { isAdmin: boolean }) {
  const [compacting, setCompacting] = useState(false);
  const { data: usage, refresh } = usePolling(() => api<Usage>("/storage"), compacting ? 3000 : 60000, [compacting]);
  const [compactError, setCompactError] = useState("");
  useEffect(() => {
    if (compacting && usage && !usage.database.compacting) setCompacting(false);
  }, [compacting, usage]);
  if (!usage) return null;
  const compact = async () => {
    const db = usage.database;
    if (!window.confirm(`DB를 압축할까요? 빈 공간 ${fileSize(db.empty)}를 디스크에 돌려줍니다. 1 GB에 1분쯤 걸리고, 그동안 기록이 잠시 멈춥니다.`))
      return;
    setCompactError("");
    try {
      await api("/storage/compact", { method: "POST" });
      setCompacting(true);
      await refresh();
    } catch (err) {
      setCompactError(err instanceof Error ? err.message : String(err));
    }
  };
  const used = usage.disk > 0 ? ((usage.disk - usage.free) / usage.disk) * 100 : 0;
  return (
    <section className="card">
      <div className="card-head">
        <div>
          <h3>저장 공간</h3>
          <div className="muted small">
            모델과 지표, 로그, 학습 · 평가가 남긴 파일은 모두 BADAMOA가 설치된 이 서버에 보관합니다. 학습 서버에는 캐시만
            남습니다.
          </div>
        </div>
        <div className="storage-total">
          <strong>{fileSize(usage.bytes)}</strong>
          <span className="muted small" title="파일 수는 Hub가 10분마다 따로 셉니다">
            파일 {usage.files.toLocaleString()}개{usage.counted_at ? ` · ${timeAgo(new Date(usage.counted_at * 1000).toISOString())} 셈` : " · 세는 중"}
          </span>
        </div>
      </div>
      {usage.path && (
        <div className="storage-path">
          <span className="muted small">위치</span> <code>{usage.path}</code>
        </div>
      )}
      <div className="env-list">
        {PART_LABEL.map(([key, label, help]) => (
          <div className="storage-part" key={key}>
            <span>
              {label} <span className="muted small">{help}</span>
            </span>
            <span className="num">{usage.parts[key] > 0 ? fileSize(usage.parts[key]) : "–"}</span>
          </div>
        ))}
      </div>
      <div className="storage-notes muted small">
        데이터셋을 등록하면 이미지 한 장에 약 {fileSize(usage.per_image)}(썸네일과 기록)가 Hub에 쌓입니다. 같은 이미지는 버전이 달라도 한 번만
        저장합니다. 등록할 이미지가 들어갈 자리가 없으면(늘 {fileSize(usage.keep_free)}는 남겨 둠) 이미지를 읽기 전에 멈춥니다.
      </div>
      {usage.old_thumbs > 0 && (
        <div className="muted small">
          예전 크기(320px JPEG)의 썸네일 {usage.old_thumbs.toLocaleString()}장을{" "}
          {usage.shrinking ? "256px WebP로 줄이고 있습니다" : "등록이 없을 때 256px WebP로 줄입니다"}. 원본은 다시 읽지 않습니다.
        </div>
      )}
      {usage.database.empty > 0 && (
        <div className="storage-compact">
          <span className="small">
            기록(DB) 안의 빈 공간 <strong>{fileSize(usage.database.empty)}</strong>
            <span className="muted"> · 지운 기록의 자리입니다. 압축하면 디스크에 돌아옵니다 (휴지통을 비운 뒤 많이 남으면 Hub가 쉴 때 알아서 함)</span>
          </span>
          {isAdmin && (
            <button className="small" onClick={() => void compact()} disabled={compacting || usage.database.compacting}>
              {compacting || usage.database.compacting ? "압축하는 중…" : "DB 압축"}
            </button>
          )}
        </div>
      )}
      {usage.database.last_compact && (
        <div className="muted small">
          마지막 압축: {fileSize(usage.database.last_compact.before)} → {fileSize(usage.database.last_compact.after)} (
          {timeAgo(new Date(usage.database.last_compact.at * 1000).toISOString())})
        </div>
      )}
      {compactError && <div className="error">{compactError}</div>}
      <div className="storage-disk">
        <div className="meter">
          <div className={used > 90 ? "meter-fill warn" : "meter-fill"} style={{ width: `${Math.min(100, used).toFixed(1)}%` }} />
        </div>
        <span className={used > 90 ? "small warn-text" : "muted small"}>
          디스크 여유 {fileSize(usage.free)} / 전체 {fileSize(usage.disk)}
        </span>
      </div>
      {usage.projects.length > 0 && (
        <details className="fold inner">
          <summary>
            프로젝트별 <span className="muted">({usage.projects.length})</span>
          </summary>
          <div className="table-wrap">
            <table className="plain">
              <thead>
                <tr>
                  <th>프로젝트</th>
                  <th className="right">이미지</th>
                  <th className="right">데이터셋</th>
                  <th className="right">모델</th>
                  <th className="right">산출물</th>
                  <th className="right">체크포인트</th>
                  <th className="right">합계</th>
                </tr>
              </thead>
              <tbody>
                {usage.projects.map((p) => (
                  <tr key={p.id}>
                    <td>{p.name}</td>
                    <td className="num right">{p.images > 0 ? `${p.images.toLocaleString()}장` : "–"}</td>
                    {(["datasets", "models", "outputs", "checkpoints", "total"] as const).map((k) => (
                      <td key={k} className="num right">
                        {p[k] > 0 ? fileSize(p[k]) : "–"}
                      </td>
                    ))}
                  </tr>
                ))}
              </tbody>
            </table>
          </div>
        </details>
      )}
      {isAdmin && <Retention policy={usage.policy} onChanged={refresh} />}
    </section>
  );
}

/** How long outputs and checkpoints are kept, and cleaning up now (after seeing what would go). */
function Retention({ policy, onChanged }: { policy: Policy; onChanged: () => Promise<unknown> }) {
  const [draft, setDraft] = useState({
    outputs: String(policy.outputs_days),
    checkpoints: String(policy.checkpoint_days),
    trash: String(policy.trash_days),
    workdirs: String(policy.workdir_days),
  });
  const [preview, setPreview] = useState<CleanupReport | null>(null);
  const [message, setMessage] = useState<{ kind: "ok" | "error"; text: string } | null>(null);
  const [busy, setBusy] = useState(false);
  const changed =
    draft.outputs !== String(policy.outputs_days) ||
    draft.checkpoints !== String(policy.checkpoint_days) ||
    draft.trash !== String(policy.trash_days) ||
    draft.workdirs !== String(policy.workdir_days);

  const act = async (work: () => Promise<void>) => {
    setBusy(true);
    setMessage(null);
    try {
      await work();
    } catch (e) {
      setMessage({ kind: "error", text: e instanceof Error ? e.message : String(e) });
    } finally {
      setBusy(false);
    }
  };
  const save = () =>
    act(async () => {
      await api("/storage/policy", {
        method: "PUT",
        body: {
          outputs_days: Number(draft.outputs) || 0,
          checkpoint_days: Number(draft.checkpoints) || 0,
          trash_days: Number(draft.trash) || 0,
          workdir_days: Number(draft.workdirs) || 0,
        },
      });
      setPreview(null);
      await onChanged();
      setMessage({ kind: "ok", text: "보관 기간을 저장했습니다. BADAMOA가 몇 시간마다 이 기준으로 정리합니다." });
    });
  const look = () => act(async () => setPreview(await api<CleanupReport>("/storage/cleanup")));
  const clean = () =>
    act(async () => {
      const done = await api<CleanupReport>("/storage/cleanup", { method: "POST" });
      setPreview(null);
      await onChanged();
      setMessage({ kind: "ok", text: `${fileSize(done.removed.bytes)}를 비웠습니다 (파일 ${done.removed.files.toLocaleString()}개).` });
    });
  const days = (v: string) => (Number(v) > 0 ? `${v}일` : "계속 보관");

  return (
    <div className="retention">
      <div className="eyebrow">보관 기간</div>
      <div className="form-row">
        <label>
          실행 · 평가의 산출물 <span className="muted small">{days(draft.outputs)}</span>
          <input type="number" min={0} max={3650} value={draft.outputs} onChange={(e) => setDraft({ ...draft, outputs: e.target.value })} />
        </label>
        <label>
          중단된 실행의 체크포인트 <span className="muted small">{days(draft.checkpoints)}</span>
          <input
            type="number"
            min={0}
            max={3650}
            value={draft.checkpoints}
            onChange={(e) => setDraft({ ...draft, checkpoints: e.target.value })}
          />
        </label>
      </div>
      <div className="form-row">
        <label>
          휴지통에 둔 것 <span className="muted small">{Number(draft.trash) > 0 ? `${draft.trash}일 뒤 완전히 지움` : "비울 때까지 보관"}</span>
          <input type="number" min={0} max={3650} value={draft.trash} onChange={(e) => setDraft({ ...draft, trash: e.target.value })} />
        </label>
        <label>
          노드의 작업 폴더 <span className="muted small">{days(draft.workdirs)}</span>
          <input
            type="number"
            min={0}
            max={3650}
            value={draft.workdirs}
            onChange={(e) => setDraft({ ...draft, workdirs: e.target.value })}
          />
        </label>
      </div>
      <p className="muted small">
        0이면 계속 보관합니다. 모델(가중치), 지표, 로그는 기간과 상관없이 남습니다. alias가 붙은 모델의 실행과 "보관" 태그를 붙인
        실행의 산출물도 남습니다. 지운 실행 · 모델의 파일은 하루가 지나면 정리됩니다. 노드의 작업 폴더는 실패하거나 멈춘 학습이 남긴
        것으로, 이어서 실행할 체크포인트는 Hub에도 있어 지워도 이어 갈 수 있습니다. 노드는 작업이 없을 때 한 시간에 한 번 정리합니다.
      </p>
      <div className="head-actions">
        <button onClick={save} disabled={busy || !changed}>
          저장
        </button>
        <button onClick={look} disabled={busy}>
          지금 정리하면…
        </button>
      </div>
      {preview && preview.removed.files + preview.outputs.files + preview.checkpoints.runs === 0 && (
        <div className="notice">지금 정리할 것이 없습니다.</div>
      )}
      {preview && preview.removed.files + preview.outputs.files + preview.checkpoints.runs > 0 && (
        <div className="notice cleanup-preview">
          <div>
            지금 정리하면 <strong>{fileSize(preview.removed.bytes)}</strong>가 비워집니다: 오래된 산출물 기록{" "}
            {preview.outputs.files.toLocaleString()}개, 체크포인트 {preview.checkpoints.runs}개, 쓰이지 않는 파일 포함 저장소 파일{" "}
            {preview.removed.files.toLocaleString()}개. 되돌릴 수 없습니다.
          </div>
          <div className="head-actions">
            <button onClick={() => setPreview(null)} disabled={busy}>
              취소
            </button>
            <button className="danger" onClick={clean} disabled={busy}>
              정리
            </button>
          </div>
        </div>
      )}
      {message && <div className={message.kind === "ok" ? "notice" : "error"}>{message.text}</div>}
    </div>
  );
}

// ---- model templates -----------------------------------------------------------------

/** "환경 관리": install a model kind's environment on chosen servers now, or take it off one. Only servers that train qualify. */
function PrepareDialog({
  template,
  onStart,
  onRemove,
  onClose,
}: {
  template: Template;
  onStart: (nodeIds: string[]) => Promise<void>;
  onRemove: (nodeId: string) => Promise<void>;
  onClose: () => void;
}) {
  const env = template.environment;
  const servers = env.servers ?? [];
  const [picked, setPicked] = useState<string[]>(() => servers.filter((s) => !s.why && !s.state).map((s) => s.id));
  const [busy, setBusy] = useState(false);
  const toggle = (id: string) => setPicked(picked.includes(id) ? picked.filter((x) => x !== id) : [...picked, id]);
  const docker = env.kind === "docker";
  const what = docker ? `Docker 이미지 ${env.image}` : `가상환경, 패키지 ${env.requirements.length}개`;
  const remove = (id: string, name: string) => {
    if (!window.confirm(`${name}에서 ${template.title}의 환경을 지울까요? 다음 학습 때 다시 설치합니다.`)) return;
    setBusy(true);
    void onRemove(id).finally(() => setBusy(false));
  };
  return (
    <Modal title={`${template.title} 환경 관리`} onClose={onClose}>
      <p className="muted small">
        고른 서버에 이 모델 종류의 학습 환경({what})을 지금 {docker ? "받습니다" : "설치합니다"}. 첫 학습이 기다리지 않습니다.
        학습용 · 오토라벨용 서버만 고를 수 있습니다. 준비된 서버에서는 지울 수 있습니다
        {docker ? " (BADAMOA가 받기 전부터 있던 이미지는 남김)" : ""}.
      </p>
      <div className="node-pick">
        {servers.map((s) => {
          const why = s.why || (s.state === "ready" ? "준비됨" : s.state === "preparing" ? "준비 중" : "");
          return (
            <label key={s.id} className={s.why ? "off" : ""}>
              <input type="checkbox" checked={picked.includes(s.id)} disabled={Boolean(why)} onChange={() => toggle(s.id)} />
              <span className="node-pick-name">
                <i className={`dot ${s.status}`} />
                {s.name}
              </span>
              {s.state === "ready" ? (
                <button
                  type="button"
                  className="mini"
                  disabled={busy}
                  onClick={(e) => {
                    e.preventDefault();
                    remove(s.id, s.name);
                  }}
                >
                  지우기
                </button>
              ) : (
                <span />
              )}
              <span className={`tag ${why ? "muted" : ""}`}>{why || (docker ? "받기" : "설치")}</span>
            </label>
          );
        })}
      </div>
      {servers.length > 0 && !servers.some((s) => !s.why) && (
        <p className="error">
          {servers.some((s) => s.trains)
            ? "Docker가 있는 학습 서버가 없습니다."
            : "학습용 · 오토라벨용 서버가 없습니다. 노드의 설정에서 \u201c학습용\u201d이나 \u201c오토라벨용\u201d을 켜세요."}
        </p>
      )}
      <div className="modal-actions">
        <button type="button" onClick={onClose}>
          취소
        </button>
        <button
          className="primary"
          disabled={busy || picked.length === 0}
          onClick={() => {
            setBusy(true);
            void onStart(picked).finally(() => setBusy(false));
          }}
        >
          {picked.length ? `서버 ${picked.length}대에 설치` : "설치할 서버를 고르세요"}
        </button>
      </div>
    </Modal>
  );
}

/** What someone who may upload model kinds runs on their own PC, with this Hub's address filled in. */
function TemplateKit() {
  const { data: kit, error } = usePolling(() => api<TemplateKitInfo>("/templates/kit"), 600000);
  const [token, setToken] = useState<string | null>(null);
  const [failed, setFailed] = useState<string | null>(null);
  const makeToken = async () => {
    setFailed(null);
    try {
      const made = await api<{ token: string }>("/tokens", { body: { name: "모델 종류 올리기" } });
      setToken(made.token);
    } catch (e) {
      setFailed(e instanceof Error ? e.message : String(e));
    }
  };
  if (error) return <div className="error">{error}</div>;
  if (!kit) return <p className="muted">불러오는 중…</p>;
  return (
    <div className="kit-steps">
      <div className="howto-guide">
        <a className="btn primary-link" href="#/guides/templates">
          <Icon name="book" size={14} /> 템플릿 가이드 열기
        </a>
        <span className="muted small">선언 파일 · 학습 · 평가 · 내보내기 · 오토라벨 코드, SDK 함수, 시험 실행과 허용, 자주 나는 실패</span>
      </div>
      <CopyBlock label="1. 내 PC에 BADAMOA 설치 (uv)" text={kit.install_uv} />
      <CopyBlock label="pip를 쓴다면" text={kit.install_pip} />
      {kit.wheel_url && (
        <p className="muted small">
          설치 파일을 직접 받으려면:{" "}
          <a className="link" href={kit.wheel_url}>
            {kit.wheel_url.split("/").pop()}
          </a>
        </p>
      )}
      <CopyBlock label="2. 뼈대 만들기 — 만들어진 학습 코드 파일을 내 코드로 고칩니다" text={kit.init} />
      <CopyBlock label="3. 내 PC에서 시험 실행 — 학습 · 테스트 · 평가 · 내보내기를 해 보고 보고서를 남깁니다" text={kit.check} />
      <div className="eyebrow">4. 올릴 때 쓸 API 토큰</div>
      {token ? (
        <>
          <CopyBlock text={token} />
          <p className="warn-text small">이 토큰은 지금 한 번만 보입니다. 잃어버리면 새로 만드세요 (에이전트 연결 화면에서 지울 수 있음).</p>
        </>
      ) : (
        <button className="small" onClick={() => void makeToken()}>
          토큰 만들기
        </button>
      )}
      {failed && <div className="error">{failed}</div>}
      <CopyBlock label="5. 올리기 — 3의 보고서를 함께 보냅니다" text={token ? kit.push.replace("<토큰>", token) : kit.push} />
      <p className="muted small">
        {kit.draft
          ? "올린 것은 초안으로, 나만 보고 학습에 쓸 수 있습니다. 관리자가 서버에서 시험 실행하고 허용하면 모두의 학습 화면에 나옵니다. 3을 통과하지 않은 것은 Hub가 받지 않습니다."
          : '올린 것은 "시험 전"으로 등록됩니다. 아래 목록에서 "시험 실행" 후 통과한 것을 "허용"하면 모두에게 보입니다.'}
        {" "}PyPI에 없는 사내 패키지는 설정 → 환경의 "사내 패키지"에 wheel로 올리고 requirements에 이름을 적습니다.
      </p>
    </div>
  );
}

interface TemplateKitInfo {
  hub_url: string;
  install_uv: string;
  install_pip: string;
  wheel_url: string;
  init: string;
  check: string;
  push: string;
  draft: boolean;
}

/** A model kind's description: what the table shows under its name. Admins write it here; empty, the bundle's own. */
function TemplateAbout({ t, isAdmin, onClose, onSaved }: { t: Template; isAdmin: boolean; onClose: () => void; onSaved: () => void }) {
  const [text, setText] = useState(t.description);
  const [busy, setBusy] = useState(false);
  const [error, setError] = useState<string | null>(null);
  const declared = t.declared_description ?? "";
  const save = async (value: string) => {
    setBusy(true);
    setError(null);
    try {
      await api(`/templates/${t.id}/description`, { method: "PUT", body: { text: value } });
      onSaved();
      onClose();
    } catch (e) {
      setError(e instanceof Error ? e.message : String(e));
      setBusy(false);
    }
  };
  return (
    <Modal title={t.title} onClose={onClose}>
      <p className="muted small">
        {t.name} · 지금 버전 {t.version}
        {t.versions > 1 ? ` (버전 ${t.versions}개)` : ""}
        {t.builtin ? " · 기본 제공" : t.uploaded_by ? ` · 올린 사람 ${t.uploaded_by}` : ""}
      </p>
      {isAdmin ? (
        <form
          onSubmit={(e) => {
            e.preventDefault();
            void save(text);
          }}
        >
          <label>
            설명
            <textarea rows={7} value={text} onChange={(e) => setText(e.target.value)} placeholder={declared || "이 모델 종류가 무엇인지, 언제 쓰는지"} />
          </label>
          <p className="muted small">
            이 표와 학습 창의 모델 종류 아래에 나옵니다. 비우고 저장하면 올린 파일(badamoa.toml)의 설명으로 돌아갑니다
            {declared ? `: "${declared}"` : "."}
          </p>
          {error && <div className="error">{error}</div>}
          <div className="modal-actions">
            {text.trim() !== declared && declared && (
              <button type="button" onClick={() => setText(declared)} disabled={busy}>
                올린 파일의 설명으로
              </button>
            )}
            <button type="button" onClick={onClose}>
              취소
            </button>
            <button className="primary" disabled={busy || text.trim() === t.description}>
              {busy ? "저장 중…" : "저장"}
            </button>
          </div>
        </form>
      ) : (
        <p className="memo-text">{t.description || "설명이 없습니다."}</p>
      )}
    </Modal>
  );
}

const VERSION_STATE: Record<TemplateCheck["state"], string> = {
  pending: "시험 전",
  checking: "시험 실행 중",
  passed: "통과 · 허용 대기",
  ok: "허용됨",
  failed: "시험 실패",
  allowed: "통과 없이 허용",
  legacy: "시험 실행 전에 등록",
};

/** One version's memo: read, or for admins written in place. */
function VersionMemo({ v, isAdmin, onSaved }: { v: TemplateVersion; isAdmin: boolean; onSaved: () => void }) {
  const [text, setText] = useState(v.memo);
  const [busy, setBusy] = useState(false);
  const [error, setError] = useState<string | null>(null);
  useEffect(() => setText(v.memo), [v.memo]);
  if (!isAdmin) return v.memo ? <p className="memo-text">{v.memo}</p> : null;
  const save = async () => {
    setBusy(true);
    setError(null);
    try {
      await api(`/templates/${v.id}/memo`, { method: "PUT", body: { text } });
      onSaved();
    } catch (e) {
      setError(e instanceof Error ? e.message : String(e));
    } finally {
      setBusy(false);
    }
  };
  return (
    <div className="version-memo">
      <textarea rows={3} value={text} onChange={(e) => setText(e.target.value)} placeholder="이 버전의 메모 — 무엇이 바뀌었는지, 쓸 때 주의할 점" />
      <div className="version-memo-actions">
        {error && <span className="bad-text small">{error}</span>}
        <button type="button" className="mini" onClick={() => void save()} disabled={busy || text.trim() === v.memo}>
          {busy ? "저장 중…" : "메모 저장"}
        </button>
      </div>
    </div>
  );
}

/** Every version of a model kind, newest first: when and by whom, its trial, the bundle's description, the memo. */
function TemplateVersions({ t, isAdmin, onClose, onSaved }: { t: Template; isAdmin: boolean; onClose: () => void; onSaved: () => void }) {
  const [versions, setVersions] = useState<TemplateVersion[] | null>(null);
  const [error, setError] = useState<string | null>(null);
  const load = () =>
    api<TemplateVersion[]>(`/templates/${t.id}/versions`)
      .then(setVersions)
      .catch((e) => setError(e instanceof Error ? e.message : String(e)));
  useEffect(() => {
    void load();
  }, [t.id]); // eslint-disable-line react-hooks/exhaustive-deps
  return (
    <Modal title={`${t.title} — 버전`} onClose={onClose}>
      {error && <div className="error">{error}</div>}
      {!versions && !error && <p className="muted">불러오는 중…</p>}
      <div className="version-list">
        {versions?.map((v) => (
          <div key={v.id} className={`version-item ${v.id === t.id ? "current" : ""}`}>
            <div className="version-head">
              <strong>{v.version}</strong>
              {v.id === t.id && <span className="tag">지금 보이는 버전</span>}
              <span className={`small ${v.check.state === "failed" ? "bad-text" : v.offered ? "" : "warn-text"}`}>
                {v.builtin ? "기본 제공" : (VERSION_STATE[v.check.state] ?? v.check.state)}
                {!v.builtin && !v.offered && v.check.state !== "failed" ? " · 다른 사람에게 안 보임" : ""}
              </span>
              <span className="muted small">
                {v.uploaded_by ? `${v.uploaded_by} · ` : ""}
                {timeAgo(v.created_at)}
              </span>
            </div>
            {v.title !== t.title && <div className="muted small">이 버전의 이름: {v.title}</div>}
            {v.description && <div className="muted small">올린 파일의 설명: {v.description}</div>}
            <VersionMemo
              v={v}
              isAdmin={isAdmin}
              onSaved={() => {
                void load();
                onSaved();
              }}
            />
          </div>
        ))}
      </div>
      <div className="modal-actions">
        <button type="button" onClick={onClose}>
          닫기
        </button>
      </div>
    </Modal>
  );
}

function Templates({ isAdmin, user }: { isAdmin: boolean; user: User }) {
  const { data: registryImages } = useRegistryImages(isAdmin);
  const [watching, setWatching] = useState(false);
  const { data: templates, refresh } = usePolling(() => api<Template[]>("/templates"), watching ? 3000 : 20000, [watching]);
  const preparing = templates?.some((t) => t.environment.preparing.length > 0 || t.check?.state === "checking") ?? false;
  useEffect(() => setWatching(preparing), [preparing]);
  const [message, setMessage] = useState<{ kind: "ok" | "error"; text: string } | null>(null);
  const [busy, setBusy] = useState(false);
  const [guide, setGuide] = useState(false);
  const [about, setAbout] = useState<Template | null>(null);
  const [versionsOf, setVersionsOf] = useState<Template | null>(null);
  const picker = useRef<HTMLInputElement>(null);

  const upload = async (file: File | undefined) => {
    if (!file) return;
    setBusy(true);
    setMessage(null);
    try {
      const added = await apiUpload<Template>("/templates", file);
      const next =
        added.check?.state === "pending"
          ? ' 아직 다른 사람에게는 보이지 않습니다. "시험 실행"으로 돌려 보고 통과하면 "허용"을 누르세요.'
          : "";
      setMessage({ kind: "ok", text: `${added.title} ${added.version} 버전을 등록했습니다.${next}` });
      await refresh();
    } catch (e) {
      setMessage({ kind: "error", text: e instanceof Error ? e.message : String(e) });
    } finally {
      setBusy(false);
      if (picker.current) picker.current.value = "";
    }
  };
  const [preparingFor, setPreparingFor] = useState<Template | null>(null);
  const prepare = async (t: Template, nodeIds: string[]) => {
    setMessage(null);
    try {
      await api(`/templates/${t.id}/prepare`, { body: { node_ids: nodeIds } });
      setMessage({ kind: "ok", text: `${t.title}의 환경을 고른 서버에 설치하기 시작했습니다. 첫 학습이 설치를 기다리지 않게 됩니다.` });
      setPreparingFor(null);
      setWatching(true);
      await refresh();
    } catch (e) {
      setMessage({ kind: "error", text: e instanceof Error ? e.message : String(e) });
      setPreparingFor(null);
    }
  };
  const removeEnv = async (t: Template, nodeId: string) => {
    setMessage(null);
    try {
      await api(`/templates/${t.id}/environment/${nodeId}`, { method: "DELETE" });
      setMessage({ kind: "ok", text: `${t.title}의 환경을 지우기 시작했습니다.` });
      setWatching(true);
      window.setTimeout(() => void refresh(), 4000); // the server reports what it has left after the job
    } catch (e) {
      setMessage({ kind: "error", text: e instanceof Error ? e.message : String(e) });
    }
  };
  const trial = async (t: Template) => {
    setMessage(null);
    try {
      await api(`/templates/${t.id}/check`, { method: "POST" });
      setMessage({ kind: "ok", text: `${t.title} ${t.version}을(를) 학습 서버에서 시험 실행합니다. 통과하면 "허용"을 누르세요.` });
      setWatching(true);
      await refresh();
    } catch (e) {
      setMessage({ kind: "error", text: e instanceof Error ? e.message : String(e) });
    }
  };
  const allow = async (t: Template) => {
    let reason = "";
    if (t.check.state === "passed") {
      if (!window.confirm(`시험 실행을 통과한 ${t.title} ${t.version}을(를) 모두의 학습 화면에 내보낼까요?`)) return;
    } else {
      reason =
        window.prompt(
          `${t.title} ${t.version}은(는) 시험 실행을 통과하지 않았습니다. 그래도 모두가 쓰게 할까요? 이유를 적어 주세요 (기록에 남습니다).`,
        ) ?? "";
      if (!reason.trim()) return;
    }
    setMessage(null);
    try {
      await api(`/templates/${t.id}/allow`, { body: { reason } });
      setMessage({ kind: "ok", text: `${t.title} ${t.version}을(를) 허용했습니다. 이제 모두의 학습 화면에 나옵니다.` });
      await refresh();
    } catch (e) {
      setMessage({ kind: "error", text: e instanceof Error ? e.message : String(e) });
    }
  };
  const remove = async (t: Template) => {
    if (!window.confirm(`${t.title} 모델 종류를 휴지통으로 옮길까요? 휴지통에서 되살릴 수 있고, 이것으로 만든 학습 기록과 모델은 남습니다.`))
      return;
    await api(`/templates/${t.id}`, { method: "DELETE" });
    announceTrash();
    await refresh();
  };

  return (
    <section className="card">
      <div className="card-head">
        <div>
          <h3>모델 종류</h3>
          <div className="muted small">학습·평가·추론 코드를 묶어 등록한 것입니다. 학습을 시작할 때 "모델 종류"로 나옵니다.</div>
        </div>
        {isAdmin && (
          <div className="head-actions">
            <button className="ghost-btn" onClick={() => setGuide(true)}>
              만드는 방법
            </button>
            <button onClick={() => picker.current?.click()} disabled={busy}>
              <Icon name="plus" />
              {busy ? "올리는 중…" : "올리기 (.zip)"}
            </button>
            <input ref={picker} type="file" accept=".zip" hidden onChange={(e) => void upload(e.target.files?.[0])} />
          </div>
        )}
      </div>
      {message && <div className={message.kind === "ok" ? "notice ok" : "error"}>{message.text}</div>}
      {!isAdmin && user.can_upload_templates && (
        <details className="kit-card">
          <summary>
            <strong>내 PC에서 만들어 올리기</strong>{" "}
            <span className="muted small">모델 종류를 올릴 수 있는 계정입니다. 올린 것은 초안으로 시작합니다.</span>
          </summary>
          <TemplateKit />
        </details>
      )}
      <div className="table-wrap">
        <table className="plain">
          <thead>
            <tr>
              <th>이름</th>
              <th>버전</th>
              <th>할 수 있는 것</th>
              <th>기본 환경</th>
              <th className="right">모델</th>
              {isAdmin && <th aria-label="동작" />}
            </tr>
          </thead>
          <tbody>
            {templates?.map((t) => (
              <tr key={t.id}>
                <td>
                  <button type="button" className="link template-name" onClick={() => setAbout(t)} title="설명 보기 · 고치기">
                    <strong>{t.title}</strong>
                  </button>{" "}
                  {t.builtin && <span className="tag">기본 제공</span>}
                  {t.draft && !isAdmin && <span className="tag">내 초안</span>}
                  <div className="muted small template-about">{t.description || t.name}</div>
                </td>
                <td className="num version-cell">
                  <button type="button" className="link" onClick={() => setVersionsOf(t)} title="버전마다 메모 · 시험 실행 보기">
                    {t.version}
                  </button>
                  {t.versions > 1 && <span className="muted small"> 외 {t.versions - 1}개</span>}
                  {t.memo && <div className="muted small template-about">{t.memo}</div>}
                  {(isAdmin || (t.draft && t.uploaded_by === user.username)) && !t.builtin && t.check && <TrialLine t={t} />}
                </td>
                <td>
                  {t.entrypoints.map((e) => (
                    <span key={e} className="tag">
                      {ENTRY_LABEL[e] ?? e}
                    </span>
                  ))}
                </td>
                <td className="env-cell">
                  {t.environment.default || (t.environment.kind === "docker" ? `Docker ${t.environment.image}` : "자동 설치")}
                  {t.environment.kind === "docker" && t.environment.image && <ImageInRegistry image={t.environment.image} rows={registryImages} />}
                  {t.environment.installs && (
                    <div className="muted small">
                      {t.environment.preparing.length > 0
                        ? `준비 중: ${t.environment.preparing.join(", ")}`
                        : t.environment.ready.length > 0
                          ? `준비됨: ${t.environment.ready.join(", ")}`
                          : "아직 설치한 서버 없음 (첫 학습 때 설치)"}
                    </div>
                  )}
                </td>
                <td className="num right">{t.models}</td>
                {isAdmin && (
                  <td className="right">
                    {t.environment.installs && (t.environment.servers ?? []).some((s) => !s.why) && (
                      <button
                        className="mini"
                        onClick={() => setPreparingFor(t)}
                        disabled={t.environment.preparing.length > 0}
                        title="고른 서버에 이 모델 종류의 학습 환경을 지금 설치하거나 지웁니다"
                      >
                        환경 관리
                      </button>
                    )}{" "}
                    {!t.builtin && t.check && t.check.state !== "checking" && (
                      <button
                        className={t.check.state === "pending" ? "mini primary" : "mini"}
                        onClick={() => void trial(t)}
                        title="학습 서버 한 대에서 작은 합성 데이터로 학습 → 테스트 · 평가 · 내보내기를 짧게 돌려 봅니다"
                      >
                        시험 실행
                      </button>
                    )}{" "}
                    {!t.builtin && t.check?.state === "passed" && (
                      <button className="mini primary" onClick={() => void allow(t)} title="통과한 이 버전을 모두의 학습 화면에 내보냅니다">
                        허용
                      </button>
                    )}{" "}
                    {!t.builtin && (t.check?.state === "failed" || t.check?.state === "pending") && (
                      <button className="mini" onClick={() => void allow(t)} title="시험 실행 통과 없이, 이유를 적고 모두가 쓰게 합니다">
                        통과 없이 허용
                      </button>
                    )}{" "}
                    {!t.builtin && (
                      <button className="mini" onClick={() => void remove(t)}>
                        지우기
                      </button>
                    )}
                  </td>
                )}
              </tr>
            ))}
          </tbody>
        </table>
      </div>
      {about && (
        <TemplateAbout
          t={templates?.find((x) => x.id === about.id) ?? about}
          isAdmin={isAdmin}
          onClose={() => setAbout(null)}
          onSaved={() => void refresh()}
        />
      )}
      {versionsOf && (
        <TemplateVersions
          t={templates?.find((x) => x.id === versionsOf.id) ?? versionsOf}
          isAdmin={isAdmin}
          onClose={() => setVersionsOf(null)}
          onSaved={() => void refresh()}
        />
      )}
      {preparingFor && (
        <PrepareDialog
          template={templates?.find((t) => t.id === preparingFor.id) ?? preparingFor}
          onStart={(ids) => prepare(preparingFor, ids)}
          onRemove={(nodeId) => removeEnv(preparingFor, nodeId)}
          onClose={() => setPreparingFor(null)}
        />
      )}
      {guide && (
        <Modal title="모델 종류 추가하기" onClose={() => setGuide(false)}>
          <p className="muted">
            사내 학습 코드를 폴더 하나로 묶고 선언 파일(badamoa.toml)을 더하면 됩니다. 폴더를 zip으로 묶어 이 화면의
            "올리기"로 등록해도 같습니다. 코드를 바꿔 다시 올릴 때는 선언 파일의 version을 올립니다.
          </p>
          <TemplateKit />
          <div className="modal-actions">
            <button onClick={() => setGuide(false)}>닫기</button>
          </div>
        </Modal>
      )}
    </section>
  );
}

// ---- in-house packages -----------------------------------------------------------------

interface HubPackage {
  id: string;
  name: string;
  version: string;
  filename: string;
  sha256: string;
  bytes: number;
  created_at: string;
}

/** Wheels built in-house: servers find them on the Hub while building a model kind's environment. */
function Packages({ isAdmin }: { isAdmin: boolean }) {
  const { data: packages, refresh } = usePolling(() => api<HubPackage[]>("/packages"), 30000);
  const [message, setMessage] = useState<{ kind: "ok" | "error"; text: string } | null>(null);
  const [busy, setBusy] = useState(false);
  const picker = useRef<HTMLInputElement>(null);
  const upload = async (files: FileList | null) => {
    if (!files?.length) return;
    setBusy(true);
    setMessage(null);
    try {
      for (const file of Array.from(files)) {
        await apiUpload<HubPackage>(`/packages?filename=${encodeURIComponent(file.name)}`, file, "application/octet-stream");
      }
      setMessage({ kind: "ok", text: `${files.length}개를 올렸습니다. 이제 서버가 환경을 만들 때 여기서도 찾습니다.` });
      await refresh();
    } catch (e) {
      setMessage({ kind: "error", text: e instanceof Error ? e.message : String(e) });
    } finally {
      setBusy(false);
      if (picker.current) picker.current.value = "";
    }
  };
  const remove = async (p: HubPackage) => {
    if (!window.confirm(`${p.filename}을(를) 목록에서 뺄까요? 이미 만든 환경에 설치된 것은 그대로입니다.`)) return;
    await api(`/packages/${p.id}`, { method: "DELETE" });
    await refresh();
  };
  return (
    <section className="card">
      <div className="card-head">
        <div>
          <h3>사내 패키지</h3>
          <div className="muted small">
            PyPI에 없는 사내 wheel(.whl). 모델 종류의 requirements에 이름을 적으면 서버가 환경을 만들 때 여기서 받아 설치합니다.
          </div>
        </div>
        {isAdmin && (
          <div className="head-actions">
            <button onClick={() => picker.current?.click()} disabled={busy}>
              <Icon name="plus" />
              {busy ? "올리는 중…" : "올리기 (.whl)"}
            </button>
            <input ref={picker} type="file" accept=".whl" multiple hidden onChange={(e) => void upload(e.target.files)} />
          </div>
        )}
      </div>
      {message && <div className={message.kind === "ok" ? "notice ok" : "error"}>{message.text}</div>}
      {packages && packages.length === 0 ? (
        <p className="muted small" style={{ padding: "0 16px 12px" }}>
          올린 패키지가 없습니다.
        </p>
      ) : (
        <div className="table-wrap">
          <table className="plain">
            <thead>
              <tr>
                <th>패키지</th>
                <th>버전</th>
                <th>파일</th>
                <th className="right">크기</th>
                <th>올린 때</th>
                {isAdmin && <th aria-label="동작" />}
              </tr>
            </thead>
            <tbody>
              {packages?.map((p) => (
                <tr key={p.id}>
                  <td>
                    <strong>{p.name}</strong>
                  </td>
                  <td className="num">{p.version}</td>
                  <td className="mono small" title={`sha256 ${p.sha256}`}>
                    {p.filename}
                  </td>
                  <td className="num right">{fileSize(p.bytes)}</td>
                  <td className="muted">{timeAgo(p.created_at)}</td>
                  {isAdmin && (
                    <td className="right">
                      <button className="mini" onClick={() => void remove(p)}>
                        지우기
                      </button>
                    </td>
                  )}
                </tr>
              ))}
            </tbody>
          </table>
        </div>
      )}
    </section>
  );
}

// ---- environments --------------------------------------------------------------------

function nodeState(node: EnvironmentNode): { kind: string; text: string } {
  if (node.ok === true) return { kind: "succeeded", text: "준비됨" };
  if (node.ok === false) return { kind: "failed", text: "사용 불가" };
  return { kind: "idle", text: "확인 전" };
}

function nodeDetail(node: EnvironmentNode): string {
  const d = node.detail;
  if (node.ok === false) return d.error ?? "확인에 실패했습니다.";
  if (node.ok !== true) return "";
  const packages = Object.entries(d.packages ?? {})
    .filter(([name]) => name !== "numpy")
    .map(([name, version]) => `${name} ${version}`);
  const gpu = d.cuda === true ? "GPU 사용 가능" : d.cuda === false ? "GPU 사용 불가 (CPU)" : "";
  return [`Python ${d.python}`, ...packages, gpu].filter(Boolean).join(" · ");
}

function Environments({ isAdmin, nodes }: { isAdmin: boolean; nodes: NodeInfo[] }) {
  const { data: registryImages } = useRegistryImages(isAdmin);
  const [checking, setChecking] = useState(false);
  const { data: environments, refresh } = usePolling(() => api<Environment[]>("/environments"), checking ? 2000 : 20000, [checking]);
  const [editing, setEditing] = useState<Environment | "new" | null>(null);
  const [error, setError] = useState<string | null>(null);

  const check = async (env: Environment) => {
    setError(null);
    try {
      await api(`/environments/${env.id}/check`, { method: "POST" });
      setChecking(true);
      setTimeout(() => setChecking(false), 20000); // results arrive as the servers answer
      await refresh();
    } catch (e) {
      setError(e instanceof Error ? e.message : String(e));
    }
  };
  const remove = async (env: Environment) => {
    if (!window.confirm(`'${env.name}' 환경을 지울까요? 서버에 있는 환경 자체는 그대로 둡니다.`)) return;
    await api(`/environments/${env.id}`, { method: "DELETE" });
    await refresh();
  };

  return (
    <section className="card">
      <div className="card-head">
        <div>
          <h3>환경</h3>
          <div className="muted small">학습 코드가 실행되는 곳입니다. 따로 등록하지 않으면 필요한 것을 서버가 자동으로 설치합니다.</div>
        </div>
        {isAdmin && (
          <button onClick={() => setEditing("new")}>
            <Icon name="plus" />
            환경 추가
          </button>
        )}
      </div>
      {error && <div className="error">{error}</div>}
      <div className="env-list">
        <div className="env">
          <div className="env-head">
            <strong>자동 설치</strong> <span className="tag">기본 제공</span>
          </div>
          <div className="muted small">
            모델 종류가 필요로 하는 패키지를 서버가 처음 학습할 때 설치합니다. GPU 드라이버에 맞는 PyTorch를 고릅니다.
          </div>
        </div>
        {environments?.map((env) => (
          <div className="env" key={env.id}>
            <div className="env-head">
              <strong>{env.name}</strong> <span className="tag">{env.kind === "docker" ? "Docker 이미지" : "서버에 있는 환경"}</span>
              {isAdmin && (
                <span className="env-actions">
                  <button className="mini" onClick={() => void check(env)}>
                    연결 확인
                  </button>
                  <button className="mini" onClick={() => setEditing(env)}>
                    수정
                  </button>
                  <button className="mini" onClick={() => void remove(env)}>
                    지우기
                  </button>
                </span>
              )}
            </div>
            {env.kind === "docker" && <div className="mono small">{env.image}</div>}
            {env.kind === "docker" && env.image && <ImageInRegistry image={env.image} rows={registryImages} />}
            {env.description && <div className="muted small">{env.description}</div>}
            {env.nodes.length === 0 && (
              <div className="muted small">
                {env.kind === "docker"
                  ? "아직 확인하지 않았습니다. 연결 확인을 누르면 Docker가 있는 학습 서버가 이미지를 받아 실행해 봅니다."
                  : "아직 어느 서버에도 연결하지 않았습니다."}
              </div>
            )}
            {env.nodes.map((node) => {
              const state = nodeState(node);
              return (
                <div className="env-node" key={node.node_id}>
                  <span className="env-node-name">{node.node_name}</span>
                  <Badge kind={state.kind}>{state.text}</Badge>
                  <span className="muted small">
                    {nodeDetail(node)}
                    {node.checked_at ? ` · ${timeAgo(node.checked_at)} 확인` : ""}
                  </span>
                </div>
              );
            })}
          </div>
        ))}
      </div>
      {editing && (
        <EnvironmentForm
          env={editing === "new" ? null : editing}
          nodes={nodes}
          onClose={() => setEditing(null)}
          onSaved={(saved) => {
            setEditing(null);
            void check(saved);
          }}
        />
      )}
    </section>
  );
}

function EnvironmentForm({
  env,
  nodes,
  onClose,
  onSaved,
}: {
  env: Environment | null;
  nodes: NodeInfo[];
  onClose: () => void;
  onSaved: (env: Environment) => void;
}) {
  const [name, setName] = useState(env?.name ?? "");
  const [kind, setKind] = useState(env?.kind ?? "python");
  const [image, setImage] = useState(env?.image ?? "");
  const [description, setDescription] = useState(env?.description ?? "");
  const [paths, setPaths] = useState<Record<string, string>>(() =>
    Object.fromEntries((env?.nodes ?? []).map((n) => [n.node_id, n.path ?? ""])),
  );
  const [error, setError] = useState<string | null>(null);

  const submit = async (e: FormEvent) => {
    e.preventDefault();
    try {
      setError(null);
      const body = { name, description, paths, kind, image };
      const saved = env
        ? await api<Environment>(`/environments/${env.id}`, { method: "PUT", body })
        : await api<Environment>("/environments", { body });
      onSaved(saved);
    } catch (err) {
      setError(err instanceof Error ? err.message : String(err));
    }
  };

  return (
    <Modal title={env ? "환경 수정" : "환경 추가"} onClose={onClose}>
      <form onSubmit={submit}>
        <div className="segmented" role="radiogroup" aria-label="환경의 방식">
          <span className="eyebrow">방식</span>
          {[
            ["python", "서버에 있는 Python"],
            ["docker", "Docker 이미지"],
          ].map(([value, label]) => (
            <button key={value} type="button" disabled={Boolean(env)} className={kind === value ? "active" : ""} onClick={() => setKind(value)}>
              {label}
            </button>
          ))}
        </div>
        <p className="muted">
          {kind === "docker"
            ? "레지스트리에 있는 학습 이미지를 이름으로 등록합니다. Docker가 있는 학습 서버가 이 이미지로 학습하고, 모델 종류의 코드와 데이터는 서버가 넣어 줍니다. 저장하면 서버들이 이미지를 받아 실행해 봅니다."
            : "서버에 이미 만들어 둔 Python 환경(conda, venv)을 이름으로 등록합니다. 저장하면 각 서버에서 실제로 실행되는지 바로 확인합니다."}
        </p>
        <div className="form-row">
          <label>
            이름
            <input value={name} onChange={(e) => setName(e.target.value)} required placeholder="예: 연구실 공용 환경" autoFocus />
          </label>
          <label>
            설명 (선택)
            <input value={description} onChange={(e) => setDescription(e.target.value)} placeholder="예: torch 2.5 · cu121" />
          </label>
        </div>
        {kind === "docker" && (
          <label>
            이미지
            <input
              className="mono"
              value={image}
              onChange={(e) => setImage(e.target.value)}
              required
              placeholder="예: registry.local:5000/team/train:1.2"
            />
          </label>
        )}
        {kind === "python" && <div className="eyebrow">서버별 위치 — 이 환경이 있는 서버에만 적습니다</div>}
        {kind === "python" && nodes.length === 0 && <p className="muted small">연결된 서버가 없습니다.</p>}
        {kind === "python" &&
          nodes.map((node) => (
          <label key={node.id} className="env-path">
            <span>{node.name}</span>
            <input
              className="mono"
              value={paths[node.id] ?? ""}
              onChange={(e) => setPaths({ ...paths, [node.id]: e.target.value })}
              placeholder={node.os === "windows" ? "C:\\envs\\lab\\python.exe 또는 환경 폴더" : "/opt/conda/envs/lab/bin/python 또는 환경 폴더"}
            />
          </label>
        ))}
        {error && <div className="error">{error}</div>}
        <div className="modal-actions">
          <button type="button" onClick={onClose}>
            취소
          </button>
          <button className="primary">저장하고 확인</button>
        </div>
      </form>
    </Modal>
  );
}

// ---- project defaults ----------------------------------------------------------------

function ProjectDefaults() {
  const { data: projects, refresh } = usePolling(() => api<Project[]>("/projects"), 30000);
  const { data: templates } = usePolling(() => api<Template[]>("/templates"), 30000);
  if (!projects) return null;
  if (projects.length === 0) return <section className="card empty">프로젝트가 아직 없습니다.</section>;

  const change = async (project: Project, template: string) => {
    await api(`/projects/${project.id}/settings`, { method: "PUT", body: { template } });
    await refresh();
  };

  return (
    <section className="card">
      <div className="card-head">
        <div>
          <h3>프로젝트 기본값</h3>
          <div className="muted small">학습 대화상자를 열었을 때 미리 골라져 있을 모델 종류입니다.</div>
        </div>
      </div>
      <div className="env-list">
        {projects.map((project) => (
          <label className="env-path" key={project.id}>
            <span>
              {project.name} <span className="muted small">버전 {project.releases}개</span>
            </span>
            <select value={project.settings?.template ?? ""} onChange={(e) => void change(project, e.target.value)}>
              <option value="">정하지 않음 ({templates?.[0]?.title ?? "기본 제공"})</option>
              {templates?.map((t) => (
                <option key={t.id} value={t.name}>
                  {t.title}
                </option>
              ))}
            </select>
          </label>
        ))}
      </div>
    </section>
  );
}
