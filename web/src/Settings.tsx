import { type FormEvent, useEffect, useRef, useState } from "react";
import { fileSize } from "./Files";
import { type NodeInfo, api, apiUpload } from "./api";
import type { Project } from "./catalogTypes";
import type { Environment, EnvironmentNode, Template } from "./trainTypes";
import { Badge, CopyBlock, Icon, Modal, timeAgo, usePolling } from "./ui";

const ENTRY_LABEL: Record<string, string> = { train: "학습", eval: "평가", predict: "추론", export: "내보내기" };

/**
 * What is set up once so that everyone else only picks by name: model templates,
 * environments, project defaults. Members can look; only admins change things.
 */
export function Settings({ isAdmin, nodes }: { isAdmin: boolean; nodes: NodeInfo[] | null }) {
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
      <Templates isAdmin={isAdmin} />
      <Environments isAdmin={isAdmin} nodes={nodes ?? []} />
      {isAdmin && <ProjectDefaults />}
      <StorageUsage isAdmin={isAdmin} />
      <ExternalMonitoring />
    </>
  );
}

// ---- for teams that already run Prometheus and Grafana ---------------------------------

function ExternalMonitoring() {
  const target = window.location.host;
  const scheme = window.location.protocol.replace(":", "");
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
    <details className="card fold">
      <summary>
        Prometheus · Grafana로 보내기 <span className="muted small">(사내에 이미 있다면)</span>
      </summary>
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
    </details>
  );
}

// ---- where everything is kept ----------------------------------------------------------

interface Usage {
  path: string | null; // admins only
  bytes: number;
  files: number;
  free: number;
  disk: number;
  parts: { models: number; outputs: number; checkpoints: number; other: number };
  policy: Policy;
  projects: { id: string; name: string; models: number; outputs: number; checkpoints: number; total: number }[];
}

interface Policy {
  outputs_days: number;
  checkpoint_days: number;
}

interface CleanupReport {
  policy: Policy;
  outputs: { files: number; bytes: number };
  checkpoints: { runs: number; bytes: number };
  removed: { files: number; bytes: number; partial_uploads: number };
}

const PART_LABEL: [keyof Usage["parts"], string, string][] = [
  ["models", "모델", "등록된 모델의 가중치"],
  ["outputs", "산출물", "학습 · 평가 · 내보내기 · 테스트가 남긴 파일"],
  ["checkpoints", "체크포인트", "끝나지 않은 실행을 이어 가기 위한 것"],
  ["other", "그 밖", "썸네일, 모델 종류의 코드, 기록(DB)"],
];

function StorageUsage({ isAdmin }: { isAdmin: boolean }) {
  const { data: usage, refresh } = usePolling(() => api<Usage>("/storage"), 60000);
  if (!usage) return null;
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
          <span className="muted small">파일 {usage.files.toLocaleString()}개</span>
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
                    {(["models", "outputs", "checkpoints", "total"] as const).map((k) => (
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
  const [draft, setDraft] = useState({ outputs: String(policy.outputs_days), checkpoints: String(policy.checkpoint_days) });
  const [preview, setPreview] = useState<CleanupReport | null>(null);
  const [message, setMessage] = useState<{ kind: "ok" | "error"; text: string } | null>(null);
  const [busy, setBusy] = useState(false);
  const changed = draft.outputs !== String(policy.outputs_days) || draft.checkpoints !== String(policy.checkpoint_days);

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
        body: { outputs_days: Number(draft.outputs) || 0, checkpoint_days: Number(draft.checkpoints) || 0 },
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
      <p className="muted small">
        0이면 계속 보관합니다. 모델(가중치), 지표, 로그는 기간과 상관없이 남습니다. alias가 붙은 모델의 실행과 "보관" 태그를 붙인
        실행의 산출물도 남습니다. 지운 실행 · 모델의 파일은 하루가 지나면 정리됩니다.
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

function Templates({ isAdmin }: { isAdmin: boolean }) {
  const [watching, setWatching] = useState(false);
  const { data: templates, refresh } = usePolling(() => api<Template[]>("/templates"), watching ? 3000 : 20000, [watching]);
  const preparing = templates?.some((t) => t.environment.preparing.length > 0) ?? false;
  useEffect(() => setWatching(preparing), [preparing]);
  const [message, setMessage] = useState<{ kind: "ok" | "error"; text: string } | null>(null);
  const [busy, setBusy] = useState(false);
  const [guide, setGuide] = useState(false);
  const picker = useRef<HTMLInputElement>(null);

  const upload = async (file: File | undefined) => {
    if (!file) return;
    setBusy(true);
    setMessage(null);
    try {
      const added = await apiUpload<Template>("/templates", file);
      setMessage({ kind: "ok", text: `${added.title} ${added.version} 버전을 등록했습니다.` });
      await refresh();
    } catch (e) {
      setMessage({ kind: "error", text: e instanceof Error ? e.message : String(e) });
    } finally {
      setBusy(false);
      if (picker.current) picker.current.value = "";
    }
  };
  const prepare = async (t: Template) => {
    setMessage(null);
    try {
      await api(`/templates/${t.id}/prepare`, { body: { node_ids: [] } });
      setMessage({ kind: "ok", text: `${t.title}의 환경을 서버에 설치하기 시작했습니다. 첫 학습이 설치를 기다리지 않게 됩니다.` });
      setWatching(true);
      await refresh();
    } catch (e) {
      setMessage({ kind: "error", text: e instanceof Error ? e.message : String(e) });
    }
  };
  const remove = async (t: Template) => {
    if (!window.confirm(`${t.title} 템플릿을 목록에서 지울까요? 이 템플릿으로 만든 학습 기록과 모델은 남습니다.`)) return;
    await api(`/templates/${t.id}`, { method: "DELETE" });
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
                  <strong>{t.title}</strong> {t.builtin && <span className="tag">기본 제공</span>}
                  <div className="muted small">{t.description || t.name}</div>
                </td>
                <td className="num">
                  {t.version}
                  {t.versions > 1 && <span className="muted small"> 외 {t.versions - 1}개</span>}
                </td>
                <td>
                  {t.entrypoints.map((e) => (
                    <span key={e} className="tag">
                      {ENTRY_LABEL[e] ?? e}
                    </span>
                  ))}
                </td>
                <td>
                  {t.environment.default || "자동 설치"}
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
                    {t.environment.installs && t.environment.ready.length < (t.environment.nodes ?? 1) && (
                      <button
                        className="mini"
                        onClick={() => void prepare(t)}
                        disabled={t.environment.preparing.length > 0}
                        title="모든 서버에 이 모델 종류의 학습 환경을 지금 설치합니다"
                      >
                        미리 준비
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
      {guide && (
        <Modal title="모델 종류 추가하기" onClose={() => setGuide(false)}>
          <p className="muted">
            사내 학습 코드를 폴더 하나로 묶고 선언 파일(badamoa.toml)을 더하면 됩니다. 아래 명령이 뼈대를 만들어 줍니다.
          </p>
          <CopyBlock label="1. 뼈대 만들기" text="badamoa template init my-model" />
          <CopyBlock label="2. train.py를 고친 뒤 확인" text="badamoa template check my-model" />
          <CopyBlock
            label="3. 올리기 (에이전트 연결 화면에서 발급한 관리자 토큰)"
            text={`badamoa template push my-model --hub ${window.location.origin} --token <토큰>`}
          />
          <p className="muted small">
            폴더를 zip으로 묶어 이 화면의 "올리기"로 등록해도 같습니다. 코드를 바꿔 다시 올릴 때는 선언 파일의 version을
            올립니다.
          </p>
          <div className="modal-actions">
            <button onClick={() => setGuide(false)}>닫기</button>
          </div>
        </Modal>
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
              <strong>{env.name}</strong> <span className="tag">서버에 있는 환경</span>
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
            {env.description && <div className="muted small">{env.description}</div>}
            {env.nodes.length === 0 && <div className="muted small">아직 어느 서버에도 연결하지 않았습니다.</div>}
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
  const [description, setDescription] = useState(env?.description ?? "");
  const [paths, setPaths] = useState<Record<string, string>>(() =>
    Object.fromEntries((env?.nodes ?? []).map((n) => [n.node_id, n.path ?? ""])),
  );
  const [error, setError] = useState<string | null>(null);

  const submit = async (e: FormEvent) => {
    e.preventDefault();
    try {
      setError(null);
      const body = { name, description, paths };
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
        <p className="muted">
          서버에 이미 만들어 둔 Python 환경(conda, venv)을 이름으로 등록합니다. 저장하면 각 서버에서 실제로 실행되는지 바로
          확인합니다.
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
        <div className="eyebrow">서버별 위치 — 이 환경이 있는 서버에만 적습니다</div>
        {nodes.length === 0 && <p className="muted small">연결된 서버가 없습니다.</p>}
        {nodes.map((node) => (
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
  if (!projects || projects.length === 0) return null;

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
              {project.name} <span className="muted small">릴리스 {project.releases}개</span>
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
