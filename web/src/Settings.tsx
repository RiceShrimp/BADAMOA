import { type FormEvent, useRef, useState } from "react";
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
      <StorageUsage />
    </>
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
}

const PART_LABEL: [keyof Usage["parts"], string, string][] = [
  ["models", "모델", "등록된 모델의 가중치"],
  ["outputs", "산출물", "학습 · 평가 · 내보내기 · 테스트가 남긴 파일"],
  ["checkpoints", "체크포인트", "끝나지 않은 실행을 이어 가기 위한 것"],
  ["other", "그 밖", "썸네일, 모델 종류의 코드, 기록(DB)"],
];

function StorageUsage() {
  const { data: usage } = usePolling(() => api<Usage>("/storage"), 60000);
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
    </section>
  );
}

// ---- model templates -----------------------------------------------------------------

function Templates({ isAdmin }: { isAdmin: boolean }) {
  const { data: templates, refresh } = usePolling(() => api<Template[]>("/templates"), 20000);
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
                <td>{t.environment.default || "자동 설치"}</td>
                <td className="num right">{t.models}</td>
                {isAdmin && (
                  <td className="right">
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
