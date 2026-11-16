import { type ApiTokenInfo, type Job, type NodeInfo, type User, api } from "./api";
import { bytesLabel } from "./catalogTypes";
import { METRIC_NAME, ProjectCard, type ProjectCardInfo } from "./DataBoard";
import { Badge, Icon, STATUS_LABEL, Stat, VIA_LABEL, duration, gb, timeAgo, usePolling } from "./ui";

interface Activity {
  id: number;
  ts: string;
  username: string | null;
  via: string;
  action: string;
  target: string;
  detail: Record<string, unknown>;
}

interface HomeInfo {
  totals: Record<"projects" | "folders" | "datasets" | "releases" | "items" | "boxes" | "runs" | "models", number>;
  projects: ProjectCardInfo[];
  runs: {
    id: string;
    name: string;
    project: { id: string; name: string };
    status: string;
    epoch: number;
    epochs: number;
    metric: string;
    value: number | null;
    created_at: string;
  }[];
}

interface StorageInfo {
  bytes: number;
  free: number;
  disk: number;
}

const ACTION_LABEL: Record<string, string> = {
  "setup.admin_created": "관리자 계정을 만들었습니다",
  "user.create": "사용자를 추가했습니다",
  "user.update": "사용자 계정을 바꿨습니다",
  "user.password": "비밀번호를 바꿨습니다",
  "gpu.busy": "GPU 배정 기준을 바꿨습니다",
  "token.create": "에이전트 토큰을 발급했습니다",
  "node.join_token": "노드 추가 명령을 만들었습니다",
  "node.register": "노드가 등록되었습니다",
  "node.update": "노드 설정을 바꿨습니다",
  "node.delete": "노드를 제거했습니다",
  "job.create": "작업을 실행했습니다",
  "job.cancel": "작업을 취소했습니다",
  "project.create": "프로젝트를 만들었습니다",
  "project.update": "프로젝트를 바꿨습니다",
  "project.settings": "프로젝트 설정을 바꿨습니다",
  "project.delete": "프로젝트를 휴지통으로 옮겼습니다",
  "folder.create": "폴더를 만들었습니다",
  "folder.rename": "폴더 이름을 바꿨습니다",
  "folder.delete": "폴더를 지웠습니다",
  "project.restore": "프로젝트를 되살렸습니다",
  "items.names": "파일 이름에서 사이트 · 카메라를 채웠습니다",
  "release.create": "버전을 만들었습니다",
  "release.delete": "버전을 지웠습니다",
  "release.restore": "버전을 되살렸습니다",
  "release.export": "버전을 내려받았습니다",
  "run.update": "학습 정보를 바꿨습니다",
  "run.delete": "학습을 지웠습니다",
  "model.create": "모델을 등록했습니다",
  "model.update": "모델을 바꿨습니다",
  "model.delete": "모델을 지웠습니다",
  "sweep.create": "하이퍼파라미터 탐색을 시작했습니다",
  "sweep.stop": "탐색을 멈췄습니다",
  "environment.save": "실행 환경을 저장했습니다",
  "environment.prepare": "실행 환경을 미리 준비했습니다",
  "environment.delete": "실행 환경을 지웠습니다",
  "template.register": "학습 템플릿을 등록했습니다",
  "template.delete": "학습 템플릿을 지웠습니다",
  "storage.cleanup": "저장 공간을 정리했습니다",
  "trash.empty": "휴지통을 비웠습니다",
  "storage.policy": "보관 기간을 바꿨습니다",
};

const n = (value?: number) => (value ?? 0).toLocaleString();

export function Home({ user, nodes }: { user: User; nodes: NodeInfo[] | null }) {
  const { data: home } = usePolling(() => api<HomeInfo>("/home"), 15000);
  const { data: storage } = usePolling(() => api<StorageInfo>("/storage"), 60000);
  const { data: jobs } = usePolling(() => api<Job[]>("/jobs?limit=50"), 5000);
  const { data: tokens } = usePolling(() => api<ApiTokenInfo[]>("/tokens"), 30000);
  const { data: activity } = usePolling(() => api<Activity[]>("/activity?limit=8"), 8000);

  const totals = home?.totals;
  const online = nodes?.filter((node) => node.status === "online") ?? [];
  const gpus = online.flatMap((node) => node.info.gpus);
  const vram = gpus.reduce((sum, g) => sum + (g.memory_total_mb ?? 0), 0);
  const running = jobs?.filter((j) => j.status === "running").length ?? 0;
  const queued = jobs?.filter((j) => j.status === "queued").length ?? 0;
  const failed = jobs?.filter((j) => j.status === "failed").length ?? 0;

  // Shown until everything is done, then out of the way.
  const steps = [
    { done: (nodes?.length ?? 0) > 1, title: "원격 서버를 노드로 추가", text: "학습·데이터 서버에서 한 줄 명령을 실행하면 연결됩니다.", href: "#/nodes" },
    { done: (totals?.datasets ?? 0) > 0, title: "첫 데이터셋 가져오기", text: "노드에 있는 YOLO · COCO 데이터셋을 복사 없이 등록합니다.", href: "#/datasets" },
    { done: (totals?.runs ?? 0) > 0, title: "첫 학습", text: "데이터셋 버전을 골라 YOLO로 학습하고 모델을 남깁니다.", href: "#/training" },
    { done: (tokens?.length ?? 0) > 0, title: "에이전트 연결", text: "Claude Code 같은 에이전트가 MCP로 Hub를 조작할 수 있습니다.", href: "#/agents" },
  ];
  const showSteps = home != null && tokens != null && steps.some((s) => !s.done);

  return (
    <>
      <div className="page-head">
        <div>
          <h1>안녕하세요, {user.username}님</h1>
          <p className="muted">데이터의 바다를 모아, 모델로. 지금 Hub의 상태입니다.</p>
        </div>
        <div className="head-actions">
          <a className="btn" href="#/training">
            <Icon name="chart" size={14} />
            학습
          </a>
          <a className="btn primary-link" href="#/datasets">
            <Icon name="database" size={14} />
            데이터
          </a>
        </div>
      </div>

      <div className="stats six">
        <Stat icon="folder" label="프로젝트" value={n(totals?.projects)} sub={`폴더 ${n(totals?.folders)}개`} />
        <Stat icon="database" label="데이터셋" value={n(totals?.datasets)} sub={`버전 ${n(totals?.releases)}개`} />
        <Stat icon="images" label="이미지" value={n(totals?.items)} sub={`박스 ${n(totals?.boxes)}개`} />
        <Stat icon="box" label="모델" value={n(totals?.models)} sub={`학습 ${n(totals?.runs)}회`} />
        <Stat
          icon="server"
          label="온라인 노드"
          value={
            <>
              {online.length}
              <small> / {nodes?.length ?? 0}</small>
            </>
          }
          sub={gpus.length ? `GPU ${gpus.length}개 · VRAM ${gb(vram)}` : "GPU 없음"}
        />
        <Stat
          icon="disk"
          label="저장 공간"
          value={storage ? bytesLabel(storage.bytes) : "–"}
          sub={storage ? `남은 공간 ${bytesLabel(storage.free)}` : " "}
        />
      </div>

      {showSteps && (
        <section className="card">
          <div className="card-head">
            <h3>시작하기</h3>
            <span className="muted small">
              {steps.filter((s) => s.done).length} / {steps.length} 완료
            </span>
          </div>
          <ol className="steps">
            {steps.map((s) => (
              <li key={s.title} className={s.done ? "done" : ""}>
                <span className="step-mark">{s.done && <Icon name="check" size={14} />}</span>
                <div>
                  <strong>{s.title}</strong>
                  <p>{s.text}</p>
                </div>
                <a className="btn" href={s.href}>
                  {s.done ? "보기" : "시작"}
                  <Icon name="arrow" size={14} />
                </a>
              </li>
            ))}
          </ol>
        </section>
      )}

      <section className="home-block">
        <div className="card-head">
          <h3>최근 프로젝트</h3>
          <a className="link" href="#/datasets">
            모든 프로젝트 <Icon name="arrow" size={14} />
          </a>
        </div>
        {home && home.projects.length === 0 ? (
          <div className="card empty-state">
            <p className="muted">아직 프로젝트가 없습니다. 데이터 화면에서 프로젝트를 만들고 데이터셋을 가져오세요.</p>
          </div>
        ) : (
          <div className="card-grid">
            {home?.projects.map((card) => (
              <ProjectCard key={card.id} card={card} />
            ))}
          </div>
        )}
      </section>

      <div className="two-col">
        <section className="card table-wrap">
          <div className="card-head">
            <h3>최근 학습</h3>
            <a className="link" href="#/training">
              학습 화면 <Icon name="arrow" size={14} />
            </a>
          </div>
          <table>
            <tbody>
              {home?.runs.length === 0 && (
                <tr>
                  <td className="empty">아직 학습한 적이 없습니다.</td>
                </tr>
              )}
              {home?.runs.map((run) => (
                <tr key={run.id} className="clickable" onClick={() => (location.hash = `#/training/${run.id}`)}>
                  <td>
                    <strong>{run.name}</strong>
                    <div className="muted small">{run.project.name}</div>
                  </td>
                  <td>
                    <Badge kind={run.status}>{STATUS_LABEL[run.status] ?? run.status}</Badge>
                  </td>
                  <td className="num muted">
                    {run.epoch}/{run.epochs}
                  </td>
                  <td className="num">
                    {run.value != null ? (
                      <>
                        {run.value.toFixed(3)} <span className="muted small">{METRIC_NAME[run.metric] ?? run.metric}</span>
                      </>
                    ) : (
                      <span className="muted">–</span>
                    )}
                  </td>
                  <td className="muted right">{timeAgo(run.created_at)}</td>
                </tr>
              ))}
            </tbody>
          </table>
        </section>

        <section className="card">
          <div className="card-head">
            <h3>최근 활동</h3>
          </div>
          {activity?.length === 0 && <p className="muted">아직 활동이 없습니다.</p>}
          <ul className="feed">
            {activity?.map((a) => (
              <li key={a.id}>
                <span className="feed-dot" />
                <div>
                  <div>
                    <strong>{a.username ?? "시스템"}</strong> · {ACTION_LABEL[a.action] ?? a.action}
                    {typeof a.detail.type === "string" && <span className="mono muted"> {a.detail.type}</span>}
                    {typeof a.detail.name === "string" && <span className="muted"> {a.detail.name}</span>}
                    {typeof a.detail.version === "string" && <span className="muted"> {a.detail.version}</span>}
                  </div>
                  <div className="muted small">
                    {timeAgo(a.ts)} · {VIA_LABEL[a.via] ?? a.via}
                  </div>
                </div>
              </li>
            ))}
          </ul>
        </section>
      </div>

      <section className="card table-wrap">
        <div className="card-head">
          <h3>최근 작업</h3>
          <span className="muted small">
            실행 중 {running} · 대기 {queued} · 실패 {failed}
          </span>
          <a className="link" href="#/jobs">
            전체 보기 <Icon name="arrow" size={14} />
          </a>
        </div>
        <table>
          <tbody>
            {jobs?.length === 0 && (
              <tr>
                <td className="empty">아직 실행한 작업이 없습니다.</td>
              </tr>
            )}
            {jobs?.slice(0, 5).map((job) => (
              <tr key={job.id}>
                <td>
                  <strong>{job.title}</strong> <span className="mono muted small">{job.type}</span>
                </td>
                <td>
                  <Badge kind={job.status}>{STATUS_LABEL[job.status]}</Badge>
                </td>
                <td>{job.node_name ?? <span className="muted">배정 대기</span>}</td>
                <td className="muted">{duration(job.duration_s)}</td>
                <td className="muted right">{timeAgo(job.created_at)}</td>
              </tr>
            ))}
          </tbody>
        </table>
      </section>
    </>
  );
}
