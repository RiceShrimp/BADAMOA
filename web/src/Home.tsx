import { type ApiTokenInfo, type Job, type NodeInfo, type User, api } from "./api";
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

const ACTION_LABEL: Record<string, string> = {
  "setup.admin_created": "관리자 계정을 만들었습니다",
  "user.create": "사용자를 추가했습니다",
  "token.create": "에이전트 토큰을 발급했습니다",
  "node.join_token": "노드 추가 명령을 만들었습니다",
  "node.register": "노드가 등록되었습니다",
  "node.update": "노드 설정을 바꿨습니다",
  "node.delete": "노드를 제거했습니다",
  "job.create": "작업을 실행했습니다",
  "job.cancel": "작업을 취소했습니다",
};

/** The four BADA stages and what each will hold; mirrors the sidebar and the roadmap. */
const STAGES = [
  { key: "Build", icon: "database", title: "데이터", text: "가져오기 · 검색 · 큐레이션 · 버전", phase: "Phase 1–2" },
  { key: "Annotate", icon: "wand", title: "라벨", text: "오토라벨 · 검수", phase: "Phase 3" },
  { key: "Develop", icon: "chart", title: "모델", text: "학습 · 평가 · 레지스트리 · export", phase: "Phase 1, 4" },
  { key: "Automate", icon: "rocket", title: "배포", text: "재학습 · 배포 · 피드백", phase: "Phase 5" },
];

export function Home({ user, nodes }: { user: User; nodes: NodeInfo[] | null }) {
  const { data: jobs } = usePolling(() => api<Job[]>("/jobs?limit=50"), 5000);
  const { data: tokens } = usePolling(() => api<ApiTokenInfo[]>("/tokens"), 15000);
  const { data: activity } = usePolling(() => api<Activity[]>("/activity?limit=8"), 8000);

  const online = nodes?.filter((n) => n.status === "online") ?? [];
  const gpus = online.flatMap((n) => n.info.gpus);
  const vram = gpus.reduce((sum, g) => sum + (g.memory_total_mb ?? 0), 0);
  const running = jobs?.filter((j) => j.status === "running").length ?? 0;
  const queued = jobs?.filter((j) => j.status === "queued").length ?? 0;
  const failed = jobs?.filter((j) => j.status === "failed").length ?? 0;

  const steps = [
    {
      done: (nodes?.length ?? 0) > 1,
      title: "원격 서버를 노드로 추가",
      text: "학습·데이터 서버에서 한 줄 명령을 실행하면 연결됩니다.",
      href: "#/nodes",
    },
    {
      done: (tokens?.length ?? 0) > 0,
      title: "에이전트 연결",
      text: "Claude Code 같은 에이전트가 MCP로 Hub를 조작할 수 있습니다.",
      href: "#/agents",
    },
    {
      done: (jobs?.length ?? 0) > 0,
      title: "첫 작업 실행",
      text: "시스템 점검 작업으로 노드의 GPU와 드라이버를 확인합니다.",
      href: "#/jobs",
    },
  ];

  return (
    <>
      <div className="page-head">
        <div>
          <h1>안녕하세요, {user.username}님</h1>
          <p className="muted">데이터의 바다를 모아, 모델로. 지금 Hub의 상태입니다.</p>
        </div>
      </div>

      <div className="stats">
        <Stat
          icon="server"
          label="온라인 노드"
          value={
            <>
              {online.length}
              <small> / {nodes?.length ?? 0}</small>
            </>
          }
          sub={online.map((n) => n.os).filter(Boolean).join(" · ") || "연결된 노드 없음"}
        />
        <Stat icon="chip" label="GPU" value={gpus.length} sub={vram ? `VRAM 합계 ${gb(vram)}` : "GPU 없음"} />
        <Stat icon="activity" label="실행 중" value={running} sub={`대기 ${queued}`} />
        <Stat icon="alert" label="실패" value={failed} sub="최근 50개 작업 기준" />
      </div>

      <section className="card">
        <div className="card-head">
          <h3>파이프라인</h3>
          <span className="muted small">BADA 단계별로 기능이 채워집니다</span>
        </div>
        <div className="stages">
          {STAGES.map((s, i) => (
            <div className="stage" key={s.key}>
              <div className="stage-icon">
                <Icon name={s.icon} size={18} />
              </div>
              <div>
                <div className="eyebrow">
                  {String(i + 1).padStart(2, "0")} {s.key}
                </div>
                <strong>{s.title}</strong>
                <p>{s.text}</p>
                <span className="tag">{s.phase}</span>
              </div>
            </div>
          ))}
        </div>
      </section>

      <div className="two-col">
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
