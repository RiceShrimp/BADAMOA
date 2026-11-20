import { type FormEvent, useEffect, useState } from "react";
import { type User, api } from "./api";
import { Badge, CopyBlock, Icon, Modal, timeAgo, usePolling } from "./ui";

// ---- people's accounts (admin) --------------------------------------------------------------

interface Account extends User {
  created_at: string;
  disabled: boolean;
  last_seen_at: string | null;
  api_tokens: number;
  online?: boolean; // used the Hub in the last three minutes
  ips?: string[]; // from where, when online
}

/** A password to hand over once: 14 letters and digits, without look-alikes (0/O, 1/l). */
function randomPassword(): string {
  const letters = "abcdefghjkmnpqrstuvwxyzABCDEFGHJKLMNPQRSTUVWXYZ23456789";
  const bytes = crypto.getRandomValues(new Uint8Array(14));
  return [...bytes].map((b) => letters[b % letters.length]).join("");
}

function PasswordField({ value, onChange, label }: { value: string; onChange: (v: string) => void; label: string }) {
  return (
    <label>
      {label}
      <div className="input-with-button">
        <input value={value} onChange={(e) => onChange(e.target.value)} minLength={8} required autoComplete="new-password" className="mono" />
        <button type="button" className="small" onClick={() => onChange(randomPassword())}>
          무작위로
        </button>
      </div>
      <span className="muted small">8자 이상. 만든 뒤 그 사람에게 직접 전하세요. 이 화면에는 다시 나오지 않습니다.</span>
    </label>
  );
}

export function UsersPanel({ me }: { me: User }) {
  const { data: users, error, refresh } = usePolling(() => api<Account[]>("/users"), 30000);
  const [adding, setAdding] = useState(false);
  const [resetting, setResetting] = useState<Account | null>(null);
  const [failed, setFailed] = useState<string | null>(null);

  const change = async (user: Account, body: Partial<{ role: string; disabled: boolean; template_upload: boolean }>) => {
    try {
      setFailed(null);
      await api(`/users/${user.id}`, { method: "PATCH", body });
      await refresh();
    } catch (e) {
      setFailed(e instanceof Error ? e.message : String(e));
    }
  };
  const stop = (user: Account) => {
    const text = `${user.username} 계정을 중지할까요?\n로그인할 수 없고 API 토큰도 멈춥니다. 그 사람이 만든 데이터 · 학습 · 모델은 그대로 남고, 다시 사용하게 할 수 있습니다.`;
    if (window.confirm(text)) void change(user, { disabled: true });
  };

  return (
    <section className="card">
      <div className="card-head">
        <div>
          <h3>사용자</h3>
          <div className="muted small">관리자는 설정 · 노드 · 사용자 · 삭제까지 다루고, 멤버는 데이터를 올리고 학습 · 평가를 합니다.</div>
        </div>
        <button className="primary" onClick={() => setAdding(true)}>
          <Icon name="plus" />
          사용자 추가
        </button>
      </div>
      {error && !users && <div className="error">{error}</div>}
      {!error && !users && <p className="muted">불러오는 중…</p>}
      {failed && <div className="error">{failed}</div>}
      {users && (
        <div className="table-wrap">
          <table className="plain">
            <thead>
              <tr>
                <th>아이디</th>
                <th>역할</th>
                <th title="멤버가 모델 종류(학습 코드)를 초안으로 올릴 수 있게 합니다. 관리자가 시험 실행하고 허용하면 모두에게 보입니다">
                  모델 종류 올리기
                </th>
                <th>상태</th>
                <th>마지막 사용</th>
                <th className="right">API 토큰</th>
                <th>만든 때</th>
                <th />
              </tr>
            </thead>
            <tbody>
              {users.map((u) => (
                <tr key={u.id} className={u.disabled ? "muted-row" : ""}>
                  <td>
                    <strong>{u.username}</strong>
                    {u.id === me.id && <span className="tag">나</span>}
                  </td>
                  <td>
                    <select value={u.role} onChange={(e) => void change(u, { role: e.target.value })} aria-label={`${u.username}의 역할`}>
                      <option value="member">멤버</option>
                      <option value="admin">관리자</option>
                    </select>
                  </td>
                  <td>
                    {u.role === "admin" ? (
                      <span className="muted small">관리자는 항상</span>
                    ) : (
                      <label className="inline-check">
                        <input
                          type="checkbox"
                          checked={!!u.template_upload}
                          onChange={(e) => void change(u, { template_upload: e.target.checked })}
                          aria-label={`${u.username}의 모델 종류 올리기`}
                        />
                        초안으로
                      </label>
                    )}
                  </td>
                  <td>{u.disabled ? <Badge kind="disabled">중지됨</Badge> : <Badge kind="online">사용 중</Badge>}</td>
                  <td>
                    {u.online ? (
                      <span title={u.ips?.length ? `접속한 곳: ${u.ips.join(", ")}` : undefined}>
                        <Badge kind="online">접속 중</Badge>
                        {u.ips?.length ? <span className="muted small"> {u.ips.join(", ")}</span> : null}
                      </span>
                    ) : u.last_seen_at ? (
                      timeAgo(u.last_seen_at)
                    ) : (
                      <span className="muted">아직 없음</span>
                    )}
                  </td>
                  <td className="num right">{u.api_tokens}</td>
                  <td>{u.created_at.slice(0, 10)}</td>
                  <td className="right row-buttons">
                    <button className="small" onClick={() => setResetting(u)}>
                      비밀번호 재설정
                    </button>
                    {u.disabled ? (
                      <button className="small" onClick={() => void change(u, { disabled: false })}>
                        다시 사용
                      </button>
                    ) : (
                      u.id !== me.id && (
                        <button className="small danger" onClick={() => stop(u)}>
                          중지
                        </button>
                      )
                    )}
                  </td>
                </tr>
              ))}
            </tbody>
          </table>
        </div>
      )}
      <p className="muted small">계정은 지우지 않고 중지합니다. 누가 무엇을 했는지 기록이 남아 있어야 하기 때문입니다.</p>
      {adding && (
        <NewAccount
          onClose={(made) => {
            setAdding(false);
            if (made) void refresh();
          }}
        />
      )}
      {resetting && (
        <ResetPassword
          user={resetting}
          onClose={(done) => {
            setResetting(null);
            if (done) void refresh();
          }}
        />
      )}
    </section>
  );
}

function NewAccount({ onClose }: { onClose: (made: boolean) => void }) {
  const [username, setUsername] = useState("");
  const [password, setPassword] = useState(randomPassword);
  const [role, setRole] = useState("member");
  const [error, setError] = useState<string | null>(null);
  const submit = async (e: FormEvent) => {
    e.preventDefault();
    try {
      await api("/users", { body: { username: username.trim(), password, role } });
      onClose(true);
    } catch (err) {
      setError(err instanceof Error ? err.message : String(err));
    }
  };
  return (
    <Modal title="사용자 추가" onClose={() => onClose(false)}>
      <form onSubmit={submit}>
        <label>
          아이디
          <input value={username} onChange={(e) => setUsername(e.target.value)} required autoFocus placeholder="영문 · 숫자 · . _ - 3~64자" />
        </label>
        <PasswordField label="처음 비밀번호" value={password} onChange={setPassword} />
        <label>
          역할
          <select value={role} onChange={(e) => setRole(e.target.value)}>
            <option value="member">멤버 — 데이터 · 학습 · 모델</option>
            <option value="admin">관리자 — 설정 · 노드 · 사용자까지</option>
          </select>
        </label>
        {error && <div className="error">{error}</div>}
        <div className="modal-actions">
          <button type="button" onClick={() => onClose(false)}>
            취소
          </button>
          <button className="primary">추가</button>
        </div>
      </form>
    </Modal>
  );
}

function ResetPassword({ user, onClose }: { user: Account; onClose: (done: boolean) => void }) {
  const [password, setPassword] = useState(randomPassword);
  const [error, setError] = useState<string | null>(null);
  const submit = async (e: FormEvent) => {
    e.preventDefault();
    try {
      await api(`/users/${user.id}`, { method: "PATCH", body: { password } });
      onClose(true);
    } catch (err) {
      setError(err instanceof Error ? err.message : String(err));
    }
  };
  return (
    <Modal title={`${user.username} 비밀번호 재설정`} onClose={() => onClose(false)}>
      <form onSubmit={submit}>
        <PasswordField label="새 비밀번호" value={password} onChange={setPassword} />
        <p className="muted small">그 사람의 브라우저 로그인은 모두 끝납니다. 에이전트용 API 토큰은 그대로 동작합니다.</p>
        {error && <div className="error">{error}</div>}
        <div className="modal-actions">
          <button type="button" onClick={() => onClose(false)}>
            취소
          </button>
          <button className="primary">바꾸기</button>
        </div>
      </form>
    </Modal>
  );
}

// ---- your own account --------------------------------------------------------------------------

export function MyAccount({ me }: { me: User }) {
  const [current, setCurrent] = useState("");
  const [next, setNext] = useState("");
  const [again, setAgain] = useState("");
  const [error, setError] = useState<string | null>(null);
  const [done, setDone] = useState(false);
  const submit = async (e: FormEvent) => {
    e.preventDefault();
    setDone(false);
    if (next !== again) return setError("새 비밀번호 두 칸이 다릅니다.");
    try {
      setError(null);
      await api("/auth/password", { body: { current, new: next } });
      setCurrent("");
      setNext("");
      setAgain("");
      setDone(true);
    } catch (err) {
      setError(err instanceof Error ? err.message : String(err));
    }
  };
  return (
    <section className="card narrow-card">
      <div className="card-head">
        <div>
          <h3>내 계정</h3>
          <div className="muted small">
            {me.username} · {me.role === "admin" ? "관리자" : "멤버"} · 에이전트용 토큰은 <a href="#/agents">에이전트 연결</a>에서
          </div>
        </div>
      </div>
      <form onSubmit={submit}>
        <label>
          지금 비밀번호
          <input type="password" value={current} onChange={(e) => setCurrent(e.target.value)} required autoComplete="current-password" />
        </label>
        <div className="form-row">
          <label>
            새 비밀번호
            <input type="password" value={next} onChange={(e) => setNext(e.target.value)} minLength={8} required autoComplete="new-password" />
          </label>
          <label>
            새 비밀번호 확인
            <input type="password" value={again} onChange={(e) => setAgain(e.target.value)} minLength={8} required autoComplete="new-password" />
          </label>
        </div>
        <p className="muted small">바꾸면 다른 곳의 로그인은 끝나고 지금 이 창은 그대로입니다.</p>
        {error && <div className="error">{error}</div>}
        {done && <div className="notice">비밀번호를 바꿨습니다.</div>}
        <div className="modal-actions">
          <button className="primary">비밀번호 바꾸기</button>
        </div>
      </form>
    </section>
  );
}

// ---- when a GPU counts as busy ----------------------------------------------------------------

interface GpuLoad {
  node: string;
  index: number;
  name: string;
  memory_pct: number;
  util_pct: number;
  state: string;
  reserved: boolean;
  users: string[];
}

interface GpuBusy {
  memory_pct: number;
  util_pct: number;
  source: "screen" | "launch";
  launch: { memory_pct: number; util_pct: number };
  gpus: GpuLoad[];
}

const STATE_TEXT: Record<string, [string, string]> = {
  free: ["online", "비어 있음"],
  job: ["running", "BADAMOA 작업"],
  busy: ["failed", "누가 쓰는 중"],
  reserved: ["queued", "사람 전용"],
};
const OFF = 101; // "never busy": the Hub takes 101 % as a level no GPU reaches

/** The state a GPU would have under other thresholds: jobs stay jobs, the rest is judged by its load now. */
function wouldBe(gpu: GpuLoad, memory: number, util: number): string {
  if (gpu.state === "job") return "job";
  if (gpu.memory_pct >= memory || gpu.util_pct >= util) return "busy";
  return gpu.reserved ? "reserved" : "free";
}

export function GpuBusyPanel({ isAdmin }: { isAdmin: boolean }) {
  const { data, refresh } = usePolling(() => api<GpuBusy>("/settings/gpu-busy"), 10000);
  const [memory, setMemory] = useState<number | null>(null);
  const [util, setUtil] = useState<number | null>(null);
  const [error, setError] = useState<string | null>(null);
  const [saved, setSaved] = useState(false);

  // The form starts from the values in force, and follows them until someone edits it.
  useEffect(() => {
    if (data && memory === null) setMemory(data.memory_pct);
    if (data && util === null) setUtil(data.util_pct);
  }, [data, memory, util]);
  if (!data || memory === null || util === null) return <section className="card muted">불러오는 중…</section>;

  const off = memory >= OFF && util >= OFF;
  const dirty = memory !== data.memory_pct || util !== data.util_pct;
  const save = async (body: { memory_pct?: number; util_pct?: number }) => {
    try {
      setError(null);
      const view = await api<GpuBusy>("/settings/gpu-busy", { method: "PUT", body });
      setMemory(view.memory_pct);
      setUtil(view.util_pct);
      setSaved(true);
      await refresh();
    } catch (e) {
      setError(e instanceof Error ? e.message : String(e));
    }
  };
  const submit = (e: FormEvent) => {
    e.preventDefault();
    void save({ memory_pct: memory, util_pct: util });
  };
  const pct = (v: number) => (v >= OFF ? "보지 않음" : `${v}%`);

  return (
    <section className="card">
      <div className="card-head">
        <div>
          <h3>GPU 배정 기준</h3>
          <div className="muted small">
            BADAMOA가 내주지 않은 GPU라도 메모리나 사용률이 이 값을 넘으면 누가 직접 쓰는 것으로 보고 작업을 보내지 않습니다. 공유 서버에서
            사람의 작업과 부딪히지 않게 하는 기준입니다.
          </div>
        </div>
      </div>
      <p className="small">
        지금: 메모리 <strong>{pct(data.memory_pct)}</strong> · 사용률 <strong>{pct(data.util_pct)}</strong> —{" "}
        {data.source === "screen" ? "이 화면에서 정한 값" : "Hub 실행 옵션의 값"}
        {data.source === "screen" && (
          <span className="muted">
            {" "}
            (실행 옵션: 메모리 {pct(data.launch.memory_pct)} · 사용률 {pct(data.launch.util_pct)})
          </span>
        )}
      </p>
      {isAdmin && (
        <form onSubmit={submit} className="gpu-busy-form">
          <div className="form-row">
            <label>
              메모리 사용이 넘으면 (%)
              <input
                type="number"
                min={1}
                max={100}
                value={off ? "" : Math.min(memory, 100)}
                disabled={off}
                onChange={(e) => (setMemory(Number(e.target.value)), setSaved(false))}
                required={!off}
              />
            </label>
            <label>
              사용률이 넘으면 (%)
              <input
                type="number"
                min={1}
                max={100}
                value={off ? "" : Math.min(util, 100)}
                disabled={off}
                onChange={(e) => (setUtil(Number(e.target.value)), setSaved(false))}
                required={!off}
              />
            </label>
          </div>
          <label className="check">
            <input
              type="checkbox"
              checked={off}
              onChange={(e) => {
                setSaved(false);
                if (e.target.checked) (setMemory(OFF), setUtil(OFF));
                else (setMemory(data.launch.memory_pct < OFF ? data.launch.memory_pct : 10), setUtil(data.launch.util_pct < OFF ? data.launch.util_pct : 50));
              }}
            />
            보지 않음 — 누가 써도 비어 있는 것으로 봄 (BADAMOA만 쓰는 서버)
          </label>
          {error && <div className="error">{error}</div>}
          {saved && !dirty && <div className="notice">저장했습니다. 다음 배정부터 이 기준을 씁니다.</div>}
          <div className="modal-actions">
            {data.source === "screen" && (
              <button type="button" className="push-left" onClick={() => void save({})}>
                실행 옵션 값으로
              </button>
            )}
            <button className="primary" disabled={!dirty}>
              저장
            </button>
          </div>
        </form>
      )}
      {data.gpus.length === 0 ? (
        <p className="muted small">온라인 노드에 GPU가 없습니다.</p>
      ) : (
        <div className="table-wrap">
          <table className="plain">
            <thead>
              <tr>
                <th>GPU</th>
                <th className="right">메모리</th>
                <th className="right">사용률</th>
                <th>지금</th>
                {isAdmin && dirty && <th>이 값이면</th>}
              </tr>
            </thead>
            <tbody>
              {data.gpus.map((g) => {
                const [kind, text] = STATE_TEXT[g.state] ?? ["idle", g.state];
                const then = wouldBe(g, memory, util);
                const [thenKind, thenText] = STATE_TEXT[then] ?? ["idle", then];
                return (
                  <tr key={`${g.node}-${g.index}`}>
                    <td>
                      {g.node} · GPU {g.index} <span className="muted small">{g.name}</span>
                      {g.users.length > 0 && <span className="muted small"> · {g.users.join(", ")}</span>}
                    </td>
                    <td className="num right">{g.memory_pct.toFixed(1)}%</td>
                    <td className="num right">{g.util_pct}%</td>
                    <td>
                      <Badge kind={kind}>{text}</Badge>
                    </td>
                    {isAdmin && dirty && (
                      <td>
                        {then === g.state ? <span className="muted small">같음</span> : <Badge kind={thenKind}>{thenText}</Badge>}
                      </td>
                    )}
                  </tr>
                );
              })}
            </tbody>
          </table>
        </div>
      )}
      <p className="muted small">
        GPU 하나를 사람에게만 남겨 두려면 <a href="#/nodes">노드</a> 화면의 GPU 자리표에서 "사람 전용"으로 지정하세요.
      </p>
    </section>
  );
}

interface HubAutostart {
  supported: boolean;
  on: boolean;
  when: "" | "boot" | "logon";
  can_boot: boolean;
  admin_command: string;
  name: string;
  log: string;
  launch: string | null;
}

/** Whether the Hub comes back by itself when its PC restarts (admins). */
export function HubAutostartPanel() {
  const { data, refresh } = usePolling(() => api<HubAutostart>("/settings/hub-autostart"), 30000);
  const [busy, setBusy] = useState(false);
  const [error, setError] = useState<string | null>(null);
  if (!data) return <section className="card muted">불러오는 중…</section>;

  const turn = async (on: boolean) => {
    setBusy(true);
    try {
      setError(null);
      await api<HubAutostart>("/settings/hub-autostart", { method: "PUT", body: { on } });
      await refresh();
    } catch (e) {
      setError(e instanceof Error ? e.message : String(e));
    } finally {
      setBusy(false);
    }
  };
  const state = !data.on
    ? "꺼짐 — 이 컴퓨터를 다시 켜면 Hub를 직접 실행해야 합니다"
    : data.when === "boot"
      ? "켜짐 — 이 컴퓨터가 켜지면 로그인하지 않아도 실행됩니다"
      : "켜짐 — Hub를 실행한 계정이 이 컴퓨터에 로그인하면 실행됩니다";

  return (
    <section className="card">
      <div className="card-head">
        <div>
          <h3>재부팅 시 자동 실행</h3>
          <div className="muted small">
            Hub가 도는 컴퓨터(Windows · Ubuntu)가 다시 켜지면 Hub를 지금과 같은 설정(데이터 폴더 · 포트 · 주소)으로 실행합니다. 노드는 Hub가
            켜지면 알아서 다시 붙습니다.
          </div>
        </div>
        {data.supported && (
          <button className={data.on ? "" : "primary"} onClick={() => void turn(!data.on)} disabled={busy}>
            {data.on ? "끄기" : "켜기"}
          </button>
        )}
      </div>
      <p className="small">
        <strong style={data.on ? { color: "var(--ok)" } : undefined}>{state}</strong>
      </p>
      {!data.supported && <p className="muted small">이 OS에서는 자동 실행을 등록하지 못합니다.</p>}
      {data.on && data.admin_command && (
        <>
          <p className="small">
            로그인하지 않아도 부팅 때 실행되게 하려면, 이 컴퓨터의 <strong>관리자 PowerShell</strong>에서 한 번 실행하세요 (지금 Hub는 관리자
            권한 없이 돌고 있습니다).
          </p>
          <CopyBlock text={data.admin_command} />
        </>
      )}
      {error && <div className="error">{error}</div>}
      {data.on && (
        <div className="muted small">
          {data.name} · 로그 {data.log}
          {data.launch && <div title="자동 실행이 쓰는 명령">{data.launch}</div>}
        </div>
      )}
    </section>
  );
}
