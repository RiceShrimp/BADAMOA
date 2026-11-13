import { type FormEvent, useEffect, useState } from "react";
import { Agents } from "./Agents";
import { ApiError, type User, api, getToken, setToken } from "./api";
import { Jobs } from "./Jobs";
import { Nodes } from "./Nodes";

type Page = "nodes" | "jobs" | "agents";
const PAGES: { id: Page; label: string }[] = [
  { id: "nodes", label: "노드" },
  { id: "jobs", label: "작업" },
  { id: "agents", label: "에이전트 연결" },
];

function pageFromHash(): Page {
  const id = window.location.hash.replace("#/", "");
  return PAGES.some((p) => p.id === id) ? (id as Page) : "nodes";
}

export function App() {
  const [state, setState] = useState<"loading" | "setup" | "login" | "ready">("loading");
  const [user, setUser] = useState<User | null>(null);
  const [version, setVersion] = useState("");
  const [page, setPage] = useState<Page>(pageFromHash);

  useEffect(() => {
    const onHash = () => setPage(pageFromHash());
    window.addEventListener("hashchange", onHash);
    return () => window.removeEventListener("hashchange", onHash);
  }, []);

  useEffect(() => {
    void (async () => {
      try {
        const status = await api<{ needs_setup: boolean; version: string }>("/setup/status");
        setVersion(status.version);
        if (status.needs_setup) return setState("setup");
        if (!getToken()) return setState("login");
        setUser(await api<User>("/auth/me"));
        setState("ready");
      } catch (e) {
        if (e instanceof ApiError && e.status === 401) setToken(null);
        setState("login");
      }
    })();
  }, []);

  const onAuthed = (token: string, u: User) => {
    setToken(token);
    setUser(u);
    setState("ready");
  };

  const logout = () => {
    setToken(null);
    setUser(null);
    setState("login");
  };

  if (state === "loading") return <div className="center muted">불러오는 중…</div>;
  if (state === "setup" || state === "login") return <AuthForm mode={state} onAuthed={onAuthed} />;

  return (
    <div className="shell">
      <aside>
        <div className="brand">
          <strong>BADAMOA</strong>
          <span>바다모아</span>
        </div>
        <nav>
          {PAGES.map((p) => (
            <a key={p.id} href={`#/${p.id}`} className={page === p.id ? "active" : ""}>
              {p.label}
            </a>
          ))}
        </nav>
        <footer>
          <div>
            {user?.username} <span className="muted">· {user?.role === "admin" ? "관리자" : "멤버"}</span>
          </div>
          <button className="ghost" onClick={logout}>
            로그아웃
          </button>
          <div className="muted small">v{version}</div>
        </footer>
      </aside>
      <main>
        {page === "nodes" && <Nodes isAdmin={user?.role === "admin"} />}
        {page === "jobs" && <Jobs />}
        {page === "agents" && <Agents />}
      </main>
    </div>
  );
}

function AuthForm({ mode, onAuthed }: { mode: "setup" | "login"; onAuthed: (token: string, user: User) => void }) {
  const [username, setUsername] = useState(mode === "setup" ? "admin" : "");
  const [password, setPassword] = useState("");
  const [confirm, setConfirm] = useState("");
  const [error, setError] = useState<string | null>(null);
  const [busy, setBusy] = useState(false);

  const submit = async (e: FormEvent) => {
    e.preventDefault();
    if (mode === "setup" && password !== confirm) return setError("비밀번호가 서로 다릅니다.");
    setBusy(true);
    setError(null);
    try {
      const res = await api<{ token: string; user: User }>(mode === "setup" ? "/setup" : "/auth/login", {
        body: { username, password },
      });
      onAuthed(res.token, res.user);
    } catch (err) {
      setError(err instanceof Error ? err.message : String(err));
    } finally {
      setBusy(false);
    }
  };

  return (
    <div className="center">
      <form className="card auth" onSubmit={submit}>
        <div className="brand">
          <strong>BADAMOA</strong>
          <span>Build · Annotate · Develop · Automate</span>
        </div>
        <h1>{mode === "setup" ? "관리자 계정 만들기" : "로그인"}</h1>
        {mode === "setup" && <p className="muted">처음 실행하셨네요. 이 Hub를 관리할 계정을 만들어 주세요.</p>}
        <label>
          아이디
          <input value={username} onChange={(e) => setUsername(e.target.value)} autoFocus required />
        </label>
        <label>
          비밀번호
          <input type="password" value={password} onChange={(e) => setPassword(e.target.value)} required minLength={8} />
        </label>
        {mode === "setup" && (
          <label>
            비밀번호 확인
            <input type="password" value={confirm} onChange={(e) => setConfirm(e.target.value)} required />
          </label>
        )}
        {error && <div className="error">{error}</div>}
        <button className="primary" disabled={busy}>
          {mode === "setup" ? "시작하기" : "로그인"}
        </button>
      </form>
    </div>
  );
}
