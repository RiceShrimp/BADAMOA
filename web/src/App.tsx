import { type FormEvent, useEffect, useState } from "react";
import { Agents } from "./Agents";
import { ApiError, type NodeInfo, type User, api, getToken, setToken } from "./api";
import { Home } from "./Home";
import { Jobs } from "./Jobs";
import { Nodes } from "./Nodes";
import { ReleaseNotes, hasUnseenRelease } from "./ReleaseNotes";
import { Icon, Logo, usePolling, useTheme } from "./ui";

type Page = "home" | "nodes" | "jobs" | "agents";

interface NavItem {
  label: string;
  icon: string;
  page?: Page; // set when the screen exists
  phase?: number; // otherwise: the roadmap phase that delivers it
}

/** The sidebar follows the product's own structure: the BADA pipeline stages, then MOA (operations). */
const NAV: { group?: string; hint?: string; items: NavItem[] }[] = [
  { items: [{ label: "홈", icon: "home", page: "home" }] },
  {
    group: "Build",
    hint: "데이터",
    items: [
      { label: "샘플 라이브러리", icon: "images", phase: 2 },
      { label: "데이터셋", icon: "database", phase: 1 },
    ],
  },
  {
    group: "Annotate",
    hint: "라벨",
    items: [
      { label: "오토라벨", icon: "wand", phase: 3 },
      { label: "검수", icon: "review", phase: 3 },
    ],
  },
  {
    group: "Develop",
    hint: "모델",
    items: [
      { label: "학습", icon: "chart", phase: 1 },
      { label: "모델", icon: "box", phase: 1 },
    ],
  },
  {
    group: "Automate",
    hint: "배포",
    items: [
      { label: "배포", icon: "rocket", phase: 4 },
      { label: "파이프라인", icon: "flow", phase: 5 },
    ],
  },
  {
    group: "MOA",
    hint: "운영",
    items: [
      { label: "노드", icon: "server", page: "nodes" },
      { label: "작업", icon: "activity", page: "jobs" },
      { label: "에이전트", icon: "bot", page: "agents" },
    ],
  },
];

const PAGES = NAV.flatMap((g) => g.items.filter((i) => i.page).map((i) => ({ ...i, group: g.group })));

function pageFromHash(): Page {
  const id = window.location.hash.replace("#/", "");
  return (PAGES.find((p) => p.page === id)?.page ?? "home") as Page;
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

  if (state === "loading") return <div className="center muted">불러오는 중…</div>;
  if (state === "setup" || state === "login" || !user) {
    return (
      <AuthForm
        mode={state === "setup" ? "setup" : "login"}
        onAuthed={(token, u) => {
          setToken(token);
          setUser(u);
          setState("ready");
        }}
      />
    );
  }

  const logout = () => {
    setToken(null);
    setUser(null);
    setState("login");
  };

  return <Shell user={user} version={version} page={page} onLogout={logout} />;
}

function Shell({ user, version, page, onLogout }: { user: User; version: string; page: Page; onLogout: () => void }) {
  const [theme, toggleTheme] = useTheme();
  const [notesOpen, setNotesOpen] = useState(false);
  const [unseen, setUnseen] = useState(() => hasUnseenRelease(version));
  const { data: nodes } = usePolling(() => api<NodeInfo[]>("/nodes"), 10000);
  const current = PAGES.find((p) => p.page === page);
  const online = nodes?.filter((n) => n.status === "online").length ?? 0;
  const isAdmin = user.role === "admin";

  return (
    <div className="shell">
      <aside className="sidebar">
        <a className="brand" href="#/home">
          <Logo />
          <span>
            <strong>BADAMOA</strong>
            <small>바다모아</small>
          </span>
        </a>
        <nav>
          {NAV.map((section, i) => (
            <div className="nav-group" key={section.group ?? i}>
              {section.group && (
                <div className="nav-label">
                  {section.group}
                  <span>{section.hint}</span>
                </div>
              )}
              {section.items.map((item) =>
                item.page ? (
                  <a key={item.label} href={`#/${item.page}`} className={`nav-item ${page === item.page ? "active" : ""}`}>
                    <Icon name={item.icon} />
                    {item.label}
                  </a>
                ) : (
                  <div
                    key={item.label}
                    className="nav-item soon"
                    title={`Phase ${item.phase}에서 추가됩니다`}
                    aria-disabled="true"
                  >
                    <Icon name={item.icon} />
                    {item.label}
                    <em>P{item.phase}</em>
                  </div>
                ),
              )}
            </div>
          ))}
        </nav>
        <div className="sidebar-foot">
          <div className="avatar">{user.username.slice(0, 1).toUpperCase()}</div>
          <div className="who">
            <strong>{user.username}</strong>
            <small>{isAdmin ? "관리자" : "멤버"}</small>
          </div>
          <button className="icon-btn" onClick={onLogout} aria-label="로그아웃" title="로그아웃">
            <Icon name="logout" />
          </button>
        </div>
      </aside>

      <div className="main">
        <header className="topbar">
          <div className="crumbs">
            {current?.group && <span>{current.group}</span>}
            <strong>{current?.label}</strong>
          </div>
          <div className="topbar-right">
            <a className={`pill ${online > 0 ? "ok" : ""}`} href="#/nodes">
              <i />
              노드 {online}/{nodes?.length ?? 0} 온라인
            </a>
            <button
              className={`pill plain ${unseen ? "new" : ""}`}
              onClick={() => {
                setNotesOpen(true);
                setUnseen(false);
              }}
              aria-label={`릴리즈 노트 (v${version})`}
              title="릴리즈 노트"
            >
              {unseen && <i />}v{version}
            </button>
            <button
              className="icon-btn"
              onClick={toggleTheme}
              aria-label="테마 전환"
              title={theme === "dark" ? "밝은 테마로" : "어두운 테마로"}
            >
              <Icon name={theme === "dark" ? "sun" : "moon"} />
            </button>
          </div>
        </header>
        <main className="content">
          {page === "home" && <Home user={user} nodes={nodes} />}
          {page === "nodes" && <Nodes isAdmin={isAdmin} />}
          {page === "jobs" && <Jobs />}
          {page === "agents" && <Agents />}
        </main>
        {notesOpen && <ReleaseNotes version={version} onClose={() => setNotesOpen(false)} />}
      </div>
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
    <div className="center auth-bg">
      <form className="card auth" onSubmit={submit}>
        <div className="brand">
          <Logo size={36} />
          <span>
            <strong>BADAMOA</strong>
            <small>Build · Annotate · Develop · Automate</small>
          </span>
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
