import { ImportTray } from "./Imports";
import { type FormEvent, useEffect, useState } from "react";
import { Agents } from "./Agents";
import { ApiError, type NodeInfo, type User, api, getToken, setToken } from "./api";
import { Datasets } from "./Datasets";
import { TRASH_EVENT } from "./catalogTypes";
import { Home } from "./Home";
import { Jobs } from "./Jobs";
import { Labeling } from "./Labeling";
import { Review } from "./Review";
import { Models } from "./Models";
import { NodeDetail } from "./NodeDetail";
import { Nodes } from "./Nodes";
import { GuidesPage } from "./Guides";
import { ReleasesPage, hasUnseenRelease } from "./ReleaseNotes";
import { Settings } from "./Settings";
import { Training } from "./Training";
import { TrashPage } from "./Trash";
import { Icon, Logo, usePolling, useTheme } from "./ui";

type Page = "home" | "datasets" | "labeling" | "review" | "training" | "models" | "nodes" | "jobs" | "agents" | "settings" | "trash" | "releases" | "guides";

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
      { label: "데이터셋", icon: "database", page: "datasets" },
    ],
  },
  {
    group: "Annotate",
    hint: "라벨",
    items: [
      { label: "라벨링", icon: "edit", page: "labeling" }, // labeling sets, with their review
      { label: "검수", icon: "review", page: "review" }, // finding missing and wrong labels
      { label: "오토라벨", icon: "wand", phase: 3 },
    ],
  },
  {
    group: "Develop",
    hint: "모델",
    items: [
      { label: "학습", icon: "chart", page: "training" },
      { label: "모델", icon: "box", page: "models" },
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
      { label: "설정", icon: "settings", page: "settings" },
    ],
  },
  // Whatever was deleted (data, runs, models, templates), one click away from every screen.
  { items: [{ label: "휴지통", icon: "trash", page: "trash" }] },
];

const PAGES = NAV.flatMap((g) => g.items.filter((i) => i.page).map((i) => ({ ...i, group: g.group })));

function pageFromHash(): Page {
  if (window.location.hash.startsWith("#/datasets/trash")) window.history.replaceState(null, "", "#/trash"); // its old place
  const id = window.location.hash.replace("#/", "").split("/")[0];
  if (id === "releases") return "releases"; // reached from the version in the top bar, not the sidebar
  if (id === "guides") return "guides"; // reached from 설명서 in the top bar
  return (PAGES.find((p) => p.page === id)?.page ?? "home") as Page;
}

export function App() {
  const [state, setState] = useState<"loading" | "setup" | "login" | "ready">("loading");
  const [user, setUser] = useState<User | null>(null);
  const [version, setVersion] = useState("");
  const [page, setPage] = useState<Page>(pageFromHash);
  const [hash, setHash] = useState(window.location.hash);

  useEffect(() => {
    const onHash = () => {
      setPage(pageFromHash());
      setHash(window.location.hash);
    };
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

  const [, , sub, arg] = hash.split("/"); // #/<page>/<id>/<extra>
  return <Shell user={user} version={version} page={page} sub={sub || undefined} arg={arg || undefined} onLogout={logout} />;
}

function Shell({
  user,
  version,
  page,
  sub,
  arg,
  onLogout,
}: {
  user: User;
  version: string;
  page: Page;
  sub?: string;
  arg?: string;
  onLogout: () => void;
}) {
  const [theme, toggleTheme] = useTheme();
  const { data: nodes } = usePolling(() => api<NodeInfo[]>("/nodes"), 10000);
  const { data: binned, refresh: recount } = usePolling(() => api<{ total: number }>("/trash/count"), 30000);
  useEffect(() => {
    const again = () => void recount();
    window.addEventListener(TRASH_EVENT, again);
    return () => window.removeEventListener(TRASH_EVENT, again);
  }, [recount]);
  // The Hub was updated while this page stayed open: its screens are the old ones until reloaded.
  const { data: running, refresh: recheck } = usePolling(() => api<{ version: string }>("/setup/status"), 60000);
  useEffect(() => {
    const back = () => document.visibilityState === "visible" && void recheck(); // coming back to the tab
    document.addEventListener("visibilitychange", back);
    return () => document.removeEventListener("visibilitychange", back);
  }, [recheck]);
  // The version this page was loaded with; when that first question went unanswered (the Hub was restarting
  // just then), the one the Hub gives on its next check, instead of a bare "v".
  const loaded = version || running?.version || "";
  const [unseen, setUnseen] = useState(() => hasUnseenRelease(loaded));
  useEffect(() => {
    if (loaded) setUnseen(page !== "releases" && hasUnseenRelease(loaded));
  }, [loaded, page]);
  const updated = running && loaded && running.version !== loaded ? running.version : null;
  const current =
    page === "releases"
      ? { group: undefined, label: "릴리즈" }
      : page === "guides"
        ? { group: undefined, label: "설명서" }
        : PAGES.find((p) => p.page === page);
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
                  <a
                    key={item.label}
                    href={`#/${item.page}`}
                    className={`nav-item ${page === item.page ? "active" : ""}`}
                    title={item.page === "trash" ? "지운 것을 되살리거나 완전히 지웁니다" : undefined}
                  >
                    <Icon name={item.icon} />
                    {item.label}
                    {item.page === "trash" && (binned?.total ?? 0) > 0 && <span className="nav-count">{binned?.total}</span>}
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
          <a className="who" href="#/settings/account" title="내 계정 · 비밀번호">
            <strong>{user.username}</strong>
            <small>{isAdmin ? "관리자" : "멤버"}</small>
          </a>
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
            <ImportTray />
            <a className={`pill ${online > 0 ? "ok" : ""}`} href="#/nodes">
              <i />
              노드 {online}/{nodes?.length ?? 0} 온라인
            </a>
            <a className="pill plain" href="#/guides/manual" title="사용 설명서 · 템플릿 가이드">
              <Icon name="book" size={13} />
              설명서
            </a>
            <a
              className={`pill plain ${unseen ? "new" : ""}`}
              href="#/releases"
              aria-label={`릴리즈 (v${loaded})`}
              title={unseen ? "새 버전입니다 — 무엇이 바뀌었는지 보기" : "릴리즈"}
            >
              {unseen && <i />}
              {loaded ? `v${loaded}` : "릴리즈"}
            </a>
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
        {updated && (
          <div className="update-bar" role="status">
            BADAMOA가 업데이트되었습니다(v{updated}). 새로고침하면 바뀐 화면을 씁니다.
            <button className="small primary" onClick={() => window.location.reload()}>
              새로고침
            </button>
          </div>
        )}
        <main className="content">
          {page === "home" && <Home user={user} nodes={nodes} />}
          {page === "datasets" && <Datasets sub={sub} arg={arg} isAdmin={isAdmin} />}
          {page === "labeling" && <Labeling setId={sub} arg={arg} />}
          {page === "review" && <Review releaseId={sub} />}
          {page === "training" && <Training runId={sub} arg={arg} />}
          {page === "models" && <Models modelId={sub} arg={arg} isAdmin={isAdmin} />}
          {page === "nodes" && (sub ? <NodeDetail nodeId={sub} isAdmin={isAdmin} /> : <Nodes isAdmin={isAdmin} />)}
          {page === "jobs" && <Jobs />}
          {page === "agents" && <Agents />}
          {page === "settings" && <Settings isAdmin={isAdmin} nodes={nodes} user={user} sub={sub} />}
          {page === "trash" && <TrashPage isAdmin={isAdmin} />}
          {page === "releases" && <ReleasesPage version={loaded} selected={sub} />}
          {page === "guides" && <GuidesPage name={sub} />}
        </main>
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
