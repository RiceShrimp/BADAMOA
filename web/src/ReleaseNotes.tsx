import { type ReactNode, useEffect, useMemo, useState } from "react";
import { api } from "./api";
import "./releases.css";
import { Icon } from "./ui";

interface Release {
  version: string;
  date: string | null;
  summary: string;
  sections: { title: string; items: string[] }[];
}

const SEEN_KEY = "badamoa.seenVersion";
const PER_PAGE = 10;
// The kinds of change, in the order the release notes use, each with its own mark.
const KIND: Record<string, string> = { 추가: "added", 변경: "changed", 수정: "fixed", "알려진 문제": "known", 제거: "removed" };

/** True until the user has opened the notes for this version (e.g. right after an upgrade). */
export function hasUnseenRelease(version: string): boolean {
  try {
    return localStorage.getItem(SEEN_KEY) !== version;
  } catch {
    return false;
  }
}

/** `code` and **bold** in a release note line, the two marks the notes use. */
function inline(text: string): ReactNode[] {
  return text.split(/(`[^`]+`|\*\*[^*]+\*\*)/g).map((part, i) => {
    if (part.startsWith("`") && part.endsWith("`") && part.length > 2) return <code key={i}>{part.slice(1, -1)}</code>;
    if (part.startsWith("**") && part.endsWith("**") && part.length > 4) return <strong key={i}>{part.slice(2, -2)}</strong>;
    return part;
  });
}

const matches = (release: Release, needle: string) =>
  `v${release.version} ${release.date ?? ""} ${release.summary} ${release.sections.flatMap((s) => [s.title, ...s.items]).join(" ")}`
    .toLowerCase()
    .includes(needle);

/**
 * Every version and what changed in it, newest first, laid out like a code host's releases page:
 * the date and tag beside each release, the newest marked, one page of ten at a time.
 * ``#/releases/<version>`` shows one release on its own, with the ones before and after it.
 */
export function ReleasesPage({ version, selected }: { version: string; selected?: string }) {
  const [releases, setReleases] = useState<Release[] | null>(null);
  const [error, setError] = useState<string | null>(null);
  const [query, setQuery] = useState("");
  const [page, setPage] = useState(0);

  useEffect(() => {
    try {
      localStorage.setItem(SEEN_KEY, version);
    } catch {
      /* storage unavailable: the "new" marker simply stays */
    }
    api<{ releases: Release[] }>("/release-notes")
      .then((res) => setReleases(res.releases))
      .catch((e) => setError(e instanceof Error ? e.message : String(e)));
  }, [version]);
  useEffect(() => {
    setPage(0);
  }, [query]);
  useEffect(() => {
    window.scrollTo({ top: 0 }); // a new page starts at its top
  }, [page, selected]);

  const needle = query.trim().toLowerCase();
  const shown = useMemo(() => (releases ?? []).filter((r) => !needle || matches(r, needle)), [releases, needle]);
  const latest = releases?.[0]?.version;

  if (error) return <div className="error">{error}</div>;
  if (!releases) return <p className="muted">불러오는 중…</p>;

  if (selected) {
    const at = releases.findIndex((r) => r.version === selected.replace(/^v/, ""));
    if (at < 0) {
      return (
        <div className="card empty-state">
          <strong>v{selected.replace(/^v/, "")} 릴리즈를 찾을 수 없습니다</strong>
          <a href="#/releases">모든 릴리즈 보기</a>
        </div>
      );
    }
    const newer = releases[at - 1];
    const older = releases[at + 1];
    return (
      <div className="releases-page">
        <a className="back-link" href="#/releases">
          ← 모든 릴리즈
        </a>
        <ReleaseEntry release={releases[at]} latest={latest} installed={version} />
        <nav className="release-steps" aria-label="이전 · 다음 릴리즈">
          {older ? (
            <a href={`#/releases/${older.version}`}>
              <small>이전 릴리즈</small>v{older.version}
            </a>
          ) : (
            <span />
          )}
          {newer && (
            <a href={`#/releases/${newer.version}`} className="right">
              <small>다음 릴리즈</small>v{newer.version}
            </a>
          )}
        </nav>
      </div>
    );
  }

  const pages = Math.max(1, Math.ceil(shown.length / PER_PAGE));
  const visible = shown.slice(page * PER_PAGE, (page + 1) * PER_PAGE);
  return (
    <div className="releases-page">
      <div className="page-head">
        <div>
          <h1>
            릴리즈 <span className="count-badge">{releases.length}</span>
          </h1>
          <p className="muted">버전마다 무엇이 추가되고 바뀌고 고쳐졌는지입니다. 지금 이 Hub는 v{version}입니다.</p>
        </div>
        <input
          type="search"
          className="release-search"
          value={query}
          onChange={(e) => setQuery(e.target.value)}
          placeholder="릴리즈 찾기"
          aria-label="릴리즈 찾기"
        />
      </div>
      {shown.length === 0 && <div className="card empty">"{query}"가 들어간 릴리즈가 없습니다.</div>}
      {visible.map((release) => (
        <ReleaseEntry key={release.version} release={release} latest={latest} installed={version} linked />
      ))}
      {pages > 1 && (
        <nav className="release-pages" aria-label="페이지">
          <button onClick={() => setPage(page - 1)} disabled={page === 0}>
            ‹ 이전
          </button>
          {Array.from({ length: pages }, (_, i) => (
            <button key={i} className={i === page ? "active" : ""} onClick={() => setPage(i)} aria-current={i === page}>
              {i + 1}
            </button>
          ))}
          <button onClick={() => setPage(page + 1)} disabled={page >= pages - 1}>
            다음 ›
          </button>
        </nav>
      )}
    </div>
  );
}

function ReleaseEntry({
  release,
  latest,
  installed,
  linked = false,
}: {
  release: Release;
  latest?: string;
  installed: string;
  linked?: boolean;
}) {
  const href = `#/releases/${release.version}`;
  const counts = release.sections.filter((s) => s.items.length > 0);
  return (
    <section className="release-entry" id={`v${release.version}`}>
      <aside className="release-meta">
        {release.date && <time dateTime={release.date}>{release.date}</time>}
        <a className="release-tag" href={href} title="이 릴리즈만 보기">
          <Icon name="tag" size={14} />v{release.version}
        </a>
        {release.version === installed && <span className="release-installed">설치된 버전</span>}
      </aside>
      <article className="card release-card">
        <header>
          <h2>{linked ? <a href={href}>v{release.version}</a> : `v${release.version}`}</h2>
          {release.version === latest && <span className="release-latest">Latest</span>}
        </header>
        {release.summary && <p className="release-summary">{inline(release.summary)}</p>}
        {release.sections.map((section) => (
          <div key={section.title} className={`release-section kind-${KIND[section.title] ?? "other"}`}>
            <h3>{section.title}</h3>
            <ul>
              {section.items.map((item) => (
                <li key={item}>{inline(item)}</li>
              ))}
            </ul>
          </div>
        ))}
        {counts.length > 0 && (
          <footer className="release-foot">
            {counts.map((s) => (
              <span key={s.title} className={`kind-${KIND[s.title] ?? "other"}`}>
                <i />
                {s.title} {s.items.length}
              </span>
            ))}
          </footer>
        )}
      </article>
    </section>
  );
}
