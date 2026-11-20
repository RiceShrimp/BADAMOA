import { type ReactNode, useEffect, useRef } from "react";
import ReactMarkdown from "react-markdown";
import remarkGfm from "remark-gfm";
import { api } from "./api";
import "./guides.css";
import { Icon, usePolling } from "./ui";

export interface GuideInfo {
  name: string;
  title: string;
  summary: string;
  file: string;
  ready: boolean;
}

interface GuideText {
  name: string;
  title: string;
  file: string;
  markdown: string;
}

const REPO = "https://github.com/RiceShrimp/BADAMOA/blob/main/";

/** A heading's anchor as GitHub makes it, so the guides' own tables of contents work here too. */
function slug(text: string): string {
  return text
    .trim()
    .toLowerCase()
    .replace(/[^\p{L}\p{N}\s-]/gu, "")
    .replace(/\s/g, "-");
}

function textOf(node: ReactNode): string {
  if (typeof node === "string" || typeof node === "number") return String(node);
  if (Array.isArray(node)) return node.map(textOf).join("");
  if (node && typeof node === "object" && "props" in node) return textOf((node as { props: { children?: ReactNode } }).props.children);
  return "";
}

/** The user manual and the template guide, read here like the release notes, to download or print. */
export function GuidesPage({ name }: { name?: string }) {
  const { data: list } = usePolling(() => api<GuideInfo[]>("/guides"), 0);
  const wanted = name || "manual";
  const { data: guide, error } = usePolling(() => api<GuideText>(`/guides/${wanted}`), 0, [wanted]);
  const body = useRef<HTMLDivElement>(null);

  // A link to a heading (#3-데이터셋) scrolls there instead of leaving the guide.
  useEffect(() => {
    if (!guide) return;
    body.current?.scrollTo?.({ top: 0 });
    window.scrollTo({ top: 0 });
  }, [guide]);

  const byFile = new Map((list ?? []).map((g) => [g.file, g.name]));
  const heading =
    (Tag: "h1" | "h2" | "h3" | "h4") =>
    ({ children }: { children?: ReactNode }) => <Tag id={slug(textOf(children))}>{children}</Tag>;
  const link = ({ href = "", children }: { href?: string; children?: ReactNode }) => {
    if (href.startsWith("#") && !href.startsWith("#/")) {
      const id = decodeURIComponent(href.slice(1));
      return (
        <a
          href={href}
          onClick={(e) => {
            e.preventDefault();
            document.getElementById(id)?.scrollIntoView({ behavior: "smooth", block: "start" });
          }}
        >
          {children}
        </a>
      );
    }
    const file = href.split("#")[0].split("/").pop() ?? "";
    if (byFile.has(file)) return <a href={`#/guides/${byFile.get(file)}`}>{children}</a>; // another guide, here
    if (/^https?:\/\//.test(href)) {
      return (
        <a href={href} target="_blank" rel="noreferrer">
          {children}
        </a>
      );
    }
    // a file of the repository (README, CHANGELOG …): on GitHub
    const path = href.replace(/^\.\.\//, "").replace(/^\.\//, "");
    return (
      <a href={`${REPO}${path.startsWith("docs/") || path.includes("/") || !path.endsWith(".md") ? path : `docs/${path}`}`} target="_blank" rel="noreferrer">
        {children}
      </a>
    );
  };

  const download = () => {
    if (!guide) return;
    const url = URL.createObjectURL(new Blob([guide.markdown], { type: "text/markdown;charset=utf-8" }));
    const a = document.createElement("a");
    a.href = url;
    a.download = `MOA-${guide.file}`;
    a.click();
    URL.revokeObjectURL(url);
  };

  return (
    <div className="guides">
      <nav className="guides-list" aria-label="설명서">
        <div className="eyebrow">설명서</div>
        {list?.map((g) => (
          <a key={g.name} href={`#/guides/${g.name}`} className={`guides-item ${g.name === wanted ? "active" : ""} ${g.ready ? "" : "off"}`}>
            <strong>{g.title}</strong>
            <span className="muted small">{g.ready ? g.summary : "준비 중"}</span>
          </a>
        ))}
        <a className="guides-item" href="#/releases">
          <strong>릴리즈 노트</strong>
          <span className="muted small">버전마다 바뀐 것</span>
        </a>
      </nav>
      <article className="guides-body" ref={body}>
        {guide && (
          <div className="guides-actions">
            <button onClick={download} title={`${guide.file} 파일로 받습니다`}>
              <Icon name="download" size={14} /> 내려받기 (.md)
            </button>
            <button onClick={() => window.print()} title="인쇄하거나 PDF로 저장합니다">
              인쇄 · PDF
            </button>
          </div>
        )}
        {error && !guide && <div className="notice">{error}</div>}
        {!guide && !error && <p className="muted">불러오는 중…</p>}
        {guide && (
          <div className="markdown">
            <ReactMarkdown
              remarkPlugins={[remarkGfm]}
              components={{ h1: heading("h1"), h2: heading("h2"), h3: heading("h3"), h4: heading("h4"), a: link }}
            >
              {guide.markdown}
            </ReactMarkdown>
          </div>
        )}
      </article>
    </div>
  );
}
