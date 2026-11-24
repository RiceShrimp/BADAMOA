import { type ReactNode, useCallback, useEffect, useLayoutEffect, useMemo, useRef, useState } from "react";
import ReactMarkdown, { type Components } from "react-markdown";
import remarkGfm from "remark-gfm";
import { api } from "./api";
import "./guides.css";
import { copyText, Icon, usePolling } from "./ui";

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
  images?: Record<string, [number, number]>; // the manual's pictures: name → width, height
}

/** The manual's pictures (img/manual/x.png) come from the Hub; the rest of a src stays as written. */
const PICTURES = "img/manual/";

/** A screen in the manual: fetched only when scrolled near, its size kept so the page does not jump while it loads,
 *  its alt text as the caption, and a click opens it full size. */
function Picture({ src = "", alt = "", sizes }: { src?: string; alt?: string; sizes: GuideText["images"] }) {
  const name = src.startsWith(PICTURES) ? src.slice(PICTURES.length) : "";
  const url = name ? `/api/guides/images/${name}` : src;
  const [width, height] = (name && sizes?.[name]) || [];
  return (
    <span className="guide-picture">
      <a href={url} target="_blank" rel="noreferrer" title="크게 보기">
        <img src={url} alt={alt} width={width} height={height} loading="lazy" decoding="async" />
      </a>
      {alt && <span className="guide-picture-caption">{alt}</span>}
    </span>
  );
}

/** A heading of the guide on screen, for the table of contents, the reading position and the search. */
interface Head {
  id: string;
  level: 2 | 3;
  num: string; // "3.2" of "3.2 데이터셋 가져오기", "" when unnumbered
  text: string; // the rest
  el: HTMLElement;
}

/** Every place the search found, in reading order, and the section (h2/h3) each one is in. */
interface Hits {
  ranges: Range[];
  section: string[];
}

const NO_HITS: Hits = { ranges: [], section: [] };
// How far below the window's top a heading counts as the one being read (top bar 52px + room).
const READING_LINE = 140;

/** A heading's anchor: lower case, punctuation dropped, spaces as hyphens, so the guides' own links and a link copied here both work. */
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

/** The guide without its own "## 차례": the side shows one that follows the reading. Downloads keep it. */
function withoutToc(md: string): string {
  const lines = md.split("\n");
  const start = lines.findIndex((l) => /^##\s+차례\s*$/.test(l));
  if (start < 0) return md;
  let end = start + 1;
  while (end < lines.length && !/^##\s/.test(lines[end])) end++;
  return [...lines.slice(0, start), ...lines.slice(end)].join("\n");
}

// ---- markdown plugins: alert boxes (> [!NOTE]) and heading ids ----

interface MdNode {
  type: string;
  value?: string;
  tagName?: string;
  children?: MdNode[];
  properties?: Record<string, unknown>;
  data?: { hProperties?: Record<string, unknown> };
}

const ALERTS: Record<string, string> = { NOTE: "참고", TIP: "팁", IMPORTANT: "중요", WARNING: "주의", CAUTION: "조심" };

/** `> [!NOTE]` and its kin become a box with a label. */
function remarkAlerts() {
  const walk = (node: MdNode) => {
    if (node.type === "blockquote") {
      const first = node.children?.[0];
      const lead = first?.type === "paragraph" ? first.children?.[0] : undefined;
      const mark = lead?.type === "text" ? /^\[!(NOTE|TIP|IMPORTANT|WARNING|CAUTION)\][ \t]*(?:\r?\n)?/.exec(lead.value ?? "") : null;
      if (first && lead && mark) {
        lead.value = (lead.value ?? "").slice(mark[0].length);
        if (!lead.value) first.children?.shift();
        if (!first.children?.length) node.children?.shift();
        node.data = {
          ...node.data,
          hProperties: { className: ["callout", `callout-${mark[1].toLowerCase()}`], dataLabel: ALERTS[mark[1]] },
        };
      }
    }
    node.children?.forEach(walk);
  };
  return (tree: MdNode) => walk(tree);
}

function plainText(node: MdNode): string {
  return node.type === "text" ? (node.value ?? "") : (node.children ?? []).map(plainText).join("");
}

/** Each heading's id: its slug, numbered from the second time the same one comes (-1, -2). */
function rehypeHeadingIds() {
  return (tree: MdNode) => {
    const seen = new Map<string, number>();
    const walk = (node: MdNode) => {
      if (node.type === "element" && /^h[1-4]$/.test(node.tagName ?? "")) {
        const base = slug(plainText(node));
        const count = seen.get(base) ?? 0;
        seen.set(base, count + 1);
        node.properties = { ...node.properties, id: count ? `${base}-${count}` : base };
      }
      node.children?.forEach(walk);
    };
    walk(tree);
  };
}

// ---- small pieces of the page ----

/** Icons this page needs that the shared set has not. */
function Glyph({ name, size = 15 }: { name: "search" | "list" | "up" | "down" | "chevron"; size?: number }) {
  const paths = {
    search: (
      <>
        <circle cx="11" cy="11" r="6.5" />
        <path d="M16 16l4.5 4.5" />
      </>
    ),
    list: <path d="M9 6h11M9 12h11M9 18h11M4.5 6h.01M4.5 12h.01M4.5 18h.01" />,
    up: <path d="M6 15l6-6 6 6" />,
    down: <path d="M6 9l6 6 6-6" />,
    chevron: <path d="M6 9l6 6 6-6" />,
  }[name];
  return (
    <svg
      className="icon"
      width={size}
      height={size}
      viewBox="0 0 24 24"
      fill="none"
      stroke="currentColor"
      strokeWidth="1.8"
      strokeLinecap="round"
      strokeLinejoin="round"
      aria-hidden="true"
    >
      {paths}
    </svg>
  );
}

/** The "#" beside a heading: goes there and copies a link to it. */
function HeadingAnchor({ href, id }: { href: string; id: string }) {
  const [copied, setCopied] = useState(false);
  return (
    <a
      className={`heading-anchor ${copied ? "copied" : ""}`}
      href={href}
      aria-label="이 부분의 링크 복사"
      title="이 부분의 링크 복사"
      onClick={async (e) => {
        e.preventDefault();
        window.history.replaceState(null, "", href);
        document.getElementById(id)?.scrollIntoView({ behavior: "smooth", block: "start" });
        if (await copyText(window.location.href)) {
          setCopied(true);
          window.setTimeout(() => setCopied(false), 1500);
        }
      }}
    >
      {copied ? "링크 복사됨" : "#"}
    </a>
  );
}

/** A command or code: a copy button in its corner. */
function CodeBlock({ children }: { children?: ReactNode }) {
  const [state, setState] = useState<"" | "copied" | "select">("");
  const box = useRef<HTMLDivElement>(null);
  const shown = useRef<HTMLPreElement>(null);
  const copy = async () => {
    const done = await copyText(textOf(children).replace(/\n$/, ""), box.current);
    if (!done && shown.current) window.getSelection()?.selectAllChildren(shown.current); // copying by hand is one key away
    setState(done ? "copied" : "select");
    window.setTimeout(() => setState(""), done ? 1500 : 4000);
  };
  return (
    <div className="code-block" ref={box}>
      <pre ref={shown}>{children}</pre>
      <button type="button" className="code-copy" onClick={copy}>
        <Icon name={state === "copied" ? "check" : "copy"} size={13} />
        {state === "copied" ? "복사됨" : state === "select" ? "Ctrl+C로 복사" : "복사"}
      </button>
    </div>
  );
}

/** A table scrolls sideways on its own instead of widening the page. */
function Table({ children }: { children?: ReactNode }) {
  return (
    <div className="table-wrap">
      <table>{children}</table>
    </div>
  );
}

/** "3.2 데이터셋 가져오기" → "3.2" and "데이터셋 가져오기", for the table of contents. */
function splitNumber(text: string): { num: string; text: string } {
  const m = /^(\d+(?:\.\d+)*)\.?\s+(.+)$/.exec(text.trim());
  return m ? { num: m[1], text: m[2] } : { num: "", text: text.trim() };
}

/** Every text in the guide that holds the query, and the h2/h3 section it is in; the "#" and copy buttons are not text. */
function findAll(root: HTMLElement, query: string): Hits {
  const needle = query.toLowerCase();
  const ranges: Range[] = [];
  const section: string[] = [];
  let current = "";
  const walker = document.createTreeWalker(root, NodeFilter.SHOW_ELEMENT | NodeFilter.SHOW_TEXT, {
    acceptNode: (node) =>
      node instanceof HTMLElement && (node.classList.contains("heading-anchor") || node.classList.contains("code-copy"))
        ? NodeFilter.FILTER_REJECT
        : NodeFilter.FILTER_ACCEPT,
  });
  for (let node = walker.nextNode(); node; node = walker.nextNode()) {
    if (node instanceof HTMLElement) {
      if ((node.tagName === "H2" || node.tagName === "H3") && node.id) current = node.id;
      continue;
    }
    const text = (node.nodeValue ?? "").toLowerCase();
    for (let at = text.indexOf(needle); at >= 0; at = text.indexOf(needle, at + needle.length)) {
      const range = document.createRange();
      range.setStart(node, at);
      range.setEnd(node, at + needle.length);
      ranges.push(range);
      section.push(current);
    }
  }
  return { ranges, section };
}

type HighlightClass = new (...ranges: Range[]) => object;

/** Marks the places found, all of them and the current one, with the browser's highlight API where it has one. */
function paint(hits: Hits, at: number) {
  const registry = (globalThis.CSS as unknown as { highlights?: Map<string, object> } | undefined)?.highlights;
  const Highlight = (globalThis as unknown as { Highlight?: HighlightClass }).Highlight;
  if (!registry || !Highlight) return;
  registry.delete("guide-hit");
  registry.delete("guide-hit-now");
  if (hits.ranges.length === 0) return;
  registry.set("guide-hit", new Highlight(...hits.ranges));
  const now = hits.ranges[at];
  if (now) registry.set("guide-hit-now", new Highlight(now));
}

function bringIntoView(range: Range, smooth: boolean) {
  const box = range.getBoundingClientRect();
  if (box.top >= READING_LINE && box.bottom <= window.innerHeight - 40) return; // already in sight
  window.scrollTo({ top: window.scrollY + box.top - window.innerHeight / 3, behavior: smooth ? "smooth" : "auto" });
}

/** The user manual and the template guide, read here like the release notes, to download or print. */
export function GuidesPage({ name, section }: { name?: string; section?: string }) {
  const { data: list } = usePolling(() => api<GuideInfo[]>("/guides"), 0);
  const wanted = name || "manual";
  const { data: guide, error } = usePolling(() => api<GuideText>(`/guides/${wanted}`), 0, [wanted]);
  const shown = guide && guide.name === wanted ? guide : null; // not the last guide while the next one loads
  const body = useRef<HTMLDivElement>(null);
  const tocRef = useRef<HTMLElement>(null);
  const searchRef = useRef<HTMLInputElement>(null);
  const [heads, setHeads] = useState<Head[]>([]);
  const [active, setActive] = useState("");
  const [tocOpen, setTocOpen] = useState(false); // narrow screens: the contents drop down from a bar
  const [query, setQuery] = useState("");
  const [hits, setHits] = useState<Hits>(NO_HITS);
  const [at, setAt] = useState(0);

  const target = (id: string) => `#/guides/${wanted}/${id}`;

  // The guide as it reads on screen; kept as it is while the page around it changes (the reading position,
  // the search), so its headings and text stay the same elements the contents and the search point at.
  const article = useMemo(() => {
    if (!shown) return null;
    const byFile = new Map((list ?? []).map((g) => [g.file, g.name]));
    const heading =
      (Tag: "h1" | "h2" | "h3" | "h4") =>
      ({ id = "", children }: { id?: string; children?: ReactNode }) => (
        <Tag id={id}>
          <span className="heading-text">{children}</span>
          {Tag !== "h1" && id && <HeadingAnchor href={`#/guides/${shown.name}/${id}`} id={id} />}
        </Tag>
      );
    const link = ({ href = "", children }: { href?: string; children?: ReactNode }) => {
      if (href.startsWith("#") && !href.startsWith("#/")) {
        const id = decodeURIComponent(href.slice(1));
        const to = `#/guides/${shown.name}/${id}`;
        return (
          <a
            href={to}
            onClick={(e) => {
              e.preventDefault();
              window.history.replaceState(null, "", to);
              document.getElementById(id)?.scrollIntoView({ behavior: "smooth", block: "start" });
            }}
          >
            {children}
          </a>
        );
      }
      const [path, anchor] = href.split("#");
      const file = path.split("/").pop() ?? "";
      if (byFile.has(file)) {
        // another guide, here (and the place in it)
        return <a href={`#/guides/${byFile.get(file)}${anchor ? `/${anchor}` : ""}`}>{children}</a>;
      }
      if (/^https?:\/\//.test(href)) {
        return (
          <a href={href} target="_blank" rel="noreferrer">
            {children}
          </a>
        );
      }
      // the release notes have their own screen
      if (file === "CHANGELOG.md") return <a href="#/releases">{children}</a>;
      // any other file is not in the Hub: the words stay, without a link out
      return <span>{children}</span>;
    };
    const components: Components = {
      h1: heading("h1"),
      h2: heading("h2"),
      h3: heading("h3"),
      h4: heading("h4"),
      a: link,
      pre: CodeBlock,
      table: Table,
      img: ({ src, alt }) => <Picture src={typeof src === "string" ? src : ""} alt={alt} sizes={shown.images} />,
    };
    return (
      <ReactMarkdown remarkPlugins={[remarkGfm, remarkAlerts]} rehypePlugins={[rehypeHeadingIds]} components={components}>
        {withoutToc(shown.markdown)}
      </ReactMarkdown>
    );
  }, [shown, list]);

  // The headings on screen, once the guide is drawn.
  useLayoutEffect(() => {
    const root = body.current;
    if (!root || !article) return setHeads([]);
    const found: Head[] = [];
    root.querySelectorAll<HTMLElement>("h2[id], h3[id]").forEach((el) => {
      const words = el.querySelector(".heading-text")?.textContent ?? el.textContent ?? "";
      found.push({ id: el.id, level: el.tagName === "H2" ? 2 : 3, ...splitNumber(words), el });
    });
    setHeads(found);
  }, [article]);

  // Where to start reading: the heading a link names (#/guides/manual/3-데이터셋), else the top.
  const placed = useRef("");
  useLayoutEffect(() => {
    if (!article || (heads.length === 0 && section)) return;
    const id = section ? safeDecode(section) : "";
    const key = `${wanted}/${id}`;
    if (placed.current === key) return;
    placed.current = key;
    const el = id ? document.getElementById(id) : null;
    if (el) el.scrollIntoView({ block: "start" });
    else window.scrollTo({ top: 0 });
  }, [article, heads, section, wanted]);

  // The section being read: the last heading above the reading line.
  useEffect(() => {
    if (heads.length === 0) return;
    let frame = 0;
    const update = () => {
      frame = 0;
      let current = "";
      for (const h of heads) {
        if (h.el.getBoundingClientRect().top <= READING_LINE) current = h.id;
        else break;
      }
      const bottom = window.innerHeight + window.scrollY >= document.documentElement.scrollHeight - 4;
      if (bottom && window.scrollY > 0) current = heads[heads.length - 1].id;
      setActive(current);
    };
    const onScroll = () => {
      if (!frame) frame = window.requestAnimationFrame(update);
    };
    update();
    window.addEventListener("scroll", onScroll, { passive: true });
    window.addEventListener("resize", onScroll);
    return () => {
      window.removeEventListener("scroll", onScroll);
      window.removeEventListener("resize", onScroll);
      if (frame) window.cancelAnimationFrame(frame);
    };
  }, [heads]);

  // The contents follow the reading: the current entry stays in sight in its own scroll.
  useEffect(() => {
    const box = tocRef.current;
    const item = box?.querySelector<HTMLElement>("a.current");
    if (!box || !item) return;
    const outer = box.getBoundingClientRect();
    const inner = item.getBoundingClientRect();
    if (inner.top < outer.top + 8 || inner.bottom > outer.bottom - 8) box.scrollTop += inner.top - outer.top - outer.height / 3;
  }, [active, tocOpen]);

  // The search: every place the words are, painted, and the count; it starts over with each guide.
  // Once the typing pauses the first place comes into view, unless Enter or a click went somewhere already.
  const firstLook = useRef(0);
  useEffect(() => {
    const root = body.current;
    const words = query.trim();
    const found = root && article && words ? findAll(root, words) : NO_HITS;
    setHits(found);
    setAt(0);
    paint(found, 0);
    if (found.ranges.length === 0) return;
    firstLook.current = window.setTimeout(() => bringIntoView(found.ranges[0], false), 250);
    return () => window.clearTimeout(firstLook.current);
  }, [query, article]);
  useEffect(() => () => paint(NO_HITS, 0), []); // nothing stays painted after leaving

  const step = useCallback(
    (by: number) => {
      const n = hits.ranges.length;
      if (n === 0) return;
      window.clearTimeout(firstLook.current);
      const next = (at + by + n) % n;
      setAt(next);
      paint(hits, next);
      bringIntoView(hits.ranges[next], true);
    },
    [at, hits],
  );
  const goToHit = (index: number) => {
    window.clearTimeout(firstLook.current);
    setAt(index);
    paint(hits, index);
    bringIntoView(hits.ranges[index], true);
    setTocOpen(false);
  };

  // "/" puts the cursor in the search, as on many documentation sites.
  useEffect(() => {
    const onKey = (e: KeyboardEvent) => {
      const typing = e.target instanceof HTMLElement && (e.target.closest("input, textarea, select") || e.target.isContentEditable);
      if (e.key !== "/" || typing || e.ctrlKey || e.metaKey || e.altKey) return;
      e.preventDefault();
      setTocOpen(true);
      window.setTimeout(() => searchRef.current?.focus(), 0);
    };
    window.addEventListener("keydown", onKey);
    return () => window.removeEventListener("keydown", onKey);
  }, []);

  const go = (id: string) => {
    window.history.replaceState(null, "", target(id));
    document.getElementById(id)?.scrollIntoView({ behavior: "smooth", block: "start" });
    setTocOpen(false);
  };

  const download = () => {
    if (!shown) return;
    const url = URL.createObjectURL(new Blob([shown.markdown], { type: "text/markdown;charset=utf-8" }));
    const a = document.createElement("a");
    a.href = url;
    a.download = `MOA-${shown.file}`;
    a.click();
    URL.revokeObjectURL(url);
  };

  // The chapter being read (its h2) and the entries under it.
  const activeIndex = heads.findIndex((h) => h.id === active);
  let chapter: Head | undefined;
  for (let i = activeIndex; i >= 0; i--) {
    if (heads[i].level === 2) {
      chapter = heads[i];
      break;
    }
  }
  const chapters = heads.filter((h) => h.level === 2);
  const sectionsOf = (h2: Head) => {
    const from = heads.indexOf(h2) + 1;
    const out: Head[] = [];
    for (let i = from; i < heads.length && heads[i].level === 3; i++) out.push(heads[i]);
    return out;
  };

  // Search results by section: how many places in each, in reading order.
  const bySection: { head?: Head; first: number; count: number }[] = [];
  hits.section.forEach((id, i) => {
    const last = bySection[bySection.length - 1];
    if (last && (last.head?.id ?? "") === id) last.count += 1;
    else bySection.push({ head: heads.find((h) => h.id === id), first: i, count: 1 });
  });
  const searching = query.trim() !== "";
  const current = list?.find((g) => g.name === wanted);

  const entry = (h: Head) => (
    <a
      key={h.id}
      href={target(h.id)}
      className={`toc-entry level-${h.level} ${h.id === active ? "current" : ""} ${h === chapter ? "in" : ""}`}
      aria-current={h.id === active ? "location" : undefined}
      onClick={(e) => {
        e.preventDefault();
        go(h.id);
      }}
    >
      {h.num && <span className="toc-num">{h.num}</span>}
      <span>{h.text}</span>
    </a>
  );

  return (
    <div className="guides">
      <aside className={`guides-side ${tocOpen ? "open" : ""}`}>
        <button
          type="button"
          className="guides-bar"
          aria-expanded={tocOpen}
          aria-controls="guides-panel"
          onClick={() => setTocOpen((o) => !o)}
        >
          <Glyph name="list" />
          <span className="guides-bar-title">
            {chapter ? (
              <>
                {chapter.num && <span className="toc-num">{chapter.num}</span>}
                {chapter.text}
              </>
            ) : (
              (current?.title ?? "차례")
            )}
          </span>
          <span className="guides-bar-hint muted small">{tocOpen ? "닫기" : "차례 · 찾기"}</span>
          <Glyph name="chevron" />
        </button>
        <div className="guides-panel" id="guides-panel">
          <nav className="guides-list" aria-label="설명서">
            {list?.map((g) => (
              <a
                key={g.name}
                href={`#/guides/${g.name}`}
                className={`guides-item ${g.name === wanted ? "active" : ""} ${g.ready ? "" : "off"}`}
                title={g.ready ? g.summary : "준비 중"}
                onClick={() => {
                  if (g.name === wanted) window.scrollTo({ top: 0, behavior: "smooth" });
                  setTocOpen(false);
                }}
              >
                <Icon name="book" size={14} />
                <span>{g.title}</span>
              </a>
            ))}
            <a className="guides-item" href="#/releases" title="버전마다 바뀐 것">
              <Icon name="tag" size={14} />
              <span>릴리즈 노트</span>
            </a>
          </nav>

          <div className={`guides-search ${searching ? "on" : ""}`} role="search">
            <Glyph name="search" size={14} />
            <input
              ref={searchRef}
              type="search"
              value={query}
              placeholder={`${current?.title ?? "설명서"}에서 찾기`}
              aria-label="설명서에서 찾기"
              onChange={(e) => setQuery(e.target.value)}
              onKeyDown={(e) => {
                if (e.key === "Enter") {
                  e.preventDefault();
                  step(e.shiftKey ? -1 : 1);
                } else if (e.key === "Escape") {
                  setQuery("");
                }
              }}
            />
            {searching ? (
              <>
                <span className="guides-search-count" aria-live="polite">
                  {hits.ranges.length ? `${at + 1}/${hits.ranges.length}` : "0"}
                </span>
                <button type="button" className="icon-btn" onClick={() => step(-1)} disabled={!hits.ranges.length} aria-label="이전" title="이전 (Shift+Enter)">
                  <Glyph name="up" size={14} />
                </button>
                <button type="button" className="icon-btn" onClick={() => step(1)} disabled={!hits.ranges.length} aria-label="다음" title="다음 (Enter)">
                  <Glyph name="down" size={14} />
                </button>
              </>
            ) : (
              <kbd title="/ 를 누르면 여기로">/</kbd>
            )}
          </div>

          <nav className="guides-toc" ref={tocRef} aria-label={searching ? "찾은 곳" : "차례"}>
            {searching ? (
              bySection.length === 0 ? (
                <p className="muted small guides-empty">“{query.trim()}”이(가) 없습니다.</p>
              ) : (
                <>
                  <div className="eyebrow">찾은 곳 {hits.ranges.length}</div>
                  {bySection.map((s) => (
                    <a
                      key={`${s.head?.id ?? "top"}-${s.first}`}
                      href={s.head ? target(s.head.id) : `#/guides/${wanted}`}
                      className={`toc-entry level-${s.head?.level ?? 2} ${hits.section[at] === (s.head?.id ?? "") ? "current" : ""}`}
                      onClick={(e) => {
                        e.preventDefault();
                        goToHit(s.first);
                      }}
                    >
                      {s.head?.num && <span className="toc-num">{s.head.num}</span>}
                      <span>{s.head?.text ?? "머리말"}</span>
                      <span className="toc-count">{s.count}</span>
                    </a>
                  ))}
                </>
              )
            ) : (
              <>
                <div className="eyebrow">차례</div>
                {chapters.map((h2) => (
                  <div key={h2.id} className="toc-chapter">
                    {entry(h2)}
                    {h2 === chapter && sectionsOf(h2).length > 0 && <div className="toc-sections">{sectionsOf(h2).map(entry)}</div>}
                  </div>
                ))}
              </>
            )}
          </nav>
        </div>
      </aside>

      <article className="guides-body">
        {shown && (
          <div className="guides-actions">
            <span className="muted small mono">{shown.file}</span>
            <span className="spacer" />
            <button onClick={download} title={`${shown.file} 파일로 받습니다`}>
              <Icon name="download" size={14} /> 내려받기 (.md)
            </button>
            <button onClick={() => window.print()} title="인쇄하거나 PDF로 저장합니다">
              인쇄 · PDF
            </button>
          </div>
        )}
        {error && !shown && <div className="notice">{error}</div>}
        {!shown && !error && <p className="muted">불러오는 중…</p>}
        <div className="markdown" ref={body}>
          {article}
        </div>
      </article>
      {tocOpen && <div className="guides-scrim" onClick={() => setTocOpen(false)} aria-hidden="true" />}
    </div>
  );
}

function safeDecode(text: string): string {
  try {
    return decodeURIComponent(text);
  } catch {
    return text;
  }
}
