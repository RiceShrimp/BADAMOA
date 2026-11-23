import { type Ref, useEffect, useState } from "react";

/** How many images a page shows: the Hub sends at most 200 at a time. One choice for every image list. */
export const PAGE_SIZES = [30, 60, 100, 200];
const PAGE_SIZE_KEY = "badamoa.pageSize";

function rememberedPageSize(): number {
  try {
    const n = Number(localStorage.getItem(PAGE_SIZE_KEY));
    return PAGE_SIZES.includes(n) ? n : 60;
  } catch {
    return 60;
  }
}

/**
 * The page and how many images it shows (remembered in this browser). A new size keeps the page near the images
 * being looked at.
 */
export function usePaging(): {
  page: number;
  setPage: (page: number) => void;
  pageSize: number;
  setPageSize: (size: number) => void;
} {
  const [page, setPage] = useState(0);
  const [pageSize, setPageSizeState] = useState(rememberedPageSize);
  const setPageSize = (n: number) => {
    setPage(Math.floor((page * pageSize) / n));
    setPageSizeState(n);
    try {
      localStorage.setItem(PAGE_SIZE_KEY, String(n));
    } catch {
      /* not remembered */
    }
  };
  return { page, setPage, pageSize, setPageSize };
}

/** The page number typed in: Enter or leaving the box goes there (kept within 1 … pages). */
function PageJump({ page, pages, onGo }: { page: number; pages: number; onGo: (page: number) => void }) {
  const [typed, setTyped] = useState(String(page + 1));
  useEffect(() => setTyped(String(page + 1)), [page]);
  const go = () => {
    const n = Math.round(Number(typed));
    if (!Number.isFinite(n) || typed.trim() === "") return setTyped(String(page + 1));
    const target = Math.min(pages, Math.max(1, n)) - 1;
    setTyped(String(target + 1));
    if (target !== page) onGo(target);
  };
  return (
    <span className="page-jump">
      <input
        type="number"
        min={1}
        max={pages}
        value={typed}
        onChange={(e) => setTyped(e.target.value)}
        onKeyDown={(e) => e.key === "Enter" && go()}
        onBlur={go}
        aria-label="페이지 번호"
        style={{ width: `${Math.max(5, String(pages).length + 3)}ch` }}
      />
      <span className="num">/ {pages.toLocaleString()}</span>
    </span>
  );
}

/**
 * « ‹ [page] / pages › » and how many a page shows. Under a list (``bottom``) as well as above it: going to another
 * page from there brings the list's top (``top``) into view.
 */
export function Pager({
  page,
  pages,
  onGo,
  pageSize,
  onPageSize,
  bottom = false,
  top,
  innerRef,
}: {
  page: number;
  pages: number;
  onGo: (page: number) => void;
  pageSize: number;
  onPageSize: (size: number) => void;
  bottom?: boolean;
  top?: { current: HTMLElement | null };
  innerRef?: Ref<HTMLDivElement>;
}) {
  const go = (next: number) => {
    onGo(next);
    if (bottom) top?.current?.scrollIntoView({ block: "start", behavior: "smooth" });
  };
  return (
    <div className={`pager ${bottom ? "pager-bottom" : ""}`} ref={innerRef}>
      <button onClick={() => go(0)} disabled={page === 0} aria-label="첫 페이지">
        «
      </button>
      <button onClick={() => go(page - 1)} disabled={page === 0} aria-label="이전 페이지">
        ‹
      </button>
      <PageJump page={page} pages={pages} onGo={go} />
      <button onClick={() => go(page + 1)} disabled={page >= pages - 1} aria-label="다음 페이지">
        ›
      </button>
      <button onClick={() => go(pages - 1)} disabled={page >= pages - 1} aria-label="마지막 페이지">
        »
      </button>
      <select value={pageSize} onChange={(e) => onPageSize(Number(e.target.value))} aria-label="한 페이지에 보일 수">
        {PAGE_SIZES.map((n) => (
          <option key={n} value={n}>
            {n}개씩
          </option>
        ))}
      </select>
    </div>
  );
}
