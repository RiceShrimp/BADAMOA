import { useKept } from "./Order";
import { Icon } from "./ui";

/** How a screen lays out its cards: side by side in a grid, or one long card to a row. */
export type CardLayout = "grid" | "rows";

/**
 * A screen's card layout as this person picked it (``fallback`` until they do). It is kept on the Hub with their card
 * order, so it is the same on any PC; nobody else's screen changes.
 */
export function useCardLayout(page: string, fallback: CardLayout = "grid") {
  const [kept, keep] = useKept(`layout.${page}`);
  const picked = kept?.[0];
  const layout: CardLayout = picked === "grid" || picked === "rows" ? picked : fallback;
  const choose = (next: CardLayout) => keep(next === fallback ? [] : [next]); // the screen's own: nothing to keep
  return [layout, choose] as const;
}

/** The class a card container takes for ``layout`` (the grid is the container's own). */
export const layoutClass = (layout: CardLayout) => (layout === "rows" ? " as-rows" : "");

/** 그리드 · 가로형: the two ways to lay out a screen's cards. */
export function LayoutSwitch({ value, onChange }: { value: CardLayout; onChange: (layout: CardLayout) => void }) {
  return (
    <div className="layout-switch" role="radiogroup" aria-label="카드 보기">
      <button type="button" className={value === "grid" ? "active" : ""} aria-pressed={value === "grid"} title="그리드" onClick={() => onChange("grid")}>
        <Icon name="tiles" size={14} />
      </button>
      <button type="button" className={value === "rows" ? "active" : ""} aria-pressed={value === "rows"} title="가로형" onClick={() => onChange("rows")}>
        <Icon name="rows" size={14} />
      </button>
    </div>
  );
}
