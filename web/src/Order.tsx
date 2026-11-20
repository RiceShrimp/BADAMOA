import { DndContext, type DragEndEvent, MouseSensor, TouchSensor, closestCenter, useSensor, useSensors } from "@dnd-kit/core";
import { SortableContext, arrayMove, rectSortingStrategy, useSortable, verticalListSortingStrategy } from "@dnd-kit/sortable";
import { CSS } from "@dnd-kit/utilities";
import { type MouseEvent as ReactMouseEvent, type ReactNode, type TouchEvent as ReactTouchEvent, useMemo, useSyncExternalStore } from "react";
import { api, getToken } from "./api";

/*
 * Cards and rows a person drags into their own order (datasets, 검수, 학습, 모델, 노드 …). Each screen's order is kept
 * on the Hub per person, so it is the same on any PC; one person's order changes nothing for anyone else.
 */

// ---- every screen's order for whoever is signed in: loaded once, kept up as it changes ----

type Orders = Record<string, string[]>;
let orders: Orders | null = null;
let loadedFor: string | null = null; // the sign-in the orders are of
const listeners = new Set<() => void>();

function emit() {
  for (const listener of listeners) listener();
}

function load() {
  const token = getToken();
  if (loadedFor === token && orders) return;
  if (loadedFor !== token) orders = null;
  loadedFor = token;
  api<Orders>("/me/orders")
    .then((got) => {
      if (loadedFor === token) {
        orders = got;
        emit();
      }
    })
    .catch(() => {
      if (loadedFor === token) {
        orders = {}; // the screens' own order; dragging still works for this visit
        emit();
      }
    });
}

function subscribe(listener: () => void) {
  listeners.add(listener);
  load();
  return () => listeners.delete(listener);
}

function save(key: string, ids: string[]) {
  const next = { ...(orders ?? {}) };
  if (ids.length) next[key] = ids;
  else delete next[key];
  orders = next;
  emit();
  void api(`/me/orders/${encodeURIComponent(key)}`, { method: "PUT", body: { ids } }).catch(() => {
    /* kept for this visit only */
  });
}

/** ``items`` with the ones dragged before in their dragged order, the ones never placed after them in the screen's order. */
function arrange<T>(items: T[], idOf: (item: T) => string, saved: string[] | undefined): T[] {
  if (!saved?.length) return items;
  const rank = new Map(saved.map((id, i) => [id, i]));
  const placed = items.filter((it) => rank.has(idOf(it))).sort((a, b) => (rank.get(idOf(a)) ?? 0) - (rank.get(idOf(b)) ?? 0));
  return [...placed, ...items.filter((it) => !rank.has(idOf(it)))];
}

export interface Order<T> {
  ordered: T[];
  custom: boolean; // this person has dragged this screen's cards
  /** The cards on show (maybe a folder's or a search's) in their new order; ``base`` is the whole list as it is shown. */
  move: (shownIds: string[], base?: string[]) => void;
  reset: () => void;
}

/** The order this person dragged screen ``key``'s cards into. */
export function useOrder<T>(key: string, items: T[] | null | undefined, idOf: (item: T) => string): Order<T> {
  const all = useSyncExternalStore(subscribe, () => orders);
  const saved = all?.[key];
  // idOf is the same function each time in practice: the order changes with the items and the saved ids
  const ordered = useMemo(() => arrange(items ?? [], idOf, saved), [items, saved]);
  return {
    ordered,
    custom: !!saved?.length,
    move: (shownIds, base = ordered.map(idOf)) => {
      // Only the cards on show moved: they take each other's places, the hidden ones stay where they are.
      const shown = new Set(shownIds);
      let i = 0;
      save(
        key,
        base.map((id) => (shown.has(id) ? shownIds[i++] : id)),
      );
    },
    reset: () => save(key, []),
  };
}

/** "원래 순서로": shown once this person has dragged the screen's cards. */
export function OrderReset({ order, onReset }: { order: { custom: boolean; reset: () => void }; onReset?: () => void }) {
  if (!order.custom) return null;
  return (
    <button
      className="link small order-reset"
      onClick={() => {
        order.reset();
        onReset?.();
      }}
      title="끌어서 바꾼 순서를 지우고 이 화면의 원래 순서로 되돌립니다"
    >
      원래 순서로
    </button>
  );
}

// ---- dragging ----------------------------------------------------------------------------

// A drag that ends over a link must not also open it: the click that follows the release is dropped.
let dragging = false;
let releasedAt = 0;
const grabbed = () => {
  dragging = true;
};
const released = () => {
  dragging = false;
  releasedAt = Date.now();
};
if (typeof window !== "undefined") {
  window.addEventListener(
    "click",
    (e) => {
      if (dragging || Date.now() - releasedAt < 250) {
        e.preventDefault();
        e.stopPropagation();
      }
    },
    true,
  );
}

// Typing, picking and the like in a card are not the start of a drag.
const NOT_A_HANDLE = "input, textarea, select, [contenteditable], [data-no-drag]";
const handle = (target: EventTarget | null) => !(target instanceof Element && target.closest(NOT_A_HANDLE));

class CardMouse extends MouseSensor {
  static activators = [
    {
      eventName: "onMouseDown" as const,
      handler: ({ nativeEvent: e }: ReactMouseEvent) => e.button === 0 && handle(e.target),
    },
  ];
}

class CardTouch extends TouchSensor {
  static activators = [
    {
      eventName: "onTouchStart" as const,
      handler: ({ nativeEvent: e }: ReactTouchEvent) => handle(e.target),
    },
  ];
}

/**
 * Cards (or rows, with ``list``) that are dragged with the mouse to another place: moved some way with the button down,
 * so a plain click still opens a card; on a touch screen, held a moment first, so the page still scrolls.
 */
export function Sortable({
  ids,
  onMove,
  list = false,
  disabled = false,
  children,
}: {
  ids: string[];
  onMove: (ids: string[]) => void;
  list?: boolean;
  disabled?: boolean;
  children: ReactNode;
}) {
  const sensors = useSensors(
    useSensor(CardMouse, { activationConstraint: { distance: 6 } }),
    useSensor(CardTouch, { activationConstraint: { delay: 250, tolerance: 8 } }),
  );
  const end = ({ active, over }: DragEndEvent) => {
    released();
    if (!over || active.id === over.id) return;
    const from = ids.indexOf(String(active.id));
    const to = ids.indexOf(String(over.id));
    if (from >= 0 && to >= 0) onMove(arrayMove(ids, from, to));
  };
  return (
    <DndContext sensors={sensors} collisionDetection={closestCenter} onDragStart={grabbed} onDragEnd={end} onDragCancel={released}>
      <SortableContext items={ids} strategy={list ? verticalListSortingStrategy : rectSortingStrategy} disabled={disabled}>
        {children}
      </SortableContext>
    </DndContext>
  );
}

/** One card or row of a ``Sortable``: the whole of it is what is dragged. */
export function SortItem({ id, children }: { id: string; children: ReactNode }) {
  const { listeners, setNodeRef, transform, transition, isDragging } = useSortable({ id });
  return (
    <div
      ref={setNodeRef}
      className={`sort-item${isDragging ? " dragging" : ""}`}
      style={{ transform: CSS.Translate.toString(transform), transition }}
      {...listeners}
      onDragStart={(e) => e.preventDefault()} // not the browser's own dragging of a link or a picture
    >
      {children}
    </div>
  );
}
