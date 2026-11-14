import { api } from "./api";
import { usePolling } from "./ui";

interface LineageNode {
  id: string;
  kind: "release" | "run" | "model" | "eval" | "export" | "predict" | "sweep";
  label: string;
  sub: string;
  href: string | null;
  status?: string;
  column: number;
  current: boolean;
}

interface Lineage {
  nodes: LineageNode[];
  edges: { from: string; to: string; label: string }[];
}

const KIND_LABEL: Record<LineageNode["kind"], string> = {
  release: "릴리스",
  run: "학습 실행",
  model: "모델",
  eval: "평가",
  export: "내보내기",
  predict: "테스트",
  sweep: "탐색",
};

const W = 196;
const H = 64;
const GAP_X = 64;
const GAP_Y = 14;
const PAD = 8;

/** Roughly how wide a text is: Hangul and other wide letters take about a full em, the rest about half. */
function fit(text: string, size: number, room: number): string {
  let width = 0;
  for (let i = 0; i < text.length; i++) {
    width += (text.charCodeAt(i) > 0x2e80 ? 1 : 0.58) * size;
    if (width > room) return `${text.slice(0, Math.max(1, i - 1))}…`;
  }
  return text;
}

/**
 * Where a model or a run came from and what came of it: release → run → model → evaluations,
 * exports, tests, and the models trained further from it. Each box links to its page.
 */
export function LineageGraph({ path }: { path: string }) {
  const { data } = usePolling(() => api<Lineage>(path), 30000, [path]);
  if (!data) return <p className="muted small">계보를 불러오는 중…</p>;

  const columns = new Map<number, LineageNode[]>();
  for (const node of data.nodes) columns.set(node.column, [...(columns.get(node.column) ?? []), node]);
  const tallest = Math.max(...[...columns.values()].map((c) => c.length));
  const height = PAD * 2 + tallest * H + (tallest - 1) * GAP_Y;
  const width = PAD * 2 + columns.size * W + (columns.size - 1) * GAP_X;
  const place = new Map<string, { x: number; y: number }>();
  for (const [column, nodes] of columns) {
    const own = nodes.length * H + (nodes.length - 1) * GAP_Y;
    nodes.forEach((node, row) => {
      place.set(node.id, { x: PAD + column * (W + GAP_X), y: PAD + (height - PAD * 2 - own) / 2 + row * (H + GAP_Y) });
    });
  }

  return (
    <div className="lineage-graph">
      <svg width={width} height={height} viewBox={`0 0 ${width} ${height}`} role="img" aria-label="계보">
        <defs>
          <marker id="lineage-arrow" viewBox="0 0 8 8" refX="7" refY="4" markerWidth="7" markerHeight="7" orient="auto">
            <path d="M0,0 L8,4 L0,8 z" className="lineage-arrow" />
          </marker>
        </defs>
        {data.edges.map((edge) => {
          const a = place.get(edge.from);
          const b = place.get(edge.to);
          if (!a || !b) return null;
          const x1 = a.x + W;
          const y1 = a.y + H / 2;
          const x2 = b.x - 2;
          const y2 = b.y + H / 2;
          const bend = Math.max(24, (x2 - x1) / 2);
          return (
            <g key={`${edge.from}-${edge.to}`}>
              <path
                d={`M${x1},${y1} C${x1 + bend},${y1} ${x2 - bend},${y2} ${x2},${y2}`}
                className="lineage-edge"
                markerEnd="url(#lineage-arrow)"
              />
              {edge.label && (
                <text x={(x1 + x2) / 2} y={(y1 + y2) / 2 - 5} textAnchor="middle" className="lineage-edge-label">
                  {edge.label}
                </text>
              )}
            </g>
          );
        })}
        {data.nodes.map((node) => {
          const at = place.get(node.id);
          if (!at) return null;
          const box = (
            <g transform={`translate(${at.x},${at.y})`}>
              <title>
                {KIND_LABEL[node.kind]}: {node.label} — {node.sub}
              </title>
              <rect width={W} height={H} rx={9} className={`lineage-box kind-${node.kind} ${node.current ? "current" : ""}`} />
              <text x={12} y={18} className="lineage-kind">
                {KIND_LABEL[node.kind]}
                {node.status && ["failed", "canceled", "running", "queued"].includes(node.status)
                  ? ` · ${{ failed: "실패", canceled: "중단", running: "실행 중", queued: "대기" }[node.status]}`
                  : ""}
              </text>
              <text x={12} y={37} className="lineage-label">
                {fit(node.label, 13, W - 22)}
              </text>
              <text x={12} y={54} className="lineage-sub">
                {fit(node.sub, 11.5, W - 22)}
              </text>
            </g>
          );
          return node.href && !node.current ? (
            <a key={node.id} href={node.href}>
              {box}
            </a>
          ) : (
            <g key={node.id}>{box}</g>
          );
        })}
      </svg>
    </div>
  );
}
