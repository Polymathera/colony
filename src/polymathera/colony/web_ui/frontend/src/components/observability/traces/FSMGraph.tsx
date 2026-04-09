/** SVG renderer for FSM state graphs with zoom/pan and draggable nodes. */

import { useState, useCallback, useRef } from "react";
import type { LayoutNode, LayoutEdge } from "./useGraphLayout";
import type { FSMState, FSMTransition, FSMLoop } from "./types";

/* ── Edge path builder ──────────────────────────────────────── */

function buildEdgePath(edge: LayoutEdge): string {
  if (!edge.sections || edge.sections.length === 0) {
    return "";
  }
  const parts: string[] = [];
  for (const sec of edge.sections) {
    parts.push(`M ${sec.startPoint.x} ${sec.startPoint.y}`);
    if (sec.bendPoints) {
      for (const bp of sec.bendPoints) {
        parts.push(`L ${bp.x} ${bp.y}`);
      }
    }
    parts.push(`L ${sec.endPoint.x} ${sec.endPoint.y}`);
  }
  return parts.join(" ");
}

/* ── State colors ───────────────────────────────────────────── */

function stateColor(label: string, hasLoop: boolean): string {
  if (hasLoop) return "#f97316"; // orange for loop states
  if (label.startsWith("PLANNING")) return "#f59e0b"; // amber
  return "#3b82f6"; // blue for execution
}

/* ── Component ──────────────────────────────────────────────── */

export function FSMGraph({
  layoutNodes,
  layoutEdges,
  graphWidth,
  graphHeight,
  states,
  transitions,
  loops,
  selectedId,
  onSelectState,
  onSelectTransition,
}: {
  layoutNodes: LayoutNode[];
  layoutEdges: LayoutEdge[];
  graphWidth: number;
  graphHeight: number;
  states: FSMState[];
  transitions: FSMTransition[];
  loops: FSMLoop[];
  selectedId: string | null;
  onSelectState: (stateId: string) => void;
  onSelectTransition: (stepIndex: number) => void;
}) {
  const svgRef = useRef<SVGSVGElement>(null);
  const [pan, setPan] = useState({ x: 0, y: 0 });
  const [zoom, setZoom] = useState(1);
  const [dragging, setDragging] = useState(false);
  const dragStart = useRef({ x: 0, y: 0, panX: 0, panY: 0 });

  // Position overrides for dragged nodes
  const [posOverrides, setPosOverrides] = useState<Record<string, { x: number; y: number }>>({});
  const [draggedNodeId, setDraggedNodeId] = useState<string | null>(null);
  const nodeStart = useRef({ x: 0, y: 0, nodeX: 0, nodeY: 0 });

  // Loop state IDs for highlighting
  const loopStateIds = new Set(loops.map((l) => l.state_id));

  // Build state/transition lookup
  const stateMap = new Map(states.map((s) => [s.state_id, s]));

  const handleWheel = useCallback((e: React.WheelEvent) => {
    e.preventDefault();
    const factor = e.deltaY > 0 ? 0.9 : 1.1;
    setZoom((z) => Math.max(0.2, Math.min(5, z * factor)));
  }, []);

  const handleMouseDown = useCallback((e: React.MouseEvent) => {
    if (e.target === svgRef.current || (e.target as Element).tagName === "svg") {
      setDragging(true);
      dragStart.current = { x: e.clientX, y: e.clientY, panX: pan.x, panY: pan.y };
    }
  }, [pan]);

  const handleMouseMove = useCallback(
    (e: React.MouseEvent) => {
      if (draggedNodeId) {
        const dx = (e.clientX - nodeStart.current.x) / zoom;
        const dy = (e.clientY - nodeStart.current.y) / zoom;
        setPosOverrides((prev) => ({
          ...prev,
          [draggedNodeId]: {
            x: nodeStart.current.nodeX + dx,
            y: nodeStart.current.nodeY + dy,
          },
        }));
        return;
      }
      if (!dragging) return;
      const dx = e.clientX - dragStart.current.x;
      const dy = e.clientY - dragStart.current.y;
      setPan({ x: dragStart.current.panX + dx, y: dragStart.current.panY + dy });
    },
    [dragging, draggedNodeId, zoom],
  );

  const handleMouseUp = useCallback(() => {
    setDragging(false);
    setDraggedNodeId(null);
  }, []);

  const handleNodeMouseDown = useCallback(
    (e: React.MouseEvent, nodeId: string, x: number, y: number) => {
      e.stopPropagation();
      setDraggedNodeId(nodeId);
      nodeStart.current = { x: e.clientX, y: e.clientY, nodeX: x, nodeY: y };
    },
    [],
  );

  const padding = 40;
  const viewBox = `${-padding} ${-padding} ${graphWidth + padding * 2} ${graphHeight + padding * 2}`;

  return (
    <svg
      ref={svgRef}
      className="w-full h-full cursor-grab active:cursor-grabbing"
      viewBox={viewBox}
      onWheel={handleWheel}
      onMouseDown={handleMouseDown}
      onMouseMove={handleMouseMove}
      onMouseUp={handleMouseUp}
      onMouseLeave={handleMouseUp}
    >
      <defs>
        <marker
          id="arrow"
          viewBox="0 0 10 6"
          refX="10"
          refY="3"
          markerWidth="8"
          markerHeight="6"
          orient="auto-start-reverse"
        >
          <path d="M 0 0 L 10 3 L 0 6 z" fill="currentColor" />
        </marker>
        <marker
          id="arrow-success"
          viewBox="0 0 10 6"
          refX="10"
          refY="3"
          markerWidth="8"
          markerHeight="6"
          orient="auto-start-reverse"
        >
          <path d="M 0 0 L 10 3 L 0 6 z" fill="#10b981" />
        </marker>
        <marker
          id="arrow-error"
          viewBox="0 0 10 6"
          refX="10"
          refY="3"
          markerWidth="8"
          markerHeight="6"
          orient="auto-start-reverse"
        >
          <path d="M 0 0 L 10 3 L 0 6 z" fill="#ef4444" />
        </marker>
      </defs>

      <g transform={`translate(${pan.x},${pan.y}) scale(${zoom})`}>
        {/* Edges */}
        {layoutEdges.map((edge, i) => {
          const transition = transitions[i];
          const path = buildEdgePath(edge);
          if (!path) return null;
          const isSuccess = transition?.success;
          const color = isSuccess ? "#10b981" : "#ef4444";
          const markerId = isSuccess ? "arrow-success" : "arrow-error";

          return (
            <g
              key={edge.id}
              className="cursor-pointer"
              onClick={() => transition && onSelectTransition(transition.step_index)}
            >
              <path
                d={path}
                fill="none"
                stroke={color}
                strokeWidth={1.5}
                strokeOpacity={0.6}
                markerEnd={`url(#${markerId})`}
              />
              {/* Hover target (wider invisible path) */}
              <path
                d={path}
                fill="none"
                stroke="transparent"
                strokeWidth={12}
              />
            </g>
          );
        })}

        {/* Nodes */}
        {layoutNodes.map((node) => {
          const state = stateMap.get(node.id);
          if (!state) return null;

          const hasLoop = loopStateIds.has(node.id);
          const color = stateColor(state.label, hasLoop);
          const isSelected = selectedId === node.id;
          const pos = posOverrides[node.id] ?? { x: node.x, y: node.y };

          return (
            <g
              key={node.id}
              transform={`translate(${pos.x},${pos.y})`}
              className="cursor-pointer"
              onClick={() => onSelectState(node.id)}
              onMouseDown={(e) => handleNodeMouseDown(e, node.id, pos.x, pos.y)}
            >
              <rect
                width={node.width}
                height={node.height}
                rx={8}
                fill={`${color}15`}
                stroke={isSelected ? color : `${color}50`}
                strokeWidth={isSelected ? 2 : 1}
              />
              {/* State label */}
              <text
                x={node.width / 2}
                y={22}
                textAnchor="middle"
                className="text-[11px] font-semibold"
                fill={color}
              >
                {state.label}
              </text>
              {/* Visit count */}
              <text
                x={node.width / 2}
                y={40}
                textAnchor="middle"
                className="text-[10px]"
                fill="hsl(var(--muted-foreground))"
              >
                {state.visit_count} visit{state.visit_count !== 1 ? "s" : ""}
              </text>
              {/* Loop badge */}
              {hasLoop && (
                <text
                  x={node.width / 2}
                  y={54}
                  textAnchor="middle"
                  className="text-[9px] font-bold"
                  fill="#f97316"
                >
                  LOOP
                </text>
              )}
            </g>
          );
        })}
      </g>
    </svg>
  );
}
