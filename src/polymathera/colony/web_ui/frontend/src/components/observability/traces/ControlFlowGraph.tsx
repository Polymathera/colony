/** SVG renderer for the algorithmic control flow graph. */

import { useState, useCallback, useRef } from "react";
import type { LayoutNode, LayoutEdge } from "./useGraphLayout";
import type { CodegenIteration } from "./types";

function buildEdgePath(edge: LayoutEdge): string {
  if (!edge.sections || edge.sections.length === 0) return "";
  const parts: string[] = [];
  for (const sec of edge.sections) {
    parts.push(`M ${sec.startPoint.x} ${sec.startPoint.y}`);
    if (sec.bendPoints) {
      for (const bp of sec.bendPoints) parts.push(`L ${bp.x} ${bp.y}`);
    }
    parts.push(`L ${sec.endPoint.x} ${sec.endPoint.y}`);
  }
  return parts.join(" ");
}

/** Get the first 3 non-blank, non-comment lines as a preview. */
function codePreview(code: string, maxLines = 3): string {
  const lines = code
    .split("\n")
    .filter((l) => l.trim() && !l.trim().startsWith("#"))
    .slice(0, maxLines);
  return lines.join("\n") || "(empty)";
}

export function ControlFlowGraph({
  layoutNodes,
  layoutEdges,
  graphWidth,
  graphHeight,
  iterations,
  edges,
  selectedNodeId,
  onSelectNode,
}: {
  layoutNodes: LayoutNode[];
  layoutEdges: LayoutEdge[];
  graphWidth: number;
  graphHeight: number;
  iterations: CodegenIteration[];
  edges: Array<{ source: string; target: string; success: boolean }>;
  selectedNodeId: string | null;
  onSelectNode: (nodeId: string) => void;
}) {
  const svgRef = useRef<SVGSVGElement>(null);
  const [pan, setPan] = useState({ x: 0, y: 0 });
  const [zoom, setZoom] = useState(1);
  const [dragging, setDragging] = useState(false);
  const dragStart = useRef({ x: 0, y: 0, panX: 0, panY: 0 });

  const iterMap = new Map(iterations.map((it) => [`n${it.step_index}`, it]));

  const handleWheel = useCallback((e: React.WheelEvent) => {
    e.preventDefault();
    const factor = e.deltaY > 0 ? 0.9 : 1.1;
    setZoom((z) => Math.max(0.2, Math.min(5, z * factor)));
  }, []);

  const handleMouseDown = useCallback(
    (e: React.MouseEvent) => {
      if (e.target === svgRef.current || (e.target as Element).tagName === "svg") {
        setDragging(true);
        dragStart.current = { x: e.clientX, y: e.clientY, panX: pan.x, panY: pan.y };
      }
    },
    [pan],
  );

  const handleMouseMove = useCallback(
    (e: React.MouseEvent) => {
      if (!dragging) return;
      const dx = e.clientX - dragStart.current.x;
      const dy = e.clientY - dragStart.current.y;
      setPan({ x: dragStart.current.panX + dx, y: dragStart.current.panY + dy });
    },
    [dragging],
  );

  const handleMouseUp = useCallback(() => setDragging(false), []);

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
        <marker id="cf-arrow-ok" viewBox="0 0 10 6" refX="10" refY="3" markerWidth="8" markerHeight="6" orient="auto-start-reverse">
          <path d="M 0 0 L 10 3 L 0 6 z" fill="#10b981" />
        </marker>
        <marker id="cf-arrow-err" viewBox="0 0 10 6" refX="10" refY="3" markerWidth="8" markerHeight="6" orient="auto-start-reverse">
          <path d="M 0 0 L 10 3 L 0 6 z" fill="#ef4444" />
        </marker>
      </defs>

      <g transform={`translate(${pan.x},${pan.y}) scale(${zoom})`}>
        {/* Edges */}
        {layoutEdges.map((edge, i) => {
          const path = buildEdgePath(edge);
          if (!path) return null;
          const edgeData = edges[i];
          const color = edgeData?.success ? "#10b981" : "#ef4444";
          const markerId = edgeData?.success ? "cf-arrow-ok" : "cf-arrow-err";

          return (
            <g key={edge.id}>
              <path d={path} fill="none" stroke={color} strokeWidth={1.5} strokeOpacity={0.5} markerEnd={`url(#${markerId})`} />
              <path d={path} fill="none" stroke="transparent" strokeWidth={12} />
            </g>
          );
        })}

        {/* Code nodes */}
        {layoutNodes.map((node) => {
          const iter = iterMap.get(node.id);
          if (!iter) return null;

          const isSelected = selectedNodeId === node.id;
          const success = iter.success === true;
          const failed = iter.success === false;
          const borderColor = success ? "#10b981" : failed ? "#ef4444" : "#6b7280";

          return (
            <g
              key={node.id}
              transform={`translate(${node.x},${node.y})`}
              className="cursor-pointer"
              onClick={() => onSelectNode(node.id)}
            >
              <rect
                width={node.width}
                height={node.height}
                rx={6}
                fill={isSelected ? `${borderColor}15` : "hsl(222 47% 7%)"}
                stroke={borderColor}
                strokeWidth={isSelected ? 2 : 1}
                strokeOpacity={isSelected ? 1 : 0.5}
              />

              {/* Step number + status */}
              <text x={8} y={16} className="text-[10px] font-bold" fill={borderColor}>
                #{iter.step_index + 1} {success ? "\u2713" : failed ? "\u2717" : "..."}
              </text>

              {/* Mode badge */}
              <text x={node.width - 8} y={16} textAnchor="end" className="text-[9px]" fill="hsl(var(--muted-foreground))">
                {iter.mode?.toUpperCase()}
              </text>

              {/* Code preview */}
              <foreignObject x={6} y={22} width={node.width - 12} height={node.height - 28}>
                <pre
                  style={{
                    fontSize: "9px",
                    lineHeight: "1.3",
                    fontFamily: "monospace",
                    color: "hsl(var(--muted-foreground))",
                    overflow: "hidden",
                    margin: 0,
                    whiteSpace: "pre-wrap",
                    wordBreak: "break-all",
                  }}
                >
                  {codePreview(iter.generated_code)}
                </pre>
              </foreignObject>
            </g>
          );
        })}
      </g>
    </svg>
  );
}
