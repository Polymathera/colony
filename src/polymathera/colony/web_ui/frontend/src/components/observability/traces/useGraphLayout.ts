/** Hook for ELK-based directed graph layout computation. */

import { useState, useEffect, useRef } from "react";
import ELK, { type ElkNode, type ElkExtendedEdge } from "elkjs/lib/elk.bundled.js";

export interface LayoutNode {
  id: string;
  x: number;
  y: number;
  width: number;
  height: number;
}

export interface LayoutEdge {
  id: string;
  source: string;
  target: string;
  sections?: Array<{
    startPoint: { x: number; y: number };
    endPoint: { x: number; y: number };
    bendPoints?: Array<{ x: number; y: number }>;
  }>;
}

export interface GraphLayoutResult {
  nodes: LayoutNode[];
  edges: LayoutEdge[];
  width: number;
  height: number;
  ready: boolean;
}

interface GraphInput {
  nodes: Array<{ id: string; width?: number; height?: number }>;
  edges: Array<{ id: string; source: string; target: string }>;
}

const elk = new ELK();

export function useGraphLayout(input: GraphInput | null): GraphLayoutResult {
  const [result, setResult] = useState<GraphLayoutResult>({
    nodes: [],
    edges: [],
    width: 0,
    height: 0,
    ready: false,
  });
  const requestId = useRef(0);

  useEffect(() => {
    if (!input || input.nodes.length === 0) {
      setResult({ nodes: [], edges: [], width: 0, height: 0, ready: true });
      return;
    }

    const thisRequest = ++requestId.current;

    const elkGraph: ElkNode = {
      id: "root",
      layoutOptions: {
        "elk.algorithm": "layered",
        "elk.direction": "DOWN",
        "elk.spacing.nodeNode": "40",
        "elk.layered.spacing.nodeNodeBetweenLayers": "60",
        "elk.edgeRouting": "ORTHOGONAL",
        "elk.layered.crossingMinimization.strategy": "LAYER_SWEEP",
      },
      children: input.nodes.map((n) => ({
        id: n.id,
        width: n.width ?? 180,
        height: n.height ?? 60,
      })),
      edges: input.edges.map((e) => ({
        id: e.id,
        sources: [e.source],
        targets: [e.target],
      })),
    };

    elk
      .layout(elkGraph)
      .then((laid) => {
        // Discard stale results
        if (thisRequest !== requestId.current) return;

        const nodes: LayoutNode[] = (laid.children ?? []).map((c) => ({
          id: c.id,
          x: c.x ?? 0,
          y: c.y ?? 0,
          width: c.width ?? 180,
          height: c.height ?? 60,
        }));

        const edges: LayoutEdge[] = ((laid.edges ?? []) as ElkExtendedEdge[]).map(
          (e) => ({
            id: e.id,
            source: e.sources[0],
            target: e.targets[0],
            sections: e.sections?.map((s) => ({
              startPoint: s.startPoint,
              endPoint: s.endPoint,
              bendPoints: s.bendPoints,
            })),
          }),
        );

        setResult({
          nodes,
          edges,
          width: laid.width ?? 0,
          height: laid.height ?? 0,
          ready: true,
        });
      })
      .catch((err) => {
        console.error("ELK layout failed:", err);
        if (thisRequest === requestId.current) {
          setResult({ nodes: [], edges: [], width: 0, height: 0, ready: true });
        }
      });
  }, [input?.nodes.length, input?.edges.length]);

  return result;
}
