import { useState, useMemo, useRef, useEffect, useCallback } from "react";
import { Canvas, useThree, ThreeEvent } from "@react-three/fiber";
import { OrbitControls, Html } from "@react-three/drei";
import * as THREE from "three";
import { usePageGraphScopes, usePageGraph } from "@/api/hooks/usePageGraph";
import { useVCMPages, useLoadedPageEntries } from "@/api/hooks/useVCM";
import { MetricCard } from "../shared/MetricCard";
import { Badge } from "../shared/Badge";
import type {
  PageGraphNode,
  PageGraphEdge,
  PageGraphScope,
  PageSummary,
} from "@/api/types";

/* ── Helpers ────────────────────────────────────────────────── */

/** Compute degree (in+out) for each node from edge list. */
function computeDegreeMap(edges: PageGraphEdge[]): Map<string, number> {
  const deg = new Map<string, number>();
  for (const e of edges) {
    deg.set(e.source, (deg.get(e.source) ?? 0) + 1);
    deg.set(e.target, (deg.get(e.target) ?? 0) + 1);
  }
  return deg;
}

/** Map a 0-1 value to a color gradient: blue(0) → cyan(0.3) → green(0.5) → yellow(0.7) → red(1). */
function heatColor(t: number, color: THREE.Color): THREE.Color {
  // HSL: hue from 240° (blue) down to 0° (red)
  const hue = (1 - t) * 0.66; // 0.66=blue, 0=red
  const sat = 0.85;
  const lit = 0.45 + t * 0.15; // brighter for hot nodes
  color.setHSL(hue, sat, lit);
  return color;
}

/* ── 3D Graph Nodes (InstancedMesh) ─────────────────────────── */

function GraphNodes({
  nodes,
  degreeMap,
  maxDegree,
  sizeScale,
  sizeSpread,
  onNodeClick,
  selectedId,
}: {
  nodes: PageGraphNode[];
  degreeMap: Map<string, number>;
  maxDegree: number;
  sizeScale: number;
  sizeSpread: number;
  onNodeClick: (id: string) => void;
  selectedId: string | null;
}) {
  const meshRef = useRef<THREE.InstancedMesh>(null);
  const tempObj = useMemo(() => new THREE.Object3D(), []);

  const nodeIds = useMemo(() => nodes.map((n) => n.id), [nodes]);

  useEffect(() => {
    const mesh = meshRef.current;
    if (!mesh) return;

    const color = new THREE.Color();
    const logMax = Math.log(maxDegree + 1);

    for (let i = 0; i < nodes.length; i++) {
      const n = nodes[i];
      const deg = degreeMap.get(n.id) ?? 0;

      // Normalized degree (log scale for better distribution)
      const t = logMax > 0 ? Math.log(deg + 1) / logMax : 0;

      // Size: uniform base scaled by slider, degree adds proportional boost
      // sizeSpread=0 → all same size; sizeSpread=1 → max degree 6x bigger
      const baseRadius = 0.012 * sizeScale;
      const degreeBoost = 1 + sizeSpread * Math.pow(t, 0.6) * 5;
      const scale = baseRadius * degreeBoost;

      tempObj.position.set(n.x, n.y, n.z);
      tempObj.scale.setScalar(selectedId === n.id ? scale * 2.5 : scale);
      tempObj.updateMatrix();
      mesh.setMatrixAt(i, tempObj.matrix);

      heatColor(t, color);
      mesh.setColorAt(i, color);
    }
    mesh.instanceMatrix.needsUpdate = true;
    if (mesh.instanceColor) mesh.instanceColor.needsUpdate = true;
  }, [nodes, degreeMap, maxDegree, sizeScale, sizeSpread, tempObj, selectedId]);

  const handleClick = useCallback(
    (e: ThreeEvent<MouseEvent>) => {
      e.stopPropagation();
      if (e.instanceId !== undefined && e.instanceId < nodeIds.length) {
        onNodeClick(nodeIds[e.instanceId]);
      }
    },
    [nodeIds, onNodeClick],
  );

  if (nodes.length === 0) return null;

  return (
    <instancedMesh
      ref={meshRef}
      args={[undefined, undefined, nodes.length]}
      onClick={handleClick}
    >
      <sphereGeometry args={[1, 12, 8]} />
      <meshBasicMaterial />
    </instancedMesh>
  );
}

/* ── 3D Graph Edges (Line segments) ─────────────────────────── */

function GraphEdges({
  edges,
  nodePositions,
  opacity,
}: {
  edges: PageGraphEdge[];
  nodePositions: Map<string, [number, number, number]>;
  opacity: number;
}) {
  const geometry = useMemo(() => {
    const positions: number[] = [];

    for (const edge of edges) {
      const from = nodePositions.get(edge.source);
      const to = nodePositions.get(edge.target);
      if (!from || !to) continue;
      positions.push(from[0], from[1], from[2], to[0], to[1], to[2]);
    }

    const geo = new THREE.BufferGeometry();
    geo.setAttribute(
      "position",
      new THREE.Float32BufferAttribute(positions, 3),
    );
    return geo;
  }, [edges, nodePositions]);

  return (
    <lineSegments geometry={geometry}>
      <lineBasicMaterial color={0x334155} transparent opacity={opacity} />
    </lineSegments>
  );
}

/* ── Camera auto-fit ────────────────────────────────────────── */

function AutoFitCamera({ nodes }: { nodes: PageGraphNode[] }) {
  const { camera } = useThree();

  useEffect(() => {
    if (nodes.length === 0) return;
    let maxDist = 0;
    for (const n of nodes) {
      const d = Math.sqrt(n.x * n.x + n.y * n.y + n.z * n.z);
      if (d > maxDist) maxDist = d;
    }
    const dist = Math.max(maxDist * 2.5, 1);
    camera.position.set(dist * 0.7, dist * 0.5, dist);
    camera.lookAt(0, 0, 0);
  }, [nodes, camera]);

  return null;
}

/* ── Hover label ────────────────────────────────────────────── */

function HoverLabel({
  nodes,
  degreeMap,
  pageMap,
  loadedSet,
}: {
  nodes: PageGraphNode[];
  degreeMap: Map<string, number>;
  pageMap: Map<string, PageSummary>;
  loadedSet: Set<string>;
}) {
  const [hovered, setHovered] = useState<{
    id: string;
    pos: [number, number, number];
  } | null>(null);
  const { raycaster, camera, scene } = useThree();

  useEffect(() => {
    const canvas = document.querySelector("canvas");
    if (!canvas) return;

    const onMove = (e: PointerEvent) => {
      const rect = canvas.getBoundingClientRect();
      const mouse = new THREE.Vector2(
        ((e.clientX - rect.left) / rect.width) * 2 - 1,
        -((e.clientY - rect.top) / rect.height) * 2 + 1,
      );
      raycaster.setFromCamera(mouse, camera);
      const meshes = scene.children.filter(
        (c) => c instanceof THREE.InstancedMesh,
      );
      for (const mesh of meshes) {
        const hits = raycaster.intersectObject(mesh);
        if (hits.length > 0 && hits[0].instanceId !== undefined) {
          const node = nodes[hits[0].instanceId];
          if (node) {
            setHovered({ id: node.id, pos: [node.x, node.y, node.z] });
            canvas.style.cursor = "pointer";
            return;
          }
        }
      }
      setHovered(null);
      canvas.style.cursor = "default";
    };

    canvas.addEventListener("pointermove", onMove);
    return () => canvas.removeEventListener("pointermove", onMove);
  }, [nodes, raycaster, camera, scene]);

  if (!hovered) return null;

  const page = pageMap.get(hovered.id);
  const deg = degreeMap.get(hovered.id) ?? 0;
  return (
    <Html position={hovered.pos} center style={{ pointerEvents: "none" }}>
      <div className="rounded bg-zinc-900/90 px-2 py-1 text-[10px] text-zinc-200 whitespace-nowrap border border-zinc-700">
        <div className="font-semibold">{hovered.id.slice(0, 24)}...</div>
        <div>Degree: {deg}</div>
        {page && (
          <>
            <div>
              {page.tokens} tokens &middot; {page.source || "unknown"}
            </div>
            <div>{loadedSet.has(hovered.id) ? "Loaded" : "Not loaded"}</div>
          </>
        )}
      </div>
    </Html>
  );
}

/* ── Group Selector ─────────────────────────────────────────── */

function GroupSelector({
  groups,
  selected,
  onSelect,
}: {
  groups: PageGraphScope[];
  selected: PageGraphScope | null;
  onSelect: (g: PageGraphScope | null) => void;
}) {
  const handleChange = useCallback(
    (e: React.ChangeEvent<HTMLSelectElement>) => {
      const val = e.target.value;
      if (!val) {
        onSelect(null);
        return;
      }
      const [tenantId, colonyId] = val.split(":::");
      const found = groups.find(
        (g) => g.tenant_id === tenantId && g.colony_id === colonyId,
      );
      onSelect(found ?? null);
    },
    [groups, onSelect],
  );

  return (
    <select
      className="rounded border border-border bg-background px-2 py-1.5 text-xs font-mono"
      value={
        selected ? `${selected.tenant_id}:::${selected.colony_id}` : ""
      }
      onChange={handleChange}
    >
      <option value="">Select a page graph...</option>
      {groups.map((g) => (
        <option
          key={`${g.tenant_id}:::${g.colony_id}`}
          value={`${g.tenant_id}:::${g.colony_id}`}
        >
          {g.scope_id} ({g.colony_id.slice(0, 16)})
        </option>
      ))}
    </select>
  );
}

/* ── Node Detail Panel ──────────────────────────────────────── */

function NodeDetail({
  nodeId,
  edges,
  degreeMap,
  pageMap,
  loadedSet,
  onClose,
}: {
  nodeId: string;
  edges: PageGraphEdge[];
  degreeMap: Map<string, number>;
  pageMap: Map<string, PageSummary>;
  loadedSet: Set<string>;
  onClose: () => void;
}) {
  const page = pageMap.get(nodeId);
  const connectedEdges = useMemo(
    () => edges.filter((e) => e.source === nodeId || e.target === nodeId),
    [edges, nodeId],
  );

  return (
    <div className="rounded-lg border bg-card p-4 text-xs space-y-2">
      <div className="flex items-center justify-between">
        <span className="font-semibold text-sm">Node Detail</span>
        <button
          className="text-muted-foreground hover:text-foreground"
          onClick={onClose}
        >
          Close
        </button>
      </div>
      <div className="font-mono text-muted-foreground break-all">
        {nodeId}
      </div>
      <div className="flex flex-wrap gap-2">
        <Badge variant={loadedSet.has(nodeId) ? "success" : "default"}>
          {loadedSet.has(nodeId) ? "Loaded" : "Not loaded"}
        </Badge>
        <Badge variant="info">Degree: {degreeMap.get(nodeId) ?? 0}</Badge>
        {page && <Badge variant="info">{page.tokens} tokens</Badge>}
      </div>
      {page?.source && (
        <div>
          <span className="text-muted-foreground">Source:</span> {page.source}
        </div>
      )}
      {page?.files && page.files.length > 0 && (
        <div>
          <span className="text-muted-foreground">Files:</span>
          <div className="mt-1 max-h-20 overflow-auto font-mono text-[10px]">
            {page.files.map((f) => (
              <div key={f}>{f}</div>
            ))}
          </div>
        </div>
      )}
      <div>
        <span className="text-muted-foreground">
          Edges: {connectedEdges.length}
        </span>
        {connectedEdges.length > 0 && (
          <div className="mt-1 max-h-32 overflow-auto space-y-1">
            {connectedEdges.slice(0, 20).map((e, i) => (
              <div key={i} className="flex items-center gap-1 text-[10px]">
                <span className="font-mono">
                  {(e.source === nodeId ? e.target : e.source).slice(0, 16)}
                </span>
                <span className="text-muted-foreground">
                  w={e.weight.toFixed(2)}
                </span>
                {e.relationship_types.map((t) => (
                  <Badge key={t} variant="default">
                    {t}
                  </Badge>
                ))}
              </div>
            ))}
            {connectedEdges.length > 20 && (
              <div className="text-muted-foreground">
                +{connectedEdges.length - 20} more
              </div>
            )}
          </div>
        )}
      </div>
    </div>
  );
}

/* ── Legend ──────────────────────────────────────────────────── */

function Legend() {
  return (
    <div className="absolute bottom-3 left-3 rounded bg-zinc-900/80 px-3 py-2 text-[10px] text-zinc-300 border border-zinc-700 space-y-1">
      <div className="font-semibold mb-1">Node color = degree</div>
      <div className="flex items-center gap-2">
        <span className="inline-block h-2.5 w-8 rounded" style={{
          background: "linear-gradient(to right, hsl(240,85%,50%), hsl(180,85%,50%), hsl(120,85%,52%), hsl(60,85%,55%), hsl(0,85%,58%))",
        }} />
        <span>Low → High</span>
      </div>
      <div className="flex items-center gap-2">
        <span className="inline-block h-0.5 w-3 bg-slate-600" />
        Edge
      </div>
    </div>
  );
}

/* ── Main Component ─────────────────────────────────────────── */

export function PageGraphTab() {
  const { data: groups } = usePageGraphScopes();
  const [selectedGroup, setSelectedGroup] = useState<PageGraphScope | null>(
    null,
  );
  const [selectedNode, setSelectedNode] = useState<string | null>(null);
  const [sizeScale, setSizeScale] = useState(1.0);
  const [sizeSpread, setSizeSpread] = useState(0.5);
  const [edgeOpacity, setEdgeOpacity] = useState(0.12);

  // Auto-select first group
  useEffect(() => {
    if (!selectedGroup && groups && groups.length > 0) {
      setSelectedGroup(groups[0]);
    }
  }, [groups, selectedGroup]);

  const graphData = usePageGraph(
    selectedGroup?.tenant_id ?? "",
    selectedGroup?.colony_id ?? "",
  );

  const pages = useVCMPages();
  const loadedEntries = useLoadedPageEntries();

  const pageMap = useMemo(() => {
    const map = new Map<string, PageSummary>();
    for (const p of pages.data ?? []) {
      map.set(p.page_id, p);
    }
    return map;
  }, [pages.data]);

  const loadedSet = useMemo(() => {
    const set = new Set<string>();
    for (const entry of loadedEntries.data ?? []) {
      set.add(entry.page_id);
    }
    return set;
  }, [loadedEntries.data]);

  const nodePositions = useMemo(() => {
    const map = new Map<string, [number, number, number]>();
    for (const n of graphData.data?.nodes ?? []) {
      map.set(n.id, [n.x, n.y, n.z]);
    }
    return map;
  }, [graphData.data?.nodes]);

  const nodes = graphData.data?.nodes ?? [];
  const edges = graphData.data?.edges ?? [];

  // Degree centrality
  const degreeMap = useMemo(() => computeDegreeMap(edges), [edges]);
  const maxDegree = useMemo(() => {
    let m = 0;
    for (const d of degreeMap.values()) {
      if (d > m) m = d;
    }
    return m;
  }, [degreeMap]);

  return (
    <div className="flex h-full flex-col gap-3">
      {/* Controls */}
      <div className="flex flex-wrap items-center gap-3">
        <GroupSelector
          groups={groups ?? []}
          selected={selectedGroup}
          onSelect={setSelectedGroup}
        />
        <MetricCard
          label="Nodes"
          value={String(graphData.data?.node_count ?? 0)}
        />
        <MetricCard
          label="Edges"
          value={String(graphData.data?.edge_count ?? 0)}
        />
        {maxDegree > 0 && (
          <MetricCard label="Max Degree" value={String(maxDegree)} />
        )}

        {/* Visualization controls */}
        <div className="flex items-center gap-4">
          <div className="flex items-center gap-1.5">
            <label className="text-[10px] text-muted-foreground whitespace-nowrap">
              Size
            </label>
            <input
              type="range"
              min={0.3}
              max={4}
              step={0.1}
              value={sizeScale}
              onChange={(e) => setSizeScale(Number(e.target.value))}
              className="w-20 accent-primary"
            />
          </div>
          <div className="flex items-center gap-1.5">
            <label className="text-[10px] text-muted-foreground whitespace-nowrap">
              Spread
            </label>
            <input
              type="range"
              min={0}
              max={1}
              step={0.05}
              value={sizeSpread}
              onChange={(e) => setSizeSpread(Number(e.target.value))}
              className="w-20 accent-primary"
            />
          </div>
          <div className="flex items-center gap-1.5">
            <label className="text-[10px] text-muted-foreground whitespace-nowrap">
              Edges
            </label>
            <input
              type="range"
              min={0}
              max={0.5}
              step={0.01}
              value={edgeOpacity}
              onChange={(e) => setEdgeOpacity(Number(e.target.value))}
              className="w-20 accent-primary"
            />
          </div>
        </div>

        {graphData.isLoading && (
          <span className="text-xs text-muted-foreground animate-pulse">
            Loading graph...
          </span>
        )}
      </div>

      {/* 3D Canvas + Detail */}
      <div className="flex flex-1 min-h-0 gap-3">
        <div className="relative flex-1 rounded-lg border bg-[hsl(222,47%,5%)]">
          {nodes.length > 0 ? (
            <>
              <Canvas camera={{ fov: 60, near: 0.001, far: 200 }}>
                <GraphNodes
                  nodes={nodes}
                  degreeMap={degreeMap}
                  maxDegree={maxDegree}
                  sizeScale={sizeScale}
                  sizeSpread={sizeSpread}
                  onNodeClick={setSelectedNode}
                  selectedId={selectedNode}
                />
                <GraphEdges edges={edges} nodePositions={nodePositions} opacity={edgeOpacity} />
                <HoverLabel
                  nodes={nodes}
                  degreeMap={degreeMap}
                  pageMap={pageMap}
                  loadedSet={loadedSet}
                />
                <AutoFitCamera nodes={nodes} />
                <OrbitControls
                  enableDamping
                  dampingFactor={0.1}
                  rotateSpeed={0.5}
                />
              </Canvas>
              <Legend />
            </>
          ) : (
            <div className="flex h-full items-center justify-center text-sm text-muted-foreground">
              {graphData.isLoading
                ? "Loading page graph..."
                : !selectedGroup
                  ? "Select a page graph group"
                  : "No page graph data available for this group"}
            </div>
          )}
        </div>

        {/* Detail panel */}
        {selectedNode && (
          <div className="w-72 shrink-0">
            <NodeDetail
              nodeId={selectedNode}
              edges={edges}
              degreeMap={degreeMap}
              pageMap={pageMap}
              loadedSet={loadedSet}
              onClose={() => setSelectedNode(null)}
            />
          </div>
        )}
      </div>
    </div>
  );
}
