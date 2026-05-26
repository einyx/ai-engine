import { useEffect, useMemo, useRef, useState } from "react";
import { forceCollide } from "d3-force-3d";
import ForceGraph2D from "react-force-graph-2d";
import {
  fetchGraph,
  type GraphSnapshot,
  type GraphNode,
  type NodeKind,
  type EdgeKind,
} from "../api";

const KIND_COLOR: Record<NodeKind, string> = {
  project: "#9d8cff",
  session: "#ffb02e",
  memory: "#28e0e0",
  command: "#6c7886",
  cluster: "#ff5f56",
  chat: "#7CFFB2",     // green — live gateway requests
  model: "#ffd166",    // gold — model names
  provider: "#4ea8ff", // blue — upstreams that served traffic
};

const KIND_ORDER: NodeKind[] = [
  "project",
  "session",
  "memory",
  "command",
  "cluster",
  "chat",
  "model",
  "provider",
];

const EDGE_COLOR: Record<EdgeKind, string> = {
  owns: "rgba(157,140,255,.35)",
  produced: "rgba(40,224,224,.45)",
  links: "rgba(40,224,224,.7)",
  ran: "rgba(108,120,134,.28)",
  pipeline: "rgba(255,95,86,.6)",
  touched: "rgba(255,176,46,.45)",
  used: "rgba(124,255,178,.5)",   // chat -> model
  served: "rgba(78,168,255,.6)",  // model -> provider
};

const DIM = "rgba(108,120,134,.08)";

export function Graph() {
  const [data, setData] = useState<GraphSnapshot | null>(null);
  const [error, setError] = useState<string | null>(null);
  const [hidden, setHidden] = useState<Set<NodeKind>>(new Set());
  const [selected, setSelected] = useState<GraphNode | null>(null);
  const [query, setQuery] = useState("");
  const fgRef = useRef<any>(null);

  const load = () => {
    fetchGraph()
      .then((g) => {
        setData(g);
        setError(null);
        setSelected(null);
      })
      .catch((e) => setError(String(e)));
  };

  // Poll for live activity (new chats → model → provider). Only swap data when
  // the node/edge count actually changes, so the force sim doesn't reheat — and
  // the current selection is preserved across refreshes.
  const sigRef = useRef("");
  useEffect(() => {
    const poll = () => {
      fetchGraph()
        .then((g) => {
          const sig = `${g.nodes.length}:${g.edges.length}`;
          if (sig !== sigRef.current) {
            sigRef.current = sig;
            setData(g);
          }
          setError(null);
        })
        .catch((e) => setError(String(e)));
    };
    poll();
    const id = setInterval(poll, 4000);
    return () => clearInterval(id);
  }, []);

  // Per-kind totals for the legend (from the full, unfiltered graph).
  const counts = useMemo(() => {
    const c = {} as Record<NodeKind, number>;
    for (const k of KIND_ORDER) c[k] = 0;
    for (const n of data?.nodes ?? []) c[n.kind]++;
    return c;
  }, [data]);

  // Degree from the original edge list (link endpoints get rewritten to object
  // refs by the lib after the first tick, so we count here, once).
  const degree = useMemo(() => {
    const d = new Map<string, number>();
    for (const e of data?.edges ?? []) {
      d.set(e.source, (d.get(e.source) ?? 0) + 1);
      d.set(e.target, (d.get(e.target) ?? 0) + 1);
    }
    return d;
  }, [data]);

  // Spread the layout so nodes/edges stop piling up: stronger repulsion,
  // longer links, and a collision force sized to each node's radius. Combined
  // with curved links (linkCurvature below), converging/parallel edges
  // separate instead of overlapping as straight horizontal lines.
  useEffect(() => {
    const fg = fgRef.current;
    if (!fg) return;
    const valOf = (id: string) => 1 + Math.min(degree.get(id) ?? 0, 60) * 0.5;
    fg.d3Force("charge")?.strength(-160).distanceMax(600);
    fg.d3Force("link")?.distance(50).strength(0.25);
    fg.d3Force(
      "collide",
      forceCollide((n: unknown) => Math.sqrt(valOf((n as GraphNode).id)) * 4 + 4),
    );
    fg.d3ReheatSimulation?.();
  }, [data, degree]);

  // Undirected adjacency for neighbor highlighting (also from original edges).
  const adjacency = useMemo(() => {
    const a = new Map<string, Set<string>>();
    const add = (x: string, y: string) => {
      if (!a.has(x)) a.set(x, new Set());
      a.get(x)!.add(y);
    };
    for (const e of data?.edges ?? []) {
      add(e.source, e.target);
      add(e.target, e.source);
    }
    return a;
  }, [data]);

  // Apply kind filters: drop hidden-kind nodes and any edge touching them.
  const graphData = useMemo(() => {
    const nodes = (data?.nodes ?? []).filter((n) => !hidden.has(n.kind));
    const live = new Set(nodes.map((n) => n.id));
    const links = (data?.edges ?? [])
      .filter((e) => live.has(e.source) && live.has(e.target))
      .map((e) => ({ source: e.source, target: e.target, kind: e.kind }));
    return { nodes: nodes.map((n) => ({ ...n })), links };
  }, [data, hidden]);

  // Focused node: an explicit selection, else the first label match for the
  // search query. Null = nothing focused, everything at full strength.
  const focusId = useMemo(() => {
    if (selected) return selected.id;
    const q = query.trim().toLowerCase();
    if (!q) return null;
    const hit = (data?.nodes ?? []).find(
      (n) => !hidden.has(n.kind) && n.label.toLowerCase().includes(q),
    );
    return hit?.id ?? null;
  }, [selected, query, data, hidden]);

  const highlight = useMemo(() => {
    if (!focusId) return null;
    const set = new Set<string>([focusId]);
    for (const nb of adjacency.get(focusId) ?? []) set.add(nb);
    return set;
  }, [focusId, adjacency]);

  const toggleKind = (k: NodeKind) =>
    setHidden((prev) => {
      const next = new Set(prev);
      if (next.has(k)) next.delete(k);
      else next.add(k);
      return next;
    });

  const nodeColor = (n: GraphNode) =>
    highlight && !highlight.has(n.id) ? DIM : KIND_COLOR[n.kind];

  // Area-based: lib renders radius ~ sqrt(val), so degree maps to a gentle ramp.
  const nodeVal = (n: GraphNode) => 1 + Math.min(degree.get(n.id) ?? 0, 60) * 0.5;

  if (error) return <div className="empty">graph error: {error}</div>;
  if (data && data.nodes.length === 0)
    return <div className="empty">no graph data — scan found nothing</div>;

  return (
    <div className="graph-wrap">
      <div className="graph-controls">
        <input
          className="graph-search"
          placeholder="search nodes…"
          value={query}
          onChange={(e) => setQuery(e.target.value)}
        />
        <button className="tab" onClick={load}>
          refresh
        </button>
      </div>

      <div className="graph-legend">
        {KIND_ORDER.map((k) => (
          <button
            key={k}
            className="legend-row"
            data-off={hidden.has(k)}
            onClick={() => toggleKind(k)}
            title={hidden.has(k) ? "click to show" : "click to hide"}
          >
            <span className="legend-dot" style={{ background: KIND_COLOR[k] }} />
            <span className="legend-name">{k}</span>
            <span className="legend-count">{counts[k]}</span>
          </button>
        ))}
      </div>

      {selected && (
        <div className="graph-detail">
          <button className="detail-close" onClick={() => setSelected(null)}>
            ✕
          </button>
          <div className="detail-kind" style={{ color: KIND_COLOR[selected.kind] }}>
            {selected.kind}
          </div>
          <div className="detail-label">{selected.label}</div>
          <div className="detail-meta">
            <DetailRow k="connections" v={String(degree.get(selected.id) ?? 0)} />
            {Object.entries(selected.meta ?? {}).map(([k, v]) => (
              <DetailRow key={k} k={k} v={String(v)} />
            ))}
          </div>
        </div>
      )}

      <ForceGraph2D
        ref={fgRef}
        graphData={graphData}
        nodeId="id"
        nodeLabel={(n: GraphNode) => `${n.kind} · ${n.label}`}
        nodeColor={nodeColor}
        nodeVal={nodeVal}
        nodeRelSize={4}
        linkCurvature={0.18}
        d3VelocityDecay={0.3}
        cooldownTicks={120}
        linkColor={(l: { kind: EdgeKind; source: unknown; target: unknown }) => {
          if (highlight) {
            const s = typeof l.source === "object" ? (l.source as GraphNode).id : l.source;
            const t = typeof l.target === "object" ? (l.target as GraphNode).id : l.target;
            if (!highlight.has(s as string) || !highlight.has(t as string)) return DIM;
          }
          return EDGE_COLOR[l.kind];
        }}
        linkDirectionalParticles={(l: { kind: EdgeKind }) =>
          l.kind === "pipeline" || l.kind === "produced" ? 2 : 0
        }
        linkDirectionalParticleWidth={2}
        onNodeClick={(n: GraphNode) => setSelected(n)}
        onBackgroundClick={() => setSelected(null)}
        backgroundColor="#08090c"
      />
    </div>
  );
}

function DetailRow({ k, v }: { k: string; v: string }) {
  return (
    <div className="detail-row">
      <span className="detail-k">{k}</span>
      <span className="detail-v">{v}</span>
    </div>
  );
}
