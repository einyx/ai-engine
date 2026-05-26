import { useEffect, useMemo, useRef, useState } from "react";
import { forceCollide } from "d3-force-3d";
import ForceGraph2D from "react-force-graph-2d";
import type { ServiceInfo, NodeTopology, GpuInfo } from "../api";

// Architecture force-graph for the Cluster view. Renders the live request
// fabric — gateway → providers → models, GPUs wired to local CUDA providers —
// plus the distributed pipeline ring when a leader/worker (or p2p) cluster is
// present. Shared models surface as a single node with an edge from each
// provider, so load-balanced pools read at a glance.

type Kind = "gateway" | "provider" | "model" | "gpu" | "node";

interface TopoNode {
  id: string;
  label: string;
  kind: Kind;
  color: string;
  val: number;
  active?: boolean;
}
interface TopoLink {
  source: string;
  target: string;
  active?: boolean;
}

const COLOR: Record<Kind, string> = {
  gateway: "#28e0e0",
  provider: "#ffb02e",
  model: "#ffd166",
  gpu: "#45d67e",
  node: "#ff5f56",
};

function providerColor(kind: string): string {
  const k = kind.toLowerCase();
  if (k === "anthropic") return "#9d8cff";
  if (k === "openai") return "#28e0e0";
  if (k === "rustyllm" || k === "candle") return "#45d67e";
  return "#ffb02e";
}

const GW = "__gateway__";

function buildGraph(
  services: ServiceInfo[],
  nodes: NodeTopology[],
  gpus: GpuInfo[],
  rates: Record<string, number>,
): { nodes: TopoNode[]; links: TopoLink[] } {
  const n: TopoNode[] = [];
  const l: TopoLink[] = [];
  const seenModel = new Set<string>();

  n.push({ id: GW, label: "gateway", kind: "gateway", color: COLOR.gateway, val: 6 });

  for (const svc of services) {
    const pid = `p:${svc.id}`;
    const active = (rates[svc.id] ?? 0) > 0;
    n.push({
      id: pid,
      label: svc.device ? `${svc.id} · ${svc.device}` : svc.id,
      kind: "provider",
      color: providerColor(svc.kind),
      val: 4,
      active,
    });
    l.push({ source: GW, target: pid, active });

    for (const m of svc.models) {
      const mid = `m:${m}`;
      if (!seenModel.has(mid)) {
        seenModel.add(mid);
        n.push({ id: mid, label: m, kind: "model", color: COLOR.model, val: 2 });
      }
      l.push({ source: pid, target: mid });
    }

    // Wire local CUDA providers to the matching GPU node (created below).
    if (svc.local && svc.device && svc.device.startsWith("cuda")) {
      const idx = svc.device.split(":")[1] ?? "0";
      l.push({ source: `g:${idx}`, target: pid, active });
    }
  }

  for (const g of gpus) {
    n.push({
      id: `g:${g.index}`,
      label: `${g.name} (${g.util_pct}%)`,
      kind: "gpu",
      color: COLOR.gpu,
      val: 5,
    });
  }

  // Distributed pipeline ring (leader/worker or p2p): node → next_node.
  for (const nd of nodes) {
    const id = `n:${nd.node_id}`;
    const tag = [nd.hosts_embedding && "embed", nd.hosts_output && "output"]
      .filter(Boolean)
      .join("·");
    const active = (rates[nd.node_id] ?? 0) > 0;
    n.push({
      id,
      label: tag ? `${nd.node_id} [${tag}] L${nd.layer_start}–${nd.layer_end}` : `${nd.node_id} L${nd.layer_start}–${nd.layer_end}`,
      kind: "node",
      color: COLOR.node,
      val: 4,
      active,
    });
    if (nd.next_node) l.push({ source: id, target: `n:${nd.next_node}`, active });
    if (nd.hosts_embedding) l.push({ source: GW, target: id, active });
  }

  return { nodes: n, links: l };
}

export function ClusterTopologyGraph({
  services,
  nodes,
  gpus,
  rates,
}: {
  services: ServiceInfo[];
  nodes: NodeTopology[];
  gpus: GpuInfo[];
  rates: Record<string, number>;
}) {
  const fgRef = useRef<any>(null);
  const wrapRef = useRef<HTMLDivElement>(null);
  const [width, setWidth] = useState(800);
  const HEIGHT = 380;

  // Rebuild only when the shape (ids) changes, not on every rate tick, so the
  // simulation doesn't reheat constantly. Active flags still update in place.
  const graph = useMemo(
    () => buildGraph(services, nodes, gpus, rates),
    // eslint-disable-next-line react-hooks/exhaustive-deps
    [
      services.map((s) => `${s.id}:${s.models.join(",")}:${s.device ?? ""}`).join("|"),
      nodes.map((n) => `${n.node_id}>${n.next_node ?? ""}`).join("|"),
      gpus.map((g) => g.index).join(","),
    ],
  );

  useEffect(() => {
    const measure = () => setWidth(wrapRef.current?.clientWidth ?? 800);
    measure();
    const ro = new ResizeObserver(measure);
    if (wrapRef.current) ro.observe(wrapRef.current);
    return () => ro.disconnect();
  }, []);

  useEffect(() => {
    const fg = fgRef.current;
    if (!fg) return;
    fg.d3Force("charge")?.strength(-220).distanceMax(500);
    fg.d3Force("link")?.distance(60).strength(0.3);
    fg.d3Force("collide", forceCollide(18));
    fg.d3ReheatSimulation?.();
  }, [graph]);

  if (graph.nodes.length <= 1) return null;

  return (
    <>
      <div className="section-head" style={{ marginTop: "28px" }}>
        <h2>Architecture</h2>
        <span className="rule" />
        <span className="meta">
          {nodes.length > 0 ? "distributed pipeline" : "request fabric"} · force graph
        </span>
      </div>
      <div ref={wrapRef} className="topo-graph">
        <ForceGraph2D
          ref={fgRef}
          width={width}
          height={HEIGHT}
          graphData={graph}
          nodeId="id"
          nodeLabel={(nd: TopoNode) => `${nd.kind} · ${nd.label}`}
          nodeVal={(nd: TopoNode) => nd.val}
          nodeRelSize={4}
          nodeColor={(nd: TopoNode) => nd.color}
          linkColor={(lk: TopoLink) => (lk.active ? "rgba(40,224,224,.6)" : "rgba(108,120,134,.25)")}
          linkDirectionalParticles={(lk: TopoLink) => (lk.active ? 3 : 0)}
          linkDirectionalParticleWidth={2}
          linkWidth={(lk: TopoLink) => (lk.active ? 1.6 : 0.6)}
          d3VelocityDecay={0.3}
          cooldownTicks={120}
          backgroundColor="#08090c"
          nodeCanvasObjectMode={() => "after"}
          nodeCanvasObject={(nd: TopoNode, ctx: CanvasRenderingContext2D, scale: number) => {
            const label = nd.label;
            const fontSize = 11 / scale;
            ctx.font = `${fontSize}px 'IBM Plex Mono', monospace`;
            ctx.fillStyle = "rgba(212,218,226,.85)";
            ctx.textAlign = "center";
            ctx.textBaseline = "top";
            const r = Math.sqrt(nd.val) * 4;
            ctx.fillText(label, (nd as any).x, (nd as any).y + r + 1.5 / scale);
          }}
        />
      </div>
    </>
  );
}
