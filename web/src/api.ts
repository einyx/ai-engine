export interface ModelInfo {
  id: string;
}

export async function fetchModels(): Promise<ModelInfo[]> {
  const r = await fetch("/v1/models");
  if (!r.ok) throw new Error(`HTTP ${r.status}`);
  const j = await r.json();
  return (j.data ?? []).map((m: { id: string }) => ({ id: m.id }));
}

export interface NodeTopology {
  node_id: string;
  backend: string;
  device_index: number;
  available_memory_bytes: number;
  compute_score: number;
  link_mbps_to_leader: number;
  layer_start: number;
  layer_end: number;
  hosts_embedding: boolean;
  hosts_output: boolean;
  previous_node: string | null;
  next_node: string | null;
}

export interface TopologySnapshot {
  model_id: string | null;
  nodes: NodeTopology[];
}

export async function fetchTopology(): Promise<TopologySnapshot> {
  const r = await fetch("/cluster/topology");
  return r.json();
}

export type NodeKind =
  | "project"
  | "session"
  | "memory"
  | "command"
  | "cluster"
  | "chat"
  | "model"
  | "provider";
export type EdgeKind =
  | "owns"
  | "produced"
  | "links"
  | "ran"
  | "pipeline"
  | "touched"
  | "used"
  | "served";

export interface GraphNode {
  id: string;
  kind: NodeKind;
  label: string;
  meta?: Record<string, unknown>;
}

export interface GraphEdge {
  source: string;
  target: string;
  kind: EdgeKind;
}

export interface GraphSnapshot {
  nodes: GraphNode[];
  edges: GraphEdge[];
}

export async function fetchGraph(): Promise<GraphSnapshot> {
  const r = await fetch("/graph");
  if (!r.ok) throw new Error(`HTTP ${r.status}`);
  return r.json();
}

export interface ServiceInfo {
  id: string;
  kind: string;
  endpoint?: string | null;
  models: string[];
  /** Device spec for in-process local backends (candle/rustyllm). */
  device?: string | null;
  /** Basename of the weights file for local backends. */
  weights?: string | null;
  /** True when inference is in-process or against a localhost endpoint. */
  local?: boolean;
}

/** Live per-provider stats from the /gateway/metrics SSE `stats` map. */
export interface ProviderStats {
  tps: number;
  tokens: number;
  requests: number;
  errors: number;
  avg_latency_ms: number;
  health: { up: boolean; latency_ms: number; checked: boolean } | null;
  resources: {
    cpu_count: number;
    load1: number;
    mem_total_mb: number;
    mem_avail_mb: number;
    disk_avail_gb: number;
  } | null;
}

/** Realtime per-GPU telemetry (nvidia-smi) from the /gateway/metrics SSE. */
export interface GpuInfo {
  index: number;
  name: string;
  util_pct: number;
  mem_used_mb: number;
  mem_total_mb: number;
  temp_c: number;
  power_w: number;
  power_limit_w: number;
}

export async function fetchServices(): Promise<ServiceInfo[]> {
  const r = await fetch("/cluster/services");
  if (!r.ok) throw new Error(`HTTP ${r.status}`);
  return r.json();
}

/** Stream a chat completion; calls onToken for each text delta. */
export async function streamChat(
  model: string,
  content: string,
  onToken: (t: string) => void,
): Promise<void> {
  const resp = await fetch("/v1/chat/completions", {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify({
      model,
      stream: true,
      messages: [{ role: "user", content }],
    }),
  });
  if (!resp.ok || !resp.body) {
    const detail = await resp.text().catch(() => "");
    throw new Error(`HTTP ${resp.status}${detail ? ` — ${detail.slice(0, 200)}` : ""}`);
  }
  const reader = resp.body.getReader();
  const decoder = new TextDecoder();
  let buf = "";
  for (;;) {
    const { done, value } = await reader.read();
    if (done) break;
    buf += decoder.decode(value, { stream: true });
    const lines = buf.split("\n");
    buf = lines.pop() ?? "";
    for (const line of lines) {
      const s = line.trim();
      if (!s.startsWith("data:")) continue;
      const data = s.slice(5).trim();
      if (data === "[DONE]") return;
      try {
        const j = JSON.parse(data);
        const delta = j.choices?.[0]?.delta?.content;
        if (delta) onToken(delta);
      } catch {
        /* ignore keep-alive / partial frames */
      }
    }
  }
}
