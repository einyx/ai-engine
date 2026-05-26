import { Fragment, useEffect, useRef, useState } from "react";
import {
  fetchTopology,
  fetchServices,
  streamChat,
  type NodeTopology,
  type ServiceInfo,
  type ProviderStats,
  type GpuInfo,
} from "../api";
import { ClusterTopologyGraph } from "./ClusterTopology";

function backendClass(b: string) {
  const k = b.toLowerCase();
  if (k === "cuda") return "chip cuda";
  if (k === "metal") return "chip metal";
  return "chip cpu";
}

export function NodeCards({
  nodes,
  tps,
}: {
  nodes: NodeTopology[];
  tps: Record<string, number>;
}) {
  if (nodes.length === 0) {
    return (
      <div className="offline fade-up">
        <div className="glyph">▣ no link</div>
        <p>No cluster — running in gateway-only mode.</p>
        <p>
          Start a leader with a <code>[[cluster]]</code> config and at least one
          worker node to populate the pipeline.
        </p>
      </div>
    );
  }

  const totalLayers = Math.max(1, ...nodes.map((n) => n.layer_end));

  return (
    <div className="pipeline">
      {nodes.map((n, i) => {
        const rate = tps[n.node_id] ?? 0;
        const active = rate > 0;
        const span = n.layer_end - n.layer_start;
        return (
          <Fragment key={n.node_id}>
            <div
              className={`node fade-up${active ? " active" : ""}`}
              style={{ animationDelay: `${i * 70}ms` }}
            >
              <div className="node-head">
                <span className="node-id">{n.node_id}</span>
                <span className={backendClass(n.backend)}>
                  {n.backend || "—"}
                  {n.device_index > 0 ? `:${n.device_index}` : ""}
                </span>
                {n.hosts_embedding && <span className="chip role">embed</span>}
                {n.hosts_output && <span className="chip role">output</span>}
              </div>

              <div className="layers-label">
                <span>layers {n.layer_start}–{n.layer_end}</span>
                <span>{span} blk</span>
              </div>
              <div className="layers-bar">
                <span
                  style={{
                    left: `${(n.layer_start / totalLayers) * 100}%`,
                    width: `${(span / totalLayers) * 100}%`,
                  }}
                />
              </div>

              <div className="stats">
                <div className="stat">
                  <span className="k">vram</span>
                  <span className="v">
                    {(n.available_memory_bytes / 1e9).toFixed(1)} GB
                  </span>
                </div>
                <div className="stat">
                  <span className="k">score</span>
                  <span className="v">{n.compute_score}</span>
                </div>
                <div className="stat">
                  <span className="k">link</span>
                  <span className="v">{n.link_mbps_to_leader} Mbps</span>
                </div>
                <div className="stat">
                  <span className="k">state</span>
                  <span className="v" style={{ color: active ? "var(--cyan)" : "var(--text-dim)" }}>
                    {active ? "streaming" : "idle"}
                  </span>
                </div>
              </div>

              <div className="tps">{rate.toFixed(1)} tok/s</div>
            </div>

            {i < nodes.length - 1 && (
              <div className={`connector${active ? " flowing" : ""}`}>
                <span className="wire" />
                <span className="packet" />
              </div>
            )}
          </Fragment>
        );
      })}
    </div>
  );
}

function kindColor(kind: string): string {
  const k = kind.toLowerCase();
  if (k === "anthropic") return "var(--violet)";
  if (k === "openai") return "var(--cyan)";
  if (k === "rustyllm" || k === "candle") return "var(--green)";
  return "var(--amber)";
}

function deviceClass(device?: string | null): string {
  const k = (device ?? "").toLowerCase();
  if (k.startsWith("cuda")) return "chip cuda";
  if (k === "metal") return "chip metal";
  if (k === "cpu") return "chip cpu";
  return "chip cpu";
}

function fmtTokens(n: number): string {
  if (n >= 1_000_000) return `${(n / 1_000_000).toFixed(1)}M`;
  if (n >= 1_000) return `${(n / 1_000).toFixed(1)}k`;
  return `${n}`;
}

const SPARK_LEN = 48;

// Inline SVG sparkline. Values are normalised against their own max so a
// provider's recent throughput shape reads even when absolute rates differ.
function Sparkline({ values, color }: { values: number[]; color: string }) {
  const w = 96;
  const h = 26;
  if (values.length < 2) {
    return <svg className="spark" width={w} height={h} aria-hidden />;
  }
  const max = Math.max(1e-6, ...values);
  const step = w / (SPARK_LEN - 1);
  const offset = SPARK_LEN - values.length;
  const pts = values
    .map((v, i) => {
      const x = (offset + i) * step;
      const y = h - 2 - (v / max) * (h - 4);
      return `${x.toFixed(1)},${y.toFixed(1)}`;
    })
    .join(" ");
  const last = values[values.length - 1];
  const lastX = (SPARK_LEN - 1) * step;
  const lastY = h - 2 - (last / max) * (h - 4);
  return (
    <svg className="spark" width={w} height={h} aria-hidden>
      <polyline
        points={pts}
        fill="none"
        stroke={color}
        strokeWidth="1.5"
        strokeLinejoin="round"
        strokeLinecap="round"
      />
      {last > 0 && <circle cx={lastX} cy={lastY} r="1.6" fill={color} />}
    </svg>
  );
}

function HealthDot({ stats }: { stats?: ProviderStats }) {
  const h = stats?.health;
  if (!h || !h.checked) {
    return <span className="hdot unknown" title="not yet probed" />;
  }
  return (
    <span
      className={`hdot ${h.up ? "up" : "down"}`}
      title={h.up ? `up · ${h.latency_ms}ms probe` : "probe failed"}
    />
  );
}

function ProviderCard({
  svc,
  rate,
  tokens,
  stats,
  history,
}: {
  svc: ServiceInfo;
  rate: number;
  tokens: number;
  stats?: ProviderStats;
  history: number[];
}) {
  const [open, setOpen] = useState(false);
  const [copied, setCopied] = useState(false);
  const [testing, setTesting] = useState(false);
  const [testOut, setTestOut] = useState<string | null>(null);
  const active = rate > 0;
  const color = kindColor(svc.kind);
  const target = svc.endpoint ?? (svc.weights ? svc.weights : "in-process");

  function copy() {
    navigator.clipboard?.writeText(target).then(
      () => {
        setCopied(true);
        setTimeout(() => setCopied(false), 1200);
      },
      () => {},
    );
  }

  async function runTest() {
    const model = svc.models[0];
    if (!model || testing) return;
    setTesting(true);
    setTestOut("");
    try {
      await streamChat(model, "Reply with a short friendly greeting.", (t) =>
        setTestOut((prev) => (prev ?? "") + t),
      );
    } catch (e) {
      setTestOut(`✗ ${e instanceof Error ? e.message : String(e)}`);
    } finally {
      setTesting(false);
    }
  }

  return (
    <div className={`node provider-card fade-up${active ? " active" : ""}`}>
      <div className="node-head">
        <HealthDot stats={stats} />
        <span className="node-id">{svc.id}</span>
        <span className="chip" style={{ color }}>
          {svc.kind}
        </span>
        {svc.device && <span className={deviceClass(svc.device)}>{svc.device}</span>}
        <span className="chip role">{svc.local ? "local" : "remote"}</span>
      </div>

      <div className="layers-label">
        <span className="endpoint-line" title={target}>
          {target}
        </span>
        <button className="mini-btn" onClick={copy} title="copy">
          {copied ? "copied" : "copy"}
        </button>
      </div>

      <div className="prov-metrics">
        <div className="stat">
          <span className="k">tok/s</span>
          <span className="v" style={{ color: active ? "var(--cyan)" : "var(--text)" }}>
            {rate.toFixed(1)}
          </span>
        </div>
        <div className="stat">
          <span className="k">served</span>
          <span className="v">{fmtTokens(tokens)}</span>
        </div>
        <div className="stat">
          <span className="k">reqs</span>
          <span className="v">{stats?.requests ?? 0}</span>
        </div>
        <div className="stat">
          <span className="k">errors</span>
          <span className="v" style={{ color: (stats?.errors ?? 0) > 0 ? "var(--red)" : "var(--text)" }}>
            {stats?.errors ?? 0}
          </span>
        </div>
        <div className="stat">
          <span className="k">latency</span>
          <span className="v">
            {stats?.avg_latency_ms ? `${Math.round(stats.avg_latency_ms)}ms` : "—"}
          </span>
        </div>
        <div className="stat spark-cell">
          <span className="k">throughput</span>
          <Sparkline values={history} color={color} />
        </div>
        {stats?.resources && (
          <>
            <div className="stat">
              <span className="k">cpu</span>
              <span className="v">
                {stats.resources.cpu_count} · load {stats.resources.load1.toFixed(1)}
              </span>
            </div>
            <div className="stat">
              <span className="k">mem free</span>
              <span className="v">
                {(stats.resources.mem_avail_mb / 1024).toFixed(1)}/
                {(stats.resources.mem_total_mb / 1024).toFixed(0)} GB
              </span>
            </div>
            <div className="stat">
              <span className="k">disk free</span>
              <span className="v">{stats.resources.disk_avail_gb.toFixed(0)} GB</span>
            </div>
          </>
        )}
      </div>

      <div className="prov-foot">
        <button className="mini-btn" onClick={() => setOpen((o) => !o)}>
          {open ? "▾ hide" : `▸ ${svc.models.length} model${svc.models.length === 1 ? "" : "s"}`}
        </button>
        <button
          className="mini-btn test"
          onClick={runTest}
          disabled={testing || svc.models.length === 0}
          title={svc.models.length === 0 ? "no model routed" : "send a test prompt"}
        >
          {testing ? "testing…" : "test"}
        </button>
      </div>

      {open && (
        <div className="prov-models">
          {svc.models.length ? (
            svc.models.map((m) => (
              <span className="svc-model" key={m}>
                {m}
              </span>
            ))
          ) : (
            <span className="muted">no models routed</span>
          )}
        </div>
      )}

      {testOut !== null && (
        <div className="prov-test">
          {testOut || <span className="muted">…</span>}
        </div>
      )}
    </div>
  );
}

export function GatewayPipeline({
  services,
  rates = {},
  total = 0,
  tokens = {},
  stats = {},
  history = {},
}: {
  services: ServiceInfo[];
  rates?: Record<string, number>;
  total?: number;
  tokens?: Record<string, number>;
  stats?: Record<string, ProviderStats>;
  history?: Record<string, number[]>;
}) {
  if (services.length === 0) {
    return (
      <div className="offline fade-up">
        <div className="glyph">▣ no link</div>
        <p>No cluster and no providers configured.</p>
        <p>
          Add a <code>[[provider]]</code> (e.g. a local Ollama or a{" "}
          <code>kind = "rustyllm"</code> model) or start a{" "}
          <code>[[cluster]]</code> leader to populate the pipeline.
        </p>
      </div>
    );
  }

  const totalModels = services.reduce((a, s) => a + s.models.length, 0);

  return (
    <>
      <div className="gateway-node fade-up">
        <span className="node-id">gateway</span>
        <span className="chip role">ingress · /v1</span>
        <span className="gw-stat">{totalModels} routes</span>
        <span className="gw-stat" style={{ color: total > 0 ? "var(--cyan)" : "var(--text-dim)" }}>
          {total.toFixed(1)} tok/s
        </span>
        <Sparkline values={history["__total__"] ?? []} color="var(--cyan)" />
      </div>

      <div className="provider-grid">
        {services.map((svc) => (
          <ProviderCard
            key={svc.id}
            svc={svc}
            rate={rates[svc.id] ?? 0}
            tokens={tokens[svc.id] ?? 0}
            stats={stats[svc.id]}
            history={history[svc.id] ?? []}
          />
        ))}
      </div>
    </>
  );
}

// Load-balanced pools: any model served by 2+ providers. Shows the LB topology
// and the live split of tok/s across the pool members. Returns null (incl. its
// header) when nothing is pooled. Inline-styled to avoid coupling to styles.css.
function LbPools({
  services,
  rates,
}: {
  services: ServiceInfo[];
  rates: Record<string, number>;
}) {
  const byModel = new Map<string, string[]>();
  for (const s of services) {
    for (const m of s.models) byModel.set(m, [...(byModel.get(m) ?? []), s.id]);
  }
  const pools = [...byModel.entries()].filter(([, ids]) => ids.length > 1);
  if (pools.length === 0) return null;

  return (
    <>
      <div className="section-head" style={{ marginTop: "28px" }}>
        <h2>Load-balanced pools</h2>
        <span className="rule" />
        <span className="meta">models served by 2+ providers · least-in-flight</span>
      </div>
      <div style={{ display: "flex", flexDirection: "column", gap: "10px" }}>
        {pools.map(([model, ids]) => {
          const sum = ids.reduce((a, id) => a + (rates[id] ?? 0), 0);
          return (
            <div
              key={model}
              className="fade-up"
              style={{
                border: "1px solid var(--bg-grid, #1a1c22)",
                borderRadius: 8,
                padding: "10px 12px",
                background: "rgba(255,255,255,.015)",
              }}
            >
              <div style={{ display: "flex", alignItems: "baseline", gap: 10, marginBottom: 8 }}>
                <span className="svc-model" style={{ fontSize: 13 }}>{model}</span>
                <span style={{ fontSize: 11, color: "var(--text-faint)", letterSpacing: ".08em" }}>
                  {ids.length}-WAY
                </span>
                <span style={{ marginLeft: "auto", fontSize: 12, color: sum > 0 ? "var(--cyan)" : "var(--text-dim)" }}>
                  {sum.toFixed(1)} tok/s
                </span>
              </div>
              <div style={{ display: "flex", flexWrap: "wrap", gap: 8 }}>
                {ids.map((id) => {
                  const r = rates[id] ?? 0;
                  const share = sum > 0 ? Math.round((r / sum) * 100) : 0;
                  return (
                    <span
                      key={id}
                      style={{
                        display: "inline-flex",
                        alignItems: "center",
                        gap: 6,
                        fontSize: 12,
                        padding: "3px 8px",
                        borderRadius: 6,
                        border: "1px solid var(--bg-grid, #1a1c22)",
                        color: r > 0 ? "var(--cyan)" : "var(--text-dim)",
                      }}
                    >
                      <span
                        style={{
                          width: 6,
                          height: 6,
                          borderRadius: 99,
                          background: r > 0 ? "var(--cyan)" : "var(--text-faint)",
                        }}
                      />
                      {id} <b>{r.toFixed(1)}</b>
                      {sum > 0 && <span style={{ color: "var(--text-faint)" }}>{share}%</span>}
                    </span>
                  );
                })}
              </div>
            </div>
          );
        })}
      </div>
    </>
  );
}

// Realtime GPU panel (nvidia-smi). Hidden entirely when no GPU is present.
function meterColor(pct: number): string {
  if (pct >= 90) return "var(--red)";
  if (pct >= 70) return "var(--amber)";
  return "var(--green)";
}

function GpuPanel({ gpus }: { gpus: GpuInfo[] }) {
  if (gpus.length === 0) return null;
  return (
    <>
      <div className="section-head" style={{ marginTop: "28px" }}>
        <h2>GPU</h2>
        <span className="rule" />
        <span className="meta">
          {gpus.length} device{gpus.length === 1 ? "" : "s"} · nvidia-smi · live
        </span>
      </div>
      <div className="gpu-grid">
        {gpus.map((g) => {
          const memPct = g.mem_total_mb > 0 ? (g.mem_used_mb / g.mem_total_mb) * 100 : 0;
          const pwrPct = g.power_limit_w > 0 ? (g.power_w / g.power_limit_w) * 100 : 0;
          return (
            <div className="gpu-card fade-up" key={g.index}>
              <div className="node-head">
                <span className="hdot up" />
                <span className="node-id">{g.name}</span>
                <span className="chip cuda">cuda:{g.index}</span>
                <span className="gpu-temp">{g.temp_c}°C</span>
              </div>

              <div className="gpu-meter-row">
                <span className="k">util</span>
                <div className="meter">
                  <span style={{ width: `${g.util_pct}%`, background: meterColor(g.util_pct) }} />
                </div>
                <span className="gpu-meter-val">{g.util_pct}%</span>
              </div>

              <div className="gpu-meter-row">
                <span className="k">vram</span>
                <div className="meter">
                  <span style={{ width: `${memPct}%`, background: meterColor(memPct) }} />
                </div>
                <span className="gpu-meter-val">
                  {(g.mem_used_mb / 1024).toFixed(1)}/{(g.mem_total_mb / 1024).toFixed(1)}G
                </span>
              </div>

              <div className="gpu-meter-row">
                <span className="k">power</span>
                <div className="meter">
                  <span style={{ width: `${pwrPct}%`, background: meterColor(pwrPct) }} />
                </div>
                <span className="gpu-meter-val">
                  {Math.round(g.power_w)}/{Math.round(g.power_limit_w)}W
                </span>
              </div>
            </div>
          );
        })}
      </div>
    </>
  );
}

export function Cluster({
  gwRates = {},
  gwTotal = 0,
  gwTokens = {},
  gwStats = {},
  gpus = [],
}: {
  // Gateway throughput is streamed at App level (always sampling) and passed
  // down, so a chat's tok/s isn't missed while this tab was unmounted.
  gwRates?: Record<string, number>;
  gwTotal?: number;
  gwTokens?: Record<string, number>;
  gwStats?: Record<string, ProviderStats>;
  gpus?: GpuInfo[];
}) {
  const [nodes, setNodes] = useState<NodeTopology[]>([]);
  const [tps, setTps] = useState<Record<string, number>>({});
  const [total, setTotal] = useState(0);
  const [services, setServices] = useState<ServiceInfo[]>([]);
  const [history, setHistory] = useState<Record<string, number[]>>({});
  const seenTick = useRef(0);

  function loadServices() {
    fetchServices()
      .then(setServices)
      .catch(() => setServices([]));
  }

  useEffect(() => {
    fetchTopology()
      .then((t) => setNodes(t.nodes))
      .catch(() => setNodes([]));
    loadServices();
    const es = new EventSource("/cluster/metrics");
    es.onmessage = (e) => {
      try {
        const j = JSON.parse(e.data);
        setTps(j.per_node ?? {});
        setTotal(j.total_tps ?? 0);
      } catch {
        /* ignore */
      }
    };
    return () => es.close();
  }, []);

  // Accumulate a rolling tok/s history per provider (plus the gateway total)
  // from the App-level SSE props. Keyed off the rates object identity, which
  // changes once per second with each SSE frame.
  useEffect(() => {
    seenTick.current += 1;
    setHistory((prev) => {
      const next: Record<string, number[]> = { ...prev };
      const push = (key: string, val: number) => {
        const arr = (next[key] ?? []).concat(val);
        next[key] = arr.length > SPARK_LEN ? arr.slice(arr.length - SPARK_LEN) : arr;
      };
      for (const svc of services) push(svc.id, gwRates[svc.id] ?? 0);
      push("__total__", gwTotal);
      return next;
    });
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [gwRates, gwTotal]);

  return (
    <>
      <div className="section-head">
        <h2>{nodes.length > 0 ? "Pipeline Topology" : "AI Services"}</h2>
        <span className="rule" />
        <span className="meta">
          {nodes.length > 0
            ? `${nodes.length} node${nodes.length === 1 ? "" : "s"} · ${total.toFixed(1)} tok/s aggregate`
            : `${services.length} provider${services.length === 1 ? "" : "s"} · ${gwTotal.toFixed(1)} tok/s`}
        </span>
        {nodes.length === 0 && (
          <button className="mini-btn refresh" onClick={loadServices} title="reload providers">
            ⟳
          </button>
        )}
      </div>

      {nodes.length > 0 ? (
        <NodeCards nodes={nodes} tps={tps} />
      ) : (
        <GatewayPipeline
          services={services}
          rates={gwRates}
          total={gwTotal}
          tokens={gwTokens}
          stats={gwStats}
          history={history}
        />
      )}

      {nodes.length === 0 && <LbPools services={services} rates={gwRates} />}

      <GpuPanel gpus={gpus} />

      <ClusterTopologyGraph services={services} nodes={nodes} gpus={gpus} rates={gwRates} />
    </>
  );
}
