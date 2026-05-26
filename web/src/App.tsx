import { useEffect, useState } from "react";
import { Chat } from "./views/Chat";
import { Cluster } from "./views/Cluster";
import { Graph } from "./views/Graph";
import { fetchTopology, type ProviderStats, type GpuInfo } from "./api";

export function App() {
  const [tab, setTab] = useState<"chat" | "cluster" | "graph">("chat");
  const [clustered, setClustered] = useState<boolean | null>(null);
  // Gateway throughput lives at App level so the stream samples continuously
  // (every tab) — otherwise a chat's tok/s is missed because the Cluster tab
  // wasn't mounted while it happened.
  const [gwTotal, setGwTotal] = useState(0);
  const [gwRates, setGwRates] = useState<Record<string, number>>({});
  const [gwTokens, setGwTokens] = useState<Record<string, number>>({});
  const [gwStats, setGwStats] = useState<Record<string, ProviderStats>>({});
  const [gpus, setGpus] = useState<GpuInfo[]>([]);

  // Probe once so the topbar can report leader vs gateway-only mode.
  useEffect(() => {
    fetchTopology()
      .then((t) => setClustered(t.nodes.length > 0))
      .catch(() => setClustered(false));
  }, []);

  useEffect(() => {
    const gw = new EventSource("/gateway/metrics");
    gw.onmessage = (e) => {
      try {
        const j = JSON.parse(e.data);
        setGwTotal(j.total_tps ?? 0);
        setGwRates(j.per_provider ?? {});
        setGwTokens(j.per_provider_total ?? {});
        setGwStats(j.stats ?? {});
        setGpus(j.gpu ?? []);
      } catch {
        /* ignore */
      }
    };
    return () => gw.close();
  }, []);

  const status =
    clustered === null
      ? { cls: "idle", label: "connecting" }
      : clustered
        ? { cls: "live", label: "leader · online" }
        : { cls: "ok", label: "gateway-only" };

  return (
    <div className="shell">
      <header className="topbar">
        <div className="brand">
          <span className="mark">
            ai<b>·</b>engine
          </span>
          <span className="ver">distributed inference</span>
        </div>

        <span className="status-pill">
          <span className={`dot ${status.cls}`} />
          {status.label}
        </span>

        <span className="status-pill" title="live gateway throughput">
          <span className={`dot ${gwTotal > 0 ? "live" : "idle"}`} />
          {gwTotal.toFixed(1)} tok/s
        </span>

        <nav className="tabs">
          <button
            className="tab"
            data-active={tab === "chat"}
            onClick={() => setTab("chat")}
          >
            Chat
          </button>
          <button
            className="tab"
            data-active={tab === "cluster"}
            onClick={() => setTab("cluster")}
          >
            Cluster
          </button>
          <button
            className="tab"
            data-active={tab === "graph"}
            onClick={() => setTab("graph")}
          >
            Graph
          </button>
        </nav>
      </header>

      {/* Chat stays mounted so its session survives tab switches (it has no
          ongoing cost). Cluster/Graph mount on demand — they hold live SSE
          streams / a force-sim canvas that shouldn't run while hidden. */}
      <div style={{ display: tab === "chat" ? "contents" : "none" }}>
        <Chat />
      </div>
      {tab === "cluster" && (
        <Cluster gwRates={gwRates} gwTotal={gwTotal} gwTokens={gwTokens} gwStats={gwStats} gpus={gpus} />
      )}
      {tab === "graph" && <Graph />}
    </div>
  );
}
