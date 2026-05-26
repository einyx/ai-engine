//! Builds a heterogeneous "system knowledge graph" by scanning local Claude
//! Code data (memories, transcripts) and merging in live cluster topology.
//! Read-only and best-effort: unreadable inputs are skipped, never fatal.

mod memory;
mod transcript;

use serde::Serialize;
use std::path::{Path, PathBuf};
use ai_engine_core::activity::ChatEvent;
use ai_engine_core::cluster_view::TopologySnapshot;

/// Node category. Serializes lowercase (`"session"`, `"memory"`, ...) so the
/// frontend can color/size by kind.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize)]
#[serde(rename_all = "lowercase")]
pub enum NodeKind {
    Project,
    Session,
    Memory,
    Command,
    Cluster,
    Chat,     // a gateway inference request
    Model,    // a model name requests target
    Provider, // an upstream that served requests
}

/// Edge relationship type. Serializes lowercase.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize)]
#[serde(rename_all = "lowercase")]
pub enum EdgeKind {
    Owns,      // project -> session
    Produced,  // session -> memory
    Links,     // memory -> memory
    Ran,       // session -> command
    Pipeline,  // cluster -> cluster
    Touched,   // session -> cluster
    Used,      // chat -> model
    Served,    // model -> provider
}

#[derive(Debug, Clone, PartialEq, Serialize)]
pub struct GraphNode {
    pub id: String,
    pub kind: NodeKind,
    pub label: String,
    /// Freeform per-kind metadata (e.g. memory `type`, command count).
    #[serde(default, skip_serializing_if = "serde_json::Map::is_empty")]
    pub meta: serde_json::Map<String, serde_json::Value>,
}

#[derive(Debug, Clone, PartialEq, Serialize)]
pub struct GraphEdge {
    pub source: String,
    pub target: String,
    pub kind: EdgeKind,
}

#[derive(Debug, Clone, PartialEq, Default, Serialize)]
pub struct GraphSnapshot {
    pub nodes: Vec<GraphNode>,
    pub edges: Vec<GraphEdge>,
}

/// Resolve the projects root: `$AI_ENGINE_GRAPH_ROOT` or `~/.claude/projects`.
/// Returns `None` if neither is resolvable.
pub fn resolve_root() -> Option<PathBuf> {
    if let Ok(p) = std::env::var("AI_ENGINE_GRAPH_ROOT") {
        return Some(PathBuf::from(p));
    }
    let home = std::env::var("HOME").ok()?;
    Some(PathBuf::from(home).join(".claude").join("projects"))
}

/// Top-level entry point: resolve the root and build the merged snapshot,
/// folding in recent gateway `activity` (chat → model → provider).
pub fn scan(topology: &TopologySnapshot, activity: &[ChatEvent]) -> GraphSnapshot {
    let mut snap = match resolve_root() {
        Some(root) => build_snapshot(&root, topology),
        None => build_snapshot(Path::new("/nonexistent"), topology),
    };
    add_activity(&mut snap, activity);
    snap
}

/// Build a snapshot from every project subdir under `projects_root`, merged
/// with live `topology`. Best-effort: unreadable dirs are skipped.
pub fn build_snapshot(projects_root: &Path, topology: &TopologySnapshot) -> GraphSnapshot {
    let mut snap = GraphSnapshot::default();

    if let Ok(entries) = std::fs::read_dir(projects_root) {
        for entry in entries.flatten() {
            let proj_dir = entry.path();
            if !proj_dir.is_dir() {
                continue;
            }
            add_project(&mut snap, &proj_dir);
        }
    }

    add_cluster(&mut snap, topology);
    snap
}

/// Fold recent gateway requests into the graph as chat → model → provider.
/// Model and provider nodes are deduped; each chat is its own node.
fn add_activity(snap: &mut GraphSnapshot, activity: &[ChatEvent]) {
    use std::collections::HashSet;
    let mut seen_model: HashSet<String> = HashSet::new();
    let mut seen_provider: HashSet<String> = HashSet::new();
    let mut seen_served: HashSet<(String, String)> = HashSet::new();

    for ev in activity {
        let model_id = format!("model:{}", ev.model);
        let provider_id = format!("provider:{}", ev.provider);

        if seen_model.insert(ev.model.clone()) {
            snap.nodes.push(GraphNode {
                id: model_id.clone(),
                kind: NodeKind::Model,
                label: ev.model.clone(),
                meta: serde_json::Map::new(),
            });
        }
        if seen_provider.insert(ev.provider.clone()) {
            snap.nodes.push(GraphNode {
                id: provider_id.clone(),
                kind: NodeKind::Provider,
                label: ev.provider.clone(),
                meta: serde_json::Map::new(),
            });
        }

        let chat_id = format!("chat:{}", ev.request_id);
        let mut meta = serde_json::Map::new();
        meta.insert("model".into(), serde_json::Value::String(ev.model.clone()));
        meta.insert("provider".into(), serde_json::Value::String(ev.provider.clone()));
        meta.insert("tokens".into(), serde_json::Value::from(ev.tokens));
        meta.insert("duration_ms".into(), serde_json::Value::from(ev.duration_ms));
        meta.insert("status".into(), serde_json::Value::from(ev.status));
        meta.insert("ts".into(), serde_json::Value::String(ev.ts.clone()));
        if let Some(p) = &ev.prompt {
            meta.insert("prompt".into(), serde_json::Value::String(p.clone()));
        }
        snap.nodes.push(GraphNode {
            id: chat_id.clone(),
            kind: NodeKind::Chat,
            // Short label: the time portion of the timestamp if available.
            label: ev.ts.split('T').nth(1).unwrap_or(&ev.ts).to_string(),
            meta,
        });

        // chat -> model
        snap.edges.push(GraphEdge {
            source: chat_id,
            target: model_id.clone(),
            kind: EdgeKind::Used,
        });
        // model -> provider (deduped)
        if seen_served.insert((ev.model.clone(), ev.provider.clone())) {
            snap.edges.push(GraphEdge {
                source: model_id,
                target: provider_id,
                kind: EdgeKind::Served,
            });
        }
    }
}

fn add_project(snap: &mut GraphSnapshot, proj_dir: &Path) {
    let proj_name = proj_dir
        .file_name()
        .and_then(|n| n.to_str())
        .unwrap_or("project")
        .to_string();
    let proj_id = format!("project:{proj_name}");

    let memories = memory::scan_memory_dir(&proj_dir.join("memory"));
    let sessions = transcript::scan_transcript_dir(proj_dir);

    // Only emit a project node if it has content worth showing.
    if memories.is_empty() && sessions.is_empty() {
        return;
    }

    snap.nodes.push(GraphNode {
        id: proj_id.clone(),
        kind: NodeKind::Project,
        label: proj_name.clone(),
        meta: serde_json::Map::new(),
    });

    // Memory name -> node id, for resolving [[wikilinks]].
    let mut mem_ids = std::collections::HashMap::new();
    for m in &memories {
        let id = format!("memory:{}", m.name);
        mem_ids.insert(m.name.clone(), id.clone());
        let mut meta = serde_json::Map::new();
        if let Some(t) = &m.mem_type {
            meta.insert("type".into(), serde_json::Value::String(t.clone()));
        }
        meta.insert("description".into(), serde_json::Value::String(m.description.clone()));
        if !m.body.is_empty() {
            meta.insert("body".into(), serde_json::Value::String(m.body.clone()));
        }
        meta.insert("links".into(), serde_json::Value::from(m.links.len()));
        snap.nodes.push(GraphNode {
            id,
            kind: NodeKind::Memory,
            label: m.name.clone(),
            meta,
        });
    }
    // memory -> memory links (only to memories we actually have nodes for).
    for m in &memories {
        let src = format!("memory:{}", m.name);
        for link in &m.links {
            if let Some(target) = mem_ids.get(link) {
                snap.edges.push(GraphEdge {
                    source: src.clone(),
                    target: target.clone(),
                    kind: EdgeKind::Links,
                });
            }
        }
    }

    for s in &sessions {
        let sid = format!("session:{}", s.session_id);
        let command_total: u32 = s.commands.values().sum();
        let mut meta = serde_json::Map::new();
        meta.insert("project".into(), serde_json::Value::String(proj_name.clone()));
        meta.insert("command_total".into(), serde_json::Value::from(command_total));
        meta.insert("session_id".into(), serde_json::Value::String(s.session_id.clone()));
        snap.nodes.push(GraphNode {
            id: sid.clone(),
            kind: NodeKind::Session,
            label: s.label.clone(),
            meta,
        });
        snap.edges.push(GraphEdge {
            source: proj_id.clone(),
            target: sid.clone(),
            kind: EdgeKind::Owns,
        });

        // session -> command (aggregated by tool name)
        for (tool, count) in &s.commands {
            let cmd_id = format!("command:{}:{}", s.session_id, tool);
            let mut meta = serde_json::Map::new();
            meta.insert("count".into(), serde_json::Value::from(*count));
            snap.nodes.push(GraphNode {
                id: cmd_id.clone(),
                kind: NodeKind::Command,
                label: format!("{tool} ×{count}"),
                meta,
            });
            snap.edges.push(GraphEdge {
                source: sid.clone(),
                target: cmd_id,
                kind: EdgeKind::Ran,
            });
        }
    }

    // session -> memory (produced), matched via originSessionId.
    for m in &memories {
        let Some(origin) = &m.origin_session else { continue };
        let sid = format!("session:{origin}");
        if sessions.iter().any(|s| &s.session_id == origin) {
            snap.edges.push(GraphEdge {
                source: sid,
                target: format!("memory:{}", m.name),
                kind: EdgeKind::Produced,
            });
        }
    }
}

fn add_cluster(snap: &mut GraphSnapshot, topology: &TopologySnapshot) {
    for n in &topology.nodes {
        let mut meta = serde_json::Map::new();
        meta.insert("backend".into(), serde_json::Value::String(n.backend.clone()));
        meta.insert("layer_start".into(), serde_json::Value::from(n.layer_start));
        meta.insert("layer_end".into(), serde_json::Value::from(n.layer_end));
        snap.nodes.push(GraphNode {
            id: format!("cluster:{}", n.node_id),
            kind: NodeKind::Cluster,
            label: n.node_id.clone(),
            meta,
        });
    }
    // pipeline edges from next_node links — only when the target node exists,
    // so we never emit a dangling edge to a node absent from the topology.
    let node_ids: std::collections::HashSet<&str> =
        topology.nodes.iter().map(|n| n.node_id.as_str()).collect();
    for n in &topology.nodes {
        if let Some(next) = &n.next_node {
            if node_ids.contains(next.as_str()) {
                snap.edges.push(GraphEdge {
                    source: format!("cluster:{}", n.node_id),
                    target: format!("cluster:{next}"),
                    kind: EdgeKind::Pipeline,
                });
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use ai_engine_core::cluster_view::{NodeTopology, TopologySnapshot};
    use std::path::Path;

    fn projects_root() -> &'static Path {
        Path::new(concat!(env!("CARGO_MANIFEST_DIR"), "/tests/fixtures/projects"))
    }

    #[test]
    fn builds_nodes_and_edges_from_fixtures() {
        let topo = TopologySnapshot::default();
        let g = build_snapshot(projects_root(), &topo);

        // project + session + 2 memories + 2 commands (Bash, Skill)
        assert!(g.nodes.iter().any(|n| n.kind == NodeKind::Project));
        assert!(g.nodes.iter().any(|n| n.kind == NodeKind::Session));
        assert_eq!(g.nodes.iter().filter(|n| n.kind == NodeKind::Memory).count(), 2);
        assert_eq!(g.nodes.iter().filter(|n| n.kind == NodeKind::Command).count(), 2);

        // produced edge from session to the memory whose originSessionId matches
        assert!(g.edges.iter().any(|e| e.kind == EdgeKind::Produced));
        // links edge between the two memories
        assert!(g.edges.iter().any(|e| e.kind == EdgeKind::Links));
        // ran edges session -> command
        assert_eq!(g.edges.iter().filter(|e| e.kind == EdgeKind::Ran).count(), 2);
    }

    #[test]
    fn merges_cluster_topology() {
        let topo = TopologySnapshot {
            model_id: Some("m".into()),
            nodes: vec![NodeTopology {
                node_id: "node-a".into(),
                backend: "Cuda".into(),
                device_index: 0,
                available_memory_bytes: 0,
                compute_score: 0,
                link_mbps_to_leader: 0,
                layer_start: 0,
                layer_end: 1,
                hosts_embedding: true,
                hosts_output: false,
                previous_node: None,
                next_node: Some("node-b".into()),
            }],
        };
        let g = build_snapshot(Path::new("/no/such/dir"), &topo);
        assert!(g.nodes.iter().any(|n| n.kind == NodeKind::Cluster && n.id == "cluster:node-a"));
        // node-b is absent, so no pipeline edge should be emitted.
        assert!(!g.edges.iter().any(|e| e.kind == EdgeKind::Pipeline));
    }

    #[test]
    fn pipeline_edge_only_when_both_nodes_present() {
        fn cnode(id: &str, next: Option<&str>) -> NodeTopology {
            NodeTopology {
                node_id: id.into(),
                backend: "Cuda".into(),
                device_index: 0,
                available_memory_bytes: 0,
                compute_score: 0,
                link_mbps_to_leader: 0,
                layer_start: 0,
                layer_end: 1,
                hosts_embedding: false,
                hosts_output: false,
                previous_node: None,
                next_node: next.map(|s| s.to_string()),
            }
        }
        // node-a -> node-b, both present: one pipeline edge.
        let topo = TopologySnapshot {
            model_id: None,
            nodes: vec![cnode("node-a", Some("node-b")), cnode("node-b", None)],
        };
        let g = build_snapshot(Path::new("/no/such/dir"), &topo);
        assert_eq!(g.edges.iter().filter(|e| e.kind == EdgeKind::Pipeline).count(), 1);

        // node-a -> missing target: no pipeline edge (no dangling edge).
        let topo2 = TopologySnapshot {
            model_id: None,
            nodes: vec![cnode("node-a", Some("ghost"))],
        };
        let g2 = build_snapshot(Path::new("/no/such/dir"), &topo2);
        assert_eq!(g2.edges.iter().filter(|e| e.kind == EdgeKind::Pipeline).count(), 0);
    }

    #[test]
    fn missing_root_yields_only_cluster() {
        let g = build_snapshot(Path::new("/no/such/dir"), &TopologySnapshot::default());
        assert!(g.nodes.is_empty());
        assert!(g.edges.is_empty());
    }

    #[test]
    fn empty_snapshot_serializes_to_empty_arrays() {
        let json = serde_json::to_value(GraphSnapshot::default()).unwrap();
        assert_eq!(json, serde_json::json!({"nodes": [], "edges": []}));
    }

    #[test]
    fn node_kind_serializes_lowercase() {
        let json = serde_json::to_value(NodeKind::Session).unwrap();
        assert_eq!(json, serde_json::json!("session"));
    }
}
