use crate::capability::Capability;
use serde::{Deserialize, Serialize};
use sha2::{Digest, Sha256};
use std::ops::Range;

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct PartitionManifest {
    pub model_id: String,
    pub model_config_hash: [u8; 32],
    pub assignments: Vec<NodeAssignment>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct NodeAssignment {
    pub node_id: String,
    pub layer_range: Range<usize>,
    pub hosts_embedding: bool,
    pub hosts_output: bool,
    pub previous_node: Option<String>,
    pub next_node: Option<String>,
}

/// DP layer-cut solver. Capabilities provide pipeline order (config order from caller).
///
/// Memory cost per node: `assigned_layers * layer_bytes + per_node_overhead`
/// (per_node_overhead is KV cache budget). The leader (index 0) additionally
/// holds `embed_output_bytes` for embedding + output projection.
///
/// Minimizes: `max_i (assigned_layers_i / compute_score_i)` (transport overhead
/// is uniform and ignored for v0.2).
pub fn auto_partition(
    model_id: &str,
    caps: &[Capability],
    n_layers: usize,
    layer_bytes: u64,
    embed_output_bytes: u64,
    per_node_overhead: u64,
) -> anyhow::Result<PartitionManifest> {
    let n = caps.len();
    if n == 0 {
        anyhow::bail!("no nodes in cluster");
    }
    if n_layers == 0 {
        anyhow::bail!("model has 0 layers");
    }

    // Memory feasibility check per node: max layers that fit.
    let max_layers_per_node: Vec<usize> = caps
        .iter()
        .enumerate()
        .map(|(i, c)| {
            let overhead = if i == 0 {
                embed_output_bytes + per_node_overhead
            } else {
                per_node_overhead
            };
            if c.available_memory_bytes <= overhead {
                return 0;
            }
            ((c.available_memory_bytes - overhead) / layer_bytes) as usize
        })
        .collect();

    // Feasibility: total layer capacity must >= n_layers.
    let total_cap: usize = max_layers_per_node.iter().sum();
    if total_cap < n_layers {
        anyhow::bail!(
            "model {model_id} does not fit any partition across this cluster \
             (n_layers={n_layers}, total capacity={total_cap}, total memory={} bytes)",
            caps.iter().map(|c| c.available_memory_bytes).sum::<u64>(),
        );
    }

    // DP: cost[k][i] = minimum max-stage-cost of assigning first i layers
    // across first k nodes, subject to per-node memory caps.
    //
    // Transition:
    //   cost[k][i] = min over j in [0..i] of  max(cost[k-1][j], stage_cost(j..i, node_k))
    //   where stage_cost(j..i, node_k) = (i - j) * 1000 / node_k.compute_score
    //   (×1000 to keep arithmetic in integers since compute_score is also an integer)
    //
    // Boundary: cost[0][0] = 0; cost[0][i > 0] = infeasible.

    const INF: u64 = u64::MAX / 4;
    let mut cost = vec![vec![INF; n_layers + 1]; n + 1];
    let mut back = vec![vec![0usize; n_layers + 1]; n + 1];

    cost[0][0] = 0;
    for k in 1..=n {
        let node = &caps[k - 1];
        let max_l = max_layers_per_node[k - 1];
        for i in 0..=n_layers {
            for j in 0..=i {
                if cost[k - 1][j] == INF {
                    continue;
                }
                let assigned = i - j;
                if assigned > max_l {
                    continue;
                }
                let stage = if node.compute_score == 0 {
                    INF
                } else {
                    (assigned as u64) * 1000 / (node.compute_score as u64)
                };
                let candidate = cost[k - 1][j].max(stage);
                if candidate < cost[k][i] {
                    cost[k][i] = candidate;
                    back[k][i] = j;
                }
            }
        }
    }

    if cost[n][n_layers] == INF {
        anyhow::bail!(
            "model {model_id} does not fit any partition (DP found no feasible assignment)"
        );
    }

    // Recover assignment by backtracking.
    let mut cuts = vec![n_layers];
    let mut cur = n_layers;
    for k in (1..=n).rev() {
        let prev = back[k][cur];
        cuts.push(prev);
        cur = prev;
    }
    cuts.reverse(); // cuts = [0, c1, c2, ..., n_layers]

    let mut assignments = Vec::with_capacity(n);
    for (idx, win) in cuts.windows(2).enumerate() {
        let start = win[0];
        let end = win[1];
        let prev = if idx == 0 {
            None
        } else {
            Some(caps[idx - 1].node_id.clone())
        };
        let next = if idx + 1 == n {
            None
        } else {
            Some(caps[idx + 1].node_id.clone())
        };
        assignments.push(NodeAssignment {
            node_id: caps[idx].node_id.clone(),
            layer_range: start..end,
            hosts_embedding: idx == 0,
            hosts_output: idx + 1 == n,
            previous_node: prev,
            next_node: next,
        });
    }

    // Content addressing: hash (model_id, n_layers, [(node_id, layer_range)]) so
    // a given input yields a stable hash.
    let model_config_hash = compute_manifest_hash(model_id, n_layers, &assignments);

    Ok(PartitionManifest {
        model_id: model_id.into(),
        model_config_hash,
        assignments,
    })
}

/// Build a manifest from an explicit (node_id, layer_range) list.
/// Validates: contiguous, non-overlapping, complete cover of 0..n_layers.
/// Also validates memory feasibility.
pub fn manual_partition(
    model_id: &str,
    caps: &[Capability],
    n_layers: usize,
    ranges: Vec<(String, Range<usize>)>,
    layer_bytes: u64,
    embed_output_bytes: u64,
    per_node_overhead: u64,
) -> anyhow::Result<PartitionManifest> {
    // Sort by start.
    let mut sorted: Vec<(String, Range<usize>)> = ranges;
    sorted.sort_by_key(|(_, r)| r.start);

    // Contiguous + complete cover.
    let mut expected_start = 0;
    for (node, r) in &sorted {
        if r.start != expected_start {
            anyhow::bail!(
                "partition_override is not contiguous: expected start={}, got {}..{} for node {}",
                expected_start,
                r.start,
                r.end,
                node
            );
        }
        if r.start >= r.end {
            anyhow::bail!("empty range {}..{} for node {}", r.start, r.end, node);
        }
        expected_start = r.end;
    }
    if expected_start != n_layers {
        anyhow::bail!(
            "partition_override does not cover all layers: covered {} of {}",
            expected_start,
            n_layers
        );
    }

    // Memory feasibility per node.
    for (i, (node, r)) in sorted.iter().enumerate() {
        let cap = caps
            .iter()
            .find(|c| &c.node_id == node)
            .ok_or_else(|| anyhow::anyhow!("unknown node {node} in partition_override"))?;
        let overhead = if i == 0 {
            embed_output_bytes + per_node_overhead
        } else {
            per_node_overhead
        };
        let need = (r.len() as u64) * layer_bytes + overhead;
        if need > cap.available_memory_bytes {
            anyhow::bail!(
                "node {} layers {}..{} need {} bytes but only {} available",
                node,
                r.start,
                r.end,
                need,
                cap.available_memory_bytes
            );
        }
    }

    let n = sorted.len();
    let mut assignments: Vec<NodeAssignment> = Vec::with_capacity(n);
    for (idx, (node, range)) in sorted.into_iter().enumerate() {
        let prev = if idx == 0 {
            None
        } else {
            Some(assignments[idx - 1].node_id.clone())
        };
        // next is filled in second pass after we know all node_ids.
        assignments.push(NodeAssignment {
            node_id: node,
            layer_range: range,
            hosts_embedding: idx == 0,
            hosts_output: idx + 1 == n,
            previous_node: prev,
            next_node: None,
        });
    }
    for i in 0..(assignments.len().saturating_sub(1)) {
        assignments[i].next_node = Some(assignments[i + 1].node_id.clone());
    }

    let model_config_hash = compute_manifest_hash(model_id, n_layers, &assignments);
    Ok(PartitionManifest {
        model_id: model_id.into(),
        model_config_hash,
        assignments,
    })
}

pub fn compute_manifest_hash(
    model_id: &str,
    n_layers: usize,
    assignments: &[NodeAssignment],
) -> [u8; 32] {
    let mut h = Sha256::new();
    h.update(model_id.as_bytes());
    h.update((n_layers as u64).to_le_bytes());
    for a in assignments {
        h.update(a.node_id.as_bytes());
        h.update((a.layer_range.start as u64).to_le_bytes());
        h.update((a.layer_range.end as u64).to_le_bytes());
    }
    let digest = h.finalize();
    let mut out = [0u8; 32];
    out.copy_from_slice(&digest);
    out
}
