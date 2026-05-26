import { render, screen } from "@testing-library/react";
import { NodeCards } from "./Cluster";
import type { NodeTopology } from "../api";

const nodes: NodeTopology[] = [
  {
    node_id: "node-a",
    backend: "Cuda",
    device_index: 0,
    available_memory_bytes: 8_000_000_000,
    compute_score: 100,
    link_mbps_to_leader: 1000,
    layer_start: 0,
    layer_end: 16,
    hosts_embedding: true,
    hosts_output: false,
    previous_node: null,
    next_node: "node-b",
  },
];

test("renders a node card with id and layer range", () => {
  render(<NodeCards nodes={nodes} tps={{ "node-a": 42 }} />);
  expect(screen.getByText("node-a")).toBeTruthy();
  expect(screen.getByText(/layers 0–16/)).toBeTruthy();
  expect(screen.getByText(/42\.0 tok\/s/)).toBeTruthy();
});

test("renders empty state with no nodes", () => {
  render(<NodeCards nodes={[]} tps={{}} />);
  expect(screen.getByText(/no cluster/i)).toBeTruthy();
});
