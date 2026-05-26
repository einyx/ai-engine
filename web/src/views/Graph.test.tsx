import { render, screen, waitFor } from "@testing-library/react";
import { describe, it, expect, vi, beforeEach } from "vitest";
import { Graph } from "./Graph";

// react-force-graph-2d uses canvas/WebGL which jsdom lacks; stub it so the
// test exercises our data-mapping + empty-state logic, not the renderer.
vi.mock("react-force-graph-2d", () => ({
  default: ({ graphData }: { graphData: { nodes: unknown[] } }) => (
    <div data-testid="fg">{graphData.nodes.length} nodes</div>
  ),
}));

describe("Graph", () => {
  beforeEach(() => {
    vi.restoreAllMocks();
  });

  it("renders node count from /graph", async () => {
    vi.stubGlobal(
      "fetch",
      vi.fn().mockResolvedValue({
        ok: true,
        json: async () => ({
          nodes: [{ id: "a", kind: "session", label: "s" }],
          edges: [],
        }),
      }),
    );
    render(<Graph />);
    await waitFor(() => expect(screen.getByTestId("fg")).toBeTruthy());
    expect(screen.getByTestId("fg").textContent).toBe("1 nodes");
  });

  it("shows empty state when graph has no nodes", async () => {
    vi.stubGlobal(
      "fetch",
      vi.fn().mockResolvedValue({
        ok: true,
        json: async () => ({ nodes: [], edges: [] }),
      }),
    );
    render(<Graph />);
    await waitFor(() => expect(screen.getByText(/no graph data/i)).toBeTruthy());
  });
});
