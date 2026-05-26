# Design: System Knowledge Graph (Graph tab)

## Goal
A third "Graph" tab in the ai-engine UI rendering a force-directed graph that
unifies four worlds into one heterogeneous knowledge graph:
inference **cluster** + Claude Code **sessions**, **memories**, and **commands**.

## Data sources (all local to the host running the leader)
| Node type | Source | Notes |
|-----------|--------|-------|
| `cluster` | `state.cluster.topology()` (in-process) | already served by `/cluster/topology` |
| `session` | transcript `*.jsonl` filenames / `sessionId` | one per session file |
| `memory`  | memory dir `*.md` frontmatter (`type`, `[[links]]`) | already graph-shaped |
| `command` | transcript tool-call events, **aggregated by (session, tool-kind)** | e.g. `Bash ×142` |

Scan strategy: **auto-discover** — glob `~/.claude/projects/*/` and merge sessions,
memories, and transcripts across all projects into one graph (plus each project's
`.remember/` if present). Root overridable via `AI_ENGINE_GRAPH_ROOT`.
**Absent/unreadable → empty graph** (mirrors gateway-only topology fallback).
Add a `project` node kind so cross-project nodes stay visually grouped
(`project --owns--> session`), keeping the merged view legible.

## Edges
- `session --produced--> memory`  (frontmatter `originSessionId`)
- `memory  --links-----> memory`  (`[[wikilink]]` bodies)
- `session --ran-------> command` (aggregated tool events)
- `cluster --pipeline--> cluster` (topology edges)
- `session --touched---> cluster` (v1: only if session hit cluster endpoints; likely empty — keep edge type, populate later)

## Backend
- New crate module `ai-engine-graph` (scanner) — pure, file-in / `GraphSnapshot`-out, fully unit-testable against fixture dirs.
- New route `GET /graph` in `ai-engine-http` → `Json(GraphSnapshot)`.
- `GraphSnapshot { nodes: Vec<GraphNode>, edges: Vec<GraphEdge> }`; nodes carry
  `id, kind, label, meta` so the frontend can color/size by `kind`.

## Frontend
- `react-force-graph-2d` (canvas, d3-force; glowing nodes, particle links, draggable).
- New view `web/src/views/Graph.tsx`; `fetchGraph()` in `api.ts`; tab wired in `App.tsx`.
- Color by `kind`, size by degree, link particles for `pipeline`/`ran` edges.

## Scope guardrails (v1)
- Commands aggregated by type — no per-call nodes.
- `/graph` is request/response (no SSE); refresh button re-fetches.
- Scanner is read-only and best-effort: unreadable/missing files are skipped, never fatal.
