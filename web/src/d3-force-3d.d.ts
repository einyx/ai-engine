// d3-force-3d ships no TypeScript types; we only use forceCollide. This minimal
// declaration keeps `tsc` happy without pulling a full @types package.
declare module "d3-force-3d" {
  export function forceCollide(radius?: number | ((node: unknown) => number)): unknown;
}
