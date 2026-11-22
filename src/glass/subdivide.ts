// Loop subdivision of a triangle mesh, the smooth surface a subdivision modifier (such as
// Blender's) shows, including the boundary rules that keep open edges (Suzanne's eye holes
// and the backs of her eye cups) as smooth curves instead of letting them shrink away.
// Shared by the renderer (a smooth close-up mesh) and the offline volume bake
// (scripts/bake-suzanne-sdf.ts), so both describe the same surface. Kept free of imports so
// Node can run the bake straight from this TypeScript file.

export interface IndexedMesh {
  /** xyz per vertex. */
  positions: ArrayLike<number>;
  /** Three vertex indices per triangle. */
  indices: ArrayLike<number>;
}

/** One level of Loop subdivision (each triangle into four). */
function subdivideOnce(mesh: IndexedMesh): { positions: Float64Array; indices: Uint32Array } {
  const { positions, indices } = mesh;
  const vertexCount = positions.length / 3;
  const triCount = indices.length / 3;

  // Each edge: its two ends, the vertices opposite it (2 inside the mesh, 1 on an open edge),
  // and the index of the new vertex at its middle.
  const edgeIndex = new Map<number, number>();
  const edgeEnds: number[] = [];
  const edgeOpposite: number[][] = [];
  const key = (a: number, b: number) => (a < b ? a * vertexCount + b : b * vertexCount + a);
  const edgeOf = (a: number, b: number, opposite: number) => {
    const k = key(a, b);
    let e = edgeIndex.get(k);
    if (e === undefined) {
      e = edgeEnds.length / 2;
      edgeIndex.set(k, e);
      edgeEnds.push(a, b);
      edgeOpposite.push([]);
    }
    edgeOpposite[e].push(opposite);
    return e;
  };
  const triEdges = new Uint32Array(triCount * 3);
  for (let t = 0; t < triCount; t++) {
    const a = indices[t * 3];
    const b = indices[t * 3 + 1];
    const c = indices[t * 3 + 2];
    triEdges[t * 3] = edgeOf(a, b, c);
    triEdges[t * 3 + 1] = edgeOf(b, c, a);
    triEdges[t * 3 + 2] = edgeOf(c, a, b);
  }
  const edgeCount = edgeEnds.length / 2;

  // Neighbours of each vertex, and the ones joined to it by an open (boundary) edge.
  const neighbours: number[][] = Array.from({ length: vertexCount }, () => []);
  const boundaryNeighbours: number[][] = Array.from({ length: vertexCount }, () => []);
  for (let e = 0; e < edgeCount; e++) {
    const a = edgeEnds[e * 2];
    const b = edgeEnds[e * 2 + 1];
    neighbours[a].push(b);
    neighbours[b].push(a);
    if (edgeOpposite[e].length === 1) {
      boundaryNeighbours[a].push(b);
      boundaryNeighbours[b].push(a);
    }
  }

  const out = new Float64Array((vertexCount + edgeCount) * 3);
  // Old vertices move toward their neighbours (on an open edge, only along it).
  for (let v = 0; v < vertexCount; v++) {
    const ring = boundaryNeighbours[v].length === 2 ? boundaryNeighbours[v] : neighbours[v];
    const onBoundary = boundaryNeighbours[v].length === 2;
    const n = ring.length;
    const beta = onBoundary ? 1 / 8 : n === 3 ? 3 / 16 : 3 / (8 * n);
    const self = 1 - n * beta;
    for (let d = 0; d < 3; d++) {
      let sum = 0;
      for (const w of ring) sum += positions[w * 3 + d];
      out[v * 3 + d] = self * positions[v * 3 + d] + beta * sum;
    }
  }
  // New vertices at the middle of each edge.
  for (let e = 0; e < edgeCount; e++) {
    const a = edgeEnds[e * 2];
    const b = edgeEnds[e * 2 + 1];
    const opp = edgeOpposite[e];
    const o = (vertexCount + e) * 3;
    for (let d = 0; d < 3; d++) {
      const ends = positions[a * 3 + d] + positions[b * 3 + d];
      out[o + d] =
        opp.length === 2 ? 0.375 * ends + 0.125 * (positions[opp[0] * 3 + d] + positions[opp[1] * 3 + d]) : 0.5 * ends;
    }
  }

  const tris = new Uint32Array(triCount * 12);
  for (let t = 0; t < triCount; t++) {
    const a = indices[t * 3];
    const b = indices[t * 3 + 1];
    const c = indices[t * 3 + 2];
    const ab = vertexCount + triEdges[t * 3];
    const bc = vertexCount + triEdges[t * 3 + 1];
    const ca = vertexCount + triEdges[t * 3 + 2];
    tris.set([a, ab, ca, b, bc, ab, c, ca, bc, ab, bc, ca], t * 12);
  }
  return { positions: out, indices: tris };
}

/** `levels` levels of Loop subdivision (each multiplies the triangle count by 4). */
export function loopSubdivide(mesh: IndexedMesh, levels: number): { positions: Float64Array; indices: Uint32Array } {
  let current: { positions: Float64Array; indices: Uint32Array } = {
    positions: Float64Array.from(mesh.positions),
    indices: Uint32Array.from(mesh.indices),
  };
  for (let i = 0; i < levels; i++) current = subdivideOnce(current);
  return current;
}
