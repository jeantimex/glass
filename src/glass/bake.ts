// Turn any triangle mesh into a solid of glass: a 128^3 signed-distance volume that the glass
// tracer follows light through (exact exits on every internal bounce), for MeshGlassMaterial's
// `volume` and the instanced renderer's baked shapes.
//
// Real-world meshes are rarely clean solids (Blender's Suzanne is an open head with eye cups
// plugged into holes; the Utah teapot is a set of open patches), so the bake makes them solid
// the way a cast glass object would be:
//   1. weld vertices that share a position, drop degenerate triangles;
//   2. split the mesh into its connected pieces;
//   3. optionally Loop-subdivide each piece (the smooth surface a subdivision modifier shows);
//   4. cap every hole of every piece, so each is a closed solid;
//   5. compute each piece's exact signed distance (inside by ray parity, distance to its
//      triangles in a band around the surface, carried outwards by closest-point sweeps);
//   6. merge the pieces (their union, optionally with a smooth fillet where they meet).
//
// Layout (as src/glass/tsl/sdf.ts reads it): HalfFloat, x fastest then y then z, texel i at
// q = ((i + 0.5) / 128 - 0.5) * 2.2 in volume units, distances in volume units, negative inside.
// Volume units are the mesh's own coordinates moved by -`center` and divided by `halfWidth`
// (by default fitted so the mesh spans [-1, 1] along its longest axis).
//
// Dependency-free (besides subdivide.ts), so it runs in Node (scripts/) and in the browser
// (src/glass/volume.ts, in a worker).
import { loopSubdivide } from './subdivide.ts';

export const VOLUME_SIZE = 128;
/** The volume covers [-1.1, 1.1] volume units on each axis. */
export const VOLUME_EXTENT = 2.2;

export interface BakeOptions {
  /** Loop subdivision levels per piece before baking (default 0). */
  subdivisions?: number;
  /** Smooth-union radius where pieces meet, in volume units (default 0: their plain union). */
  fillet?: number;
  /** Mesh point at the volume's centre (default: the middle of its bounding box). */
  center?: [number, number, number];
  /** Mesh units per volume unit (default: half the bounding box's longest side). */
  halfWidth?: number;
  /** Vertices closer than this (mesh units) are welded (default: 1e-6 of the mesh's size). */
  weldTolerance?: number;
  /** Width of the exactly computed band around the surface, in voxels (default 3). */
  band?: number;
}

export interface BakeResult {
  /** 128^3 HalfFloat signed distances (see the layout above). */
  data: Uint16Array;
  center: [number, number, number];
  halfWidth: number;
  stats: {
    pieces: number;
    holesCapped: number;
    /** Columns whose inside test stayed ambiguous (left outside); 0 for a sound mesh. */
    degenerateColumns: number;
    triangles: number;
  };
}

const SIZE = VOLUME_SIZE;
const EXTENT = VOLUME_EXTENT;

/** An undirected edge's key. */
const key = (a: number, b: number) => (a < b ? `${a}_${b}` : `${b}_${a}`);

/** Close every hole of a piece with a fan of triangles to the hole's centroid. */
function capHoles(tris: number[], positions: number[]): { tris: number[]; holes: number } {
  const count = new Map<string, number>();
  for (let t = 0; t < tris.length; t += 3) {
    for (let e = 0; e < 3; e++) {
      const k = key(tris[t + e], tris[t + ((e + 1) % 3)]);
      count.set(k, (count.get(k) ?? 0) + 1);
    }
  }
  const next = new Map<number, number>();
  for (let t = 0; t < tris.length; t += 3) {
    for (let e = 0; e < 3; e++) {
      const a = tris[t + e];
      const b = tris[t + ((e + 1) % 3)];
      if (count.get(key(a, b)) === 1) next.set(a, b);
    }
  }
  const out = [...tris];
  let holes = 0;
  const visited = new Set<number>();
  for (const start of next.keys()) {
    if (visited.has(start)) continue;
    const loop: number[] = [];
    for (let v = start; !visited.has(v); v = next.get(v)!) {
      visited.add(v);
      loop.push(v);
    }
    const c = [0, 0, 0];
    for (const v of loop) for (let d = 0; d < 3; d++) c[d] += positions[v * 3 + d] / loop.length;
    const centre = positions.length / 3;
    positions.push(c[0], c[1], c[2]);
    for (let i = 0; i < loop.length; i++) out.push(loop[(i + 1) % loop.length], loop[i], centre);
    holes++;
  }
  return { tris: out, holes };
}

/* Closest point on a triangle (Ericson, Real-Time Collision Detection 5.1.5): squared distance. */
function triDist2(px: number, py: number, pz: number, tri: Float64Array, o: number): number {
  const ax = tri[o],
    ay = tri[o + 1],
    az = tri[o + 2];
  const abx = tri[o + 3] - ax,
    aby = tri[o + 4] - ay,
    abz = tri[o + 5] - az;
  const acx = tri[o + 6] - ax,
    acy = tri[o + 7] - ay,
    acz = tri[o + 8] - az;
  const apx = px - ax,
    apy = py - ay,
    apz = pz - az;
  const d1 = abx * apx + aby * apy + abz * apz;
  const d2 = acx * apx + acy * apy + acz * apz;
  let qx: number, qy: number, qz: number;
  if (d1 <= 0 && d2 <= 0) {
    qx = ax;
    qy = ay;
    qz = az;
  } else {
    const bpx = px - tri[o + 3],
      bpy = py - tri[o + 4],
      bpz = pz - tri[o + 5];
    const d3 = abx * bpx + aby * bpy + abz * bpz;
    const d4 = acx * bpx + acy * bpy + acz * bpz;
    const cpx = px - tri[o + 6],
      cpy = py - tri[o + 7],
      cpz = pz - tri[o + 8];
    const d5 = abx * cpx + aby * cpy + abz * cpz;
    const d6 = acx * cpx + acy * cpy + acz * cpz;
    const vc = d1 * d4 - d3 * d2;
    const vb = d5 * d2 - d1 * d6;
    const va = d3 * d6 - d5 * d4;
    if (d3 >= 0 && d4 <= d3) {
      qx = tri[o + 3];
      qy = tri[o + 4];
      qz = tri[o + 5];
    } else if (d6 >= 0 && d5 <= d6) {
      qx = tri[o + 6];
      qy = tri[o + 7];
      qz = tri[o + 8];
    } else if (vc <= 0 && d1 >= 0 && d3 <= 0) {
      const v = d1 / (d1 - d3);
      qx = ax + abx * v;
      qy = ay + aby * v;
      qz = az + abz * v;
    } else if (vb <= 0 && d2 >= 0 && d6 <= 0) {
      const w = d2 / (d2 - d6);
      qx = ax + acx * w;
      qy = ay + acy * w;
      qz = az + acz * w;
    } else if (va <= 0 && d4 - d3 >= 0 && d5 - d6 >= 0) {
      const w = (d4 - d3) / (d4 - d3 + (d5 - d6));
      qx = tri[o + 3] + (tri[o + 6] - tri[o + 3]) * w;
      qy = tri[o + 4] + (tri[o + 7] - tri[o + 4]) * w;
      qz = tri[o + 5] + (tri[o + 8] - tri[o + 5]) * w;
    } else {
      const denom = 1 / (va + vb + vc);
      const v = vb * denom;
      const w = vc * denom;
      qx = ax + abx * v + acx * w;
      qy = ay + aby * v + acy * w;
      qz = az + abz * v + acz * w;
    }
  }
  const dx = px - qx,
    dy = py - qy,
    dz = pz - qz;
  return dx * dx + dy * dy + dz * dz;
}

const coord = (i: number) => ((i + 0.5) / SIZE - 0.5) * EXTENT;
const cell = (v: number) => (v / EXTENT + 0.5) * SIZE - 0.5;
const N3 = SIZE * SIZE * SIZE;

/** Signed distance field (unit-size units) of one closed piece. */
function pieceField(tris: number[], positions: number[], band: number): { field: Float32Array; degenerate: number } {
  const triCount = tris.length / 3;
  const tri = new Float64Array(triCount * 9);
  for (let t = 0; t < triCount; t++) {
    for (let k = 0; k < 3; k++) for (let d = 0; d < 3; d++) tri[t * 9 + k * 3 + d] = positions[tris[t * 3 + k] * 3 + d];
  }

  // Inside: parity of crossings along +z through each (x, y) column (nudged off the exact
  // texel centre so a column never runs exactly along a shared edge).
  const inside = new Uint8Array(N3);
  const hits: number[][] = Array.from({ length: SIZE * SIZE }, () => []);
  const nudgeX = 1.37e-5;
  const nudgeY = 2.11e-5;
  for (let t = 0; t < triCount; t++) {
    const o = t * 9;
    const xs = [tri[o], tri[o + 3], tri[o + 6]];
    const ys = [tri[o + 1], tri[o + 4], tri[o + 7]];
    const i0 = Math.max(0, Math.ceil(cell(Math.min(...xs)) - 1e-9));
    const i1 = Math.min(SIZE - 1, Math.floor(cell(Math.max(...xs))));
    const j0 = Math.max(0, Math.ceil(cell(Math.min(...ys)) - 1e-9));
    const j1 = Math.min(SIZE - 1, Math.floor(cell(Math.max(...ys))));
    for (let j = j0; j <= j1; j++) {
      const py = coord(j) + nudgeY;
      for (let i = i0; i <= i1; i++) {
        const px = coord(i) + nudgeX;
        // Barycentric coordinates of (px, py) in the triangle's xy projection.
        const x0 = xs[0],
          y0 = ys[0];
        const e1x = xs[1] - x0,
          e1y = ys[1] - y0,
          e2x = xs[2] - x0,
          e2y = ys[2] - y0;
        const det = e1x * e2y - e2x * e1y;
        if (Math.abs(det) < 1e-14) continue;
        const u = ((px - x0) * e2y - e2x * (py - y0)) / det;
        const v = (e1x * (py - y0) - (px - x0) * e1y) / det;
        if (u < 0 || v < 0 || u + v > 1) continue;
        hits[i + j * SIZE].push(tri[o + 2] + (tri[o + 5] - tri[o + 2]) * u + (tri[o + 8] - tri[o + 2]) * v);
      }
    }
  }
  /** Crossings of every triangle by the column at (px, py) (for columns to cast again). */
  const castColumn = (px: number, py: number): number[] => {
    const zs: number[] = [];
    for (let t = 0; t < triCount; t++) {
      const o = t * 9;
      const x0 = tri[o],
        y0 = tri[o + 1];
      const e1x = tri[o + 3] - x0,
        e1y = tri[o + 4] - y0,
        e2x = tri[o + 6] - x0,
        e2y = tri[o + 7] - y0;
      const det = e1x * e2y - e2x * e1y;
      if (Math.abs(det) < 1e-14) continue;
      const u = ((px - x0) * e2y - e2x * (py - y0)) / det;
      const v = (e1x * (py - y0) - (px - x0) * e1y) / det;
      if (u < 0 || v < 0 || u + v > 1) continue;
      zs.push(tri[o + 2] + (tri[o + 5] - tri[o + 2]) * u + (tri[o + 8] - tri[o + 2]) * v);
    }
    return zs;
  };
  let recast = 0;
  let oddColumns = 0;
  for (let c = 0; c < SIZE * SIZE; c++) {
    let zs = hits[c].sort((a, b) => a - b);
    // A column through an edge or vertex can count a crossing twice or not at all: cast it
    // again a little to the side until the count is even (closed surfaces cross evenly).
    for (let attempt = 1; zs.length % 2 && attempt <= 8; attempt++) {
      recast++;
      const px = coord(c % SIZE) + nudgeX * (1 + attempt * 7.3);
      const py = coord(Math.floor(c / SIZE)) - nudgeY * (1 + attempt * 5.1);
      zs = castColumn(px, py).sort((a, b) => a - b);
    }
    if (zs.length % 2) {
      oddColumns++;
      continue; // (still degenerate: leave the column outside; reported below)
    }
    for (let h = 0; h < zs.length; h += 2) {
      const k0 = Math.max(0, Math.ceil(cell(zs[h])));
      const k1 = Math.min(SIZE - 1, Math.floor(cell(zs[h + 1])));
      for (let k = k0; k <= k1; k++) inside[c + k * SIZE * SIZE] = 1;
    }
  }

  // Unsigned distance: exact in a band around every triangle, then carried outwards by
  // sweeping each texel's nearest triangle to its neighbours (a closest-point transform).
  const dist2 = new Float64Array(N3).fill(Infinity);
  const nearest = new Int32Array(N3).fill(-1);
  for (let t = 0; t < triCount; t++) {
    const o = t * 9;
    const lo = [0, 1, 2].map((d) =>
      Math.max(0, Math.floor(cell(Math.min(tri[o + d], tri[o + 3 + d], tri[o + 6 + d]))) - band),
    );
    const hi = [0, 1, 2].map((d) =>
      Math.min(SIZE - 1, Math.ceil(cell(Math.max(tri[o + d], tri[o + 3 + d], tri[o + 6 + d]))) + band),
    );
    for (let k = lo[2]; k <= hi[2]; k++) {
      for (let j = lo[1]; j <= hi[1]; j++) {
        for (let i = lo[0]; i <= hi[0]; i++) {
          const idx = i + j * SIZE + k * SIZE * SIZE;
          const d2 = triDist2(coord(i), coord(j), coord(k), tri, o);
          if (d2 < dist2[idx]) {
            dist2[idx] = d2;
            nearest[idx] = t;
          }
        }
      }
    }
  }
  const neighbours: [number, number, number][] = [];
  for (let dz = -1; dz <= 1; dz++)
    for (let dy = -1; dy <= 1; dy++) for (let dx = -1; dx <= 1; dx++) if (dx || dy || dz) neighbours.push([dx, dy, dz]);
  const sweep = (forward: boolean) => {
    const order = forward ? neighbours.slice(0, 13) : neighbours.slice(13);
    for (let s = 0; s < N3; s++) {
      const idx = forward ? s : N3 - 1 - s;
      const i = idx % SIZE;
      const j = Math.floor(idx / SIZE) % SIZE;
      const k = Math.floor(idx / (SIZE * SIZE));
      for (const [dx, dy, dz] of order) {
        const ni = i + dx,
          nj = j + dy,
          nk = k + dz;
        if (ni < 0 || nj < 0 || nk < 0 || ni >= SIZE || nj >= SIZE || nk >= SIZE) continue;
        const t = nearest[ni + nj * SIZE + nk * SIZE * SIZE];
        if (t < 0 || t === nearest[idx]) continue;
        const d2 = triDist2(coord(i), coord(j), coord(k), tri, t * 9);
        if (d2 < dist2[idx]) {
          dist2[idx] = d2;
          nearest[idx] = t;
        }
      }
    }
  };
  for (let pass = 0; pass < 3; pass++) {
    sweep(true);
    sweep(false);
  }

  const field = new Float32Array(N3);
  for (let idx = 0; idx < N3; idx++) field[idx] = (inside[idx] ? -1 : 1) * Math.sqrt(dist2[idx]);
  void recast;
  return { field, degenerate: oddColumns };
}

/** Polynomial smooth minimum (Quilez): a fillet of radius ~k where two surfaces meet. */
function smoothMin(a: number, b: number, k: number): number {
  if (k <= 0) return Math.min(a, b);
  const h = Math.max(k - Math.abs(a - b), 0) / k;
  return Math.min(a, b) - h * h * k * 0.25;
}

/** Exact float32 -> IEEE half conversion (round to nearest even). */
const f32 = new Float32Array(1);
const u32 = new Uint32Array(f32.buffer);
function toHalf(value: number): number {
  f32[0] = value;
  const x = u32[0];
  const sign = (x >>> 16) & 0x8000;
  const exp = ((x >>> 23) & 0xff) - 127 + 15;
  const mant = x & 0x7fffff;
  if (exp <= 0) {
    if (exp < -10) return sign;
    return sign | ((((mant | 0x800000) >>> (1 - exp)) + 0x1000) >>> 13);
  }
  if (exp >= 31) return sign | 0x7c00;
  const h = sign | (exp << 10) | (mant >>> 13);
  const rest = mant & 0x1fff;
  return rest > 0x1000 || (rest === 0x1000 && h & 1) ? h + 1 : h;
}

/**
 * Bake the signed-distance volume of the solid a mesh describes. `positions`: xyz per vertex
 * (mesh units); `indices`: three per triangle (or null: every three vertices are a triangle).
 */
export function bakeGlassVolume(
  positions: ArrayLike<number>,
  indices: ArrayLike<number> | null,
  options: BakeOptions = {},
): BakeResult {
  const vertexCount = positions.length / 3;
  const index: ArrayLike<number> = indices ?? Array.from({ length: vertexCount }, (_, i) => i);

  /* Fit the mesh into the volume. */
  const lo = [Infinity, Infinity, Infinity];
  const hi = [-Infinity, -Infinity, -Infinity];
  for (let v = 0; v < vertexCount; v++) {
    for (let d = 0; d < 3; d++) {
      lo[d] = Math.min(lo[d], positions[v * 3 + d]);
      hi[d] = Math.max(hi[d], positions[v * 3 + d]);
    }
  }
  const center = options.center ?? ([0, 1, 2].map((d) => (lo[d] + hi[d]) / 2) as [number, number, number]);
  const halfWidth = options.halfWidth ?? Math.max((hi[0] - lo[0]) / 2, (hi[1] - lo[1]) / 2, (hi[2] - lo[2]) / 2, 1e-9);

  /* 1. Weld (each vertex onto the first one at its position) and drop degenerate triangles. */
  const tolerance = options.weldTolerance ?? 1e-6 * halfWidth;
  const firstAt = new Map<string, number>();
  const weld = new Uint32Array(vertexCount);
  for (let v = 0; v < vertexCount; v++) {
    const k = `${Math.round(positions[v * 3] / tolerance)},${Math.round(positions[v * 3 + 1] / tolerance)},${Math.round(positions[v * 3 + 2] / tolerance)}`;
    let first = firstAt.get(k);
    if (first === undefined) {
      first = v;
      firstAt.set(k, v);
    }
    weld[v] = first;
  }
  const tris: number[] = [];
  for (let t = 0; t + 2 < index.length; t += 3) {
    const a = weld[index[t]];
    const b = weld[index[t + 1]];
    const c = weld[index[t + 2]];
    if (a !== b && b !== c && c !== a) tris.push(a, b, c);
  }

  /* 2. Connected pieces. */
  const parent = Array.from({ length: vertexCount }, (_, i) => i);
  const find = (x: number): number => (parent[x] === x ? x : (parent[x] = find(parent[x])));
  for (let t = 0; t < tris.length; t += 3) {
    parent[find(tris[t + 1])] = find(tris[t]);
    parent[find(tris[t + 2])] = find(tris[t]);
  }
  const pieces = new Map<number, number[]>();
  for (let t = 0; t < tris.length; t += 3) {
    const root = find(tris[t]);
    if (!pieces.has(root)) pieces.set(root, []);
    pieces.get(root)!.push(tris[t], tris[t + 1], tris[t + 2]);
  }

  /* 3. Each piece in volume units (subdivided while still open), into one vertex list. */
  const subdivisions = options.subdivisions ?? 0;
  const volumePositions: number[] = [];
  const pieceTris = [...pieces.values()].map((pieceIndices) => {
    const local = new Map<number, number>();
    const localPositions: number[] = [];
    const localIndices: number[] = [];
    for (const v of pieceIndices) {
      let l = local.get(v);
      if (l === undefined) {
        l = localPositions.length / 3;
        local.set(v, l);
        for (let d = 0; d < 3; d++) localPositions.push((positions[v * 3 + d] - center[d]) / halfWidth);
      }
      localIndices.push(l);
    }
    const smooth =
      subdivisions > 0
        ? loopSubdivide({ positions: localPositions, indices: localIndices }, subdivisions)
        : { positions: localPositions, indices: localIndices };
    const base = volumePositions.length / 3;
    for (const x of smooth.positions) volumePositions.push(x);
    return Array.from(smooth.indices, (i) => i + base);
  });

  /* 4-5. Close each piece and compute its signed distance (largest first). */
  let holesCapped = 0;
  let degenerateColumns = 0;
  let triangles = 0;
  const fields: Float32Array[] = [];
  for (const piece of pieceTris.sort((a, b) => b.length - a.length)) {
    const { tris: closed, holes } = capHoles(piece, volumePositions);
    holesCapped += holes;
    triangles += closed.length / 3;
    const { field, degenerate } = pieceField(closed, volumePositions, options.band ?? 3);
    degenerateColumns += degenerate;
    fields.push(field);
  }

  /* 6. Merge. */
  const fillet = Math.max(options.fillet ?? 0, 0);
  const data = new Uint16Array(N3);
  for (let idx = 0; idx < N3; idx++) {
    let d = fields.length ? fields[0][idx] : EXTENT;
    for (let f = 1; f < fields.length; f++) d = smoothMin(d, fields[f][idx], fillet);
    data[idx] = toHalf(d);
  }
  return { data, center, halfWidth, stats: { pieces: fields.length, holesCapped, degenerateColumns, triangles } };
}
