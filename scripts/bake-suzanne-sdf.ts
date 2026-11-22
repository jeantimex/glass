// Bake Suzanne's 128^3 signed-distance volume to public/suzanne_sdf_128.bin.
//
//   node scripts/bake-suzanne-sdf.ts [fillet]
//
// Blender's Suzanne is made for opaque rendering: an open head shell with a hole for each
// eye, plugged by an open half-sphere eye cup. That is not a solid, so "inside the glass" is
// undefined around the eyes (an earlier bake turned it into an air pocket behind thin
// floating lenses, with sub-voxel walls that rendered as a crinkled ring). Here she is made
// the solid a cast glass figurine would be: every hole is capped so the head and the two
// eyes are closed solids, then they are merged. The holes are a little smaller than the
// eyes, which bulge out through them, so the merge keeps each hole's rim as a crease around
// its eye, as in the model, softened by `fillet` (unit-size units; default 0.01, about half
// a voxel, so the crease does not stair-step at the volume's resolution; 0 keeps it crisp).
// Each piece is Loop-subdivided first (src/glass/subdivide.ts), as the smooth model shows.
//
// Same layout as the other volumes (see src/glass/tsl/sdf.ts): HalfFloat, x fastest then
// y then z, texel i at q = ((i + 0.5) / 128 - 0.5) * 2.2 in unit-size coordinates (the mesh
// is already unit size: half-width 1), distances in those units, negative inside.
import { readFileSync, writeFileSync } from 'node:fs';
import { fileURLToPath } from 'node:url';
import { loopSubdivide } from '../src/glass/subdivide.ts';

const SIZE = 128;
const EXTENT = 2.2;
const VOXEL = EXTENT / SIZE;
/** Smooth-union radius where the eyes meet the head (unit-size units; 0: a crisp crease). */
const FILLET = Math.max(Number(process.argv[2] ?? 0.01) || 0, 0);
/** Width of the exactly computed band around the surface, in voxels. */
const BAND = 3;
/**
 * Loop subdivision levels before baking: the smooth surface Suzanne is meant to show (as
 * with Blender's subdivision modifier), not the low-poly cage's flat facets.
 */
const SUBDIVISIONS = 3;

const meshUrl = new URL('../public/suzanne_mesh.bin', import.meta.url);
const outUrl = new URL('../public/suzanne_sdf_128.bin', import.meta.url);

/* Load the mesh: [vertexCount, indexCount, pad] + positions + normals + indices. */
const bytes = readFileSync(meshUrl);
const buffer = bytes.buffer.slice(bytes.byteOffset, bytes.byteOffset + bytes.byteLength);
const [vertexCount, indexCount] = new Uint32Array(buffer, 0, 2);
const cage = new Float32Array(buffer, 12, vertexCount * 3);
const indices = new Uint32Array(buffer, 12 + vertexCount * 24, indexCount);

/* Split into connected pieces (head, two eyes). */
const parent = Array.from({ length: vertexCount }, (_, i) => i);
const find = (x: number): number => (parent[x] === x ? x : (parent[x] = find(parent[x])));
for (let t = 0; t < indexCount; t += 3) {
  parent[find(indices[t + 1])] = find(indices[t]);
  parent[find(indices[t + 2])] = find(indices[t]);
}
const pieces = new Map<number, number[]>(); // root -> triangle vertex indices (flat)
for (let t = 0; t < indexCount; t += 3) {
  const root = find(indices[t]);
  if (!pieces.has(root)) pieces.set(root, []);
  pieces.get(root)!.push(indices[t], indices[t + 1], indices[t + 2]);
}

/*
 * Subdivide each piece while it is still open (the boundary rules keep the eye holes' rims
 * and the eye cups' edges as smooth lips, as the subdivided model shows them), into one
 * shared vertex list.
 */
const positions: number[] = [];
const smoothPieces = [...pieces.values()].map((tris) => {
  const local = new Map<number, number>();
  const localPositions: number[] = [];
  const localIndices: number[] = [];
  for (const v of tris) {
    if (!local.has(v)) {
      local.set(v, localPositions.length / 3);
      localPositions.push(cage[v * 3], cage[v * 3 + 1], cage[v * 3 + 2]);
    }
    localIndices.push(local.get(v)!);
  }
  const smooth = loopSubdivide({ positions: localPositions, indices: localIndices }, SUBDIVISIONS);
  const base = positions.length / 3;
  for (const x of smooth.positions) positions.push(x);
  return Array.from(smooth.indices, (i) => i + base);
});

/** An undirected edge's key. */
const key = (a: number, b: number) => (a < b ? `${a}_${b}` : `${b}_${a}`);

/** Close every hole of a piece with a fan of triangles to the hole's centroid. */
function capHoles(tris: number[]): { tris: number[]; holes: number } {
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
function pieceField(tris: number[]): Float32Array {
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
      Math.max(0, Math.floor(cell(Math.min(tri[o + d], tri[o + 3 + d], tri[o + 6 + d]))) - BAND),
    );
    const hi = [0, 1, 2].map((d) =>
      Math.min(SIZE - 1, Math.ceil(cell(Math.max(tri[o + d], tri[o + 3 + d], tri[o + 6 + d]))) + BAND),
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
  console.log(`  piece: ${triCount} triangles, ${recast} column(s) recast, ${oddColumns} left degenerate`);
  return field;
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

const started = Date.now();
const fields: Float32Array[] = [];
for (const tris of smoothPieces.sort((a, b) => b.length - a.length)) {
  const { tris: closed, holes } = capHoles(tris);
  console.log(`closing a piece of ${tris.length / 3} triangles: ${holes} hole(s) capped`);
  fields.push(pieceField(closed));
}
const out = new Uint16Array(N3);
let minD = Infinity;
let maxD = -Infinity;
for (let idx = 0; idx < N3; idx++) {
  let d = fields[0][idx];
  for (let f = 1; f < fields.length; f++) d = smoothMin(d, fields[f][idx], FILLET);
  minD = Math.min(minD, d);
  maxD = Math.max(maxD, d);
  out[idx] = toHalf(d);
}
writeFileSync(outUrl, new Uint8Array(out.buffer));
console.log(
  `wrote ${fileURLToPath(outUrl)} in ${((Date.now() - started) / 1000).toFixed(1)} s: ` +
    `${fields.length} solids, fillet ${FILLET} (${(FILLET / VOXEL).toFixed(1)} voxels), d in [${minD.toFixed(4)}, ${maxD.toFixed(4)}]`,
);
