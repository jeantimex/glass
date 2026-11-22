// Bake the torus knot's exact 128^3 signed-distance volume to public/torusknot_sdf_128.bin.
//
//   node scripts/bake-torusknot-sdf.ts
//
// Same layout as the Suzanne and dragon volumes (see src/glass/tsl/sdf.ts): HalfFloat,
// x fastest then y then z, texel i at q = ((i + 0.5) / 128 - 0.5) * 2.2 in unit-size
// coordinates, distances in unit-size units, negative inside.
//
// A tube is exactly the set of points within `tube` of its centre curve, so its signed
// distance is (distance to the curve) - tube. The curve is a fine polyline; each texel
// first finds its distance to a coarse polyline, then refines only the coarse segments
// that could still hold the nearest point.
import { writeFileSync } from 'node:fs';
import { fileURLToPath } from 'node:url';
import { TORUS_KNOT, UNIT_TORUS_KNOT, torusKnotCurve } from '../src/glass/torusKnot.ts';

const SIZE = 128;
const EXTENT = 2.2;
const COARSE = 512;
const FINE_PER_COARSE = 16;
const FINE = COARSE * FINE_PER_COARSE;

const { centerY, scale } = UNIT_TORUS_KNOT;
const tube = TORUS_KNOT.tube * scale;
const turns = Math.PI * 2 * TORUS_KNOT.p;

// Fine polyline points (closed: point FINE equals point 0), in unit-size coordinates.
const pts = new Float64Array((FINE + 1) * 3);
for (let i = 0; i <= FINE; i++) {
  const [x, y, z] = torusKnotCurve(((i % FINE) / FINE) * turns);
  pts.set([x * scale, (y - centerY) * scale, z * scale], i * 3);
}

/** Squared distance from (px, py, pz) to the segment between fine points a and b. */
function segDist2(px: number, py: number, pz: number, a: number, b: number): number {
  const ax = pts[a * 3];
  const ay = pts[a * 3 + 1];
  const az = pts[a * 3 + 2];
  const dx = pts[b * 3] - ax;
  const dy = pts[b * 3 + 1] - ay;
  const dz = pts[b * 3 + 2] - az;
  const wx = px - ax;
  const wy = py - ay;
  const wz = pz - az;
  const len2 = dx * dx + dy * dy + dz * dz;
  const t = len2 > 0 ? Math.min(Math.max((wx * dx + wy * dy + wz * dz) / len2, 0), 1) : 0;
  const ex = wx - dx * t;
  const ey = wy - dy * t;
  const ez = wz - dz * t;
  return ex * ex + ey * ey + ez * ez;
}

// How far the true curve strays from each coarse chord (bounded by its fine points).
let sag = 0;
for (let c = 0; c < COARSE; c++) {
  const a = c * FINE_PER_COARSE;
  for (let k = 1; k < FINE_PER_COARSE; k++)
    sag = Math.max(
      sag,
      Math.sqrt(segDist2(pts[(a + k) * 3], pts[(a + k) * 3 + 1], pts[(a + k) * 3 + 2], a, a + FINE_PER_COARSE)),
    );
}
const margin = sag * 1.5 + 1e-4;

/** Exact float32 -> IEEE half conversion (round to nearest even). */
const f32 = new Float32Array(1);
const u32 = new Uint32Array(f32.buffer);
function toHalf(v: number): number {
  f32[0] = v;
  const x = u32[0];
  const sign = (x >>> 16) & 0x8000;
  const exp = ((x >>> 23) & 0xff) - 127 + 15;
  const mant = x & 0x7fffff;
  if (exp <= 0) {
    if (exp < -10) return sign;
    const m = (mant | 0x800000) >>> (1 - exp);
    return sign | ((m + 0x1000) >>> 13);
  }
  if (exp >= 31) return sign | 0x7c00;
  const h = sign | (exp << 10) | (mant >>> 13);
  const rest = mant & 0x1fff;
  return rest > 0x1000 || (rest === 0x1000 && h & 1) ? h + 1 : h;
}

const out = new Uint16Array(SIZE * SIZE * SIZE);
const coarseD2 = new Float64Array(COARSE);
const started = Date.now();
let minD = Infinity;
let maxD = -Infinity;
for (let k = 0; k < SIZE; k++) {
  const pz = ((k + 0.5) / SIZE - 0.5) * EXTENT;
  for (let j = 0; j < SIZE; j++) {
    const py = ((j + 0.5) / SIZE - 0.5) * EXTENT;
    for (let i = 0; i < SIZE; i++) {
      const px = ((i + 0.5) / SIZE - 0.5) * EXTENT;
      let bestCoarse = Infinity;
      for (let c = 0; c < COARSE; c++) {
        const d2 = segDist2(px, py, pz, c * FINE_PER_COARSE, (c + 1) * FINE_PER_COARSE);
        coarseD2[c] = d2;
        if (d2 < bestCoarse) bestCoarse = d2;
      }
      // The true nearest distance is at most sqrt(bestCoarse) + margin; any coarse
      // segment farther than that minus its own margin cannot hold it.
      const limit = Math.sqrt(bestCoarse) + 2 * margin;
      const limit2 = limit * limit;
      let best = Infinity;
      for (let c = 0; c < COARSE; c++) {
        if (coarseD2[c] > limit2) continue;
        const a = c * FINE_PER_COARSE;
        for (let s = 0; s < FINE_PER_COARSE; s++) {
          const d2 = segDist2(px, py, pz, a + s, a + s + 1);
          if (d2 < best) best = d2;
        }
      }
      const d = Math.sqrt(best) - tube;
      minD = Math.min(minD, d);
      maxD = Math.max(maxD, d);
      out[i + j * SIZE + k * SIZE * SIZE] = toHalf(d);
    }
  }
  if (k % 16 === 15) console.log(`slice ${k + 1}/${SIZE} (${((Date.now() - started) / 1000).toFixed(1)}s)`);
}

const target = fileURLToPath(new URL('../public/torusknot_sdf_128.bin', import.meta.url));
writeFileSync(target, new Uint8Array(out.buffer));
console.log(
  `wrote ${target}: tube ${tube.toFixed(4)}, chord margin ${margin.toExponential(2)}, d in [${minD.toFixed(4)}, ${maxD.toFixed(4)}]`,
);
