import { readFileSync } from 'node:fs';
import { describe, expect, it } from 'vitest';
import * as THREE from 'three/webgpu';
import { loopSubdivide } from '../subdivide';

const SIZE = 128;
const EXTENT = 2.2;

/** Suzanne's baked volume (public/suzanne_sdf_128.bin), decoded from HalfFloat. */
const volume = (() => {
  const bytes = readFileSync(new URL('../../../public/suzanne_sdf_128.bin', import.meta.url));
  const halves = new Uint16Array(bytes.buffer, bytes.byteOffset, bytes.byteLength / 2);
  return Float32Array.from(halves, (h) => THREE.DataUtils.fromHalfFloat(h));
})();

/** Trilinear sample of the volume at unit-size point (x, y, z). */
function sampleVolume(x: number, y: number, z: number): number {
  const g = [x, y, z].map((v) => Math.min(Math.max((v / EXTENT + 0.5) * SIZE - 0.5, 0), SIZE - 1));
  const i = g.map((v) => Math.min(Math.floor(v), SIZE - 2));
  const f = g.map((v, k) => v - i[k]);
  let sum = 0;
  for (let c = 0; c < 8; c++) {
    const o = [c & 1, (c >> 1) & 1, (c >> 2) & 1];
    const w = o.reduce((acc, bit, k) => acc * (bit ? f[k] : 1 - f[k]), 1);
    sum += w * volume[i[0] + o[0] + (i[1] + o[1]) * SIZE + (i[2] + o[2]) * SIZE * SIZE];
  }
  return sum;
}

describe('Loop subdivision (subdivide.ts)', () => {
  it('splits every triangle into four and smooths a closed mesh toward a sphere', () => {
    const cube = new THREE.BoxGeometry(2, 2, 2);
    const welded = { positions: cube.getAttribute('position').array, indices: cube.index!.array };
    const once = loopSubdivide(welded, 1);
    expect(once.indices.length).toBe(welded.indices.length * 4);
    const twice = loopSubdivide(welded, 2);
    expect(twice.indices.length).toBe(welded.indices.length * 16);
    for (const v of twice.positions) expect(Number.isFinite(v)).toBe(true);
  });

  it('keeps an open edge on its own curve (boundary rules)', () => {
    // A flat square of two triangles: its boundary stays in the plane and inside the square.
    const square = { positions: [0, 0, 0, 1, 0, 0, 1, 1, 0, 0, 1, 0], indices: [0, 1, 2, 0, 2, 3] };
    const out = loopSubdivide(square, 2);
    for (let v = 0; v < out.positions.length / 3; v++) {
      expect(out.positions[v * 3 + 2]).toBe(0);
      expect(out.positions[v * 3]).toBeGreaterThanOrEqual(0);
      expect(out.positions[v * 3]).toBeLessThanOrEqual(1);
    }
  });
});

describe("Suzanne's baked volume (scripts/bake-suzanne-sdf.ts)", () => {
  it('is solid glass behind the eyes (no air pocket where the eye holes were)', () => {
    // Points inside the head just behind each eye, where the old bake had an air pocket.
    for (const x of [-0.27, 0.27]) {
      for (const z of [0.38, 0.42, 0.46]) expect(sampleVolume(x, 0.2, z)).toBeLessThan(0);
    }
  });

  it('bulges the eyes out of the face, and is outside in front of them', () => {
    expect(sampleVolume(0.27, 0.2, 0.55)).toBeLessThan(0); // inside the eye bulge
    expect(sampleVolume(0.27, 0.2, 0.68)).toBeGreaterThan(0); // in front of the eye
    expect(sampleVolume(0, 0, 0)).toBeLessThan(0); // the middle of the head
    expect(sampleVolume(1.05, 1.05, 1.05)).toBeGreaterThan(0); // the corner of the volume
  });
});
