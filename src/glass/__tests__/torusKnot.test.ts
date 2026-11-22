import { readFileSync } from 'node:fs';
import { describe, expect, it } from 'vitest';
import * as THREE from 'three/webgpu';
import { createShapeGeometries } from '../geometries';
import { KIND } from '../objects';
import { TORUS_KNOT, UNIT_TORUS_KNOT, torusKnotCurve } from '../torusKnot';

const SIZE = 128;
const EXTENT = 2.2;

/** The baked volume (public/torusknot_sdf_128.bin), decoded from HalfFloat. */
const volume = (() => {
  const bytes = readFileSync(new URL('../../../public/torusknot_sdf_128.bin', import.meta.url));
  const halves = new Uint16Array(bytes.buffer, bytes.byteOffset, bytes.byteLength / 2);
  return Float32Array.from(halves, (h) => THREE.DataUtils.fromHalfFloat(h));
})();

/** Trilinear sample of the volume at unit-size point q, as the shader's `volumeFast` does. */
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

describe('torus knot shape (torusKnot.ts) and its baked distance volume', () => {
  it('maps the tube onto y in [-1, 1] exactly', () => {
    const { centerY, scale } = UNIT_TORUS_KNOT;
    let minY = Infinity;
    let maxY = -Infinity;
    for (let i = 0; i < 4096; i++) {
      const y = torusKnotCurve((i / 4096) * Math.PI * 2 * TORUS_KNOT.p)[1];
      minY = Math.min(minY, y);
      maxY = Math.max(maxY, y);
    }
    expect((minY - TORUS_KNOT.tube - centerY) * scale).toBeCloseTo(-1, 4);
    expect((maxY + TORUS_KNOT.tube - centerY) * scale).toBeCloseTo(1, 4);
  });

  it('bakes a volume that is zero on the rendered mesh surface', () => {
    expect(volume).toHaveLength(SIZE * SIZE * SIZE);
    const meshes = {
      suzanne: new THREE.BoxGeometry(1, 1, 1),
      dragon: new THREE.IcosahedronGeometry(1, 1),
      dragonLod: new THREE.IcosahedronGeometry(1, 1),
    };
    const position = createShapeGeometries(meshes).fineBases[KIND.torusknot].getAttribute('position');
    // Trilinear filtering of a 2.2 / 128 grid rounds the tube a little; within a fifth of a texel.
    const tolerance = (EXTENT / SIZE) * 0.2;
    let worst = 0;
    for (let v = 0; v < position.count; v += 7) {
      worst = Math.max(worst, Math.abs(sampleVolume(position.getX(v), position.getY(v), position.getZ(v))));
    }
    expect(worst).toBeLessThan(tolerance);
  });

  it('is negative inside the tube and positive outside it', () => {
    const { centerY, scale } = UNIT_TORUS_KNOT;
    const [x, y, z] = torusKnotCurve(0.7);
    const onCurve = [x * scale, (y - centerY) * scale, z * scale] as const;
    // Deepest on the centre curve (trilinear filtering rounds off that sharp minimum a little).
    expect(sampleVolume(...onCurve)).toBeLessThan(-TORUS_KNOT.tube * scale * 0.9);
    expect(sampleVolume(0, 0, 0.95)).toBeGreaterThan(0);
  });
});
