import { describe, expect, it } from 'vitest';
import * as THREE from 'three/webgpu';
import { bakeGlassVolume, VOLUME_EXTENT, VOLUME_SIZE } from '../bake';
import { MeshGlassMaterial } from '../material';

/** Trilinear sample of a baked volume at volume-unit point (x, y, z). */
function sample(data: Uint16Array, x: number, y: number, z: number): number {
  const g = [x, y, z].map((v) => Math.min(Math.max((v / VOLUME_EXTENT + 0.5) * VOLUME_SIZE - 0.5, 0), VOLUME_SIZE - 1));
  const i = g.map((v) => Math.min(Math.floor(v), VOLUME_SIZE - 2));
  const f = g.map((v, k) => v - i[k]);
  let sum = 0;
  for (let c = 0; c < 8; c++) {
    const o = [c & 1, (c >> 1) & 1, (c >> 2) & 1];
    const w = o.reduce((acc, bit, k) => acc * (bit ? f[k] : 1 - f[k]), 1);
    const idx = i[0] + o[0] + (i[1] + o[1]) * VOLUME_SIZE + (i[2] + o[2]) * VOLUME_SIZE * VOLUME_SIZE;
    sum += w * THREE.DataUtils.fromHalfFloat(data[idx]);
  }
  return sum;
}

const arrays = (geometry: THREE.BufferGeometry) => ({
  positions: geometry.getAttribute('position').array,
  indices: geometry.index ? geometry.index.array : null,
});

// Each bake sweeps the full 128^3 volume (about 1-2 s), so these get a longer timeout.
describe('bakeGlassVolume (bake.ts): any mesh into a solid glass volume', { timeout: 30_000 }, () => {
  it('fits a closed mesh into the volume and measures exact distances', () => {
    // A sphere of radius 3 centred at (5, 0, 0): fitted to half-width 3, centre (5, 0, 0).
    const { positions, indices } = arrays(new THREE.IcosahedronGeometry(3, 6).translate(5, 0, 0));
    const { data, center, halfWidth, stats } = bakeGlassVolume(positions, indices);
    expect(center[0]).toBeCloseTo(5, 3);
    // (Half the bounding box: the icosphere's vertices fall just short of the poles' radius.)
    expect(halfWidth).toBeCloseTo(3, 1);
    expect(stats).toMatchObject({ pieces: 1, holesCapped: 0, degenerateColumns: 0 });
    const radius = 3 / halfWidth; // in volume units
    expect(sample(data, 0, 0, 0)).toBeCloseTo(-radius, 1); // the centre: one radius inside
    expect(Math.abs(sample(data, radius * 0.999, 0, 0))).toBeLessThan(0.02); // on the surface
    expect(sample(data, 0, 0, 1.05)).toBeGreaterThan(0); // outside
  });

  it('welds unwelded triangles and caps open holes, so open meshes become solids', () => {
    // An open-ended tube as separate, unwelded triangles: one piece once welded, with its two
    // open ends capped into a solid cylinder.
    const tube = new THREE.CylinderGeometry(0.5, 0.5, 2, 48, 1, true).toNonIndexed();
    const { positions, indices } = arrays(tube);
    const { data, stats } = bakeGlassVolume(positions, indices, { center: [0, 0, 0], halfWidth: 1 });
    expect(stats.pieces).toBe(1);
    expect(stats.holesCapped).toBe(2);
    expect(stats.degenerateColumns).toBe(0);
    expect(sample(data, 0, 0, 0)).toBeLessThan(-0.4); // inside the (now solid) tube
    expect(sample(data, 0, 0.9, 0)).toBeLessThan(0); // near a capped end, still inside
    expect(sample(data, 0, 1.05, 0)).toBeGreaterThan(0); // beyond the cap
  });

  it('merges overlapping pieces into their union, optionally with a fillet', () => {
    // Two overlapping balls in one non-indexed triangle list (as the icosahedron comes).
    const a = new THREE.IcosahedronGeometry(0.5, 4).translate(-0.3, 0, 0);
    const b = new THREE.IcosahedronGeometry(0.5, 4).translate(0.3, 0, 0);
    const positions = Float32Array.from([...a.getAttribute('position').array, ...b.getAttribute('position').array]);
    const indices = null;
    const plain = bakeGlassVolume(positions, indices, { center: [0, 0, 0], halfWidth: 1 });
    expect(plain.stats.pieces).toBe(2);
    expect(sample(plain.data, -0.3, 0, 0)).toBeLessThan(-0.4); // inside each ball
    expect(sample(plain.data, 0.3, 0, 0)).toBeLessThan(-0.4);
    expect(sample(plain.data, 0, 0, 0)).toBeLessThan(0); // where they overlap
    const filleted = bakeGlassVolume(positions, indices, { center: [0, 0, 0], halfWidth: 1, fillet: 0.1 });
    // The fillet fills the crease where the balls meet (deeper inside there than the plain union).
    expect(sample(filleted.data, 0, 0.4, 0)).toBeLessThan(sample(plain.data, 0, 0.4, 0));
  });

  it('gives MeshGlassMaterial a volume with its own centre and scale', () => {
    const texture = new THREE.Data3DTexture(new Uint16Array(8), 2, 2, 2);
    const mat = new MeshGlassMaterial({ volume: { texture, center: [5, 0, 0], halfWidth: 3 } });
    expect(mat.volume?.halfWidth).toBe(3);
    expect(mat.colorNode).toBeTruthy();
  });
});
