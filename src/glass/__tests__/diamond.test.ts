import { describe, expect, it } from 'vitest';
import * as THREE from 'three/webgpu';
import {
  DIAMOND_CUTS,
  DIAMOND_LEAN,
  DIAMOND_PLANES,
  DIAMOND_PLANE_COUNT,
  DIAMOND_REST,
  diamondGeometry,
  type DiamondCut,
} from '../diamond';

const ALL_CUTS = Object.keys(DIAMOND_CUTS) as DiamondCut[];

describe('diamond cuts & convex polyhedron geometry', () => {
  it('builds the round brilliant from exactly 110 triangles', () => {
    const geometry = diamondGeometry('round');
    const count = (geometry.index ? geometry.index.count : geometry.getAttribute('position').count) / 3;
    expect(count).toBe(110);
  });

  it('exposes all 6 cuts with valid outward unit normals and positive offsets', () => {
    expect(ALL_CUTS).toEqual(['round', 'princess', 'emerald', 'asscher', 'cushion', 'radiant']);
    for (const cut of ALL_CUTS) {
      const spec = DIAMOND_CUTS[cut];
      expect(spec.planes.length).toBe(spec.planeCount * 4);
      expect(spec.planeCount).toBeGreaterThanOrEqual(30);
      for (let p = 0; p < spec.planeCount; p++) {
        const nx = spec.planes[p * 4];
        const ny = spec.planes[p * 4 + 1];
        const nz = spec.planes[p * 4 + 2];
        const d = spec.planes[p * 4 + 3];
        expect(Math.hypot(nx, ny, nz)).toBeCloseTo(1, 5);
        expect(d).toBeGreaterThan(0);
      }
    }
  });

  it('keeps legacy Round Brilliant exports in sync with DIAMOND_CUTS.round', () => {
    expect(DIAMOND_PLANES).toBe(DIAMOND_CUTS.round.planes);
    expect(DIAMOND_PLANE_COUNT).toBe(DIAMOND_CUTS.round.planeCount);
    expect(DIAMOND_REST).toBe(DIAMOND_CUTS.round.rest);
    expect(DIAMOND_LEAN).toBe(DIAMOND_CUTS.round.lean);
  });

  for (const cut of ALL_CUTS) {
    it(`builds a watertight convex mesh for "${cut}" matching its half-space planes`, () => {
      const spec = DIAMOND_CUTS[cut];
      const geom = diamondGeometry(cut);
      const pos = geom.getAttribute('position');
      const nrm = geom.getAttribute('normal');
      expect(pos.count).toBeGreaterThanOrEqual(90);
      expect(pos.count % 3).toBe(0);
      expect(nrm.count).toBe(pos.count);

      // Every vertex must be finite and lie inside (or on) every facet plane.
      for (let i = 0; i < pos.count; i++) {
        const vx = pos.getX(i);
        const vy = pos.getY(i);
        const vz = pos.getZ(i);
        expect(Number.isFinite(vx) && Number.isFinite(vy) && Number.isFinite(vz)).toBe(true);
        for (let p = 0; p < spec.planeCount; p++) {
          const signed =
            spec.planes[p * 4] * vx +
            spec.planes[p * 4 + 1] * vy +
            spec.planes[p * 4 + 2] * vz -
            spec.planes[p * 4 + 3];
          expect(signed).toBeLessThanOrEqual(1e-4);
        }
      }

      // Every triangle winding must agree with its outward normal.
      const a = new THREE.Vector3();
      const b = new THREE.Vector3();
      const c = new THREE.Vector3();
      const ab = new THREE.Vector3();
      const ac = new THREE.Vector3();
      const cross = new THREE.Vector3();
      const fn = new THREE.Vector3();
      for (let i = 0; i < pos.count; i += 3) {
        a.fromBufferAttribute(pos, i);
        b.fromBufferAttribute(pos, i + 1);
        c.fromBufferAttribute(pos, i + 2);
        fn.fromBufferAttribute(nrm, i);
        cross.crossVectors(ab.subVectors(b, a), ac.subVectors(c, a));
        expect(cross.dot(fn)).toBeGreaterThan(0);
      }

      // Resting pose: leaning by -spec.lean around Z and lifting by spec.rest places bottom flush on y = 0.
      const q = new THREE.Quaternion().setFromAxisAngle(new THREE.Vector3(0, 0, 1), -spec.lean);
      let minY = Infinity;
      for (let i = 0; i < pos.count; i++) {
        a.fromBufferAttribute(pos, i).applyQuaternion(q);
        if (a.y + spec.rest < minY) minY = a.y + spec.rest;
      }
      expect(minY).toBeCloseTo(0, 3);
    });
  }

  it('has symmetric X/Z extents for square/round cuts and elongated Z > X for emerald/radiant', () => {
    for (const cut of ['round', 'princess', 'asscher', 'cushion'] as DiamondCut[]) {
      const g = diamondGeometry(cut);
      g.computeBoundingBox();
      const b = g.boundingBox!;
      expect(b.max.x - b.min.x).toBeCloseTo(b.max.z - b.min.z, 2);
    }
    for (const cut of ['emerald', 'radiant'] as DiamondCut[]) {
      const g = diamondGeometry(cut);
      g.computeBoundingBox();
      const b = g.boundingBox!;
      expect(b.max.z - b.min.z).toBeGreaterThan((b.max.x - b.min.x) * 1.2);
    }
  });
});
