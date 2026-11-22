import { describe, expect, it } from 'vitest';
import * as THREE from 'three/webgpu';
import { createShapeGeometries, triangleCount } from '../geometries';
import { KIND } from '../objects';

describe('unit shape geometries (geometries.ts)', () => {
  const meshes = {
    suzanne: new THREE.BoxGeometry(1, 1, 1),
    dragon: new THREE.IcosahedronGeometry(1, 3),
    dragonLod: new THREE.IcosahedronGeometry(1, 1),
  };

  it('builds valid coarse and fine geometries for all 13 KINDs', () => {
    const { bases, fineBases } = createShapeGeometries(meshes);
    const kindCount = Object.keys(KIND).length;
    expect(bases).toHaveLength(kindCount);
    expect(fineBases).toHaveLength(kindCount);

    for (let k = 0; k < kindCount; k++) {
      for (const geo of [bases[k], fineBases[k]]) {
        expect(triangleCount(geo)).toBeGreaterThan(0);
        expect(geo.getAttribute('position')).toBeTruthy();
        expect(geo.getAttribute('normal')).toBeTruthy();
      }
    }
  });

  it('uses higher tessellation in fineBases for curved shapes and full dragon mesh', () => {
    const { bases, fineBases } = createShapeGeometries(meshes);
    for (const kind of [
      KIND.sphere,
      KIND.cylinder,
      KIND.capsule,
      KIND.cone,
      KIND.torus,
      KIND.suzanne,
      KIND.dragon,
      KIND.torusknot,
    ]) {
      expect(triangleCount(fineBases[kind])).toBeGreaterThan(triangleCount(bases[kind]));
    }
  });

  it('centers the unit torus knot with its bottom resting at y = -1', () => {
    // The frame comes from the exact tube (shared with its baked distance volume), so
    // each mesh, its polygons cutting just inside the round tube, stays within [-1, 1].
    const { bases, fineBases } = createShapeGeometries(meshes);
    for (const geo of [bases[KIND.torusknot], fineBases[KIND.torusknot]]) {
      geo.computeBoundingBox();
      expect(geo.boundingBox!.min.y).toBeGreaterThanOrEqual(-1 - 1e-6);
      expect(geo.boundingBox!.max.y).toBeLessThanOrEqual(1 + 1e-6);
      expect(geo.boundingBox!.min.y).toBeLessThan(-0.99);
      expect(geo.boundingBox!.max.y).toBeGreaterThan(0.99);
    }
  });

  it('keeps the whole torus knot inside its baked distance volume (|q| < 1.09)', () => {
    const { fineBases } = createShapeGeometries(meshes);
    const geo = fineBases[KIND.torusknot];
    geo.computeBoundingBox();
    const { min, max } = geo.boundingBox!;
    for (const v of [min.x, min.y, min.z, max.x, max.y, max.z]) expect(Math.abs(v)).toBeLessThan(1.09);
  });
});
