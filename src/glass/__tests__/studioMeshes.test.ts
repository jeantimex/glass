import { describe, expect, it } from 'vitest';
import * as THREE from 'three/webgpu';
import { KIND } from '../objects';
import { studioShapeOf, studioSize } from '../studioMeshes';

describe('studio objects from standard meshes', () => {
  it('reads built-in solids and declared shapes', () => {
    expect(studioShapeOf(new THREE.BoxGeometry(2, 4, 6))?.kind).toBe(KIND.box);
    const declared = new THREE.BufferGeometry();
    declared.userData.glassShape = { kind: KIND.suzanne, size: [1, 1, 1], axis: 'y' };
    expect(studioShapeOf(declared)?.kind).toBe(KIND.suzanne);
    expect(studioShapeOf(new THREE.BufferGeometry())).toBeNull();
  });

  it("gives sizes in the renderer's conventions", () => {
    const box = studioShapeOf(new THREE.BoxGeometry(2, 4, 6))!;
    expect(studioSize(box, new THREE.Vector3(1, 2, 3))).toEqual([1, 4, 9]);
    // A sphere or octahedron: its radius in all three.
    const sphere = studioShapeOf(new THREE.IcosahedronGeometry(0.5, 12))!;
    expect(studioSize(sphere, new THREE.Vector3(2, 2, 2))).toEqual([1, 1, 1]);
    // A torus is built around z: its tube scales with the geometry's z (the shape's y).
    const torus = studioShapeOf(new THREE.TorusGeometry(1, 0.25, 8, 16))!;
    expect(studioSize(torus, new THREE.Vector3(2, 3, 4))).toEqual([2, 1, 0.75]);
  });
});
