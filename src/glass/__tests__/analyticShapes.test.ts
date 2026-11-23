import { describe, expect, it } from 'vitest';
import * as THREE from 'three/webgpu';
import { analyticShape } from '../analyticShapes';
import { MeshGlassMaterial } from '../material';
import { KIND } from '../objects';

describe('analyticShape (analyticShapes.ts): three.js built-in solids traced exactly', () => {
  it('reads each built-in solid into its shape and size', () => {
    expect(analyticShape(new THREE.SphereGeometry(0.7, 32, 16))).toEqual({
      kind: KIND.sphere,
      size: [0.7, 0, 0],
      axis: 'y',
    });
    expect(analyticShape(new THREE.IcosahedronGeometry(0.4, 5))).toMatchObject({
      kind: KIND.sphere,
      size: [0.4, 0, 0],
    });
    expect(analyticShape(new THREE.BoxGeometry(2, 1, 0.5))).toMatchObject({ kind: KIND.box, size: [1, 0.5, 0.25] });
    expect(analyticShape(new THREE.CylinderGeometry(0.3, 0.3, 2))).toMatchObject({
      kind: KIND.cylinder,
      size: [0.3, 1, 0.3],
    });
    expect(analyticShape(new THREE.CylinderGeometry(0, 0.5, 1.5))).toMatchObject({
      kind: KIND.cone,
      size: [0.5, 0.75, 0.5],
    });
    expect(analyticShape(new THREE.ConeGeometry(0.6, 1.2))).toMatchObject({ kind: KIND.cone, size: [0.6, 0.6, 0.6] });
    // CapsuleGeometry's height is its straight middle section.
    expect(analyticShape(new THREE.CapsuleGeometry(0.25, 1))).toMatchObject({
      kind: KIND.capsule,
      size: [0.25, 0.5, 0.25],
    });
    expect(analyticShape(new THREE.OctahedronGeometry(0.9))).toMatchObject({
      kind: KIND.octahedron,
      size: [0.9, 0, 0],
    });
    // three.js builds the torus around z; tsl/shapes.ts has it around y.
    expect(analyticShape(new THREE.TorusGeometry(1, 0.3))).toEqual({
      kind: KIND.torus,
      size: [1, 0.3, 0.3],
      axis: 'z',
    });
  });

  it('declines what is not a whole built-in solid', () => {
    expect(analyticShape(new THREE.SphereGeometry(1, 16, 8, 0, Math.PI))).toBeNull(); // half a sphere
    expect(analyticShape(new THREE.CylinderGeometry(1, 1, 1, 16, 1, true))).toBeNull(); // open tube
    expect(analyticShape(new THREE.CylinderGeometry(0.5, 1, 1))).toBeNull(); // truncated cone
    expect(analyticShape(new THREE.TorusGeometry(1, 0.3, 12, 48, Math.PI))).toBeNull(); // part of a ring
    expect(analyticShape(new THREE.IcosahedronGeometry(1, 1))).toBeNull(); // faceted, not a sphere
    expect(analyticShape(new THREE.OctahedronGeometry(1, 2))).toBeNull(); // subdivided octahedron
    expect(analyticShape(new THREE.TorusKnotGeometry())).toBeNull();
    expect(analyticShape(new THREE.PlaneGeometry())).toBeNull();
    expect(analyticShape(new THREE.BufferGeometry())).toBeNull();
    expect(analyticShape(null)).toBeNull();
  });

  it('is on by default and can be turned off on the material', () => {
    const mat = new MeshGlassMaterial();
    expect(mat.exactShapes).toBe(true);
    expect(new MeshGlassMaterial({ exactShapes: false }).exactShapes).toBe(false);
    mat.exactShapes = false;
    expect(mat.clone().exactShapes).toBe(false);
  });
});
