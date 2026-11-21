// Unit-scale BufferGeometries for all 13 glass shape KINDs (coarse LOD for large grids
// and high-tessellation fine meshes for close-up / small counts).
import * as THREE from 'three/webgpu';
import { diamondGeometry } from './diamond';
import { KIND } from './objects';

/** Tube radius of the unit torus geometry (ring radius = 1). */
export const UNIT_TUBE = 0.25;

export interface BakedMeshGeometries {
  suzanne: THREE.BufferGeometry;
  dragon: THREE.BufferGeometry;
  dragonLod: THREE.BufferGeometry;
}

/** Convert an indexed geometry to non-indexed flat-shaded triangles with face normals. */
function makeFlatShaded(geometry: THREE.BufferGeometry): THREE.BufferGeometry {
  const flat = geometry.index ? geometry.toNonIndexed() : geometry;
  flat.computeVertexNormals();
  return flat;
}

/** Build a unit torus knot centered vertically and scaled so its bottom rests at y = -1. */
function buildUnitTorusKnot(tubularSegments: number, radialSegments: number): THREE.BufferGeometry {
  const geometry = new THREE.TorusKnotGeometry(0.62, 0.2, tubularSegments, radialSegments);
  geometry.computeBoundingBox();
  const bounds = geometry.boundingBox!;
  const centerY = (bounds.min.y + bounds.max.y) * 0.5;
  const scale = 2 / (bounds.max.y - bounds.min.y);
  return geometry.translate(0, -centerY, 0).scale(scale, scale, scale);
}

/** Count triangles in an indexed or non-indexed BufferGeometry. */
export function triangleCount(geometry: THREE.BufferGeometry): number {
  return (geometry.index ? geometry.index.count : geometry.getAttribute('position').count) / 3;
}

/** Build coarse (`bases`) and high-tessellation (`fineBases`) unit geometries indexed by `KIND`. */
export function createShapeGeometries(meshes: BakedMeshGeometries): {
  bases: THREE.BufferGeometry[];
  fineBases: THREE.BufferGeometry[];
} {
  // Triangular prism: apex at y = 1, base at y = -0.5, extruded along z in [-1, 1].
  const triangle = new THREE.Shape([
    new THREE.Vector2(0, 1),
    new THREE.Vector2(-0.8660254, -0.5),
    new THREE.Vector2(0.8660254, -0.5),
  ]);
  const hexRadius = 1 / Math.cos(Math.PI / 6);

  const bases: THREE.BufferGeometry[] = [
    new THREE.BoxGeometry(2, 2, 2),
    new THREE.IcosahedronGeometry(1, 4),
    new THREE.CylinderGeometry(1, 1, 2, 40, 1),
    new THREE.CapsuleGeometry(1, 2, 10, 32),
    new THREE.ConeGeometry(1, 2, 48, 1),
    new THREE.TorusGeometry(1, UNIT_TUBE, 20, 64).rotateX(Math.PI / 2),
    makeFlatShaded(new THREE.OctahedronGeometry(1)),
    makeFlatShaded(new THREE.ExtrudeGeometry(triangle, { depth: 2, bevelEnabled: false }).translate(0, 0, -1)),
    makeFlatShaded(new THREE.CylinderGeometry(hexRadius, hexRadius, 2, 6, 1).rotateY(Math.PI / 2)),
    meshes.suzanne,
    diamondGeometry(),
    meshes.dragonLod,
    buildUnitTorusKnot(96, 16),
  ];

  // High-tessellation meshes for close-up views (<= FINE_UP_TO objects).
  const fineBases = [...bases];
  fineBases[KIND.sphere] = new THREE.IcosahedronGeometry(1, 12);
  fineBases[KIND.cylinder] = new THREE.CylinderGeometry(1, 1, 2, 120, 1);
  fineBases[KIND.capsule] = new THREE.CapsuleGeometry(1, 2, 24, 96);
  fineBases[KIND.cone] = new THREE.ConeGeometry(1, 2, 120, 1);
  fineBases[KIND.torus] = new THREE.TorusGeometry(1, UNIT_TUBE, 40, 144).rotateX(Math.PI / 2);
  fineBases[KIND.dragon] = meshes.dragon;
  fineBases[KIND.torusknot] = buildUnitTorusKnot(220, 36);

  return { bases, fineBases };
}
