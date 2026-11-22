// Unit-scale BufferGeometries for all 13 glass shape KINDs (coarse LOD for large grids
// and high-tessellation fine meshes for close-up / small counts).
import * as THREE from 'three/webgpu';
import { diamondGeometry } from './diamond';
import { KIND } from './objects';
import { loopSubdivide } from './subdivide';
import { TORUS_KNOT, UNIT_TORUS_KNOT } from './torusKnot';

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

/**
 * Build a unit torus knot centered vertically and scaled so its bottom rests at y = -1,
 * in exactly the frame of its baked distance volume (see torusKnot.ts).
 */
function buildUnitTorusKnot(tubularSegments: number, radialSegments: number): THREE.BufferGeometry {
  const { radius, tube, p, q } = TORUS_KNOT;
  const { centerY, scale } = UNIT_TORUS_KNOT;
  const geometry = new THREE.TorusKnotGeometry(radius, tube, tubularSegments, radialSegments, p, q);
  return geometry.translate(0, -centerY, 0).scale(scale, scale, scale);
}

/**
 * The smooth surface of a low-poly indexed mesh (Loop subdivision, `levels` times), with
 * smooth vertex normals: Suzanne as the subdivided model shows her, matching her baked
 * distance volume (scripts/bake-suzanne-sdf.ts) instead of the cage's flat facets.
 */
export function subdividedGeometry(geometry: THREE.BufferGeometry, levels: number): THREE.BufferGeometry {
  const smooth = loopSubdivide(
    { positions: geometry.getAttribute('position').array, indices: geometry.index!.array },
    levels,
  );
  const out = new THREE.BufferGeometry();
  out.setAttribute('position', new THREE.BufferAttribute(Float32Array.from(smooth.positions), 3));
  out.setIndex(new THREE.BufferAttribute(smooth.indices, 1));
  out.computeVertexNormals();
  return out;
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
  // Suzanne's low-poly cage, subdivided into her smooth surface for close-ups.
  fineBases[KIND.suzanne] = subdividedGeometry(meshes.suzanne, 2);
  fineBases[KIND.torusknot] = buildUnitTorusKnot(220, 36);

  return { bases, fineBases };
}
