// The demo's objects as standard three.js meshes: one InstancedMesh per shape kind (a plain
// THREE.Mesh for a single object), with MeshGlassMaterial in studio mode on the studio's own
// geometries. Each object is an instance: its pose and size in the instance matrix (the
// geometries are unit-sized; a capsule's or torus's declared deformation keeps its round
// sections), its colour by setColorAt, its own glass in the per-instance `glassVary` attribute.
import * as THREE from 'three/webgpu';
import { MeshGlassMaterial, type GlassObjects, type GlassStudio } from '../glass';

const position = new THREE.Vector3();
const quaternion = new THREE.Quaternion();
const scale = new THREE.Vector3();
const matrix = new THREE.Matrix4();
const colour = new THREE.Color();

/** Object `i`'s pose with its size as the scale (the studio's geometries are unit-sized). */
function poseOf(objects: GlassObjects, i: number): THREE.Matrix4 {
  position.fromArray(objects.center, i * 3);
  quaternion.fromArray(objects.quat, i * 4);
  scale.fromArray(objects.size, i * 3);
  return matrix.compose(position, quaternion, scale);
}

/**
 * Standard meshes for `objects` in `studio`: `cut` the diamonds' cut, `coarse` the studio's
 * lighter geometries (for many objects, as the renderer draws them).
 */
export function buildGlassMeshes(
  objects: GlassObjects,
  studio: GlassStudio,
  options: { cut: string; coarse: boolean },
): THREE.Mesh[] {
  const material = new MeshGlassMaterial({ studio });
  const geometryOf = (kind: number) => studio.shapeGeometry(kind, options.cut, options.coarse);
  if (objects.count === 1) {
    const mesh = new THREE.Mesh(geometryOf(objects.kind[0]), material);
    poseOf(objects, 0).decompose(mesh.position, mesh.quaternion, mesh.scale);
    return [mesh];
  }
  const groups = new Map<number, number[]>();
  for (let i = 0; i < objects.count; i++) {
    const kind = objects.kind[i];
    const members = groups.get(kind);
    if (members) members.push(i);
    else groups.set(kind, [i]);
  }
  const meshes: THREE.Mesh[] = [];
  for (const [kind, members] of groups) {
    const geometry = geometryOf(kind);
    const vary = new Float32Array(members.length * 3);
    const mesh = new THREE.InstancedMesh(geometry, material, members.length);
    members.forEach((i, k) => {
      mesh.setMatrixAt(k, poseOf(objects, i));
      mesh.setColorAt(k, colour.fromArray(objects.tint, i * 3));
      vary.set(objects.vary.subarray(i * 3, i * 3 + 3), k * 3);
    });
    geometry.setAttribute('glassVary', new THREE.InstancedBufferAttribute(vary, 3));
    meshes.push(mesh);
  }
  return meshes;
}

/** Free what `buildGlassMeshes` made. */
export function disposeGlassMeshes(meshes: THREE.Mesh[]): void {
  const materials = new Set<THREE.Material>();
  for (const mesh of meshes) {
    mesh.geometry.dispose();
    materials.add(mesh.material as THREE.Material);
    if ((mesh as THREE.InstancedMesh).isInstancedMesh) (mesh as THREE.InstancedMesh).dispose();
  }
  for (const material of materials) material.dispose();
}
