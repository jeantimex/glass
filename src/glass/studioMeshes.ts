// Standard meshes as the studio's objects: each THREE.Mesh or InstancedMesh with
// MeshGlassMaterial (studio mode) on a solid the studio can trace. The studio reads every
// instance into its object records (kind, centre, turn, size, own glass, colour), so its
// layering, shadow framing and budgets work from them as from any objects; it draws them
// through proxies: per mesh, one InstancedMesh for the nearer half of its instances and one for
// the farther (filled in the studio's order, nearest first), with wireframe twins, plus the
// mesh's proxies in the other passes (studioPasses.ts).
import * as THREE from 'three/webgpu';
import { analyticShape, type AnalyticShape } from './analyticShapes';
import { KIND, MAX_OBJECTS, type GlassObjects } from './objects';
import { addStudioPasses, type StudioPasses } from './studioPasses';
import type { GlassStudio } from './studioGlass';
import { triangleCount } from './geometries';

/** The solid a mesh's geometry is (declared, or a three.js built-in), or null. */
export function studioShapeOf(geometry: THREE.BufferGeometry): AnalyticShape | null {
  return (geometry.userData.glassShape as AnalyticShape | undefined) ?? analyticShape(geometry);
}

/**
 * The studio's size of an object (its half-extents in the conventions of tsl/shapes.ts, as the
 * renderer's own objects carry it) from its shape and its scale along the geometry's axes.
 */
export function studioSize(shape: AnalyticShape, scale: THREE.Vector3): [number, number, number] {
  // The shape's own axes may be turned from the geometry's (three.js builds the torus around z).
  const s = shape.axis === 'z' ? [scale.x, scale.z, scale.y] : [scale.x, scale.y, scale.z];
  const size: [number, number, number] = [shape.size[0] * s[0], shape.size[1] * s[1], shape.size[2] * s[2]];
  // A sphere or an octahedron is given by its radius alone: all three, as the renderer's are.
  if (shape.kind === KIND.sphere || shape.kind === KIND.octahedron) size[1] = size[2] = size[0];
  return size;
}

interface Drawn {
  mesh: THREE.Mesh;
  shape: AnalyticShape;
  /** Its instances' proxies in the nearer and farther glass scene, and their wireframes. */
  near: THREE.InstancedMesh;
  far: THREE.InstancedMesh;
  nearWire: THREE.InstancedMesh;
  farWire: THREE.InstancedMesh;
  passes: StudioPasses;
}

const matrix = new THREE.Matrix4();
const instance = new THREE.Matrix4();
const position = new THREE.Vector3();
const quaternion = new THREE.Quaternion();
const scale = new THREE.Vector3();
const colour = new THREE.Color();

export class StudioMeshes {
  private drawn: Drawn[] = [];
  /** Per object: which mesh and which of its instances it is, and its world matrix. */
  private objectMesh = new Uint32Array(MAX_OBJECTS);
  private objectInstance = new Uint32Array(MAX_OBJECTS);
  private objectMatrix = new Float32Array(MAX_OBJECTS * 16);
  private compiled: Promise<void> | null = null;
  ready = false;

  constructor(
    private readonly studio: GlassStudio,
    private readonly makeWireMaterial: () => THREE.Material,
  ) {}

  get meshes(): THREE.Mesh[] {
    return this.drawn.map((d) => d.mesh);
  }

  /** Use these meshes (null or empty: none); builds their programs without blocking. */
  set(meshes: THREE.Mesh[]): void {
    this.clear();
    for (const mesh of meshes) {
      const shape = studioShapeOf(mesh.geometry);
      if (!shape) throw new Error('StudioMeshes: not a solid the studio can trace');
      const capacity = Math.max(
        1,
        (mesh as THREE.InstancedMesh).isInstancedMesh ? (mesh as THREE.InstancedMesh).count : 1,
      );
      const proxy = (scene: THREE.Scene, material: THREE.Material, geometry: THREE.BufferGeometry) => {
        const p = new THREE.InstancedMesh(geometry, material, capacity);
        p.matrixAutoUpdate = false;
        p.frustumCulled = false;
        p.userData.studioProxy = true;
        scene.add(p);
        return p;
      };
      // Each half has its own instance order, so its own copy of the per-instance attributes
      // (the vertex data itself is shared).
      const halfGeometry = () => {
        const g = new THREE.BufferGeometry();
        g.index = mesh.geometry.index;
        for (const [name, attribute] of Object.entries(mesh.geometry.attributes)) {
          if (name !== 'glassVary') g.setAttribute(name, attribute);
        }
        g.setAttribute('glassVary', new THREE.InstancedBufferAttribute(new Float32Array(capacity * 3), 3));
        // (A plain geometry: the solid has to be declared, as its type and parameters are not
        // carried over.)
        g.userData = { ...mesh.geometry.userData, glassShape: shape };
        g.groups = mesh.geometry.groups;
        return g;
      };
      const material = mesh.material as THREE.Material;
      const near = proxy(this.studio.nearScene, material, halfGeometry());
      const far = proxy(this.studio.farScene, material, halfGeometry());
      for (const p of [near, far])
        p.instanceColor = new THREE.InstancedBufferAttribute(new Float32Array(capacity * 3), 3);
      const wire = (of: THREE.InstancedMesh, scene: THREE.Scene) => {
        const w = proxy(scene, this.makeWireMaterial(), of.geometry);
        w.instanceMatrix = of.instanceMatrix;
        w.userData.studioWire = true;
        return w;
      };
      this.drawn.push({
        mesh,
        shape,
        near,
        far,
        nearWire: wire(near, this.studio.nearScene),
        farWire: wire(far, this.studio.farScene),
        passes: addStudioPasses(this.studio, mesh),
      });
    }
    this.ready = this.drawn.length === 0;
    const all = this.drawn;
    this.compiled = (async () => {
      for (const d of all) {
        const objects: THREE.Object3D[] = [d.near, d.far, ...d.passes.proxies];
        for (const object of objects) await this.studio.compile(object);
      }
      if (this.drawn === all) this.ready = true;
    })();
  }

  /** Done building the programs of the current meshes. */
  whenReady(): Promise<void> {
    return this.compiled ?? Promise.resolve();
  }

  /** Read every instance into `objects` (after the meshes moved: update their passes too). */
  derive(objects: GlassObjects): void {
    let n = 0;
    this.drawn.forEach((d, m) => {
      const { mesh, shape } = d;
      mesh.updateWorldMatrix(true, false);
      d.passes.update();
      const instanced = (mesh as THREE.InstancedMesh).isInstancedMesh === true;
      const count = instanced ? (mesh as THREE.InstancedMesh).count : 1;
      const vary = mesh.geometry.getAttribute('glassVary') as THREE.BufferAttribute | undefined;
      const perInstance = (vary as THREE.InstancedBufferAttribute | undefined)?.isInstancedBufferAttribute === true;
      const colours = instanced ? (mesh as THREE.InstancedMesh).instanceColor : null;
      for (let k = 0; k < count && n < MAX_OBJECTS; k++, n++) {
        if (instanced) (mesh as THREE.InstancedMesh).getMatrixAt(k, instance);
        else instance.identity();
        matrix.multiplyMatrices(mesh.matrixWorld, instance);
        matrix.toArray(this.objectMatrix, n * 16);
        matrix.decompose(position, quaternion, scale);
        this.objectMesh[n] = m;
        this.objectInstance[n] = k;
        objects.kind[n] = shape.kind;
        objects.center.set([position.x, position.y, position.z], n * 3);
        objects.quat.set([quaternion.x, quaternion.y, quaternion.z, quaternion.w], n * 4);
        objects.size.set(studioSize(shape, scale), n * 3);
        const v = perInstance ? k : 0;
        objects.vary.set(vary ? [vary.getX(v), vary.getY(v), vary.getZ(v)] : [0, 1, 0], n * 3);
        if (colours) colour.fromArray(colours.array, k * 3);
        else colour.setRGB(1, 1, 1);
        objects.tint.set([colour.r, colour.g, colour.b], n * 3);
      }
    });
    objects.count = n;
  }

  /**
   * Fill the proxies after the studio's split: `order` the objects nearest first, `inNear`
   * whether each is in the nearer half. Returns whether each half has anything to draw.
   */
  distribute(objects: GlassObjects, order: Uint32Array, inNear: Uint8Array): { hasNear: boolean; hasFar: boolean } {
    for (const d of this.drawn) d.near.count = d.far.count = 0;
    let hasNear = false;
    let hasFar = false;
    for (let k = 0; k < objects.count; k++) {
      const i = order[k];
      const d = this.drawn[this.objectMesh[i]];
      const half = inNear[i] ? d.near : d.far;
      const slot = half.count++;
      matrix.fromArray(this.objectMatrix, i * 16);
      half.setMatrixAt(slot, matrix);
      half.setColorAt(slot, colour.fromArray(objects.tint, i * 3));
      (half.geometry.getAttribute('glassVary') as THREE.BufferAttribute).setXYZ(
        slot,
        objects.vary[i * 3],
        objects.vary[i * 3 + 1],
        objects.vary[i * 3 + 2],
      );
      if (inNear[i]) hasNear = true;
      else hasFar = true;
    }
    for (const d of this.drawn) {
      for (const half of [d.near, d.far]) {
        half.instanceMatrix.needsUpdate = true;
        if (half.instanceColor) half.instanceColor.needsUpdate = true;
        half.geometry.getAttribute('glassVary').needsUpdate = true;
      }
      d.nearWire.count = d.near.count;
      d.farWire.count = d.far.count;
    }
    return { hasNear, hasFar };
  }

  /** Show the glass or its wireframe (and the passes with the glass only). */
  setVisibility(wire: boolean): void {
    for (const d of this.drawn) {
      d.near.visible = !wire && d.near.count > 0;
      d.far.visible = !wire && d.far.count > 0;
      d.nearWire.visible = wire && d.near.count > 0;
      d.farWire.visible = wire && d.far.count > 0;
      d.passes.setMirrorVisible(!wire);
    }
  }

  /** Triangles drawn for the objects (each object's mesh once). */
  triangles(): number {
    let total = 0;
    for (const d of this.drawn) total += triangleCount(d.mesh.geometry) * (d.near.count + d.far.count);
    return total;
  }

  clear(): void {
    for (const d of this.drawn) {
      for (const p of [d.near, d.far, d.nearWire, d.farWire]) {
        p.removeFromParent();
        if (p === d.near || p === d.far) p.geometry.dispose();
        if (p === d.nearWire || p === d.farWire) (p.material as THREE.Material).dispose();
      }
      d.passes.dispose();
    }
    this.drawn = [];
    this.compiled = null;
    this.ready = true;
  }
}
