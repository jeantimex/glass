// The instanced renderer's other passes over the glass, drawn from standard meshes: the
// mirror image in the floor, the depth along the lamp (shadows on the glass), the light
// map's filter (shadows and tint on the floor) and the real caustics' rays. Each pass gets a
// proxy of the mesh (sharing its geometry and instance data) in the renderer's scene for it,
// with a material that computes what the renderer's own does, through the mesh's frame.
/* eslint-disable @typescript-eslint/no-explicit-any -- TSL node types are too loose to spell out */
import * as THREE from 'three/webgpu';
import * as TSL from 'three/tsl';
import { analyticShape, type AnalyticShape } from './analyticShapes';
import { instanceMatrixNode, MeshGlassMaterial } from './material';
import { studioDeformedPosition, studioShapeInputs, type GlassStudio } from './studioGlass';

const { attribute, instanceColor, inverse, modelWorldMatrix, modelWorldMatrixInverse, positionWorld, vec3 } =
  TSL as any;

/** The data passes (the mirror image is glass: MeshGlassMaterial draws it). */
export type StudioPass = 'lampDepth' | 'caustic' | 'photon';

/** One of the renderer's data passes for a standard mesh, set up as the renderer's own. */
export class StudioPassMaterial extends THREE.NodeMaterial {
  constructor(
    readonly studio: GlassStudio,
    readonly shape: AnalyticShape,
    readonly pass: StudioPass,
  ) {
    super();
    if (pass === 'lampDepth') {
      this.side = THREE.DoubleSide;
    } else if (pass === 'caustic') {
      this.side = THREE.DoubleSide;
      this.depthTest = false;
      this.depthWrite = false;
      this.transparent = true;
      this.blending = THREE.CustomBlending;
      this.blendEquation = THREE.AddEquation;
      this.blendSrc = THREE.DstColorFactor; // result = source x destination
      this.blendDst = THREE.ZeroFactor;
    } else {
      // Its output is data (negative as often as not): written untouched.
      this.transparent = true;
      this.blending = THREE.NoBlending;
    }
  }

  override customProgramCacheKey(): string {
    return `${super.customProgramCacheKey()}|studioPass:${this.pass}`;
  }

  override setup(builder: any): any {
    const object = builder.object;
    const instanced = Boolean(object?.isInstancedMesh && object?.instanceMatrix);
    const instanceMatrix = instanced ? instanceMatrixNode(builder, object) : null;
    const toWorld = instanceMatrix ? modelWorldMatrix.mul(instanceMatrix) : modelWorldMatrix;
    const toGeometry = instanceMatrix ? inverse(instanceMatrix).mul(modelWorldMatrixInverse) : modelWorldMatrixInverse;
    const tint = object?.instanceColor ? instanceColor : vec3(1);
    const vary = object?.geometry?.getAttribute('glassVary') ? attribute('glassVary', 'vec3') : vec3(0, 1, 0);
    const { inputs, normal } = studioShapeInputs(this.studio, this.shape, toWorld, toGeometry, tint, vary);
    // (A deformed capsule or torus is drawn as the renderer draws it.)
    this.positionNode = studioDeformedPosition(this.shape, toWorld, instanceMatrix);
    const nodes = this.studio.passNodes;
    this.fragmentNode =
      this.pass === 'lampDepth'
        ? nodes.lampDepth()
        : this.pass === 'caustic'
          ? nodes.caustic(normal, positionWorld, inputs)
          : nodes.photon(this.shape.kind, positionWorld, normal, inputs, this.shape.diamondPlanes);
    // The shape's size is per object: this build is not shared with other objects.
    builder.objectSpecific = true;
    return super.setup(builder);
  }
}

/** The floor's mirror (y to -y). */
const MIRROR = new THREE.Matrix4().makeScale(1, -1, 1);

/** A standard mesh's proxies in the renderer's passes; `update()` after moving the mesh. */
export interface StudioPasses {
  /** The proxies, one per pass (the mirror image first). */
  readonly proxies: THREE.Mesh[];
  update(): void;
  /** Show or hide the mirror image (hidden while the glass is shown as a wireframe). */
  setMirrorVisible(visible: boolean): void;
  dispose(): void;
}

/**
 * Draw a standard mesh (a THREE.Mesh or InstancedMesh with MeshGlassMaterial in studio mode,
 * on a solid the studio can trace) in all of the renderer's other passes, as its own objects
 * are: the mirror image, shadows, the light map and the real caustics.
 */
export function addStudioPasses(studio: GlassStudio, mesh: THREE.Mesh): StudioPasses {
  const shape = (mesh.geometry.userData.glassShape as AnalyticShape | undefined) ?? analyticShape(mesh.geometry);
  if (!shape) throw new Error('addStudioPasses: not a solid the studio can trace');
  const instanced = (mesh as THREE.InstancedMesh).isInstancedMesh === true;
  const proxies: { proxy: THREE.Mesh; mirrored: boolean }[] = [];
  const add = (material: THREE.Material, scene: THREE.Scene, mirrored = false) => {
    let proxy: THREE.Mesh;
    if (instanced) {
      const source = mesh as THREE.InstancedMesh;
      const copy = new THREE.InstancedMesh(mesh.geometry, material, source.count);
      copy.instanceMatrix = source.instanceMatrix;
      copy.instanceColor = source.instanceColor;
      proxy = copy;
    } else {
      proxy = new THREE.Mesh(mesh.geometry, material);
    }
    proxy.matrixAutoUpdate = false;
    proxy.frustumCulled = false;
    proxy.userData.studioProxy = true;
    scene.add(proxy);
    proxies.push({ proxy, mirrored });
  };
  const scenes = studio.passScenes;
  add(new MeshGlassMaterial({ studio, studioMirror: true }), scenes.mirror, true);
  add(new StudioPassMaterial(studio, shape, 'lampDepth'), scenes.lampDepth);
  add(new StudioPassMaterial(studio, shape, 'caustic'), scenes.light);
  add(new StudioPassMaterial(studio, shape, 'photon'), scenes.photon);
  const update = () => {
    mesh.updateWorldMatrix(true, false);
    for (const { proxy, mirrored } of proxies) {
      if (mirrored) proxy.matrix.multiplyMatrices(MIRROR, mesh.matrixWorld);
      else proxy.matrix.copy(mesh.matrixWorld);
      proxy.matrixWorldNeedsUpdate = true;
    }
  };
  update();
  return {
    get proxies() {
      return proxies.map((entry) => entry.proxy);
    },
    update,
    setMirrorVisible: (visible: boolean) => {
      for (const { proxy, mirrored } of proxies) if (mirrored) proxy.visible = visible;
    },
    dispose: () => {
      for (const { proxy } of proxies) {
        proxy.removeFromParent();
        (proxy.material as THREE.Material).dispose();
      }
      proxies.length = 0;
    },
  };
}
