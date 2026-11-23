// Glass seen through glass for MeshGlassMaterial in plain three.js scenes, as the instanced
// renderer layers it: the glass meshes are split by distance into a farther and a nearer half;
// the scene without the nearer half (opaque objects and the farther glass) is rendered into a
// layer, with each pixel's distance from the camera in alpha; and the nearer glass looks into
// that layer with the renderer's own screen-space march (tsl/lookThrough.ts, through each
// material's `behindMap`). Use it in place of renderer.render():
//
//   const layers = new GlassLayers();
//   renderer.setAnimationLoop(() => layers.render(renderer, scene, camera));
//
// Meshes sharing one material share its layer: a material counts as nearer if any of its
// meshes is (give glass that should see other glass its own material).
/* eslint-disable @typescript-eslint/no-explicit-any -- TSL node types are too loose to spell out */
import * as THREE from 'three/webgpu';
import * as TSL from 'three/tsl';
import { MeshGlassMaterial } from './material';

const { cameraProjectionMatrixInverse, float, getViewPosition, length, select, texture, uv, vec4 } = TSL as any;

/** The distance written where nothing was drawn (the renderer's backdrop value). */
const BACKDROP_DISTANCE = 1000;

export interface GlassLayersOptions {
  /** Resolution of the layer relative to the drawing buffer (default 1). */
  resolutionScale?: number;
  /**
   * Multisampling of the layer (default 4): matching the canvas's antialiasing keeps edges seen
   * through glass the same as transmission shows them.
   */
  samples?: number;
}

export class GlassLayers {
  readonly resolutionScale: number;
  /** The farther layer the nearer glass looks into (rgb, distance from the camera in alpha). */
  readonly target: THREE.RenderTarget;
  private readonly sceneTarget: THREE.RenderTarget;
  private readonly composeQuad: THREE.QuadMesh;
  private readonly colorNode: any;
  private readonly depthNode: any;
  private readonly size = new THREE.Vector2();
  private readonly centre = new THREE.Vector3();
  private readonly forward = new THREE.Vector3();
  /** Materials given the layer last frame (so they can be released when they move away). */
  private readonly assigned = new Set<MeshGlassMaterial>();

  constructor(options: GlassLayersOptions = {}) {
    this.resolutionScale = Math.max(0.25, Math.min(1, options.resolutionScale ?? 1));
    const settings = {
      type: THREE.HalfFloatType,
      format: THREE.RGBAFormat,
      minFilter: THREE.LinearFilter,
      magFilter: THREE.LinearFilter,
    };
    this.sceneTarget = new THREE.RenderTarget(4, 4, { ...settings, depthBuffer: true, samples: options.samples ?? 4 });
    this.sceneTarget.depthTexture = new THREE.DepthTexture(4, 4);
    this.target = new THREE.RenderTarget(4, 4, { ...settings, depthBuffer: false });

    // The scene's colour, and its depth turned into distance from the camera.
    this.colorNode = texture(this.sceneTarget.texture);
    this.depthNode = texture(this.sceneTarget.depthTexture);
    const depth = this.depthNode.sample(uv()).r;
    const viewPosition = getViewPosition(uv(), depth, cameraProjectionMatrixInverse);
    const distance = select(depth.greaterThanEqual(0.999999), float(BACKDROP_DISTANCE), length(viewPosition));
    const material = new THREE.NodeMaterial();
    material.fragmentNode = vec4(this.colorNode.sample(uv()).rgb, distance);
    material.depthTest = material.depthWrite = false;
    material.blending = THREE.NoBlending;
    this.composeQuad = new THREE.QuadMesh(material);
  }

  /** Render `scene` with glass seen through glass. */
  render(renderer: THREE.WebGPURenderer, scene: THREE.Scene, camera: THREE.Camera): void {
    const glass: { mesh: THREE.Mesh; material: MeshGlassMaterial; distance: number }[] = [];
    camera.updateMatrixWorld();
    camera.getWorldDirection(this.forward);
    scene.traverseVisible((object) => {
      const mesh = object as THREE.Mesh;
      if (!mesh.isMesh || !(mesh.material instanceof MeshGlassMaterial)) return;
      if (!mesh.geometry.boundingSphere) mesh.geometry.computeBoundingSphere();
      this.centre.copy(mesh.geometry.boundingSphere!.center).applyMatrix4(mesh.matrixWorld);
      glass.push({ mesh, material: mesh.material, distance: this.centre.sub(camera.position).dot(this.forward) });
    });
    if (glass.length < 2) {
      // Nothing to see through: plain rendering, every material on its own transmission.
      this.release(new Set());
      renderer.render(scene, camera);
      return;
    }
    glass.sort((a, b) => a.distance - b.distance);
    const near = glass.slice(0, glass.length >> 1);
    const nearMaterials = new Set(near.map((g) => g.material));

    // The farther layer: the scene without the nearer glass (and without its layer).
    renderer.getDrawingBufferSize(this.size);
    const w = Math.max(1, Math.round(this.size.x * this.resolutionScale));
    const h = Math.max(1, Math.round(this.size.y * this.resolutionScale));
    if (this.target.width !== w || this.target.height !== h) {
      this.sceneTarget.setSize(w, h);
      this.target.setSize(w, h);
    }
    this.release(nearMaterials);
    const hidden: THREE.Object3D[] = [];
    for (const g of glass) {
      if (nearMaterials.has(g.material) && g.mesh.visible) {
        g.mesh.visible = false;
        hidden.push(g.mesh);
      }
    }
    const previousTarget = renderer.getRenderTarget();
    try {
      renderer.setRenderTarget(this.sceneTarget);
      renderer.render(scene, camera);
      renderer.setRenderTarget(this.target);
      this.composeQuad.render(renderer);
    } finally {
      for (const mesh of hidden) mesh.visible = true;
      renderer.setRenderTarget(previousTarget);
    }

    // The nearer glass looks into it.
    for (const material of nearMaterials) {
      if (material.behindMap !== this.target.texture) material.behindMap = this.target.texture;
      this.assigned.add(material);
    }
    renderer.render(scene, camera);
  }

  /** Release the GPU resources (and the materials' layer). */
  dispose(): void {
    this.release(new Set());
    this.sceneTarget.dispose();
    this.target.dispose();
  }

  /** Take the layer back from materials that are no longer nearer. */
  private release(keep: Set<MeshGlassMaterial>): void {
    for (const material of this.assigned) {
      if (keep.has(material)) continue;
      if (material.behindMap === this.target.texture) material.behindMap = null;
      this.assigned.delete(material);
    }
  }
}
