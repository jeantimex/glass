// Generic back-face depth & outward-normal pre-pass for arbitrary concave meshes
// (Suzanne, Stanford Dragon, TorusKnot, custom GLTFs). Feeds `MeshGlassMaterial.backfaceMap`
// so two-interface refraction uses exact chord thickness and exact back-face exit normals.
/* eslint-disable @typescript-eslint/no-explicit-any -- TSL node types are too loose to spell out */
import * as THREE from 'three/webgpu';
import * as TSL from 'three/tsl';
import { MeshGlassMaterial } from './material';

const { cameraPosition, length, normalWorld, normalize, positionWorld, vec4 } = TSL as any;

export interface GlassBackfacePassOptions {
  /** Render target resolution scale relative to the renderer drawing buffer (default `0.5`). */
  resolutionScale?: number;
}

/** Check whether a mesh uses `MeshGlassMaterial` (directly or in a material array). */
function hasGlassMaterial(mesh: THREE.Mesh): boolean {
  const mat = mesh.material;
  if (Array.isArray(mat)) return mat.some((m) => m instanceof MeshGlassMaterial);
  return mat instanceof MeshGlassMaterial;
}

/**
 * Offscreen `THREE.BackSide` pre-pass that records each glass mesh's nearest back-face
 * world-space outward normal (`rgb`) and distance from the camera (`a`) into a half-float
 * texture. Bind `pass.texture` to `MeshGlassMaterial.backfaceMap` for exact exit normals
 * and Beer–Lambert chord thickness on arbitrary concave geometries (no offline SDF needed).
 *
 * ```ts
 * const backfacePass = new GlassBackfacePass();
 * const glass = new MeshGlassMaterial({
 *   ior: 1.52,
 *   dispersion: 0.6,
 *   transmission: 1,
 *   backfaceMap: backfacePass.texture,
 * });
 *
 * // Each frame before renderer.render(scene, camera):
 * backfacePass.render(renderer, scene, camera);
 * renderer.render(scene, camera);
 * ```
 */
export class GlassBackfacePass {
  readonly target: THREE.RenderTarget;
  readonly material: THREE.NodeMaterial;
  readonly resolutionScale: number;
  private readonly sizeScratch = new THREE.Vector2();
  private readonly clearColorScratch = new THREE.Color();

  constructor(options: GlassBackfacePassOptions = {}) {
    this.resolutionScale = Math.max(0.125, Math.min(1, options.resolutionScale ?? 0.5));
    this.target = new THREE.RenderTarget(4, 4, {
      type: THREE.HalfFloatType,
      format: THREE.RGBAFormat,
      depthBuffer: true,
      minFilter: THREE.LinearFilter,
      magFilter: THREE.LinearFilter,
    });

    // Pack outward back-face world normal in RGB and camera distance in A (0 = no back face).
    this.material = new THREE.NodeMaterial();
    this.material.side = THREE.BackSide;
    this.material.transparent = false;
    this.material.blending = THREE.NoBlending;
    this.material.fragmentNode = vec4(normalize(normalWorld), length(positionWorld.sub(cameraPosition)));
  }

  /** Half-float RGBA texture (`rgb = outward back-face normal`, `a = camera distance`). */
  get texture(): THREE.Texture {
    return this.target.texture;
  }

  /** Resize the internal render target explicitly. */
  setSize(width: number, height: number): void {
    const w = Math.max(1, Math.round(width * this.resolutionScale));
    const h = Math.max(1, Math.round(height * this.resolutionScale));
    if (this.target.width !== w || this.target.height !== h) {
      this.target.setSize(w, h);
    }
  }

  /**
   * Render back-faces of all visible `MeshGlassMaterial` meshes in `scene` into `this.target`.
   * Non-glass meshes are temporarily hidden and restored in a `finally` block.
   */
  render(renderer: THREE.WebGPURenderer, scene: THREE.Scene, camera: THREE.Camera): void {
    renderer.getDrawingBufferSize(this.sizeScratch);
    this.setSize(this.sizeScratch.x, this.sizeScratch.y);

    const hidden: THREE.Object3D[] = [];
    scene.traverse((obj) => {
      if (!obj.visible) return;
      if ((obj as THREE.Mesh).isMesh && !hasGlassMaterial(obj as THREE.Mesh)) {
        obj.visible = false;
        hidden.push(obj);
      }
    });

    const prevOverride = scene.overrideMaterial;
    const prevTarget = renderer.getRenderTarget();
    const prevAutoClear = renderer.autoClear;
    renderer.getClearColor(this.clearColorScratch);
    const prevClearAlpha = renderer.getClearAlpha();

    try {
      scene.overrideMaterial = this.material;
      renderer.setRenderTarget(this.target);
      renderer.autoClear = true;
      renderer.setClearColor(0x000000, 0);
      renderer.render(scene, camera);
    } finally {
      scene.overrideMaterial = prevOverride;
      for (let i = 0; i < hidden.length; i++) hidden[i].visible = true;
      renderer.setRenderTarget(prevTarget);
      renderer.autoClear = prevAutoClear;
      renderer.setClearColor(this.clearColorScratch, prevClearAlpha);
    }
  }

  /** Release GPU render target and material resources. */
  dispose(): void {
    this.target.dispose();
    this.material.dispose();
  }
}
