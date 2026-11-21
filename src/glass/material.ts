// Standalone, geometry-agnostic Three.js NodeMaterial for real-time physical glass
// (WebGPU & WebGL 2). Matches MeshPhysicalMaterial ergonomics: attach to any THREE.Mesh,
// optionally set `envMap` for HDRI lighting and `transmission` (0–1) for automatic
// mipmapped scene refraction via Three.js `viewportOpaqueMipTexture`.
/* eslint-disable @typescript-eslint/no-explicit-any -- TSL node types are too loose to spell out */
import * as THREE from 'three/webgpu';
import * as TSL from 'three/tsl';
import { applyFrostVeil, perturbFrostNormal } from './tsl/frost';
import { fresnel, SPECTRAL_COUNT, spectralCauchy, spectralWeight } from './tsl/optics';
import { makeStudioEnv } from './tsl/studioEnv';

type N = any;
const {
  Fn,
  If,
  Loop,
  abs,
  batchColor,
  cameraPosition,
  clamp,
  diffuseColor,
  dot,
  exp,
  float,
  instanceColor,
  int,
  length,
  log,
  materialOpacity,
  max,
  mix,
  normalWorld,
  normalize,
  positionLocal,
  positionWorld,
  reflect,
  refract,
  screenUV,
  select,
  texture,
  uniform,
  vec2,
  vec3,
  vec4,
  vertexColor,
  viewportOpaqueMipTexture,
} = TSL as N;

export interface MeshGlassMaterialParameters {
  /** Index of refraction (1.0 – 2.5; default 1.52 for crown glass). */
  ior?: number;
  /** Chromatic dispersion strength (0 = none, 1 = vivid prism fire; default 0.5). */
  dispersion?: number;
  /** Frosted glass roughness (0 = crystal clear, 1 = fully frosted; default 0). */
  roughness?: number;
  /** Frosted glass grain fineness (higher = smaller micro-particles; default 4). */
  frostScale?: number;
  /** Volumetric glass tint color (Beer-Lambert transmittance across `thickness`; default white). */
  color?: THREE.ColorRepresentation;
  /** Characteristic object thickness in world units for refraction & absorption (default 1.0). */
  thickness?: number;
  /**
   * Automatic scene transmission (`0` – `1`; default `0`), matching `MeshPhysicalMaterial`.
   * When `> 0`, automatically refracts opaque meshes behind the glass via Three.js
   * `viewportOpaqueMipTexture` with roughness-driven mipmap blur.
   */
  transmission?: number;
  /** Optional equirectangular HDRI / environment texture (overrides the neutral fallback studio). */
  envMap?: THREE.Texture | null;
  /** Intensity multiplier for `envMap` (default 1.0). */
  envMapIntensity?: number;
  /**
   * Optional back-face normal & depth map from `GlassBackfacePass` (`rgb = outward normal`,
   * `a = camera distance`) for exact exit refraction & thickness on arbitrary concave meshes.
   */
  backfaceMap?: THREE.Texture | null;
}

/** Shared 1x1 white fallback texture so optional texture nodes bind cleanly without per-instance leaks. */
export const FALLBACK_ENV_TEXTURE = (() => {
  const tex = new THREE.DataTexture(new Uint8Array([255, 255, 255, 255]), 1, 1);
  tex.wrapS = THREE.RepeatWrapping;
  tex.wrapT = THREE.ClampToEdgeWrapping;
  tex.minFilter = THREE.LinearMipmapLinearFilter;
  tex.magFilter = THREE.LinearFilter;
  tex.generateMipmaps = true;
  tex.needsUpdate = true;
  return tex;
})();

/**
 * Standalone, geometry-agnostic physical glass material for Three.js WebGPU & WebGL 2.
 *
 * Attach directly to any `THREE.Mesh`:
 * ```ts
 * const glass = new MeshGlassMaterial({
 *   ior: 1.52,
 *   dispersion: 0.6,
 *   roughness: 0.15,
 *   thickness: 1.2,
 *   transmission: 1,
 *   envMap: hdr,
 * });
 * const mesh = new THREE.Mesh(new THREE.TorusKnotGeometry(1, 0.35, 160, 32), glass);
 * scene.add(mesh);
 * ```
 */
export class MeshGlassMaterial extends THREE.MeshBasicNodeMaterial {
  private _envMap: THREE.Texture | null = null;
  private _backfaceMap: THREE.Texture | null = null;
  private _transmission = 0;

  private readonly uIor = uniform(1.52);
  private readonly uDispersion = uniform(0.5);
  private readonly uRoughness = uniform(0);
  private readonly uFrostScale = uniform(4);
  private readonly uThickness = uniform(1);
  private readonly uTransmission = uniform(0);
  private readonly uHasEnvMap = uniform(0);
  private readonly uEnvMapIntensity = uniform(1);
  private readonly uHasBackface = uniform(0);
  private readonly envTexNode: N;
  private readonly backfaceTexNode: N;

  constructor(params: MeshGlassMaterialParameters = {}) {
    super();

    if (params.color !== undefined) this.color.set(params.color);
    this.uIor.value = params.ior ?? 1.52;
    this.uDispersion.value = params.dispersion ?? 0.5;
    this.uRoughness.value = params.roughness ?? 0;
    this.uFrostScale.value = params.frostScale ?? 4;
    this.uThickness.value = params.thickness ?? 1;
    this._transmission = Math.max(0, Math.min(1, params.transmission ?? 0));
    this.uTransmission.value = this._transmission;
    this.transparent = this._transmission > 0;
    this.uEnvMapIntensity.value = params.envMapIntensity ?? 1;

    this._envMap = params.envMap ?? null;
    this.envTexNode = texture(this._envMap ?? FALLBACK_ENV_TEXTURE);
    this.uHasEnvMap.value = this._envMap ? 1 : 0;

    this._backfaceMap = params.backfaceMap ?? null;
    this.backfaceTexNode = texture(this._backfaceMap ?? FALLBACK_ENV_TEXTURE);
    this.uHasBackface.value = this._backfaceMap ? 1 : 0;

    this.colorNode = this.buildColorNode();
  }

  /* Reactive optical properties (live GPU uniform updates, no recompile) */

  get ior(): number {
    return this.uIor.value;
  }
  set ior(v: number) {
    this.uIor.value = v;
  }

  get dispersion(): number {
    return this.uDispersion.value;
  }
  set dispersion(v: number) {
    this.uDispersion.value = v;
  }

  get roughness(): number {
    return this.uRoughness.value;
  }
  set roughness(v: number) {
    this.uRoughness.value = v;
  }

  get frostScale(): number {
    return this.uFrostScale.value;
  }
  set frostScale(v: number) {
    this.uFrostScale.value = v;
  }

  get thickness(): number {
    return this.uThickness.value;
  }
  set thickness(v: number) {
    this.uThickness.value = v;
  }

  get transmission(): number {
    return this._transmission;
  }
  set transmission(v: number) {
    const clamped = Math.max(0, Math.min(1, v));
    const wasActive = this._transmission > 0;
    const isActive = clamped > 0;
    this._transmission = clamped;
    this.uTransmission.value = clamped;
    if (wasActive !== isActive) {
      this.transparent = isActive;
      this.colorNode = this.buildColorNode();
      this.needsUpdate = true;
    }
  }

  override get envMap(): THREE.Texture | null {
    return this._envMap;
  }
  override set envMap(tex: THREE.Texture | null) {
    this._envMap = tex;
    // Guard for `super()` calling `setDefaultValues()` before subclass fields initialize.
    if (this.envTexNode) {
      this.envTexNode.value = tex ?? FALLBACK_ENV_TEXTURE;
      this.uHasEnvMap.value = tex ? 1 : 0;
    }
  }

  get envMapIntensity(): number {
    return this.uEnvMapIntensity.value;
  }
  set envMapIntensity(v: number) {
    this.uEnvMapIntensity.value = v;
  }

  get backfaceMap(): THREE.Texture | null {
    return this._backfaceMap;
  }
  set backfaceMap(tex: THREE.Texture | null) {
    this._backfaceMap = tex;
    if (this.backfaceTexNode) {
      this.backfaceTexNode.value = tex ?? FALLBACK_ENV_TEXTURE;
      this.uHasBackface.value = tex ? 1 : 0;
    }
  }

  /** Create an independent clone of this `MeshGlassMaterial` with identical optical parameters. */
  override clone(): this {
    return new MeshGlassMaterial({
      ior: this.ior,
      dispersion: this.dispersion,
      roughness: this.roughness,
      frostScale: this.frostScale,
      color: this.color.clone(),
      thickness: this.thickness,
      transmission: this.transmission,
      envMap: this._envMap,
      envMapIntensity: this.envMapIntensity,
      backfaceMap: this._backfaceMap,
    }) as this;
  }

  /**
   * Route per-instance (`InstancedMesh.setColorAt`), batched (`BatchedMesh.setColorAt`),
   * and vertex colors into volumetric Beer–Lambert absorption instead of post-multiplying
   * the final shaded output (which would tint white dielectric Fresnel highlights).
   */
  override setupDiffuseColor(builder: N): void {
    const { object, geometry } = builder;
    const hasVertexColor = this.vertexColors === true && Boolean(geometry?.hasAttribute?.('color'));
    const hasInstanceColor = Boolean(object?.instanceColor);
    const hasBatchColor = Boolean(object?.isBatchedMesh && object?._colorsTexture);
    if (!hasVertexColor && !hasInstanceColor && !hasBatchColor) {
      super.setupDiffuseColor(builder);
      return;
    }
    let tint: N = uniform(this.color);
    if (hasVertexColor) tint = tint.mul(vertexColor().rgb);
    if (hasInstanceColor) tint = tint.mul(instanceColor);
    if (hasBatchColor) tint = tint.mul(batchColor);
    diffuseColor.assign(this.buildColorNode(tint));
    const opacityNode = this.opacityNode ? float(this.opacityNode) : materialOpacity;
    diffuseColor.a.assign(diffuseColor.a.mul(opacityNode));
  }

  /** Two-interface dielectric refraction, TIR, Beer-Lambert absorption, 6-band CIE dispersion & frost. */
  private buildColorNode(tintNode: N = uniform(this.color)): N {
    const uTint = tintNode;
    const { env } = makeStudioEnv(this.uRoughness, {
      envTexNode: this.envTexNode,
      uHasEnvMap: this.uHasEnvMap,
      uEnvMapIntensity: this.uEnvMapIntensity,
    });

    const useTransmission = this._transmission > 0;
    const uIor = this.uIor;
    const uDispersion = this.uDispersion;
    const uRoughness = this.uRoughness;
    const uFrostScale = this.uFrostScale;
    const uThickness = this.uThickness;
    const uTransmission = this.uTransmission;

    const lookEnv = (rd: N, bendOffset: N) => {
      if (!useTransmission) return env(rd);
      const uv = clamp(screenUV.add(bendOffset), vec2(0.002), vec2(0.998));
      const bg = viewportOpaqueMipTexture(uv, uRoughness.mul(5)).rgb;
      return mix(env(rd), bg, uTransmission);
    };

    return Fn(() => {
      const d0 = normalize(positionWorld.sub(cameraPosition)).toVar();
      const n0 = normalize(normalWorld).toVar();
      If(dot(d0, n0).greaterThan(0), () => {
        n0.assign(n0.negate());
      });

      const reach = max(uThickness, 0.05).toVar();
      const frostGrain = float(0.5).toVar();
      perturbFrostNormal(n0, d0, positionLocal, reach, uRoughness, uFrostScale, frostGrain);

      const ior = max(uIor, 1.0001).toVar();
      const disp = uDispersion.div(12).toVar();
      const layers = select(disp.greaterThan(0).and(uRoughness.lessThan(0.35)), int(SPECTRAL_COUNT), int(1)).toVar();
      const sigma = log(max(uTint, vec3(1e-4)))
        .negate()
        .div(max(reach, 1e-4))
        .toVar();
      const col = vec3(0).toVar();

      const d0DotN0 = dot(d0, n0).toVar();
      const outerRefl = env(reflect(d0, n0)).toVar();
      const frontDist = length(positionWorld.sub(cameraPosition)).toVar();
      const bfSample = vec4(0).toVar();
      const hasExactExit = float(0).toVar();
      If(this.uHasBackface.greaterThan(0.5), () => {
        bfSample.assign(this.backfaceTexNode.sample(screenUV).level(0));
        If(bfSample.a.greaterThan(frontDist.add(1e-4)).and(dot(bfSample.rgb, bfSample.rgb).greaterThan(1e-4)), () => {
          hasExactExit.assign(1);
        });
      });

      Loop({ start: int(0), end: layers, type: 'int', condition: '<', name: 'lc' }, ({ lc }: { lc: N }) => {
        const bandOffset = select(layers.equal(1), float(0), spectralCauchy(lc));
        const iorIn = max(ior.add(disp.mul(bandOffset)), 1.0001).toVar();
        const f0 = fresnel(d0DotN0, iorIn).toVar();
        const layer = f0.mul(outerRefl).toVar();
        const thr = float(1).sub(f0).toVar();
        const t1 = refract(d0, n0, float(1).div(iorIn)).toVar();
        If(dot(t1, t1).greaterThan(1e-6), () => {
          // Use exact back-face chord thickness & outward normal when `backfaceMap` is bound;
          // otherwise fall back to analytic chord estimation from `thickness`.
          const dist = select(
            hasExactExit.greaterThan(0.5),
            max(bfSample.a.sub(frontDist), 0.01),
            reach.mul(mix(0.35, 1, abs(dot(t1, n0)))),
          ).toVar();
          const nExit = select(
            hasExactExit.greaterThan(0.5),
            normalize(bfSample.rgb),
            normalize(mix(t1, n0.negate(), 0.55)),
          ).toVar();
          const absorb = exp(sigma.mul(dist.negate())).toVar();
          const t2 = refract(t1, nExit.negate(), iorIn).toVar();
          const f1 = fresnel(dot(t1, nExit), float(1).div(iorIn)).toVar();
          If(dot(t2, t2).lessThan(1e-6), () => {
            f1.assign(1);
          });
          const outW = thr.mul(float(1).sub(f1)).toVar();
          If(outW.greaterThan(0.004), () => {
            const bend = t2.sub(d0).xy.mul(0.12);
            layer.addAssign(absorb.mul(outW).mul(lookEnv(t2, bend)));
          });
          const bounceW = thr.mul(f1).toVar();
          If(bounceW.greaterThan(0.004), () => {
            const rInt = reflect(t1, nExit.negate());
            const bend = rInt.sub(d0).xy.mul(0.12);
            layer.addAssign(absorb.mul(absorb).mul(bounceW).mul(lookEnv(rInt, bend)));
          });
        });
        const weight = select(layers.equal(1), vec3(1), spectralWeight(lc));
        col.addAssign(layer.mul(weight));
      });

      // Creamy frosted veil + crisp Fresnel rim when roughness > 0.
      applyFrostVeil(col, d0, n0, uTint, vec3(0.75), uRoughness, frostGrain);
      return vec4(col, 1);
    })();
  }
}
