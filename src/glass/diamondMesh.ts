// Standalone Three.js DiamondMesh pairing watertight convex facet geometry with
// exact multi-bounce internal raytracing across all 6 diamond cuts (WebGPU & WebGL 2).
/* eslint-disable @typescript-eslint/no-explicit-any -- TSL node types are too loose to spell out */
import * as THREE from 'three/webgpu';
import * as TSL from 'three/tsl';
import { DIAMOND_CUTS, diamondGeometry, type DiamondCut } from './diamond';
import { FALLBACK_ENV_TEXTURE } from './material';
import { KIND } from './objects';
import { applyFrostVeil, perturbFrostNormal } from './tsl/frost';
import { fresnel, spectralJitter, spectralSample, spectralSampleCount } from './tsl/optics';
import { shapeFunctions } from './tsl/shapes';
import { makeStudioEnv } from './tsl/studioEnv';

type N = any;
const {
  Break,
  Fn,
  If,
  Loop,
  cameraPosition,
  clamp,
  dot,
  exp,
  float,
  int,
  log,
  max,
  mix,
  modelWorldMatrix,
  modelWorldMatrixInverse,
  normalLocal,
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
  viewportOpaqueMipTexture,
} = TSL as N;

export interface DiamondMeshParameters {
  /** Facet cut (`'round' | 'princess' | 'emerald' | 'asscher' | 'cushion' | 'radiant'`; default `'round'`). */
  cut?: DiamondCut;
  /** Index of refraction (default `2.42` for real diamond). */
  ior?: number;
  /** Chromatic dispersion / spectral fire (default `0.4`). */
  dispersion?: number;
  /** Maximum internal reflection/refraction bounces (default `6`). */
  bounces?: number;
  /** Frosted surface roughness (`0` = polished crystal, `1` = frosted; default `0`). */
  roughness?: number;
  /** Frosted grain fineness (default `4`). */
  frostScale?: number;
  /** Volumetric diamond tint color (Beer-Lambert absorption; default white). */
  color?: THREE.ColorRepresentation;
  /** Automatic scene transmission (`0` – `1`; default `0`), matching `MeshPhysicalMaterial`. */
  transmission?: number;
  /** Optional equirectangular HDRI / environment texture. */
  envMap?: THREE.Texture | null;
  /** Intensity multiplier for `envMap` (default `1.0`). */
  envMapIntensity?: number;
}

/**
 * Self-contained Three.js diamond mesh with exact multi-bounce internal facet raytracing.
 *
 * Owns both the watertight facet `BufferGeometry` (`diamondGeometry(cut)`) and the
 * matching TSL facet-plane raytracing shader so changing `diamond.cut` atomically updates
 * both geometry and optics:
 * ```ts
 * const diamond = new DiamondMesh({ cut: 'princess', envMap: hdr });
 * diamond.restOnFloor();
 * scene.add(diamond);
 *
 * diamond.cut = 'emerald'; // swaps geometry + internal facet shader
 * ```
 */
export class DiamondMesh extends THREE.Mesh {
  private _cut: DiamondCut;
  private _envMap: THREE.Texture | null = null;
  private _transmission = 0;

  private readonly uIor = uniform(2.42);
  private readonly uDispersion = uniform(0.4);
  private readonly uBounces = uniform(6);
  private readonly uRoughness = uniform(0);
  private readonly uFrostScale = uniform(4);
  private readonly uTransmission = uniform(0);
  private readonly uHasEnvMap = uniform(0);
  private readonly uEnvMapIntensity = uniform(1);
  private readonly envTexNode: N;
  private readonly diamondMaterial: THREE.MeshBasicNodeMaterial;

  constructor(params: DiamondMeshParameters = {}) {
    const cut: DiamondCut = params.cut ?? 'round';
    const mat = new THREE.MeshBasicNodeMaterial();
    super(diamondGeometry(cut), mat);

    this._cut = cut;
    this.diamondMaterial = mat;

    if (params.color !== undefined) mat.color.set(params.color);
    this.uIor.value = params.ior ?? 2.42;
    this.uDispersion.value = params.dispersion ?? 0.4;
    this.uBounces.value = params.bounces ?? 6;
    this.uRoughness.value = params.roughness ?? 0;
    this.uFrostScale.value = params.frostScale ?? 4;
    this._transmission = Math.max(0, Math.min(1, params.transmission ?? 0));
    this.uTransmission.value = this._transmission;
    mat.transparent = this._transmission > 0;
    this.uEnvMapIntensity.value = params.envMapIntensity ?? 1;

    this._envMap = params.envMap ?? null;
    this.envTexNode = texture(this._envMap ?? FALLBACK_ENV_TEXTURE);
    this.uHasEnvMap.value = this._envMap ? 1 : 0;

    mat.colorNode = this.buildDiamondNode();
  }

  /** Active diamond facet cut. Updating replaces the mesh geometry and rebuilds the facet shader. */
  get cut(): DiamondCut {
    return this._cut;
  }
  set cut(v: DiamondCut) {
    if (v === this._cut) return;
    this._cut = v;
    this.geometry.dispose();
    this.geometry = diamondGeometry(v);
    this.diamondMaterial.colorNode = this.buildDiamondNode();
    this.diamondMaterial.needsUpdate = true;
  }

  /** Resting center height above `y = 0` for the active cut at unit scale. */
  get restHeight(): number {
    return DIAMOND_CUTS[this._cut].rest;
  }

  /** Tilt angle (radians around Z) to sit flush on a pavilion facet. */
  get restLean(): number {
    return DIAMOND_CUTS[this._cut].lean;
  }

  /** Position and tilt the diamond so it rests naturally on a horizontal surface at `y`. */
  restOnFloor(y = 0): this {
    this.rotation.z = -this.restLean;
    this.position.y = y + this.restHeight * this.scale.y;
    return this;
  }

  get color(): THREE.Color {
    return this.diamondMaterial.color;
  }

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

  get bounces(): number {
    return this.uBounces.value;
  }
  set bounces(v: number) {
    this.uBounces.value = v;
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
      this.diamondMaterial.transparent = isActive;
      this.diamondMaterial.colorNode = this.buildDiamondNode();
      this.diamondMaterial.needsUpdate = true;
    }
  }

  get envMap(): THREE.Texture | null {
    return this._envMap;
  }
  set envMap(tex: THREE.Texture | null) {
    this._envMap = tex;
    this.envTexNode.value = tex ?? FALLBACK_ENV_TEXTURE;
    this.uHasEnvMap.value = tex ? 1 : 0;
  }

  get envMapIntensity(): number {
    return this.uEnvMapIntensity.value;
  }
  set envMapIntensity(v: number) {
    this.uEnvMapIntensity.value = v;
  }

  /** Dispose the diamond's geometry and material resources. */
  dispose(): void {
    this.geometry.dispose();
    this.diamondMaterial.dispose();
  }

  private buildDiamondNode(): N {
    const uTint = uniform(this.diamondMaterial.color);
    const { env } = makeStudioEnv(this.uRoughness, {
      envTexNode: this.envTexNode,
      uHasEnvMap: this.uHasEnvMap,
      uEnvMapIntensity: this.uEnvMapIntensity,
    });
    const planes = DIAMOND_CUTS[this._cut].planes;
    const { declare, entry, exit } = shapeFunctions(KIND.diamond, { bevel: float(0) }, planes);

    const useTransmission = this._transmission > 0;
    const uIor = this.uIor;
    const uDispersion = this.uDispersion;
    const uBounces = this.uBounces;
    const uRoughness = this.uRoughness;
    const uFrostScale = this.uFrostScale;
    const uTransmission = this.uTransmission;

    const lookEnv = (rd: N, bendOffset: N) => {
      if (!useTransmission) return env(rd);
      const uv = clamp(screenUV.add(bendOffset), vec2(0.002), vec2(0.998));
      const bg = viewportOpaqueMipTexture(uv, uRoughness.mul(5)).rgb;
      return mix(env(rd), bg, uTransmission);
    };

    return Fn(() => {
      declare();
      const toLocalDir = (v: N) => normalize(modelWorldMatrixInverse.mul(vec4(v, 0)).xyz);
      const toWorldDir = (v: N) => normalize(modelWorldMatrix.mul(vec4(v, 0)).xyz);

      const d0World = normalize(positionWorld.sub(cameraPosition)).toVar();
      const d0 = toLocalDir(d0World).toVar();
      const size = vec3(1).toVar();

      const { pos: p0, nrm: n0 } = entry(positionLocal, size, normalize(normalLocal));
      If(dot(d0, n0).greaterThan(0), () => {
        n0.assign(n0.negate());
      });

      const frostGrain = float(0.5).toVar();
      perturbFrostNormal(n0, d0, p0, float(1), uRoughness, uFrostScale, frostGrain);

      const ior = max(uIor, 1.0001).toVar();
      const disp = uDispersion.div(12).toVar();
      const maxEvents = uBounces.toInt();
      const layers = select(
        disp.greaterThan(0).and(uRoughness.lessThan(0.35)),
        spectralSampleCount(disp, false),
        int(1),
      ).toVar();
      const spectralU = spectralJitter().toVar();
      const weightSum = vec3(0).toVar();
      const sigma = log(max(uTint, vec3(1e-4)))
        .negate()
        .div(2)
        .toVar();

      const col = vec3(0).toVar();
      const d0DotN0 = dot(d0, n0).toVar();
      const outerRefl = env(toWorldDir(reflect(d0, n0))).toVar();

      Loop({ start: int(0), end: layers, type: 'int', condition: '<', name: 'lc' }, ({ lc }: { lc: N }) => {
        const sample = spectralSample(lc, layers, spectralU);
        const bandOffset = select(layers.equal(1), float(0), sample.cauchy);
        const iorIn = max(ior.add(disp.mul(bandOffset)), 1.0001).toVar();
        const f0 = fresnel(d0DotN0, iorIn).toVar();
        const layer = f0.mul(outerRefl).toVar();
        const thr = float(1).sub(f0).toVar();
        const dir = refract(d0, n0, float(1).div(iorIn)).toVar();
        const pos = vec3(p0).toVar();
        const filtered = vec3(1).toVar();
        If(thr.greaterThan(0.012).and(dot(dir, dir).greaterThan(1e-6)), () => {
          Loop({ start: int(1), end: maxEvents, type: 'int', condition: '<', name: 'ev' }, () => {
            const hit = exit(pos, dir, size);
            pos.assign(hit.pos);
            const nrm = hit.nrm.toVar();
            filtered.mulAssign(exp(sigma.mul(hit.t.negate())));
            const rf = refract(dir, nrm.negate(), iorIn).toVar();
            const f = fresnel(dot(dir, nrm), float(1).div(iorIn)).toVar();
            If(dot(rf, rf).lessThan(1e-6), () => {
              f.assign(1);
            });
            const outW = thr.mul(float(1).sub(f)).toVar();
            If(outW.greaterThan(0.01), () => {
              const wDir = toWorldDir(rf);
              const bend = wDir.sub(d0World).xy.mul(0.12);
              layer.addAssign(filtered.mul(outW.mul(lookEnv(wDir, bend))));
            });
            thr.mulAssign(f);
            If(thr.lessThan(0.015), () => {
              Break();
            });
            dir.assign(reflect(dir, nrm.negate()));
          });
          If(thr.greaterThanEqual(0.015), () => {
            const wDir = toWorldDir(dir);
            const bend = wDir.sub(d0World).xy.mul(0.12);
            layer.addAssign(filtered.mul(thr.mul(lookEnv(wDir, bend))));
          });
        });
        const weight = select(layers.equal(1), vec3(1), sample.weight).toVar();
        weightSum.addAssign(weight);
        col.addAssign(layer.mul(weight));
      });
      col.divAssign(max(weightSum, vec3(1e-6)));

      applyFrostVeil(col, d0, n0, uTint, vec3(0.75), uRoughness, frostGrain);
      return vec4(col, 1);
    })();
  }
}
