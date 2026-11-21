// Standalone Three.js NodeMaterial for real-time physical glass (WebGPU & WebGL 2).
// Works on ANY THREE.Mesh / BufferGeometry out of the box, with optional exact
// multi-bounce analytic raytracing for primitives and all 6 diamond cuts.
/* eslint-disable @typescript-eslint/no-explicit-any -- TSL node types are too loose to spell out */
import * as THREE from 'three/webgpu';
import * as TSL from 'three/tsl';
import { DIAMOND_CUTS, type DiamondCut } from './diamond';
import { KIND } from './objects';
import { tones } from './tsl/finish';
import { CMY_FACE_ORDERS, cmyTint, fresnel, frostNoise3, shapeFunctions, SPECTRAL_COUNT, spectralCauchy, spectralWeight, type CmyFaceOrder } from './tsl/glass';

type N = any;
const {
  Break, Fn, If, Loop, abs, atan, cameraPosition, clamp, cos, dot, exp, float, int, log, max, mix,
  modelWorldMatrix, modelWorldMatrixInverse, normalLocal, normalWorld, normalize, positionLocal, positionWorld,
  reflect, refract, screenUV, select, sin, smoothstep, sqrt, texture, uniform, vec2, vec3, vec4,
} = TSL as N;

/** Analytic shape for exact multi-bounce internal raytracing, or `'mesh'` for any BufferGeometry. */
export type GlassShape =
  | 'mesh'
  | 'box'
  | 'sphere'
  | 'cylinder'
  | 'capsule'
  | 'cone'
  | 'torus'
  | 'octahedron'
  | 'prism'
  | 'hexprism'
  | 'diamond';

const SHAPE_KIND: Record<Exclude<GlassShape, 'mesh'>, number> = {
  box: KIND.box,
  sphere: KIND.sphere,
  cylinder: KIND.cylinder,
  capsule: KIND.capsule,
  cone: KIND.cone,
  torus: KIND.torus,
  octahedron: KIND.octahedron,
  prism: KIND.prism,
  hexprism: KIND.hexprism,
  diamond: KIND.diamond,
};

export interface GlassMaterialParameters {
  /** Index of refraction (1.0 – 2.5; default 1.52 for crown glass). */
  ior?: number;
  /** Chromatic dispersion strength (0 = none, 1 = strong fire; default 0.5). */
  dispersion?: number;
  /** Frosted glass roughness (0 = crystal clear, 1 = fully frosted; default 0). */
  roughness?: number;
  /** Frosted glass grain fineness (higher = smaller micro-particles; default 4). */
  frostScale?: number;
  /** Glass tint color (Beer-Lambert transmittance across thickness; default white). */
  color?: THREE.ColorRepresentation;
  /** Strength of glass tint (0 = clear, 1 = full color; default 1). */
  colorAlpha?: number;
  /** Characteristic object thickness in world units for `'mesh'` mode (default 1.0). */
  thickness?: number;
  /** Maximum internal reflection/refraction bounces for analytic shapes (default 6). */
  bounces?: number;
  /** Analytic shape for exact internal raytracing, or `'mesh'` for any BufferGeometry (default `'mesh'`). */
  shape?: GlassShape;
  /** Local-space size parameters for analytic shapes (default [1, 1, 1]). */
  shapeSize?: [number, number, number] | THREE.Vector3;
  /** Diamond cut when `shape === 'diamond'` (default `'round'`). */
  diamondCut?: DiamondCut;
  /** Edge bevel in local units when `shape === 'box'` (default 0). */
  bevel?: number;
  /** CMY dichroic cube filter strength when `shape === 'box'` (0 = off; default 0). */
  cmy?: number;
  /** CMY face color permutation for [X, Y, Z] faces when `cmy > 0` (default `'cym'`). */
  cmyOrder?: CmyFaceOrder;
  /** Use dark studio environment instead of bright photo studio (default false). */
  dark?: boolean;
  /** Include a procedural ground plane in reflections/refractions (default true). */
  ground?: boolean;
  /** Ground plane albedo color (default #eeeeee). */
  groundColor?: THREE.ColorRepresentation;
  /** Direction toward the main studio lamp (default [-0.6, 0.7, 0.4]). */
  lampDirection?: THREE.Vector3 | [number, number, number];
  /** Exposure multiplier (default 1.0). */
  exposure?: number;
  /** Apply Khronos neutral filmic tone mapping (default true). */
  filmic?: boolean;
  /** Optional screen-space background texture to refract through the glass. */
  backgroundTexture?: THREE.Texture | null;
  /** Optional blurred background texture for frosted transmission. */
  backgroundBlurTexture?: THREE.Texture | null;
}

/** Procedural studio environment (matching scene.ts) for standalone GlassMaterial. */
function makeStudioEnv(
  uDark: N,
  uGround: N,
  uGroundColor: N,
  uLampDir: N,
  uRoughness: N,
) {
  const env = Fn(([d]: N[]) => {
    const r = uRoughness.toVar();
    const r2 = r.mul(r).toVar();
    const sharp = float(1).sub(r.mul(0.85)).toVar();
    const az = atan(d.z, d.x).toVar();
    const lamp = normalize(uLampDir).toVar();
    const out = vec3(0).toVar();
    If(uDark.greaterThan(0.5), () => {
      const k = mix(0.11, 0.035, smoothstep(0, 0.6, d.y)).toVar();
      const spotGlow = smoothstep(float(0.988).sub(r.mul(0.55)), float(0.996).add(r.mul(0.004)), dot(d, lamp));
      k.addAssign(float(40).div(r2.mul(28).add(1)).mul(spotGlow));
      const ring = smoothstep(float(0.78).sub(r.mul(0.35)), float(0.86).add(r.mul(0.1)), d.y).mul(
        float(1).sub(smoothstep(0.94, 0.97, d.y).mul(sharp)),
      );
      k.addAssign(ring.mul(2.6).div(r2.mul(3).add(1)).mul(mix(smoothstep(-0.2, 0.3, cos(az.mul(4))), 0.5, r)));
      const tall = smoothstep(0.05, 0.15, d.y).mul(float(1).sub(smoothstep(0.55, 0.7, d.y))).toVar();
      const stripLo = float(0.95).sub(r.mul(0.55));
      const stripHi = float(0.985).add(r.mul(0.015));
      const stripGain = float(4).div(r2.mul(5).add(1));
      k.addAssign(tall.mul(stripGain).mul(smoothstep(stripLo, stripHi, cos(az.sub(2.5)))));
      k.addAssign(tall.mul(stripGain).mul(smoothstep(stripLo, stripHi, cos(az.add(0.63)))));
      k.addAssign(r.mul(0.12));
      out.assign(vec3(k));
    }).Else(() => {
      const c = mix(mix(0.15, 0.38, r), mix(0.72, 0.65, r), smoothstep(-0.1, 0.95, d.y)).toVar();
      const band = smoothstep(0.03, 0.1, d.y).mul(float(1).sub(smoothstep(0.68, 0.82, d.y)));
      c.assign(mix(c, 0.04, band.mul(0.96).mul(sharp).mul(smoothstep(-0.1, 0.15, sin(az.mul(5).add(1))))));
      c.assign(mix(c, 0.05, smoothstep(0.9, 0.95, d.y).mul(sharp)));
      const mainLo = float(0.86).sub(r.mul(0.55));
      const mainHi = float(0.96).add(r.mul(0.04));
      c.addAssign(float(3).div(r2.mul(3).add(1)).mul(smoothstep(mainLo, mainHi, dot(d, lamp))));
      const smallLo = float(0.985).sub(r.mul(0.55));
      const smallGain = float(3).div(r2.mul(8).add(1));
      c.addAssign(smoothstep(smallLo, 0.995, dot(d, normalize(vec3(Math.cos(3), 0.6, Math.sin(3))))).mul(smallGain));
      c.addAssign(smoothstep(smallLo, 0.995, dot(d, normalize(vec3(Math.cos(5), 0.9, Math.sin(5))))).mul(smallGain));
      c.addAssign(r.mul(0.08));
      out.assign(vec3(c));
    });
    return out;
  }).setLayout({ name: 'gmEnv', type: 'vec3', inputs: [{ name: 'd', type: 'vec3' }] });

  const sampleScene = Fn(([ro, rd]: N[]) => {
    const out = vec3(0).toVar();
    If(uGround.greaterThan(0.5).and(rd.y.lessThan(-1e-5)), () => {
      const t = max(ro.y.negate().div(rd.y), 0).toVar();
      const horizon = env(normalize(vec3(rd.x, 0.03, rd.z)));
      const skyLit = select(uDark.greaterThan(0.5), vec3(0.05), vec3(0.52));
      const sunLit = select(uDark.greaterThan(0.5), vec3(0.28), vec3(0.42));
      const floorCol = uGroundColor.mul(skyLit.add(sunLit));
      out.assign(mix(floorCol, horizon, smoothstep(60, 240, t)));
    }).Else(() => {
      out.assign(env(rd));
    });
    return out;
  }).setLayout({ name: 'gmScene', type: 'vec3', inputs: [{ name: 'ro', type: 'vec3' }, { name: 'rd', type: 'vec3' }] });

  return { env, sampleScene };
}

/**
 * Standalone physical glass material for Three.js WebGPU & WebGL 2.
 *
 * Attach directly to any `THREE.Mesh`:
 * ```ts
 * const glass = new GlassMaterial({ ior: 1.52, dispersion: 0.6, roughness: 0.15 });
 * const mesh = new THREE.Mesh(new THREE.TorusKnotGeometry(1, 0.35, 160, 32), glass);
 * scene.add(mesh);
 * ```
 * Or specify an analytic `shape` (e.g. `shape: 'diamond', diamondCut: 'princess'`)
 * for exact multi-bounce internal raytracing.
 */
export class GlassMaterial extends THREE.MeshBasicNodeMaterial {
  readonly shape: GlassShape;
  private _diamondCut: DiamondCut;
  private _cmyOrder: CmyFaceOrder;

  private readonly uIor = uniform(1.52);
  private readonly uDispersion = uniform(0.5);
  private readonly uRoughness = uniform(0);
  private readonly uFrostScale = uniform(4);
  private readonly uColorAlpha = uniform(1);
  private readonly uThickness = uniform(1);
  private readonly uBounces = uniform(6);
  private readonly uBevel = uniform(0);
  private readonly uCmy = uniform(0);
  private readonly uCmyOrder = uniform(new THREE.Vector3(0, 1, 2));
  private readonly uDark = uniform(0);
  private readonly uGround = uniform(1);
  private readonly uExposure = uniform(1);
  private readonly uFilmic = uniform(1);
  private readonly uGroundColor: N;
  private readonly uLampDir: N;
  private readonly uShapeSize: N;
  private readonly uHasBg = uniform(0);
  private readonly uHasBgBlur = uniform(0);
  private readonly bgTexNode: N;
  private readonly bgBlurTexNode: N;

  constructor(params: GlassMaterialParameters = {}) {
    super();

    this.shape = params.shape ?? 'mesh';
    this._diamondCut = params.diamondCut ?? 'round';
    this._cmyOrder = params.cmyOrder ?? 'cym';
    this.uCmyOrder.value.set(...CMY_FACE_ORDERS[this._cmyOrder]);

    if (params.color !== undefined) this.color.set(params.color);
    this.uIor.value = params.ior ?? 1.52;
    this.uDispersion.value = params.dispersion ?? 0.5;
    this.uRoughness.value = params.roughness ?? 0;
    this.uFrostScale.value = params.frostScale ?? 4;
    this.uColorAlpha.value = params.colorAlpha ?? 1;
    this.uThickness.value = params.thickness ?? 1;
    this.uBounces.value = params.bounces ?? 6;
    this.uBevel.value = params.bevel ?? 0;
    this.uCmy.value = params.cmy ?? 0;
    this.uDark.value = params.dark ? 1 : 0;
    this.uGround.value = (params.ground ?? true) ? 1 : 0;
    this.uExposure.value = params.exposure ?? 1;
    this.uFilmic.value = (params.filmic ?? true) ? 1 : 0;

    const gCol = new THREE.Color(params.groundColor ?? 0xeeeeee);
    this.uGroundColor = uniform(gCol);

    const lamp = Array.isArray(params.lampDirection)
      ? new THREE.Vector3(...params.lampDirection)
      : params.lampDirection
        ? params.lampDirection.clone()
        : new THREE.Vector3(-0.6, 0.7, 0.4);
    lamp.normalize();
    this.uLampDir = uniform(lamp);

    const sz = Array.isArray(params.shapeSize)
      ? new THREE.Vector3(...params.shapeSize)
      : params.shapeSize
        ? params.shapeSize.clone()
        : new THREE.Vector3(1, 1, 1);
    this.uShapeSize = uniform(sz);

    // 1x1 fallback white texture so texture nodes always bind cleanly even without backgroundTexture.
    const fallbackTex = new THREE.DataTexture(new Uint8Array([255, 255, 255, 255]), 1, 1);
    fallbackTex.needsUpdate = true;
    this.bgTexNode = texture(params.backgroundTexture ?? fallbackTex);
    this.bgBlurTexNode = texture(params.backgroundBlurTexture ?? params.backgroundTexture ?? fallbackTex);
    this.uHasBg.value = params.backgroundTexture ? 1 : 0;
    this.uHasBgBlur.value = params.backgroundBlurTexture ? 1 : 0;

    this.colorNode = this.buildColorNode();
  }

  // ---- Reactive optical properties (live GPU uniform updates, no recompile) ----

  get diamondCut(): DiamondCut {
    return this._diamondCut;
  }
  set diamondCut(v: DiamondCut) {
    if (v === this._diamondCut) return;
    this._diamondCut = v;
    if (this.shape === 'diamond') {
      this.colorNode = this.buildColorNode();
      this.needsUpdate = true;
    }
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

  get colorAlpha(): number {
    return this.uColorAlpha.value;
  }
  set colorAlpha(v: number) {
    this.uColorAlpha.value = v;
  }

  get thickness(): number {
    return this.uThickness.value;
  }
  set thickness(v: number) {
    this.uThickness.value = v;
  }

  get bounces(): number {
    return this.uBounces.value;
  }
  set bounces(v: number) {
    this.uBounces.value = v;
  }

  get bevel(): number {
    return this.uBevel.value;
  }
  set bevel(v: number) {
    this.uBevel.value = v;
  }

  get cmy(): number {
    return this.uCmy.value;
  }
  set cmy(v: number) {
    this.uCmy.value = v;
  }

  get cmyOrder(): CmyFaceOrder {
    return this._cmyOrder;
  }
  set cmyOrder(v: CmyFaceOrder) {
    this._cmyOrder = v;
    this.uCmyOrder.value.set(...CMY_FACE_ORDERS[v]);
  }

  get dark(): boolean {
    return this.uDark.value > 0.5;
  }
  set dark(v: boolean) {
    this.uDark.value = v ? 1 : 0;
  }

  get ground(): boolean {
    return this.uGround.value > 0.5;
  }
  set ground(v: boolean) {
    this.uGround.value = v ? 1 : 0;
  }

  get groundColor(): THREE.Color {
    return this.uGroundColor.value;
  }
  set groundColor(v: THREE.Color) {
    this.uGroundColor.value.copy(v);
  }

  get lampDirection(): THREE.Vector3 {
    return this.uLampDir.value;
  }
  set lampDirection(v: THREE.Vector3) {
    this.uLampDir.value.copy(v).normalize();
  }

  get exposure(): number {
    return this.uExposure.value;
  }
  set exposure(v: number) {
    this.uExposure.value = v;
  }

  get filmic(): boolean {
    return this.uFilmic.value > 0.5;
  }
  set filmic(v: boolean) {
    this.uFilmic.value = v ? 1 : 0;
  }

  get shapeSize(): THREE.Vector3 {
    return this.uShapeSize.value;
  }
  set shapeSize(v: THREE.Vector3) {
    this.uShapeSize.value.copy(v);
  }

  /** Attach or update a screen-space background texture (and optional blurred copy for frost). */
  setBackground(sharp: THREE.Texture | null, blurred: THREE.Texture | null = null): void {
    if (sharp) {
      this.bgTexNode.value = sharp;
      this.uHasBg.value = 1;
    } else {
      this.uHasBg.value = 0;
    }
    if (blurred) {
      this.bgBlurTexNode.value = blurred;
      this.uHasBgBlur.value = 1;
    } else if (sharp) {
      this.bgBlurTexNode.value = sharp;
      this.uHasBgBlur.value = 0;
    } else {
      this.uHasBgBlur.value = 0;
    }
  }

  private buildColorNode(): N {
    const uTint = uniform(this.color);
    const { sampleScene } = makeStudioEnv(this.uDark, this.uGround, this.uGroundColor, this.uLampDir, this.uRoughness);

    const lookEnv = Fn(([ro, rd, bendOffset]: N[]) => {
      const out = sampleScene(ro, rd).toVar();
      If(this.uHasBg.greaterThan(0.5), () => {
        const uv = clamp(screenUV.add(bendOffset), vec2(0.002), vec2(0.998)).toVar();
        const sharp = this.bgTexNode.sample(uv).level(0).rgb;
        out.assign(sharp);
        If(this.uRoughness.greaterThan(0.001).and(this.uHasBgBlur.greaterThan(0.5)), () => {
          const blurMix = clamp(sqrt(this.uRoughness), 0, 1);
          out.assign(mix(sharp, this.bgBlurTexNode.sample(uv).level(0).rgb, blurMix));
        });
      });
      return out;
    }).setLayout({
      name: 'gmLookEnv',
      type: 'vec3',
      inputs: [{ name: 'ro', type: 'vec3' }, { name: 'rd', type: 'vec3' }, { name: 'bendOffset', type: 'vec2' }],
    });

    if (this.shape === 'mesh') {
      return this.buildMeshNode(uTint, sampleScene, lookEnv);
    }
    return this.buildAnalyticNode(uTint, sampleScene, lookEnv);
  }

  /** Arbitrary BufferGeometry mode: two-interface dielectric refraction, TIR, Beer-Lambert, dispersion & frost. */
  private buildMeshNode(uTint: N, sampleScene: N, lookEnv: N): N {
    const uIor = this.uIor;
    const uDispersion = this.uDispersion;
    const uRoughness = this.uRoughness;
    const uFrostScale = this.uFrostScale;
    const uColorAlpha = this.uColorAlpha;
    const uThickness = this.uThickness;
    const uDark = this.uDark;
    const uExposure = this.uExposure;
    const uFilmic = this.uFilmic;

    return Fn(() => {
      const d0 = normalize(positionWorld.sub(cameraPosition)).toVar();
      const n0 = normalize(normalWorld).toVar();
      If(dot(d0, n0).greaterThan(0), () => {
        n0.assign(n0.negate());
      });

      const reach = max(uThickness, 0.05).toVar();
      const frostGrain = float(0.5).toVar();
      If(uRoughness.greaterThan(0.001), () => {
        const fs = max(uFrostScale, 0.25).toVar();
        const fp = positionLocal.div(reach).mul(fs).toVar();
        const fn0 = frostNoise3(fp).toVar();
        frostGrain.assign(fn0);
        const eps = float(0.008).toVar();
        const gx = frostNoise3(fp.add(vec3(eps, 0, 0))).sub(fn0);
        const gy = frostNoise3(fp.add(vec3(0, eps, 0))).sub(fn0);
        const gz = frostNoise3(fp.add(vec3(0, 0, eps))).sub(fn0);
        const perturbed = normalize(n0.add(vec3(gx, gy, gz).mul(uRoughness.mul(0.45)))).toVar();
        If(dot(d0, perturbed).lessThan(-0.02), () => {
          n0.assign(perturbed);
        });
      });

      const ior = max(uIor, 1.0001).toVar();
      const disp = uDispersion.div(12).toVar();
      const layers = select(disp.greaterThan(0), int(SPECTRAL_COUNT), int(1)).toVar();
      const baseTint = mix(vec3(1), uTint, uColorAlpha).toVar();
      const sigma = log(max(baseTint, vec3(1e-4))).negate().div(max(reach, 1e-4)).toVar();
      const col = vec3(0).toVar();

      const d0DotN0 = dot(d0, n0).toVar();
      const outerRefl = sampleScene(positionWorld.add(n0.mul(0.003)), reflect(d0, n0)).toVar();

      Loop({ start: int(0), end: layers, type: 'int', condition: '<', name: 'lc' }, ({ lc }: { lc: N }) => {
        const bandOffset = select(layers.equal(1), float(0), spectralCauchy(lc));
        const iorIn = max(ior.add(disp.mul(bandOffset)), 1.0001).toVar();
        const f0 = fresnel(d0DotN0, iorIn).toVar();
        const layer = f0.mul(outerRefl).toVar();
        const thr = float(1).sub(f0).toVar();
        const t1 = refract(d0, n0, float(1).div(iorIn)).toVar();
        If(dot(t1, t1).greaterThan(1e-6), () => {
          // Estimate internal chord length & curved back-face exit normal for arbitrary meshes.
          const dist = reach.mul(mix(0.35, 1, abs(dot(t1, n0)))).toVar();
          const pExit = positionWorld.add(t1.mul(dist)).toVar();
          const nExit = normalize(mix(t1, n0.negate(), 0.55)).toVar();
          const absorb = exp(sigma.mul(dist.negate())).toVar();
          const t2 = refract(t1, nExit.negate(), iorIn).toVar();
          const f1 = fresnel(dot(t1, nExit), float(1).div(iorIn)).toVar();
          If(dot(t2, t2).lessThan(1e-6), () => {
            f1.assign(1);
          });
          const outW = thr.mul(float(1).sub(f1)).toVar();
          If(outW.greaterThan(0.004), () => {
            const bend = t2.sub(d0).xy.mul(0.12);
            layer.addAssign(absorb.mul(outW).mul(lookEnv(pExit, t2, bend)));
          });
          const bounceW = thr.mul(f1).toVar();
          If(bounceW.greaterThan(0.004), () => {
            const rInt = reflect(t1, nExit.negate());
            const bend = rInt.sub(d0).xy.mul(0.12);
            layer.addAssign(absorb.mul(absorb).mul(bounceW).mul(lookEnv(pExit, rInt, bend)));
          });
        });
        const weight = select(layers.equal(1), vec3(1), spectralWeight(lc));
        col.addAssign(layer.mul(weight));
      });

      // Creamy frosted veil + crisp Fresnel rim when roughness > 0.
      If(uRoughness.greaterThan(0.001), () => {
        const veilOpacity = uRoughness.mul(0.42).mul(frostGrain.mul(0.24).add(0.88)).toVar();
        const studioLight = select(uDark.greaterThan(0.5), vec3(0.22), vec3(0.75)).toVar();
        const frostTint = baseTint.mul(studioLight).toVar();
        const lifted = col.mul(1.03).add(frostTint.mul(0.22));
        const frosted = mix(mix(col, lifted, veilOpacity), frostTint, veilOpacity.mul(0.55));
        const ndotv = clamp(dot(d0, n0).negate(), 0, 1);
        const rim = float(1).sub(ndotv).toVar();
        const rimGlow = rim.mul(rim).mul(rim).mul(uRoughness.mul(0.35)).mul(studioLight);
        col.assign(frosted.add(rimGlow));
      });

      col.mulAssign(uExposure);
      If(uFilmic.greaterThan(0.5), () => {
        col.assign(tones(col));
      });
      return vec4(col, 1);
    })();
  }

  /** Analytic shape mode: exact multi-bounce internal raytracing in the mesh's local space. */
  private buildAnalyticNode(uTint: N, sampleScene: N, lookEnv: N): N {
    const kind = SHAPE_KIND[this.shape as Exclude<GlassShape, 'mesh'>];
    const diamondPlanes = kind === KIND.diamond ? DIAMOND_CUTS[this.diamondCut].planes : undefined;
    const { declare, entry, exit } = shapeFunctions(kind, { bevel: this.uBevel }, diamondPlanes);
    const cmyBox = kind === KIND.box;

    const uIor = this.uIor;
    const uDispersion = this.uDispersion;
    const uRoughness = this.uRoughness;
    const uFrostScale = this.uFrostScale;
    const uColorAlpha = this.uColorAlpha;
    const uBounces = this.uBounces;
    const uCmy = this.uCmy;
    const uDark = this.uDark;
    const uExposure = this.uExposure;
    const uFilmic = this.uFilmic;
    const uShapeSize = this.uShapeSize;

    return Fn(() => {
      declare();
      const toLocalDir = (v: N) => normalize(modelWorldMatrixInverse.mul(vec4(v, 0)).xyz);
      const toWorldPos = (p: N) => modelWorldMatrix.mul(vec4(p, 1)).xyz;
      const toWorldDir = (v: N) => normalize(modelWorldMatrix.mul(vec4(v, 0)).xyz);

      const d0World = normalize(positionWorld.sub(cameraPosition)).toVar();
      const d0 = toLocalDir(d0World).toVar();
      const size = uShapeSize.toVar();
      const reach = max(size.x, max(size.y, size.z)).toVar();

      const { pos: p0, nrm: n0 } = entry(positionLocal, size, normalize(normalLocal));
      If(dot(d0, n0).greaterThan(0), () => {
        n0.assign(n0.negate());
      });

      const frostGrain = float(0.5).toVar();
      If(uRoughness.greaterThan(0.001), () => {
        const fs = max(uFrostScale, 0.25).toVar();
        const fp = p0.div(max(reach, 0.05)).mul(fs).toVar();
        const fn0 = frostNoise3(fp).toVar();
        frostGrain.assign(fn0);
        const eps = float(0.008).toVar();
        const gx = frostNoise3(fp.add(vec3(eps, 0, 0))).sub(fn0);
        const gy = frostNoise3(fp.add(vec3(0, eps, 0))).sub(fn0);
        const gz = frostNoise3(fp.add(vec3(0, 0, eps))).sub(fn0);
        const perturbed = normalize(n0.add(vec3(gx, gy, gz).mul(uRoughness.mul(0.45)))).toVar();
        If(dot(d0, perturbed).lessThan(-0.02), () => {
          n0.assign(perturbed);
        });
      });

      const ior = max(uIor, 1.0001).toVar();
      const disp = uDispersion.div(12).toVar();
      const maxEvents = uBounces.toInt();
      const layers = select(disp.greaterThan(0), int(SPECTRAL_COUNT), int(1)).toVar();
      const baseTint = mix(vec3(1), uTint, uColorAlpha).toVar();
      const sigma = log(max(baseTint, vec3(1e-4))).negate().div(max(reach.mul(2), 1e-4)).toVar();
      const entryFace = vec3(1).toVar();
      if (cmyBox)
        If(uCmy.greaterThan(0), () => {
          entryFace.assign(cmyTint(n0, uCmy, this.uCmyOrder));
        });

      const col = vec3(0).toVar();
      const d0DotN0 = dot(d0, n0).toVar();
      const outerRefl = sampleScene(positionWorld.add(toWorldDir(n0).mul(0.003)), toWorldDir(reflect(d0, n0))).toVar();
      Loop({ start: int(0), end: layers, type: 'int', condition: '<', name: 'lc' }, ({ lc }: { lc: N }) => {
        const bandOffset = select(layers.equal(1), float(0), spectralCauchy(lc));
        const iorIn = max(ior.add(disp.mul(bandOffset)), 1.0001).toVar();
        const f0 = fresnel(d0DotN0, iorIn).toVar();
        const layer = f0.mul(outerRefl).toVar();
        const thr = float(1).sub(f0).toVar();
        const dir = refract(d0, n0, float(1).div(iorIn)).toVar();
        const pos = vec3(p0).toVar();
        const filtered = vec3(entryFace).toVar();
        If(thr.greaterThan(0.004).and(dot(dir, dir).greaterThan(1e-6)), () => {
          Loop({ start: int(1), end: maxEvents, type: 'int', condition: '<', name: 'ev' }, () => {
            const hit = exit(pos, dir, size);
            pos.assign(hit.pos);
            const nrm = hit.nrm.toVar();
            filtered.mulAssign(exp(sigma.mul(hit.t.negate())));
            const face = vec3(1).toVar();
            if (cmyBox)
              If(uCmy.greaterThan(0), () => {
                face.assign(cmyTint(nrm, uCmy, this.uCmyOrder));
              });
            const rf = refract(dir, nrm.negate(), iorIn).toVar();
            const f = fresnel(dot(dir, nrm), float(1).div(iorIn)).toVar();
            If(dot(rf, rf).lessThan(1e-6), () => {
              f.assign(1);
            });
            const outW = thr.mul(float(1).sub(f)).toVar();
            If(outW.greaterThan(0.004), () => {
              const wDir = toWorldDir(rf);
              const bend = wDir.sub(d0World).xy.mul(0.12);
              const seen = lookEnv(toWorldPos(pos.add(nrm.mul(0.003))), wDir, bend);
              layer.addAssign(filtered.mul(face).mul(outW.mul(seen)));
            });
            thr.mulAssign(f);
            If(thr.lessThan(0.004), () => {
              Break();
            });
            dir.assign(reflect(dir, nrm.negate()));
            filtered.mulAssign(face);
          });
          If(thr.greaterThanEqual(0.004), () => {
            const wDir = toWorldDir(dir);
            const bend = wDir.sub(d0World).xy.mul(0.12);
            layer.addAssign(filtered.mul(thr.mul(lookEnv(toWorldPos(pos), wDir, bend))));
          });
        });
        const weight = select(layers.equal(1), vec3(1), spectralWeight(lc));
        col.addAssign(layer.mul(weight));
      });

      If(uRoughness.greaterThan(0.001), () => {
        const veilOpacity = uRoughness.mul(0.42).mul(frostGrain.mul(0.24).add(0.88)).toVar();
        const studioLight = select(uDark.greaterThan(0.5), vec3(0.22), vec3(0.75)).toVar();
        const frostTint = baseTint.mul(studioLight).toVar();
        const lifted = col.mul(1.03).add(frostTint.mul(0.22));
        const frosted = mix(mix(col, lifted, veilOpacity), frostTint, veilOpacity.mul(0.55));
        const ndotv = clamp(dot(d0, n0).negate(), 0, 1);
        const rim = float(1).sub(ndotv).toVar();
        const rimGlow = rim.mul(rim).mul(rim).mul(uRoughness.mul(0.35)).mul(studioLight);
        col.assign(frosted.add(rimGlow));
      });

      col.mulAssign(uExposure);
      If(uFilmic.greaterThan(0.5), () => {
        col.assign(tones(col));
      });
      return vec4(col, 1);
    })();
  }
}
