// Standalone, geometry-agnostic Three.js NodeMaterial for real-time physical glass
// (WebGPU & WebGL 2). Matches MeshPhysicalMaterial ergonomics: attach to any THREE.Mesh,
// optionally set `envMap` for HDRI lighting and `transmission` (0–1) for automatic
// mipmapped scene refraction via Three.js `viewportOpaqueMipTexture`.
//
// It shades with the same glass tracer as the instanced renderer (tsl/trace.ts): every
// internal event with exact Fresnel and total internal reflection, continuous spectral
// dispersion and Beer-Lambert absorption. Where each internal ray leaves the glass comes
// from the best source it has: a baked signed-distance `volume` (exact, every bounce), a
// `backfaceMap` from GlassBackfacePass (exact first exit), or else a smooth local-sphere
// estimate from `thickness`.
/* eslint-disable @typescript-eslint/no-explicit-any -- TSL node types are too loose to spell out */
import * as THREE from 'three/webgpu';
import * as TSL from 'three/tsl';
import { applyFrostVeil, perturbFrostNormal, perturbImperfections } from './tsl/frost';
import { spectralSampleCount } from './tsl/optics';
import { createSdfVolume } from './tsl/sdf';
import { makeStudioEnv } from './tsl/studioEnv';
import { traceGlass, type GlassHit } from './tsl/trace';

type N = any;
const {
  Fn,
  If,
  batchColor,
  buffer,
  cameraPosition,
  cameraProjectionMatrix,
  cameraViewMatrix,
  clamp,
  diffuseColor,
  dot,
  float,
  instanceColor,
  instancedBufferAttribute,
  instancedDynamicBufferAttribute,
  instanceIndex,
  int,
  inverse,
  length,
  log,
  mat4,
  materialOpacity,
  max,
  mix,
  modelWorldMatrix,
  modelWorldMatrixInverse,
  normalize,
  normalWorld,
  positionLocal,
  positionWorld,
  referenceAttribute,
  reflect,
  screenUV,
  select,
  texture,
  texture3D,
  uniform,
  vec2,
  vec3,
  vec4,
  vertexColor,
  vertexStage,
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
  /**
   * Optional baked signed-distance volume of the mesh (see `GlassVolume`): exact exits for
   * every internal bounce, as the instanced renderer uses for Suzanne, the dragon and the
   * torus knot. Takes precedence over `backfaceMap`.
   */
  volume?: GlassVolume | null;
  /** Internal events followed per ray (default 6, or 4 with a `volume`; 1 exit + internal reflections). */
  bounces?: number;
  /** Faint surface waviness (0: perfect glass, the default; 1: clearly hand-made). */
  imperfections?: number;
}

/**
 * A baked 128^3 signed-distance volume (HalfFloat, red channel), as `bakeGlassVolume`
 * (src/glass/bake.ts) and the scripts in `scripts/` write it: distances in volume units,
 * negative inside, covering +-1.1 volume units. Volume units are the mesh's local coordinates
 * moved by -`center` and divided by `halfWidth`. Several volumes can be stacked along z in one
 * texture (`layer` of `layerCount`). `createGlassVolume` / `bakeGlassVolumeAsync`
 * (src/glass/volume.ts) bake one for any geometry. Works on `Mesh` and `InstancedMesh` (each
 * instance's matrix is followed) with uniform scale.
 */
export interface GlassVolume {
  texture: THREE.Data3DTexture;
  /** Which stacked 128^3 layer holds this mesh (default 0). */
  layer?: number;
  /** How many layers are stacked in `texture` (default: its depth / 128). */
  layerCount?: number;
  /** Mesh-local point at the volume's centre (default the origin). */
  center?: [number, number, number];
  /** Mesh-local units per volume unit (default 1: the mesh is already unit size). */
  halfWidth?: number;
}

/** Vertex buffers a render pipeline may bind (WebGPU's guaranteed minimum). */
const MAX_VERTEX_BUFFERS = 8;

/**
 * This instance's matrix (geometry space to the object's space) in the fragment stage, read
 * the way three.js reads it for instancing (r187 `Instance.js`): bound per object when the
 * vertex buffers allow, else from a uniform buffer or the instance attribute directly. Passed
 * from the vertex stage as its four columns (WGSL cannot pass matrices between stages).
 */
function instanceMatrixNode(builder: N, mesh: THREE.InstancedMesh): N {
  const matrices = mesh.instanceMatrix;
  const count = Math.max(matrices.count, 1);
  const instanceBuffers = 1 + (mesh.instanceColor !== null ? 1 : 0) + (builder.needsPreviousData?.() ? 1 : 0);
  let matrix: N;
  if (
    builder.renderer?.isRenderer === true &&
    Object.keys(mesh.geometry.attributes).length + instanceBuffers <= MAX_VERTEX_BUFFERS
  ) {
    matrix = referenceAttribute('instanceMatrix', 'mat4');
  } else if (count * 16 * 4 <= builder.getUniformBufferLimit()) {
    matrix = buffer(matrices.array, 'mat4', count).element(instanceIndex);
  } else {
    const attributeFn =
      matrices.usage === THREE.DynamicDrawUsage ? instancedDynamicBufferAttribute : instancedBufferAttribute;
    matrix = mat4(
      ...[0, 4, 8, 12].map((offset) =>
        attributeFn(new THREE.InterleavedBufferAttribute(matrices as unknown as THREE.InterleavedBuffer, 4, offset)),
      ),
    );
  }
  const column = (k: number) =>
    vertexStage(matrix.mul(vec4(k === 0 ? 1 : 0, k === 1 ? 1 : 0, k === 2 ? 1 : 0, k === 3 ? 1 : 0)));
  return mat4(column(0), column(1), column(2), column(3));
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
  private _volume: GlassVolume | null = null;

  private readonly uIor = uniform(1.52);
  private readonly uDispersion = uniform(0.5);
  private readonly uRoughness = uniform(0);
  private readonly uFrostScale = uniform(4);
  private readonly uThickness = uniform(1);
  private readonly uTransmission = uniform(0);
  private readonly uHasEnvMap = uniform(0);
  private readonly uEnvMapIntensity = uniform(1);
  private readonly uHasBackface = uniform(0);
  private readonly uBounces = uniform(6);
  private readonly uImperfections = uniform(0);
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

    this._volume = params.volume ?? null;
    // Baked-volume meshes default to 4 events (as in the instanced renderer): their folded
    // surfaces scatter deeper bounces chaotically from pixel to pixel.
    this.uBounces.value = params.bounces ?? (this._volume ? 4 : 6);
    this.uImperfections.value = params.imperfections ?? 0;

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

  /** Internal events followed per ray (1 exit + internal reflections). */
  get bounces(): number {
    return this.uBounces.value;
  }
  set bounces(v: number) {
    this.uBounces.value = Math.max(1, Math.round(v));
  }

  /** Faint surface waviness (0: perfect glass; 1: clearly hand-made). */
  get imperfections(): number {
    return this.uImperfections.value;
  }
  set imperfections(v: number) {
    this.uImperfections.value = Math.min(Math.max(v, 0), 1);
  }

  /** Baked signed-distance volume for exact exits (changing it rebuilds the shader). */
  get volume(): GlassVolume | null {
    return this._volume;
  }
  set volume(v: GlassVolume | null) {
    this._volume = v;
    if (this.uBounces) {
      this.colorNode = this.buildColorNode();
      this.needsUpdate = true;
    }
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
      volume: this._volume,
      bounces: this.bounces,
      imperfections: this.imperfections,
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
    // Exits that live in the geometry's own space (a baked volume) must follow each instance's
    // matrix as well as the object's.
    const instanced = Boolean(object?.isInstancedMesh && object?.instanceMatrix) && this._volume !== null;
    if (!hasVertexColor && !hasInstanceColor && !hasBatchColor && !instanced) {
      super.setupDiffuseColor(builder);
      return;
    }
    let tint: N = uniform(this.color);
    if (hasVertexColor) tint = tint.mul(vertexColor().rgb);
    if (hasInstanceColor) tint = tint.mul(instanceColor);
    if (hasBatchColor) tint = tint.mul(batchColor);
    diffuseColor.assign(this.buildColorNode(tint, instanced ? instanceMatrixNode(builder, object) : null));
    const opacityNode = this.opacityNode ? float(this.opacityNode) : materialOpacity;
    diffuseColor.a.assign(diffuseColor.a.mul(opacityNode));
  }

  /**
   * The glass shader: every internal event through the shared tracer (tsl/trace.ts), with
   * exits from the baked volume, the back-face map or the local-sphere estimate, and what
   * leaving rays see from the environment and (with transmission) the scene behind.
   */
  private buildColorNode(tintNode: N = uniform(this.color), instanceMatrix: N = null): N {
    const uTint = tintNode;
    const { env } = makeStudioEnv(this.uRoughness, {
      envTexNode: this.envTexNode,
      uHasEnvMap: this.uHasEnvMap,
      uEnvMapIntensity: this.uEnvMapIntensity,
    });

    const useTransmission = this._transmission > 0;
    const uRoughness = this.uRoughness;
    const uThickness = this.uThickness;
    const uTransmission = this.uTransmission;
    const volume = this._volume;
    const viewProj = cameraProjectionMatrix.mul(cameraViewMatrix);

    return Fn(() => {
      const d0 = normalize(positionWorld.sub(cameraPosition)).toVar();
      const n0 = normalize(normalWorld).toVar();
      If(dot(d0, n0).greaterThan(0), () => {
        n0.assign(n0.negate());
      });

      // Object scale: the characteristic thickness, and the radius of the local-sphere model.
      const reach = max(uThickness, 0.05).toVar();
      const sphereRadius = reach.mul(0.5).toVar();
      const frostGrain = float(0.5).toVar();
      perturbFrostNormal(n0, d0, positionLocal, reach, uRoughness, this.uFrostScale, frostGrain);
      perturbImperfections(n0, positionLocal, reach, this.uImperfections);

      const ior = max(this.uIor, 1.0001).toVar();
      const dispersionSpread = this.uDispersion.div(12).toVar();
      // A plain three.js render shows each frame on its own (no frames averaged), so the
      // wavelengths are evenly spaced and fixed rather than jittered per pixel (which would be
      // static grain), and twice as many are taken to keep the fringes smooth.
      const bandCount = select(
        dispersionSpread.greaterThan(0).and(uRoughness.lessThan(0.35)),
        spectralSampleCount(dispersionSpread.mul(2), false),
        int(1),
      ).toVar();
      // Beer-Lambert: `color` is what survives `thickness` of glass.
      const absorptionSigma = log(max(uTint, vec3(1e-4)))
        .negate()
        .div(reach)
        .toVar();

      /**
       * What a ray leaving the glass at `pos` along `dir` sees: the environment, and with
       * transmission the opaque scene behind, read where the ray crosses the screen a little
       * way past the glass (blurred by roughness and the ray's cone).
       */
      const look = (pos: N, dir: N, cone: N) => {
        const seen = env(dir).toVar();
        if (useTransmission) {
          const clip = viewProj.mul(vec4(pos.add(dir.mul(reach.mul(1.5))), 1)).toVar();
          If(clip.w.greaterThan(0.01), () => {
            const ndc = clip.xy.div(clip.w);
            const uv = clamp(vec2(ndc.x, ndc.y.negate()).mul(0.5).add(0.5), vec2(0.002), vec2(0.998));
            const behind = viewportOpaqueMipTexture(uv, max(uRoughness.mul(5), cone.mul(8))).rgb;
            seen.assign(mix(seen, behind, uTransmission));
          });
        }
        return seen;
      };

      /** Local-sphere model: from a point with outward normal `n`, a ray `dir` crosses a sphere of the object's size. */
      const sphereExit = (pos: N, dir: N, n: N): GlassHit => {
        const t = sphereRadius.mul(max(dot(dir, n).negate().mul(2), 0.05)).toVar();
        return { pos: pos.add(dir.mul(t)).toVar(), nrm: normalize(n.sub(dir.mul(dot(dir, n).mul(2)))), t };
      };

      // The normal of the last surface met, for the local-sphere model's next exit.
      const lastNormal = vec3(n0).toVar();
      let exit: (pos: N, dir: N, ev: N) => GlassHit;
      if (volume) {
        // Exact exits through the baked volume: world -> mesh-local -> volume units and back.
        const layerCount = volume.layerCount ?? Math.max(1, Math.round(volume.texture.image.depth / 128));
        const sdf = createSdfVolume(texture3D(volume.texture), volume.layer ?? 0, layerCount);
        const [cx, cy, cz] = volume.center ?? [0, 0, 0];
        const volumeCenter = vec3(cx, cy, cz);
        const halfWidth = float(volume.halfWidth ?? 1);
        // Geometry space to world: the object's matrix, times the instance's on an InstancedMesh.
        const toWorld = instanceMatrix ? modelWorldMatrix.mul(instanceMatrix).toVar() : modelWorldMatrix;
        const toGeometry = instanceMatrix
          ? inverse(instanceMatrix).mul(modelWorldMatrixInverse).toVar()
          : modelWorldMatrixInverse;
        const worldScale = instanceMatrix
          ? length(toWorld.mul(vec4(1, 0, 0, 0)).xyz).toVar()
          : length(modelWorldMatrix[0].xyz).toVar();
        exit = (pos: N, dir: N) => {
          const localPos = toGeometry.mul(vec4(pos, 1)).xyz;
          const localDir = normalize(toGeometry.mul(vec4(dir, 0)).xyz);
          const hit = sdf.exit(localPos.sub(volumeCenter).div(halfWidth), localDir, float(1));
          const hitLocal = hit.pos.mul(halfWidth).add(volumeCenter);
          return {
            pos: toWorld.mul(vec4(hitLocal, 1)).xyz.toVar(),
            nrm: normalize(toWorld.mul(vec4(hit.nrm, 0)).xyz),
            t: hit.t.mul(halfWidth).mul(worldScale),
          };
        };
      } else {
        // The back-face map's exit (exact where it is bound and found behind this pixel) for
        // the first event, then the local-sphere model from the last surface met.
        const frontDist = length(positionWorld.sub(cameraPosition)).toVar();
        const backface = this.backfaceTexNode.sample(screenUV).level(0).toVar();
        const exactFirst = this.uHasBackface
          .greaterThan(0.5)
          .and(backface.a.greaterThan(frontDist.add(1e-4)))
          .and(dot(backface.rgb, backface.rgb).greaterThan(1e-4))
          .toVar();
        exit = (pos: N, dir: N, ev: N) => {
          const fromNormal = select(ev.equal(1), n0, lastNormal);
          const estimate = sphereExit(pos, dir, fromNormal);
          const useMap = exactFirst.and(ev.equal(1));
          const t = select(useMap, max(backface.a.sub(frontDist), 0.01), estimate.t).toVar();
          const nrm = select(useMap, normalize(backface.rgb), estimate.nrm).toVar();
          lastNormal.assign(nrm);
          return { pos: pos.add(dir.mul(t)).toVar(), nrm, t };
        };
      }

      const traced = traceGlass({
        entryPos: positionWorld,
        entryNormal: n0,
        viewDir: d0,
        ior,
        dispersionSpread,
        bandCount,
        spectralJitter: float(0.5),
        absorptionSigma,
        maxEvents: this.uBounces.toInt(),
        outerReflection: env(reflect(d0, n0)),
        exit,
        look,
        perturbNormal: (normal: N, pos: N) => perturbImperfections(normal, pos, reach, this.uImperfections),
      });
      const col = traced.color.div(max(traced.weightSum, vec3(1e-6))).toVar();

      // Creamy frosted veil + crisp Fresnel rim when roughness > 0.
      applyFrostVeil(col, d0, n0, uTint, vec3(0.75), uRoughness, frostGrain);
      return vec4(col, 1);
    })();
  }
}
