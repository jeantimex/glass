// Caustics for MeshGlassMaterial in plain three.js scenes (rendered with renderer.render()):
// the light of a DirectionalLight traced through every glass mesh onto a receiving floor
// plane, as real caustics: one photon pass per wavelength band (dispersion colours them, tinted
// glass filters them), each ray spread over the patch of floor it lights, softened by the
// lamp's size. The result is how much of the lamp's light reaches each point of the floor
// (shadows and focused light, in colour), applied by scaling the light's colour there.
//
//   const caustics = new GlassCaustics({ light: sun });
//   caustics.attach(); // the light's colour now includes the caustics, on every lit material
//   renderer.setAnimationLoop(() => {
//     caustics.update(renderer, scene); // redraws only when glass, light or floor changed
//     renderer.render(scene, camera);
//   });
//
// The photon pass follows the same exits as MeshGlassMaterial: three.js built-in solids
// exactly, other meshes through their (automatic or given) volume, and the material's colour
// and IOR. The floor is one horizontal plane (`floorY`); light reaching it through several
// glass objects in a row is followed through the first only.
/* eslint-disable @typescript-eslint/no-explicit-any -- TSL node types are too loose to spell out */
import * as THREE from 'three/webgpu';
import * as TSL from 'three/tsl';
import { analyticShape } from './analyticShapes';
import { autoGlassVolume } from './autoVolume';
import { createBlur, createCausticSplat } from './causticSplat';
import { instanceMatrixNode, MeshGlassMaterial, type GlassVolume } from './material';
import { fresnel, spectralBands } from './tsl/optics';
import { createSdfVolume } from './tsl/sdf';
import { shapeFunctions } from './tsl/shapes';

type N = any;
const {
  Break,
  Fn,
  If,
  Loop,
  bool,
  dot,
  float,
  int,
  inverse,
  length,
  log,
  max,
  min,
  modelWorldMatrix,
  modelWorldMatrixInverse,
  normalWorld,
  normalize,
  positionWorld,
  reflect,
  refract,
  smoothstep,
  texture,
  texture3D,
  transpose,
  uniform,
  vec2,
  vec3,
  vec4,
  exp,
} = TSL as N;

/** A half-float map for the caustic passes. */
function causticMap(size: number, linear: boolean, depth = false): THREE.RenderTarget {
  return new THREE.RenderTarget(size, size, {
    type: THREE.HalfFloatType,
    format: THREE.RGBAFormat,
    depthBuffer: depth,
    minFilter: linear ? THREE.LinearFilter : THREE.NearestFilter,
    magFilter: linear ? THREE.LinearFilter : THREE.NearestFilter,
  });
}

export interface GlassCausticsOptions {
  /** The lamp whose light is traced through the glass. */
  light: THREE.DirectionalLight;
  /** Height of the receiving floor plane (default 0). */
  floorY?: number;
  /** Half-width of the square of floor covered, around the floor's origin (default 4). */
  extent?: number;
  /** How wide the lamp is, as the blur of its shadows and caustics on the floor (m, default 0.05). */
  softness?: number;
  /** Rays traced each way (default 512: 512^2 rays). */
  resolution?: number;
  /**
   * How far above the open floor's light focused caustics may rise before being eased off, in
   * extra multiples of it (default 24, about what one glass lens reaches; lower for piles,
   * where light through several objects is counted as if through the top one only).
   */
  headroom?: number;
}

/**
 * Real caustics from every `MeshGlassMaterial` mesh in a scene, lit by one `DirectionalLight`,
 * on a horizontal floor plane. See the file comment for use.
 */
export class GlassCaustics {
  readonly light: THREE.DirectionalLight;
  floorY: number;
  extent: number;
  softness: number;
  headroom: number;

  /** The lamp's light reaching each point of the floor, in the light's view (rgb, 1: unchanged). */
  readonly target: THREE.RenderTarget;
  private readonly resolution: number;
  private readonly photonMap: THREE.RenderTarget;
  private readonly openMap: THREE.RenderTarget;
  private readonly landedMap: THREE.RenderTarget;
  private readonly gathered: THREE.RenderTarget;
  private readonly scratch: THREE.RenderTarget;
  private readonly lightCamera = new THREE.OrthographicCamera(-1, 1, 1, -1, 0.1, 100);
  private readonly photonScene = new THREE.Scene();
  private readonly uBand = uniform(new THREE.Vector4(0, 1, 1, 1));
  /** Light view: projection of a world point onto the map (xy in clip units). */
  private readonly uLightViewProj = uniform(new THREE.Matrix4());
  private readonly uLightDir = uniform(new THREE.Vector3(0, -1, 0));
  private readonly uLightRight = uniform(new THREE.Vector3(1, 0, 0));
  private readonly uLightUp = uniform(new THREE.Vector3(0, 0, -1));
  private readonly uFloorY = uniform(0);
  private readonly uHeadroom = uniform(24);
  private readonly mapNode: N;
  private splat: ReturnType<typeof createCausticSplat> | null = null;
  private blur: ReturnType<typeof createBlur>['blur'] | null = null;
  private readonly composeQuad: THREE.QuadMesh;
  private readonly composeOpen: N;
  private readonly composeLanded: N;
  /** One photon mesh per glass mesh found (rebuilt when the set of glass meshes changes). */
  private readonly proxies = new Map<THREE.Mesh, { proxy: THREE.Mesh; material: THREE.NodeMaterial; key: string }>();
  private lastState = '';
  private originalColorNode: N = null;

  constructor(options: GlassCausticsOptions) {
    this.light = options.light;
    this.floorY = options.floorY ?? 0;
    this.extent = options.extent ?? 4;
    this.softness = options.softness ?? 0.05;
    this.headroom = options.headroom ?? 24;
    this.resolution = options.resolution ?? 512;
    this.photonMap = causticMap(this.resolution, false, true);
    this.openMap = causticMap(this.resolution, true);
    this.landedMap = causticMap(this.resolution, true);
    this.gathered = causticMap(this.resolution, true);
    this.scratch = causticMap(this.resolution, true);
    this.target = causticMap(this.resolution, true);
    this.mapNode = texture(this.target.texture);

    // open + landed * 1, then (in update) blurred by the lamp's size into `target`.
    this.composeOpen = texture(this.openMap.texture);
    this.composeLanded = texture(this.landedMap.texture);
    const composeMaterial = new THREE.NodeMaterial();
    composeMaterial.fragmentNode = vec4(this.composeOpen.rgb.add(this.composeLanded.rgb), 1);
    composeMaterial.depthTest = composeMaterial.depthWrite = false;
    this.composeQuad = new THREE.QuadMesh(composeMaterial);
  }

  /**
   * How much of the lamp's light reaches a world point on the floor (rgb; 1 unchanged, 0 in full
   * shadow, above 1 in focused caustics), eased above the open floor's level by `headroom`.
   * For a custom floor material; `attach()` applies it to the light instead.
   */
  lightAt(worldPos: N = positionWorld): N {
    return Fn(() => {
      const clip = this.uLightViewProj.mul(vec4(worldPos, 1)).toVar();
      const uv = vec2(clip.x.mul(0.5).add(0.5), float(0.5).sub(clip.y.mul(0.5))).toVar();
      const out = vec3(1).toVar();
      const inside = uv.x.greaterThan(0).and(uv.x.lessThan(1)).and(uv.y.greaterThan(0)).and(uv.y.lessThan(1));
      // Only at the floor's height (objects above it are lit as usual).
      If(inside.and(worldPos.y.sub(this.uFloorY).abs().lessThan(0.02)), () => {
        const raw = max(this.mapNode.sample(uv).level(0).rgb, vec3(0)).toVar();
        const over = max(raw.sub(1), vec3(0));
        out.assign(min(raw, vec3(1)).add(over.div(over.div(this.uHeadroom).add(1))));
      });
      return out;
    })();
  }

  /** Scale the light's colour by `lightAt()`, so every lit material in the scene gets the caustics. */
  attach(): void {
    if (this.originalColorNode !== null) return;
    const light = this.light as THREE.DirectionalLight & { colorNode?: N };
    this.originalColorNode = light.colorNode ?? null;
    const lightColor = uniform(light.color).mul(uniform(light.intensity).onObjectUpdate(() => light.intensity));
    light.colorNode = lightColor.mul(this.lightAt());
  }

  /** Undo `attach()`. */
  detach(): void {
    const light = this.light as THREE.DirectionalLight & { colorNode?: N };
    light.colorNode = this.originalColorNode ?? undefined;
    this.originalColorNode = null;
  }

  /** Redraw the caustics if the glass, the light or the settings changed since the last call. */
  update(renderer: THREE.WebGPURenderer, scene: THREE.Scene): void {
    const glass = this.findGlass(scene);
    const state = this.stateKey(glass);
    if (state === this.lastState) return;
    this.lastState = state;
    this.render(renderer, glass);
  }

  /** Release the GPU resources. */
  dispose(): void {
    for (const t of [this.photonMap, this.openMap, this.landedMap, this.gathered, this.scratch, this.target])
      t.dispose();
    for (const { material } of this.proxies.values()) material.dispose();
    this.detach();
  }

  /* --- internals --- */

  private findGlass(scene: THREE.Scene): THREE.Mesh[] {
    const found: THREE.Mesh[] = [];
    scene.traverseVisible((object) => {
      const mesh = object as THREE.Mesh;
      if (mesh.isMesh && mesh.material instanceof MeshGlassMaterial) found.push(mesh);
    });
    return found;
  }

  /** Everything that changes the caustics: the glass (where, what, its optics), the light, the settings. */
  private stateKey(glass: THREE.Mesh[]): string {
    this.light.updateMatrixWorld();
    const parts: (string | number)[] = [
      this.light.matrixWorld.elements.join(),
      this.light.target.matrixWorld.elements.join(),
      this.floorY,
      this.extent,
      this.softness,
    ];
    for (const mesh of glass) {
      mesh.updateMatrixWorld();
      const m = mesh.material as MeshGlassMaterial;
      parts.push(mesh.id, mesh.matrixWorld.elements.join(), mesh.geometry.id, m.ior, m.dispersion, m.color.getHex());
      parts.push(m.version);
      const inst = mesh as THREE.InstancedMesh;
      if (inst.isInstancedMesh) parts.push(inst.count, inst.instanceMatrix.version, inst.instanceColor?.version ?? -1);
    }
    return parts.join('|');
  }

  /** The light's camera: looking along the light, covering the floor's square. */
  private placeLightCamera(): void {
    const dir = new THREE.Vector3()
      .subVectors(
        this.light.target.getWorldPosition(new THREE.Vector3()),
        this.light.getWorldPosition(new THREE.Vector3()),
      )
      .normalize();
    const half = this.extent;
    const cam = this.lightCamera;
    cam.left = cam.bottom = -half;
    cam.right = cam.top = half;
    cam.near = 0.1;
    cam.far = half * 8;
    const centre = new THREE.Vector3(0, this.floorY, 0);
    cam.position.copy(centre).addScaledVector(dir, -half * 4);
    cam.up.set(0, 0, -1);
    cam.lookAt(centre);
    cam.updateProjectionMatrix();
    cam.updateMatrixWorld();
    this.uLightViewProj.value.multiplyMatrices(cam.projectionMatrix, cam.matrixWorldInverse);
    this.uLightDir.value.copy(dir);
    const e = cam.matrixWorld.elements;
    this.uLightRight.value.set(e[0], e[1], e[2]).divideScalar(half);
    this.uLightUp.value.set(e[4], e[5], e[6]).divideScalar(half);
    this.uFloorY.value = this.floorY;
    this.uHeadroom.value = this.headroom;
  }

  private render(renderer: THREE.WebGPURenderer, glass: THREE.Mesh[]): void {
    this.placeLightCamera();
    this.syncProxies(glass);
    if (!this.splat) this.splat = createCausticSplat(this.photonMap, this.resolution, this.uBand);
    if (!this.blur) this.blur = createBlur(renderer, this.scratch).blur;

    const previousTarget = renderer.getRenderTarget();
    const previousClear = renderer.getClearColor(new THREE.Color());
    const previousAlpha = renderer.getClearAlpha();
    const previousAutoClear = renderer.autoClear;

    // Wavelength bands: more with stronger dispersion, three for coloured glass, else one.
    let dispersion = 0;
    let coloured = false;
    for (const mesh of glass) {
      const m = mesh.material as MeshGlassMaterial;
      dispersion = Math.max(dispersion, m.dispersion);
      coloured ||=
        Math.min(m.color.r, m.color.g, m.color.b) < 0.999 || Boolean((mesh as THREE.InstancedMesh).instanceColor);
    }
    const count = dispersion > 0 ? Math.min(Math.max(Math.ceil((dispersion / 12) * 64), 4), 10) : coloured ? 3 : 1;
    const bands = spectralBands(count).map((b) => new THREE.Vector4(b.cauchy, b.rgb[0], b.rgb[1], b.rgb[2]));

    // Each band: where its rays land, then splat them (added up over the bands).
    renderer.setClearColor(0x000000, 0);
    renderer.autoClear = true;
    for (let b = 0; b < bands.length; b++) {
      this.uBand.value.copy(bands[b]);
      renderer.setRenderTarget(this.photonMap);
      renderer.render(this.photonScene, this.lightCamera);
      if (b === 0) {
        // The light meeting no glass (the same for every band).
        renderer.setRenderTarget(this.openMap);
        this.splat.openQuad.render(renderer);
        renderer.setClearColor(0x000000, 1);
        renderer.setRenderTarget(this.landedMap);
        renderer.clear();
      }
      renderer.setRenderTarget(this.landedMap);
      renderer.autoClear = false;
      renderer.render(this.splat.landScene, this.lightCamera);
      renderer.autoClear = true;
    }
    // Open + landed, blurred together by the lamp's size (texture coordinates per metre: 0.5 / extent).
    renderer.setRenderTarget(this.gathered);
    this.composeQuad.render(renderer);
    this.blur(this.gathered.texture, this.target, (this.softness * 0.5) / this.extent, undefined, this.scratch, 0, 2);

    renderer.setRenderTarget(previousTarget);
    renderer.setClearColor(previousClear, previousAlpha);
    renderer.autoClear = previousAutoClear;
  }

  /** Keep one photon proxy per glass mesh, sharing its geometry, matrices and instances. */
  private syncProxies(glass: THREE.Mesh[]): void {
    const keep = new Set(glass);
    for (const [mesh, entry] of this.proxies) {
      if (!keep.has(mesh)) {
        this.photonScene.remove(entry.proxy);
        entry.material.dispose();
        this.proxies.delete(mesh);
      }
    }
    for (const mesh of glass) {
      const m = mesh.material as MeshGlassMaterial;
      const shape =
        m.volume === null || (m.volume !== 'auto' && m.volume)
          ? null
          : m.exactShapes
            ? analyticShape(mesh.geometry)
            : null;
      const given = m.volume !== 'auto' ? m.volume : null;
      const volume: GlassVolume | null =
        given ??
        (shape === null && m.volume === 'auto' ? autoGlassVolume(mesh.geometry, () => (this.lastState = '')) : null);
      const key = `${mesh.geometry.id}:${shape ? shape.kind : 'v'}:${volume ? volume.texture.id : 'none'}`;
      let entry = this.proxies.get(mesh);
      if (entry && entry.key !== key) {
        this.photonScene.remove(entry.proxy);
        entry.material.dispose();
        entry = undefined;
      }
      if (!entry) {
        const material = this.photonMaterial(m, shape, volume);
        const proxy = (mesh as THREE.InstancedMesh).isInstancedMesh
          ? new THREE.InstancedMesh(mesh.geometry, material, (mesh as THREE.InstancedMesh).count)
          : new THREE.Mesh(mesh.geometry, material);
        proxy.matrixAutoUpdate = false;
        proxy.frustumCulled = false;
        this.photonScene.add(proxy);
        entry = { proxy, material, key };
        this.proxies.set(mesh, entry);
      }
      entry.proxy.matrix.copy(mesh.matrixWorld);
      entry.proxy.matrixWorld.copy(mesh.matrixWorld);
      if ((mesh as THREE.InstancedMesh).isInstancedMesh) {
        const src = mesh as THREE.InstancedMesh;
        const dst = entry.proxy as THREE.InstancedMesh;
        dst.instanceMatrix = src.instanceMatrix;
        dst.instanceColor = src.instanceColor;
        dst.count = src.count;
      }
    }
  }

  /**
   * The photon material for one glass mesh: traces the lamp's ray through it (same exits as
   * MeshGlassMaterial) to the floor, writing `vec4(landing offset, light carried, 1)` as the
   * instanced renderer's photon pass does (tsl/caustics.ts).
   */
  private photonMaterial(
    glass: MeshGlassMaterial,
    shape: ReturnType<typeof analyticShape>,
    volume: GlassVolume | null,
  ): THREE.NodeMaterial {
    const material = new THREE.NodeMaterial();
    material.transparent = true;
    material.blending = THREE.NoBlending;
    material.side = THREE.FrontSide;
    const uIor = uniform(glass.ior).onObjectUpdate(() => glass.ior);
    const uDispersion = uniform(glass.dispersion).onObjectUpdate(() => glass.dispersion);
    const uTint = uniform(glass.color);
    const band = this.uBand;
    const lightDir = this.uLightDir;
    const lightRight = this.uLightRight;
    const lightUp = this.uLightUp;
    const floorY = this.uFloorY;
    material.fragmentNode = Fn((builder: N) => {
      // Geometry space to world (times the instance's matrix on an InstancedMesh).
      const object = builder.object as THREE.InstancedMesh;
      const instanceMatrix = object?.isInstancedMesh ? instanceMatrixNode(builder, object) : null;
      const toWorld = instanceMatrix ? modelWorldMatrix.mul(instanceMatrix).toVar() : modelWorldMatrix;
      const toGeometry = instanceMatrix
        ? inverse(instanceMatrix).mul(modelWorldMatrixInverse).toVar()
        : modelWorldMatrixInverse;
      const ior = max(uIor.add(uDispersion.div(12).mul(band.x)), 1.0001).toVar();
      /** The lamp's ray through this point never enters the glass (see the volume exits). */
      const missed = bool(false).toVar();
      const rayDir = vec3(lightDir).toVar();
      const pos = vec3(positionWorld).toVar();
      const n = normalize(normalWorld).toVar();
      // The exits, as MeshGlassMaterial finds them.
      let exit: (p: N, d: N) => { pos: N; nrm: N; t: N };
      if (shape) {
        const toShape = (v: N) => (shape.axis === 'z' ? vec3(v.x, v.z.negate(), v.y) : v);
        const fromShape = (v: N) => (shape.axis === 'z' ? vec3(v.x, v.z, v.y.negate()) : v);
        const normalToWorld = (v: N) => normalize(transpose(toGeometry).mul(vec4(fromShape(v), 0)).xyz);
        const size = uniform(new THREE.Vector3(...shape.size));
        const fns = shapeFunctions(shape.kind, { bevel: float(0) });
        const snapped = fns.entry(
          toShape(toGeometry.mul(vec4(positionWorld, 1)).xyz),
          size,
          normalize(toShape(transpose(toWorld).mul(vec4(normalWorld, 0)).xyz)),
        );
        pos.assign(toWorld.mul(vec4(fromShape(snapped.pos), 1)).xyz);
        n.assign(normalToWorld(snapped.nrm));
        exit = (p: N, d: N) => {
          const hit = fns.exit(
            toShape(toGeometry.mul(vec4(p, 1)).xyz).toVar(),
            normalize(toShape(toGeometry.mul(vec4(d, 0)).xyz)).toVar(),
            size,
          );
          const w = toWorld.mul(vec4(fromShape(hit.pos), 1)).xyz.toVar();
          return { pos: w, nrm: normalToWorld(hit.nrm).toVar(), t: length(w.sub(p)).toVar() };
        };
      } else if (volume) {
        const layerCount = volume.layerCount ?? Math.max(1, Math.round(volume.texture.image.depth / 128));
        const sdf = createSdfVolume(texture3D(volume.texture), volume.layer ?? 0, layerCount);
        const [cx, cy, cz] = volume.center ?? [0, 0, 0];
        const c = vec3(cx, cy, cz);
        const hw = float(volume.halfWidth ?? 1);
        // Enter where the light really meets the solid: from the rasterized point, along the
        // light, onto the volume's surface, with its normal. A mesh made of overlapping open
        // patches (the teapot) puts seams inside the solid, which the lamp would otherwise see
        // as the nearest surface and enter through at a wrong, often grazing, angle.
        const lightLocal = normalize(toGeometry.mul(vec4(rayDir, 0)).xyz).toVar();
        const start = toGeometry.mul(vec4(positionWorld, 1)).xyz.sub(c).div(hw).toVar();
        const back = float(0).toVar(); // distance back toward the lamp (volume units)
        Loop({ start: int(0), end: int(32), type: 'int', condition: '<', name: 'en' }, () => {
          const d = sdf.volumeFast(start.sub(lightLocal.mul(back)), float(1)).toVar();
          If(d.greaterThanEqual(0), () => {
            Break();
          });
          back.addAssign(max(d.negate(), 0.004));
        });
        Loop({ start: int(0), end: int(4), type: 'int', condition: '<', name: 'eb' }, () => {
          back.subAssign(sdf.volumeFast(start.sub(lightLocal.mul(back)), float(1)));
        });
        // Where the rasterized point lies outside the solid (thin edges the volume rounds away),
        // look ahead along the light for it; a ray that never enters passes as if no glass were
        // there (it lands where it would anyway, carrying all its light).
        const ahead = float(0).toVar();
        If(sdf.volumeFast(start, float(1)).greaterThan(0), () => {
          missed.assign(true);
          Loop({ start: int(0), end: int(24), type: 'int', condition: '<', name: 'ea' }, () => {
            const d = sdf.volumeFast(start.add(lightLocal.mul(ahead)), float(1)).toVar();
            If(d.lessThan(0.002), () => {
              missed.assign(false);
              Break();
            });
            ahead.addAssign(max(d, 0.004));
            If(ahead.greaterThan(0.25), () => {
              Break();
            });
          });
        });
        const entryLocal = start.sub(lightLocal.mul(back)).add(lightLocal.mul(ahead)).toVar();
        pos.assign(toWorld.mul(vec4(entryLocal.mul(hw).add(c), 1)).xyz);
        n.assign(normalize(transpose(toGeometry).mul(vec4(sdf.volumeSmoothNormal(entryLocal, float(1)), 0)).xyz));
        exit = (p: N, d: N) => {
          const lp = toGeometry.mul(vec4(p, 1)).xyz;
          const ld = normalize(toGeometry.mul(vec4(d, 0)).xyz);
          const hit = sdf.exit(lp.sub(c).div(hw), ld, float(1));
          const w = toWorld.mul(vec4(hit.pos.mul(hw).add(c), 1)).xyz.toVar();
          return { pos: w, nrm: normalize(toWorld.mul(vec4(hit.nrm, 0)).xyz).toVar(), t: length(w.sub(p)).toVar() };
        };
      } else {
        // No exact exits yet (an automatic volume still baking): the thickness estimate.
        const radius = max(float(glass.thickness), 0.05).mul(0.5);
        exit = (p: N, d: N) => {
          const t = radius.mul(max(dot(d, n).negate().mul(2), 0.05)).toVar();
          return { pos: p.add(d.mul(t)).toVar(), nrm: normalize(n.sub(d.mul(dot(d, n).mul(2)))).toVar(), t };
        };
      }
      If(dot(rayDir, n).greaterThan(0), () => {
        n.assign(n.negate());
      });
      const transmitted = float(1)
        .sub(fresnel(dot(rayDir, n), ior))
        .mul(smoothstep(0.02, 0.09, dot(rayDir, n).negate()))
        .toVar();
      // Beer-Lambert: `color` is what survives `thickness` of glass.
      const sigma = log(max(uTint, vec3(1e-4)))
        .negate()
        .div(max(float(glass.thickness), 0.05))
        .toVar();
      const absorbed = vec3(1).toVar();
      const dir = refract(rayDir, n, float(1).div(ior)).toVar();
      const exitDir = vec3(0).toVar();
      const escaped = bool(false).toVar();
      If(dot(dir, dir).greaterThan(1e-6), () => {
        Loop({ start: int(0), end: int(8), type: 'int', condition: '<', name: 'ev' }, () => {
          const hit = exit(pos, dir);
          const hitNormal = vec3(hit.nrm).toVar();
          // An exit faces the way the ray leaves. Where a volume rounds a thin part away (a
          // teapot's spout or handle) its gradient can point back at the ray, and refract()
          // would then quietly reflect it instead of letting it out.
          If(dot(dir, hitNormal).lessThan(0), () => {
            hitNormal.assign(hitNormal.negate());
          });
          absorbed.mulAssign(exp(sigma.mul(hit.t.negate())));
          pos.assign(hit.pos);
          const refracted = refract(dir, hitNormal.negate(), ior).toVar();
          If(dot(refracted, refracted).greaterThan(1e-6), () => {
            transmitted.mulAssign(float(1).sub(fresnel(dot(dir, hitNormal), float(1).div(ior))));
            exitDir.assign(refracted);
            escaped.assign(true);
            Break();
          });
          dir.assign(reflect(dir, hitNormal.negate()));
        });
      });
      const landing = vec2(0).toVar();
      const energy = float(0).toVar();
      If(missed, () => {
        // Straight on, with all its light: the same as light meeting no glass.
        energy.assign(1);
      });
      If(missed.not().and(escaped).and(exitDir.y.lessThan(-0.02)), () => {
        const delta = pos
          .add(exitDir.mul(floorY.sub(pos.y).div(exitDir.y)))
          .sub(positionWorld)
          .toVar();
        landing.assign(vec2(dot(delta, lightRight), dot(delta, lightUp)));
        const share = band.yzw;
        energy.assign(transmitted.mul(dot(absorbed, share).div(max(dot(share, vec3(1)), 1e-4))));
      });
      return vec4(landing, energy, 1);
    })();
    return material;
  }
}
