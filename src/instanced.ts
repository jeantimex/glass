// Real-time renderer for thousands of glass objects (see instanced.wgsl).
//
// Per frame:
//   1. caustic map   every object drawn from the main lamp's direction, multiplying
//                    up how much of its light each surface lets through
//                    (only when something moved or the light changed)
//   2. background    ground (lit through the caustic map) and sky
//   3. far glass     the farther half of the objects, over the background
//   4. near glass    the nearer half, which sees the far half refracted
//   5. finishing     contact shadows, then depth of field, glow and tones
//
// The layers carry each pixel's distance from the camera in their alpha, which is
// what the finishing passes work from.
import * as THREE from 'three/webgpu';
import { diamondGeometry, diamondWgsl } from './diamond';

export { diamondPoints } from './diamond';
import { attribute, cross, normalLocal, positionGeometry, positionWorld, sampler, screenCoordinate, texture, texture3D, uniform, vec2, vec3, wgslFn, float, uv, vec4, varying, instanceIndex } from 'three/tsl';
import source from './instanced.wgsl?raw';

export const MAX_OBJECTS = 20000;

/** Shape ids, shared with instanced.wgsl. */
export const KIND = { box: 0, sphere: 1, cylinder: 2, capsule: 3, cone: 4, torus: 5, octahedron: 6, prism: 7, hexprism: 8, suzanne: 9, diamond: 10, dragon: 11 } as const;
const KIND_COUNT = 12;

/** Tube radius of the unit torus geometry (its ring radius is 1). */
const UNIT_TUBE = 0.25;

/**
 * The objects to draw. `size` is per shape (see the table in instanced.wgsl):
 * box half-extents; sphere radius; cylinder, capsule, cone and hex prism radius
 * in x and half-height in y; torus ring and tube radius; octahedron size;
 * triangular prism size in x and half-length in z; Suzanne and dragon half-width;
 * diamond size (see diamond.ts). Whoever
 * moves the objects writes `center` and `quat` in place.
 */
export interface GlassObjects {
  count: number;
  kind: Uint8Array;
  center: Float32Array;
  quat: Float32Array;
  size: Float32Array;
  /**
   * Per object: added to the IOR, multiplying the dispersion (so piles are not
   * uniform), and how strongly it is a CMY cube (0: plain glass).
   */
  vary: Float32Array;
  /** Per object: the colour of its glass (linear rgb; white for clear). */
  tint: Float32Array;
}

export function createGlassObjects(): GlassObjects {
  return {
    count: 0,
    kind: new Uint8Array(MAX_OBJECTS),
    center: new Float32Array(MAX_OBJECTS * 3),
    quat: new Float32Array(MAX_OBJECTS * 4),
    size: new Float32Array(MAX_OBJECTS * 3),
    vary: new Float32Array(MAX_OBJECTS * 3),
    tint: new Float32Array(MAX_OBJECTS * 3).fill(1),
  };
}

export interface InstancedLighting {
  ground: boolean;
  /** The dark studio (black room, a few lamps) instead of the white one. */
  dark: boolean;
  /** Glass shows other glass through it, along the bent rays (off: only the floor and the studio). */
  seeThrough: boolean;
  /** Real caustics: the lamp's light traced through the glass onto the floor, and how soft they are (m). */
  realCaustics: boolean;
  causticSoftness: number;
  /** How much of the light gathered by the glass reaches the floor (1: all of it). */
  causticStrength: number;
  /** Contact shadows: strength (0 to 1) and how far they reach (m). */
  contactShadows: number;
  contactRadius: number;
  /** The camera: focus distance (m), aperture (0: everything sharp), glow, film-like tones. */
  focus: number;
  depthOfField: number;
  glow: number;
  filmic: boolean;
  /** Stage spotlight: radius of its beam where it meets the floor, in metres (0: none). */
  spotRadius: number;
  /** How reflective the floor is: 0 (matt) to 1. */
  floorReflection: number;
  /** Studio shadows: how far their edges are blurred (m), and how dark they are (0 to 1). */
  shadowSoftness: number;
  shadowDarkness: number;
  /** Objects cast the lamp's shadow on each other (see icLampVisible). */
  objectShadows: boolean;
  /** A grid of lines drawn on the floor, and how thick the fog over it is (0: none). */
  floorGrid: boolean;
  fog: number;
  groundColor: [number, number, number];
  exposure: number;
  /** Irradiance / pi on the ground: from the room all round, and from the main lamp. */
  skyE: [number, number, number];
  sunE: [number, number, number];
  /** Unit vector toward the main lamp. */
  sunDir: [number, number, number];
}

const CAUSTIC_MAP_SIZE = 2048;
/** The lamp's depth map, for shadows on the objects themselves. */
const LAMP_DEPTH_SIZE = 1024;
const SHADE_MAP_SIZE = 1024;
/** Real caustics: this many rays of light each way (so its square in all). */
const PHOTON_MAP_SIZE = 1024;
/** Up to this many objects, the round shapes are drawn with their fine meshes. */
const FINE_UP_TO = 64;
/** Above this many objects, the less visible passes are drawn more cheaply. */
const MANY_OBJECTS = 3000;

type Pass = 'CAUSTIC' | 'GROUND' | 'GLASS' | 'BLUR' | 'SHADE' | 'LENS' | 'PHOTON';
function programFor(pass: Pass): string {
  const kept: string[] = [];
  let keep = true;
  for (const line of source.split('\n')) {
    const open = line.match(/^\s*\/\/#if (.+)$/);
    if (open) keep = open[1].trim().split(/\s+/).includes(pass);
    else if (/^\s*\/\/#endif/.test(line)) keep = true;
    else if (keep) kept.push(line);
  }
  return kept.join('\n');
}

/** One batch of instances of one shape: per-instance centre, rotation and size. */
class Batch {
  geometry = new THREE.InstancedBufferGeometry();
  center: THREE.InstancedBufferAttribute;
  quat: THREE.InstancedBufferAttribute;
  size: THREE.InstancedBufferAttribute;
  vary: THREE.InstancedBufferAttribute;
  tint: THREE.InstancedBufferAttribute;
  count = 0;

  /** @param capacity the most instances it will ever hold */
  constructor(base: THREE.BufferGeometry, capacity = MAX_OBJECTS) {
    this.center = new THREE.InstancedBufferAttribute(new Float32Array(capacity * 3), 3);
    this.quat = new THREE.InstancedBufferAttribute(new Float32Array(capacity * 4), 4);
    this.size = new THREE.InstancedBufferAttribute(new Float32Array(capacity * 3), 3);
    this.vary = new THREE.InstancedBufferAttribute(new Float32Array(capacity * 3), 3);
    this.tint = new THREE.InstancedBufferAttribute(new Float32Array(capacity * 3), 3);
    this.geometry.index = base.index;
    this.geometry.setAttribute('position', base.getAttribute('position'));
    this.geometry.setAttribute('normal', base.getAttribute('normal'));
    // Left as ordinary (static) buffers on purpose: three.js uploads a buffer marked
    // dynamic in full on every render pass, changed or not, which at thousands of
    // objects costs more than everything else. These are sent when commit() says so.
    this.geometry.setAttribute('instCenter', this.center);
    this.geometry.setAttribute('instQuat', this.quat);
    this.geometry.setAttribute('instSize', this.size);
    this.geometry.setAttribute('instVary', this.vary);
    this.geometry.setAttribute('instTint', this.tint);
    this.geometry.instanceCount = 0;
  }

  push(objects: GlassObjects, i: number) {
    const k = this.count++;
    // (Copied number by number: for a few values that is far quicker than subarray and set.)
    const center = this.center.array;
    const quat = this.quat.array;
    const size = this.size.array;
    const vary = this.vary.array;
    center[k * 3] = objects.center[i * 3];
    center[k * 3 + 1] = objects.center[i * 3 + 1];
    center[k * 3 + 2] = objects.center[i * 3 + 2];
    quat[k * 4] = objects.quat[i * 4];
    quat[k * 4 + 1] = objects.quat[i * 4 + 1];
    quat[k * 4 + 2] = objects.quat[i * 4 + 2];
    quat[k * 4 + 3] = objects.quat[i * 4 + 3];
    size[k * 3] = objects.size[i * 3];
    size[k * 3 + 1] = objects.size[i * 3 + 1];
    size[k * 3 + 2] = objects.size[i * 3 + 2];
    vary[k * 3] = objects.vary[i * 3];
    vary[k * 3 + 1] = objects.vary[i * 3 + 1];
    vary[k * 3 + 2] = objects.vary[i * 3 + 2];
    const tint = this.tint.array;
    tint[k * 3] = objects.tint[i * 3];
    tint[k * 3 + 1] = objects.tint[i * 3 + 1];
    tint[k * 3 + 2] = objects.tint[i * 3 + 2];
  }

  /** Instances uploaded last time: nothing needs sending while a batch stays empty. */
  private sent = 0;

  commit() {
    this.geometry.instanceCount = this.count;
    if (!this.count && !this.sent) return;
    this.sent = this.count;
    // Send only the instances in use, not the whole (mostly empty) buffers.
    for (const a of [this.center, this.quat, this.size, this.vary, this.tint]) {
      a.clearUpdateRanges();
      a.addUpdateRange(0, Math.max(this.count, 1) * a.itemSize);
      a.needsUpdate = true;
    }
  }
}

export function createInstancedGlass(
  renderer: THREE.WebGPURenderer,
  camera: THREE.PerspectiveCamera,
  objects: GlassObjects,
  /**
   * The shapes made from meshes: Suzanne's mesh, the dragon's (in full, and
   * simplified for piles), and their baked distance volumes, one after the other in
   * a single 3D texture (Suzanne's, then the dragon's).
   */
  meshes: { suzanne: THREE.BufferGeometry; dragon: THREE.BufferGeometry; dragonLod: THREE.BufferGeometry; volume: THREE.Data3DTexture },
) {
  // ------------------------------------------------------------ targets ---
  const causticMap = new THREE.RenderTarget(CAUSTIC_MAP_SIZE, CAUSTIC_MAP_SIZE, {
    type: THREE.HalfFloatType,
    format: THREE.RGBAFormat,
    depthBuffer: false,
    minFilter: THREE.LinearFilter,
    magFilter: THREE.LinearFilter,
  });
  // How far toward the lamp the nearest object reaches, seen from the lamp (red, plus
  // 100 so that 0, as cleared, means nothing there): the shadows objects cast on
  // each other (see icLampVisible).
  const lampDepth = new THREE.RenderTarget(LAMP_DEPTH_SIZE, LAMP_DEPTH_SIZE, {
    type: THREE.HalfFloatType,
    format: THREE.RGBAFormat,
    depthBuffer: true,
    minFilter: THREE.NearestFilter,
    magFilter: THREE.NearestFilter,
  });
  // The caustic map blurred, for soft shadows: a little, and far wider.
  const makeShade = () =>
    new THREE.RenderTarget(SHADE_MAP_SIZE, SHADE_MAP_SIZE, {
      type: THREE.HalfFloatType,
      format: THREE.RGBAFormat,
      depthBuffer: false,
      minFilter: THREE.LinearFilter,
      magFilter: THREE.LinearFilter,
    });
  const shadeScratch = makeShade();
  // Real caustics: where each ray of the lamp's light lands, and the rays added up there.
  const photonMap = new THREE.RenderTarget(PHOTON_MAP_SIZE, PHOTON_MAP_SIZE, {
    type: THREE.HalfFloatType,
    format: THREE.RGBAFormat,
    depthBuffer: true,
    minFilter: THREE.NearestFilter,
    magFilter: THREE.NearestFilter,
  });
  const landedMap = new THREE.RenderTarget(PHOTON_MAP_SIZE, PHOTON_MAP_SIZE, {
    type: THREE.HalfFloatType,
    format: THREE.RGBAFormat,
    depthBuffer: false,
    minFilter: THREE.LinearFilter,
    magFilter: THREE.LinearFilter,
  });
  const shadeNear = makeShade();
  const shadeFar = makeShade();
  // Real caustics, on the way to the near map: the plain shadow, and the gathered light blurred.
  const openMap = makeShade();
  const gatheredSoft = makeShade();
  const makeTarget = (depthBuffer: boolean) =>
    new THREE.RenderTarget(4, 4, {
      type: THREE.HalfFloatType,
      format: THREE.RGBAFormat,
      depthBuffer,
      // The layers that glass is drawn into are multisampled, like the canvas.
      samples: depthBuffer ? 4 : 0,
      minFilter: THREE.LinearFilter,
      magFilter: THREE.LinearFilter,
    });
  const background = makeTarget(false); // ground and sky
  const farLayer = makeTarget(true); // background plus the farther half of the glass
  const nearLayer = makeTarget(true); // plus the nearer half: the whole picture
  const shadedLayer = makeTarget(false); // with contact shadows
  // The glow of the brightest glints, at a quarter of the size.
  const glowScratch = makeTarget(false);
  const glowLayer = makeTarget(false);
  // The objects mirrored in the floor, at half size. Not multisampled: it is blurred.
  const mirrorLayer = new THREE.RenderTarget(4, 4, {
    type: THREE.HalfFloatType,
    format: THREE.RGBAFormat,
    depthBuffer: true,
    minFilter: THREE.LinearFilter,
    magFilter: THREE.LinearFilter,
  });

  // ----------------------------------------------------------- uniforms ---
  const uRes = uniform(new THREE.Vector2(4, 4));
  // 1 when few enough objects are on show for light to enter a baked mesh through its volume (see icEntry).
  const uMeshEntry = uniform(0);
  const uCamPos = uniform(new THREE.Vector3());
  const uCamRight = uniform(new THREE.Vector4());
  const uCamUp = uniform(new THREE.Vector3());
  const uCamFwd = uniform(new THREE.Vector4());
  const uViewProj = uniform(new THREE.Matrix4());
  const uRoom = uniform(1); // 1 if there is a floor, plus 2 for the dark studio
  const uSkyE = uniform(new THREE.Vector3());
  const uSunE = uniform(new THREE.Vector3());
  const uAlbedo = uniform(new THREE.Vector3());
  const sunDir = new THREE.Vector3(0, 1, 0); // toward the main lamp
  const uSunDir = uniform(sunDir);
  const uLightC = uniform(new THREE.Vector3());
  const uLightR = uniform(new THREE.Vector3());
  const uLightU = uniform(new THREE.Vector3());
  const uGlass = uniform(new THREE.Vector4(1.45, 1, 6, 0)); // ior, dispersion, max events, steps to look for other glass
  const uShade = uniform(new THREE.Vector4()); // contact shadows: strength, radius (m), exposure
  const uLens = uniform(new THREE.Vector4()); // focus distance (m), aperture, glow, tones on
  const uBlurCut = uniform(0);
  const uBlurReach = uniform(6);
  const uTint = uniform(new THREE.Vector4(1, 1, 1, 1)); // the colour of all the glass (linear rgb), and how strongly it applies
  const uCmy = uniform(0); // above 0: boxes are CMY cubes, with filters this strong
  const uBevel = uniform(0); // how much is cut off the edges of boxes (m)
  const uSpot = uniform(new THREE.Vector4()); // stage spotlight: beam radius at the floor (0: off), distance of the lamp
  const uFloor = uniform(new THREE.Vector4()); // x: how reflective the floor is; y: how dark its shadows are
  const uBlurDir = uniform(new THREE.Vector2());
  const uMirrorCam = uniform(new THREE.Vector3()); // the camera, mirrored in the floor

  // Named so the shader's helper functions can read them as globals.
  const lightNode = texture(causticMap.texture).setName('icLightTex');
  const behindNode = texture(background.texture).setName('icBehindTex');
  const lampDepthNode = texture(lampDepth.texture).setName('icLampDepthTex');
  const mirrorNode = texture(mirrorLayer.texture);
  const shadeNode = texture(shadeNear.texture).setName('icShadeTex');
  const shadeFarNode = texture(shadeFar.texture).setName('icShadeFarTex');
  const blurSource = texture(causticMap.texture);
  const volumeNode = texture3D(meshes.volume).setName('icSdfTex');

  // ------------------------------------------------- per-instance nodes ---
  const instCenter = attribute('instCenter', 'vec3');
  const instQuat = attribute('instQuat', 'vec4');
  const instSize = attribute('instSize', 'vec3');
  const instVary = attribute('instVary', 'vec3');
  const instTint = attribute('instTint', 'vec3');
  // v rotated by the instance quaternion: v + 2 q.xyz x (q.xyz x v + q.w v)
  // eslint-disable-next-line @typescript-eslint/no-explicit-any -- TSL node types are too loose to spell out here
  const rotate = (v: any) => v.add(cross(instQuat.xyz, cross(instQuat.xyz, v).add(v.mul(instQuat.w))).mul(2));
  // Unit geometry scaled to the instance's size, then rotated and moved into place.
  const scaledPosition = rotate(positionGeometry.mul(instSize)).add(instCenter);
  // A capsule cannot simply be scaled (its caps would stop being round): the unit
  // capsule's caps are moved to the ends of the straight part instead.
  const capY = positionGeometry.y.clamp(-1, 1);
  const capsuleLocal = vec3(
    positionGeometry.x.mul(instSize.x),
    positionGeometry.y.sub(capY).mul(instSize.x).add(capY.mul(instSize.y)),
    positionGeometry.z.mul(instSize.x),
  );
  const capsulePosition = rotate(capsuleLocal).add(instCenter);
  // Likewise a torus: the unit torus's ring is scaled to the ring radius and its
  // tube to the tube radius.
  const ringDir = vec2(positionGeometry.x, positionGeometry.z).normalize();
  const ringPoint = vec3(ringDir.x, 0, ringDir.y);
  const torusLocal = ringPoint.mul(instSize.x).add(positionGeometry.sub(ringPoint).mul(instSize.y.div(UNIT_TUBE)));
  const torusPosition = rotate(torusLocal).add(instCenter);
  // Normals: capsule and torus keep the unit shape's normals; the scaled shapes
  // need them divided by the scale to stay perpendicular to the surface.
  const plainNormal = rotate(normalLocal);
  const scaledNormal = rotate(normalLocal.div(instSize).normalize());

  // ---------------------------------------------------------- materials ---
  const causticProgram = wgslFn(programFor('CAUSTIC'));
  // eslint-disable-next-line @typescript-eslint/no-explicit-any
  const makeLampDepthMaterial = (position: any) => {
    const material = new THREE.NodeMaterial();
    material.positionNode = position;
    material.fragmentNode = vec4(positionWorld.dot(uSunDir).add(100), 0, 0, 1);
    material.side = THREE.DoubleSide;
    return material;
  };
  // eslint-disable-next-line @typescript-eslint/no-explicit-any
  const makeCausticMaterial = (position: any, normal: any) => {
    const material = new THREE.NodeMaterial();
    material.positionNode = position;
    material.fragmentNode = causticProgram({
      normalW: normal,
      worldPos: positionWorld,
      size: instSize,
      quat: instQuat,
      cmyAll: uCmy,
      vary: instVary,
      tintAll: uTint,
      tintOwn: instTint,
    });
    material.side = THREE.DoubleSide;
    material.depthTest = false;
    material.depthWrite = false;
    material.transparent = true;
    material.blending = THREE.CustomBlending;
    material.blendEquation = THREE.AddEquation;
    material.blendSrc = THREE.DstColorFactor; // result = source x destination
    material.blendDst = THREE.ZeroFactor;
    return material;
  };

  const groundMaterial = new THREE.NodeMaterial();
  groundMaterial.fragmentNode = wgslFn(programFor('GROUND'))({
    fragCoord: screenCoordinate,
    res: uRes,
    camPos: uCamPos,
    camRight: uCamRight,
    camUp: uCamUp,
    camFwd: uCamFwd,
    room: uRoom,
    skyE: uSkyE,
    sunE: uSunE,
    albedo: uAlbedo,
    lightC: uLightC,
    lightR: uLightR,
    lightU: uLightU,
    floorP: uFloor,
    spot: uSpot, lamp: uSunDir,
    lightTex: lightNode,
    lightSamp: sampler(lightNode),
    mirrorTex: mirrorNode,
    mirrorSamp: sampler(mirrorNode),
    shadeTex: shadeNode,
    shadeSamp: sampler(shadeNode),
    shadeFarTex: shadeFarNode,
    shadeFarSamp: sampler(shadeFarNode),
  });
  // (Its alpha is the distance of each pixel, so it is written as it is.)
  groundMaterial.transparent = true;
  groundMaterial.blending = THREE.NoBlending;
  groundMaterial.depthTest = false;
  groundMaterial.depthWrite = false;
  const groundQuad = new THREE.QuadMesh(groundMaterial);

  const blurMaterial = new THREE.NodeMaterial();
  blurMaterial.fragmentNode = wgslFn(programFor('BLUR'))({ uv: uv(), dir: uBlurDir, cut: uBlurCut, reach: uBlurReach, src: blurSource, srcSamp: sampler(blurSource) });
  blurMaterial.depthTest = blurMaterial.depthWrite = false;
  const blurQuad = new THREE.QuadMesh(blurMaterial);
  /**
   * Blur `from` into `to` (through a scratch target) by `radiusX` and `radiusY`, in
   * texture coordinates, keeping only what is brighter than `cut`.
   */
  const blur = (from: THREE.Texture, to: THREE.RenderTarget, radiusX: number, radiusY = radiusX, scratch = shadeScratch, cut = 0) => {
    // Samples no more than about a texel and a half apart, or they leave streaks:
    // up to 12 each side of the middle in one pass, and a wider blur as several
    // narrower ones in a row (which add up to the same blur).
    const texels = Math.max(radiusX * scratch.width, radiusY * scratch.height);
    const passes = Math.min(Math.max(Math.ceil((texels / 18) ** 2), 1), 5);
    const share = 1 / Math.sqrt(passes);
    uBlurReach.value = Math.min(Math.max(Math.ceil((texels * share) / 1.5), 3), 12);
    for (let pass = 0; pass < passes; pass++) {
      blurSource.value = pass === 0 ? from : to.texture;
      uBlurCut.value = pass === 0 ? cut : 0;
      uBlurDir.value.set(radiusX * share, 0);
      renderer.setRenderTarget(scratch);
      blurQuad.render(renderer);
      blurSource.value = scratch.texture;
      uBlurCut.value = 0;
      uBlurDir.value.set(0, radiusY * share);
      renderer.setRenderTarget(to);
      blurQuad.render(renderer);
    }
  };

  const glassSource = programFor('GLASS').replace('/*DIAMOND*/', diamondWgsl('IC_DIAMOND'));
  const photonSource = programFor('PHOTON').replace('/*DIAMOND*/', diamondWgsl('IC_DIAMOND'));
  // eslint-disable-next-line @typescript-eslint/no-explicit-any
  const makePhotonMaterial = (kind: number, position: any, normal: any) => {
    // Its output is data (which way each ray is thrown, negative as often as not, and
    // alpha too), so it goes out as the whole fragment, untouched: as the colour of
    // a basic material, three.js would clamp the negative half away.
    const material = new THREE.NodeMaterial();
    material.transparent = true;
    material.blending = THREE.NoBlending;
    material.positionNode = position;
    const program = wgslFn(photonSource.replace('const IC_SHAPE = 0;', `const IC_SHAPE = ${kind};`));
    material.fragmentNode = program({
      worldPos: positionWorld,
      nrmW: normal,
      center: instCenter,
      quat: instQuat,
      size: instSize,
      vary: instVary,
      tintAll: uTint,
      tintOwn: instTint,
      sunDir: uSunDir,
      lightR: uLightR,
      lightU: uLightU,
      glassP: uGlass,
      cmyAll: uCmy,
      bevel: uBevel,
      raysAcross: float(PHOTON_MAP_SIZE),
      meshEntry: uMeshEntry,
      sdfTex: volumeNode,
    });
    return material;
  };

  // The second half of real caustics. First the light that reaches the floor
  // without meeting any glass ...
  const photonNode = texture(photonMap.texture);
  const openMaterial = new THREE.NodeMaterial();
  openMaterial.fragmentNode = vec4(vec3(float(1).sub(photonNode.a.min(1))), 1);
  openMaterial.depthTest = openMaterial.depthWrite = false;
  const openQuad = new THREE.QuadMesh(openMaterial);
  // ... then every ray that went through glass, as a square one texel wide (wider
  // if the glass fanned it out: see icPhoton), moved to where it landed and added on.
  const rayIndex = float(instanceIndex);
  const rayRow = rayIndex.div(PHOTON_MAP_SIZE).floor();
  const rayAt = vec2(rayIndex.sub(rayRow.mul(PHOTON_MAP_SIZE)), rayRow).add(0.5).div(PHOTON_MAP_SIZE);
  const ray = texture(photonMap.texture, rayAt);
  const landMaterial = new THREE.NodeMaterial();
  landMaterial.vertexNode = vec4(
    vec2(rayAt.x.mul(2).sub(1), float(1).sub(rayAt.y.mul(2)))
      .add(ray.xy)
      .add(vec2(positionGeometry.x, positionGeometry.y).mul(ray.a.sub(1).max(1).mul(2 / PHOTON_MAP_SIZE))),
    0,
    1,
  );
  landMaterial.fragmentNode = vec4(vec3(varying(ray.z)), 1);
  landMaterial.transparent = true;
  landMaterial.blending = THREE.AdditiveBlending;
  landMaterial.depthTest = landMaterial.depthWrite = false;
  const landGeometry = new THREE.InstancedBufferGeometry();
  const square = new THREE.PlaneGeometry(1, 1);
  landGeometry.index = square.index;
  landGeometry.setAttribute('position', square.getAttribute('position'));
  landGeometry.instanceCount = PHOTON_MAP_SIZE * PHOTON_MAP_SIZE;
  const landMesh = new THREE.Mesh(landGeometry, landMaterial);
  landMesh.frustumCulled = false;
  const landScene = new THREE.Scene();
  landScene.add(landMesh);
  // eslint-disable-next-line @typescript-eslint/no-explicit-any
  const makeGlassMaterial = (kind: number, position: any, normal: any, mirrored = false) => {
    const material = new THREE.MeshBasicNodeMaterial();
    // Its alpha is data (distance, or how much a reflection has faded), so it has to
    // be written as it is: an opaque material would write 1.
    material.transparent = true;
    material.blending = THREE.NoBlending;
    // The mirror image: the same object drawn upside down below the floor, but
    // shaded as itself, seen from the mirror image of the camera.
    const upsideDown = vec3(1, -1, 1);
    material.positionNode = mirrored ? position.mul(upsideDown) : position;
    if (mirrored) {
      material.side = THREE.BackSide; // mirroring turns the triangles inside out
    }
    // One program per shape, so each only contains (and compiles) its own formulas.
    const program = wgslFn(glassSource.replace('const IC_SHAPE = 0;', `const IC_SHAPE = ${kind};`));
    // The WGSL function returns a vec4; the typings cannot see that.
    material.colorNode = program({
      worldPos: mirrored ? positionWorld.mul(upsideDown) : positionWorld,
      nrmW: normal,
      center: instCenter,
      quat: instQuat,
      size: instSize,
      vary: instVary,
      tintAll: uTint,
      tintOwn: instTint,
      camPos: mirrored ? uMirrorCam : uCamPos,
      glassP: uGlass,
      viewProj: uViewProj,
      room: uRoom,
      skyE: uSkyE,
      sunE: uSunE,
      albedo: uAlbedo,
      lightC: uLightC,
      lightR: uLightR,
      lightU: uLightU,
      cmyAll: uCmy,
      mirror: float(mirrored ? 1 : 0),
      bevel: uBevel,
      spot: uSpot, lamp: uSunDir,
      lightTex: lightNode,
      lightSamp: sampler(lightNode),
      behindTex: behindNode,
      behindSamp: sampler(behindNode),
      lampDepthTex: lampDepthNode,
      meshEntry: uMeshEntry,
      sdfTex: volumeNode,
    }) as unknown as THREE.MeshBasicNodeMaterial['colorNode'];
    return material;
  };

  // Wireframe: the edges of the triangles each object is made of, in one flat
  // colour. Its alpha is the distance from the camera, as the glass writes it.
  const uWireColor = uniform(new THREE.Color(0.1, 0.1, 0.1));
  const wireMaterials: THREE.MeshBasicNodeMaterial[] = [];
  // eslint-disable-next-line @typescript-eslint/no-explicit-any
  const makeWireMaterial = (position: any) => {
    const material = new THREE.MeshBasicNodeMaterial();
    material.wireframe = true;
    material.transparent = true;
    // Its colour is mixed over the picture by a set amount (its opacity, in the
    // blend constant), while its alpha, the distance, is written as it is.
    material.blending = THREE.CustomBlending;
    material.blendSrc = THREE.ConstantColorFactor;
    material.blendDst = THREE.OneMinusConstantColorFactor;
    material.blendSrcAlpha = THREE.OneFactor;
    material.blendDstAlpha = THREE.ZeroFactor;
    wireMaterials.push(material);
    material.positionNode = position;
    material.colorNode = vec4(uWireColor, positionWorld.sub(uCamPos).length()) as unknown as THREE.MeshBasicNodeMaterial['colorNode'];
    return material;
  };

  // A copy of a layer into the next one (alpha and all).
  const copySource = texture(background.texture);
  const copyMaterial = new THREE.NodeMaterial();
  copyMaterial.fragmentNode = copySource;
  copyMaterial.transparent = true;
  copyMaterial.blending = THREE.NoBlending;
  copyMaterial.depthTest = copyMaterial.depthWrite = false;
  const copyQuad = new THREE.QuadMesh(copyMaterial);
  // And one layer added onto another.
  const addSource = texture(gatheredSoft.texture);
  const uAddGain = uniform(1);
  const addMaterial = new THREE.NodeMaterial();
  addMaterial.fragmentNode = addSource.mul(uAddGain);
  addMaterial.transparent = true;
  addMaterial.blending = THREE.AdditiveBlending;
  addMaterial.depthTest = addMaterial.depthWrite = false;
  const addQuad = new THREE.QuadMesh(addMaterial);

  // The finishing passes.
  const shadeSource = texture(nearLayer.texture);
  const shadeMaterial = new THREE.NodeMaterial();
  shadeMaterial.fragmentNode = wgslFn(programFor('SHADE'))({
    fragCoord: screenCoordinate,
    res: uRes,
    camPos: uCamPos,
    camRight: uCamRight,
    camUp: uCamUp,
    camFwd: uCamFwd,
    p: uShade,
    src: shadeSource,
    srcSamp: sampler(shadeSource),
  });
  shadeMaterial.transparent = true;
  shadeMaterial.blending = THREE.NoBlending;
  shadeMaterial.depthTest = shadeMaterial.depthWrite = false;
  const shadeQuad = new THREE.QuadMesh(shadeMaterial);
  const lensSource = texture(shadedLayer.texture);
  const glowNode = texture(glowLayer.texture);
  // (A basic material, so that three.js converts its colour for the screen.)
  const lensMaterial = new THREE.MeshBasicNodeMaterial();
  lensMaterial.colorNode = (
    wgslFn(programFor('LENS'))({
      fragCoord: screenCoordinate,
      res: uRes,
      p: uLens,
      src: lensSource,
      srcSamp: sampler(lensSource),
      glowTex: glowNode,
      glowSamp: sampler(glowNode),
    }) as unknown as { rgb: THREE.MeshBasicNodeMaterial['colorNode'] }
  ).rgb;
  lensMaterial.depthTest = lensMaterial.depthWrite = false;
  const lensQuad = new THREE.QuadMesh(lensMaterial);

  // ------------------------------------------------------------- scenes ---
  // Unit geometry per shape, indexed by KIND.
  const flat = (g: THREE.BufferGeometry) => {
    const out = g.index ? g.toNonIndexed() : g;
    out.computeVertexNormals();
    return out;
  };
  // Triangular prism: apex at y = 1, base at y = -0.5, from z = -1 to 1.
  const triangle = new THREE.Shape([new THREE.Vector2(0, 1), new THREE.Vector2(-0.8660254, -0.5), new THREE.Vector2(0.8660254, -0.5)]);
  const bases: THREE.BufferGeometry[] = [
    new THREE.BoxGeometry(2, 2, 2),
    new THREE.IcosahedronGeometry(1, 4),
    new THREE.CylinderGeometry(1, 1, 2, 40, 1),
    new THREE.CapsuleGeometry(1, 2, 10, 32),
    new THREE.ConeGeometry(1, 2, 48, 1),
    new THREE.TorusGeometry(1, UNIT_TUBE, 20, 64).rotateX(Math.PI / 2),
    flat(new THREE.OctahedronGeometry(1)),
    flat(new THREE.ExtrudeGeometry(triangle, { depth: 2, bevelEnabled: false }).translate(0, 0, -1)),
    // Hexagon with corners on the X axis; 1 is the distance to its side faces.
    flat(new THREE.CylinderGeometry(1 / Math.cos(Math.PI / 6), 1 / Math.cos(Math.PI / 6), 2, 6, 1).rotateY(Math.PI / 2)),
    meshes.suzanne,
    diamondGeometry(),
    meshes.dragonLod,
  ];
  // The round shapes again, far finer: with only a few objects on show, each is
  // large on screen, and the corners of the coarse ones show on their outlines.
  // (Thousands of fine ones would cost too many triangles.)
  const fineBases = [...bases];
  fineBases[KIND.sphere] = new THREE.IcosahedronGeometry(1, 12);
  fineBases[KIND.cylinder] = new THREE.CylinderGeometry(1, 1, 2, 120, 1);
  fineBases[KIND.capsule] = new THREE.CapsuleGeometry(1, 2, 24, 96);
  fineBases[KIND.cone] = new THREE.ConeGeometry(1, 2, 120, 1);
  fineBases[KIND.torus] = new THREE.TorusGeometry(1, UNIT_TUBE, 40, 144).rotateX(Math.PI / 2);
  fineBases[KIND.dragon] = meshes.dragon; // the scan in full
  const farScene = new THREE.Scene();
  const nearScene = new THREE.Scene();
  const lightScene = new THREE.Scene();
  const lampDepthScene = new THREE.Scene();
  const mirrorScene = new THREE.Scene();
  const photonScene = new THREE.Scene();
  const far: Batch[] = [];
  const near: Batch[] = [];
  // The same objects drawn as glass, mirrored in the floor, or as wireframe.
  const glassMeshes: THREE.Mesh[] = [];
  const mirrorMeshes: THREE.Mesh[] = [];
  const wireMeshes: THREE.Mesh[] = [];
  let wireOn = false;
  const fineFar: Batch[] = [];
  const fineNear: Batch[] = [];
  for (let kind = 0; kind < KIND_COUNT; kind++) {
    const position = kind === KIND.capsule ? capsulePosition : kind === KIND.torus ? torusPosition : scaledPosition;
    const normal = kind === KIND.capsule || kind === KIND.torus ? plainNormal : scaledNormal;
    const glass = makeGlassMaterial(kind, position, normal);
    const caustic = makeCausticMaterial(position, normal);
    const lampDepthMaterial = makeLampDepthMaterial(position);
    const mirrored = makeGlassMaterial(kind, position, normal, true);
    const photon = makePhotonMaterial(kind, position, normal);
    const wire = makeWireMaterial(position);
    for (const [batches, scene, fine] of [[far, farScene, false], [near, nearScene, false], [fineFar, farScene, true], [fineNear, nearScene, true]] as const) {
      const batch = fine ? new Batch(fineBases[kind], FINE_UP_TO) : new Batch(bases[kind]);
      batches.push(batch);
      for (const [material, target] of [[glass, scene], [caustic, lightScene], [lampDepthMaterial, lampDepthScene], [mirrored, mirrorScene], [photon, photonScene], [wire, scene]] as const) {
        const m = new THREE.Mesh(batch.geometry, material);
        m.frustumCulled = false;
        target.add(m);
        if (material === glass) glassMeshes.push(m);
        if (material === mirrored) mirrorMeshes.push(m);
        if (material === wire) {
          m.visible = false;
          wireMeshes.push(m);
        }
      }
    }
  }
  const lightCamera = new THREE.OrthographicCamera(-1, 1, 1, -1, 0.1, 100);
  /** Triangles in one object of each shape, and how many were drawn last frame. */
  const triangleCount = (g: THREE.BufferGeometry) => (g.index ? g.index.count : g.getAttribute('position').count) / 3;
  const trianglesOf = bases.map(triangleCount);
  const fineTrianglesOf = fineBases.map(triangleCount);
  let lastTriangles = 0;

  // -------------------------------------------------------------- state ---
  let lightDirty = true;
  let lastSun = '';
  const depth = new Float32Array(MAX_OBJECTS);
  const slices = new Uint32Array(1024);
  const order = new Uint32Array(MAX_OBJECTS); // the objects, nearest first
  let lastView = '';
  const forward = new THREE.Vector3();
  const viewProj = new THREE.Matrix4();

  /**
   * Split the objects into a far half and a near half for this view. Only when
   * something changed: an object moved, or the camera did.
   */
  function sortIntoLayers(moved: boolean) {
    camera.getWorldDirection(forward);
    const c = camera.position;
    const n = objects.count;
    const view = `${n} ${c.x} ${c.y} ${c.z} ${forward.x} ${forward.y} ${forward.z}`;
    if (!moved && view === lastView) return;
    lastView = view;
    let nearest = Infinity;
    let farthest = -Infinity;
    for (let i = 0; i < n; i++) {
      const d = (objects.center[i * 3] - c.x) * forward.x + (objects.center[i * 3 + 1] - c.y) * forward.y + (objects.center[i * 3 + 2] - c.z) * forward.z;
      depth[i] = d;
      // (Read back, so the limits are rounded to 32 bits just as the stored depths are.)
      if (depth[i] < nearest) nearest = depth[i];
      if (depth[i] > farthest) farthest = depth[i];
    }
    // The depth that half of them are beyond, from a count of how many fall in each
    // of 1024 depth slices (much cheaper than sorting them all).
    slices.fill(0);
    const perSlice = 1023.999 / Math.max(farthest - nearest, 1e-6);
    for (let i = 0; i < n; i++) slices[((depth[i] - nearest) * perSlice) | 0] += 1;
    let middle = 0;
    for (let seen = 0; middle < 1024 && seen + slices[middle] <= n >> 1; middle++) seen += slices[middle];
    const split = nearest + middle / perSlice;
    // A few objects are drawn with the fine shapes, many with the coarse ones.
    const fine = n <= FINE_UP_TO;
    uMeshEntry.value = fine ? 1 : 0;
    const farSet = fine ? fineFar : far;
    const nearSet = fine ? fineNear : near;
    const all = [...far, ...near, ...fineFar, ...fineNear];
    for (const b of all) b.count = 0;
    // Nearest first, using the slices counted above (a counting sort, no more costly
    // than counting). The GPU skips shading any pixel already covered by something
    // nearer, so a deep pile is shaded about once per pixel, not once for every
    // object behind it, and looks exactly the same.
    for (let s = 0, start = 0; s < 1024; s++) {
      const inSlice = slices[s];
      slices[s] = start;
      start += inSlice;
    }
    for (let i = 0; i < n; i++) order[slices[((depth[i] - nearest) * perSlice) | 0]++] = i;
    for (let k = 0; k < n; k++) {
      const i = order[k];
      (depth[i] >= split ? farSet : nearSet)[objects.kind[i]].push(objects, i);
    }
    for (const b of all) b.commit();
  }

  /** Half-width of the square of ground the objects (and their shadows) cover. */
  function groundReach(): number {
    let reach = 0;
    // How far along the floor a shadow runs for each metre of height (twice over: an
    // object's top is about twice as high as its centre).
    const slope = 2 * Math.max(1, Math.hypot(sunDir.x, sunDir.z) / Math.max(sunDir.y, 0.1));
    for (let i = 0; i < objects.count; i++) {
      // Higher objects cast their shadows and caustics farther away, the more so the lower the lamp.
      const spread = Math.max(Math.abs(objects.center[i * 3]), Math.abs(objects.center[i * 3 + 2])) + objects.center[i * 3 + 1] * slope;
      if (spread > reach) reach = spread;
    }
    return Math.ceil((reach + 3) / 4) * 4; // in 4 m steps, so it does not shimmer
  }

  function resize(width: number, height: number) {
    background.setSize(width, height);
    farLayer.setSize(width, height);
    nearLayer.setSize(width, height);
    shadedLayer.setSize(width, height);
    glowScratch.setSize(Math.max(1, width >> 2), Math.max(1, height >> 2));
    glowLayer.setSize(Math.max(1, width >> 2), Math.max(1, height >> 2));
    mirrorLayer.setSize(Math.max(1, width >> 1), Math.max(1, height >> 1));
    uRes.value.set(width, height);
  }

  /** `moved`: some object changed since the last frame (redraws the caustic map). */
  function render(glass: { ior: number; dispersion: number; cmy?: number; bevel?: number; color?: [number, number, number, number]; wireframe?: boolean; wireColor?: [number, number, number, number] }, light: InstancedLighting, moved: boolean) {
    marks = timed ? [] : null;
    // Wireframe only: the edges in place of the glass, and nothing mirrored in the floor.
    const wantWire = glass.wireframe ?? false;
    if (wantWire !== wireOn) {
      wireOn = wantWire;
      for (const m of glassMeshes) m.visible = !wireOn;
      for (const m of mirrorMeshes) m.visible = !wireOn;
      for (const m of wireMeshes) m.visible = wireOn;
    }
    // The colour of the lines (linear rgb) and how opaque they are.
    const [wr, wg, wb, wa] = glass.wireColor ?? [0.01, 0.01, 0.01, 1];
    uWireColor.value.setRGB(wr, wg, wb);
    for (const m of wireMaterials) m.blendColor.setRGB(wa, wa, wa);
    uTint.value.fromArray(glass.color ?? [1, 1, 1, 1]);
    uBevel.value = glass.bevel ?? 0;
    if (uCmy.value !== (glass.cmy ?? 0)) lightDirty = true; // its shadow is coloured too
    uCmy.value = glass.cmy ?? 0;
    // Camera.
    camera.updateMatrixWorld();
    const e = camera.matrixWorld.elements;
    const tanHalf = Math.tan(THREE.MathUtils.degToRad(camera.fov) / 2);
    uCamPos.value.set(e[12], e[13], e[14]);
    uCamRight.value.set(e[0], e[1], e[2], camera.aspect);
    uCamUp.value.set(e[4], e[5], e[6]);
    uCamFwd.value.set(-e[8], -e[9], -e[10], tanHalf);
    viewProj.multiplyMatrices(camera.projectionMatrix, camera.matrixWorldInverse);
    uViewProj.value.copy(viewProj);

    // Lighting and glass settings.
    uRoom.value = (light.ground ? 1 : 0) + (light.dark ? 2 : 0);
    uSpot.value.set(light.spotRadius, light.spotRadius * 2.2, 0, 0);
    uSkyE.value.fromArray(light.skyE);
    uSunE.value.fromArray(light.sunE);
    uAlbedo.value.fromArray(light.groundColor);
    sunDir.fromArray(light.sunDir);
    const sunKey = light.sunDir.join() + light.shadowSoftness + light.causticSoftness + '/' + light.causticStrength + glass.ior + (glass.cmy ?? 0) + (glass.bevel ?? 0) + (glass.color ?? []).join() + String(light.realCaustics) + String(light.objectShadows);
    if (sunKey !== lastSun) {
      lastSun = sunKey;
      lightDirty = true;
    }
    if (moved) lightDirty = true;

    sortIntoLayers(moved || lightDirty);
    // Triangles drawn this frame: every object once, again for its mirror image,
    // and again when the shadow map is redrawn.
    let drawn = 0;
    for (let kind = 0; kind < KIND_COUNT; kind++) {
      drawn += trianglesOf[kind] * (far[kind].count + near[kind].count) + fineTrianglesOf[kind] * (fineFar[kind].count + fineNear[kind].count);
    }
    const passes = 1 + (light.ground && light.floorReflection > 0 ? 1 : 0) + (lightDirty ? 1 : 0);
    lastTriangles = drawn * passes + 2 * 4; // plus the full-screen passes

    // 1. Caustic map, from the main lamp's direction, covering everything.
    if (lightDirty) {
      lightDirty = false;
      const half = groundReach();
      const sun = sunDir;
      lightCamera.left = lightCamera.bottom = -half;
      lightCamera.right = lightCamera.top = half;
      lightCamera.near = 0.1;
      lightCamera.far = half * 6;
      lightCamera.position.copy(sun).multiplyScalar(half * 3);
      lightCamera.up.set(0, 0, -1);
      lightCamera.lookAt(0, 0, 0);
      lightCamera.updateProjectionMatrix();
      lightCamera.updateMatrixWorld();
      const m = lightCamera.matrixWorld.elements;
      uLightC.value.set(0, 0, 0);
      uLightR.value.set(m[0], m[1], m[2]).divideScalar(half);
      uLightU.value.set(m[4], m[5], m[6]).divideScalar(half);
      const previousClear = renderer.getClearColor(new THREE.Color());
      const previousAlpha = renderer.getClearAlpha();
      mark('Shadow map');
      renderer.setClearColor(0xffffff, 1);
      renderer.setRenderTarget(causticMap);
      renderer.render(lightScene, lightCamera);
      // ... and how far toward the lamp the objects reach, for their shadows on each other.
      renderer.setClearColor(0x000000, 0);
      renderer.setRenderTarget(lampDepth);
      // (Without object shadows it is left cleared: nothing stands in any object's way.)
      if (light.objectShadows) renderer.render(lampDepthScene, lightCamera);
      else renderer.clear();
      renderer.setClearColor(previousClear, previousAlpha);
      // Soft shadows: the map blurred a little, and again far wider.
      const perMetre = 0.5 / half; // texture coordinates per metre of floor
      blur(causticMap.texture, shadeNear, light.shadowSoftness * perMetre);
      blur(shadeNear.texture, shadeFar, Math.max(light.shadowSoftness * 3, 0.25) * perMetre);
      // Real caustics take the place of the near map: every ray of the lamp's light
      // followed to where it lands (see icPhoton), then added up there.
      if (light.realCaustics) {
        uGlass.value.set(glass.ior, glass.dispersion, 6, 0);
        mark('Caustics');
        renderer.setClearColor(0x000000, 0);
        renderer.setRenderTarget(photonMap);
        renderer.render(photonScene, lightCamera);
        // The shadow itself (the light that meets no glass), with its edge as soft
        // as the shadows are set to be ...
        renderer.setRenderTarget(openMap);
        openQuad.render(renderer);
        blur(openMap.texture, shadeNear, light.shadowSoftness * perMetre);
        // ... plus the light that came through the glass, gathered where it lands
        // and blurred as much as the lamp is wide: a small lamp throws sharp
        // caustics, a large soft one smooth pools of light.
        renderer.setClearColor(0x000000, 1);
        renderer.setRenderTarget(landedMap);
        renderer.render(landScene, lightCamera);
        renderer.setClearColor(previousClear, previousAlpha);
        blur(landedMap.texture, gatheredSoft, light.causticSoftness * perMetre);
        renderer.setRenderTarget(shadeNear);
        renderer.autoClear = false;
        uAddGain.value = light.causticStrength;
        addQuad.render(renderer);
        renderer.autoClear = true;
      }
    }

    // 2. The objects mirrored in the floor, if it is polished.
    // With thousands of objects each is small on screen, so the passes that are
    // seen least directly do less: the mirror image (blurred and faint anyway) and
    // the far half (seen through the near half) skip dispersion and follow the
    // light through fewer bounces.
    const many = objects.count > MANY_OBJECTS;
    uGlass.value.set(glass.ior, many ? 0 : glass.dispersion, many ? 3 : 4, 0);
    uFloor.value.x = light.ground ? light.floorReflection : 0;
    uFloor.value.y = light.realCaustics ? -light.shadowDarkness : light.shadowDarkness; // (negative: see icGroundColor)
    uFloor.value.z = light.floorGrid ? 1 : 0;
    uFloor.value.w = light.fog;
    if (uFloor.value.x > 0) {
      uMirrorCam.value.set(e[12], -e[13], e[14]);
      const previousClear = renderer.getClearColor(new THREE.Color());
      const previousAlpha = renderer.getClearAlpha();
      mark('Reflection');
      renderer.setClearColor(0x000000, 0);
      renderer.setRenderTarget(mirrorLayer);
      renderer.render(mirrorScene, camera);
      renderer.setClearColor(previousClear, previousAlpha);
    }

    // 3. Ground and sky.
    mark('Ground');
    renderer.setRenderTarget(background);
    groundQuad.render(renderer);

    // 4. Far half of the glass over a copy of the background.
    uGlass.value.set(glass.ior, many ? 0 : glass.dispersion, many ? 4 : 6, 0);
    mark('Far glass');
    renderer.setRenderTarget(farLayer);
    copySource.value = background.texture;
    copyQuad.render(renderer);
    renderer.autoClear = false;
    behindNode.value = background.texture;
    renderer.render(farScene, camera);

    // 5. Near half over a copy of that, seeing the far half refracted.
    renderer.autoClear = true;
    mark('Near glass');
    renderer.setRenderTarget(nearLayer);
    copySource.value = farLayer.texture;
    copyQuad.render(renderer);
    renderer.autoClear = false;
    // With glass showing through glass, the near half sees the far half: it is given
    // the picture with the far half in it, and looks for it along its refracted rays
    // (see icLook). Without, it is given the floor alone, as the far half was.
    uGlass.value.set(glass.ior, glass.dispersion, many ? 5 : 6, light.seeThrough ? (many ? 5 : 8) : 0);
    behindNode.value = light.seeThrough ? farLayer.texture : background.texture;
    renderer.render(nearScene, camera);
    renderer.autoClear = true;

    // 6. Finishing: contact shadows (and exposure) ...
    uShade.value.set(light.contactShadows, light.contactRadius, light.exposure, 0);
    mark('Finishing');
    renderer.setRenderTarget(shadedLayer);
    shadeQuad.render(renderer);
    // ... the glow of whatever is brighter than white, blurred wide ...
    if (light.glow > 0) {
      const aspect = uRes.value.x / uRes.value.y;
      blur(shadedLayer.texture, glowLayer, 0.012, 0.012 * aspect, glowScratch, 1);
      blur(glowLayer.texture, glowLayer, 0.03, 0.03 * aspect, glowScratch);
    }
    // ... and the camera: depth of field, glow and tones, onto the screen.
    uLens.value.set(light.focus, light.depthOfField, light.glow, light.filmic ? 1 : 0);
    renderer.setRenderTarget(null);
    lensQuad.render(renderer);
    if (marks) {
      markedFrames.set(renderer.info.frame, marks);
      for (const frame of markedFrames.keys()) if (frame < renderer.info.frame - 30) markedFrames.delete(frame);
    }
  }

  // ------------------------------------------------------------- timing ---
  // With the renderer set to keep GPU timestamps (?timing), how long each part of
  // the frame takes on the GPU. Every render call in a frame is numbered; before
  // each part, the number of the next call is noted with its name.
  const timed = (renderer.backend as unknown as { trackTimestamp?: boolean }).trackTimestamp === true;
  let marks: [number, string][] | null = null;
  const markedFrames = new Map<number, [number, string][]>();
  function mark(name: string) {
    if (marks) marks.push([renderer.info.render.frameCalls + 1, name]);
  }
  /** GPU time (ms) of each part of the latest frame timed, or null if not timing. */
  async function passTimes(): Promise<Record<string, number> | null> {
    if (!timed) return null;
    await renderer.resolveTimestampsAsync(THREE.TimestampQuery.RENDER);
    const pool = (renderer.backend as unknown as { timestampQueryPool: Record<string, { timestamps: Map<string, number> } | null> }).timestampQueryPool.render;
    if (!pool) return null;
    // The calls of each frame: number -> GPU time.
    const frames = new Map<number, Map<number, number>>();
    for (const [uid, ms] of pool.timestamps) {
      const match = uid.match(/^r:(\d+):.*:f(\d+)$/);
      if (!match) continue;
      const frame = Number(match[2]);
      if (!frames.has(frame)) frames.set(frame, new Map());
      frames.get(frame)!.set(Number(match[1]), ms);
    }
    const frame = [...frames.keys()].filter((f) => markedFrames.has(f)).sort((a, b) => b - a)[0];
    if (frame === undefined) return null;
    const frameMarks = markedFrames.get(frame)!;
    const out: Record<string, number> = {};
    for (const [call, ms] of frames.get(frame)!) {
      let name = 'Other';
      for (const [from, label] of frameMarks) if (call >= from) name = label;
      out[name] = (out[name] ?? 0) + ms;
    }
    return out;
  }

  return { render, resize, markChanged: () => (lightDirty = true), triangles: () => lastTriangles, passTimes };
}
