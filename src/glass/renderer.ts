// Real-time renderer for thousands of glass objects. The shaders are in tsl/, in
// TSL: three.js compiles them to WGSL for WebGPU or GLSL for WebGL 2.
//
// Per frame:
//   1. caustic map   every object drawn from the main lamp's direction, multiplying
//                    up how much of its light each surface lets through
//                    (only when something moved or the light changed)
//   2. background    ground (lit through the caustic map) and sky
//   3. far glass     the farther half of the objects, over the background
//   4. near glass    the nearer half, which sees the far half refracted
//   5. finishing     contact shadows, then depth of field, glow and tones
//   6. temporal AA   every frame is drawn with the camera nudged by a fraction of a
//                    pixel (and the wavelengths moved on) and merged into a history:
//                    while anything moves, the history is reprojected to where each
//                    surface was and clipped to the new frame's neighbourhood; once the
//                    view is still it becomes an exact running average, up to
//                    `ACCUMULATION_SAMPLES` frames. This settles the per-pixel aliasing
//                    of light bouncing inside the glass, the dispersion grain and edges.
//
// The layers carry each pixel's distance from the camera in their alpha, which is
// what the finishing passes work from.
import * as THREE from 'three/webgpu';
import { InstanceBatch } from './batch';
import { DIAMOND_CUTS, diamondGeometry, type DiamondCut } from './diamond';
import { createShapeGeometries, triangleCount, UNIT_TUBE } from './geometries';
import { MeshGlassMaterial } from './material';
import {
  Fn,
  abs,
  attribute,
  clamp,
  dot,
  floor,
  length,
  max,
  normalize,
  smoothstep,
  sqrt,
  cross,
  normalLocal,
  positionGeometry,
  positionWorld,
  screenCoordinate,
  select,
  texture,
  texture3D,
  uniform,
  vec2,
  vec3,
  float,
  mix,
  uv,
  vec4,
  varying,
  instanceIndex,
} from 'three/tsl';
import { KIND, MAX_OBJECTS, type GlassObjects } from './objects';
import { createPassTimer } from './timing';
import { SPECTRAL_FRAME_OFFSET, spectralBands } from './tsl/optics';
import { causticNode, photonNode as photonRayNode } from './tsl/caustics';
import { CMY_FACE_ORDERS, type CmyFaceOrder } from './tsl/cmy';
import { bloomPrefilterNode, blurNode, contactShadeNode, lensNode } from './tsl/finish';
import { glassNode } from './tsl/glass';
import { groundNode } from './tsl/ground';

const KIND_COUNT = 13;

export interface InstancedLighting {
  ground: boolean;
  /** The dark studio (black room, a few lamps) instead of the white one. */
  dark: boolean;
  /** Glass shows other glass through it, along the bent rays (off: only the floor and the studio). */
  seeThrough: boolean;
  /**
   * Real caustics: the lamp's light traced through the glass onto the floor, and how soft they
   * are (m). With real caustics on, shadows and caustics are softened together by the larger of
   * this and `shadowSoftness` (both stand for the lamp's size), so their edges meet exactly.
   */
  realCaustics: boolean;
  causticSoftness: number;
  /** How much of the light gathered by the glass reaches the floor (1: all of it, physically right). */
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
  /** Objects cast the lamp's shadow on each other (see lampVisible in tsl/glass.ts). */
  objectShadows: boolean;
  /** Floor surface linear rgba color (defaults to groundColor with alpha 1). */
  floorColor?: [number, number, number, number];
  /** A grid of lines drawn on the floor, its linear rgba color, and how thick the fog over it is (0: none). */
  floorGrid: boolean;
  floorGridColor?: [number, number, number, number];
  fog: number;
  groundColor: [number, number, number];
  exposure: number;
  /** Irradiance / pi on the ground: from the room all round, and from the main lamp. */
  skyE: [number, number, number];
  sunE: [number, number, number];
  /** Unit vector toward the main lamp. */
  sunDir: [number, number, number];
  /** Optional equirectangular HDRI environment texture (overrides procedural studio env). */
  envMap?: THREE.Texture | null;
  /** Intensity multiplier for `envMap` (default 1). */
  envMapIntensity?: number;
  /** CMY dichroic face-filter strength for box instances (0 = off). */
  cmy?: number;
  /** CMY face color permutation for box instances (default `'cym'`). */
  cmyOrder?: CmyFaceOrder;
  /** Diamond cut used for diamond instances (default `'round'`). */
  diamondCut?: DiamondCut;
  /** Edge bevel of faceted shapes (box, octahedron, prism, hex prism), in metres (default `0`). */
  bevel?: number;
  /** Blend strength for `MeshGlassMaterial.color` (0 = clear, 1 = full tint; default `1`). */
  colorAlpha?: number;
  /** Faint waviness of the glass surfaces (0: perfect glass, the default; 1: clearly hand-made). */
  imperfections?: number;
}

const CAUSTIC_MAP_SIZE = 1024;
/** The lamp's depth map, for shadows on the objects themselves. */
const LAMP_DEPTH_SIZE = 1024;
const SHADE_MAP_SIZE = 1024;
/** Wide ambient shadow blur runs at 1/4 the shade map width (16x fewer pixels). */
const SHADE_FAR_SIZE = 256;
/** Real caustics: this many rays of light each way (so its square in all). */
const PHOTON_MAP_SIZE = 768;
/** Up to this many objects, the round shapes are drawn with their fine meshes. */
const FINE_UP_TO = 64;
/** Above this many objects, the less visible passes are drawn more cheaply. */
const MANY_OBJECTS = 3000;
/** Frames the still view is averaged over before it is considered settled. */
const ACCUMULATION_SAMPLES = 32;
/** How many still frames the (reprojected) history counts for when motion stops. */
const TAA_HISTORY_WEIGHT = 8;
/** Length of the sub-pixel jitter cycle (Halton 2, 3). */
const JITTER_CYCLE = 16;

/** A full-size HDR target for frame accumulation (resized with the picture). */
function makeAccumTarget(): THREE.RenderTarget {
  return new THREE.RenderTarget(4, 4, {
    type: THREE.HalfFloatType,
    format: THREE.RGBAFormat,
    depthBuffer: false,
    // The history is read at reprojected, sub-pixel positions.
    minFilter: THREE.LinearFilter,
    magFilter: THREE.LinearFilter,
  });
}

/** Whether texture coordinates `q` (a vec2 node) lie on the screen. */
// eslint-disable-next-line @typescript-eslint/no-explicit-any
function onScreen(q: any) {
  return q.x.greaterThanEqual(0).and(q.x.lessThanEqual(1)).and(q.y.greaterThanEqual(0)).and(q.y.lessThanEqual(1));
}

/** The radical inverse of `index` in `base` (Halton sequence): a well-spread value in [0, 1). */
function halton(index: number, base: number): number {
  let f = 1;
  let r = 0;
  for (let i = index; i > 0; i = Math.floor(i / base)) {
    f /= base;
    r += f * (i % base);
  }
  return r;
}

/**
 * Create the multi-pass instanced studio glass renderer supporting up to `MAX_OBJECTS` shapes,
 * photon-traced caustics, floor reflections, half-res bilateral contact shadows, bloom, and DoF.
 *
 * @param renderer Initialized Three.js `WebGPURenderer` (WebGPU or WebGL 2 backend).
 * @param camera   Active scene perspective camera.
 * @param objects  Flat-array instance container read on each `render()` call.
 * @param meshes   Baked triangle meshes, and the stacked 128^3 SDF 3D texture for Suzanne, Dragon and the torus knot.
 */
export function createInstancedGlass(
  renderer: THREE.WebGPURenderer,
  camera: THREE.PerspectiveCamera,
  objects: GlassObjects,
  /**
   * The shapes made from meshes: Suzanne's mesh, the dragon's (in full, and
   * simplified for piles), and their baked distance volumes, one after the other in
   * a single 3D texture (Suzanne's, the dragon's, then the torus knot's;
   * see SDF_LAYER in tsl/sdf.ts).
   */
  meshes: {
    suzanne: THREE.BufferGeometry;
    dragon: THREE.BufferGeometry;
    dragonLod: THREE.BufferGeometry;
    volume: THREE.Data3DTexture;
  },
) {
  /* Render targets */
  const isMobile =
    typeof navigator !== 'undefined' &&
    (/Mobi|Android|iPhone|iPad|iPod/i.test(navigator.userAgent) ||
      (navigator.maxTouchPoints > 1 &&
        typeof window !== 'undefined' &&
        Math.min(window.innerWidth, window.innerHeight) < 900));
  const dpr = typeof window !== 'undefined' ? window.devicePixelRatio || 1 : 1;
  const causticMapSize = isMobile ? 512 : CAUSTIC_MAP_SIZE;
  const lampDepthSize = isMobile ? 512 : LAMP_DEPTH_SIZE;
  const shadeMapSize = isMobile ? 512 : SHADE_MAP_SIZE;
  const photonMapSize = isMobile ? 512 : PHOTON_MAP_SIZE;
  // High-DPI Retina displays already supersample edges; avoid costly 2x RGBA16F MSAA resolves on Apple Silicon & mobile.
  const msaaSamples = isMobile || dpr >= 1.5 ? 0 : 2;
  const causticMap = new THREE.RenderTarget(causticMapSize, causticMapSize, {
    type: THREE.HalfFloatType,
    format: THREE.RGBAFormat,
    depthBuffer: false,
    minFilter: THREE.LinearFilter,
    magFilter: THREE.LinearFilter,
  });
  // How far toward the lamp the nearest object reaches, seen from the lamp (red, plus
  // 100 so that 0, as cleared, means nothing there): the shadows objects cast on
  // each other (see lampVisible in tsl/glass.ts).
  const lampDepth = new THREE.RenderTarget(lampDepthSize, lampDepthSize, {
    type: THREE.HalfFloatType,
    format: THREE.RGBAFormat,
    depthBuffer: true,
    minFilter: THREE.NearestFilter,
    magFilter: THREE.NearestFilter,
  });
  // The caustic map blurred, for soft shadows: a little, and far wider.
  const makeShade = (size = shadeMapSize) =>
    new THREE.RenderTarget(size, size, {
      type: THREE.HalfFloatType,
      format: THREE.RGBAFormat,
      depthBuffer: false,
      minFilter: THREE.LinearFilter,
      magFilter: THREE.LinearFilter,
    });
  const shadeScratch = makeShade();
  const shadeFarScratch = makeShade(SHADE_FAR_SIZE);
  // Real caustics: where each ray of the lamp's light lands, and the rays added up there.
  const photonMap = new THREE.RenderTarget(photonMapSize, photonMapSize, {
    type: THREE.HalfFloatType,
    format: THREE.RGBAFormat,
    depthBuffer: true,
    minFilter: THREE.NearestFilter,
    magFilter: THREE.NearestFilter,
  });
  const landedMap = new THREE.RenderTarget(photonMapSize, photonMapSize, {
    type: THREE.HalfFloatType,
    format: THREE.RGBAFormat,
    depthBuffer: false,
    minFilter: THREE.LinearFilter,
    magFilter: THREE.LinearFilter,
  });
  const shadeNear = makeShade();
  const shadeFar = makeShade(SHADE_FAR_SIZE);
  // Real caustics, on the way to the near map: the plain shadow, and the gathered light blurred.
  const openMap = makeShade();
  const gatheredSoft = makeShade();
  const makeTarget = (depthBuffer: boolean) =>
    new THREE.RenderTarget(4, 4, {
      type: THREE.HalfFloatType,
      format: THREE.RGBAFormat,
      depthBuffer,
      // The layers that glass is drawn into are multisampled, like the canvas.
      samples: depthBuffer ? msaaSamples : 0,
      minFilter: THREE.LinearFilter,
      magFilter: THREE.LinearFilter,
    });
  const background = makeTarget(false); // ground and sky
  const farLayer = makeTarget(true); // background plus the farther half of the glass
  const nearLayer = makeTarget(true); // plus the nearer half: the whole picture
  // Half-resolution contact shadows (shade + depth), bilaterally upsampled in lensNode.
  const aoLayer = makeTarget(false);
  // The glow of the brightest glints, at a quarter of the size.
  const glowScratch = makeTarget(false);
  const glowLayer = makeTarget(false);
  // Blurred background/far layer for frosted glass transmission, at quarter size.
  const frostLayer = makeTarget(false);
  // The objects mirrored in the floor, at half size. Not multisampled: it is blurred.
  const mirrorLayer = new THREE.RenderTarget(4, 4, {
    type: THREE.HalfFloatType,
    format: THREE.RGBAFormat,
    depthBuffer: true,
    minFilter: THREE.LinearFilter,
    magFilter: THREE.LinearFilter,
  });

  /* Uniforms */
  const uRes = uniform(new THREE.Vector2(4, 4));
  const uAoRes = uniform(new THREE.Vector2(4, 4));
  // 1 when few enough objects are on show for light to enter a baked mesh through its volume (see entry in tsl/glass.ts).
  const uMeshEntry = uniform(0);
  const uCamPos = uniform(new THREE.Vector3());
  const uCamRight = uniform(new THREE.Vector4());
  const uCamUp = uniform(new THREE.Vector3());
  const uCamFwd = uniform(new THREE.Vector4());
  const uViewProj = uniform(new THREE.Matrix4());
  // This frame's sub-pixel camera jitter in texture coordinates (y down), for the passes
  // that rebuild a pixel's ray from its screen position.
  const uJitterUv = uniform(new THREE.Vector2());
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
  const uFinish = uniform(new THREE.Vector4(1, 0, 0.001, 0.001)); // exposure, aoActive (0/1), 1/aoW, 1/aoH
  const uBlurCut = uniform(0);
  const uBlurReach = uniform(6);
  const uGlowStep = uniform(new THREE.Vector2(0.001, 0.001)); // 1 / fullRes for 4x4 bilinear bloom prefilter
  const uTint = uniform(new THREE.Vector4(1, 1, 1, 1)); // the colour of all the glass (linear rgb), and how strongly it applies
  const uCmy = uniform(0); // above 0: boxes are CMY cubes, with filters this strong
  const uCmyOrder = uniform(new THREE.Vector3(0, 1, 2)); // CMY face color permutation (X, Y, Z -> 0=C, 1=Y, 2=M)
  const uBevel = uniform(0); // how much is cut off the edges of boxes (m)
  const uRoughness = uniform(0); // frosted glass roughness (0: clear, 1: frosted)
  const uFrostScale = uniform(4); // frosted glass grain density (higher = smaller particles)
  const uImperfection = uniform(0); // faint surface waviness (0: perfect glass)
  const uSpot = uniform(new THREE.Vector4()); // stage spotlight: beam radius at the floor (0: off), distance of the lamp
  const uFloor = uniform(new THREE.Vector4()); // x: how reflective the floor is; y: how dark its shadows are
  const uFloorColor = uniform(new THREE.Vector4(0.85, 0.85, 0.85, 1)); // floor linear rgb + alpha
  const uGridColor = uniform(new THREE.Vector4(0.033, 0.033, 0.033, 1)); // floor grid linear rgb + alpha
  const uBlurDir = uniform(new THREE.Vector2());
  const uMirrorCam = uniform(new THREE.Vector3()); // the camera, mirrored in the floor
  const uEnvP = uniform(new THREE.Vector2(0, 1)); // x: envMap enabled, y: intensity
  const fallbackEnvTex = new THREE.DataTexture(new Uint8Array([255, 255, 255, 255]), 1, 1);
  fallbackEnvTex.wrapS = THREE.RepeatWrapping;
  fallbackEnvTex.wrapT = THREE.ClampToEdgeWrapping;
  fallbackEnvTex.minFilter = THREE.LinearMipmapLinearFilter;
  fallbackEnvTex.magFilter = THREE.LinearFilter;
  fallbackEnvTex.generateMipmaps = true;
  fallbackEnvTex.needsUpdate = true;
  const envTexNode = texture(fallbackEnvTex);

  const lightNode = texture(causticMap.texture);
  const behindNode = texture(background.texture);
  const frostNode = texture(frostLayer.texture);
  const lampDepthNode = texture(lampDepth.texture);
  const mirrorNode = texture(mirrorLayer.texture);
  const shadeNode = texture(shadeNear.texture);
  const shadeFarNode = texture(shadeFar.texture);
  const blurSource = texture(causticMap.texture);
  const volumeNode = texture3D(meshes.volume);

  /* Per-instance nodes */
  const instCenter = attribute('instCenter', 'vec3');
  const instQuat = attribute('instQuat', 'vec4');
  const instSize = attribute('instSize', 'vec3');
  const instVary = attribute('instVary', 'vec3');
  const instTint = attribute('instTint', 'vec3');
  /** What every glass pass reads of the object being drawn, and of the glass settings. */
  const glassInputs = {
    center: instCenter,
    quat: instQuat,
    size: instSize,
    vary: instVary,
    tintOwn: instTint,
    tintAll: uTint,
    cmyAll: uCmy,
    cmyOrder: uCmyOrder,
    bevel: uBevel,
    meshEntry: uMeshEntry,
    glassP: uGlass,
    sdfTex: volumeNode,
    roughness: uRoughness,
    frostScale: uFrostScale,
    imperfection: uImperfection,
  };
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

  /* Pass materials */
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
    material.fragmentNode = causticNode(normal, positionWorld, glassInputs);
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
  // Headroom for gathered caustic light above the open floor's (see causticHeadroom in tsl/scene.ts).
  const uCausticHeadroom = uniform(2);
  const sceneInputs = {
    causticHeadroom: uCausticHeadroom,
    room: uRoom,
    skyE: uSkyE,
    sunE: uSunE,
    albedo: uAlbedo,
    lightC: uLightC,
    lightR: uLightR,
    lightU: uLightU,
    floorP: uFloor,
    floorColor: uFloorColor,
    gridColor: uGridColor,
    spot: uSpot,
    lamp: uSunDir,
    lightTex: lightNode,
    lightTexSize: causticMapSize,
    envTex: envTexNode,
    envP: uEnvP,
  };
  /** The camera as the passes over the whole picture see it. */
  const screenCamera = {
    fragCoord: screenCoordinate,
    res: uRes,
    camPos: uCamPos,
    camRight: uCamRight,
    camUp: uCamUp,
    camFwd: uCamFwd,
    jitterUv: uJitterUv,
  };
  groundMaterial.fragmentNode = groundNode(
    screenCamera,
    { ...sceneInputs, shadeTex: shadeNode, shadeFarTex: shadeFarNode },
    mirrorNode,
  );
  // (Its alpha is the distance of each pixel, so it is written as it is.)
  groundMaterial.transparent = true;
  groundMaterial.blending = THREE.NoBlending;
  groundMaterial.depthTest = false;
  groundMaterial.depthWrite = false;
  const groundQuad = new THREE.QuadMesh(groundMaterial);

  const blurMaterial = new THREE.NodeMaterial();
  blurMaterial.fragmentNode = blurNode(uv(), uBlurDir, uBlurCut, uBlurReach, blurSource);
  blurMaterial.depthTest = blurMaterial.depthWrite = false;
  const blurQuad = new THREE.QuadMesh(blurMaterial);
  /**
   * Blur `from` into `to` (through a scratch target) by `radiusX` and `radiusY`, in
   * texture coordinates, keeping only what is brighter than `cut`.
   */
  const blur = (
    from: THREE.Texture,
    to: THREE.RenderTarget,
    radiusX: number,
    radiusY = radiusX,
    scratch = shadeScratch,
    cut = 0,
    maxPasses = 4,
  ) => {
    // Each bilinear-paired tap covers 2 texels (up to 12 pairs = 25 fetches per pass).
    const texels = Math.max(radiusX * scratch.width, radiusY * scratch.height);
    const passes = Math.min(Math.max(Math.ceil((texels / 22) ** 2), 1), maxPasses);
    const share = 1 / Math.sqrt(passes);
    // Taps cover the kernel out to 3 sigma (1.5 times `radius`, which is 2 sigma), about a
    // texel apart: farther apart, sharp content (caustic splats) breaks up into a grid.
    uBlurReach.value = Math.min(Math.max(Math.ceil((texels * share * 1.5) / 2.2), 2), 12);
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

  // Real caustics are traced once per wavelength band: x its Cauchy IOR offset, yzw the
  // linear-sRGB share of the light it carries (the bands' shares add up to white).
  const uPhotonBand = uniform(new THREE.Vector4(0, 1, 1, 1));
  // eslint-disable-next-line @typescript-eslint/no-explicit-any
  const makePhotonMaterial = (kind: number, position: any, normal: any, diamondPlanes?: number[]) => {
    // Its output is data (which way each ray is thrown, negative as often as not, and
    // alpha too), so it goes out as the whole fragment, untouched: as the colour of
    // a basic material, three.js would clamp the negative half away.
    const material = new THREE.NodeMaterial();
    material.transparent = true;
    material.blending = THREE.NoBlending;
    material.positionNode = position;
    material.fragmentNode = photonRayNode(
      kind,
      positionWorld,
      normal,
      glassInputs,
      uSunDir,
      uLightR,
      uLightU,
      diamondPlanes,
      uPhotonBand,
    );
    return material;
  };

  // The second half of real caustics. First the light that reaches the floor
  // without meeting any glass ...
  const photonNode = texture(photonMap.texture);
  const openMaterial = new THREE.NodeMaterial();
  openMaterial.fragmentNode = vec4(vec3(float(1).sub(photonNode.a.min(1))), 1);
  openMaterial.depthTest = openMaterial.depthWrite = false;
  const openQuad = new THREE.QuadMesh(openMaterial);
  // ... then every ray that went through glass, spread over the patch of floor it
  // lights and added on. Each ray stands for one texel of the lamp's view; the glass
  // bends that texel into a parallelogram on the floor, spanned by where the rays on
  // either side of it land. Its light is shared out over that area, so the patches of
  // neighbouring rays tile the floor without gaps or overlaps (a fixed square per ray
  // left a grid of gaps wherever the glass stretched the light one way).
  const rayIndex = float(instanceIndex);
  const rayRow = rayIndex.div(photonMapSize).floor();
  const rayTexel = vec2(rayIndex.sub(rayRow.mul(photonMapSize)), rayRow);
  const texelStep = 1 / photonMapSize;
  /** Where the ray of the texel `offset` texels away lands (lamp clip units), and whether it does. */
  const landingOf = (offset: [number, number]) => {
    const at = rayTexel.add(vec2(offset[0], offset[1])).add(0.5).mul(texelStep);
    const r = texture(photonMap.texture, at.clamp(float(0), float(1))).level(float(0));
    const inside = at.x.greaterThan(0).and(at.x.lessThan(1)).and(at.y.greaterThan(0)).and(at.y.lessThan(1));
    return {
      pos: vec2(at.x.mul(2).sub(1), float(1).sub(at.y.mul(2))).add(r.xy),
      lands: inside.and(r.z.greaterThan(1e-5)),
      light: r.z,
    };
  };
  const self = landingOf([0, 0]);
  /** Footprint edge along one axis of the lamp map: central difference, one-sided at an edge, else one texel. */
  const footprintEdge = (axis: [number, number]) => {
    const ahead = landingOf(axis);
    const behind = landingOf([-axis[0], -axis[1]]);
    // A neighbour that lands far away went round the other side of the object (or
    // through another surface): not part of the same smooth bundle of rays.
    const reach = 24 * 2 * texelStep;
    const aheadOk = ahead.lands.and(ahead.pos.sub(self.pos).length().lessThan(reach));
    const behindOk = behind.lands.and(behind.pos.sub(self.pos).length().lessThan(reach));
    const oneTexel = vec2(axis[0] * 2 * texelStep, -axis[1] * 2 * texelStep);
    return select(
      aheadOk.and(behindOk),
      ahead.pos.sub(behind.pos).mul(0.5),
      select(aheadOk, ahead.pos.sub(self.pos), select(behindOk, self.pos.sub(behind.pos), oneTexel)),
    );
  };
  const edgeX = footprintEdge([1, 0]).toVar();
  const edgeY = footprintEdge([0, 1]).toVar();
  // One clip unit of the lamp map is photonMapSize / 2 texels.
  const texelsPerUnit = photonMapSize / 2;
  // Each ray's light is spread by a bilinear "hat" over its own footprint: twice as wide
  // along each footprint edge and falling linearly to 0, so the hats of neighbouring rays
  // (one footprint apart) add up to a perfectly smooth spread, whether the footprint is
  // small, large, or long and thin. (Hard-edged shapes snapped to whole texels, leaving a
  // grid of squares; one square per ray turned long, thin footprints into blocks.) Edges
  // shorter than a texel are stretched to one, so no light falls between texel centres.
  // eslint-disable-next-line @typescript-eslint/no-explicit-any
  const atLeastOneTexel = (edge: any, fallback: any) => {
    const texels = length(edge).mul(texelsPerUnit);
    return select(texels.lessThan(1e-4), fallback, edge.mul(max(float(1).div(max(texels, 1e-4)), 1)));
  };
  /* eslint-disable @typescript-eslint/no-explicit-any -- select() loses the vector types */
  const hatX: any = atLeastOneTexel(edgeX, vec2(2 * texelStep, 0)).toVar();
  const hatY: any = atLeastOneTexel(edgeY, vec2(0, -2 * texelStep)).toVar();
  const hatArea = abs(hatX.x.mul(hatY.y).sub(hatX.y.mul(hatY.x))).mul(texelsPerUnit * texelsPerUnit);
  // Edges (nearly) along one line, where a fold of the caustic squeezes the footprint flat:
  // a one-texel hat instead.
  const flat = hatArea.lessThan(0.5);
  const corner = vec2(positionGeometry.x, positionGeometry.y);
  const spanX: any = select(flat, vec2(2 * texelStep, 0), hatX);
  const spanY: any = select(flat, vec2(0, -2 * texelStep), hatY);
  /* eslint-enable @typescript-eslint/no-explicit-any */
  const landPos = self.pos.add(spanX.mul(corner.x.mul(2)).add(spanY.mul(corner.y.mul(2))));
  // The hat over (-1..1)^2 of the footprint integrates to the footprint's area.
  const landLight = self.light.div(select(flat, float(1), hatArea));
  // Where in the hat this corner is (-1 to 1 along each edge).
  const tentAt = varying(corner.mul(2));
  const tentWeight = float(1)
    .sub(abs(tentAt.x))
    .mul(float(1).sub(abs(tentAt.y)))
    .max(0);
  const landMaterial = new THREE.NodeMaterial();
  // Cull empty photon rays (no glass hit or zero carried light) offscreen before rasterization.
  landMaterial.vertexNode = vec4(select(self.lands, landPos.x, 2), select(self.lands, landPos.y, 2), 0, 1);
  landMaterial.fragmentNode = vec4(uPhotonBand.yzw.mul(varying(landLight).mul(tentWeight)), 1);
  // A footprint's edges can wind either way (the lamp map's rows run down, clip space up,
  // and a caustic can fold the light over): draw both sides, or many would be culled.
  landMaterial.side = THREE.DoubleSide;
  landMaterial.transparent = true;
  landMaterial.blending = THREE.AdditiveBlending;
  landMaterial.depthTest = landMaterial.depthWrite = false;
  const landGeometry = new THREE.InstancedBufferGeometry();
  const square = new THREE.PlaneGeometry(1, 1);
  landGeometry.index = square.index;
  landGeometry.setAttribute('position', square.getAttribute('position'));
  landGeometry.instanceCount = photonMapSize * photonMapSize;
  const landMesh = new THREE.Mesh(landGeometry, landMaterial);
  landMesh.frustumCulled = false;
  const landScene = new THREE.Scene();
  landScene.add(landMesh);
  // eslint-disable-next-line @typescript-eslint/no-explicit-any
  const makeGlassMaterial = (kind: number, position: any, normal: any, mirrored = false, diamondPlanes?: number[]) => {
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
    const worldPos = mirrored ? positionWorld.mul(upsideDown) : positionWorld;
    material.colorNode = glassNode(
      kind,
      mirrored,
      worldPos,
      normal,
      glassInputs,
      sceneInputs,
      {
        camPos: mirrored ? uMirrorCam : uCamPos,
        viewProj: uViewProj,
        behindTex: behindNode,
        behindBlurTex: frostNode,
        lampDepthTex: lampDepthNode,
        lampDepthSize,
      },
      diamondPlanes,
    );
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
    material.colorNode = vec4(
      uWireColor,
      positionWorld.sub(uCamPos).length(),
    ) as unknown as THREE.MeshBasicNodeMaterial['colorNode'];
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
  // Colour only, alpha 1: additive blending multiplies the colour by the source alpha, and
  // the gathered-light map's alpha counts the splats that touched each texel (4 or more in a
  // pool), which multiplied the caustics' light several times over.
  addMaterial.fragmentNode = vec4(addSource.rgb.mul(uAddGain), 1);
  addMaterial.transparent = true;
  addMaterial.blending = THREE.AdditiveBlending;
  addMaterial.depthTest = addMaterial.depthWrite = false;
  const addQuad = new THREE.QuadMesh(addMaterial);

  // The finishing passes.
  const shadeSource = texture(nearLayer.texture);
  const aoCamera = {
    fragCoord: screenCoordinate,
    res: uAoRes,
    camPos: uCamPos,
    camRight: uCamRight,
    camUp: uCamUp,
    camFwd: uCamFwd,
    jitterUv: uJitterUv,
  };
  const shadeMaterial = new THREE.NodeMaterial();
  shadeMaterial.fragmentNode = contactShadeNode(aoCamera, uShade, shadeSource, isMobile ? 6 : 8);
  shadeMaterial.transparent = true;
  shadeMaterial.blending = THREE.NoBlending;
  shadeMaterial.depthTest = shadeMaterial.depthWrite = false;
  const shadeQuad = new THREE.QuadMesh(shadeMaterial);
  const glowPrefilterMaterial = new THREE.NodeMaterial();
  glowPrefilterMaterial.fragmentNode = bloomPrefilterNode(uv(), uGlowStep, uBlurCut, float(6), shadeSource);
  glowPrefilterMaterial.depthTest = glowPrefilterMaterial.depthWrite = false;
  const glowPrefilterQuad = new THREE.QuadMesh(glowPrefilterMaterial);
  const aoNode = texture(aoLayer.texture);
  const glowNode = texture(glowLayer.texture);
  // (A basic material, so that three.js converts its colour for the screen.)
  const lensMaterial = new THREE.MeshBasicNodeMaterial();
  lensMaterial.colorNode = (
    lensNode(screenCoordinate, uRes, uLens, uFinish, shadeSource, glowNode, aoNode) as {
      rgb: THREE.MeshBasicNodeMaterial['colorNode'];
    }
  ).rgb;
  lensMaterial.depthTest = lensMaterial.depthWrite = false;
  const lensQuad = new THREE.QuadMesh(lensMaterial);

  // Temporal AA: the finished frame (linear, before the screen's conversion), the history
  // (two, written in turn), and the copy of it to the screen.
  const frameTarget = makeAccumTarget();
  const historyTargets = [makeAccumTarget(), makeAccumTarget()];
  let historyRead = 0;
  /** Whether the history holds a picture of this scene at this size. */
  let historyValid = false;
  /** Still frames the history amounts to (motion leaves it at TAA_HISTORY_WEIGHT). */
  let stillCount = 0;
  let frameCounter = 0;
  let lastStateKey = '';
  const viewProjNow = new THREE.Matrix4();
  const viewProjLast = new THREE.Matrix4();
  const probeNow = new THREE.Vector3();
  const probeLast = new THREE.Vector3();
  /**
   * Largest on-screen movement (px) since the last frame of a few points of the scene: the
   * first objects' centres and a point straight ahead of the camera.
   */
  const screenMotionPx = () => {
    let most = 0;
    const measure = (x: number, y: number, z: number) => {
      probeNow.set(x, y, z).applyMatrix4(viewProjNow);
      probeLast.set(x, y, z).applyMatrix4(viewProjLast);
      const dx = (probeNow.x - probeLast.x) * 0.5 * uRes.value.x;
      const dy = (probeNow.y - probeLast.y) * 0.5 * uRes.value.y;
      most = Math.max(most, Math.hypot(dx, dy));
    };
    const step = Math.max(1, Math.floor(objects.count / 16));
    for (let i = 0; i < objects.count; i += step) {
      measure(objects.center[i * 3], objects.center[i * 3 + 1], objects.center[i * 3 + 2]);
    }
    const ahead = camera.getWorldDirection(probeNow).multiplyScalar(3).add(camera.position);
    measure(ahead.x, ahead.y, ahead.z);
    return most;
  };
  const taaHistory = texture(historyTargets[0].texture);
  const taaFrame = texture(frameTarget.texture);
  const taaSurface = texture(nearLayer.texture); // alpha: distance to what the pixel shows
  const taaBehind = texture(background.texture); // alpha: distance to the floor / backdrop behind it
  const uPrevViewProj = uniform(new THREE.Matrix4());
  const uTaaStillWeight = uniform(1); // 1 / (n + 1) while still
  const uTaaMoving = uniform(0); // 1: reproject, clip and blend for motion
  const taaMaterial = new THREE.NodeMaterial();
  /** One bilinear history read (for the Catmull-Rom filter below). */
  // eslint-disable-next-line @typescript-eslint/no-explicit-any
  const tap = (x: any, y: any) => taaHistory.sample(vec2(x, y)).level(float(0)).rgb;
  taaMaterial.fragmentNode = Fn(() => {
    const uvHere = screenCoordinate.xy.div(uRes).toVar();
    const texel = vec2(1).div(uRes).toVar();
    const current = taaFrame.sample(uvHere).level(float(0)).rgb.toVar();

    // The history lives on the unjittered pixel grid: follow the ray through this pixel's
    // centre (not the jittered sample, which would read the history up to half a pixel off
    // and soften it a little more every frame) to where a point at distance d along it was
    // on the previous frame's screen.
    const ndc = uvHere.mul(2).sub(1).mul(vec2(1, -1)).toVar();
    const ray = normalize(
      uCamFwd.xyz.add(uCamRight.xyz.mul(ndc.x.mul(uCamFwd.w).mul(uCamRight.w))).add(uCamUp.mul(ndc.y.mul(uCamFwd.w))),
    ).toVar();
    // eslint-disable-next-line @typescript-eslint/no-explicit-any
    const previousUv = (dist: any) => {
      const clip = uPrevViewProj.mul(vec4(uCamPos.add(ray.mul(dist)), 1)).toVar();
      const prevNdc = clip.xy.div(max(clip.w, 1e-6));
      return vec2(prevNdc.x.mul(0.5).add(0.5), float(0.5).sub(prevNdc.y.mul(0.5))).toVar();
    };
    /**
     * Catmull-Rom history read in 5 bilinear taps: the history is resampled at a sub-pixel
     * offset every moving frame, and plain bilinear reads would soften it each time.
     */
    // eslint-disable-next-line @typescript-eslint/no-explicit-any
    const historyAt = (q: any) => {
      const samplePos = q.mul(uRes);
      const tex1 = floor(samplePos.sub(0.5)).add(0.5);
      const f = samplePos.sub(tex1);
      const w0 = f.mul(f.mul(f.mul(-0.5).add(1)).sub(0.5));
      const w1 = f.mul(f).mul(f.mul(1.5).sub(2.5)).add(1);
      const w2 = f.mul(f.mul(f.mul(-1.5).add(2)).add(0.5));
      const w3 = f.mul(f).mul(f.mul(0.5).sub(0.5));
      const w12 = w1.add(w2);
      const tex0 = tex1.sub(1).mul(texel);
      const tex3 = tex1.add(2).mul(texel);
      const tex12 = tex1.add(w2.div(w12)).mul(texel);
      const c0 = w12.x.mul(w0.y);
      const c1 = w0.x.mul(w12.y);
      const c2 = w12.x.mul(w12.y);
      const c3 = w3.x.mul(w12.y);
      const c4 = w12.x.mul(w3.y);
      return max(
        tap(tex12.x, tex0.y)
          .mul(c0)
          .add(tap(tex0.x, tex12.y).mul(c1))
          .add(tap(tex12.x, tex12.y).mul(c2))
          .add(tap(tex3.x, tex12.y).mul(c3))
          .add(tap(tex12.x, tex3.y).mul(c4))
          .div(c0.add(c1).add(c2).add(c3).add(c4)),
        vec3(0),
      ).toVar();
    };

    const surfaceDist = taaSurface.sample(uvHere).level(float(0)).a.toVar();
    const behindDist = taaBehind.sample(uvHere).level(float(0)).a.toVar();
    const prevSurface = previousUv(surfaceDist);
    const historySurface = historyAt(prevSurface);
    // Glass: what is seen through it moves like the world behind it, not like its surface.
    const isGlass = surfaceDist.lessThan(behindDist.sub(1e-3)).toVar();
    const prevBehind = previousUv(behindDist);
    const historyBehind = historyAt(prevBehind);

    // The current 3x3 neighbourhood's colour range, to clip stale history to.
    // eslint-disable-next-line @typescript-eslint/no-explicit-any
    let m1: any = vec3(0);
    // eslint-disable-next-line @typescript-eslint/no-explicit-any
    let m2: any = vec3(0);
    for (let y = -1; y <= 1; y++) {
      for (let x = -1; x <= 1; x++) {
        const c = taaFrame.sample(uvHere.add(texel.mul(vec2(x, y)))).level(float(0)).rgb;
        m1 = m1.add(c);
        m2 = m2.add(c.mul(c));
      }
    }
    const mean = m1.div(9).toVar();
    const sigma = sqrt(max(m2.div(9).sub(mean.mul(mean)), vec3(0))).toVar();
    const luma = vec3(0.2126, 0.7152, 0.0722);
    const useBehind = isGlass
      .and(dot(historyBehind.sub(current).abs(), luma).lessThan(dot(historySurface.sub(current).abs(), luma)))
      .and(onScreen(prevBehind));
    const history = select(useBehind, historyBehind, historySurface).toVar();
    const prevUv = select(useBehind, prevBehind, prevSurface);
    // On-screen speed (px per frame). The glide after letting go moves well under a pixel a
    // frame, where the history is still nearly exact: glass only turns strict as speed builds.
    const motionPx = length(prevSurface.sub(uvHere).mul(uRes)).toVar();
    const strict = smoothstep(0.25, 2.0, motionPx).toVar();
    const gamma = select(isGlass, mix(1.25, 0.75, strict), float(1.25));
    const clipped = clamp(history, mean.sub(sigma.mul(gamma)), mean.add(sigma.mul(gamma)));
    // Slow motion keeps more history (less noise), fast motion takes more of the new frame.
    const speedAlpha = mix(0.08, 0.25, clamp(motionPx.div(16), 0, 1)).toVar();
    const glassAlpha = mix(speedAlpha, max(speedAlpha, 0.3), strict);
    const moving = uTaaMoving.greaterThan(0.5);
    const blend = select(
      onScreen(prevUv),
      select(moving, select(isGlass, glassAlpha, speedAlpha), uTaaStillWeight),
      float(1),
    );
    return vec4(mix(select(moving, clipped, history), current, blend), 1);
  })();
  taaMaterial.transparent = true;
  taaMaterial.blending = THREE.NoBlending;
  taaMaterial.depthTest = taaMaterial.depthWrite = false;
  const taaQuad = new THREE.QuadMesh(taaMaterial);
  const presentSource = texture(historyTargets[0].texture);
  // (A basic material, as the lens pass was, so three.js converts its colour for the screen.)
  const presentMaterial = new THREE.MeshBasicNodeMaterial();
  presentMaterial.colorNode = presentSource.rgb as unknown as THREE.MeshBasicNodeMaterial['colorNode'];
  presentMaterial.depthTest = presentMaterial.depthWrite = false;
  const presentQuad = new THREE.QuadMesh(presentMaterial);
  const present = () => {
    presentSource.value = historyTargets[historyRead].texture;
    renderer.setRenderTarget(null);
    presentQuad.render(renderer);
  };

  /* Instanced scenes */
  const { bases, fineBases } = createShapeGeometries(meshes);
  const farScene = new THREE.Scene();
  const nearScene = new THREE.Scene();
  const lightScene = new THREE.Scene();
  const lampDepthScene = new THREE.Scene();
  const mirrorScene = new THREE.Scene();
  const photonScene = new THREE.Scene();
  const far: InstanceBatch[] = [];
  const near: InstanceBatch[] = [];
  // The same objects drawn as glass, mirrored in the floor, or as wireframe.
  /**
   * Every mesh, with its batch and what it draws. A mesh is only visible while its
   * batch holds objects: three.js compiles a program for whatever is visible, and
   * with every shape's programs compiled up front the first frame took seconds on
   * WebGPU and over a minute on WebGL. A shape is compiled when it first appears.
   */
  const drawn: { mesh: THREE.Mesh; batch: InstanceBatch; role: 'glass' | 'wire' | 'other' }[] = [];
  let wireOn = false;
  const updateVisibility = () => {
    for (const { mesh, batch, role } of drawn) {
      // Wireframe only: the edges in place of the glass, and nothing mirrored in the floor.
      mesh.visible = batch.count > 0 && (role === 'other' || (role === 'wire') === wireOn);
    }
  };
  const fineFar: InstanceBatch[] = [];
  const fineNear: InstanceBatch[] = [];
  type DiamondCutEntry = {
    geometry: THREE.BufferGeometry;
    glass: THREE.MeshBasicNodeMaterial;
    mirrored: THREE.MeshBasicNodeMaterial;
    photon: THREE.NodeMaterial;
  };
  const diamondCutCache: Partial<Record<DiamondCut, DiamondCutEntry>> = {};
  const diamondGlassMeshes: THREE.Mesh[] = [];
  const diamondMirrorMeshes: THREE.Mesh[] = [];
  const diamondPhotonMeshes: THREE.Mesh[] = [];
  let currentDiamondCut: DiamondCut = 'round';
  /** Each shape's meshes (all passes, coarse and fine), with the scene each is drawn in. */
  const kindMeshes: { mesh: THREE.Mesh; scene: THREE.Scene }[][] = Array.from({ length: KIND_COUNT }, () => []);
  for (let kind = 0; kind < KIND_COUNT; kind++) {
    const position = kind === KIND.capsule ? capsulePosition : kind === KIND.torus ? torusPosition : scaledPosition;
    const normal = kind === KIND.capsule || kind === KIND.torus ? plainNormal : scaledNormal;
    const glass = makeGlassMaterial(kind, position, normal);
    const caustic = makeCausticMaterial(position, normal);
    const lampDepthMaterial = makeLampDepthMaterial(position);
    const mirrored = makeGlassMaterial(kind, position, normal, true);
    const photon = makePhotonMaterial(kind, position, normal);
    const wire = makeWireMaterial(position);
    if (kind === KIND.diamond) diamondCutCache.round = { geometry: bases[kind], glass, mirrored, photon };
    for (const [batches, scene, fine] of [
      [far, farScene, false],
      [near, nearScene, false],
      [fineFar, farScene, true],
      [fineNear, nearScene, true],
    ] as const) {
      const batch = fine ? new InstanceBatch(fineBases[kind], FINE_UP_TO) : new InstanceBatch(bases[kind]);
      batches.push(batch);
      for (const [material, target] of [
        [glass, scene],
        [caustic, lightScene],
        [lampDepthMaterial, lampDepthScene],
        [mirrored, mirrorScene],
        [photon, photonScene],
        [wire, scene],
      ] as const) {
        const m = new THREE.Mesh(batch.geometry, material);
        m.frustumCulled = false;
        m.visible = false;
        target.add(m);
        // (Wireframe lines are cheap and rarely shown: compiled when first drawn.)
        if (material !== wire) kindMeshes[kind].push({ mesh: m, scene: target });
        drawn.push({
          mesh: m,
          batch,
          role: material === glass || material === mirrored ? 'glass' : material === wire ? 'wire' : 'other',
        });
        if (kind === KIND.diamond) {
          if (material === glass) diamondGlassMeshes.push(m);
          else if (material === mirrored) diamondMirrorMeshes.push(m);
          else if (material === photon) diamondPhotonMeshes.push(m);
        }
      }
    }
  }
  const lightCamera = new THREE.OrthographicCamera(-1, 1, 1, -1, 0.1, 100);
  const trianglesOf = bases.map(triangleCount);
  const fineTrianglesOf = fineBases.map(triangleCount);
  const setDiamondCut = (cut: DiamondCut) => {
    if (cut === currentDiamondCut) return;
    let entry = diamondCutCache[cut];
    if (!entry) {
      const planes = DIAMOND_CUTS[cut].planes;
      entry = {
        geometry: diamondGeometry(cut),
        glass: makeGlassMaterial(KIND.diamond, scaledPosition, scaledNormal, false, planes),
        mirrored: makeGlassMaterial(KIND.diamond, scaledPosition, scaledNormal, true, planes),
        photon: makePhotonMaterial(KIND.diamond, scaledPosition, scaledNormal, planes),
      };
      diamondCutCache[cut] = entry;
    }
    for (const batch of [far[KIND.diamond], near[KIND.diamond], fineFar[KIND.diamond], fineNear[KIND.diamond]]) {
      batch.geometry.index = entry.geometry.index;
      batch.geometry.setAttribute('position', entry.geometry.getAttribute('position'));
      batch.geometry.setAttribute('normal', entry.geometry.getAttribute('normal'));
    }
    const tris = triangleCount(entry.geometry);
    trianglesOf[KIND.diamond] = tris;
    fineTrianglesOf[KIND.diamond] = tris;
    for (const m of diamondGlassMeshes) m.material = entry.glass;
    for (const m of diamondMirrorMeshes) m.material = entry.mirrored;
    for (const m of diamondPhotonMeshes) m.material = entry.photon;
    currentDiamondCut = cut;
    lightDirty = true;
  };
  let lastTriangles = 0;

  /* Shader compilation */
  // Each shape's programs are built the first time it is needed. Building them in the
  // middle of a frame froze the page (a third of a second here, seconds on slower GPUs
  // and Windows), so they are built with compileAsync instead (WebGPU's asynchronous
  // pipeline creation): until a shape is ready, the last picture stays up.
  /** Which programs are built: one per shape, and per cut for the diamond. */
  const compiledKeys = new Set<string>();
  const compiling = new Map<string, Promise<void>>();
  const programKey = (kind: number) => (kind === KIND.diamond ? `${kind}:${currentDiamondCut}` : String(kind));
  /** Shapes on show this frame are not compiled yet (the last picture is kept meanwhile). */
  let waitingForPrograms = false;
  /**
   * Build the programs of shape `kind` (as it is now: the diamond in its current cut)
   * without blocking, pass by pass, each into the target and with the camera it is drawn
   * with, so they are found again when the shape is drawn.
   */
  const compileKind = (kind: number): Promise<void> => {
    const key = programKey(kind);
    if (compiledKeys.has(key)) return Promise.resolve();
    const pending = compiling.get(key);
    if (pending) return pending;
    const passes: [THREE.Scene, THREE.Camera, THREE.RenderTarget][] = [
      [farScene, camera, farLayer],
      [nearScene, camera, nearLayer],
      [mirrorScene, camera, mirrorLayer],
      [lightScene, lightCamera, causticMap],
      [lampDepthScene, lightCamera, lampDepth],
      [photonScene, lightCamera, photonMap],
    ];
    const job = (async () => {
      for (const [scene, passCamera, target] of passes) {
        const passMeshes = kindMeshes[kind].filter((entry) => entry.scene === scene).map((entry) => entry.mesh);
        if (passMeshes.length === 0) continue;
        // compileAsync gathers what to build synchronously, from what is visible: show the
        // shape's meshes just for that, then put everything back before anything is drawn.
        const previousTarget = renderer.getRenderTarget();
        renderer.setRenderTarget(target);
        for (const mesh of passMeshes) mesh.visible = true;
        let built: Promise<unknown>;
        try {
          built = renderer.compileAsync(scene, passCamera);
        } finally {
          updateVisibility();
          renderer.setRenderTarget(previousTarget);
        }
        await built;
      }
      compiledKeys.add(key);
      compiling.delete(key);
    })();
    compiling.set(key, job);
    return job;
  };
  /** Build the programs of every shape among `objects` (and wait for them). */
  const compileObjects = async (): Promise<void> => {
    const kinds = new Set<number>();
    for (let i = 0; i < objects.count; i++) kinds.add(objects.kind[i]);
    for (const kind of kinds) await compileKind(kind);
  };
  /** Build the programs of every shape in the background, one after another. */
  const precompileAll = async (): Promise<void> => {
    for (let kind = 0; kind < KIND_COUNT; kind++) {
      await compileKind(kind);
      // Leave the GPU and the page a moment between shapes.
      await new Promise((resolve) => setTimeout(resolve, 50));
    }
  };

  /**
   * The wavelength bands real caustics are traced in for glass of dispersion `dispersion`:
   * one (white, no offset) without dispersion, else more the stronger it is, evenly across
   * the spectrum, each with its Cauchy offset and linear-sRGB share (the shares add up to
   * white, so clear glass still throws white light where the colours overlap).
   */
  const causticBandCache = new Map<number, THREE.Vector4[]>();
  const causticBands = (dispersion: number, coloured: boolean): THREE.Vector4[] => {
    // Coloured glass (tinted, CMY) needs at least three bands to carry its colour, even
    // without dispersion: each band is filtered by how much of its light the glass passes.
    const fewest = coloured ? 3 : 1;
    const count = dispersion > 0 ? Math.min(Math.max(Math.ceil((dispersion / 12) * 64), 4), isMobile ? 4 : 10) : fewest;
    let bands = causticBandCache.get(count);
    if (!bands) {
      bands = spectralBands(count).map((b) => new THREE.Vector4(b.cauchy, b.rgb[0], b.rgb[1], b.rgb[2]));
      causticBandCache.set(count, bands);
    }
    return bands;
  };

  /* Layer-sorting state */
  let lightDirty = true;
  let lastSun = '';
  const depth = new Float32Array(MAX_OBJECTS);
  const slices = new Uint32Array(1024);
  const order = new Uint32Array(MAX_OBJECTS); // the objects, nearest first
  let lastView = '';
  let lastCount = -1;
  let hasFar = false;
  let hasNear = false;
  const forward = new THREE.Vector3();
  const viewProj = new THREE.Matrix4();

  /**
   * Split the objects into a far half and a near half for this view. Only when
   * something changed: an object moved, or the camera did.
   */
  function sortIntoLayers(moved: boolean) {
    const n = objects.count;
    // A single object needs no camera-dependent layer sorting; skip re-uploading its instance buffers on orbit.
    if (!moved && n <= 1 && lastCount === n) return;
    camera.getWorldDirection(forward);
    const c = camera.position;
    const view = `${n} ${c.x} ${c.y} ${c.z} ${forward.x} ${forward.y} ${forward.z}`;
    if (!moved && view === lastView) return;
    lastView = view;
    lastCount = n;
    let nearest = Infinity;
    let farthest = -Infinity;
    for (let i = 0; i < n; i++) {
      const d =
        (objects.center[i * 3] - c.x) * forward.x +
        (objects.center[i * 3 + 1] - c.y) * forward.y +
        (objects.center[i * 3 + 2] - c.z) * forward.z;
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
    hasFar = farSet.some((b) => b.count > 0);
    hasNear = nearSet.some((b) => b.count > 0);
    for (const b of all) b.commit();
    updateVisibility();
  }

  /** Half-width of the square of ground the objects (and their shadows) cover. */
  function groundReach(): number {
    let reach = 0;
    // How far along the floor a shadow runs for each metre of height (twice over: an
    // object's top is about twice as high as its centre).
    const slope = 2 * Math.max(1, Math.hypot(sunDir.x, sunDir.z) / Math.max(sunDir.y, 0.1));
    for (let i = 0; i < objects.count; i++) {
      // Higher objects cast their shadows and caustics farther away, the more so the lower the lamp.
      const spread =
        Math.max(Math.abs(objects.center[i * 3]), Math.abs(objects.center[i * 3 + 2])) +
        objects.center[i * 3 + 1] * slope;
      if (spread > reach) reach = spread;
    }
    return Math.ceil((reach + 3) / 4) * 4; // in 4 m steps, so it does not shimmer
  }

  function resize(width: number, height: number) {
    const halfW = Math.max(1, width >> 1);
    const halfH = Math.max(1, height >> 1);
    background.setSize(width, height);
    farLayer.setSize(width, height);
    nearLayer.setSize(width, height);
    aoLayer.setSize(halfW, halfH);
    glowScratch.setSize(Math.max(1, width >> 2), Math.max(1, height >> 2));
    glowLayer.setSize(Math.max(1, width >> 2), Math.max(1, height >> 2));
    frostLayer.setSize(Math.max(1, width >> 2), Math.max(1, height >> 2));
    mirrorLayer.setSize(halfW, halfH);
    frameTarget.setSize(width, height);
    for (const t of historyTargets) t.setSize(width, height);
    historyValid = false;
    uRes.value.set(width, height);
    uAoRes.value.set(halfW, halfH);
    uGlowStep.value.set(1 / Math.max(1, width), 1 / Math.max(1, height));
  }

  /** `moved`: some object changed since the last frame (redraws the caustic map). */
  function render(
    glass:
      | MeshGlassMaterial
      | {
          ior: number;
          dispersion: number;
          roughness?: number;
          frostScale?: number;
          diamondCut?: DiamondCut;
          cmy?: number;
          cmyOrder?: CmyFaceOrder;
          bevel?: number;
          color?: [number, number, number, number];
          wireframe?: boolean;
          wireColor?: [number, number, number, number];
        },
    light: InstancedLighting,
    moved: boolean,
    wireColorOverride?: [number, number, number, number],
  ) {
    const isMat = glass instanceof MeshGlassMaterial;
    const ior = glass.ior;
    const dispersion = glass.dispersion;
    const roughness = glass.roughness ?? 0;
    const frostScale = glass.frostScale ?? 4;
    const diamondCut = light.diamondCut ?? (isMat ? 'round' : (glass.diamondCut ?? 'round'));
    const cmy = light.cmy ?? (isMat ? 0 : (glass.cmy ?? 0));
    const cmyOrder: CmyFaceOrder = light.cmyOrder ?? (isMat ? 'cym' : (glass.cmyOrder ?? 'cym'));
    const bevel = light.bevel ?? (isMat ? 0 : (glass.bevel ?? 0));
    const color: [number, number, number, number] = isMat
      ? [glass.color.r, glass.color.g, glass.color.b, light.colorAlpha ?? 1]
      : (glass.color ?? [1, 1, 1, 1]);
    const wantWire = glass.wireframe ?? false;
    const wireColor = wireColorOverride ?? (!isMat ? glass.wireColor : undefined) ?? [0.01, 0.01, 0.01, 1];
    setDiamondCut(diamondCut);

    // A shape whose programs are not built yet: build them in the background and keep the
    // last picture up meanwhile, instead of freezing on building them in this frame.
    waitingForPrograms = false;
    for (let i = 0; i < objects.count; i++) {
      const kind = objects.kind[i];
      if (compiledKeys.has(programKey(kind))) continue;
      waitingForPrograms = true;
      void compileKind(kind);
    }
    if (waitingForPrograms) {
      if (historyValid) present();
      return;
    }

    // Temporal AA: anything that changes the picture (the camera, the size, any setting,
    // objects moving) counts as motion; a still view is averaged until it settles.
    camera.updateMatrixWorld();
    viewProjNow.multiplyMatrices(camera.projectionMatrix, camera.matrixWorldInverse);
    const stateKey = JSON.stringify([
      uRes.value.x,
      uRes.value.y,
      objects.count,
      [ior, dispersion, roughness, frostScale, diamondCut, cmy, cmyOrder, bevel, color, wantWire, wireColor],
      JSON.stringify(light, (k, v) => {
        if (k === 'envMap') return (v as THREE.Texture | null)?.uuid ?? null;
        // The focus distance only matters with depth of field (it follows the orbiting camera).
        if (k === 'focus' && light.depthOfField <= 0) return 0;
        // Ignore float noise in the last digits (values recomputed every frame).
        return typeof v === 'number' ? Math.round(v * 1e5) / 1e5 : v;
      }),
    ]);
    const contentChanged = stateKey !== lastStateKey;
    lastStateKey = stateKey;
    const viewChanged = !viewProjNow.equals(viewProjLast);
    // The camera counts as moving only while the picture moves by a noticeable fraction of a
    // pixel: the long, ever slower glide of the orbit damping after letting go is averaged
    // like a still view (the history is reprojected either way, so it follows the drift).
    const moving = moved || contentChanged || (historyValid && viewChanged && screenMotionPx() > 0.25);
    if (moving) stillCount = TAA_HISTORY_WEIGHT;
    else if (historyValid && !viewChanged && stillCount >= ACCUMULATION_SAMPLES) {
      present(); // settled and unchanged: show the stored result again
      return;
    }
    // A Halton (2, 3) sub-pixel jitter every frame (TAA needs it while moving too).
    frameCounter += 1;
    const cycle = ((frameCounter - 1) % JITTER_CYCLE) + 1;
    const jitterX = halton(cycle, 2) - 0.5;
    const jitterY = halton(cycle, 3) - 0.5;
    const frameW = uRes.value.x;
    const frameH = uRes.value.y;
    camera.setViewOffset(frameW, frameH, jitterX, jitterY, frameW, frameH);
    uJitterUv.value.set(jitterX / frameW, jitterY / frameH);
    // Each frame moves every pixel's wavelengths on by the golden ratio of a step.
    SPECTRAL_FRAME_OFFSET.value = (frameCounter * 0.6180339887) % 1;
    timer.beginFrame();
    // Wireframe only: the edges in place of the glass, and nothing mirrored in the floor.
    if (wantWire !== wireOn) {
      wireOn = wantWire;
      updateVisibility();
    }
    // The colour of the lines (linear rgb) and how opaque they are.
    const [wr, wg, wb, wa] = wireColor;
    uWireColor.value.setRGB(wr, wg, wb);
    for (const m of wireMaterials) m.blendColor.setRGB(wa, wa, wa);
    uTint.value.fromArray(color);
    uBevel.value = bevel;
    uRoughness.value = roughness;
    uFrostScale.value = frostScale;
    uImperfection.value = Math.min(Math.max(light.imperfections ?? 0, 0), 1);
    if (uCmy.value !== cmy) lightDirty = true; // its shadow is coloured too
    uCmy.value = cmy;
    const [ox, oy, oz] = CMY_FACE_ORDERS[cmyOrder];
    if (uCmyOrder.value.x !== ox || uCmyOrder.value.y !== oy || uCmyOrder.value.z !== oz) {
      uCmyOrder.value.set(ox, oy, oz);
      lightDirty = true;
    }
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
    const objShadowsActive = light.objectShadows && objects.count > 1;
    uRoom.value = (light.ground ? 1 : 0) + (light.dark ? 2 : 0);
    // z: object-on-object shadows active (skips 4 lampDepth loads when count <= 1 or disabled);
    // w: 3-band CIE fast spectral mode (mobile / secondary layers).
    uSpot.value.set(light.spotRadius, light.spotRadius * 2.2, objShadowsActive ? 1 : 0, isMobile ? 1 : 0);
    uSkyE.value.fromArray(light.skyE);
    uSunE.value.fromArray(light.sunE);
    uAlbedo.value.fromArray(light.groundColor);
    sunDir.fromArray(light.sunDir);
    envTexNode.value = light.envMap ?? fallbackEnvTex;
    uEnvP.value.set(light.envMap ? 1 : 0, light.envMapIntensity ?? 1);
    const sunKey =
      light.sunDir.join() +
      light.shadowSoftness +
      light.causticSoftness +
      '/' +
      light.causticStrength +
      ior +
      cmy +
      cmyOrder +
      bevel +
      '/' +
      dispersion +
      color.join() +
      String(light.realCaustics) +
      String(light.objectShadows);
    if (sunKey !== lastSun) {
      lastSun = sunKey;
      lightDirty = true;
    }
    if (moved) lightDirty = true;

    sortIntoLayers(moved || lightDirty);
    // Triangles of the objects on show (each object's mesh once: a round diamond is 110),
    // not counting the extra passes that draw them again (floor mirror, shadow map).
    let sceneTriangles = 0;
    for (let kind = 0; kind < KIND_COUNT; kind++) {
      sceneTriangles +=
        trianglesOf[kind] * (far[kind].count + near[kind].count) +
        fineTrianglesOf[kind] * (fineFar[kind].count + fineNear[kind].count);
    }
    lastTriangles = sceneTriangles;

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
      // Only needed when multiple objects can shadow each other.
      if (objShadowsActive) {
        renderer.setClearColor(0x000000, 0);
        renderer.setRenderTarget(lampDepth);
        renderer.render(lampDepthScene, lightCamera);
      }
      renderer.setClearColor(previousClear, previousAlpha);
      // Soft shadows: wide ambient blur runs at 256x256 (16x fewer pixels).
      // When realCaustics is on, shadeNear is overwritten below, so skip the wasted shadeNear blur.
      const perMetre = 0.5 / half; // texture coordinates per metre of floor
      if (!light.realCaustics) {
        blur(causticMap.texture, shadeNear, light.shadowSoftness * perMetre);
        blur(
          shadeNear.texture,
          shadeFar,
          Math.max(light.shadowSoftness * 3, 0.25) * perMetre,
          undefined,
          shadeFarScratch,
          0,
          2,
        );
      } else {
        blur(
          causticMap.texture,
          shadeFar,
          Math.max(light.shadowSoftness * 3, 0.25) * perMetre,
          undefined,
          shadeFarScratch,
          0,
          2,
        );
        uGlass.value.set(glass.ior, glass.dispersion, 6, 0);
        mark('Caustics');
        let coloured = cmy > 0 || (color[3] > 0 && Math.min(color[0], color[1], color[2]) < 0.999);
        for (let i = 0; i < objects.count * 3 && !coloured; i++) coloured = objects.tint[i] < 0.999;
        const bands = causticBands(dispersion, coloured);
        uPhotonBand.value.copy(bands[0]);
        renderer.setClearColor(0x000000, 0);
        renderer.setRenderTarget(photonMap);
        renderer.render(photonScene, lightCamera);
        // The shadow itself (the light that meets no glass), with its edge as soft
        // as the shadows are set to be ...
        renderer.setRenderTarget(openMap);
        openQuad.render(renderer);
        // ... plus the light that came through the glass, gathered where it lands. Each
        // wavelength band is traced and gathered in turn and added up, so dispersion fans
        // the caustics out into their colours.
        renderer.setClearColor(0x000000, 1);
        renderer.setRenderTarget(landedMap);
        renderer.render(landScene, lightCamera);
        for (let b = 1; b < bands.length; b++) {
          uPhotonBand.value.copy(bands[b]);
          renderer.setClearColor(0x000000, 0);
          renderer.setRenderTarget(photonMap);
          renderer.render(photonScene, lightCamera);
          renderer.setRenderTarget(landedMap);
          renderer.autoClear = false;
          renderer.render(landScene, lightCamera);
          renderer.autoClear = true;
        }
        renderer.setClearColor(previousClear, previousAlpha);
        // The two added up, then blurred together as much as the lamp is wide: a small lamp
        // throws sharp shadows and caustics, a large soft one soft shadows and smooth pools.
        // (Blurred apart by different amounts, the shadow's edge faded out faster than the
        // light refracted across it faded in, leaving a dark ring round every silhouette.)
        copySource.value = openMap.texture;
        renderer.setRenderTarget(gatheredSoft);
        copyQuad.render(renderer);
        addSource.value = landedMap.texture;
        uAddGain.value = light.causticStrength;
        renderer.autoClear = false;
        addQuad.render(renderer);
        renderer.autoClear = true;
        const lampWidth = Math.max(light.shadowSoftness, light.causticSoftness);
        blur(gatheredSoft.texture, shadeNear, lampWidth * perMetre, undefined, shadeScratch, 0, 2);
        renderer.autoClear = true;
      }
    }

    // 2. The objects mirrored in the floor, if it is polished.
    // With thousands of objects each is small on screen, so the passes that are
    // seen least directly do less: the mirror image (blurred and faint anyway) and
    // the far half (seen through the near half) skip dispersion and follow the
    // light through fewer bounces.
    const many = objects.count > MANY_OBJECTS;
    // Floor mirror reflection is half-res and blurred: skip dispersion & cap bounces to 2.
    uGlass.value.set(ior, 0, 2, 0);
    uFloor.value.x = light.ground ? light.floorReflection : 0;
    uFloor.value.y = light.realCaustics ? -light.shadowDarkness : light.shadowDarkness; // (negative: see icGroundColor)
    // One object: its real focused caustic (up to ~25x the open floor). More: pools from
    // objects above one another add up (only the top layer is traced), so the headroom
    // falls quickly toward the pile's ~3x (12 objects: under 4x).
    uCausticHeadroom.value = 2 + 22 / Math.max(objects.count, 1);
    uFloor.value.z = light.floorGrid ? 1 : 0;
    uFloor.value.w = light.fog;
    if (light.floorColor) uFloorColor.value.fromArray(light.floorColor);
    else uFloorColor.value.set(light.groundColor[0], light.groundColor[1], light.groundColor[2], 1);
    uGridColor.value.fromArray(light.floorGridColor ?? [0.033, 0.033, 0.033, 1]);
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
    const rough = uRoughness.value;
    const frostBlur = (source: THREE.Texture) => {
      const aspect = uRes.value.x / uRes.value.y;
      const r1 = 0.012 + rough * 0.03;
      const r2 = 0.026 + rough * 0.065;
      // Capped to 1 pass each (4 total render passes max vs up to 18 previously).
      blur(source, frostLayer, r1, r1 * aspect, glowScratch, 0, 1);
      blur(frostLayer.texture, frostLayer, r2, r2 * aspect, glowScratch, 0, 1);
    };
    if (rough > 0.001) frostBlur(background.texture);
    const twoLayer = light.seeThrough && objects.count > 1;
    if (!twoLayer) {
      // Single-pass fast path (single object, or Glass through glass disabled):
      // skip farLayer and its full-screen copy, and disable lookThrough raymarching.
      uSpot.value.w = isMobile || many ? 1 : 0;
      uGlass.value.set(ior, dispersion, many ? 5 : isMobile ? 5 : 6, 0);
      mark('Glass');
      renderer.setRenderTarget(nearLayer);
      copySource.value = background.texture;
      copyQuad.render(renderer);
      renderer.autoClear = false;
      behindNode.value = background.texture;
      if (hasFar) renderer.render(farScene, camera);
      if (hasNear) renderer.render(nearScene, camera);
      renderer.autoClear = true;
    } else {
      // Far layer is seen refracted behind near glass: use 3-band CIE spectral mode (~2x faster).
      uSpot.value.w = 1;
      uGlass.value.set(ior, many ? 0 : dispersion, many ? 4 : isMobile ? 4 : 5, 0);
      mark('Far glass');
      renderer.setRenderTarget(farLayer);
      copySource.value = background.texture;
      copyQuad.render(renderer);
      renderer.autoClear = false;
      behindNode.value = background.texture;
      renderer.render(farScene, camera);

      // 5. Near half over a copy of that, seeing the far half refracted.
      renderer.autoClear = true;
      if (rough > 0.001) frostBlur(farLayer.texture);
      mark('Near glass');
      renderer.setRenderTarget(nearLayer);
      copySource.value = farLayer.texture;
      copyQuad.render(renderer);
      renderer.autoClear = false;
      uSpot.value.w = isMobile || many ? 1 : 0;
      uGlass.value.set(ior, dispersion, many ? 4 : isMobile ? 5 : 6, many ? 4 : isMobile ? 4 : 6);
      behindNode.value = farLayer.texture;
      renderer.render(nearScene, camera);
      renderer.autoClear = true;
    }

    // 6. Finishing: half-res contact shadows (skipped when disabled) ...
    const aoActive = light.contactShadows > 0.001;
    mark('Finishing');
    if (aoActive) {
      // Glass lets most of the light it blocks through (and focuses it nearby), so the
      // darkening where it meets the floor is only as strong as the glass is opaque: about
      // a third for clear glass, the full setting as its tint gets dark.
      const tintR = 1 + (color[0] - 1) * color[3];
      const tintG = 1 + (color[1] - 1) * color[3];
      const tintB = 1 + (color[2] - 1) * color[3];
      const transmitted = 0.2126 * tintR + 0.7152 * tintG + 0.0722 * tintB;
      const glassOpacity = 1 - 0.65 * Math.min(Math.max(transmitted, 0), 1);
      uShade.value.set(light.contactShadows * glassOpacity, light.contactRadius, light.exposure, 0);
      renderer.setRenderTarget(aoLayer);
      shadeQuad.render(renderer);
    }
    // ... the glow of whatever is brighter than white, prefiltered to quarter-res (4x4 box + soft HDR cap) then blurred wide ...
    if (light.glow > 0) {
      const aspect = uRes.value.x / uRes.value.y;
      uBlurCut.value = 1 / Math.max(light.exposure, 1e-4);
      renderer.setRenderTarget(glowLayer);
      glowPrefilterQuad.render(renderer);
      blur(
        glowLayer.texture,
        glowLayer,
        isMobile ? 0.022 : 0.014,
        (isMobile ? 0.022 : 0.014) * aspect,
        glowScratch,
        0,
        1,
      );
      if (!isMobile) {
        blur(glowLayer.texture, glowLayer, 0.032, 0.032 * aspect, glowScratch, 0, 2);
      }
    }
    // ... and the camera: bilateral contact shadow composite, exposure, depth of field, glow and tones, onto the screen.
    uFinish.value.set(light.exposure, aoActive ? 1 : 0, 1 / uAoRes.value.x, 1 / uAoRes.value.y);
    uLens.value.set(light.focus, light.depthOfField, light.glow, light.filmic ? 1 : 0);
    renderer.setRenderTarget(frameTarget);
    lensQuad.render(renderer);
    camera.clearViewOffset();

    // 7. Merge it into the history, and show that.
    uTaaMoving.value = moving && historyValid ? 1 : 0;
    uTaaStillWeight.value = historyValid ? 1 / (Math.min(stillCount, ACCUMULATION_SAMPLES) + 1) : 1;
    if (!moving) stillCount = Math.min(stillCount + 1, ACCUMULATION_SAMPLES);
    uPrevViewProj.value.copy(historyValid ? viewProjLast : viewProjNow);
    taaHistory.value = historyTargets[historyRead].texture;
    const write = 1 - historyRead;
    renderer.setRenderTarget(historyTargets[write]);
    taaQuad.render(renderer);
    historyRead = write;
    historyValid = true;
    viewProjLast.copy(viewProjNow);
    present();
    timer.endFrame();
  }

  const timer = createPassTimer(renderer);
  const mark = (name: string) => timer.mark(name);

  return {
    render,
    resize,
    markChanged: () => {
      lightDirty = true;
      lastStateKey = ''; // next frame counts as motion
    },
    /** Forget the history (a new scene): the next frame starts it afresh. */
    resetHistory: () => {
      historyValid = false;
      lastStateKey = '';
    },
    /** Whether the view is still settling (keep calling `render` until false). */
    accumulating: () => waitingForPrograms || !historyValid || stillCount < ACCUMULATION_SAMPLES,
    /** Build the programs of the shapes now among the objects, without blocking (await it). */
    compileObjects,
    /** Build every shape's programs in the background, one shape after another. */
    precompileAll,
    triangles: () => lastTriangles,
    passTimes: timer.passTimes,
  };
}
