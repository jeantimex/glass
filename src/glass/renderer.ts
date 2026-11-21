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
//
// The layers carry each pixel's distance from the camera in their alpha, which is
// what the finishing passes work from.
import * as THREE from 'three/webgpu';
import { InstanceBatch } from './batch';
import { DIAMOND_CUTS, diamondGeometry, type DiamondCut } from './diamond';
import { createShapeGeometries, triangleCount, UNIT_TUBE } from './geometries';
import { GlassMaterial } from './material';
import {
  attribute,
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
  uv,
  vec4,
  varying,
  instanceIndex,
} from 'three/tsl';
import { KIND, MAX_OBJECTS, type GlassObjects } from './objects';
import { createPassTimer } from './timing';
import { causticNode, photonNode as photonRayNode } from './tsl/caustics';
import { CMY_FACE_ORDERS, type CmyFaceOrder } from './tsl/cmy';
import { blurNode, contactShadeNode, lensNode } from './tsl/finish';
import { glassNode } from './tsl/glass';
import { groundNode } from './tsl/ground';

const KIND_COUNT = 13;

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

export function createInstancedGlass(
  renderer: THREE.WebGPURenderer,
  camera: THREE.PerspectiveCamera,
  objects: GlassObjects,
  /**
   * The shapes made from meshes: Suzanne's mesh, the dragon's (in full, and
   * simplified for piles), and their baked distance volumes, one after the other in
   * a single 3D texture (Suzanne's, then the dragon's).
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
  const uTint = uniform(new THREE.Vector4(1, 1, 1, 1)); // the colour of all the glass (linear rgb), and how strongly it applies
  const uCmy = uniform(0); // above 0: boxes are CMY cubes, with filters this strong
  const uCmyOrder = uniform(new THREE.Vector3(0, 1, 2)); // CMY face color permutation (X, Y, Z -> 0=C, 1=Y, 2=M)
  const uBevel = uniform(0); // how much is cut off the edges of boxes (m)
  const uRoughness = uniform(0); // frosted glass roughness (0: clear, 1: frosted)
  const uFrostScale = uniform(4); // frosted glass grain density (higher = smaller particles)
  const uSpot = uniform(new THREE.Vector4()); // stage spotlight: beam radius at the floor (0: off), distance of the lamp
  const uFloor = uniform(new THREE.Vector4()); // x: how reflective the floor is; y: how dark its shadows are
  const uFloorColor = uniform(new THREE.Vector4(0.85, 0.85, 0.85, 1)); // floor linear rgb + alpha
  const uGridColor = uniform(new THREE.Vector4(0.033, 0.033, 0.033, 1)); // floor grid linear rgb + alpha
  const uBlurDir = uniform(new THREE.Vector2());
  const uMirrorCam = uniform(new THREE.Vector3()); // the camera, mirrored in the floor

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
  const sceneInputs = {
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
  };
  /** The camera as the passes over the whole picture see it. */
  const screenCamera = {
    fragCoord: screenCoordinate,
    res: uRes,
    camPos: uCamPos,
    camRight: uCamRight,
    camUp: uCamUp,
    camFwd: uCamFwd,
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
    // Each bilinear-paired tap covers 2 texels (up to 6 pairs = 13 fetches per pass).
    const texels = Math.max(radiusX * scratch.width, radiusY * scratch.height);
    const passes = Math.min(Math.max(Math.ceil((texels / 22) ** 2), 1), maxPasses);
    const share = 1 / Math.sqrt(passes);
    uBlurReach.value = Math.min(Math.max(Math.ceil((texels * share) / 2.2), 2), 6);
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
      photonMapSize,
      diamondPlanes,
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
  // ... then every ray that went through glass, as a square one texel wide (wider
  // if the glass fanned it out: see photonNode in tsl/glass.ts), moved to where it landed and added on.
  const rayIndex = float(instanceIndex);
  const rayRow = rayIndex.div(photonMapSize).floor();
  const rayAt = vec2(rayIndex.sub(rayRow.mul(photonMapSize)), rayRow)
    .add(0.5)
    .div(photonMapSize);
  const ray = texture(photonMap.texture, rayAt);
  const landMaterial = new THREE.NodeMaterial();
  const landPos = vec2(rayAt.x.mul(2).sub(1), float(1).sub(rayAt.y.mul(2)))
    .add(ray.xy)
    .add(
      vec2(positionGeometry.x, positionGeometry.y).mul(
        ray.a
          .sub(1)
          .max(1)
          .mul(2 / photonMapSize),
      ),
    );
  // Cull empty photon rays (no glass hit or zero carried light) offscreen before rasterization.
  const landActive = ray.z.greaterThan(1e-5);
  landMaterial.vertexNode = vec4(select(landActive, landPos.x, 2), select(landActive, landPos.y, 2), 0, 1);
  landMaterial.fragmentNode = vec4(vec3(varying(ray.z)), 1);
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
  addMaterial.fragmentNode = addSource.mul(uAddGain);
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
  };
  const shadeMaterial = new THREE.NodeMaterial();
  shadeMaterial.fragmentNode = contactShadeNode(aoCamera, uShade, shadeSource, isMobile ? 6 : 8);
  shadeMaterial.transparent = true;
  shadeMaterial.blending = THREE.NoBlending;
  shadeMaterial.depthTest = shadeMaterial.depthWrite = false;
  const shadeQuad = new THREE.QuadMesh(shadeMaterial);
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
    uRes.value.set(width, height);
    uAoRes.value.set(halfW, halfH);
  }

  /** `moved`: some object changed since the last frame (redraws the caustic map). */
  function render(
    glass:
      | GlassMaterial
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
    const isMat = glass instanceof GlassMaterial;
    const ior = glass.ior;
    const dispersion = glass.dispersion;
    const roughness = glass.roughness ?? 0;
    const frostScale = glass.frostScale ?? 4;
    const diamondCut = glass.diamondCut ?? 'round';
    const cmy = glass.cmy ?? 0;
    const cmyOrder: CmyFaceOrder = glass.cmyOrder ?? 'cym';
    const bevel = glass.bevel ?? 0;
    const color: [number, number, number, number] = isMat
      ? [glass.color.r, glass.color.g, glass.color.b, glass.colorAlpha]
      : (glass.color ?? [1, 1, 1, 1]);
    const wantWire = glass.wireframe ?? false;
    const wireColor = wireColorOverride ?? (!isMat ? glass.wireColor : undefined) ?? [0.01, 0.01, 0.01, 1];
    setDiamondCut(diamondCut);
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
      color.join() +
      String(light.realCaustics) +
      String(light.objectShadows);
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
      drawn +=
        trianglesOf[kind] * (far[kind].count + near[kind].count) +
        fineTrianglesOf[kind] * (fineFar[kind].count + fineNear[kind].count);
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
        renderer.setClearColor(0x000000, 0);
        renderer.setRenderTarget(photonMap);
        renderer.render(photonScene, lightCamera);
        // The shadow itself (the light that meets no glass), with its edge as soft
        // as the shadows are set to be ...
        renderer.setRenderTarget(openMap);
        openQuad.render(renderer);
        blur(openMap.texture, shadeNear, light.shadowSoftness * perMetre, undefined, shadeScratch, 0, 2);
        // ... plus the light that came through the glass, gathered where it lands
        // and blurred as much as the lamp is wide: a small lamp throws sharp
        // caustics, a large soft one smooth pools of light.
        renderer.setClearColor(0x000000, 1);
        renderer.setRenderTarget(landedMap);
        renderer.render(landScene, lightCamera);
        renderer.setClearColor(previousClear, previousAlpha);
        blur(landedMap.texture, gatheredSoft, light.causticSoftness * perMetre, undefined, shadeScratch, 0, 2);
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
    // Floor mirror reflection is half-res and blurred: skip dispersion & cap bounces to 2.
    uGlass.value.set(ior, 0, 2, 0);
    uFloor.value.x = light.ground ? light.floorReflection : 0;
    uFloor.value.y = light.realCaustics ? -light.shadowDarkness : light.shadowDarkness; // (negative: see icGroundColor)
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
      uShade.value.set(light.contactShadows, light.contactRadius, light.exposure, 0);
      renderer.setRenderTarget(aoLayer);
      shadeQuad.render(renderer);
    }
    // ... the glow of whatever is brighter than white, blurred wide (cut = 1/exposure, scaled by exposure in lensNode) ...
    if (light.glow > 0) {
      const aspect = uRes.value.x / uRes.value.y;
      const cut = 1 / Math.max(light.exposure, 1e-4);
      blur(
        nearLayer.texture,
        glowLayer,
        isMobile ? 0.022 : 0.014,
        (isMobile ? 0.022 : 0.014) * aspect,
        glowScratch,
        cut,
        1,
      );
      if (!isMobile) {
        blur(glowLayer.texture, glowLayer, 0.032, 0.032 * aspect, glowScratch, 0, 1);
      }
    }
    // ... and the camera: bilateral contact shadow composite, exposure, depth of field, glow and tones, onto the screen.
    uFinish.value.set(light.exposure, aoActive ? 1 : 0, 1 / uAoRes.value.x, 1 / uAoRes.value.y);
    uLens.value.set(light.focus, light.depthOfField, light.glow, light.filmic ? 1 : 0);
    renderer.setRenderTarget(null);
    lensQuad.render(renderer);
    timer.endFrame();
  }

  const timer = createPassTimer(renderer);
  const mark = (name: string) => timer.mark(name);

  return {
    render,
    resize,
    markChanged: () => (lightDirty = true),
    triangles: () => lastTriangles,
    passTimes: timer.passTimes,
  };
}
