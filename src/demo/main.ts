import * as THREE from 'three/webgpu';
import { OrbitControls } from 'three/addons/controls/OrbitControls.js';
import GUI from 'three/addons/libs/lil-gui.module.min.js';
import { GlassMaterial, KIND, createGlassObjects, createInstancedGlass, type CmyFaceOrder, type DiamondCut } from '../glass';
import { CMY_STRENGTH, layoutGrid, restingPose, type GridLayout } from './grid';

/** The volumes of the shapes made from meshes: 128^3 each, Suzanne's then the dragon's. */
const MESH_SDF_URLS = ['suzanne_sdf_128.bin', 'dragon_sdf_128.bin'];
const MESH_SDF_SIZE = 128;

type RGB = { r: number; g: number; b: number };

const settings = {
  // Lighting: a white photo studio, or a dark one.
  dark: false,
  /** In the dark studio: one spotlight from overhead, like on a stage. */
  spotlight: true,
  /** How bright its main lamp is (the one that casts the shadows). */
  lamp: 1,
  /** Where it is: how high above the floor, and which way round (degrees). */
  lampHeight: 45,
  lampTurn: 150,
  ground: true,
  groundColor: { r: 0xee / 255, g: 0xee / 255, b: 0xee / 255 } as RGB, // light grey, #eeeeee
  exposure: 1,

  // The floor.
  /** Floor surface colour and opacity (0: transparent, 1: solid). */
  floorColor: { r: 0xee / 255, g: 0xee / 255, b: 0xee / 255 } as RGB,
  floorAlpha: 1,
  /** How reflective the floor is. */
  floorReflection: 0.4,
  /** Studio shadows: blur of their edges (m) and darkness (0 to 1). */
  shadowSoftness: 0.15,
  shadowDarkness: 0.85,
  /** Objects cast the lamp's shadow on each other, not only on the floor. */
  objectShadows: true,
  /** Grid lines on the floor (with color & alpha), and fog (0: none). */
  floorGrid: true,
  floorGridColor: { r: 0x33 / 255, g: 0x33 / 255, b: 0x33 / 255 } as RGB,
  floorGridAlpha: 1,
  fog: 0.15,

  /** Real caustics: the lamp's light traced through the glass onto the floor. */
  realCaustics: true,
  /** Glass shows other glass through it, along the bent rays. */
  seeThrough: true,
  /** How soft they are (m): the wider the lamp, the softer. */
  causticSoftness: 0.3,
  /** How much of the light gathered by the glass reaches the floor. */
  causticStrength: 0.3,

  // Finishing.
  /** Darkening where objects meet the floor and each other (0 to 1). */
  contactShadows: 0.7,
  /** The camera: how soft what is out of focus gets (0: all sharp), glow, film-like tones. */
  depthOfField: 0,
  glow: 0.35,
  filmic: true,

  // The glass (of a grid of objects; one on its own has its own settings).
  ior: 1.52,
  dispersion: 1,
  /** Frosted glass roughness (0: crystal clear, 1: fully frosted). */
  roughness: 0,
  /** Frosted glass grain fineness (higher = smaller micro-particles). */
  frostScale: 4,
  /** How much is cut off the edges of boxes (m). */
  bevel: 0,
  /** Diamond cut style (Round Brilliant, Princess, Emerald, Asscher, Cushion, Radiant). */
  diamondCut: 'round' as DiamondCut,
  /** CMY cube face color permutation (X, Y, Z -> Cyan, Yellow, Magenta). */
  cmyOrder: 'cym' as CmyFaceOrder,
  /** The colour of the glass, and how strongly it applies (0: clear, whatever the colour). */
  glassColor: { r: 1, g: 1, b: 1 } as RGB,
  glassAlpha: 1,
  /** Give every object in a grid a random colour of its own. */
  colorize: false,
  /** Draw only the edges of the triangles the objects are made of, in this colour and opacity. */
  wireframe: false,
  // Blender's orange for the active object (its default theme, #ffaa40).
  wireColor: { r: 0xff / 255, g: 0xaa / 255, b: 0x40 / 255 } as RGB,
  wireAlpha: 1,
};

const query = new URLSearchParams(location.search);
/** The settings as the page opened with them: what choosing a shape goes back to. */
const DEFAULTS = structuredClone(settings);

/**
 * The shapes to choose from. One of them on its own is placed close up, like a
 * product photo, with settings of its own; more are set out in a grid.
 */
type Shape = {
  /** The kinds of object it is made of (null: every kind, mixed). */
  kinds: number[] | null;
  cubic?: boolean;
  cmy?: boolean;
  ior?: number;
  dispersion?: number;
  /** On its own: the settings it opens with, in place of the usual ones. */
  look?: Partial<typeof settings>;
  /** On its own: where the camera starts (position, then what it looks at). */
  camera?: [number, number, number, number, number, number];
};
const SHAPES: Record<string, Shape> = {
  'Mixed shapes': { kinds: null, ior: 1.52 },
  // Diamonds get diamond's IOR and (small) dispersion.
  // On its own the diamond is seen from lower down, looking up at it a little.
  Diamond: { kinds: [KIND.diamond], ior: 2.42, dispersion: 0.4, camera: [3.0, 1.2, 4.1, 0.35, 0.4, 0] },
  // A glass ball on its own is lit from behind by a low lamp: its shadow comes
  // toward the camera, with the light it gathers pooled inside it.
  Sphere: {
    kinds: [KIND.sphere],
    ior: 1.52, // optical crown glass (BK7)
    look: { spotlight: true, lamp: 1.7, lampHeight: 32, lampTurn: -93, fog: 0, causticSoftness: 0.26, glow: 0.5, depthOfField: 0 },
    camera: [2.35, 2.76, 3.1, 0.35, 0.3, 0.04],
  },
  Cube: { kinds: [KIND.box], cubic: true, ior: 1.52 }, // crown glass
  'CMY cube': { kinds: [KIND.box], cubic: true, cmy: true, ior: 1.49 }, // acrylic (PMMA)
  Box: { kinds: [KIND.box], ior: 1.52 }, // soda-lime glass
  Cylinder: { kinds: [KIND.cylinder], ior: 1.47 }, // borosilicate (Pyrex)
  Capsule: { kinds: [KIND.capsule], ior: 1.47 }, // borosilicate glass
  Cone: { kinds: [KIND.cone], ior: 1.52 }, // crown glass
  Torus: { kinds: [KIND.torus], ior: 1.52 }, // blown glass
  'Torus knot': { kinds: [KIND.torusknot], ior: 1.52 }, // blown crystal knot
  Octahedron: { kinds: [KIND.octahedron], ior: 1.55 }, // quartz crystal
  Prism: { kinds: [KIND.prism], ior: 1.62 }, // flint glass
  'Hex prism': { kinds: [KIND.hexprism], ior: 1.55 }, // quartz crystal
  Suzanne: { kinds: [KIND.suzanne], ior: 1.52 }, // crown glass
  Dragon: { kinds: [KIND.dragon], ior: 1.66 }, // lead crystal
};
/** Up to this many objects. */
const MAX_COUNT = 15000;
// The address picks what opens: ?shape=cmy-cube&count=500. Only letters and digits
// count, and plurals do too. (The old ?demo=pile-diamonds or single-diamond still work.)
const lettersOnly = (text: string) => text.toLowerCase().replace(/[^a-z0-9]+/g, '');
const findShape = (text: string): string | undefined => {
  const wanted = lettersOnly(text);
  if (!wanted) return undefined;
  // The longest name that it starts with, or that starts with it ("cmycubes" is the CMY cube, not the cube).
  const matches = Object.keys(SHAPES).filter((name) => wanted.startsWith(lettersOnly(name)) || lettersOnly(name).startsWith(wanted));
  return matches.sort((a, b) => b.length - a.length)[0];
};
const oldDemo = lettersOnly(query.get('demo') ?? '');
const view = {
  shape: findShape(query.get('shape') ?? oldDemo.replace(/^(single|pile)/, '')) ?? 'Diamond',
  count: query.has('count') ? Number(query.get('count')) : oldDemo.startsWith('pile') ? 2000 : 1,
  layout: (query.get('layout') === 'grid' ? 'grid' : 'flat') as GridLayout,
};
view.count = Math.min(Math.max(Math.round(view.count) || 1, 1), MAX_COUNT);
/** A shape's name as it is written in the address: "CMY cube" is cmy-cube. */
const slug = (name: string) => name.toLowerCase().replace(/[^a-z0-9]+/g, '-');

/**
 * The loading screen: a bar and a line of text over the canvas until the first
 * frame is on screen. Downloads fill most of the bar; the rest is compiling the
 * shaders.
 */
const loading = (() => {
  const root = document.getElementById('loading')!;
  const fill = document.getElementById('loading-fill')!;
  const text = document.getElementById('loading-text')!;
  /** Per file: bytes so far and the size, when the server says (and isn't compressing). */
  const files = new Map<string, { loaded: number; total: number | null; done: boolean }>();
  const DOWNLOAD_SHARE = 0.85;
  const set = (fraction: number, message: string) => {
    fill.style.width = `${(Math.min(Math.max(fraction, 0), 1) * 100).toFixed(1)}%`;
    text.textContent = message;
  };
  const showDownloads = () => {
    const all = [...files.values()];
    const loaded = all.reduce((s, f) => s + f.loaded, 0);
    const known = all.every((f) => f.total !== null);
    const total = all.reduce((s, f) => s + (f.total ?? 0), 0);
    // With every size known, by bytes; otherwise by files finished.
    const fraction = known && total > 0 ? loaded / total : all.filter((f) => f.done).length / Math.max(all.length, 1);
    const mb = (n: number) => (n / 1048576).toFixed(1);
    set(fraction * DOWNLOAD_SHARE, known ? `Downloading models… ${mb(loaded)} / ${mb(total)} MB` : `Downloading models… ${mb(loaded)} MB`);
  };
  return {
    stage: (fraction: number, message: string) => set(fraction, message),
    /** Fetch a file as bytes, counting its progress on the bar. */
    async fetch(url: string): Promise<Response & { buffer: () => Promise<ArrayBuffer> }> {
      const res = await fetch(url);
      const length = res.headers.get('content-length');
      // A compressed response's length is the compressed size, not what the stream yields.
      const total = length && !res.headers.get('content-encoding') ? Number(length) : null;
      const entry = { loaded: 0, total, done: false };
      files.set(url, entry);
      const buffer = async () => {
        if (!res.body) return res.arrayBuffer();
        const chunks: Uint8Array[] = [];
        const reader = res.body.getReader();
        for (;;) {
          const { done, value } = await reader.read();
          if (done) break;
          chunks.push(value);
          entry.loaded += value.byteLength;
          showDownloads();
        }
        entry.done = true;
        showDownloads();
        const out = new Uint8Array(entry.loaded);
        let at = 0;
        for (const c of chunks) {
          out.set(c, at);
          at += c.byteLength;
        }
        return out.buffer;
      };
      return Object.assign(res, { buffer });
    },
    done() {
      set(1, 'Ready');
      root.classList.add('done');
      setTimeout(() => root.remove(), 500);
    },
  };
})();

function fail(message: string): never {
  document.getElementById('loading')?.remove();
  const el = document.getElementById('error')!;
  el.textContent = message;
  el.style.display = 'flex';
  throw new Error(message);
}

const srgbToLinear = (c: number) => (c <= 0.04045 ? c / 12.92 : Math.pow((c + 0.055) / 1.055, 2.4));


/**
 * The shapes of the meshes: 128^3 volumes of signed distances to their surfaces,
 * one after the other along z in a single texture.
 */
async function loadMeshVolumes(): Promise<THREE.Data3DTexture> {
  const n = MESH_SDF_SIZE;
  const all = new Uint16Array(n * n * n * MESH_SDF_URLS.length);
  await Promise.all(
    MESH_SDF_URLS.map(async (url, k) => {
      const res = await loading.fetch(url);
      if (!res.ok) fail(`Could not load the mesh volume (${url}).`);
      all.set(new Uint16Array(await res.buffer()), k * n * n * n);
    }),
  );
  const tex = new THREE.Data3DTexture(all, n, n, n * MESH_SDF_URLS.length);
  tex.format = THREE.RedFormat;
  tex.type = THREE.HalfFloatType;
  tex.minFilter = tex.magFilter = THREE.LinearFilter;
  tex.wrapS = tex.wrapT = tex.wrapR = THREE.ClampToEdgeWrapping;
  tex.needsUpdate = true;
  return tex;
}

/**
 * A mesh shape as triangles, in the same space as its volume.
 * File layout: vertex, index and hull-point counts (uint32), then positions,
 * normals (float32 x 3 each), indices (uint32) and hull points (float32 x 3, the
 * convex hull once used for collisions, not read).
 */
async function loadMesh(url: string): Promise<{ geometry: THREE.BufferGeometry }> {
  const res = await loading.fetch(url);
  if (!res.ok) fail(`Could not load the mesh (${url}).`);
  const buffer = await res.buffer();
  const [vertices, indices] = new Uint32Array(buffer, 0, 2);
  let offset = 12;
  const take = <T>(make: (start: number, length: number) => T, length: number): T => {
    const out = make(offset, length);
    offset += length * 4;
    return out;
  };
  const positions = take((o, n) => new Float32Array(buffer.slice(o, o + n * 4)), vertices * 3);
  const normals = take((o, n) => new Float32Array(buffer.slice(o, o + n * 4)), vertices * 3);
  const index = take((o, n) => new Uint32Array(buffer.slice(o, o + n * 4)), indices);
  const geometry = new THREE.BufferGeometry();
  geometry.setAttribute('position', new THREE.BufferAttribute(positions, 3));
  geometry.setAttribute('normal', new THREE.BufferAttribute(normals, 3));
  geometry.setIndex(new THREE.BufferAttribute(index, 1));
  return { geometry };
}


async function main() {
  const canvas = document.getElementById('view') as HTMLCanvasElement;
  // Anti-aliased (4x multisampling): it smooths the edges of the glass.
  // GPU timestamps: the time each part of the frame takes, for the stats panel.
  // (They can only be turned on here, before the renderer starts; they cost little.)
  // WebGPU where the browser has it, otherwise WebGL 2 (the shaders are TSL, which
  // three.js compiles for either). ?renderer=webgl uses WebGL 2 even so, to try it.
  // Canvas antialias is off because geometry AA already runs in nearLayer and the
  // final pass (lensQuad) is a single full-screen quad — avoiding 4x swapchain MSAA bandwidth.
  loading.stage(0, 'Starting the renderer…');
  const forceWebGL = query.get('renderer') === 'webgl';
  const renderer = new THREE.WebGPURenderer({ canvas, antialias: false, trackTimestamp: true, forceWebGL });
  try {
    await renderer.init();
  } catch {
    fail('This demo needs WebGPU or WebGL 2.\nOpen it in a recent Chrome, Edge, Safari or Firefox.');
  }
  /** The GPU device, with WebGPU (to know when the first frame is done); null on WebGL. */
  const device = (renderer.backend as { isWebGPUBackend?: boolean }).isWebGPUBackend
    ? (renderer.backend as unknown as { device: GPUDevice }).device
    : null;
  renderer.toneMapping = THREE.NoToneMapping;

  // ------------------------------------------------------------ camera ---
  const camera = new THREE.PerspectiveCamera(28.8, 1, 0.1, 500);
  const orbit = new OrbitControls(camera, canvas);
  // Inertia: the camera glides on a little after the mouse lets go.
  orbit.enableDamping = true;
  orbit.dampingFactor = 0.08;
  // Zooming goes toward (and away from) whatever is under the pointer, not the
  // middle of the picture.
  orbit.zoomToCursor = true;
  // Never lower than 8 degrees above the floor: from floor level the floor is seen
  // edge on, there is nothing to show an object standing on it, and it seems to hover.
  orbit.maxPolarAngle = Math.PI / 2 - THREE.MathUtils.degToRad(8);

  // ---------------------------------------------------------- textures ---
  // The Stanford dragon (Stanford Computer Graphics Laboratory, via Morgan McGuire's
  // archive and the Khronos glTF samples): free for non-commercial use, with credit.
  const [meshVolume, suzanneMesh, dragonMesh, dragonLod] = await Promise.all([
    loadMeshVolumes(),
    loadMesh('suzanne_mesh.bin'),
    loadMesh('dragon_mesh.bin'),
    loadMesh('dragon_mesh_lod.bin'),
  ]);

  // ------------------------------------------------------------- scene ---
  // One object on its own, or thousands of random ones set out in a grid.
  const glassObjects = createGlassObjects();
  /** Half-width of the grid the objects are set out in (m). */
  let gridReach = 1;
  let gridCenterY = 0;
  /**
   * The single-shape demos: one object placed by hand in the middle of the floor,
   * about a metre tall, resting the way it would lie. Each demo keeps its own
   * settings: the shape's measurements (named per shape), how far it is turned,
   * its IOR and its dispersion.
   */
  type Single = { measures: number[]; turn: number; ior: number; dispersion: number; roughness: number; frostScale: number; bevel: number; colour: number; cmyOrder: CmyFaceOrder; glassColor: RGB; glassAlpha: number };
  const SINGLE_SHAPES: Record<number, { labels: string[]; measures: number[]; turn: number }> = {
    [KIND.box]: { labels: ['Half width', 'Half height', 'Half depth'], measures: [0.6, 0.6, 0.6], turn: -15 },
    [KIND.sphere]: { labels: ['Radius'], measures: [0.6], turn: 0 },
    [KIND.cylinder]: { labels: ['Radius', 'Half height'], measures: [0.5, 0.6], turn: 0 },
    [KIND.capsule]: { labels: ['Radius', 'Half length'], measures: [0.4, 0.4], turn: 0 },
    [KIND.cone]: { labels: ['Radius', 'Half height'], measures: [0.55, 0.6], turn: 0 },
    [KIND.torus]: { labels: ['Ring radius', 'Tube radius'], measures: [0.62, 0.22], turn: 0 },
    [KIND.octahedron]: { labels: ['Size'], measures: [0.85], turn: 20 },
    [KIND.prism]: { labels: ['Size', 'Half length'], measures: [0.65, 0.6], turn: 55 },
    [KIND.hexprism]: { labels: ['Radius', 'Half height'], measures: [0.5, 0.5], turn: -15 },
    [KIND.suzanne]: { labels: ['Size'], measures: [0.8], turn: -20 },
    [KIND.diamond]: { labels: ['Size'], measures: [0.6], turn: 240 },
    [KIND.dragon]: { labels: ['Size'], measures: [0.9], turn: -30 },
    [KIND.torusknot]: { labels: ['Size'], measures: [0.75], turn: -20 },
  };
  const singles: Record<string, Single> = {};
  const singleKind = (name: string): number => SHAPES[name].kinds?.[0] ?? KIND.box;
  /** The settings of a shape on its own (its defaults, the first time). */
  const singleFor = (name: string): Single => {
    const demo = SHAPES[name];
    const shape = SINGLE_SHAPES[singleKind(name)];
    return (singles[name] ??= {
      // A box is not a cube: longer one way, flatter.
      measures: name === 'Box' ? [0.75, 0.42, 0.55] : [...shape.measures],
      turn: shape.turn,
      ior: demo.ior ?? 1.52,
      dispersion: demo.dispersion ?? 1,
      roughness: 0,
      frostScale: 4,
      // A CMY cube gets fine bevels: its edges catch the light, as cut glass does.
      bevel: demo.cmy ? 0.008 : 0,
      colour: CMY_STRENGTH,
      cmyOrder: 'cym',
      glassColor: { r: 1, g: 1, b: 1 },
      glassAlpha: 1,
    });
  };
  /** Put a shape on its own in the middle of the floor, resting the way it would lie. */
  const placeShowcase = (name: string) => {
    const kind = singleKind(name);
    const single = singleFor(name);
    const [a, b = a, c = a] = single.measures;
    let size: [number, number, number] = [a, a, a];
    if (kind === KIND.box) size = [a, b, c];
    else if (kind === KIND.cylinder || kind === KIND.cone || kind === KIND.hexprism || kind === KIND.capsule) size = [a, b, a];
    else if (kind === KIND.torus) size = [a, b, b];
    else if (kind === KIND.prism) size = [a, a, b];
    const { height, quat } = restingPose(kind, size, THREE.MathUtils.degToRad(single.turn), settings.diamondCut);
    glassObjects.count = 1;
    glassObjects.kind[0] = kind;
    glassObjects.size.set(size, 0);
    glassObjects.center.set([0, height, 0], 0);
    glassObjects.quat.set([quat.x, quat.y, quat.z, quat.w], 0);
    glassObjects.vary.set([0, 1, 0], 0); // exactly the IOR and dispersion it is set to
    glassObjects.tint.set([1, 1, 1], 0); // and the colour
  };
  const glassMaterial = new GlassMaterial({
    ior: settings.ior,
    dispersion: settings.dispersion,
    roughness: settings.roughness,
    frostScale: settings.frostScale,
    diamondCut: settings.diamondCut,
  });
  const instanced = createInstancedGlass(renderer, camera, glassObjects, {
    suzanne: suzanneMesh.geometry,
    dragon: dragonMesh.geometry,
    dragonLod: dragonLod.geometry,
    volume: meshVolume,
  });
  let dirtyFrames = 2;
  const markDirty = () => {
    dirtyFrames = Math.max(dirtyFrames, 2);
  };
  /** The shape chosen. */
  const current = () => SHAPES[view.shape];
  /** One object on its own (one of the mix is a grid of one). */
  const isSingle = () => view.count === 1 && current().kinds !== null;
  /** The settings and camera the scene opens with, when they are not the usual ones. */
  const lookOf = () => (isSingle() ? (current().look ?? {}) : {});
  /**
   * Give the objects of a grid their own colours: a random one each if asked for,
   * otherwise none (white). CMY cubes keep to their own colours.
   */
  /** A new one each time random colours are turned on, for a new set of colours. */
  let colourSeed = 0;
  const colourObjects = () => {
    const colour = new THREE.Color();
    for (let i = 0; i < glassObjects.count; i++) {
      // Each object's colour is a hash of its number (and the seed), so it keeps it
      // as long as the seed stays the same.
      const pick = (n: number) => {
        const x = Math.sin((i + 1) * n + colourSeed) * 43758.5453;
        return x - Math.floor(x);
      };
      if (settings.colorize && glassObjects.vary[i * 3 + 2] === 0) colour.setHSL(pick(12.9898), 0.55 + 0.4 * pick(78.233), 0.5 + 0.2 * pick(37.719));
      else colour.setRGB(1, 1, 1);
      glassObjects.tint.set([colour.r, colour.g, colour.b], i * 3);
    }
    instanced.markChanged();
    markDirty();
  };
  /** The colour of all the glass on show (linear rgb) and how strongly it applies. */
  const glassColour = (): [number, number, number, number] => {
    const from = isSingle() ? singleFor(view.shape) : settings;
    return [srgbToLinear(from.glassColor.r), srgbToLinear(from.glassColor.g), srgbToLinear(from.glassColor.b), from.glassAlpha];
  };
  /** How much is cut off the edges of boxes: set on a single cube, one value for a grid. */
  const boxBevel = () => (isSingle() ? singleFor(view.shape).bevel : settings.bevel);
  /** How strong the colour filters of a CMY cube are (0: not a CMY cube). */
  const cmyStrength = () => {
    if (!current().cmy) return 0;
    return isSingle() ? Math.max(singleFor(view.shape).colour, 0.001) : CMY_STRENGTH;
  };
  /** Which CMY face colour sits on X, Y (top), Z. */
  const cmyFaceOrder = (): CmyFaceOrder => (isSingle() ? singleFor(view.shape).cmyOrder : settings.cmyOrder);
  /** Set the scene up: a shape on its own resting in place, or a grid of them. */
  const placeObjects = (keepGlass = false) => {
    const shape = current();
    if (isSingle()) {
      const single = singleFor(view.shape);
      if (!keepGlass) {
        settings.ior = single.ior;
        settings.dispersion = single.dispersion;
        settings.roughness = single.roughness;
        settings.frostScale = single.frostScale;
      }
      placeShowcase(view.shape);
    } else {
      // Each shape sets its own glass when switching shapes (kept when only changing count/layout).
      if (!keepGlass) {
        settings.ior = shape.ior ?? 1.52;
        settings.dispersion = shape.dispersion ?? 1;
      }
      const bounds = layoutGrid(glassObjects, view.count, shape.kinds, shape.cubic ?? false, view.layout, settings.diamondCut);
      gridReach = bounds.reach;
      gridCenterY = bounds.centerY;
    }
    colourObjects(); // (white, unless random colours are on)
    instanced.markChanged();
    markDirty();
  };
  Object.assign(settings, lookOf());
  placeObjects();
  /** Put the camera where the whole scene is in view. */
  const frameScene = () => {
    const start = isSingle() ? current().camera : undefined;
    if (start) {
      camera.position.set(start[0], start[1], start[2]);
      orbit.target.set(start[3], start[4], start[5]);
    } else if (isSingle()) {
      camera.position.set(3.017, 1.9, 4.119); // 30 degrees round to the right of straight on
      orbit.target.set(0.35, 0.22, 0);
    } else {
      // From in front, above, a little to the right: the whole grid in view.
      const r = gridReach;
      camera.position.set(r * 1.3 + 2, gridCenterY + r * 1.6 + 2, r * 2.8 + 3);
      orbit.target.set(0, gridCenterY, 0);
    }
    orbit.update();
  };

  const isMobile =
    /Mobi|Android|iPhone|iPad|iPod/i.test(navigator.userAgent) ||
    (navigator.maxTouchPoints > 1 && Math.min(window.innerWidth, window.innerHeight) < 900);
  function resize() {
    const w = window.innerWidth;
    const h = window.innerHeight;
    const screenRatio = window.devicePixelRatio || 1;
    // Cap total rendered pixel budget so large Retina displays (e.g. MacBook Pro 2x) and
    // high-DPR phones stay crisp without overflowing HalfFloat fragment bandwidth.
    const maxPixels = isMobile ? 550_000 : 2_400_000;
    const budgetRatio = Math.sqrt(maxPixels / Math.max(w * h, 1));
    const pixelRatio = Math.max(1, Math.min(screenRatio, isMobile ? 1.5 : 2, budgetRatio));
    renderer.setPixelRatio(pixelRatio);
    renderer.setSize(w, h, false);
    camera.aspect = w / h;
    camera.updateProjectionMatrix();
    instanced.resize(Math.round(w * pixelRatio), Math.round(h * pixelRatio));
  }
  window.addEventListener('resize', resize);

  // --------------------------------------------------------------- GUI ---
  const gui = new GUI({ title: 'Settings', width: 300 });
  // Folded to its title bar at first, so the scene is not covered.
  gui.close();
  const status = { fps: '0', triangles: '0' };
  /** Settings of the page itself, kept whatever the scene (?stats=1 shows the panel). */
  const ui = { stats: query.get('stats') === '1' };

  /** Keep the address in step, so that reloading or sharing it opens the same scene. */
  const updateAddress = () => {
    const address = new URL(location.href);
    address.searchParams.delete('demo');
    address.searchParams.set('shape', slug(view.shape));
    address.searchParams.set('count', String(view.count));
    if (view.count > 1 && view.layout === 'grid') address.searchParams.set('layout', 'grid');
    else address.searchParams.delete('layout');
    history.replaceState(null, '', address);
  };
  let wasSingle = isSingle();
  /** A new count: keeps all lighting, scene, camera, and glass settings intact. */
  const changeCount = () => {
    view.count = Math.min(Math.max(Math.round(view.count) || 1, 1), MAX_COUNT);
    const nowSingle = isSingle();
    if (nowSingle !== wasSingle) {
      const single = singleFor(view.shape);
      if (wasSingle && !nowSingle) {
        // Carry single-object glass settings over to the grid.
        settings.ior = single.ior;
        settings.dispersion = single.dispersion;
        settings.roughness = single.roughness;
        settings.frostScale = single.frostScale;
        Object.assign(settings.glassColor, single.glassColor);
        settings.glassAlpha = single.glassAlpha;
      } else if (!wasSingle && nowSingle) {
        // Carry grid glass settings over to the single object.
        single.ior = settings.ior;
        single.dispersion = settings.dispersion;
        single.roughness = settings.roughness;
        single.frostScale = settings.frostScale;
        Object.assign(single.glassColor, settings.glassColor);
        single.glassAlpha = settings.glassAlpha;
      }
      updateAddress();
      placeObjects(true);
      frameScene();
      showPanels();
      resize(); // a shape on its own is drawn larger than the screen
      gui.controllersRecursive().forEach((c) => c.updateDisplay());
    } else {
      updateAddress();
      placeObjects(true);
      layoutController.show(view.count > 1);
    }
    wasSingle = nowSingle;
  };
  // A new shape keeps everything else as it is (the scene, lighting and camera
  // settings, the camera, the count): only the objects change.
  gui.add(view, 'shape', Object.keys(SHAPES)).name('Shape').onChange(() => {
    updateAddress();
    placeObjects();
    showPanels();
    resize(); // a shape on its own is drawn larger than the screen
    gui.controllersRecursive().forEach((c) => c.updateDisplay());
    wasSingle = isSingle();
  });
  gui.add(view, 'count', 1, MAX_COUNT, 1).name('Count').onFinishChange(changeCount);
  const layoutController = gui
    .add(view, 'layout', { Flat: 'flat', '3D Grid': 'grid' })
    .name('Layout')
    .onChange(() => {
      updateAddress();
      placeObjects(true);
      frameScene();
    });
  // The stats panel in the corner: frame rate, triangles, and GPU time per part.
  gui.add(ui, 'stats').name('Show stats').onChange((on: boolean) => {
    const panel = document.getElementById('stats');
    if (panel) panel.style.display = on ? 'grid' : 'none';
  });
  type Folder = ReturnType<typeof gui.addFolder>;
  /**
   * A Reset button at the bottom of a folder: puts each of its settings back as the
   * page starts (colours too), then `after` (for whatever else that needs redoing).
   */
  const addReset = (folder: Folder, after?: () => void) =>
    folder.add({ reset: () => resetFolder(folder, after) }, 'reset').name('Reset');
  const resetFolder = (folder: Folder, after?: () => void) => {
    const all = settings as unknown as Record<string, unknown>;
    const start = DEFAULTS as unknown as Record<string, unknown>;
    for (const c of folder.controllers) {
      if (c.object !== settings || !(c.property in start)) continue;
      const value = start[c.property];
      if (value !== null && typeof value === 'object') Object.assign(all[c.property] as object, value);
      else all[c.property] = value;
      c.updateDisplay();
    }
    instanced.markChanged();
    markDirty();
    after?.();
  };
  const scene = gui.addFolder('Scene').close(); // folded until opened
  scene.addColor(settings, 'floorColor').name('Floor color');
  scene.add(settings, 'floorAlpha', 0, 1, 0.01).name('Floor alpha');
  scene.add(settings, 'floorReflection', 0, 1, 0.01).name('Floor reflection');
  scene.add(settings, 'shadowSoftness', 0.01, 0.6, 0.01).name('Shadow softness');
  scene.add(settings, 'shadowDarkness', 0, 1, 0.01).name('Shadow darkness');
  scene.add(settings, 'objectShadows').name('Object shadows');
  const gridCtrl = scene.add(settings, 'floorGrid').name('Floor grid');
  const gridOptions = [
    scene.addColor(settings, 'floorGridColor').name('Grid color'),
    scene.add(settings, 'floorGridAlpha', 0, 1, 0.01).name('Grid alpha'),
  ];
  const showGridOptions = () => gridOptions.forEach((c) => c.show(settings.floorGrid));
  gridCtrl.onChange(showGridOptions);
  showGridOptions();
  scene.add(settings, 'fog', 0, 2, 0.01).name('Fog');
  addReset(scene, showGridOptions);
  /**
   * Two folders, rebuilt for each shape and count:
   * - Glass settings, the same for every shape: wireframe, colour, IOR, dispersion
   *   (and on its own, how far it is turned). On its own they are that shape's;
   *   in a grid, the whole grid's.
   * - The shape's own settings, only what belongs to that shape: on its own, its
   *   measurements (and a box's bevel, a CMY cube's colour strength); in a grid,
   *   the bevel of its boxes. (Left out when there is nothing.)
   */
  let glassFolder: ReturnType<typeof gui.addFolder> | null = null;
  let glassOpen = false;
  let shapeFolder: ReturnType<typeof gui.addFolder> | null = null;
  function rebuildShapeFolder() {
    if (glassFolder) glassOpen = !(glassFolder as unknown as { _closed: boolean })._closed;
    glassFolder?.destroy();
    shapeFolder?.destroy();
    shapeFolder = null;
    const name = view.shape;
    const demo = SHAPES[name];
    const single = isSingle() ? singleFor(name) : null;
    const place = () => {
      placeShowcase(name);
      colourObjects();
    };

    // ---- Glass settings, for every shape.
    const glass = gui.addFolder('Glass settings');
    glassFolder = glass;
    // Folded until opened, and kept as it was when the folder is rebuilt for a new shape.
    if (!glassOpen) glass.close();
    const wire = glass.add(settings, 'wireframe').name('Wireframe only');
    const wireOptions = [glass.addColor(settings, 'wireColor').name('Wire color'), glass.add(settings, 'wireAlpha', 0, 1, 0.01).name('Wire alpha')];
    // (Its colour and opacity only while it is on.)
    const showWire = () => wireOptions.forEach((c) => c.show(settings.wireframe));
    wire.onChange(showWire);
    showWire();
    // A colour for the glass (CMY cubes have colours of their own) ...
    const own = single ?? settings;
    const colourOptions = demo.cmy ? [] : [glass.addColor(own, 'glassColor').name('Glass color'), glass.add(own, 'glassAlpha', 0, 1, 0.01).name('Color alpha')];
    // ... or a random colour for each object instead (a new set every time it is
    // turned on; on its own, a new colour): the colour above is then not used, and
    // greyed out.
    const showColour = () => colourOptions.forEach((c) => c.enable(!settings.colorize));
    glass.add(settings, 'colorize').name('Random colors').onChange((on: boolean) => {
      if (on) colourSeed = Math.random() * 1000;
      colourObjects();
      showColour();
    });
    showColour();
    if (single) {
      glass.add(single, 'ior', 1, 2.5, 0.01).name('IOR').onChange((v: number) => (settings.ior = v));
      glass.add(single, 'dispersion', 0, 5, 0.05).name('Dispersion').onChange((v: number) => (settings.dispersion = v));
      const roughCtrl = glass.add(single, 'roughness', 0, 1, 0.01).name('Roughness');
      const frostGrainCtrl = glass.add(single, 'frostScale', 0.5, 12, 0.1).name('Frost grain').onChange((v: number) => (settings.frostScale = v));
      roughCtrl.onChange((v: number) => {
        settings.roughness = v;
        frostGrainCtrl.show(v > 0);
      });
      frostGrainCtrl.show(single.roughness > 0);
      glass.add(single, 'turn', -180, 180, 1).name('Turn').onChange(place);
    } else {
      glass.add(settings, 'ior', 1, 2.5, 0.01).name('IOR').listen();
      glass.add(settings, 'dispersion', 0, 5, 0.05).name('Dispersion').listen();
      const roughCtrl = glass.add(settings, 'roughness', 0, 1, 0.01).name('Roughness').listen();
      const frostGrainCtrl = glass.add(settings, 'frostScale', 0.5, 12, 0.1).name('Frost grain').listen();
      roughCtrl.onChange((v: number) => frostGrainCtrl.show(v > 0));
      frostGrainCtrl.show(settings.roughness > 0);
    }
    glass.add(settings, 'seeThrough').name('Glass through glass');
    glass.add(settings, 'realCaustics').name('Real caustics');
    glass.add(settings, 'causticSoftness', 0.005, 0.3, 0.005).name('Caustic softness');
    glass.add(settings, 'causticStrength', 0, 3, 0.01).name('Caustic strength');
    // Reset: the wireframe as the page starts, and the glass as the shape starts
    // (its own, on its own: its measurements are left as they are).
    glass.add({
      reset: () => {
        Object.assign(settings, {
          wireframe: DEFAULTS.wireframe,
          wireAlpha: DEFAULTS.wireAlpha,
          colorize: DEFAULTS.colorize,
          seeThrough: DEFAULTS.seeThrough,
          realCaustics: DEFAULTS.realCaustics,
          causticSoftness: DEFAULTS.causticSoftness,
          causticStrength: DEFAULTS.causticStrength,
        });
        Object.assign(settings.wireColor, DEFAULTS.wireColor);
        if (single) {
          const { measures, bevel, colour, cmyOrder } = single;
          delete singles[name];
          Object.assign(singleFor(name), { measures, bevel, colour, cmyOrder });
          placeObjects();
        } else {
          settings.ior = demo.ior ?? 1.52;
          settings.dispersion = demo.dispersion ?? 1;
          settings.roughness = DEFAULTS.roughness;
          settings.frostScale = DEFAULTS.frostScale;
          Object.assign(settings.glassColor, DEFAULTS.glassColor);
          settings.glassAlpha = DEFAULTS.glassAlpha;
          colourObjects();
        }
        instanced.markChanged();
        rebuildShapeFolder();
      },
    }, 'reset').name('Reset');

    // ---- The shape's own settings.
    const hasBoxes = !demo.kinds || demo.kinds.includes(KIND.box);
    const hasDiamonds = !demo.kinds || demo.kinds.includes(KIND.diamond);
    if (!single && !hasBoxes && !hasDiamonds) return;
    const folder = gui.addFolder(single ? `${name} shape` : `${name} grid shape`);
    shapeFolder = folder;
    const diamondCutOptions: Record<string, DiamondCut> = {
      'Round Brilliant': 'round',
      Princess: 'princess',
      Emerald: 'emerald',
      Asscher: 'asscher',
      Cushion: 'cushion',
      Radiant: 'radiant',
    };
    const cmyOrderList: CmyFaceOrder[] = ['cym', 'myc', 'ycm', 'mcy', 'cmy', 'ymc'];
    const cmyOrderOptions: Record<string, CmyFaceOrder> = {
      'Yellow top (C / Y / M)': 'cym',
      'Yellow top (M / Y / C)': 'myc',
      'Cyan top (Y / C / M)': 'ycm',
      'Cyan top (M / C / Y)': 'mcy',
      'Magenta top (C / M / Y)': 'cmy',
      'Magenta top (Y / M / C)': 'ymc',
    };
    const addCmyFaceControls = (target: { cmyOrder: CmyFaceOrder }) => {
      const orderCtrl = folder.add(target, 'cmyOrder', cmyOrderOptions).name('Face colors');
      folder
        .add(
          {
            rotate: () => {
              const next = cmyOrderList[(cmyOrderList.indexOf(target.cmyOrder) + 1) % cmyOrderList.length];
              target.cmyOrder = next;
              orderCtrl.updateDisplay();
              markDirty();
            },
          },
          'rotate',
        )
        .name('Rotate faces');
    };
    if (!single) {
      if (hasBoxes) folder.add(settings, 'bevel', 0, 0.05, 0.001).name('Box bevel');
      if (demo.cmy) addCmyFaceControls(settings);
      if (hasDiamonds) folder.add(settings, 'diamondCut', diamondCutOptions).name('Diamond cut').onChange(() => placeObjects(true));
      addReset(folder, () => placeObjects(true));
      return;
    }
    SINGLE_SHAPES[singleKind(name)].labels.forEach((label, i) =>
      folder.add(single.measures as unknown as Record<string, number>, String(i), 0.1, 1.5, 0.01).name(label).onChange(place),
    );
    if (singleKind(name) === KIND.diamond) {
      folder.add(settings, 'diamondCut', diamondCutOptions).name('Cut').onChange(() => placeObjects(true));
    }
    if (singleKind(name) === KIND.box) folder.add(single, 'bevel', 0, 0.15, 0.001).name('Bevel');
    if (demo.cmy) {
      folder.add(single, 'colour', 0, 1, 0.01).name('Colour strength');
      addCmyFaceControls(single);
    }
    // Reset: the shape's measurements as it starts (its glass is left as it is).
    const reset = () => {
      const { ior, dispersion, roughness, frostScale, glassColor, glassAlpha, turn } = single;
      delete singles[name];
      Object.assign(singleFor(name), { ior, dispersion, roughness, frostScale, glassColor, glassAlpha, turn });
      if (singleKind(name) === KIND.diamond) settings.diamondCut = DEFAULTS.diamondCut;
      placeObjects(true);
      rebuildShapeFolder();
    };
    folder.add({ reset }, 'reset').name('Reset');
  }
  /** A grid has one IOR for all; a single has its own settings. */
  function showPanels() {
    layoutController.show(view.count > 1);
    rebuildShapeFolder();
  }

  const lighting = gui.addFolder('Lighting');
  lighting.add(settings, 'dark').name('Dark studio').onChange(() => instanced.markChanged());
  lighting.add(settings, 'spotlight').name('Spotlight (dark studio)').onChange(() => instanced.markChanged());
  lighting.add(settings, 'lamp', 0, 3, 0.01).name('Main lamp');
  lighting.add(settings, 'lampHeight', 10, 85, 1).name('Lamp height (deg)');
  lighting.add(settings, 'lampTurn', -180, 180, 1).name('Lamp direction (deg)');
  lighting.add(settings, 'ground').name('Ground');
  lighting.addColor(settings, 'groundColor').name('Ground color');
  lighting.add(settings, 'exposure', 0.1, 4, 0.01).name('Exposure');
  addReset(lighting);
  lighting.close();

  const cameraFolder = gui.addFolder('Camera');
  cameraFolder.add(settings, 'contactShadows', 0, 1, 0.01).name('Contact shadows');
  cameraFolder.add(settings, 'depthOfField', 0, 1, 0.01).name('Depth of field');
  cameraFolder.add(settings, 'glow', 0, 1, 0.01).name('Glow');
  cameraFolder.add(settings, 'filmic').name('Film-like tones');
  addReset(cameraFolder);
  cameraFolder.close();


  // -------------------------------------------------------------- loop ---
  resize();
  frameScene();
  showPanels();
  let fpsFrames = 0;
  let fpsSince = performance.now();

  const scaled = (v: number): [number, number, number] => [v * settings.lamp, v * settings.lamp, v * settings.lamp];
  let renderedSincePoll = false;

  {
    // Two columns: names on the left, values lined up on the right.
    const panel = document.createElement('div');
    panel.id = 'stats';
    panel.style.cssText =
      'position:fixed;left:12px;bottom:12px;display:grid;grid-template-columns:auto auto;column-gap:16px;padding:8px 10px;font:12px/1.5 ui-monospace,monospace;font-variant-numeric:tabular-nums;color:#eee;background:rgba(20,20,20,0.8);border-radius:6px;pointer-events:none;z-index:10';
    panel.style.display = ui.stats ? 'grid' : 'none'; // (hidden from the start, not after its first update)
    document.body.append(panel);
    /** The panel's rows: [name, value], or a heading (a name alone, across both columns). */
    const fill = (rows: ([string, string] | [string])[]) => {
      panel.replaceChildren(
        ...rows.flatMap((row) => {
          const name = document.createElement('span');
          name.textContent = row[0];
          if (row.length === 1) {
            name.style.cssText = 'grid-column:1 / -1;margin-top:6px;color:#aaa';
            return [name];
          }
          const value = document.createElement('span');
          value.textContent = row[1];
          value.style.textAlign = 'right';
          return [name, value];
        }),
      );
    };
    // Some parts are drawn only on some frames (the lamp's maps, when something has
    // moved): each part shows its latest time, for as long as it was seen lately.
    const seen = new Map<string, { ms: number; at: number }>();
    const order = ['Shadow map', 'Caustics', 'Reflection', 'Ground', 'Glass', 'Far glass', 'Near glass', 'Finishing', 'Other'];
    const show = async () => {
      const now = performance.now();
      // Only read back GPU timestamps when frames were actually rendered since the last poll.
      if (renderedSincePoll) {
        renderedSincePoll = false;
        for (const [name, ms] of Object.entries((await instanced.passTimes()) ?? {})) seen.set(name, { ms, at: now });
      }
      panel.style.display = ui.stats ? 'grid' : 'none';
      const recent = order.filter((name) => seen.has(name) && now - seen.get(name)!.at < 2000).map((name) => [name, seen.get(name)!.ms] as const);
      const total = recent.reduce((a, [, ms]) => a + ms, 0);
      fill([
        ['FPS', status.fps],
        ['Triangles', status.triangles],
        ['GPU time per frame'],
        ...recent.map(([name, ms]) => [name, `${ms.toFixed(2)} ms`] as [string, string]),
        ['Drawing', `${total.toFixed(2)} ms`],
      ]);
      setTimeout(show, 500);
    };
    show();
  }

  // The first frame compiles the shaders, which can take a moment: keep the loading
  // screen up until the GPU has finished that frame.
  loading.stage(0.94, 'Compiling shaders…');
  let firstFrame = true;
  let lastFrameSig = '';
  gui.onChange(markDirty);
  orbit.addEventListener('change', markDirty);
  window.addEventListener('resize', markDirty);
  const frameSignature = () => {
    const p = camera.position;
    const q = camera.quaternion;
    const t = orbit.target;
    return (
      `${p.x.toFixed(5)},${p.y.toFixed(5)},${p.z.toFixed(5)}|` +
      `${q.x.toFixed(5)},${q.y.toFixed(5)},${q.z.toFixed(5)},${q.w.toFixed(5)}|` +
      `${t.x.toFixed(5)},${t.y.toFixed(5)},${t.z.toFixed(5)}|` +
      `${window.innerWidth}x${window.innerHeight}`
    );
  };
  renderer.setAnimationLoop(() => {
    if (firstFrame) {
      firstFrame = false;
      requestAnimationFrame(() => {
        // (WebGL cannot say when the GPU is done: by the next frame is close enough.)
        if (device) device.queue.onSubmittedWorkDone().then(() => loading.done(), () => loading.done());
        else requestAnimationFrame(() => loading.done());
      });
    }
    orbit.update();
    const sig = frameSignature();
    if (sig !== lastFrameSig) {
      lastFrameSig = sig;
      dirtyFrames = Math.max(dirtyFrames, 1);
    }
    if (dirtyFrames <= 0) {
      fpsFrames = 0;
      fpsSince = performance.now();
      return;
    }
    dirtyFrames -= 1;
    renderedSincePoll = true;
    // The page knows which studio is on, so what is drawn over it (the link in the
    // corner) can be dark on the white one and light on the dark one.
    if (document.body.classList.contains('dark-studio') !== settings.dark) document.body.classList.toggle('dark-studio', settings.dark);
    fpsFrames += 1;
    const nowMs = performance.now();
    if (nowMs - fpsSince >= 500) {
      status.fps = ((fpsFrames * 1000) / (nowMs - fpsSince)).toFixed(0);
      fpsFrames = 0;
      fpsSince = nowMs;
    }
    const spot = settings.dark && settings.spotlight;
    const lampUp = THREE.MathUtils.degToRad(settings.lampHeight);
    const lampRound = THREE.MathUtils.degToRad(settings.lampTurn);
    const [gr, gg, gb, ga] = glassColour();
    glassMaterial.ior = settings.ior;
    glassMaterial.dispersion = settings.dispersion;
    glassMaterial.roughness = settings.roughness;
    glassMaterial.frostScale = settings.frostScale;
    glassMaterial.diamondCut = settings.diamondCut;
    glassMaterial.cmy = cmyStrength();
    glassMaterial.cmyOrder = cmyFaceOrder();
    glassMaterial.bevel = boxBevel();
    glassMaterial.color.setRGB(gr, gg, gb);
    glassMaterial.colorAlpha = ga;
    glassMaterial.wireframe = settings.wireframe;
    instanced.render(
      glassMaterial,
      {
        ground: settings.ground,
        dark: settings.dark,
        seeThrough: settings.seeThrough,
        realCaustics: settings.realCaustics,
        causticSoftness: settings.causticSoftness,
        causticStrength: settings.causticStrength,
        contactShadows: settings.contactShadows,
        // They reach about as far as an object is big: a single is a metre, a grid's objects a third of that.
        contactRadius: isSingle() ? 0.45 : 0.22,
        // In focus: whatever is in the middle of the picture (the shader finds it). This
        // is only the distance to fall back on when the middle is empty backdrop.
        focus: camera.position.distanceTo(orbit.target),
        depthOfField: settings.depthOfField,
        glow: settings.glow,
        filmic: settings.filmic,
        // The pool of the spotlight takes in the single object, or the whole grid.
        spotRadius: spot ? (isSingle() ? 5.5 : gridReach * 1.5 + 2) : 0,
        floorReflection: settings.floorReflection,
        shadowSoftness: settings.shadowSoftness,
        shadowDarkness: settings.shadowDarkness,
        objectShadows: settings.objectShadows,
        floorColor: [
          srgbToLinear(settings.floorColor.r) * (settings.dark ? 0.2 : 1),
          srgbToLinear(settings.floorColor.g) * (settings.dark ? 0.2 : 1),
          srgbToLinear(settings.floorColor.b) * (settings.dark ? 0.2 : 1),
          settings.floorAlpha,
        ],
        floorGrid: settings.floorGrid,
        floorGridColor: [
          srgbToLinear(settings.floorGridColor.r),
          srgbToLinear(settings.floorGridColor.g),
          srgbToLinear(settings.floorGridColor.b),
          settings.floorGridAlpha,
        ],
        fog: settings.fog,
        // In the dark studio the floor is glossy black: it shows the objects mirrored in
        // it, and the stones, looking down into it, see black between their flashes.
        groundColor: [settings.groundColor.r, settings.groundColor.g, settings.groundColor.b].map((c) => srgbToLinear(c) * (settings.dark ? 0.2 : 1)) as [number, number, number],
        exposure: settings.exposure,
        // The studio lights the ground white: mostly soft light from all round, the
        // rest from the main lamp, which is up and to the left (as in tsl/scene.ts).
        // In the dark studio the floor gets next to nothing but a pool from that lamp.
        // (One strong lamp and little fill: shadows are dark, and what the glass
        // gathers stands out in them.)
        skyE: spot ? [0.02, 0.02, 0.02] : settings.dark ? [0.004, 0.004, 0.004] : [0.26, 0.26, 0.26],
        sunE: scaled(spot ? 0.3 : settings.dark ? 0.022 : 0.55),
        sunDir: [Math.cos(lampUp) * Math.cos(lampRound), Math.sin(lampUp), Math.cos(lampUp) * Math.sin(lampRound)],
      },
      // (Nothing moves by itself: a new scene marks itself changed.)
      false,
      [srgbToLinear(settings.wireColor.r), srgbToLinear(settings.wireColor.g), srgbToLinear(settings.wireColor.b), settings.wireAlpha],
    );
    status.triangles = instanced.triangles().toLocaleString('en-US');
  });
}

main().catch((err) => {
  console.error(err);
  const el = document.getElementById('error')!;
  if (el.style.display !== 'flex') {
    el.textContent = `Failed to start: ${err instanceof Error ? err.message : String(err)}`;
    el.style.display = 'flex';
  }
});
