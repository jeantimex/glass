import * as THREE from 'three/webgpu';
import { OrbitControls } from 'three/addons/controls/OrbitControls.js';
import GUI from 'three/addons/libs/lil-gui.module.min.js';
import {
  GlassMaterial,
  KIND,
  createGlassObjects,
  createInstancedGlass,
  type CmyFaceOrder,
  type DiamondCut,
} from '../glass';
import { createLoadingScreen, loadBinaryMesh, loadMeshVolumes, showFatalError } from './assets';
import { CMY_STRENGTH, layoutGrid, restingPose } from './grid';
import {
  MAX_COUNT,
  SHAPES,
  SINGLE_SHAPE_SPECS,
  createDefaultSettings,
  parseInitialView,
  shapeSlug,
  srgbToLinear,
  type RGB,
} from './presets';
import { mountStatsPanel } from './stats';

const query = new URLSearchParams(location.search);
const settings = createDefaultSettings();
const DEFAULTS = structuredClone(settings);
const view = parseInitialView(query);
const loading = createLoadingScreen();

const singleKind = (name: string): number => SHAPES[name].kinds?.[0] ?? KIND.box;

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
    showFatalError('This demo needs WebGPU or WebGL 2.\nOpen it in a recent Chrome, Edge, Safari or Firefox.');
  }
  /** The GPU device, with WebGPU (to know when the first frame is done); null on WebGL. */
  const device = (renderer.backend as { isWebGPUBackend?: boolean }).isWebGPUBackend
    ? (renderer.backend as unknown as { device: GPUDevice }).device
    : null;
  renderer.toneMapping = THREE.NoToneMapping;

  /* Camera */
  const camera = new THREE.PerspectiveCamera(28.8, 1, 0.1, 500);
  const orbit = new OrbitControls(camera, canvas);
  // Tight tracking while dragging, smooth inertia glide on release.
  const DRAG_DAMPING = 0.3;
  const RELEASE_DAMPING = 0.1;
  orbit.enableDamping = true;
  orbit.dampingFactor = RELEASE_DAMPING;
  orbit.addEventListener('start', () => {
    orbit.dampingFactor = DRAG_DAMPING;
  });
  orbit.addEventListener('end', () => {
    orbit.dampingFactor = RELEASE_DAMPING;
  });
  // Zooming goes toward (and away from) whatever is under the pointer, not the
  // middle of the picture.
  orbit.zoomToCursor = true;
  // Never lower than 8 degrees above the floor: from floor level the floor is seen
  // edge on, there is nothing to show an object standing on it, and it seems to hover.
  orbit.maxPolarAngle = Math.PI / 2 - THREE.MathUtils.degToRad(8);

  /* Mesh & SDF assets */
  // The Stanford dragon (Stanford Computer Graphics Laboratory, via Morgan McGuire's
  // archive and the Khronos glTF samples): free for non-commercial use, with credit.
  const [meshVolume, suzanneGeo, dragonGeo, dragonLodGeo] = await Promise.all([
    loadMeshVolumes(loading),
    loadBinaryMesh(loading, 'suzanne_mesh.bin'),
    loadBinaryMesh(loading, 'dragon_mesh.bin'),
    loadBinaryMesh(loading, 'dragon_mesh_lod.bin'),
  ]);

  /* Scene objects */
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
  type Single = {
    measures: number[];
    turn: number;
    ior: number;
    dispersion: number;
    roughness: number;
    frostScale: number;
    bevel: number;
    colour: number;
    cmyOrder: CmyFaceOrder;
    glassColor: RGB;
    glassAlpha: number;
  };
  const singles: Record<string, Single> = {};
  /** The settings of a shape on its own (its defaults, the first time). */
  const singleFor = (name: string): Single => {
    const demo = SHAPES[name];
    const shape = SINGLE_SHAPE_SPECS[singleKind(name)];
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
    else if (kind === KIND.cylinder || kind === KIND.cone || kind === KIND.hexprism || kind === KIND.capsule)
      size = [a, b, a];
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
    suzanne: suzanneGeo,
    dragon: dragonGeo,
    dragonLod: dragonLodGeo,
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
      if (settings.colorize && glassObjects.vary[i * 3 + 2] === 0)
        colour.setHSL(pick(12.9898), 0.55 + 0.4 * pick(78.233), 0.5 + 0.2 * pick(37.719));
      else colour.setRGB(1, 1, 1);
      glassObjects.tint.set([colour.r, colour.g, colour.b], i * 3);
    }
    instanced.markChanged();
    markDirty();
  };
  /** The colour of all the glass on show (linear rgb) and how strongly it applies. */
  const glassColour = (): [number, number, number, number] => {
    const from = isSingle() ? singleFor(view.shape) : settings;
    return [
      srgbToLinear(from.glassColor.r),
      srgbToLinear(from.glassColor.g),
      srgbToLinear(from.glassColor.b),
      from.glassAlpha,
    ];
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
      const bounds = layoutGrid(
        glassObjects,
        view.count,
        shape.kinds,
        shape.cubic ?? false,
        view.layout,
        settings.diamondCut,
      );
      gridReach = bounds.reach;
      gridCenterY = bounds.centerY;
    }
    colourObjects(); // (white, unless random colours are on)
    instanced.markChanged();
    markDirty();
  };
  Object.assign(settings, lookOf());
  placeObjects();
  /** Put the camera where the whole scene is in view, looking at the center object. */
  const frameScene = () => {
    const start = isSingle() ? current().camera : undefined;
    if (start) {
      camera.position.set(start[0], start[1], start[2]);
      orbit.target.set(0, glassObjects.center[1], 0);
    } else if (isSingle()) {
      camera.position.set(3.017, 1.9, 4.119); // 30 degrees round to the right of straight on
      orbit.target.set(0, glassObjects.center[1], 0);
    } else {
      // From in front, above, a little to the right: the whole grid in view.
      const r = gridReach;
      camera.position.set(r * 1.3 + 2, gridCenterY + r * 1.6 + 2, r * 2.8 + 3);
      orbit.target.set(0, gridCenterY, 0);
    }
    orbit.update();
    markDirty();
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

  /* Settings GUI */
  const gui = new GUI({ title: 'Settings', width: 300 });
  // Folded to its title bar at first, so the scene is not covered.
  gui.close();
  const status = { fps: '0', triangles: '0' };
  /** Settings of the page itself, kept whatever the scene (shown by default; ?stats=0 hides it). */
  const ui = { stats: query.get('stats') !== '0' };

  /** Keep the address in step, so that reloading or sharing it opens the same scene. */
  const updateAddress = () => {
    const address = new URL(location.href);
    address.searchParams.delete('demo');
    address.searchParams.set('shape', shapeSlug(view.shape));
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
      frameScene();
      layoutController.show(view.count > 1);
    }
    wasSingle = nowSingle;
  };
  // A new shape keeps lighting and count settings, and re-frames the camera on the center object.
  gui
    .add(view, 'shape', Object.keys(SHAPES))
    .name('Shape')
    .onChange(() => {
      updateAddress();
      placeObjects();
      frameScene();
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
  gui
    .add(ui, 'stats')
    .name('Show stats')
    .onChange((on: boolean) => {
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

    /* Glass settings (shared across shapes) */
    const glass = gui.addFolder('Glass settings');
    glassFolder = glass;
    // Folded until opened, and kept as it was when the folder is rebuilt for a new shape.
    if (!glassOpen) glass.close();
    const wire = glass.add(settings, 'wireframe').name('Wireframe only');
    const wireOptions = [
      glass.addColor(settings, 'wireColor').name('Wire color'),
      glass.add(settings, 'wireAlpha', 0, 1, 0.01).name('Wire alpha'),
    ];
    // (Its colour and opacity only while it is on.)
    const showWire = () => wireOptions.forEach((c) => c.show(settings.wireframe));
    wire.onChange(showWire);
    showWire();
    // A colour for the glass (CMY cubes have colours of their own) ...
    const own = single ?? settings;
    const colourOptions = demo.cmy
      ? []
      : [
          glass.addColor(own, 'glassColor').name('Glass color'),
          glass.add(own, 'glassAlpha', 0, 1, 0.01).name('Color alpha'),
        ];
    // ... or a random colour for each object instead (a new set every time it is
    // turned on; on its own, a new colour): the colour above is then not used, and
    // greyed out.
    const showColour = () => colourOptions.forEach((c) => c.enable(!settings.colorize));
    glass
      .add(settings, 'colorize')
      .name('Random colors')
      .onChange((on: boolean) => {
        if (on) colourSeed = Math.random() * 1000;
        colourObjects();
        showColour();
      });
    showColour();
    if (single) {
      glass
        .add(single, 'ior', 1, 2.5, 0.01)
        .name('IOR')
        .onChange((v: number) => (settings.ior = v));
      glass
        .add(single, 'dispersion', 0, 5, 0.05)
        .name('Dispersion')
        .onChange((v: number) => (settings.dispersion = v));
      const roughCtrl = glass.add(single, 'roughness', 0, 1, 0.01).name('Roughness');
      const frostGrainCtrl = glass
        .add(single, 'frostScale', 0.5, 12, 0.1)
        .name('Frost grain')
        .onChange((v: number) => (settings.frostScale = v));
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
    glass
      .add(
        {
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
        },
        'reset',
      )
      .name('Reset');

    /* Shape-specific settings */
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
      if (hasDiamonds)
        folder
          .add(settings, 'diamondCut', diamondCutOptions)
          .name('Diamond cut')
          .onChange(() => placeObjects(true));
      addReset(folder, () => placeObjects(true));
      return;
    }
    SINGLE_SHAPE_SPECS[singleKind(name)].labels.forEach((label, i) =>
      folder
        .add(single.measures as unknown as Record<string, number>, String(i), 0.1, 1.5, 0.01)
        .name(label)
        .onChange(place),
    );
    if (singleKind(name) === KIND.diamond) {
      folder
        .add(settings, 'diamondCut', diamondCutOptions)
        .name('Cut')
        .onChange(() => placeObjects(true));
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
  lighting
    .add(settings, 'dark')
    .name('Dark studio')
    .onChange(() => instanced.markChanged());
  lighting
    .add(settings, 'spotlight')
    .name('Spotlight (dark studio)')
    .onChange(() => instanced.markChanged());
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

  /* Render loop */
  resize();
  frameScene();
  showPanels();
  let fpsFrames = 0;
  let fpsSince = performance.now();

  const scaled = (v: number): [number, number, number] => [v * settings.lamp, v * settings.lamp, v * settings.lamp];
  let renderedSincePoll = false;

  mountStatsPanel({
    isVisible: () => ui.stats,
    getStatus: () => status,
    shouldPollGpu: () => {
      if (!renderedSincePoll) return false;
      renderedSincePoll = false;
      return true;
    },
    readPassTimes: instanced.passTimes,
  });

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
        if (device)
          device.queue.onSubmittedWorkDone().then(
            () => loading.done(),
            () => loading.done(),
          );
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
    if (document.body.classList.contains('dark-studio') !== settings.dark)
      document.body.classList.toggle('dark-studio', settings.dark);
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
        groundColor: [settings.groundColor.r, settings.groundColor.g, settings.groundColor.b].map(
          (c) => srgbToLinear(c) * (settings.dark ? 0.2 : 1),
        ) as [number, number, number],
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
      [
        srgbToLinear(settings.wireColor.r),
        srgbToLinear(settings.wireColor.g),
        srgbToLinear(settings.wireColor.b),
        settings.wireAlpha,
      ],
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
