// Many objects at once, for testing how fast they draw: random shapes of random
// proportions, each resting on the floor in its own place in a spaced grid.
// Nothing moves (there is no physics): every frame draws the same scene.
import * as THREE from 'three/webgpu';
import { DIAMOND_CUTS, KIND, MAX_OBJECTS, type DiamondCut, type GlassObjects } from '../glass';

/** How strong a CMY cube's colour filters are, unless set otherwise. */
export const CMY_STRENGTH = 0.85;

/** Spacing of the grid (m): room for the largest object, with a gap round it. */
const CELL = 1.35;

/** Objects are scaled at random by up to this (most stay near their usual size). */
const MAX_SCALE = 1.6;
/** But no object reaches farther than this from its centre (m), to fit its place in the grid. */
const MAX_REACH = 0.6;

/** About how far a shape reaches from its centre (m). */
function reachOf(kind: number, s: [number, number, number]): number {
  if (kind === KIND.capsule || kind === KIND.torus) return s[0] + s[1];
  if (
    kind === KIND.box ||
    kind === KIND.cylinder ||
    kind === KIND.cone ||
    kind === KIND.hexprism ||
    kind === KIND.prism
  )
    return Math.hypot(s[0], s[1], s[2]);
  return 1.1 * Math.max(s[0], s[1], s[2]); // sphere, octahedron, diamond, the meshes
}

/** Where each shape sits in the 0..1 range that the mix of every shape draws from. */
const PICK_FOR_KIND: Record<number, number> = {
  [KIND.box]: 0.1,
  [KIND.sphere]: 0.2,
  [KIND.cylinder]: 0.35,
  [KIND.capsule]: 0.45,
  [KIND.cone]: 0.55,
  [KIND.torus]: 0.65,
  [KIND.octahedron]: 0.75,
  [KIND.prism]: 0.85,
  [KIND.hexprism]: 0.9,
  [KIND.diamond]: 0.95,
  [KIND.suzanne]: 0.98,
  [KIND.dragon]: 0.992,
  [KIND.torusknot]: 0.998,
};

/**
 * How a shape of this size rests on the floor, turned `turn` radians round the
 * upright: how high its centre is, and its rotation. Most stand on their base; an
 * octahedron lies on a face and a diamond on a pavilion facet, its table turned up.
 */
export function restingPose(
  kind: number,
  size: ArrayLike<number>,
  turn: number,
  diamondCut: DiamondCut = 'round',
): { height: number; quat: THREE.Quaternion } {
  const [a, b] = [size[0], size[1]];
  let height = a;
  let lean: THREE.Quaternion | null = null;
  if (
    kind === KIND.box ||
    kind === KIND.cylinder ||
    kind === KIND.cone ||
    kind === KIND.hexprism ||
    kind === KIND.torus
  )
    height = b;
  else if (kind === KIND.capsule) height = a + b;
  else if (kind === KIND.octahedron) {
    height = a * 0.57735;
    lean = new THREE.Quaternion().setFromUnitVectors(
      new THREE.Vector3(-1, -1, -1).normalize(),
      new THREE.Vector3(0, -1, 0),
    );
  } else if (kind === KIND.prism) height = a / 2;
  else if (kind === KIND.suzanne)
    height = a * 0.72; // her chin is 0.72 below her centre at size 1
  else if (kind === KIND.dragon)
    height = a * 0.7053; // its feet are 0.7053 below its centre at size 1
  else if (kind === KIND.diamond) {
    const spec = DIAMOND_CUTS[diamondCut];
    height = spec.rest * a;
    lean = new THREE.Quaternion().setFromAxisAngle(new THREE.Vector3(0, 0, 1), -spec.lean);
  }
  const quat = new THREE.Quaternion().setFromAxisAngle(new THREE.Vector3(0, 1, 0), turn);
  if (lean) quat.multiply(lean);
  return { height, quat };
}

/**
 * Fill `objects` with `count` random objects, each resting on the floor in its own
 * cell of a square grid centred on the middle of the floor.
 * @param only  shape ids to draw from; null for the mix of every shape
 * @param cubic make boxes cubes (equal sides)
 * @returns half-width of the grid (m)
 */
export type GridLayout = 'flat' | 'grid';

export function layoutGrid(
  objects: GlassObjects,
  count: number,
  only: number[] | null = null,
  cubic = false,
  layout: GridLayout = 'flat',
  diamondCut: DiamondCut = 'round',
): { reach: number; centerY: number } {
  count = Math.min(Math.max(1, Math.round(count)), MAX_OBJECTS);
  let seed = (count * 2654435761) >>> 0;
  const rnd = () => {
    seed = (Math.imul(seed, 1664525) + 1013904223) >>> 0;
    return seed / 4294967296;
  };
  const perSide = layout === 'grid' ? Math.ceil(Math.cbrt(count)) : Math.ceil(Math.sqrt(count));
  const perLayer = perSide * perSide;

  objects.count = count;
  for (let i = 0; i < count; i++) {
    // A random shape, of every kind the app supports, with random proportions.
    let pick = rnd();
    if (only && only.length) pick = PICK_FOR_KIND[only[Math.floor(rnd() * only.length)]];
    let kind: number;
    let size: [number, number, number];
    let cmyCube = false;
    if (pick < 0.16) {
      kind = KIND.box;
      size = [0.16 + rnd() * 0.14, 0.16 + rnd() * 0.14, 0.16 + rnd() * 0.14];
      // In the mix of every shape, nearly half the boxes are CMY cubes of random size.
      if (!only && pick < 0.07) {
        cmyCube = true;
        size = [size[0], size[0], size[0]];
      }
    } else if (pick < 0.3) {
      kind = KIND.sphere;
      const r = 0.18 + rnd() * 0.17;
      size = [r, r, r];
    } else if (pick < 0.4) {
      kind = KIND.cylinder;
      const r = 0.15 + rnd() * 0.15;
      size = [r, 0.15 + rnd() * 0.2, r];
    } else if (pick < 0.5) {
      kind = KIND.capsule;
      const r = 0.13 + rnd() * 0.1;
      size = [r, 0.12 + rnd() * 0.16, r];
    } else if (pick < 0.6) {
      kind = KIND.cone;
      const r = 0.18 + rnd() * 0.14;
      size = [r, 0.18 + rnd() * 0.16, r];
    } else if (pick < 0.7) {
      kind = KIND.torus;
      const tube = 0.07 + rnd() * 0.04;
      size = [0.2 + rnd() * 0.1, tube, tube];
    } else if (pick < 0.79) {
      kind = KIND.octahedron;
      const r = 0.24 + rnd() * 0.14;
      size = [r, r, r];
    } else if (pick < 0.88) {
      kind = KIND.prism;
      const t = 0.22 + rnd() * 0.14;
      size = [t, t, 0.15 + rnd() * 0.15];
    } else if (pick < 0.93) {
      kind = KIND.hexprism;
      const r = 0.17 + rnd() * 0.11;
      size = [r, 0.12 + rnd() * 0.14, r];
    } else if (pick < 0.975) {
      kind = KIND.diamond;
      const r = 0.2 + rnd() * 0.14;
      size = [r, r, r];
    } else if (pick < 0.99) {
      kind = KIND.suzanne;
      const r = 0.32 + rnd() * 0.12;
      size = [r, r, r];
    } else if (pick < 0.995) {
      // (About as long as Suzanne is wide; a little lower and flatter.)
      kind = KIND.dragon;
      const r = 0.34 + rnd() * 0.12;
      size = [r, r, r];
    } else {
      kind = KIND.torusknot;
      const r = 0.28 + rnd() * 0.1;
      size = [r, r, r];
    }
    if (cubic && kind === KIND.box) size = [size[0], size[0], size[0]];
    // Some bigger, a few much bigger (and some a little smaller): from 0.85 to 1.6
    // times the size, weighted toward the small end; within the room each has.
    const scale = Math.min(0.85 + (MAX_SCALE - 0.85) * rnd() ** 2.5, Math.max(1, MAX_REACH / reachOf(kind, size)));
    size = [size[0] * scale, size[1] * scale, size[2] * scale];

    // Resting on the floor, turned any way round, in the middle of its cell.
    const { height, quat } = restingPose(kind, size, rnd() * 2 * Math.PI, diamondCut);
    const ix = i % perSide;
    const iz = layout === 'grid' ? Math.floor(i / perSide) % perSide : Math.floor(i / perSide);
    const iy = layout === 'grid' ? Math.floor(i / perLayer) : 0;
    const position = [(ix - (perSide - 1) / 2) * CELL, height + iy * CELL, (iz - (perSide - 1) / 2) * CELL];

    objects.kind[i] = kind;
    objects.size.set(size, i * 3);
    objects.center.set(position, i * 3);
    objects.quat.set([quat.x, quat.y, quat.z, quat.w], i * 4);
    // Its own glass: the IOR up to 0.15 either side of the setting, the dispersion
    // from half to one and a half times it.
    // (A CMY cube keeps the settings as they are, with its usual colour strength.)
    const ior = (rnd() - 0.5) * 0.3;
    const dispersion = 0.5 + rnd();
    objects.vary.set(cmyCube ? [0, 1, CMY_STRENGTH] : [ior, dispersion, 0], i * 3);
  }
  const halfWidth = (perSide * CELL) / 2;
  if (layout === 'grid') {
    const layers = Math.ceil(count / perLayer);
    const totalHeight = layers * CELL;
    return { reach: Math.max(halfWidth, totalHeight / 2), centerY: totalHeight / 2 };
  }
  return { reach: halfWidth, centerY: 0 };
}
