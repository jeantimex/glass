// The objects of a pile: which shapes, how big, and where they are dropped from.
// Shared by the two physics solvers (box3d.ts on the CPU, gpu.ts on the GPU).
import { KIND, MAX_OBJECTS, type GlassObjects } from '../glass';

/** How strong a CMY cube's colour filters are, unless set otherwise. */
export const CMY_STRENGTH = 0.85;

/** Horizontal spacing of the grid the objects are dropped from (m): room for the largest. */
const CELL = 1.35;

/** Objects are scaled at random by up to this (most stay near their usual size). */
const MAX_SCALE = 1.6;
/** But no object reaches farther than this from its centre (m), to fit its place in the grid. */
const MAX_REACH = 0.6;

/** About how far a shape reaches from its centre (m). */
function reachOf(kind: number, s: [number, number, number]): number {
  if (kind === KIND.capsule || kind === KIND.torus) return s[0] + s[1];
  if (kind === KIND.box || kind === KIND.cylinder || kind === KIND.cone || kind === KIND.hexprism || kind === KIND.prism) return Math.hypot(s[0], s[1], s[2]);
  return 1.1 * Math.max(s[0], s[1], s[2]); // sphere, octahedron, diamond, the meshes
}

/** Where each shape sits in the 0..1 range that the mixed pile draws from. */
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
  [KIND.dragon]: 0.995,
};

/**
 * Fill `objects` with `count` random objects above the ground, ready to drop.
 * @param only  shape ids to draw from; null for the mix of every shape
 * @param cubic make boxes cubes (equal sides)
 * @param adjust called once they are spawned, to change any of them by hand
 * @returns half-width of the square they are dropped over (m)
 */
export function spawnObjects(objects: GlassObjects, count: number, only: number[] | null = null, cubic = false, adjust?: () => void): number {
  count = Math.min(Math.max(1, Math.round(count)), MAX_OBJECTS);
  let seed = (count * 2654435761) >>> 0;
  const rnd = () => {
    seed = (Math.imul(seed, 1664525) + 1013904223) >>> 0;
    return seed / 4294967296;
  };

  // Dropped from a column of layers over a square footprint, so they land in a pile.
  // (Wider in proportion to the spacing, so the column is no taller.)
  const footprint = Math.min(Math.max(Math.sqrt(count) * 0.11, 1.2), 12) * (CELL / 0.95);
  const perSide = Math.max(1, Math.floor((2 * footprint) / CELL));
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
    } else {
      // (About as long as Suzanne is wide; a little lower and flatter.)
      kind = KIND.dragon;
      const r = 0.34 + rnd() * 0.12;
      size = [r, r, r];
    }
    if (cubic && kind === KIND.box) size = [size[0], size[0], size[0]];
    // Some bigger, a few much bigger (and some a little smaller): from 0.85 to 1.6
    // times the size, weighted toward the small end; within the room each has.
    const scale = Math.min(0.85 + (MAX_SCALE - 0.85) * rnd() ** 2.5, Math.max(1, MAX_REACH / reachOf(kind, size)));
    size = [size[0] * scale, size[1] * scale, size[2] * scale];

    // A random orientation (uniform over rotations).
    const u1 = rnd();
    const u2 = rnd() * 2 * Math.PI;
    const u3 = rnd() * 2 * Math.PI;
    const quat: [number, number, number, number] = [
      Math.sqrt(1 - u1) * Math.sin(u2),
      Math.sqrt(1 - u1) * Math.cos(u2),
      Math.sqrt(u1) * Math.sin(u3),
      Math.sqrt(u1) * Math.cos(u3),
    ];

    const cell = i % perLayer;
    const layer = Math.floor(i / perLayer);
    const position: [number, number, number] = [
      ((cell % perSide) - (perSide - 1) / 2) * CELL + (rnd() - 0.5) * 0.15,
      1.2 + layer * CELL,
      (Math.floor(cell / perSide) - (perSide - 1) / 2) * CELL + (rnd() - 0.5) * 0.15,
    ];

    objects.kind[i] = kind;
    objects.size.set(size, i * 3);
    objects.center.set(position, i * 3);
    objects.quat.set(quat, i * 4);
    // Its own glass: the IOR up to 0.15 either side of the setting, the dispersion
    // from half to one and a half times it.
    // (A CMY cube keeps the settings as they are, with its usual colour strength.)
    const ior = (rnd() - 0.5) * 0.3;
    const dispersion = 0.5 + rnd();
    objects.vary.set(cmyCube ? [0, 1, CMY_STRENGTH] : [ior, dispersion, 0], i * 3);
  }
  adjust?.();
  return footprint;
}

/**
 * A mesh shape's extreme points along 26 directions (a coarse convex hull), at half-width 1.
 * `planes` are those directions with how far the hull reaches along each (x, y, z, reach).
 */
export function meshExtremes(hull: Float32Array): { points: number[]; planes: number[] } {
  const kept = new Set<number>();
  const planes: number[] = [];
  for (let dx = -1; dx <= 1; dx++) {
    for (let dy = -1; dy <= 1; dy++) {
      for (let dz = -1; dz <= 1; dz++) {
        if (!dx && !dy && !dz) continue;
        const length = Math.hypot(dx, dy, dz);
        let best = 0;
        let bestDot = -Infinity;
        for (let k = 0; k < hull.length / 3; k++) {
          const d = (hull[k * 3] * dx + hull[k * 3 + 1] * dy + hull[k * 3 + 2] * dz) / length;
          if (d > bestDot) {
            bestDot = d;
            best = k;
          }
        }
        kept.add(best);
        planes.push(dx / length, dy / length, dz / length, bestDot);
      }
    }
  }
  const points: number[] = [];
  for (const k of kept) points.push(hull[k * 3], hull[k * 3 + 1], hull[k * 3 + 2]);
  return { points, planes };
}
