// Rigid-body physics for the real-time renderer: Box3D (Erin Catto's 3D engine,
// through the box3d.js WebAssembly build) drops the objects so they fall into a
// pile. It owns no rendering; it only writes positions and rotations into the
// shared GlassObjects arrays.
import type { Box3DModule, b3BodyId, b3WorldId } from 'box3d.js';
import { KIND, MAX_OBJECTS, diamondPoints, type GlassObjects } from '../glass';
import type { Pile } from './pile';
import { spawnObjects, meshExtremes } from './spawn';

/** The unit torus as a ring of capsules (Box3D has no torus shape). */
const TORUS_SEGMENTS = 8;

/**
 * @param hulls points (x, y, z, ...) of the convex hulls of the shapes made from
 *   meshes (Suzanne, the dragon), at half-width 1
 */
export async function createPile(objects: GlassObjects, hulls: { suzanne: Float32Array; dragon: Float32Array }): Promise<Pile> {
  // The multithreaded build needs SharedArrayBuffer, which browsers only allow on
  // cross-origin-isolated pages (see the headers in vite.config.ts). Elsewhere it
  // falls back to one thread.
  const threaded = self.crossOriginIsolated === true;
  const factory = threaded ? (await import('box3d.js/mt-inline')).default : (await import('box3d.js/inline')).default;
  const b3: Box3DModule = await factory();
  const workers = threaded ? Math.min(Math.max(2, (navigator.hardwareConcurrency ?? 4) - 1), 12) : 0;
  const events = b3.createEventsBuffer();
  const moveEvent = b3.createBodyMoveEvent();
  let world: b3WorldId | null = null;
  /** Box3D body index -> index into the GlassObjects arrays. */
  let instanceOf = new Int32Array(0);
  let footprint = 2;

  // Box3D hulls are limited to 255 half-edges (about 40 corners), so keep only the
  // mesh shapes' extreme points along 26 directions.
  const suzannePoints = meshExtremes(hulls.suzanne).points;
  const dragonPoints = meshExtremes(hulls.dragon).points;

  /** Corner points of a convex shape, as a Box3D hull on `body`. */
  const addHull = (body: b3BodyId, shapeDef: ReturnType<Box3DModule['b3DefaultShapeDef']>, points: number[]) => {
    const hull = b3.b3CreateHull(points);
    b3.b3CreateHullShape(body, shapeDef, hull);
    hull?.delete();
  };

  /**
   * Throw away the current pile and drop `count` new random objects.
   * @param only  shape ids to draw from; null for the mix of every shape
   * @param cubic make boxes cubes (equal sides)
   * @param adjust called once they are spawned, to change any of them by hand
   */
  function reset(count: number, only: number[] | null = null, cubic = false, adjust?: () => void) {
    count = Math.min(Math.max(1, Math.round(count)), MAX_OBJECTS);
    if (world) b3.b3DestroyWorld(world);
    const worldDef = b3.b3DefaultWorldDef();
    worldDef.gravity = [0, -9.81, 0];
    worldDef.workerCount = workers;
    world = b3.b3CreateWorld(worldDef);

    const shapeDef = b3.b3DefaultShapeDef();
    // Grippy and dead (hardly any bounce), so the pile stops jostling soon.
    shapeDef.baseMaterial.friction = 0.65;
    shapeDef.baseMaterial.restitution = 0.02;
    // Without this, spheres and cylinders roll away across the flat ground for ever.
    shapeDef.baseMaterial.rollingResistance = 0.16;

    // The ground: a wide static slab whose top is at y = 0.
    const groundDef = b3.b3DefaultBodyDef();
    groundDef.position = [0, -0.5, 0];
    b3.b3CreateBoxShape(b3.b3CreateBody(world, groundDef), shapeDef, 400, 0.5, 400);

    footprint = spawnObjects(objects, count, only, cubic, adjust);
    const bodies: b3BodyId[] = [];
    for (let i = 0; i < count; i++) {
      const kind = objects.kind[i];
      const size = [objects.size[i * 3], objects.size[i * 3 + 1], objects.size[i * 3 + 2]];
      const position: [number, number, number] = [objects.center[i * 3], objects.center[i * 3 + 1], objects.center[i * 3 + 2]];
      const quat: [number, number, number, number] = [objects.quat[i * 4], objects.quat[i * 4 + 1], objects.quat[i * 4 + 2], objects.quat[i * 4 + 3]];

      const bodyDef = b3.b3DefaultBodyDef();
      bodyDef.type = b3.b3BodyType.b3_dynamicBody;
      bodyDef.position = position;
      bodyDef.rotation = quat;
      bodyDef.linearDamping = 0.08;
      bodyDef.angularDamping = 0.6;
      // Counts as at rest below this speed (m/s): well above Box3D's 0.05, so slow
      // creeping does not keep the whole pile awake.
      bodyDef.sleepThreshold = 0.25;
      const body = b3.b3CreateBody(world, bodyDef);
      if (kind === KIND.box) {
        b3.b3CreateBoxShape(body, shapeDef, size[0], size[1], size[2]);
      } else if (kind === KIND.sphere) {
        b3.b3CreateSphereShape(body, shapeDef, { center: [0, 0, 0], radius: size[0] });
      } else if (kind === KIND.cylinder) {
        // b3CreateCylinder(full height, radius, y of the base, sides)
        const hull = b3.b3CreateCylinder(2 * size[1], size[0], -size[1], 16);
        b3.b3CreateHullShape(body, shapeDef, hull);
        hull?.delete();
      } else if (kind === KIND.capsule) {
        b3.b3CreateCapsuleShape(body, shapeDef, { center1: [0, -size[1], 0], center2: [0, size[1], 0], radius: size[0] });
      } else if (kind === KIND.cone) {
        // A ring of base corners plus the apex.
        const points = [0, size[1], 0];
        for (let k = 0; k < 16; k++) {
          const a = (k / 16) * 2 * Math.PI;
          points.push(size[0] * Math.cos(a), -size[1], size[0] * Math.sin(a));
        }
        addHull(body, shapeDef, points);
      } else if (kind === KIND.torus) {
        // Capsules laid end to end around the ring, so the hole stays open.
        for (let k = 0; k < TORUS_SEGMENTS; k++) {
          const a0 = (k / TORUS_SEGMENTS) * 2 * Math.PI;
          const a1 = ((k + 1) / TORUS_SEGMENTS) * 2 * Math.PI;
          b3.b3CreateCapsuleShape(body, shapeDef, {
            center1: [size[0] * Math.cos(a0), 0, size[0] * Math.sin(a0)],
            center2: [size[0] * Math.cos(a1), 0, size[0] * Math.sin(a1)],
            radius: size[1],
          });
        }
      } else if (kind === KIND.octahedron) {
        const r = size[0];
        addHull(body, shapeDef, [r, 0, 0, -r, 0, 0, 0, r, 0, 0, -r, 0, 0, 0, r, 0, 0, -r]);
      } else if (kind === KIND.prism) {
        // Triangle with its apex at y = t and base at y = -t / 2, extruded along z.
        const t = size[0];
        const points: number[] = [];
        for (const z of [-size[2], size[2]]) points.push(0, t, z, -0.8660254 * t, -0.5 * t, z, 0.8660254 * t, -0.5 * t, z);
        addHull(body, shapeDef, points);
      } else if (kind === KIND.hexprism) {
        // Corners on the X axis; size[0] is the distance to the side faces.
        const corner = size[0] / Math.cos(Math.PI / 6);
        const points: number[] = [];
        for (let k = 0; k < 6; k++) {
          const a = (k / 6) * 2 * Math.PI;
          for (const y of [-size[1], size[1]]) points.push(corner * Math.cos(a), y, corner * Math.sin(a));
        }
        addHull(body, shapeDef, points);
      } else if (kind === KIND.diamond) {
        addHull(body, shapeDef, diamondPoints(size[0]));
      } else if (kind === KIND.dragon) {
        addHull(body, shapeDef, dragonPoints.map((v) => v * size[0]));
      } else {
        addHull(body, shapeDef, suzannePoints.map((v) => v * size[0]));
      }
      bodies.push(body);
    }

    let maxIndex = 0;
    for (const b of bodies) maxIndex = Math.max(maxIndex, b.index1);
    instanceOf = new Int32Array(maxIndex + 1).fill(-1);
    bodies.forEach((b, i) => (instanceOf[b.index1] = i));
  }

  /** Advance one 60 Hz step. Returns how many objects moved. */
  function step(): number {
    if (!world) return 0;
    b3.b3World_Step(world, 1 / 60, 4);
    b3.getEvents(events, world);
    const moved = b3.getNumBodyMoveEvents(events);
    for (let k = 0; k < moved; k++) {
      b3.getBodyMoveEventAt(moveEvent, events, k);
      const i = instanceOf[moveEvent.bodyId.index1];
      if (i === undefined || i < 0) continue;
      objects.center.set(moveEvent.position, i * 3);
      objects.quat.set(moveEvent.rotation, i * 4);
    }
    return moved;
  }

  return { reset, step, footprint: () => footprint, label: () => `${Math.max(workers, 1)} threads` };
}
