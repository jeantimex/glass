// The objects the glass renderer draws, in flat arrays (one entry per object)
// that whatever moves them (a physics engine, an animation, code placing them by
// hand) writes into. Nothing here knows about physics.

export const MAX_OBJECTS = 20000;

/** Shape ids, shared with the shaders (tsl/glass.ts). */
export const KIND = {
  box: 0,
  sphere: 1,
  cylinder: 2,
  capsule: 3,
  cone: 4,
  torus: 5,
  octahedron: 6,
  prism: 7,
  hexprism: 8,
  suzanne: 9,
  diamond: 10,
  dragon: 11,
  torusknot: 12,
} as const;

/**
 * The objects to draw. `size` is per shape (see the table in tsl/glass.ts):
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

/** Allocate an empty flat-array `GlassObjects` container with capacity for `MAX_OBJECTS` instances. */
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
