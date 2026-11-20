import { describe, expect, it } from 'vitest';
import { DIAMOND_CUTS, KIND, MAX_OBJECTS, createGlassObjects, type DiamondCut } from '../../glass';
import { layoutGrid, restingPose } from '../grid';

describe('demo grid layout & resting poses', () => {
  it('computes valid positive height and unit quaternion for every shape and diamond cut', () => {
    for (const kind of Object.values(KIND)) {
      const { height, quat } = restingPose(kind, [0.5, 0.6, 0.4], 0.7);
      expect(height).toBeGreaterThan(0);
      expect(Number.isFinite(height)).toBe(true);
      expect(quat.length()).toBeCloseTo(1, 5);
    }
    for (const cut of Object.keys(DIAMOND_CUTS) as DiamondCut[]) {
      const { height, quat } = restingPose(KIND.diamond, [0.6, 0.6, 0.6], 1.2, cut);
      expect(height).toBeCloseTo(DIAMOND_CUTS[cut].rest * 0.6, 5);
      expect(quat.length()).toBeCloseTo(1, 5);
    }
  });

  it('lays out deterministic flat floor grids and respects shape filters', () => {
    const a = createGlassObjects();
    const b = createGlassObjects();
    const boundsA = layoutGrid(a, 16, [KIND.box], true, 'flat');
    const boundsB = layoutGrid(b, 16, [KIND.box], true, 'flat');

    expect(a.count).toBe(16);
    expect(boundsA.centerY).toBe(0);
    expect(boundsA.reach).toBeGreaterThan(0);
    expect(boundsA).toEqual(boundsB);

    for (let i = 0; i < 16; i++) {
      expect(a.kind[i]).toBe(KIND.box);
      // Cubic boxes have equal X, Y, Z half-extents.
      expect(a.size[i * 3]).toBeCloseTo(a.size[i * 3 + 1], 5);
      expect(a.size[i * 3]).toBeCloseTo(a.size[i * 3 + 2], 5);
      // Deterministic positions.
      expect(a.center[i * 3]).toBe(b.center[i * 3]);
      expect(a.center[i * 3 + 1]).toBe(b.center[i * 3 + 1]);
      expect(a.center[i * 3 + 2]).toBe(b.center[i * 3 + 2]);
    }
  });

  it('stacks objects vertically in 3D grid layout and clamps invalid counts', () => {
    const objs = createGlassObjects();
    const bounds = layoutGrid(objs, 27, [KIND.sphere], false, 'grid');
    expect(objs.count).toBe(27);
    expect(bounds.centerY).toBeGreaterThan(0);

    // 27 spheres in a 3x3x3 cube should span 3 distinct vertical layers.
    const ys = Array.from({ length: 27 }, (_, i) => objs.center[i * 3 + 1]);
    expect(Math.max(...ys) - Math.min(...ys)).toBeGreaterThan(2.0);

    // Clamping: 0 -> 1, above MAX_OBJECTS -> MAX_OBJECTS.
    layoutGrid(objs, 0);
    expect(objs.count).toBe(1);
    layoutGrid(objs, MAX_OBJECTS + 500);
    expect(objs.count).toBe(MAX_OBJECTS);
  });
});
