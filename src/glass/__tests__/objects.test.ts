import { describe, expect, it } from 'vitest';
import { KIND, MAX_OBJECTS, createGlassObjects } from '../objects';

describe('GlassObjects SoA storage', () => {
  it('assigns unique contiguous shape IDs starting at 0', () => {
    const ids = Object.values(KIND).sort((a, b) => a - b);
    expect(ids).toEqual(Array.from({ length: ids.length }, (_, i) => i));
    expect(KIND.torusknot).toBe(12);
  });

  it('allocates flat typed arrays with clear white default tints', () => {
    const objs = createGlassObjects();
    expect(objs.count).toBe(0);
    expect(objs.kind).toBeInstanceOf(Uint8Array);
    expect(objs.kind.length).toBe(MAX_OBJECTS);
    expect(objs.center.length).toBe(MAX_OBJECTS * 3);
    expect(objs.quat.length).toBe(MAX_OBJECTS * 4);
    expect(objs.size.length).toBe(MAX_OBJECTS * 3);
    expect(objs.vary.length).toBe(MAX_OBJECTS * 3);
    expect(objs.tint.length).toBe(MAX_OBJECTS * 3);
    // Default tint is clear white (1, 1, 1).
    expect(objs.tint[0]).toBe(1);
    expect(objs.tint[1]).toBe(1);
    expect(objs.tint[2]).toBe(1);
    expect(objs.tint[MAX_OBJECTS * 3 - 1]).toBe(1);
  });
});
