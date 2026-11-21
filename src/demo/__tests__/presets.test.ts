import { describe, expect, it } from 'vitest';
import { KIND } from '../../glass';
import {
  createDefaultSettings,
  MAX_COUNT,
  parseInitialView,
  SHAPES,
  shapeSlug,
  SINGLE_SHAPE_SPECS,
  srgbToLinear,
} from '../presets';

describe('demo presets & helpers (presets.ts)', () => {
  it('returns independent default setting objects with depthOfField = 0', () => {
    const a = createDefaultSettings();
    const b = createDefaultSettings();
    expect(a.depthOfField).toBe(0);
    expect(a.cmyOrder).toBe('cym');
    expect(a.floorAlpha).toBe(1);
    expect(a.floorGridAlpha).toBe(1);
    a.ior = 2.0;
    expect(b.ior).toBeCloseTo(1.52);
  });

  it('provides SINGLE_SHAPE_SPECS for every KIND and valid presets in SHAPES', () => {
    for (const kind of Object.values(KIND)) {
      const spec = SINGLE_SHAPE_SPECS[kind];
      expect(spec).toBeTruthy();
      expect(spec.labels.length).toBeGreaterThan(0);
      expect(spec.measures.length).toBe(spec.labels.length);
    }
    expect(SHAPES.Diamond.kinds).toEqual([KIND.diamond]);
    expect(SHAPES['CMY cube'].cmy).toBe(true);
    expect(SHAPES['Mixed shapes'].kinds).toBeNull();
  });

  it('converts sRGB to linear monotonically across the piecewise threshold', () => {
    expect(srgbToLinear(0)).toBe(0);
    expect(srgbToLinear(1)).toBeCloseTo(1, 6);
    expect(srgbToLinear(0.04045)).toBeCloseTo(0.04045 / 12.92, 6);
    expect(srgbToLinear(0.5)).toBeCloseTo(0.214041, 4);
    expect(srgbToLinear(0.2)).toBeLessThan(srgbToLinear(0.5));
  });

  it('formats URL slugs and parses query parameters including legacy ?demo=', () => {
    expect(shapeSlug('CMY cube')).toBe('cmy-cube');
    expect(shapeSlug('Torus knot')).toBe('torus-knot');

    expect(parseInitialView(new URLSearchParams(''))).toEqual({ shape: 'Diamond', count: 1, layout: 'flat' });
    expect(parseInitialView(new URLSearchParams('shape=cmy-cube&count=27&layout=grid'))).toEqual({
      shape: 'CMY cube',
      count: 27,
      layout: 'grid',
    });
    expect(parseInitialView(new URLSearchParams('demo=pile-diamonds'))).toEqual({
      shape: 'Diamond',
      count: 2000,
      layout: 'flat',
    });
    expect(parseInitialView(new URLSearchParams('count=999999')).count).toBe(MAX_COUNT);
    expect(parseInitialView(new URLSearchParams('count=-5')).count).toBe(1);
  });
});
