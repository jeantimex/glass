import { describe, expect, it } from 'vitest';
import { float, vec3 } from 'three/tsl';
import { CMY_FACE_ORDERS, cmyChannelColor, cmyTint, type CmyFaceOrder } from '../tsl/cmy';

describe('CMY dichroic cube filters (cmy.ts)', () => {
  const allOrders = Object.keys(CMY_FACE_ORDERS) as CmyFaceOrder[];

  it('defines all 6 unique permutations of [0, 1, 2] across [X, Y, Z]', () => {
    expect(allOrders).toHaveLength(6);
    const seen = new Set<string>();
    for (const key of allOrders) {
      const tuple = CMY_FACE_ORDERS[key];
      expect(tuple).toHaveLength(3);
      expect([...tuple].sort()).toEqual([0, 1, 2]);
      seen.add(tuple.join(','));
    }
    expect(seen.size).toBe(6);
    expect(CMY_FACE_ORDERS.cym).toEqual([0, 1, 2]);
  });

  it('builds TSL nodes for cmyChannelColor and cmyTint with and without custom order', () => {
    expect(cmyChannelColor(float(0))).toBeTruthy();
    expect(cmyChannelColor(float(1))).toBeTruthy();
    expect(cmyChannelColor(float(2))).toBeTruthy();
    expect(cmyTint(vec3(1, 0, 0), float(0.85))).toBeTruthy();
    expect(cmyTint(vec3(0, 1, 0), float(0.85), vec3(1, 0, 2))).toBeTruthy();
  });
});
