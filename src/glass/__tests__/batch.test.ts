import { describe, expect, it } from 'vitest';
import * as THREE from 'three/webgpu';
import { InstanceBatch } from '../batch';
import { createGlassObjects, KIND } from '../objects';

describe('InstanceBatch (batch.ts)', () => {
  it('copies per-instance attributes on push and uploads only the active range on commit', () => {
    const base = new THREE.BoxGeometry(2, 2, 2);
    const batch = new InstanceBatch(base, 16);
    const objects = createGlassObjects();
    objects.count = 2;

    objects.kind[0] = KIND.box;
    objects.center.set([1, 2, 3], 0);
    objects.quat.set([0, 0.5, 0, 0.866], 0);
    objects.size.set([0.4, 0.5, 0.6], 0);
    objects.vary.set([0.1, 1.2, 0.85], 0);
    objects.tint.set([0.8, 0.9, 1.0], 0);

    objects.kind[1] = KIND.box;
    objects.center.set([-4, 5, -6], 3);
    objects.quat.set([0, 0, 0, 1], 4);
    objects.size.set([0.2, 0.2, 0.2], 3);
    objects.vary.set([-0.05, 0.7, 0], 3);
    objects.tint.set([1, 0.5, 0.25], 3);

    // Committing an empty batch that was never uploaded should not schedule GPU uploads.
    batch.commit();
    expect(batch.geometry.instanceCount).toBe(0);
    expect(batch.center.updateRanges).toHaveLength(0);

    batch.push(objects, 1);
    batch.push(objects, 0);
    batch.commit();

    expect(batch.geometry.instanceCount).toBe(2);
    expect(Array.from(batch.center.array.slice(0, 6))).toEqual([-4, 5, -6, 1, 2, 3]);
    expect(Array.from(batch.quat.array.slice(0, 8))).toEqual([0, 0, 0, 1, 0, 0.5, 0, Math.fround(0.866)]);
    expect(Array.from(batch.size.array.slice(0, 6))).toEqual([
      Math.fround(0.2),
      Math.fround(0.2),
      Math.fround(0.2),
      Math.fround(0.4),
      Math.fround(0.5),
      Math.fround(0.6),
    ]);
    expect(Array.from(batch.tint.array.slice(0, 6))).toEqual([1, 0.5, 0.25, Math.fround(0.8), Math.fround(0.9), 1]);
    expect(batch.center.updateRanges).toEqual([{ start: 0, count: 6 }]);
    expect(batch.quat.updateRanges).toEqual([{ start: 0, count: 8 }]);

    // Emptying a previously non-empty batch resets instanceCount to 0.
    batch.count = 0;
    batch.commit();
    expect(batch.geometry.instanceCount).toBe(0);
  });
});
