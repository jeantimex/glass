import { describe, expect, it } from 'vitest';
import * as THREE from 'three/webgpu';
import { autoGlassVolume, autoGlassVolumesSettled } from '../autoVolume';
import { MeshGlassMaterial } from '../material';

// (Without Web Workers, as here, bakeGlassVolumeAsync bakes on this thread; ~1-2 s each.)
describe('automatic glass volumes (autoVolume.ts)', { timeout: 30_000 }, () => {
  it('bakes each geometry once, tells every waiting material once, then serves the cached volume', async () => {
    const geometry = new THREE.TorusKnotGeometry(0.5, 0.15, 64, 12);
    let calls = 0;
    const onReady = () => {
      calls++;
    };
    expect(autoGlassVolume(geometry, onReady)).toBeNull(); // starts the bake
    expect(autoGlassVolume(geometry, onReady)).toBeNull(); // same bake, same waiter
    let otherCalls = 0;
    autoGlassVolume(geometry, () => otherCalls++); // a second material waiting
    await autoGlassVolumesSettled();
    expect(calls).toBe(1);
    expect(otherCalls).toBe(1);
    const volume = autoGlassVolume(geometry, onReady);
    expect(volume).not.toBeNull();
    expect(volume!.texture.image.width).toBe(128);
    expect(volume!.halfWidth).toBeGreaterThan(0);
    expect(autoGlassVolume(geometry, onReady)).toBe(volume); // cached
    expect(calls).toBe(1);
  });

  it('changes the build cache key when private settings change, so three.js rebuilds', () => {
    // three.js leaves properties starting with "_" out of its cache key: without this, setting
    // exactShapes or bounces (or an automatic volume becoming ready) would not rebuild anything.
    const mat = new MeshGlassMaterial();
    const keys = new Set([mat.customProgramCacheKey()]);
    mat.exactShapes = false;
    keys.add(mat.customProgramCacheKey());
    mat.bounces = 3;
    keys.add(mat.customProgramCacheKey());
    mat.volume = null;
    keys.add(mat.customProgramCacheKey());
    expect(keys.size).toBe(4);
  });

  it("defaults MeshGlassMaterial to 'auto', which null turns off", () => {
    expect(new MeshGlassMaterial().volume).toBe('auto');
    expect(new MeshGlassMaterial({ volume: null }).volume).toBeNull();
    // 'auto' alone does not change the bounce default (6); a clone keeps the exact settings.
    const mat = new MeshGlassMaterial();
    expect(mat.bounces).toBe(6);
    expect(mat.clone().volume).toBe('auto');
  });
});
