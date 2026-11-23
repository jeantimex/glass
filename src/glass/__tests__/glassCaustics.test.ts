import { describe, expect, it } from 'vitest';
import * as THREE from 'three/webgpu';
import { GlassCaustics } from '../glassCaustics';

type WithColorNode = THREE.DirectionalLight & { colorNode?: unknown };

describe('GlassCaustics (glassCaustics.ts)', () => {
  it('takes its defaults and options', () => {
    const light = new THREE.DirectionalLight();
    const defaults = new GlassCaustics({ light });
    expect(defaults).toMatchObject({ floorY: 0, extent: 4, softness: 0.05, headroom: 24 });
    expect(defaults.target.width).toBe(512);
    const custom = new GlassCaustics({ light, floorY: -0.5, extent: 2, softness: 0.1, resolution: 256, headroom: 4 });
    expect(custom).toMatchObject({ floorY: -0.5, extent: 2, softness: 0.1, headroom: 4 });
    expect(custom.target.width).toBe(256);
    expect(custom.photonMap.width).toBe(256);
  });

  it('attach() routes the light through the caustics, detach() restores it, and dispose() detaches', () => {
    const light = new THREE.DirectionalLight() as WithColorNode;
    const caustics = new GlassCaustics({ light });
    expect(light.colorNode).toBeUndefined();
    caustics.attach();
    const attached = light.colorNode;
    expect(attached).toBeTruthy();
    caustics.attach(); // a second attach changes nothing
    expect(light.colorNode).toBe(attached);
    caustics.detach();
    expect(light.colorNode).toBeUndefined();
    caustics.attach();
    caustics.dispose();
    expect(light.colorNode).toBeUndefined();
  });

  it('keeps a light colour node that was there before', () => {
    const light = new THREE.DirectionalLight() as WithColorNode;
    const own = { isNode: true };
    light.colorNode = own;
    const caustics = new GlassCaustics({ light });
    caustics.attach();
    expect(light.colorNode).not.toBe(own);
    caustics.detach();
    expect(light.colorNode).toBe(own);
  });

  it('builds the floor lighting lookup for a custom material', () => {
    const caustics = new GlassCaustics({ light: new THREE.DirectionalLight() });
    expect(caustics.lightAt()).toBeTruthy();
  });
});
