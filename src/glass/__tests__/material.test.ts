import { describe, expect, it } from 'vitest';
import * as THREE from 'three/webgpu';
import * as TSL from 'three/tsl';
import { MeshGlassMaterial } from '../material';

describe('MeshGlassMaterial standalone API', () => {
  it('initializes with realistic crown-glass defaults matching MeshPhysicalMaterial', () => {
    const mat = new MeshGlassMaterial();
    expect(mat).toBeInstanceOf(THREE.MeshBasicNodeMaterial);
    expect(mat.ior).toBeCloseTo(1.52);
    expect(mat.dispersion).toBeCloseTo(0.5);
    expect(mat.roughness).toBe(0);
    expect(mat.frostScale).toBe(4);
    expect(mat.thickness).toBe(1);
    expect(mat.transmission).toBe(0);
    expect(mat.transparent).toBe(false);
    expect(mat.envMap).toBeNull();
    expect(mat.envMapIntensity).toBe(1);
    expect(mat.colorNode).toBeTruthy();
  });

  it('applies constructor optical parameters and enables transparent viewport capture when transmission > 0', () => {
    const mat = new MeshGlassMaterial({
      ior: 1.66,
      dispersion: 0.8,
      roughness: 0.35,
      frostScale: 8,
      color: '#88ccff',
      thickness: 2.5,
      transmission: 1,
      envMapIntensity: 1.4,
    });
    expect(mat.ior).toBeCloseTo(1.66);
    expect(mat.dispersion).toBeCloseTo(0.8);
    expect(mat.roughness).toBeCloseTo(0.35);
    expect(mat.frostScale).toBeCloseTo(8);
    expect(mat.thickness).toBeCloseTo(2.5);
    expect(mat.transmission).toBe(1);
    expect(mat.transparent).toBe(true);
    expect(mat.envMapIntensity).toBeCloseTo(1.4);
  });

  it('supports live reactive optical updates without recompiling', () => {
    const mat = new MeshGlassMaterial();
    const originalNode = mat.colorNode;
    mat.ior = 1.66;
    mat.dispersion = 1.2;
    mat.roughness = 0.5;
    mat.frostScale = 10;
    mat.thickness = 1.8;
    expect(mat.ior).toBeCloseTo(1.66);
    expect(mat.dispersion).toBeCloseTo(1.2);
    expect(mat.roughness).toBeCloseTo(0.5);
    expect(mat.frostScale).toBeCloseTo(10);
    expect(mat.thickness).toBeCloseTo(1.8);
    expect(mat.colorNode).toBe(originalNode);
  });

  it('toggles viewport transmission graph only when crossing 0', () => {
    const mat = new MeshGlassMaterial({ transmission: 0 });
    const opaqueNode = mat.colorNode;
    mat.transmission = 0.8;
    expect(mat.transparent).toBe(true);
    const transNode = mat.colorNode;
    expect(transNode).not.toBe(opaqueNode);

    // Varying transmission within (0, 1] updates uniform without recompiling.
    mat.transmission = 0.4;
    expect(mat.colorNode).toBe(transNode);

    // Returning to 0 disables viewport copy overhead.
    mat.transmission = 0;
    expect(mat.transparent).toBe(false);
    expect(mat.colorNode).not.toBe(transNode);
  });

  it('supports equirectangular envMap, envMapIntensity, and clone() without recompiling', () => {
    const hdr = new THREE.DataTexture(new Uint8Array([255, 255, 255, 255]), 1, 1);
    const mat = new MeshGlassMaterial({
      ior: 1.66,
      roughness: 0.25,
      transmission: 0.9,
      envMap: hdr,
      envMapIntensity: 1.4,
    });
    const originalNode = mat.colorNode;
    expect(mat.envMap).toBe(hdr);
    expect(mat.envMapIntensity).toBeCloseTo(1.4);

    mat.envMap = null;
    mat.envMapIntensity = 0.8;
    expect(mat.envMap).toBeNull();
    expect(mat.envMapIntensity).toBeCloseTo(0.8);
    expect(mat.colorNode).toBe(originalNode);

    mat.envMap = hdr;
    const copy = mat.clone();
    expect(copy).not.toBe(mat);
    expect(copy.ior).toBeCloseTo(1.66);
    expect(copy.roughness).toBeCloseTo(0.25);
    expect(copy.transmission).toBeCloseTo(0.9);
    expect(copy.envMap).toBe(hdr);
    expect(copy.envMapIntensity).toBeCloseTo(0.8);
    copy.ior = 2.0;
    expect(mat.ior).toBeCloseTo(1.66);
  });

  it('routes InstancedMesh instanceColor into Beer-Lambert tint via setupDiffuseColor', () => {
    const mat = new MeshGlassMaterial({ color: '#ffffff', ior: 1.52 });
    const inst = new THREE.InstancedMesh(new THREE.BoxGeometry(1, 1, 1), mat, 2);
    inst.setColorAt(0, new THREE.Color('#ff4466'));
    inst.setColorAt(1, new THREE.Color('#44ccff'));
    expect(inst.instanceColor).toBeTruthy();

    // Verify setupDiffuseColor runs cleanly inside a TSL Fn context for an InstancedMesh with instanceColor.
    expect(() =>
      TSL.Fn(() => {
        mat.setupDiffuseColor({ object: inst, geometry: inst.geometry });
        return TSL.vec4(1);
      })(),
    ).not.toThrow();
  });

  it('binds, clears, and clones backfaceMap reactively without recompiling', () => {
    const bf = new THREE.DataTexture(new Uint8Array([128, 255, 128, 255]), 1, 1);
    const mat = new MeshGlassMaterial({ backfaceMap: bf });
    const originalNode = mat.colorNode;
    expect(mat.backfaceMap).toBe(bf);

    const copy = mat.clone();
    expect(copy.backfaceMap).toBe(bf);

    mat.backfaceMap = null;
    expect(mat.backfaceMap).toBeNull();
    expect(mat.colorNode).toBe(originalNode);
  });
});

describe('MeshGlassMaterial exits and options', () => {
  it('defaults to 6 events, or 4 with a baked volume, and clamps bounces and imperfections', () => {
    expect(new MeshGlassMaterial().bounces).toBe(6);
    const volume = { texture: new THREE.Data3DTexture(new Uint16Array(8), 2, 2, 2) };
    const withVolume = new MeshGlassMaterial({ volume });
    expect(withVolume.bounces).toBe(4);
    expect(withVolume.volume).toBe(volume);
    withVolume.bounces = 0;
    expect(withVolume.bounces).toBe(1);
    withVolume.imperfections = 3;
    expect(withVolume.imperfections).toBe(1);
  });

  it('rebuilds its shader when a volume is attached or removed, and clones it', () => {
    const mat = new MeshGlassMaterial();
    const before = mat.colorNode;
    const volume = { texture: new THREE.Data3DTexture(new Uint16Array(8), 2, 2, 2), layer: 0, layerCount: 1 };
    mat.volume = volume;
    expect(mat.colorNode).not.toBe(before);
    const copy = mat.clone();
    expect(copy.volume).toBe(volume);
    expect(copy.bounces).toBe(mat.bounces);
    mat.volume = null;
    expect(mat.volume).toBeNull();
  });
});

describe('MeshGlassMaterial temporal option', () => {
  it('is off by default, clamps spectral samples, and clones both', () => {
    const mat = new MeshGlassMaterial();
    expect(mat.temporal).toBe(false);
    expect(mat.spectralSamples).toBe(0);
    mat.temporal = true;
    mat.spectralSamples = -4;
    expect(mat.spectralSamples).toBe(0);
    mat.spectralSamples = 9.6;
    expect(mat.spectralSamples).toBe(10);
    const copy = mat.clone();
    expect(copy.temporal).toBe(true);
    expect(copy.spectralSamples).toBe(10);
  });
});
