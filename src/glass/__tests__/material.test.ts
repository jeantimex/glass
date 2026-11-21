import { describe, expect, it } from 'vitest';
import * as THREE from 'three/webgpu';
import { GlassMaterial, type GlassShape } from '../material';

const ALL_SHAPES: GlassShape[] = [
  'mesh',
  'box',
  'sphere',
  'cylinder',
  'capsule',
  'cone',
  'torus',
  'octahedron',
  'prism',
  'hexprism',
  'diamond',
];

describe('GlassMaterial standalone API', () => {
  it('initializes with realistic crown-glass defaults', () => {
    const mat = new GlassMaterial();
    expect(mat).toBeInstanceOf(THREE.MeshBasicNodeMaterial);
    expect(mat.shape).toBe('mesh');
    expect(mat.diamondCut).toBe('round');
    expect(mat.cmyOrder).toBe('cym');
    expect(mat.ior).toBeCloseTo(1.52);
    expect(mat.dispersion).toBeCloseTo(0.5);
    expect(mat.roughness).toBe(0);
    expect(mat.frostScale).toBe(4);
    expect(mat.colorAlpha).toBe(1);
    expect(mat.thickness).toBe(1);
    expect(mat.bounces).toBe(6);
    expect(mat.dark).toBe(false);
    expect(mat.ground).toBe(true);
    expect(mat.filmic).toBe(false);
    expect(mat.envMap).toBeNull();
    expect(mat.envMapIntensity).toBe(1);
    expect(mat.colorNode).toBeTruthy();
  });

  it('applies constructor parameters and normalizes lampDirection', () => {
    const mat = new GlassMaterial({
      ior: 2.42,
      dispersion: 0.4,
      roughness: 0.35,
      frostScale: 8,
      color: '#88ccff',
      colorAlpha: 0.75,
      thickness: 2.5,
      bounces: 8,
      bevel: 0.02,
      cmy: 0.85,
      dark: true,
      ground: false,
      groundColor: '#222222',
      lampDirection: [3, 4, 0],
      exposure: 1.5,
      filmic: false,
      shape: 'box',
      shapeSize: [0.5, 0.6, 0.7],
    });
    expect(mat.ior).toBeCloseTo(2.42);
    expect(mat.dispersion).toBeCloseTo(0.4);
    expect(mat.roughness).toBeCloseTo(0.35);
    expect(mat.frostScale).toBeCloseTo(8);
    expect(mat.colorAlpha).toBeCloseTo(0.75);
    expect(mat.thickness).toBeCloseTo(2.5);
    expect(mat.bounces).toBe(8);
    expect(mat.bevel).toBeCloseTo(0.02);
    expect(mat.cmy).toBeCloseTo(0.85);
    expect(mat.dark).toBe(true);
    expect(mat.ground).toBe(false);
    expect(mat.exposure).toBeCloseTo(1.5);
    expect(mat.filmic).toBe(false);
    expect(mat.lampDirection.length()).toBeCloseTo(1, 5);
    expect(mat.lampDirection.x).toBeCloseTo(0.6, 5);
    expect(mat.lampDirection.y).toBeCloseTo(0.8, 5);
    expect(mat.shapeSize.toArray()).toEqual([0.5, 0.6, 0.7]);
  });

  it('supports live reactive property updates without recompiling', () => {
    const mat = new GlassMaterial();
    const originalNode = mat.colorNode;
    mat.ior = 1.66;
    mat.dispersion = 1.2;
    mat.roughness = 0.5;
    mat.frostScale = 10;
    mat.colorAlpha = 0.4;
    mat.dark = true;
    mat.ground = false;
    mat.filmic = false;
    mat.exposure = 1.25;
    mat.cmyOrder = 'ycm';
    expect(mat.ior).toBeCloseTo(1.66);
    expect(mat.dispersion).toBeCloseTo(1.2);
    expect(mat.roughness).toBeCloseTo(0.5);
    expect(mat.frostScale).toBeCloseTo(10);
    expect(mat.colorAlpha).toBeCloseTo(0.4);
    expect(mat.dark).toBe(true);
    expect(mat.ground).toBe(false);
    expect(mat.filmic).toBe(false);
    expect(mat.exposure).toBeCloseTo(1.25);
    expect(mat.cmyOrder).toBe('ycm');
    expect(mat.colorNode).toBe(originalNode);
  });

  it('builds a valid colorNode for every supported GlassShape', () => {
    for (const shape of ALL_SHAPES) {
      const mat = new GlassMaterial({ shape });
      expect(mat.colorNode).toBeTruthy();
    }
  });

  it('rebuilds colorNode when diamondCut changes on a diamond GlassMaterial', () => {
    const mat = new GlassMaterial({ shape: 'diamond', diamondCut: 'round' });
    const roundNode = mat.colorNode;
    mat.diamondCut = 'round';
    expect(mat.colorNode).toBe(roundNode);
    mat.diamondCut = 'princess';
    expect(mat.diamondCut).toBe('princess');
    expect(mat.colorNode).not.toBe(roundNode);
  });

  it('accepts and clears screen-space background textures cleanly', () => {
    const mat = new GlassMaterial();
    const tex = new THREE.DataTexture(new Uint8Array([255, 0, 0, 255]), 1, 1);
    const blur = new THREE.DataTexture(new Uint8Array([0, 255, 0, 255]), 1, 1);
    expect(() => mat.setBackground(tex, blur)).not.toThrow();
    expect(() => mat.setBackground(tex)).not.toThrow();
    expect(() => mat.setBackground(null)).not.toThrow();
  });

  it('supports equirectangular envMap, envMapIntensity, and clone() without recompiling', () => {
    const hdr = new THREE.DataTexture(new Uint8Array([255, 255, 255, 255]), 1, 1);
    const mat = new GlassMaterial({ ior: 1.66, roughness: 0.25, envMap: hdr, envMapIntensity: 1.4, filmic: true });
    const originalNode = mat.colorNode;
    expect(mat.envMap).toBe(hdr);
    expect(mat.envMapIntensity).toBeCloseTo(1.4);
    expect(mat.filmic).toBe(true);

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
    expect(copy.envMap).toBe(hdr);
    expect(copy.envMapIntensity).toBeCloseTo(0.8);
    expect(copy.filmic).toBe(true);
    copy.ior = 2.0;
    expect(mat.ior).toBeCloseTo(1.66);
  });
});
