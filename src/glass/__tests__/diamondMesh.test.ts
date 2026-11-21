import { describe, expect, it } from 'vitest';
import * as THREE from 'three/webgpu';
import { DIAMOND_CUTS, type DiamondCut } from '../diamond';
import { DiamondMesh } from '../diamondMesh';

const ALL_CUTS: DiamondCut[] = ['round', 'princess', 'emerald', 'asscher', 'cushion', 'radiant'];

describe('DiamondMesh', () => {
  it('initializes with real diamond optics and round brilliant geometry', () => {
    const diamond = new DiamondMesh();
    expect(diamond).toBeInstanceOf(THREE.Mesh);
    expect(diamond.cut).toBe('round');
    expect(diamond.ior).toBeCloseTo(2.42);
    expect(diamond.dispersion).toBeCloseTo(0.4);
    expect(diamond.bounces).toBe(6);
    expect(diamond.roughness).toBe(0);
    expect(diamond.transmission).toBe(0);
    expect(diamond.restHeight).toBeCloseTo(DIAMOND_CUTS.round.rest);
    expect(diamond.restLean).toBeCloseTo(DIAMOND_CUTS.round.lean);
    expect(diamond.geometry.getAttribute('position').count).toBeGreaterThan(20);
    diamond.dispose();
  });

  it('switches cuts atomically (geometry + facet shader) and positions flush via restOnFloor()', () => {
    const diamond = new DiamondMesh({ cut: 'round' });
    for (const cut of ALL_CUTS) {
      diamond.cut = cut;
      expect(diamond.cut).toBe(cut);
      expect(diamond.restHeight).toBeCloseTo(DIAMOND_CUTS[cut].rest);
      diamond.restOnFloor(1.5);
      expect(diamond.rotation.z).toBeCloseTo(-DIAMOND_CUTS[cut].lean);
      expect(diamond.position.y).toBeCloseTo(1.5 + DIAMOND_CUTS[cut].rest);
    }
    diamond.dispose();
  });

  it('supports reactive optics, envMap, and transmission toggling', () => {
    const hdr = new THREE.DataTexture(new Uint8Array([255, 255, 255, 255]), 1, 1);
    const diamond = new DiamondMesh({ cut: 'princess', envMap: hdr, envMapIntensity: 1.5, transmission: 1 });
    expect(diamond.envMap).toBe(hdr);
    expect(diamond.envMapIntensity).toBeCloseTo(1.5);
    expect(diamond.transmission).toBe(1);

    diamond.ior = 2.45;
    diamond.dispersion = 0.7;
    diamond.bounces = 8;
    diamond.roughness = 0.2;
    diamond.transmission = 0;
    expect(diamond.ior).toBeCloseTo(2.45);
    expect(diamond.dispersion).toBeCloseTo(0.7);
    expect(diamond.bounces).toBe(8);
    expect(diamond.roughness).toBeCloseTo(0.2);
    expect(diamond.transmission).toBe(0);
    diamond.dispose();
  });
});
