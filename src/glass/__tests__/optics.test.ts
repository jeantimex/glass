import { describe, expect, it } from 'vitest';
import * as THREE from 'three/webgpu';
import { float, texture, texture3D, uniform, vec3, vec4 } from 'three/tsl';
import { DIAMOND_CUTS, type DiamondCut } from '../diamond';
import { KIND } from '../objects';
import {
  causticNode,
  glassNode,
  photonNode,
  SPECTRAL_BANDS,
  SPECTRAL_BANDS_FAST,
  SPECTRAL_COUNT,
  SPECTRAL_FAST_COUNT,
} from '../tsl/glass';

/** Reference CPU implementation of our exact Cook-Torrance dielectric Fresnel formula (glass.ts). */
function dielectricFresnel(cosI: number, eta: number): number {
  const c = Math.abs(cosI);
  const g2 = eta * eta - 1 + c * c;
  if (g2 <= 0) return 1;
  const g = Math.sqrt(g2);
  const a = (g - c) / (g + c);
  const b = (c * (g + c) - 1) / (c * (g - c) + 1);
  return 0.5 * a * a * (b * b + 1);
}

describe('optical physics invariants', () => {
  it('matches exact Fresnel reflectance at normal incidence, Brewster angle, grazing, and TIR', () => {
    const n = 1.52;
    // Normal incidence: R0 = ((n - 1) / (n + 1))^2 ≈ 0.04258
    const r0 = ((n - 1) / (n + 1)) ** 2;
    expect(dielectricFresnel(1, n)).toBeCloseTo(r0, 6);
    expect(dielectricFresnel(1, 1 / n)).toBeCloseTo(r0, 6);

    // Diamond normal incidence: R0 ≈ 0.172
    const r0Diamond = ((2.42 - 1) / (2.42 + 1)) ** 2;
    expect(dielectricFresnel(1, 2.42)).toBeCloseTo(r0Diamond, 6);

    // Brewster's angle (tan(theta) = n): Rp = 0, so R = 0.5 * Rs
    const cosB = 1 / Math.sqrt(1 + n * n);
    const sinB = n / Math.sqrt(1 + n * n);
    const cosT = sinB; // theta_B + theta_T = 90 deg
    const rs = (cosB - n * cosT) / (cosB + n * cosT);
    expect(dielectricFresnel(cosB, n)).toBeCloseTo(0.5 * rs * rs, 6);

    // Grazing incidence -> R = 1
    expect(dielectricFresnel(1e-6, n)).toBeCloseTo(1, 4);

    // Total Internal Reflection (glass -> air, eta = 1/n, sin(theta_i) >= 1/n)
    const criticalCos = Math.sqrt(1 - (1 / n) ** 2);
    expect(dielectricFresnel(criticalCos - 0.01, 1 / n)).toBe(1);
    expect(dielectricFresnel(criticalCos + 0.05, 1 / n)).toBeLessThan(1);
  });

  it('satisfies Cauchy dispersion Fraunhofer C/d/F ratios', () => {
    // 656.3 nm (C, red), 587.6 nm (d, yellow/green), 486.1 nm (F, blue)
    const invC2 = 1 / 0.6563 ** 2;
    const invD2 = 1 / 0.5876 ** 2;
    const invF2 = 1 / 0.4861 ** 2;
    const redWeight = (invC2 - invD2) / (invF2 - invC2);
    const blueWeight = (invF2 - invD2) / (invF2 - invC2);
    expect(redWeight).toBeCloseTo(-0.301, 3);
    expect(blueWeight).toBeCloseTo(0.699, 3);
    expect(blueWeight - redWeight).toBeCloseTo(1, 5);
  });

  it('has energy-conserving, non-negative 6-band CIE 1931 spectral weights matching Cauchy law', () => {
    expect(SPECTRAL_BANDS.length).toBe(SPECTRAL_COUNT);
    expect(SPECTRAL_COUNT).toBe(6);
    const invC2 = 1 / 656.3 ** 2;
    const invD2 = 1 / 587.6 ** 2;
    const invF2 = 1 / 486.1 ** 2;
    let sumR = 0;
    let sumG = 0;
    let sumB = 0;
    for (let i = 0; i < SPECTRAL_BANDS.length; i++) {
      const band = SPECTRAL_BANDS[i];
      const expectedCauchy = (1 / band.wavelength ** 2 - invD2) / (invF2 - invC2);
      expect(band.cauchy).toBeCloseTo(expectedCauchy, 3);
      if (i > 0) expect(band.cauchy).toBeLessThan(SPECTRAL_BANDS[i - 1].cauchy);
      expect(band.rgb[0]).toBeGreaterThanOrEqual(0);
      expect(band.rgb[1]).toBeGreaterThanOrEqual(0);
      expect(band.rgb[2]).toBeGreaterThanOrEqual(0);
      sumR += band.rgb[0];
      sumG += band.rgb[1];
      sumB += band.rgb[2];
    }
    expect(sumR).toBeCloseTo(1, 4);
    expect(sumG).toBeCloseTo(1, 4);
    expect(sumB).toBeCloseTo(1, 4);
  });

  it('has energy-conserving, non-negative 3-band CIE 1931 fast spectral weights', () => {
    expect(SPECTRAL_BANDS_FAST.length).toBe(SPECTRAL_FAST_COUNT);
    expect(SPECTRAL_FAST_COUNT).toBe(3);
    let sumR = 0;
    let sumG = 0;
    let sumB = 0;
    for (let i = 0; i < SPECTRAL_BANDS_FAST.length; i++) {
      const band = SPECTRAL_BANDS_FAST[i];
      if (i > 0) expect(band.cauchy).toBeLessThan(SPECTRAL_BANDS_FAST[i - 1].cauchy);
      expect(band.rgb[0]).toBeGreaterThanOrEqual(0);
      expect(band.rgb[1]).toBeGreaterThanOrEqual(0);
      expect(band.rgb[2]).toBeGreaterThanOrEqual(0);
      sumR += band.rgb[0];
      sumG += band.rgb[1];
      sumB += band.rgb[2];
    }
    expect(sumR).toBeCloseTo(1, 4);
    expect(sumG).toBeCloseTo(1, 4);
    expect(sumB).toBeCloseTo(1, 4);
  });

  it('satisfies Beer-Lambert volumetric absorption invariants', () => {
    const reach = 0.6;
    const beerLambert = (tint: number, dist: number) => {
      const sigma = -Math.log(Math.max(tint, 1e-4)) / (2 * reach);
      return Math.exp(-sigma * dist);
    };
    // Clear glass (white = 1) has zero absorption at any path length.
    expect(beerLambert(1, 0.1)).toBe(1);
    expect(beerLambert(1, 5.0)).toBe(1);
    // Colored glass matches target tint at characteristic diameter (2 * reach).
    expect(beerLambert(0.4, 2 * reach)).toBeCloseTo(0.4, 6);
    // Thin edges are lighter; thick multi-bounce paths are deeper.
    expect(beerLambert(0.4, 0.25 * reach)).toBeGreaterThan(0.4);
    expect(beerLambert(0.4, 4 * reach)).toBeLessThan(0.4);
  });
});

describe('TSL shader node graph construction', () => {
  const dummyTex = new THREE.DataTexture(new Uint8Array([255, 255, 255, 255]), 1, 1);
  const dummy3D = new THREE.Data3DTexture(new Float32Array(8), 2, 2, 2);

  const gInputs = {
    center: vec3(0, 1, 0),
    quat: vec4(0, 0, 0, 1),
    size: vec3(0.5, 0.5, 0.5),
    vary: vec3(0, 1, 0),
    tintOwn: vec3(1, 1, 1),
    tintAll: uniform(new THREE.Vector4(1, 1, 1, 1)),
    cmyAll: uniform(0),
    bevel: uniform(0),
    meshEntry: uniform(1),
    glassP: uniform(new THREE.Vector4(1.52, 1, 6, 4)),
    sdfTex: texture3D(dummy3D),
    roughness: uniform(0.2),
    frostScale: uniform(4),
  };

  const sInputs = {
    room: uniform(1),
    skyE: uniform(new THREE.Vector3(0.5, 0.5, 0.5)),
    sunE: uniform(new THREE.Vector3(0.5, 0.5, 0.5)),
    albedo: uniform(new THREE.Vector3(0.9, 0.9, 0.9)),
    lightC: uniform(new THREE.Vector3()),
    lightR: uniform(new THREE.Vector3(1, 0, 0)),
    lightU: uniform(new THREE.Vector3(0, 0, 1)),
    floorP: uniform(new THREE.Vector4(0.4, -0.8, 1, 0)),
    spot: uniform(new THREE.Vector4(0, 0, 0, 0)),
    lamp: uniform(new THREE.Vector3(0, 1, 0)),
    lightTex: texture(dummyTex),
    lightTexSize: 2048,
  };

  const vInputs = {
    camPos: uniform(new THREE.Vector3(0, 2, 5)),
    viewProj: uniform(new THREE.Matrix4()),
    behindTex: texture(dummyTex),
    behindBlurTex: texture(dummyTex),
    lampDepthTex: texture(dummyTex),
    lampDepthSize: 1024,
  };

  it('builds glassNode, photonNode, and causticNode for all 13 shapes without throwing', () => {
    for (const kind of Object.values(KIND)) {
      const gn = glassNode(kind, false, vec3(0, 1, 0), vec3(0, 1, 0), gInputs, sInputs, vInputs);
      const gm = glassNode(kind, true, vec3(0, -1, 0), vec3(0, 1, 0), gInputs, sInputs, vInputs);
      const pn = photonNode(kind, vec3(0, 1, 0), vec3(0, 1, 0), gInputs, sInputs.lamp, sInputs.lightR, sInputs.lightU);
      expect(gn).toBeTruthy();
      expect(gm).toBeTruthy();
      expect(pn).toBeTruthy();
    }
    expect(causticNode(vec3(0, 1, 0), vec3(0, 1, 0), gInputs)).toBeTruthy();
    void float;
  });

  it('builds glassNode and photonNode for all 6 diamond cuts without throwing', () => {
    for (const cut of Object.keys(DIAMOND_CUTS) as DiamondCut[]) {
      const planes = DIAMOND_CUTS[cut].planes;
      const gn = glassNode(KIND.diamond, false, vec3(0, 1, 0), vec3(0, 1, 0), gInputs, sInputs, vInputs, planes);
      const pn = photonNode(
        KIND.diamond,
        vec3(0, 1, 0),
        vec3(0, 1, 0),
        gInputs,
        sInputs.lamp,
        sInputs.lightR,
        sInputs.lightU,
        planes,
      );
      expect(gn).toBeTruthy();
      expect(pn).toBeTruthy();
    }
  });
});
