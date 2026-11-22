import { describe, expect, it } from 'vitest';
import * as THREE from 'three/webgpu';
import { float, texture, texture3D, uniform, vec3, vec4 } from 'three/tsl';
import { DIAMOND_CUTS, type DiamondCut } from '../diamond';
import { KIND } from '../objects';
import {
  cauchyAt,
  cmfXyz,
  SPECTRAL_SAMPLES,
  SPECTRUM_NM,
  spectralRgb,
  spectralSample,
  spectralSampleCount,
} from '../tsl/optics';
import { causticNode, glassNode, photonNode } from '../tsl/glass';

/** Wavelength (nm, 380-780) where colour-matching function `k` (0 x̄, 1 ȳ, 2 z̄) peaks. */
function cmfPeak(k: number): number {
  let best = 380;
  for (let nm = 380; nm <= 780; nm++) if (cmfXyz(nm)[k] > cmfXyz(best)[k]) best = nm;
  return best;
}

/** Which linear-sRGB channel (0 r, 1 g, 2 b) light of wavelength `nm` shows most in. */
function dominantChannel(nm: number): number {
  const rgb = spectralRgb(nm);
  return rgb.indexOf(Math.max(...rgb));
}

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

  it('follows Cauchy dispersion: shorter wavelengths bend more, zero at the d line', () => {
    expect(cauchyAt(587.6)).toBeCloseTo(0, 10);
    expect(cauchyAt(486.1) - cauchyAt(656.3)).toBeCloseTo(1, 10);
    const [lo, hi] = SPECTRUM_NM;
    for (let nm = lo; nm < hi; nm += 10) expect(cauchyAt(nm)).toBeGreaterThan(cauchyAt(nm + 10));
  });

  it('fits the CIE 1931 colour-matching functions (Wyman et al. 2013)', () => {
    // Luminosity peaks at ~555 nm with value ~1; x̄ peaks near 600 nm, z̄ near 445 nm.
    expect(Math.abs(cmfPeak(1) - 555)).toBeLessThan(10);
    expect(cmfXyz(555)[1]).toBeCloseTo(1, 1);
    expect(Math.abs(cmfPeak(0) - 600)).toBeLessThan(10);
    expect(Math.abs(cmfPeak(2) - 445)).toBeLessThan(10);
  });

  it('weights wavelengths by their non-negative sRGB colour, in spectral order', () => {
    const [lo, hi] = SPECTRUM_NM;
    for (let nm = lo; nm <= hi; nm += 5) for (const c of spectralRgb(nm)) expect(c).toBeGreaterThanOrEqual(0);
    expect(dominantChannel(450)).toBe(2); // blue
    expect(dominantChannel(530)).toBe(1); // green
    expect(dominantChannel(620)).toBe(0); // red
  });

  it('samples the spectrum evenly enough that jitter only adds faint grain', () => {
    // Per pixel the shader sums the weights of `count` evenly spaced wavelengths, offset by
    // a jitter. How much that sum changes with the jitter is how strong the grain can get.
    const [lo, hi] = SPECTRUM_NM;
    const sum = (count: number, jitter: number) => {
      const out = [0, 0, 0];
      for (let i = 0; i < count; i++) {
        const rgb = spectralRgb(lo + ((i + jitter) / count) * (hi - lo));
        for (let c = 0; c < 3; c++) out[c] += rgb[c];
      }
      return out;
    };
    for (const count of [SPECTRAL_SAMPLES.min, 12, SPECTRAL_SAMPLES.max]) {
      for (let c = 0; c < 3; c++) {
        const sums = Array.from({ length: 16 }, (_, k) => sum(count, k / 16)[c]);
        const mean = sums.reduce((a, b) => a + b) / sums.length;
        const spread = (Math.max(...sums) - Math.min(...sums)) / mean;
        expect(spread).toBeLessThan(count >= 12 ? 0.1 : 0.4);
      }
    }
  });

  it('builds the spectral sample count and samples as shader nodes', () => {
    expect(SPECTRAL_SAMPLES.min).toBeLessThan(SPECTRAL_SAMPLES.max);
    expect(spectralSampleCount(float(0.2), false)).toBeTruthy();
    expect(spectralSampleCount(float(0.2), true)).toBeTruthy();
    const sample = spectralSample(float(2), float(8), float(0.5));
    expect(sample.cauchy).toBeTruthy();
    expect(sample.weight).toBeTruthy();
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
