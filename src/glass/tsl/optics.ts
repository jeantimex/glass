// Physical optical primitives in TSL: quaternion space transforms, exact unpolarized
// dielectric Fresnel, and Cauchy dispersion (Fraunhofer C/d/F & 6-band / 3-band CIE 1931
// equal-energy spectral integration).
/* eslint-disable @typescript-eslint/no-explicit-any -- TSL node types are too loose to spell out */
import * as TSL from 'three/tsl';

type N = any;
const { Fn, If, abs, cross, float, select, sqrt, vec3 } = TSL as N;

/* Quaternion transforms */

/**
 * Rotate vector `v` from world space into an object's local space using the conjugate
 * of unit quaternion `q = (q_x, q_y, q_z, q_w)`:
 *   v' = v + 2 * (-q_xyz) × ((-q_xyz) × v + q_w * v)
 */
export const rotateToLocal = (v: N, q: N) => {
  const invAxis = q.xyz.negate();
  return v.add(cross(invAxis, cross(invAxis, v).add(v.mul(q.w))).mul(2));
};

/**
 * Rotate vector `v` from an object's local space back into world space using unit
 * quaternion `q = (q_x, q_y, q_z, q_w)`:
 *   v' = v + 2 * q_xyz × (q_xyz × v + q_w * v)
 */
export const rotateToWorld = (v: N, q: N) => v.add(cross(q.xyz, cross(q.xyz, v).add(v.mul(q.w))).mul(2));

/* Dielectric Fresnel */

/**
 * Exact unpolarized Cook-Torrance dielectric Fresnel reflectance (average of s- and p-polarized power).
 * Given `cosI = |cos(theta_i)|` and relative IOR `eta = n_transmitted / n_incident`:
 *   g² = eta² - 1 + cosI²
 *   if g² <= 0: Total Internal Reflection (R = 1)
 *   else:       R = 0.5 * ((g - c) / (g + c))² * (1 + ((c * (g + c) - 1) / (c * (g - c) + 1))²)
 */
export const fresnel = Fn(([cosThetaI, eta]: N[]) => {
  const cosI = abs(cosThetaI).toVar();
  const gSquared = eta.mul(eta).sub(1).add(cosI.mul(cosI)).toVar();
  const reflectance = float(1).toVar();
  If(gSquared.greaterThan(0), () => {
    const g = sqrt(gSquared).toVar();
    const termA = g.sub(cosI).div(g.add(cosI));
    const termB = cosI
      .mul(g.add(cosI))
      .sub(1)
      .div(cosI.mul(g.sub(cosI)).add(1));
    reflectance.assign(termA.mul(termA).mul(termB.mul(termB).add(1)).mul(0.5));
  });
  return reflectance;
}).setLayout({
  name: 'icFresnel',
  type: 'float',
  inputs: [
    { name: 'cosI', type: 'float' },
    { name: 'eta', type: 'float' },
  ],
});

/* Cauchy dispersion & CIE 1931 spectral bands */

/**
 * Legacy 3-channel Fraunhofer C/d/F IOR offset relative to yellow (587.6 nm, k = 1),
 * for red (656.3 nm, k = 0) and blue (486.1 nm, k = 2), given total spread `n_F - n_C`.
 */
export function iorOffset(spread: N, channel: number | N): N {
  if (typeof channel === 'number')
    return channel === 0 ? spread.mul(-0.301) : channel === 2 ? spread.mul(0.699) : float(0);
  return select(channel.equal(0), spread.mul(-0.301), select(channel.equal(2), spread.mul(0.699), 0));
}

export interface SpectralBand {
  /** Effective wavelength (nm) of this equal-energy CIE 1931 band. */
  wavelength: number;
  /** Cauchy IOR offset relative to Fraunhofer d (587.6 nm), normalized by (n_F - n_C). */
  cauchy: number;
  /** Normalized linear-sRGB spectral sensitivity weight (sums to [1, 1, 1] across all bands). */
  rgb: readonly [number, number, number];
}

/**
 * Six equal-energy stratified spectral bands across the visible spectrum (380–720 nm),
 * derived from the CIE 1931 2° standard observer color-matching functions (D65 linear sRGB)
 * and Cauchy's two-term dispersion equation `n(λ) = A + B / λ²`:
 *   cauchy(λ) = (1/λ² - 1/λ_d²) / (1/λ_F² - 1/λ_C²)
 *   n(λ)      = n_d + cauchy(λ) * (n_F - n_C)
 * where Fraunhofer lines are C = 656.3 nm (red), d = 587.6 nm (yellow), F = 486.1 nm (blue).
 *
 * Each band's linear-sRGB weights are non-negative and normalized so Σ rgb = (1, 1, 1),
 * guaranteeing exact energy and white-balance conservation as dispersion → 0.
 */
export const SPECTRAL_BANDS: readonly SpectralBand[] = [
  { wavelength: 432.3, cauchy: 1.2845, rgb: [0.0388, 0.0, 0.4409] }, // Violet (380–447 nm)
  { wavelength: 459.9, cauchy: 0.9586, rgb: [0.0, 0.0, 0.4313] }, // Blue (447–475 nm)
  { wavelength: 507.6, cauchy: 0.5155, rgb: [0.0, 0.4643, 0.1278] }, // Cyan-Green (475–538 nm)
  { wavelength: 556.5, cauchy: 0.1739, rgb: [0.0438, 0.4024, 0.0] }, // Yellow-Green (538–573 nm)
  { wavelength: 588.1, cauchy: -0.0025, rgb: [0.3587, 0.1333, 0.0] }, // Amber-Yellow (573–603.5 nm)
  { wavelength: 626.5, cauchy: -0.1826, rgb: [0.5587, 0.0, 0.0] }, // Red (603.5–720 nm)
] as const;

export const SPECTRAL_COUNT = SPECTRAL_BANDS.length;

/** Cauchy IOR offset multiplier for 6-band spectral index `bandIndex` (0..5). */
export function spectralCauchy(bandIndex: N): N {
  let out: N = float(SPECTRAL_BANDS[SPECTRAL_COUNT - 1].cauchy);
  for (let i = SPECTRAL_COUNT - 2; i >= 0; i--) {
    out = select(bandIndex.equal(i), float(SPECTRAL_BANDS[i].cauchy), out);
  }
  return out;
}

/** Normalized linear-sRGB spectral sensitivity weight for 6-band index `bandIndex` (0..5). */
export function spectralWeight(bandIndex: N): N {
  const last = SPECTRAL_BANDS[SPECTRAL_COUNT - 1].rgb;
  let out: N = vec3(last[0], last[1], last[2]);
  for (let i = SPECTRAL_COUNT - 2; i >= 0; i--) {
    const c = SPECTRAL_BANDS[i].rgb;
    out = select(bandIndex.equal(i), vec3(c[0], c[1], c[2]), out);
  }
  return out;
}

/**
 * Three energy-conserving overlapping CIE 1931 spectral bands (paired from SPECTRAL_BANDS),
 * used for mobile GPUs and secondary glass layers (~2x faster than 6-band while avoiding
 * non-overlapping RGB delta-spike confetti).
 */
export const SPECTRAL_BANDS_FAST: readonly SpectralBand[] = [
  { wavelength: 445.9, cauchy: 1.122, rgb: [0.0388, 0.0, 0.8722] }, // Violet-Blue (B0 + B1)
  { wavelength: 530.7, cauchy: 0.345, rgb: [0.0438, 0.8667, 0.1278] }, // Cyan-Green (B2 + B3)
  { wavelength: 610.6, cauchy: -0.111, rgb: [0.9174, 0.1333, 0.0] }, // Amber-Red (B4 + B5)
] as const;

export const SPECTRAL_FAST_COUNT = SPECTRAL_BANDS_FAST.length;

/** Cauchy IOR offset multiplier for fast 3-band spectral index `bandIndex` (0..2). */
export function spectralCauchyFast(bandIndex: N): N {
  return select(
    bandIndex.equal(0),
    float(SPECTRAL_BANDS_FAST[0].cauchy),
    select(bandIndex.equal(1), float(SPECTRAL_BANDS_FAST[1].cauchy), float(SPECTRAL_BANDS_FAST[2].cauchy)),
  );
}

/** Normalized linear-sRGB spectral sensitivity weight for fast 3-band index `bandIndex` (0..2). */
export function spectralWeightFast(bandIndex: N): N {
  const c0 = SPECTRAL_BANDS_FAST[0].rgb;
  const c1 = SPECTRAL_BANDS_FAST[1].rgb;
  const c2 = SPECTRAL_BANDS_FAST[2].rgb;
  return select(
    bandIndex.equal(0),
    vec3(c0[0], c0[1], c0[2]),
    select(bandIndex.equal(1), vec3(c1[0], c1[1], c1[2]), vec3(c2[0], c2[1], c2[2])),
  );
}
