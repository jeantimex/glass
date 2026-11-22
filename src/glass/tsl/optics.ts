// Physical optical primitives in TSL: quaternion space transforms, exact unpolarized
// dielectric Fresnel, and Cauchy dispersion with continuous, stratified CIE 1931 spectral
// sampling (as many wavelengths as the dispersion strength needs).
/* eslint-disable @typescript-eslint/no-explicit-any -- TSL node types are too loose to spell out */
import * as TSL from 'three/tsl';

type N = any;
const {
  Fn,
  If,
  abs,
  ceil,
  clamp,
  cross,
  dot,
  exp,
  float,
  fract,
  int,
  max,
  screenCoordinate,
  select,
  sqrt,
  uniform,
  vec2,
  vec3,
} = TSL as N;

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

/* Cauchy dispersion */

/**
 * Legacy 3-channel Fraunhofer C/d/F IOR offset relative to yellow (587.6 nm, k = 1),
 * for red (656.3 nm, k = 0) and blue (486.1 nm, k = 2), given total spread `n_F - n_C`.
 */
export function iorOffset(spread: N, channel: number | N): N {
  if (typeof channel === 'number')
    return channel === 0 ? spread.mul(-0.301) : channel === 2 ? spread.mul(0.699) : float(0);
  return select(channel.equal(0), spread.mul(-0.301), select(channel.equal(2), spread.mul(0.699), 0));
}

const FRAUNHOFER_D = 587.6;
const FRAUNHOFER_F = 486.1;
const FRAUNHOFER_C = 656.3;
const CAUCHY_D = 1 / (FRAUNHOFER_D * FRAUNHOFER_D);
const CAUCHY_SCALE = 1 / (1 / (FRAUNHOFER_F * FRAUNHOFER_F) - 1 / (FRAUNHOFER_C * FRAUNHOFER_C));

/**
 * Cauchy's two-term dispersion `n(λ) = A + B / λ²`, as an IOR offset from the Fraunhofer d
 * line normalized by the spread `n_F - n_C`:
 *   cauchy(λ) = (1/λ² - 1/λ_d²) / (1/λ_F² - 1/λ_C²),   n(λ) = n_d + cauchy(λ) * (n_F - n_C)
 * (C = 656.3 nm red, d = 587.6 nm yellow, F = 486.1 nm blue).
 */
export function cauchyAt(nm: number): number {
  return (1 / (nm * nm) - CAUCHY_D) * CAUCHY_SCALE;
}

/* Continuous CIE 1931 spectral sampling */

/** The span of wavelengths sampled (nm): where the eye sees nearly all of the light. */
export const SPECTRUM_NM = [400, 700] as const;

/**
 * Wyman, Sloan & Shirley (2013) multi-lobe piecewise-Gaussian fit to the CIE 1931 2°
 * colour-matching functions: `[weight, mean, sigma below mean, sigma above mean]` lobes.
 */
const CMF_LOBES: Record<'x' | 'y' | 'z', ReadonlyArray<readonly [number, number, number, number]>> = {
  x: [
    [1.056, 599.8, 37.9, 31.0],
    [0.362, 442.0, 16.0, 26.7],
    [-0.065, 501.1, 20.4, 26.2],
  ],
  y: [
    [0.821, 568.8, 46.9, 40.5],
    [0.286, 530.9, 16.3, 31.1],
  ],
  z: [
    [1.217, 437.0, 11.8, 36.0],
    [0.681, 459.0, 26.0, 13.8],
  ],
};

/** CIE XYZ -> linear sRGB (D65). */
const XYZ_TO_RGB = [
  [3.2406, -1.5372, -0.4986],
  [-0.9689, 1.8758, 0.0415],
  [0.0557, -0.204, 1.057],
] as const;

/** CIE 1931 colour-matching functions (x̄, ȳ, z̄) at wavelength `nm`. */
export function cmfXyz(nm: number): [number, number, number] {
  const lobe = ([w, mu, below, above]: readonly [number, number, number, number]) => {
    const t = (nm - mu) / (nm < mu ? below : above);
    return w * Math.exp(-0.5 * t * t);
  };
  const sum = (lobes: (typeof CMF_LOBES)['x']) => lobes.reduce((acc, l) => acc + lobe(l), 0);
  return [sum(CMF_LOBES.x), sum(CMF_LOBES.y), sum(CMF_LOBES.z)];
}

/**
 * How strongly light of wavelength `nm` shows in each linear-sRGB channel: the
 * colour-matching functions in sRGB, with the parts outside the sRGB gamut (negative)
 * dropped. Unnormalized: the shader divides by the sum of the weights it used.
 */
export function spectralRgb(nm: number): [number, number, number] {
  const xyz = cmfXyz(nm);
  return XYZ_TO_RGB.map((row) => Math.max(row[0] * xyz[0] + row[1] * xyz[1] + row[2] * xyz[2], 0)) as [
    number,
    number,
    number,
  ];
}

/** The same weights as `spectralRgb`, as a shader node of wavelength `nm`. */
function spectralRgbNode(nm: N): N {
  const lobe = ([w, mu, below, above]: readonly [number, number, number, number]) => {
    const t = nm.sub(mu).div(select(nm.lessThan(mu), float(below), float(above)));
    return exp(t.mul(t).mul(-0.5)).mul(w);
  };
  const sum = (lobes: (typeof CMF_LOBES)['x']) =>
    lobes.slice(1).reduce((acc: N, l) => acc.add(lobe(l)), lobe(lobes[0]));
  const xyz = vec3(sum(CMF_LOBES.x), sum(CMF_LOBES.y), sum(CMF_LOBES.z));
  const row = (r: readonly [number, number, number]) => dot(xyz, vec3(r[0], r[1], r[2]));
  return max(vec3(row(XYZ_TO_RGB[0]), row(XYZ_TO_RGB[1]), row(XYZ_TO_RGB[2])), vec3(0));
}

/**
 * A per-pixel jitter in [0, 1) (Jimenez' interleaved gradient noise) for spectral sampling.
 * Each pixel takes its wavelengths evenly spaced across the spectrum, but offset by this
 * much of a step, so neighbouring pixels fill in the wavelengths between each other's and
 * fringes blend into a continuous rainbow rather than separate sharp copies.
 */
export const spectralJitter = (): N =>
  fract(
    float(52.9829189)
      .mul(fract(dot(screenCoordinate.xy, vec2(0.06711056, 0.00583715))))
      .add(SPECTRAL_FRAME_OFFSET),
  );

/**
 * Added to every pixel's spectral jitter: a renderer that averages frames (see
 * `createInstancedGlass`) moves it on each frame, so each pixel's wavelengths fill in
 * the whole spectrum over time. 0 (the default) gives a single frame's pattern.
 */
export const SPECTRAL_FRAME_OFFSET = uniform(0);

/** Fewest and most wavelengths traced per pixel, at full quality and in the fast mode. */
export const SPECTRAL_SAMPLES = { min: 6, max: 12, fastMin: 3, fastMax: 6 } as const;
/**
 * Wavelengths per unit of IOR spread (n_F - n_C) in one frame. Across 400-700 nm the Cauchy
 * offset spans about 2.2, so neighbouring samples differ in IOR by about 2.2 * spread / count;
 * this keeps that step near 0.045 per frame. The renderer moves the wavelengths on every
 * frame and averages frames (temporal AA), which fills in the steps between them, so a frame
 * needs only half the samples a single still image would (6 for real glass, diamond and the
 * demo's default, 12 at most).
 */
const SAMPLES_PER_SPREAD = 48;

/**
 * How many wavelengths to trace for IOR spread `spread` (n_F - n_C): more the stronger the
 * dispersion, as many as it takes to keep neighbouring wavelengths close. `fast` (a bool
 * node) halves the budget, for mobile GPUs and passes seen only indirectly.
 */
export function spectralSampleCount(spread: N, fast: N | boolean): N {
  const wanted = ceil(max(spread, 0).mul(SAMPLES_PER_SPREAD));
  if (typeof fast === 'boolean') {
    const [fewest, most] = fast
      ? [SPECTRAL_SAMPLES.fastMin, SPECTRAL_SAMPLES.fastMax]
      : [SPECTRAL_SAMPLES.min, SPECTRAL_SAMPLES.max];
    return int(clamp(fast ? wanted.mul(0.5).ceil() : wanted, float(fewest), float(most)));
  }
  const fewest = select(fast, float(SPECTRAL_SAMPLES.fastMin), float(SPECTRAL_SAMPLES.min));
  const most = select(fast, float(SPECTRAL_SAMPLES.fastMax), float(SPECTRAL_SAMPLES.max));
  return int(clamp(select(fast, wanted.mul(0.5).ceil(), wanted), fewest, most));
}

/**
 * Wavelength sample `index` of `count` (stratified across `SPECTRUM_NM`, offset by
 * `jitter` in [0, 1)): its Cauchy IOR offset and its unnormalized linear-sRGB weight.
 * Accumulate `radiance * weight` and `weight`, then divide: white light stays exactly
 * white whatever the count or jitter.
 */
export function spectralSample(index: N, count: N, jitter: N): { cauchy: N; weight: N } {
  const [lo, hi] = SPECTRUM_NM;
  const nm = float(index)
    .add(jitter)
    .div(float(count))
    .mul(hi - lo)
    .add(lo)
    .toVar();
  return {
    cauchy: float(1).div(nm.mul(nm)).sub(CAUCHY_D).mul(CAUCHY_SCALE),
    weight: spectralRgbNode(nm),
  };
}
