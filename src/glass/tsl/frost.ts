// Frosted glass (roughness) shading in TSL: 3-octave organic micro-particle noise,
// surface normal perturbation, translucent frost veil, and grazing Fresnel rim.
// Shared by both the instanced glass shader (glass.ts) and standalone MeshGlassMaterial (material.ts).
/* eslint-disable @typescript-eslint/no-explicit-any -- TSL node types are too loose to spell out */
import * as TSL from 'three/tsl';

type N = any;
const { Fn, If, clamp, dot, float, floor, fract, max, mix, normalize, sin, vec3 } = TSL as N;

/** Fast 3D hash returning [0, 1). */
const hash13 = (p: N) => fract(sin(dot(p, vec3(127.1, 311.7, 74.7))).mul(43758.5453123));

/** Smooth trilinear 3D value noise using the C^1 cubic Hermite curve `u(f) = 3f² - 2f³`. */
const valueNoise3 = Fn(([p]: N[]) => {
  const cell = floor(p).toVar();
  const frac = fract(p).toVar();
  const smooth = frac
    .mul(frac)
    .mul(vec3(3).sub(frac.mul(2)))
    .toVar();

  const n000 = hash13(cell);
  const n100 = hash13(cell.add(vec3(1, 0, 0)));
  const n010 = hash13(cell.add(vec3(0, 1, 0)));
  const n110 = hash13(cell.add(vec3(1, 1, 0)));
  const n001 = hash13(cell.add(vec3(0, 0, 1)));
  const n101 = hash13(cell.add(vec3(1, 0, 1)));
  const n011 = hash13(cell.add(vec3(0, 1, 1)));
  const n111 = hash13(cell.add(vec3(1, 1, 1)));

  const nx00 = mix(n000, n100, smooth.x);
  const nx10 = mix(n010, n110, smooth.x);
  const nx01 = mix(n001, n101, smooth.x);
  const nx11 = mix(n011, n111, smooth.x);
  return mix(mix(nx00, nx10, smooth.y), mix(nx01, nx11, smooth.y), smooth.z);
}).setLayout({ name: 'icValueNoise3', type: 'float', inputs: [{ name: 'p', type: 'vec3' }] });

/** 3-octave organic frost grain: fine (0.52) + medium (0.33) + broad (0.15). */
export const frostNoise3 = (p: N) =>
  clamp(
    valueNoise3(p.mul(36))
      .mul(0.52)
      .add(valueNoise3(p.mul(14).add(vec3(19.7, 4.3, 11.2))).mul(0.33))
      .add(valueNoise3(p.mul(5.5).add(vec3(3.1, 28.4, 7.6))).mul(0.15)),
    0,
    1,
  );

/**
 * Perturb entry surface normal `normalVar` with 3D organic frost micro-grain and write
 * the sampled grain value [0, 1] into `frostGrainVar`.
 *
 * Estimates the 3D noise gradient via forward finite differences `(N(p + ε e_i) - N(p))`
 * and rejects perturbations that would flip the normal behind the view ray (`dot(V, n') >= -0.02`).
 */
export function perturbFrostNormal(
  normalVar: N,
  viewDir: N,
  localPos: N,
  objectReach: N,
  roughness: N,
  frostScale: N,
  frostGrainVar: N,
): void {
  If(roughness.greaterThan(0.001), () => {
    const scale = max(frostScale, 0.25).toVar();
    const samplePos = localPos.div(max(objectReach, 0.05)).mul(scale).toVar();
    const centerNoise = frostNoise3(samplePos).toVar();
    frostGrainVar.assign(centerNoise);

    const eps = float(0.008).toVar();
    const gradX = frostNoise3(samplePos.add(vec3(eps, 0, 0))).sub(centerNoise);
    const gradY = frostNoise3(samplePos.add(vec3(0, eps, 0))).sub(centerNoise);
    const gradZ = frostNoise3(samplePos.add(vec3(0, 0, eps))).sub(centerNoise);

    const perturbed = normalize(normalVar.add(vec3(gradX, gradY, gradZ).mul(roughness.mul(0.45)))).toVar();
    If(dot(viewDir, perturbed).lessThan(-0.02), () => {
      normalVar.assign(perturbed);
    });
  });
}

/**
 * Tilt surface normal `normalVar` by the glass's faint imperfections at `localPos`: slow,
 * organic waviness (as in cast or hand-finished glass, far broader and gentler than frost),
 * up to about 0.03 rad at `amount` 1 (cast glass bends light by a few milliradians). A function of position only, so the same surface
 * bends every ray that meets it the same way, inside or out.
 */
export function perturbImperfections(normalVar: N, localPos: N, objectReach: N, amount: N): void {
  If(amount.greaterThan(0.001), () => {
    // A few undulations across the object (frostNoise3's octaves at 36, 14 and 5.5 times this).
    const p = localPos.div(max(objectReach, 0.05)).mul(0.07).toVar();
    const centre = frostNoise3(p).toVar();
    const eps = float(0.004);
    const gradient = vec3(
      frostNoise3(p.add(vec3(eps, 0, 0))).sub(centre),
      frostNoise3(p.add(vec3(0, eps, 0))).sub(centre),
      frostNoise3(p.add(vec3(0, 0, eps))).sub(centre),
    ).div(eps);
    // Only the part across the surface tilts it.
    const across = gradient.sub(normalVar.mul(dot(gradient, normalVar)));
    normalVar.assign(normalize(normalVar.add(across.mul(amount.mul(0.0012)))));
  });
}

/**
 * Apply the creamy translucent frost veil and grazing cubic Fresnel rim `(1 - N·V)³`
 * onto `colorVar` when `roughness > 0`.
 */
export function applyFrostVeil(
  colorVar: N,
  viewDir: N,
  normal: N,
  baseTint: N,
  studioLight: N,
  roughness: N,
  frostGrain: N,
): void {
  If(roughness.greaterThan(0.001), () => {
    const veilOpacity = roughness.mul(0.42).mul(frostGrain.mul(0.24).add(0.88)).toVar();
    const frostTint = baseTint.mul(studioLight).toVar();
    const lifted = colorVar.mul(1.03).add(frostTint.mul(0.22));
    const frosted = mix(mix(colorVar, lifted, veilOpacity), frostTint, veilOpacity.mul(0.55));
    const nDotV = clamp(dot(viewDir, normal).negate(), 0, 1);
    const rim = float(1).sub(nDotV).toVar();
    const rimGlow = rim.mul(rim).mul(rim).mul(roughness.mul(0.35)).mul(studioLight);
    colorVar.assign(frosted.add(rimGlow));
  });
}
