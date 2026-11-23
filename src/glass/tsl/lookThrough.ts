// What a ray leaving the glass sees, as the instanced renderer's glass shading looks it up
// (shared so every glass that must match it pixel for pixel uses the same code): first other
// glass and the floor drawn behind, by a short screen-space march through `behindTex` (the
// backdrop and farther glass, with each pixel's distance from the camera in alpha), then the
// floor where the ray meets it, then the room (`scene`).
/* eslint-disable @typescript-eslint/no-explicit-any -- TSL node types are too loose to spell out */
import * as TSL from 'three/tsl';

type N = any;
const { Break, Fn, If, Loop, abs, bool, clamp, float, int, length, max, mix, smoothstep, sqrt, vec2, vec3, vec4 } =
  TSL as N;

/** Clip-space xy to texture coordinates (y down). */
const toScreenUv = (ndc: N) => vec2(ndc.x, ndc.y.negate()).mul(0.5).add(0.5);

export interface LookThroughInputs {
  /** 1 when there is a floor. */
  groundOn: N;
  /** The room along a ray: `scene(origin, dir, lampLit, cone)` (tsl/scene.ts). */
  scene: (ro: N, rd: N, lampLit: N, cone: N) => N;
  /** World to clip space of the camera. */
  viewProj: N;
  camPos: N;
  /** The backdrop and farther glass (rgb), each pixel's distance from the camera in alpha. */
  behindTex: N;
  /** A blurred copy of `behindTex`, for frosted glass. */
  behindBlurTex: N;
  /** Frosted glass roughness. */
  roughness: N;
  /** Steps of the screen-space march for other glass (0: none). */
  marchSteps: N;
}

/** `lookThrough(origin, dir, reach, lampLit, cone)`: what a ray leaving the glass sees. */
export function createLookThrough(inputs: LookThroughInputs) {
  const { groundOn, scene, viewProj, camPos, behindTex, behindBlurTex, roughness } = inputs;
  const glassP = { w: inputs.marchSteps };
  return Fn(([rayOrigin, rayDir, reach, lampLit, cone]: N[]) => {
    const out = vec3(0).toVar();
    const found = bool(false).toVar();
    const hitsFloor = groundOn.greaterThan(0.5).and(rayDir.y.lessThan(-1e-5));

    // Optional screen-space march for glass-through-glass refraction.
    If(glassP.w.greaterThan(0.5), () => {
      const maxDist = float(1e9).toVar();
      If(hitsFloor, () => {
        maxDist.assign(max(rayOrigin.y.negate().div(rayDir.y), 0));
      });
      const dist = reach.mul(0.2).toVar();
      Loop({ start: int(0), end: glassP.w.toInt(), type: 'int', condition: '<', name: 'ls' }, () => {
        If(dist.greaterThanEqual(maxDist), () => {
          Break();
        });
        const samplePos = rayOrigin.add(rayDir.mul(dist)).toVar();
        const clip = viewProj.mul(vec4(samplePos, 1)).toVar();
        If(clip.w.lessThan(0.01), () => {
          Break();
        });
        const ndc = clip.xy.div(clip.w).toVar();
        If(abs(ndc.x).greaterThan(0.98).or(abs(ndc.y).greaterThan(0.98)), () => {
          Break();
        });
        const suv = toScreenUv(ndc).toVar();
        const behindSample = behindTex.sample(suv).level(0).toVar();
        const depthDelta = length(samplePos.sub(camPos)).sub(behindSample.a).toVar();
        If(depthDelta.greaterThan(0).and(depthDelta.lessThan(reach.mul(1.5))), () => {
          out.assign(behindSample.rgb);
          If(roughness.greaterThan(0.001), () => {
            const blurMix = clamp(
              sqrt(roughness).mul(float(0.65).add(smoothstep(0, reach.mul(2), dist).mul(0.35))),
              0,
              1,
            );
            out.assign(mix(behindSample.rgb, behindBlurTex.sample(suv).level(0).rgb, blurMix));
          });
          found.assign(true);
          Break();
        });
        dist.mulAssign(1.6);
      });
    });

    // Floor projection into `behindTex` (captures caustics, reflections, and grid).
    If(found.not().and(hitsFloor), () => {
      const floorDist = max(rayOrigin.y.negate().div(rayDir.y), 0);
      const clip = viewProj.mul(vec4(rayOrigin.add(rayDir.mul(floorDist)), 1)).toVar();
      If(clip.w.greaterThan(0.01), () => {
        const ndc = clip.xy.div(clip.w).toVar();
        If(abs(ndc.x).lessThan(0.98).and(abs(ndc.y).lessThan(0.98)), () => {
          const suv = toScreenUv(ndc).toVar();
          const sharp = behindTex.sample(suv).level(0).rgb;
          out.assign(sharp);
          If(roughness.greaterThan(0.001), () => {
            const blurMix = clamp(
              sqrt(roughness).mul(float(0.65).add(smoothstep(0, reach.mul(2), floorDist).mul(0.35))),
              0,
              1,
            );
            out.assign(mix(sharp, behindBlurTex.sample(suv).level(0).rgb, blurMix));
          });
          found.assign(true);
        });
      });
    });

    If(found.not(), () => {
      out.assign(scene(rayOrigin, rayDir, lampLit, cone));
    });
    return out;
  }).setLayout({
    name: 'icLookThrough',
    type: 'vec3',
    inputs: [
      { name: 'ro', type: 'vec3' },
      { name: 'rd', type: 'vec3' },
      { name: 'reach', type: 'float' },
      { name: 'lampLit', type: 'float' },
      { name: 'cone', type: 'float' },
    ],
  });
}
