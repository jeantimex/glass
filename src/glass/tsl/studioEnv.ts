// Neutral procedural studio fallback (and optional equirectangular HDRI `envMap`)
// used by standalone MeshGlassMaterial.
/* eslint-disable @typescript-eslint/no-explicit-any -- TSL node types are too loose to spell out */
import * as TSL from 'three/tsl';

type N = any;
const { Fn, If, asin, atan, clamp, dot, float, mix, normalize, sin, smoothstep, vec2, vec3 } = TSL as N;

export interface StudioEnvOptions {
  /** Optional equirectangular environment texture node (`texture(envMap)`). */
  envTexNode?: N;
  /** Uniform flag: `> 0.5` samples `envTexNode` instead of the procedural studio. */
  uHasEnvMap?: N;
  /** Scalar intensity multiplier applied to `envTexNode`. */
  uEnvMapIntensity?: N;
}

/**
 * Build standalone 360° environment sampler for `MeshGlassMaterial`.
 * When `envMap` is bound (`uHasEnvMap > 0.5`), samples the equirectangular HDRI at mip level
 * `roughness * 5` (matching Three.js equirectangular UV convention); otherwise evaluates a
 * neutral light-tent studio so `new MeshGlassMaterial()` looks crisp with zero setup.
 */
export function makeStudioEnv(uRoughness: N, opts: StudioEnvOptions = {}) {
  const { envTexNode, uHasEnvMap, uEnvMapIntensity } = opts;
  const mainLamp = normalize(vec3(-0.6, 0.7, 0.4));
  const accentLampA = normalize(vec3(Math.cos(3), 0.6, Math.sin(3)));
  const accentLampB = normalize(vec3(Math.cos(5), 0.9, Math.sin(5)));

  const env = Fn(([dir]: N[]) => {
    const rough = uRoughness.toVar();
    const out = vec3(0).toVar();

    const evalFallbackStudio = () => {
      const rough2 = rough.mul(rough).toVar();
      const sharpness = float(1).sub(rough.mul(0.85)).toVar();
      const azimuth = atan(dir.z, dir.x).toVar();

      // Soft light-tent gradient (pale overhead, darker lower hemisphere for glass contrast).
      const c = mix(mix(0.15, 0.38, rough), mix(0.72, 0.65, rough), smoothstep(-0.1, 0.95, dir.y)).toVar();
      const band = smoothstep(0.03, 0.1, dir.y).mul(float(1).sub(smoothstep(0.68, 0.82, dir.y)));
      c.assign(
        mix(
          c,
          0.04,
          band
            .mul(0.96)
            .mul(sharpness)
            .mul(smoothstep(-0.1, 0.15, sin(azimuth.mul(5).add(1)))),
        ),
      );
      c.assign(mix(c, 0.05, smoothstep(0.9, 0.95, dir.y).mul(sharpness)));

      // Soft key light + two crisp accent highlights, energy-widened by roughness.
      const mainLo = float(0.86).sub(rough.mul(0.55));
      const mainHi = float(0.96).add(rough.mul(0.04));
      c.addAssign(
        float(3)
          .div(rough2.mul(3).add(1))
          .mul(smoothstep(mainLo, mainHi, dot(dir, mainLamp))),
      );

      const smallLo = float(0.985).sub(rough.mul(0.55));
      const smallGain = float(3).div(rough2.mul(8).add(1));
      c.addAssign(smoothstep(smallLo, 0.995, dot(dir, accentLampA)).mul(smallGain));
      c.addAssign(smoothstep(smallLo, 0.995, dot(dir, accentLampB)).mul(smallGain));
      c.addAssign(rough.mul(0.08));
      out.assign(vec3(c));
    };

    if (envTexNode && uHasEnvMap && uEnvMapIntensity) {
      If(uHasEnvMap.greaterThan(0.5), () => {
        // Equirectangular spherical projection: u = atan2(z, x) / 2π + 0.5, v = asin(y) / π + 0.5.
        const u = atan(dir.z, dir.x)
          .mul(1 / (2 * Math.PI))
          .add(0.5);
        const v = asin(clamp(dir.y, -1, 1))
          .mul(1 / Math.PI)
          .add(0.5);
        out.assign(envTexNode.sample(vec2(u, v)).level(rough.mul(5)).rgb.mul(uEnvMapIntensity));
      }).Else(evalFallbackStudio);
    } else {
      evalFallbackStudio();
    }

    return out;
  }).setLayout({ name: 'gmEnv', type: 'vec3', inputs: [{ name: 'd', type: 'vec3' }] });

  return { env };
}
