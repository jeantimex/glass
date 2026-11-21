// Procedural bright/dark studio environment used by standalone GlassMaterial.
/* eslint-disable @typescript-eslint/no-explicit-any -- TSL node types are too loose to spell out */
import * as TSL from 'three/tsl';

type N = any;
const { Fn, If, atan, cos, dot, float, max, mix, normalize, select, sin, smoothstep, vec3 } = TSL as N;

/** Build standalone procedural studio environment and ground sampler nodes. */
export function makeStudioEnv(uDark: N, uGround: N, uGroundColor: N, uLampDir: N, uRoughness: N) {
  const env = Fn(([dir]: N[]) => {
    const rough = uRoughness.toVar();
    const rough2 = rough.mul(rough).toVar();
    const sharpness = float(1).sub(rough.mul(0.85)).toVar();
    const azimuth = atan(dir.z, dir.x).toVar();
    const lamp = normalize(uLampDir).toVar();
    const out = vec3(0).toVar();

    If(uDark.greaterThan(0.5), () => {
      const k = mix(0.11, 0.035, smoothstep(0, 0.6, dir.y)).toVar();
      const spotGlow = smoothstep(
        float(0.988).sub(rough.mul(0.55)),
        float(0.996).add(rough.mul(0.004)),
        dot(dir, lamp),
      );
      k.addAssign(float(40).div(rough2.mul(28).add(1)).mul(spotGlow));

      const ring = smoothstep(float(0.78).sub(rough.mul(0.35)), float(0.86).add(rough.mul(0.1)), dir.y).mul(
        float(1).sub(smoothstep(0.94, 0.97, dir.y).mul(sharpness)),
      );
      k.addAssign(
        ring
          .mul(2.6)
          .div(rough2.mul(3).add(1))
          .mul(mix(smoothstep(-0.2, 0.3, cos(azimuth.mul(4))), 0.5, rough)),
      );

      const tall = smoothstep(0.05, 0.15, dir.y)
        .mul(float(1).sub(smoothstep(0.55, 0.7, dir.y)))
        .toVar();
      const stripLo = float(0.95).sub(rough.mul(0.55));
      const stripHi = float(0.985).add(rough.mul(0.015));
      const stripGain = float(4).div(rough2.mul(5).add(1));
      k.addAssign(tall.mul(stripGain).mul(smoothstep(stripLo, stripHi, cos(azimuth.sub(2.5)))));
      k.addAssign(tall.mul(stripGain).mul(smoothstep(stripLo, stripHi, cos(azimuth.add(0.63)))));
      k.addAssign(rough.mul(0.12));
      out.assign(vec3(k));
    }).Else(() => {
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

      const mainLo = float(0.86).sub(rough.mul(0.55));
      const mainHi = float(0.96).add(rough.mul(0.04));
      c.addAssign(
        float(3)
          .div(rough2.mul(3).add(1))
          .mul(smoothstep(mainLo, mainHi, dot(dir, lamp))),
      );

      const smallLo = float(0.985).sub(rough.mul(0.55));
      const smallGain = float(3).div(rough2.mul(8).add(1));
      c.addAssign(smoothstep(smallLo, 0.995, dot(dir, normalize(vec3(Math.cos(3), 0.6, Math.sin(3))))).mul(smallGain));
      c.addAssign(smoothstep(smallLo, 0.995, dot(dir, normalize(vec3(Math.cos(5), 0.9, Math.sin(5))))).mul(smallGain));
      c.addAssign(rough.mul(0.08));
      out.assign(vec3(c));
    });
    return out;
  }).setLayout({ name: 'gmEnv', type: 'vec3', inputs: [{ name: 'd', type: 'vec3' }] });

  const sampleScene = Fn(([rayOrigin, rayDir]: N[]) => {
    const out = vec3(0).toVar();
    If(uGround.greaterThan(0.5).and(rayDir.y.lessThan(-1e-5)), () => {
      const dist = max(rayOrigin.y.negate().div(rayDir.y), 0).toVar();
      const horizon = env(normalize(vec3(rayDir.x, 0.03, rayDir.z)));
      const skyLit = select(uDark.greaterThan(0.5), vec3(0.05), vec3(0.52));
      const sunLit = select(uDark.greaterThan(0.5), vec3(0.28), vec3(0.42));
      const floorCol = uGroundColor.mul(skyLit.add(sunLit));
      out.assign(mix(floorCol, horizon, smoothstep(60, 240, dist)));
    }).Else(() => {
      out.assign(env(rayDir));
    });
    return out;
  }).setLayout({
    name: 'gmScene',
    type: 'vec3',
    inputs: [
      { name: 'ro', type: 'vec3' },
      { name: 'rd', type: 'vec3' },
    ],
  });

  return { env, sampleScene };
}
