// Caustic & shadow map passes in TSL:
// - `photonNode`: real photon-traced caustics (traces each lamp ray through the glass onto the floor).
// - `causticNode`: multiplicative shadow/transmittance map drawn from the main lamp.
/* eslint-disable @typescript-eslint/no-explicit-any -- TSL node types are too loose to spell out */
import * as TSL from 'three/tsl';
import { KIND } from '../objects';
import type { GlassInputs } from './glass';
import { cmyTint } from './cmy';
import { fresnel, rotateToLocal, rotateToWorld } from './optics';
import { shapeFunctions } from './shapes';

type N = any;
const {
  Break,
  Fn,
  If,
  Loop,
  abs,
  bool,
  cross,
  dFdx,
  dFdy,
  dot,
  float,
  int,
  max,
  min,
  mix,
  normalize,
  reflect,
  refract,
  smoothstep,
  sqrt,
  vec2,
  vec3,
  vec4,
} = TSL as N;

/**
 * Real caustics, first pass: objects are rendered from the main lamp's direction so each
 * fragment traces one lamp ray through the glass down to y = 0.
 * Outputs `vec4(offsetX, offsetY, lightIntensity / spread, splatWidth + 1)`.
 */
export function photonNode(
  shape: number,
  worldPos: N,
  worldNormal: N,
  g: GlassInputs,
  sunDir: N,
  lightRight: N,
  lightUp: N,
  raysAcross: number,
  diamondPlanes?: number[],
): N {
  const { center, quat, size, vary, tintOwn, tintAll, cmyAll, glassP } = g;
  const { declare, entry, exit } = shapeFunctions(shape, g, diamondPlanes);

  return Fn((builder: N) => {
    declare();
    const cmy = max(cmyAll, vary.z).toVar();
    const rayDir = rotateToLocal(sunDir.negate(), quat).toVar();
    const { pos, nrm: entryNormal } = entry(
      rotateToLocal(worldPos.sub(center), quat),
      size,
      rotateToLocal(normalize(worldNormal), quat),
    );
    If(dot(rayDir, entryNormal).greaterThan(0), () => {
      entryNormal.assign(entryNormal.negate());
    });

    const ior = max(glassP.x.add(vary.x), 1.0001).toVar();
    // Fade grazing rays that scatter wide into thin streaks to keep the caustic pool clean.
    const transmitted = float(1)
      .sub(fresnel(dot(rayDir, entryNormal), ior))
      .mul(smoothstep(0.05, 0.4, dot(rayDir, entryNormal).negate()))
      .toVar();
    const tint = mix(vec3(1), tintAll.rgb.mul(tintOwn), tintAll.a).toVar();
    if (shape === KIND.box) {
      If(cmy.greaterThan(0), () => {
        tint.mulAssign(cmyTint(entryNormal, cmy, g.cmyOrder));
      });
    }

    const dir = refract(rayDir, entryNormal, float(1).div(ior)).toVar();
    const exitDir = vec3(0).toVar();
    const escaped = bool(false).toVar();
    If(dot(dir, dir).greaterThan(1e-6), () => {
      Loop({ start: int(0), end: int(8), type: 'int', condition: '<', name: 'ev' }, () => {
        const hit = exit(pos, dir, size);
        pos.assign(hit.pos);
        const hitNormal = hit.nrm.toVar();
        if (shape === KIND.box) {
          If(cmy.greaterThan(0), () => {
            tint.mulAssign(cmyTint(hitNormal, cmy, g.cmyOrder));
          });
        }
        const refracted = refract(dir, hitNormal.negate(), ior).toVar();
        If(dot(refracted, refracted).greaterThan(1e-6), () => {
          transmitted.mulAssign(float(1).sub(fresnel(dot(dir, hitNormal), float(1).div(ior))));
          exitDir.assign(refracted);
          escaped.assign(true);
          Break();
        });
        dir.assign(reflect(dir, hitNormal.negate()));
      });
    });

    const landingOffset = vec2(0).toVar();
    const lightEnergy = float(0).toVar();
    If(escaped, () => {
      const exitWorld = center.add(rotateToWorld(pos, quat)).toVar();
      const downDir = rotateToWorld(exitDir, quat).toVar();
      If(downDir.y.lessThan(-0.02), () => {
        const delta = exitWorld
          .add(downDir.mul(exitWorld.y.negate().div(downDir.y)))
          .sub(worldPos)
          .toVar();
        landingOffset.assign(vec2(dot(delta, lightRight), dot(delta, lightUp)));
        lightEnergy.assign(transmitted.mul(dot(tint, vec3(1 / 3))));
      });
    });

    // Estimate ray-bundle divergence on the floor via screen-space derivatives to splat & conserve energy.
    const across = dFdx(landingOffset)
      .mul(0.5 * raysAcross)
      .toVar();
    const downward = dFdy(landingOffset)
      .mul((builder.isFlipY?.() ? -0.5 : 0.5) * raysAcross)
      .toVar();
    const spread = max(abs(across.x.add(1).mul(float(1).sub(downward.y)).add(downward.x.mul(across.y))), 1).toVar();
    return vec4(landingOffset, lightEnergy.div(spread), min(sqrt(spread), 6).add(1));
  })();
}

/**
 * Multiplicative caustic/shadow map: drawn from the main lamp with multiplicative blending
 * so each crossed glass surface attenuates and tints the light reaching the floor.
 */
export function causticNode(worldNormal: N, worldPos: N, g: GlassInputs): N {
  const { quat, size, vary, tintOwn, tintAll, cmyAll } = g;
  return Fn(() => {
    const cmy = max(cmyAll, vary.z).toVar();
    const normal = normalize(worldNormal).toVar();
    const normalizedHeight = worldPos.y.div(max(size.x, max(size.y, size.z)).mul(0.9));
    const transmittance = vec3(mix(1.32, 0.79, smoothstep(0, 1, normalizedHeight))).toVar();
    // Light crosses two surfaces (in and out), so each applies sqrt(tint).
    transmittance.mulAssign(sqrt(mix(vec3(1), tintAll.rgb.mul(tintOwn), tintAll.a)));
    If(cmy.greaterThan(0), () => {
      const invAxis = quat.xyz.negate();
      const localNormal = normal.add(cross(invAxis, cross(invAxis, normal).add(normal.mul(quat.w))).mul(2));
      transmittance.mulAssign(cmyTint(localNormal, cmy, g.cmyOrder));
    });
    return vec4(transmittance, 1);
  })();
}
