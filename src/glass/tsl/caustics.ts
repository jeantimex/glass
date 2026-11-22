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
  bool,
  cross,
  dot,
  float,
  int,
  max,
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
 * fragment traces one lamp ray through the glass down to y = 0 (at one wavelength: the
 * renderer draws a pass per wavelength band, so caustics carry dispersion's colours).
 * Outputs `vec4(offsetX, offsetY, lightCarried, 1)`: where the ray lands relative to where
 * it set out (in the lamp map's clip units), how much light it carries (0 if it never
 * reaches the floor), and 1 in alpha to mark that glass was in the way. How widely each
 * ray's light is spread on the floor is worked out when the rays are splatted
 * (renderer.ts), from where its neighbours land.
 */
export function photonNode(
  shape: number,
  worldPos: N,
  worldNormal: N,
  g: GlassInputs,
  sunDir: N,
  lightRight: N,
  lightUp: N,
  diamondPlanes?: number[],
  /**
   * This pass's wavelength band (see renderer.ts: one pass per band): x its Cauchy IOR
   * offset, yzw its linear-sRGB share of white light (to filter it by the glass's colour).
   */
  band?: N,
): N {
  const { center, quat, size, vary, tintOwn, tintAll, cmyAll, glassP } = g;
  const { declare, entry, exit } = shapeFunctions(shape, g, diamondPlanes);

  return Fn(() => {
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

    // This pass's wavelength: the IOR moves by its Cauchy offset times the spread n_F - n_C.
    const bandOffset = band ? band.x : float(0);
    const ior = max(glassP.x.add(vary.x).add(glassP.y.mul(vary.y).div(12).mul(bandOffset)), 1.0001).toVar();
    // The Fresnel reflection takes most light at grazing angles; only rays meeting the surface
    // all but edge-on (within ~5 degrees, where the rasterized mesh is least exact) are faded
    // out. (A wider fade cut away much of the light through steep faces: prisms, cones.)
    const transmitted = float(1)
      .sub(fresnel(dot(rayDir, entryNormal), ior))
      .mul(smoothstep(0.02, 0.09, dot(rayDir, entryNormal).negate()))
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
        // The share of this band's light the glass lets through: its colour filter weighed
        // by the band's place in the spectrum (the plain average for white light).
        const share = band ? band.yzw : vec3(1 / 3);
        lightEnergy.assign(transmitted.mul(dot(tint, share).div(max(dot(share, vec3(1)), 1e-4))));
      });
    });

    return vec4(landingOffset, lightEnergy, 1);
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
