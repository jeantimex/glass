// Main glass refraction & multi-bounce raytracing shader (`glassNode`) in TSL.
//
// Related focused modules:
// - `optics.ts`: Fresnel, Cauchy spectral bands (6-band & fast 3-band), CMY filters
// - `frost.ts`:  frosted glass (roughness) normal perturbation, veil, and Fresnel rim
// - `shapes.ts`: analytic & baked-SDF `entry` / `exit` geometry functions
// - `caustics.ts`: photon-traced caustics (`photonNode`) and shadow map (`causticNode`)
/* eslint-disable @typescript-eslint/no-explicit-any -- TSL node types are too loose to spell out */
import * as TSL from 'three/tsl';
import { KIND } from '../objects';
import { cmyTint } from './cmy';
import { applyFrostVeil, perturbFrostNormal } from './frost';
import { fresnel, rotateToLocal, rotateToWorld, spectralJitter, spectralSample, spectralSampleCount } from './optics';
import { sceneFunctions, type SceneInputs } from './scene';
import { isBakedMesh, isCurvedShape, shapeFunctions } from './shapes';

// Re-export modular optical, CMY, frost, shape, and caustic primitives for backwards compatibility.
export { CMY_FACE_ORDERS, cmyChannelColor, cmyTint, type CmyFaceOrder } from './cmy';
export {
  cauchyAt,
  fresnel,
  iorOffset,
  SPECTRAL_SAMPLES,
  SPECTRUM_NM,
  spectralRgb,
  spectralSample,
  spectralSampleCount,
} from './optics';
export { applyFrostVeil, frostNoise3, perturbFrostNormal } from './frost';
export { isBakedMesh, isCurvedShape, shapeFunctions } from './shapes';
export { causticNode, photonNode } from './caustics';

type N = any;
const {
  Break,
  Fn,
  If,
  Loop,
  abs,
  bool,
  clamp,
  dFdx,
  dFdy,
  dot,
  exp,
  float,
  int,
  ivec2,
  length,
  log,
  max,
  min,
  mix,
  normalize,
  reflect,
  refract,
  select,
  smoothstep,
  sqrt,
  vec2,
  vec3,
  vec4,
} = TSL as N;

/** Per-instance and global uniforms read by the glass and caustic passes. */
export interface GlassInputs {
  /** Per-object instance attributes: center, quaternion, size, variation, own linear tint. */
  center: N;
  quat: N;
  size: N;
  vary: N;
  tintOwn: N;
  /** Global glass tint (linear rgb) and strength (a). */
  tintAll: N;
  /** Global CMY dichroic filter strength for boxes (0 = off). */
  cmyAll: N;
  /** Optional CMY face permutation (vec3 of 0=Cyan, 1=Yellow, 2=Magenta for X, Y, Z). */
  cmyOrder?: N;
  /** Edge bevel width for boxes (m). */
  bevel: N;
  /** 1 when baked meshes project entry rays onto their C^2 SDF isosurface. */
  meshEntry: N;
  /** vec4(ior, dispersion, maxBounces, lookThroughSteps). */
  glassP: N;
  /** Stacked 128^3 signed-distance volumes (Suzanne, Dragon, torus knot; see sdf.ts). */
  sdfTex: N;
  /** Frosted glass roughness (0 = crystal clear, 1 = fully frosted). */
  roughness: N;
  /** Frosted glass micro-particle grain density. */
  frostScale: N;
}

/** Camera, backdrop, and shadow-depth textures read by the glass shading pass. */
export interface GlassViewInputs {
  camPos: N;
  viewProj: N;
  /** Rendered backdrop + farther glass layer (alpha stores distance from camera). */
  behindTex: N;
  /** Quarter-res blurred copy of `behindTex` for frosted glass transmission. */
  behindBlurTex: N;
  /** Lamp-view depth map (red = distance toward lamp + 100) for inter-object shadows. */
  lampDepthTex: N;
  lampDepthSize: number;
}

/**
 * Multi-bounce physical glass fragment shader for one `KIND`.
 * Traces dielectric Fresnel reflection, CIE 1931 spectral dispersion, Beer-Lambert absorption,
 * CMY dichroic filters, ray-cone angular filtering, and frosted glass roughness.
 */
export function glassNode(
  shape: number,
  mirrored: boolean,
  worldPos: N,
  worldNormal: N,
  g: GlassInputs,
  s: SceneInputs,
  v: GlassViewInputs,
  diamondPlanes?: number[],
): N {
  const { center, quat, size, vary, tintOwn, tintAll, cmyAll, glassP, roughness, frostScale } = g;
  const { camPos, viewProj, behindTex, behindBlurTex } = v;
  const { groundOn, scene } = sceneFunctions({ ...s, roughness }, false);
  // Mirror reflections are half-res and blurred: skip expensive tricubic mesh entry projection.
  const { declare, entry, exit } = shapeFunctions(shape, mirrored ? { ...g, meshEntry: float(0) } : g, diamondPlanes);

  // 4-tap PCF visibility test against the lamp's depth map (inter-object shadows).
  const lampVisible = (pointWorld: N, objectTop: N) => {
    const lit = float(1).toVar();
    const offset = pointWorld.sub(s.lightC);
    const uv = vec2(dot(offset, s.lightR).mul(0.5).add(0.5), float(0.5).sub(dot(offset, s.lightU).mul(0.5))).toVar();
    If(uv.x.greaterThan(0).and(uv.y.greaterThan(0)).and(uv.x.lessThan(1)).and(uv.y.lessThan(1)), () => {
      const dims = v.lampDepthSize;
      const texCoord = uv.mul(dims).toVar();
      const unoccluded = float(0).toVar();
      for (let k = 0; k < 4; k++) {
        const texel = clamp(
          ivec2(texCoord.add(vec2(k & 1 ? 0.75 : -0.75, k & 2 ? 0.75 : -0.75))),
          ivec2(0),
          ivec2(dims - 1),
        );
        const occluderReach = v.lampDepthTex.load(texel).r.sub(100);
        unoccluded.addAssign(select(occluderReach.greaterThan(objectTop), 0, 1));
      }
      lit.assign(unoccluded.mul(0.25));
    });
    return lit;
  };

  // Sample what a refracted ray sees after exiting the glass: screen-space behind glass,
  // rendered floor in `behindTex`, or analytic studio environment fallback.
  const toScreenUv = (ndc: N) => vec2(ndc.x, ndc.y.negate()).mul(0.5).add(0.5);
  const lookThrough = mirrored
    ? (ro: N, rd: N, _reach: N, lampLit: N, cone: N) => scene(ro, rd, lampLit, cone)
    : Fn(([rayOrigin, rayDir, reach, lampLit, cone]: N[]) => {
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

  return Fn(() => {
    declare();
    const reach = max(size.x, max(size.y, size.z)).toVar();
    const cmy = max(cmyAll, vary.z).toVar();
    const isCmyBox = shape === KIND.box;

    // Inter-object shadow visibility (skipped when s.spot.z <= 0.5).
    const uprightPos = mirrored ? worldPos.mul(vec3(1, -1, 1)) : worldPos;
    const lampLit = float(1).toVar();
    If(s.spot.z.greaterThan(0.5), () => {
      lampLit.assign(lampVisible(uprightPos, dot(center, s.lamp).add(length(size).mul(1.25))));
    });

    const viewDirLocal = rotateToLocal(normalize(worldPos.sub(camPos)), quat).toVar();
    const { pos: entryPos, nrm: entryNormal } = entry(
      rotateToLocal(worldPos.sub(center), quat),
      size,
      rotateToLocal(normalize(worldNormal), quat),
    );
    If(dot(viewDirLocal, entryNormal).greaterThan(0), () => {
      entryNormal.assign(entryNormal.negate());
    });

    // Physical ray-cone angular footprint at entry (evaluated in uniform control flow for WGSL).
    // `max(|∂n/∂x|, |∂n/∂y|)` measures the normal spread (radians) across 1 screen pixel.
    const curved = isCurvedShape(shape);
    const dnX = dFdx(entryNormal);
    const dnY = dFdy(entryNormal);
    const entryCone = curved ? clamp(sqrt(max(dot(dnX, dnX), dot(dnY, dnY))), 0, 0.85).toVar() : float(0).toVar();
    const reflCone = curved ? clamp(entryCone.mul(0.6), 0, 0.6).toVar() : float(0).toVar();

    // Organic 3-octave frosted surface normal perturbation (skipped on blurred mirror reflections).
    const frostGrain = float(0.5).toVar();
    if (!mirrored) {
      perturbFrostNormal(entryNormal, viewDirLocal, entryPos, reach, roughness, frostScale, frostGrain);
    }

    const baseIor = max(glassP.x.add(vary.x), 1.0001).toVar();
    const dispersionSpread = glassP.y.mul(vary.y).div(12).toVar();
    // Same bounce budget for every shape (the renderer lowers it for big piles and the floor mirror).
    const maxEvents = glassP.z.toInt();

    // As many wavelengths as the dispersion needs (heavily frosted glass blurs the colours
    // out: one); mobile & far layers trace half as many (s.spot.w > 0.5).
    const useFastSpectral = s.spot.w.greaterThan(0.5).toVar();
    const wantDispersion = dispersionSpread.greaterThan(0).and(roughness.lessThan(0.35));
    const bandCount = select(wantDispersion, spectralSampleCount(dispersionSpread, useFastSpectral), int(1)).toVar();
    // Where between evenly spaced wavelengths this pixel samples (continuous rainbow fringes).
    const spectralU = spectralJitter().toVar();
    // Sum of the spectral weights used, to divide by (keeps white light exactly white).
    const weightSum = vec3(0).toVar();

    // Beer-Lambert absorption coefficient across one characteristic diameter (2 * reach).
    const baseTint = mix(vec3(1), tintAll.rgb.mul(tintOwn), tintAll.a).toVar();
    const absorptionSigma = log(max(baseTint, vec3(1e-4)))
      .negate()
      .div(max(reach.mul(2), 1e-4))
      .toVar();
    const entryFaceFilter = vec3(1).toVar();
    if (isCmyBox) {
      If(cmy.greaterThan(0), () => {
        entryFaceFilter.assign(cmyTint(entryNormal, cmy, g.cmyOrder));
      });
    }

    // Hoist wavelength-independent outer reflection outside the spectral loop.
    const viewDotNormal = dot(viewDirLocal, entryNormal).toVar();
    const outerReflection = scene(
      worldPos.add(rotateToWorld(entryNormal, quat).mul(0.003)),
      rotateToWorld(reflect(viewDirLocal, entryNormal), quat),
      lampLit,
      reflCone,
    ).toVar();

    const color = vec3(0).toVar();

    const tracePerBand = () => {
      Loop({ start: int(0), end: bandCount, type: 'int', condition: '<', name: 'lc' }, ({ lc }: { lc: N }) => {
        const sample = spectralSample(lc, bandCount, spectralU);
        const bandOffset = select(bandCount.equal(1), float(0), sample.cauchy);
        const bandIor = max(baseIor.add(dispersionSpread.mul(bandOffset)), 1.0001).toVar();
        const entryFresnel = fresnel(viewDotNormal, bandIor).toVar();
        const bandRadiance = entryFresnel.mul(outerReflection).toVar();
        const throughput = float(1).sub(entryFresnel).toVar();
        const rayDir = refract(viewDirLocal, entryNormal, float(1).div(bandIor)).toVar();
        const rayPos = vec3(entryPos).toVar();
        const rayCone = entryCone.mul(0.45).toVar();
        const attenuation = vec3(entryFaceFilter).toVar();

        If(throughput.greaterThan(0.012).and(dot(rayDir, rayDir).greaterThan(1e-6)), () => {
          Loop({ start: int(1), end: maxEvents, type: 'int', condition: '<', name: 'ev' }, () => {
            const hit = exit(rayPos, rayDir, size);
            rayPos.assign(hit.pos);
            const hitNormal = hit.nrm.toVar();

            if (isBakedMesh(shape)) {
              rayCone.assign(
                min(
                  rayCone
                    .mul(1.75)
                    .add(hit.t.div(max(reach, 0.05)).mul(0.06))
                    .add(0.03),
                  0.85,
                ),
              );
            } else if (curved) {
              rayCone.assign(min(rayCone.mul(1.5).add(0.01), 0.85));
            }

            attenuation.mulAssign(exp(absorptionSigma.mul(hit.t.negate())));
            const faceFilter = vec3(1).toVar();
            if (isCmyBox) {
              If(cmy.greaterThan(0), () => {
                faceFilter.assign(cmyTint(hitNormal, cmy, g.cmyOrder));
              });
            }

            const exitRefracted = refract(rayDir, hitNormal.negate(), bandIor).toVar();
            const exitFresnel = fresnel(dot(rayDir, hitNormal), float(1).div(bandIor)).toVar();
            If(dot(exitRefracted, exitRefracted).lessThan(1e-6), () => {
              exitFresnel.assign(1);
            });

            const exitWeight = throughput.mul(float(1).sub(exitFresnel)).toVar();
            If(exitWeight.greaterThan(0.01), () => {
              const seen = lookThrough(
                center.add(rotateToWorld(rayPos.add(hitNormal.mul(0.003)), quat)),
                rotateToWorld(exitRefracted, quat),
                reach,
                lampLit,
                rayCone,
              );
              bandRadiance.addAssign(attenuation.mul(faceFilter).mul(exitWeight.mul(seen)));
            });

            throughput.mulAssign(exitFresnel);
            If(throughput.lessThan(0.015), () => {
              Break();
            });
            rayDir.assign(reflect(rayDir, hitNormal.negate()));
            attenuation.mulAssign(faceFilter);
          });

          // Release any residual trapped energy after max bounces so interiors don't go black.
          If(throughput.greaterThanEqual(0.015), () => {
            bandRadiance.addAssign(
              attenuation.mul(
                throughput.mul(
                  lookThrough(
                    center.add(rotateToWorld(rayPos, quat)),
                    rotateToWorld(rayDir, quat),
                    reach,
                    lampLit,
                    rayCone,
                  ),
                ),
              ),
            );
          });
        });

        const bandWeight = select(bandCount.equal(1), vec3(1), sample.weight).toVar();
        color.addAssign(bandRadiance.mul(bandWeight));
        weightSum.addAssign(bandWeight);
      });
    };

    if (shape === KIND.diamond) {
      // Coherent facet-sequence tracing for flat-faceted diamonds at physical dispersion:
      // traces the center wavelength once through the 37–73 facet planes and reuses the flat
      // facet normals across all spectral bands (~6x fewer plane intersections).
      const MAX_DIAMOND_BOUNCES = 5;
      If(bandCount.greaterThan(1).and(dispersionSpread.lessThanEqual(0.15)), () => {
        const hitPositions = Array.from({ length: MAX_DIAMOND_BOUNCES }, () => vec3(0).toVar());
        const hitNormals = Array.from({ length: MAX_DIAMOND_BOUNCES }, () => vec3(0, 1, 0).toVar());
        const hitLengths = Array.from({ length: MAX_DIAMOND_BOUNCES }, () => float(0).toVar());
        const hitCount = int(0).toVar();

        const centerPos = vec3(entryPos).toVar();
        const centerDir = refract(viewDirLocal, entryNormal, float(1).div(baseIor)).toVar();
        const centerThr = float(1).sub(fresnel(viewDotNormal, baseIor)).toVar();

        If(centerThr.greaterThan(0.008).and(dot(centerDir, centerDir).greaterThan(1e-6)), () => {
          for (let b = 0; b < MAX_DIAMOND_BOUNCES; b++) {
            If(
              int(b + 1)
                .lessThan(maxEvents)
                .and(centerThr.greaterThanEqual(0.008)),
              () => {
                const hit = exit(centerPos, centerDir, size);
                centerPos.assign(hit.pos);
                hitPositions[b].assign(hit.pos);
                hitNormals[b].assign(hit.nrm);
                hitLengths[b].assign(hit.t);
                hitCount.assign(int(b + 1));
                const cRefr = refract(centerDir, hit.nrm.negate(), baseIor).toVar();
                const cFres = fresnel(dot(centerDir, hit.nrm), float(1).div(baseIor)).toVar();
                If(dot(cRefr, cRefr).lessThan(1e-6), () => {
                  cFres.assign(1);
                });
                centerThr.mulAssign(cFres);
                centerDir.assign(reflect(centerDir, hit.nrm.negate()));
              },
            );
          }
        });

        Loop({ start: int(0), end: bandCount, type: 'int', condition: '<', name: 'lc' }, ({ lc }: { lc: N }) => {
          const sample = spectralSample(lc, bandCount, spectralU);
          const bandOffset = sample.cauchy;
          const bandIor = max(baseIor.add(dispersionSpread.mul(bandOffset)), 1.0001).toVar();
          const entryFresnel = fresnel(viewDotNormal, bandIor).toVar();
          const bandRadiance = entryFresnel.mul(outerReflection).toVar();
          const throughput = float(1).sub(entryFresnel).toVar();
          const rayDir = refract(viewDirLocal, entryNormal, float(1).div(bandIor)).toVar();
          const attenuation = vec3(1).toVar();

          If(throughput.greaterThan(0.012).and(dot(rayDir, rayDir).greaterThan(1e-6)), () => {
            for (let b = 0; b < MAX_DIAMOND_BOUNCES; b++) {
              If(int(b).lessThan(hitCount).and(throughput.greaterThanEqual(0.015)), () => {
                const hitNormal = hitNormals[b];
                const hitPos = hitPositions[b];
                attenuation.mulAssign(exp(absorptionSigma.mul(hitLengths[b].negate())));
                const cosHit = max(dot(rayDir, hitNormal), 1e-5);
                const exitRefracted = refract(rayDir, hitNormal.negate(), bandIor).toVar();
                const exitFresnel = fresnel(cosHit, float(1).div(bandIor)).toVar();
                If(dot(exitRefracted, exitRefracted).lessThan(1e-6), () => {
                  exitFresnel.assign(1);
                });
                const exitWeight = throughput.mul(float(1).sub(exitFresnel)).toVar();
                If(exitWeight.greaterThan(0.01), () => {
                  const seen = lookThrough(
                    center.add(rotateToWorld(hitPos.add(hitNormal.mul(0.003)), quat)),
                    rotateToWorld(exitRefracted, quat),
                    reach,
                    lampLit,
                    float(0),
                  );
                  bandRadiance.addAssign(attenuation.mul(exitWeight.mul(seen)));
                });
                throughput.mulAssign(exitFresnel);
                rayDir.assign(reflect(rayDir, hitNormal.negate()));
              });
            }
            If(throughput.greaterThanEqual(0.015), () => {
              bandRadiance.addAssign(
                attenuation.mul(
                  throughput.mul(
                    lookThrough(
                      center.add(rotateToWorld(centerPos, quat)),
                      rotateToWorld(rayDir, quat),
                      reach,
                      lampLit,
                      float(0),
                    ),
                  ),
                ),
              );
            });
          });
          const bandWeight = sample.weight.toVar();
          color.addAssign(bandRadiance.mul(bandWeight));
          weightSum.addAssign(bandWeight);
        });
      }).Else(tracePerBand);
    } else {
      tracePerBand();
    }
    color.divAssign(max(weightSum, vec3(1e-6)));

    // Creamy frosted veil + crisp Fresnel rim when roughness > 0.
    const studioLight = s.skyE.add(s.sunE.mul(mix(0.25, 0.45, lampLit))).toVar();
    applyFrostVeil(color, viewDirLocal, entryNormal, baseTint, studioLight, roughness, frostGrain);

    if (mirrored) {
      const mirrorFade = exp(max(worldPos.y, 0).mul(-3.2)).toVar();
      return vec4(color.mul(mirrorFade), mirrorFade);
    }

    // Dim objects shaded by other objects in the main lamp's path.
    const lampShare = dot(s.sunE, vec3(1)).div(max(dot(s.sunE.add(s.skyE), vec3(1)), 1e-4));
    color.mulAssign(mix(float(1).sub(lampShare.mul(0.55)), 1, lampLit));
    // Alpha carries distance from camera for contact shadows & depth of field.
    return vec4(color, length(worldPos.sub(camPos)));
  })();
}
