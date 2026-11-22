// The glass light tracer shared by every glass in the project: the instanced renderer
// (glass.ts) and the standalone MeshGlassMaterial (material.ts) both shade through it,
// so any shape gets the same optics. Per wavelength sample it follows the light in through
// the front surface and through every internal event: exact dielectric Fresnel, total
// internal reflection, Beer-Lambert absorption along each path, and the light leaving at
// each exit seen along its refracted ray. What differs between users is supplied as hooks:
// how a ray finds its next exit (analytic shapes, a baked distance volume, a back-face map,
// an estimate), what an outgoing ray sees, and optional per-surface filters.
/* eslint-disable @typescript-eslint/no-explicit-any -- TSL node types are too loose to spell out */
import * as TSL from 'three/tsl';
import { fresnel, spectralSample } from './optics';

type N = any;
const { Break, If, Loop, dot, exp, float, int, max, reflect, refract, select, vec3 } = TSL as N;

/** Where an internal ray meets the surface: position, outward unit normal, path length. */
export interface GlassHit {
  pos: N;
  nrm: N;
  t: N;
}

export interface GlassTraceInputs {
  /** Entry point on the surface, and its outward normal turned to face the viewer. */
  entryPos: N;
  entryNormal: N;
  /** Unit direction from the eye to the entry point (same space as the positions). */
  viewDir: N;
  /** Base IOR (d line) and IOR spread n_F - n_C. */
  ior: N;
  dispersionSpread: N;
  /** Wavelength samples this pixel takes (1: no dispersion), and its spectral jitter. */
  bandCount: N;
  spectralJitter: N;
  /** Beer-Lambert absorption coefficient per unit path length (vec3). */
  absorptionSigma: N;
  /** Internal events to follow (including the first exit). */
  maxEvents: N;
  /** Light reflected off the outside of the entry surface (vec3, wavelength independent). */
  outerReflection: N;
  /** Next surface along an internal ray from `pos` in direction `dir` (and the event index). */
  exit: (pos: N, dir: N, event: N) => GlassHit;
  /** What a ray leaving the glass at `pos` in direction `dir` sees, with its cone angle. */
  look: (pos: N, dir: N, cone: N) => N;
  /** Ray-cone angle at entry (filters what outgoing rays see; 0: sharp). */
  entryCone?: N;
  /** Widen the ray cone after an internal event of length `t` (default: unchanged). */
  growCone?: (cone: N, hit: GlassHit) => N;
  /** Colour filter of the entry surface (vec3, e.g. CMY dichroic faces; default none). */
  entryFilter?: N;
  /** Colour filter of an internal surface hit (default none). */
  surfaceFilter?: (normal: N) => N;
  /** Adjust a hit's normal in place (e.g. surface imperfections; default none). */
  perturbNormal?: (normal: N, pos: N) => void;
  /** Offset off a surface for the outgoing lookups (default 0.003). */
  surfaceOffset?: number;
}

/**
 * Trace the glass for one pixel: returns `{ color, weightSum }` — the spectrally weighted
 * radiance and the sum of the weights used (divide one by the other, so white light stays
 * exactly white whatever the number of wavelength samples).
 */
export function traceGlass(p: GlassTraceInputs): { color: N; weightSum: N } {
  const {
    entryPos,
    entryNormal,
    viewDir,
    ior,
    dispersionSpread,
    bandCount,
    spectralJitter,
    absorptionSigma,
    maxEvents,
    outerReflection,
    exit,
    look,
  } = p;
  const offset = p.surfaceOffset ?? 0.003;
  const color = vec3(0).toVar();
  const weightSum = vec3(0).toVar();
  const viewDotNormal = dot(viewDir, entryNormal).toVar();

  Loop({ start: int(0), end: bandCount, type: 'int', condition: '<', name: 'lc' }, ({ lc }: { lc: N }) => {
    const sample = spectralSample(lc, bandCount, spectralJitter);
    const bandOffset = select(bandCount.equal(1), float(0), sample.cauchy);
    const bandIor = max(ior.add(dispersionSpread.mul(bandOffset)), 1.0001).toVar();
    const entryFresnel = fresnel(viewDotNormal, bandIor).toVar();
    const bandRadiance = entryFresnel.mul(outerReflection).toVar();
    const throughput = float(1).sub(entryFresnel).toVar();
    const rayDir = refract(viewDir, entryNormal, float(1).div(bandIor)).toVar();
    const rayPos = vec3(entryPos).toVar();
    const rayCone = (p.entryCone ?? float(0)).mul(0.45).toVar();
    const attenuation = vec3(p.entryFilter ?? vec3(1)).toVar();

    If(throughput.greaterThan(0.012).and(dot(rayDir, rayDir).greaterThan(1e-6)), () => {
      Loop({ start: int(1), end: maxEvents, type: 'int', condition: '<', name: 'ev' }, ({ ev }: { ev: N }) => {
        // Read everything about the hit before the ray moves: an exit may hand back
        // expressions of the ray's position, which would be evaluated at the new one.
        const exitHit = exit(rayPos, rayDir, ev);
        const hitPos = vec3(exitHit.pos).toVar();
        const hitNormal = vec3(exitHit.nrm).toVar();
        const hitT = float(exitHit.t).toVar();
        rayPos.assign(hitPos);
        const hit = { pos: hitPos, nrm: hitNormal, t: hitT };
        p.perturbNormal?.(hitNormal, hitPos);
        if (p.growCone) rayCone.assign(p.growCone(rayCone, hit));

        attenuation.mulAssign(exp(absorptionSigma.mul(hitT.negate())));
        const faceFilter = p.surfaceFilter ? vec3(p.surfaceFilter(hitNormal)).toVar() : null;

        const exitRefracted = refract(rayDir, hitNormal.negate(), bandIor).toVar();
        const exitFresnel = fresnel(dot(rayDir, hitNormal), float(1).div(bandIor)).toVar();
        If(dot(exitRefracted, exitRefracted).lessThan(1e-6), () => {
          exitFresnel.assign(1);
        });

        const exitWeight = throughput.mul(float(1).sub(exitFresnel)).toVar();
        If(exitWeight.greaterThan(0.01), () => {
          const seen = look(rayPos.add(hitNormal.mul(offset)), exitRefracted, rayCone);
          const filtered = faceFilter ? attenuation.mul(faceFilter) : attenuation;
          bandRadiance.addAssign(filtered.mul(exitWeight.mul(seen)));
        });

        throughput.mulAssign(exitFresnel);
        If(throughput.lessThan(0.015), () => {
          Break();
        });
        rayDir.assign(reflect(rayDir, hitNormal.negate()));
        if (faceFilter) attenuation.mulAssign(faceFilter);
      });

      // Release any residual trapped energy after the last event so interiors don't go black.
      If(throughput.greaterThanEqual(0.015), () => {
        bandRadiance.addAssign(attenuation.mul(throughput.mul(look(rayPos, rayDir, rayCone))));
      });
    });

    const bandWeight = select(bandCount.equal(1), vec3(1), sample.weight).toVar();
    color.addAssign(bandRadiance.mul(bandWeight));
    weightSum.addAssign(bandWeight);
  });

  return { color, weightSum };
}
