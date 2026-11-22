// Baked 128^3 signed-distance field (SDF) volume sampling in TSL for complex meshes
// (Suzanne = layer 0, Stanford Dragon = layer 1, torus knot = layer 2, stacked along Z
// in one 128x128x384 3D texture).
//
// Provides both:
// - Fast 1-tap hardware-trilinear SDF & 4-tap tetrahedral normal (for sphere-tracing & internal exits)
// - Sigg & Hadwiger (2005) 8-tap tricubic B-spline SDF & C^1-smooth normal (for entry projection)
/* eslint-disable @typescript-eslint/no-explicit-any -- TSL node types are too loose to spell out */
import * as TSL from 'three/tsl';
import { KIND } from '../objects';

type N = any;
const { Break, Fn, If, Loop, clamp, float, floor, int, length, max, mix, normalize, vec3 } = TSL as N;

/** Each baked shape's layer in the stacked volume texture (the order its volumes are loaded in). */
export const SDF_LAYER: Readonly<Record<number, number>> = {
  [KIND.suzanne]: 0,
  [KIND.dragon]: 1,
  [KIND.torusknot]: 2,
};
/** How many 128^3 layers the stacked volume texture holds. */
export const SDF_LAYER_COUNT = 3;

/**
 * Build fast (trilinear) and smooth (C^2 tricubic B-spline) SDF samplers for one mesh layer
 * of a stack of `layerCount` 128^3 volumes.
 */
export function createSdfVolume(sdfTex: N, sdfLayer: number, layerCount = SDF_LAYER_COUNT) {
  const zCells = 128 * layerCount;
  /** Fast 1-tap hardware-trilinear SDF sample for sphere-tracing and internal exits. */
  const volumeFast = Fn(([p, scale]: N[]) => {
    const q = p.div(scale).toVar();
    const inner = 1.1 * (1 - 1 / 128);
    const clamped = clamp(q, vec3(-inner), vec3(inner)).toVar();
    const gridPos = clamp(clamped.div(2.2).add(0.5).mul(128), vec3(0.5), vec3(127.5)).toVar();
    const uvw = vec3(gridPos.x.mul(1 / 128), gridPos.y.mul(1 / 128), gridPos.z.add(128 * sdfLayer).mul(1 / zCells));
    return sdfTex
      .sample(uvw)
      .level(0)
      .r.add(length(q.sub(clamped)))
      .mul(scale);
  }).setLayout({
    name: 'icVolumeFast',
    type: 'float',
    inputs: [
      { name: 'p', type: 'vec3' },
      { name: 'scale', type: 'float' },
    ],
  });

  /**
   * Fast 4-tap tetrahedral gradient from the hardware-trilinear SDF.
   * Evaluates at the 4 vertices of a regular tetrahedron `(±e, ±e, ±e)`:
   *   ∇f(p) ≈ (1 / 4e) Σ k_i * f(p + e * k_i)
   * which is O(e²) accurate like 6-tap central differences while saving 2 texture fetches.
   */
  const volumeFastNormal = Fn(([p, scale]: N[]) => {
    const e = scale.mul(0.014).toVar();
    const xyy = vec3(e, e.negate(), e.negate());
    const yyx = vec3(e.negate(), e.negate(), e);
    const yxy = vec3(e.negate(), e, e.negate());
    const xxx = vec3(e, e, e);
    return normalize(
      xyy
        .mul(volumeFast(p.add(xyy), scale))
        .add(yyx.mul(volumeFast(p.add(yyx), scale)))
        .add(yxy.mul(volumeFast(p.add(yxy), scale)))
        .add(xxx.mul(volumeFast(p.add(xxx), scale))),
    );
  }).setLayout({
    name: 'icVolumeFastNormal',
    type: 'vec3',
    inputs: [
      { name: 'p', type: 'vec3' },
      { name: 'scale', type: 'float' },
    ],
  });

  /**
   * Sigg & Hadwiger (2005) 8-tap hardware-trilinear tricubic B-spline SDF (`C^2` smooth, `C^1` normals).
   *
   * Along each axis, the 4 uniform cubic B-spline basis weights at fractional position `f ∈ [0, 1)`:
   *   w0 = (1 - f)³ / 6,   w1 = (3f³ - 6f² + 4) / 6,   w2 = (-3f³ + 3f² + 3f + 1) / 6,   w3 = f³ / 6
   * are paired into two hardware linear fetches with weights `g0 = w0 + w1`, `g1 = w2 + w3 = 1 - g0`
   * at texel offsets `t0 = i + w1/g0 - 0.5` and `t1 = i + w3/g1 + 1.5`, reducing 4³ = 64 taps to 8.
   */
  const volumeSmooth = Fn(([p, scale]: N[]) => {
    const q = p.div(scale).toVar();
    const inner = 1.1 * (1 - 1 / 128);
    const clamped = clamp(q, vec3(-inner), vec3(inner)).toVar();
    const gridPos = clamped.div(2.2).add(0.5).mul(128).sub(0.5).toVar();
    const i = floor(gridPos).toVar();
    const f = gridPos.sub(i).toVar();
    const f2 = f.mul(f).toVar();
    const f3 = f2.mul(f).toVar();
    const omf = vec3(1).sub(f).toVar();
    const w0 = omf.mul(omf).mul(omf).div(6).toVar();
    const w1 = f3.mul(3).sub(f2.mul(6)).add(4).div(6).toVar();
    const w2 = f3.mul(-3).add(f2.mul(3)).add(f.mul(3)).add(1).div(6).toVar();
    const w3 = f3.div(6).toVar();
    const g0 = w0.add(w1).toVar();
    const g1 = w2.add(w3).toVar();
    // Clamp in-layer texel coordinates to [0.5, 127.5] so adjacent mesh layers never bleed along Z.
    const t0 = clamp(i.add(w1.div(g0)).sub(0.5), vec3(0.5), vec3(127.5)).toVar();
    const t1 = clamp(i.add(w3.div(g1)).add(1.5), vec3(0.5), vec3(127.5)).toVar();
    const invN = float(1 / 128);
    const invZ = float(1 / zCells);
    const zBase = float(128 * sdfLayer);
    const u0 = t0.x.mul(invN).toVar();
    const u1 = t1.x.mul(invN).toVar();
    const v0 = t0.y.mul(invN).toVar();
    const v1 = t1.y.mul(invN).toVar();
    const z0 = t0.z.add(zBase).mul(invZ).toVar();
    const z1 = t1.z.add(zBase).mul(invZ).toVar();
    const sampleAt = (u: N, v: N, z: N) => sdfTex.sample(vec3(u, v, z)).level(0).r;
    const dist = mix(
      mix(
        mix(sampleAt(u0, v0, z0), sampleAt(u1, v0, z0), g1.x),
        mix(sampleAt(u0, v1, z0), sampleAt(u1, v1, z0), g1.x),
        g1.y,
      ),
      mix(
        mix(sampleAt(u0, v0, z1), sampleAt(u1, v0, z1), g1.x),
        mix(sampleAt(u0, v1, z1), sampleAt(u1, v1, z1), g1.x),
        g1.y,
      ),
      g1.z,
    );
    return dist.add(length(q.sub(clamped))).mul(scale);
  }).setLayout({
    name: 'icVolume',
    type: 'float',
    inputs: [
      { name: 'p', type: 'vec3' },
      { name: 'scale', type: 'float' },
    ],
  });

  /** C^1-smooth outward normal from the tricubic B-spline SDF (4-tap tetrahedral estimator). */
  const volumeSmoothNormal = Fn(([p, scale]: N[]) => {
    const e = scale.mul(0.012).toVar();
    const xyy = vec3(e, e.negate(), e.negate());
    const yyx = vec3(e.negate(), e.negate(), e);
    const yxy = vec3(e.negate(), e, e.negate());
    const xxx = vec3(e, e, e);
    return normalize(
      xyy
        .mul(volumeSmooth(p.add(xyy), scale))
        .add(yyx.mul(volumeSmooth(p.add(yyx), scale)))
        .add(yxy.mul(volumeSmooth(p.add(yxy), scale)))
        .add(xxx.mul(volumeSmooth(p.add(xxx), scale))),
    );
  }).setLayout({
    name: 'icVolumeNormal',
    type: 'vec3',
    inputs: [
      { name: 'p', type: 'vec3' },
      { name: 'scale', type: 'float' },
    ],
  });

  /**
   * Trace an internal ray from `pos` along `dir` (volume-local coordinates, object half-width
   * `scale`) to where it leaves the solid: `{ pos, nrm, t }`. Steps inside first if the
   * rasterized mesh slightly leads the isosurface, sphere-traces to the exit wall, bisects it,
   * and takes the 4-tap trilinear normal there.
   */
  const exit = (pos: N, dir: N, scale: N) => {
    const t = scale.mul(0.004).toVar();
    Loop({ start: int(0), end: int(8), type: 'int', condition: '<', name: 'm0' }, () => {
      const d = volumeFast(pos.add(dir.mul(t)), scale).toVar();
      If(d.lessThan(0), () => {
        Break();
      });
      t.addAssign(max(d, scale.mul(0.002)));
    });
    const tIn = float(t).toVar();
    Loop({ start: int(0), end: int(32), type: 'int', condition: '<', name: 'm1' }, () => {
      const d = volumeFast(pos.add(dir.mul(t)), scale)
        .negate()
        .toVar();
      If(d.lessThan(scale.mul(0.0005)), () => {
        Break();
      });
      tIn.assign(t);
      t.addAssign(max(d, scale.mul(0.008)));
    });
    tIn.assign(max(tIn.sub(scale.mul(0.004)), 0));
    t.addAssign(scale.mul(0.004));
    Loop({ start: int(0), end: int(4), type: 'int', condition: '<', name: 'm2' }, () => {
      const mid = tIn.add(t).mul(0.5).toVar();
      If(volumeFast(pos.add(dir.mul(mid)), scale).lessThan(0), () => {
        tIn.assign(mid);
      }).Else(() => {
        t.assign(mid);
      });
    });
    return { pos: pos.add(dir.mul(t)).toVar(), nrm: volumeFastNormal(pos.add(dir.mul(t)), scale), t };
  };

  return { volumeFast, volumeFastNormal, volumeSmooth, volumeSmoothNormal, exit };
}
