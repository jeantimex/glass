// The glass itself, in TSL: every fragment follows the light through its own
// object with exact ray/shape formulas (only the torus and the baked meshes take
// steps along the ray), and looks up whatever the light sees after leaving. Also
// the two passes that light the floor through the glass: the caustic map (light
// multiplied down through every surface) and real caustics (each ray of the lamp
// followed to where it lands).
//
// Each program is built for one shape, so the JavaScript below picks that shape's
// formulas while the shader is built, and each program holds only its own.
//
// (Values used again after something they read has changed are made variables
// with toVar(): a TSL expression is worked out where it is used, not where it is
// written.)
/* eslint-disable @typescript-eslint/no-explicit-any -- TSL node types are too loose to spell out */
import * as TSL from 'three/tsl';
import { KIND } from '../objects';
import { DIAMOND_PLANES } from '../diamond';
import { sceneFunctions, type SceneInputs } from './scene';

type N = any;
const {
  Break, Fn, If, Loop, abs, array, bool, clamp, cross, dFdx, dFdy, dot, exp, float, floor, fract, int, ivec2, length, log, max, min, mix, normalize,
  reflect, refract, select, sign, sin, smoothstep, sqrt, vec2, vec3, vec4,
} = TSL as N;

/** What the glass passes read: the object being drawn, and the settings. */
export interface GlassInputs {
  /** Per object (instance attributes): centre, rotation, size, variation, own colour. */
  center: N;
  quat: N;
  size: N;
  vary: N;
  tintOwn: N;
  /** The colour of all the glass (linear rgb) and how strongly it applies. */
  tintAll: N;
  /** Above 0: boxes are CMY cubes, with filters this strong. */
  cmyAll: N;
  /** Optional CMY face color permutation (vec3 of 0=Cyan, 1=Yellow, 2=Magenta for X, Y, Z). */
  cmyOrder?: N;
  /** How much is cut off the edges of boxes (m). */
  bevel: N;
  /** 1 when light enters a baked mesh through its volume's surface (see entry). */
  meshEntry: N;
  /** IOR, dispersion, how many times light is followed round inside, steps to look for other glass. */
  glassP: N;
  /** The baked meshes' distance volumes, one after the other along z. */
  sdfTex: N;
  /** Frosted glass roughness (0: crystal clear, 1: fully frosted). */
  roughness: N;
  /** Frosted glass grain density (higher = smaller micro-particles). */
  frostScale: N;
}

// ------------------------------------------------------------- helpers ---

/** v rotated into the object's own space, and back. */
const toLocal = (v: N, q: N) => {
  const u = q.xyz.negate();
  return v.add(cross(u, cross(u, v).add(v.mul(q.w))).mul(2));
};
const toWorld = (v: N, q: N) => v.add(cross(q.xyz, cross(q.xyz, v).add(v.mul(q.w))).mul(2));

/** Unpolarised dielectric Fresnel. eta = n_transmitted / n_incident. */
export const fresnel = Fn(([cosI, eta]: N[]) => {
  const c = abs(cosI).toVar();
  const g2 = eta.mul(eta).sub(1).add(c.mul(c)).toVar();
  const out = float(1).toVar();
  If(g2.greaterThan(0), () => {
    const g = sqrt(g2).toVar();
    const a = g.sub(c).div(g.add(c));
    const b = c.mul(g.add(c)).sub(1).div(c.mul(g.sub(c)).add(1));
    out.assign(a.mul(a).mul(b.mul(b).add(1)).mul(0.5));
  });
  return out;
}).setLayout({ name: 'icFresnel', type: 'float', inputs: [{ name: 'cosI', type: 'float' }, { name: 'eta', type: 'float' }] });

/**
 * How much the IOR for red (k = 0) and blue (k = 2) light differs from the quoted
 * IOR (which is for yellow light, 587.6 nm; green, k = 1, uses it as it is), when
 * blue and red are `spread` apart. From Cauchy's law, n = A + B / wavelength^2, at
 * the wavelengths glass makers measure at: red 656.3 nm, blue 486.1 nm. Blue bends
 * away from yellow more than twice as far as red does.
 */
export function iorOffset(spread: N, k: number | N): N {
  if (typeof k === 'number') return k === 0 ? spread.mul(-0.301) : k === 2 ? spread.mul(0.699) : float(0);
  return select(k.equal(0), spread.mul(-0.301), select(k.equal(2), spread.mul(0.699), 0));
}

export interface SpectralBand {
  /** Effective wavelength (nm) of this equal-energy CIE 1931 band. */
  wavelength: number;
  /** Cauchy IOR offset relative to Fraunhofer d (587.6 nm), normalized by (n_F - n_C). */
  cauchy: number;
  /** Normalized linear-sRGB spectral weight (sums to [1, 1, 1] across all bands). */
  rgb: readonly [number, number, number];
}

/**
 * Six equal-energy stratified spectral bands across the visible spectrum (380–720 nm),
 * derived from the CIE 1931 2° standard observer color-matching functions (D65 linear sRGB)
 * and Cauchy's dispersion law relative to Fraunhofer d (587.6 nm), C (656.3 nm), F (486.1 nm).
 *
 * Compared to 3 non-overlapping RGB delta spikes, integrating 6 overlapping spectral bands
 * produces true continuous rainbows on smooth/prismatic surfaces and lets high-curvature
 * geometry naturally average overlapping wavelengths instead of breaking into RGB confetti.
 */
export const SPECTRAL_BANDS: readonly SpectralBand[] = [
  { wavelength: 432.3, cauchy: 1.2845, rgb: [0.0388, 0.0, 0.4409] },   // Violet (380–447 nm)
  { wavelength: 459.9, cauchy: 0.9586, rgb: [0.0, 0.0, 0.4313] },      // Blue (447–475 nm)
  { wavelength: 507.6, cauchy: 0.5155, rgb: [0.0, 0.4643, 0.1278] },   // Cyan-Green (475–538 nm)
  { wavelength: 556.5, cauchy: 0.1739, rgb: [0.0438, 0.4024, 0.0] },   // Yellow-Green (538–573 nm)
  { wavelength: 588.1, cauchy: -0.0025, rgb: [0.3587, 0.1333, 0.0] },  // Amber-Yellow (573–603.5 nm)
  { wavelength: 626.5, cauchy: -0.1826, rgb: [0.5587, 0.0, 0.0] },     // Red (603.5–720 nm)
] as const;

export const SPECTRAL_COUNT = SPECTRAL_BANDS.length;

/** Cauchy IOR offset multiplier for spectral band index `k` (0..5). */
export function spectralCauchy(k: N): N {
  let out: N = float(SPECTRAL_BANDS[SPECTRAL_COUNT - 1].cauchy);
  for (let i = SPECTRAL_COUNT - 2; i >= 0; i--) {
    out = select(k.equal(i), float(SPECTRAL_BANDS[i].cauchy), out);
  }
  return out;
}

/** Normalized linear-sRGB spectral sensitivity weight for band index `k` (0..5). */
export function spectralWeight(k: N): N {
  const last = SPECTRAL_BANDS[SPECTRAL_COUNT - 1].rgb;
  let out: N = vec3(last[0], last[1], last[2]);
  for (let i = SPECTRAL_COUNT - 2; i >= 0; i--) {
    const c = SPECTRAL_BANDS[i].rgb;
    out = select(k.equal(i), vec3(c[0], c[1], c[2]), out);
  }
  return out;
}

export type CmyFaceOrder = 'cym' | 'myc' | 'ycm' | 'mcy' | 'cmy' | 'ymc';

/** Axis color assignments (0 = Cyan, 1 = Yellow, 2 = Magenta) for [X, Y, Z] faces. */
export const CMY_FACE_ORDERS: Record<CmyFaceOrder, readonly [number, number, number]> = {
  cym: [0, 1, 2],
  myc: [2, 1, 0],
  ycm: [1, 0, 2],
  mcy: [2, 0, 1],
  cmy: [0, 2, 1],
  ymc: [1, 2, 0],
};

const cmyColor = (id: N) =>
  select(id.lessThan(0.5), vec3(0.1, 0.92, 1), select(id.lessThan(1.5), vec3(1, 0.9, 0.2), vec3(1, 0.14, 0.92)));

/**
 * CMY cube: each pair of opposite faces is a filter that holds back one primary
 * (cyan, yellow, magenta). `n` is the face normal in the cube's own space;
 * `strength` runs from clear glass (0) to the full filter (1). `order` maps
 * (X, Y, Z) faces to (0=Cyan, 1=Yellow, 2=Magenta). The filters pass their own
 * colours almost untouched, which keeps the cube luminous, and each lets a little
 * of the blocked one through, so three in a row stay a deep colour instead of going grey.
 */
export function cmyTint(n: N, strength: N, order?: N): N {
  const a = abs(n).toVar();
  const xCol = order ? cmyColor(order.x) : vec3(0.1, 0.92, 1);
  const yCol = order ? cmyColor(order.y) : vec3(1, 0.9, 0.2);
  const zCol = order ? cmyColor(order.z) : vec3(1, 0.14, 0.92);
  const passed = select(
    a.x.greaterThanEqual(a.y).and(a.x.greaterThanEqual(a.z)),
    xCol,
    select(a.y.greaterThanEqual(a.z), yCol, zCol),
  );
  return mix(vec3(1), passed, strength);
}

/** 3D value noise for organic frosted-glass micro-texture (ported from glass-effect-webgpu's 3-octave frost_noise). */
const hash13 = (p: N) => fract(sin(dot(p, vec3(127.1, 311.7, 74.7))).mul(43758.5453123));
const valueNoise3 = Fn(([p]: N[]) => {
  const i = floor(p).toVar();
  const f = fract(p).toVar();
  const u = f.mul(f).mul(vec3(3).sub(f.mul(2))).toVar();
  const n000 = hash13(i);
  const n100 = hash13(i.add(vec3(1, 0, 0)));
  const n010 = hash13(i.add(vec3(0, 1, 0)));
  const n110 = hash13(i.add(vec3(1, 1, 0)));
  const n001 = hash13(i.add(vec3(0, 0, 1)));
  const n101 = hash13(i.add(vec3(1, 0, 1)));
  const n011 = hash13(i.add(vec3(0, 1, 1)));
  const n111 = hash13(i.add(vec3(1, 1, 1)));
  const nx00 = mix(n000, n100, u.x);
  const nx10 = mix(n010, n110, u.x);
  const nx01 = mix(n001, n101, u.x);
  const nx11 = mix(n011, n111, u.x);
  return mix(mix(nx00, nx10, u.y), mix(nx01, nx11, u.y), u.z);
}).setLayout({ name: 'icValueNoise3', type: 'float', inputs: [{ name: 'p', type: 'vec3' }] });

/** 3-octave organic frost grain: fine (0.52) + medium (0.33) + broad (0.15). */
export const frostNoise3 = (p: N) =>
  clamp(
    valueNoise3(p.mul(36)).mul(0.52)
      .add(valueNoise3(p.mul(14).add(vec3(19.7, 4.3, 11.2))).mul(0.33))
      .add(valueNoise3(p.mul(5.5).add(vec3(3.1, 28.4, 7.6))).mul(0.15)),
    0,
    1,
  );

// -------------------------------------------------------------- shapes ---
//   0 box         half-extents size.xyz
//   1 sphere      radius size.x
//   2 cylinder    radius size.x, half-height size.y, axis Y
//   3 capsule     radius size.x, half-length of the middle part size.y, axis Y
//   4 cone        base radius size.x, half-height size.y, apex up
//   5 torus       ring radius size.x, tube radius size.y, axis Y
//   6 octahedron  vertices at distance size.x
//   7 prism       triangular: apex at y = size.x, base at y = -size.x / 2, half-length size.z
//   8 hex prism   distance from centre to a side face size.x, half-height size.y
//   9 mesh        Suzanne, baked into a distance volume; half-width size.x
//  10 diamond     round brilliant cut (diamond.ts) of size size.x
//  11 mesh        the Stanford dragon, baked like Suzanne; half-width size.x

const isMesh = (shape: number) => shape === KIND.suzanne || shape === KIND.dragon;

/**
 * The flat faces of a polyhedron as planes (xyz = outward normal, w = offset),
 * each a TSL vec4, or null for the diamond, whose planes are read in a loop. A box
 * counts as one once its edges are bevelled: its six faces plus a flat cut at 45
 * degrees along each of its twelve edges.
 */
function planesOf(shape: number, size: N, bevel: N): N[] | null {
  const s = (k: number) => (k & 1 ? 1 : -1);
  if (shape === KIND.box) {
    const faces = [0, 1, 2, 3, 4, 5].map((k) => {
      const axis = k >> 1;
      const n = [0, 0, 0];
      n[axis] = s(k);
      return vec4(n[0], n[1], n[2], [size.x, size.y, size.z][axis]);
    });
    const r = Math.SQRT1_2;
    const edges = Array.from({ length: 12 }, (_, e) => {
      const a = s(e);
      const b = e & 2 ? 1 : -1;
      const pair = e >> 2;
      if (pair === 0) return vec4(a * r, b * r, 0, size.x.add(size.y).sub(bevel).mul(r));
      if (pair === 1) return vec4(0, a * r, b * r, size.y.add(size.z).sub(bevel).mul(r));
      return vec4(a * r, 0, b * r, size.x.add(size.z).sub(bevel).mul(r));
    });
    return [...faces, ...edges];
  }
  if (shape === KIND.octahedron) {
    const r = 1 / Math.sqrt(3);
    return Array.from({ length: 8 }, (_, k) => vec4(s(k) * r, (k & 2 ? 1 : -1) * r, (k & 4 ? 1 : -1) * r, size.x.mul(r)));
  }
  if (shape === KIND.prism) {
    const h = size.x.mul(0.5);
    return [vec4(0, -1, 0, h), vec4(0.8660254, 0.5, 0, h), vec4(-0.8660254, 0.5, 0, h), vec4(0, 0, 1, size.z), vec4(0, 0, -1, size.z)];
  }
  if (shape === KIND.hexprism) {
    const sides = Array.from({ length: 6 }, (_, k) => {
      const a = ((30 + 60 * k) * Math.PI) / 180;
      return vec4(Math.cos(a), 0, Math.sin(a), size.x);
    });
    return [...sides, vec4(0, 1, 0, size.y), vec4(0, -1, 0, size.y)];
  }
  return null; // the diamond
}

/** The shape functions for one shape: snapping a surface point onto it, and finding where a ray inside leaves. */
export function shapeFunctions(shape: number, g: Pick<GlassInputs, 'bevel'> & Partial<Pick<GlassInputs, 'sdfTex' | 'meshEntry'>>, diamondPlanes: number[] = DIAMOND_PLANES) {
  const { bevel, sdfTex } = g;
  const diamondPlaneCount = diamondPlanes.length / 4;
  // The diamond's facets at size 1 (diamond.ts), read by index: a constant array
  // in the shader (declared by `declare`, at the top of the program). (Read from a
  // uniform buffer instead, the diamond's glass took three times as long.)
  let diamond: N = null;
  const declare = () => {
    if (shape !== KIND.diamond) return;
    const rows = Array.from({ length: diamondPlaneCount }, (_, k) => vec4(...diamondPlanes.slice(k * 4, k * 4 + 4)));
    diamond = array(rows).toConst('icDiamond');
  };

  /** Run `body` for each face plane of the polyhedron. */
  const eachPlane = (size: N, body: (plane: N) => void) => {
    const planes = planesOf(shape, size, bevel);
    if (planes) for (const plane of planes) body(plane);
    else
      Loop({ start: int(0), end: int(diamondPlaneCount), type: 'int', condition: '<', name: 'kp' }, ({ kp }: { kp: N }) => {
        const facet = diamond.element(kp).toVar();
        body(vec4(facet.xyz, facet.w.mul(size.x)));
      });
  };

  // Signed distance to a baked mesh: a 128^3 grid covering a cube of half-width 1.1
  // around the mesh, which is scaled to half-width 1. The volumes of the meshes lie
  // one after another along z in the same texture; this mesh's is `layer`.
  // Fast 1-tap hardware-trilinear SDF sample for coarse sphere-tracing (m0/m1),
  // saving ~8x 3D texture fetches per march step while m2 bisection & volumeNormal
  // evaluate the C^2 tricubic surface below.
  const layer = shape === KIND.dragon ? 1 : 0;
  const volumeFast = Fn(([p, scale]: N[]) => {
    const q = p.div(scale).toVar();
    const inner = 1.1 * (1 - 1 / 128);
    const c = clamp(q, vec3(-inner), vec3(inner)).toVar();
    const gp = clamp(c.div(2.2).add(0.5).mul(128), vec3(0.5), vec3(127.5)).toVar();
    const uvw = vec3(gp.x.mul(1 / 128), gp.y.mul(1 / 128), gp.z.add(128 * layer).mul(1 / 256));
    return sdfTex.sample(uvw).level(0).r.add(length(q.sub(c))).mul(scale);
  }).setLayout({ name: 'icVolumeFast', type: 'float', inputs: [{ name: 'p', type: 'vec3' }, { name: 'scale', type: 'float' }] });

  // Uses Sigg & Hadwiger's 8-tap hardware-trilinear tricubic B-spline filter so the
  // reconstructed SDF is C^2 smooth (C^1 smooth normals, eliminating voxel-grid creases).
  const volume = Fn(([p, scale]: N[]) => {
    const q = p.div(scale).toVar();
    const inner = 1.1 * (1 - 1 / 128);
    const c = clamp(q, vec3(-inner), vec3(inner)).toVar();
    const gp = c.div(2.2).add(0.5).mul(128).sub(0.5).toVar();
    const i = floor(gp).toVar();
    const f = gp.sub(i).toVar();
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
    const invZ = float(1 / 256);
    const zBase = float(128 * layer);
    const u0 = t0.x.mul(invN).toVar();
    const u1 = t1.x.mul(invN).toVar();
    const v0 = t0.y.mul(invN).toVar();
    const v1 = t1.y.mul(invN).toVar();
    const z0 = t0.z.add(zBase).mul(invZ).toVar();
    const z1 = t1.z.add(zBase).mul(invZ).toVar();
    const at = (u: N, v: N, z: N) => sdfTex.sample(vec3(u, v, z)).level(0).r;
    const d = mix(
      mix(mix(at(u0, v0, z0), at(u1, v0, z0), g1.x), mix(at(u0, v1, z0), at(u1, v1, z0), g1.x), g1.y),
      mix(mix(at(u0, v0, z1), at(u1, v0, z1), g1.x), mix(at(u0, v1, z1), at(u1, v1, z1), g1.x), g1.y),
      g1.z,
    );
    return d.add(length(q.sub(c))).mul(scale);
  }).setLayout({ name: 'icVolume', type: 'float', inputs: [{ name: 'p', type: 'vec3' }, { name: 'scale', type: 'float' }] });

  // The outward direction of a baked mesh's surface at p (from its C^2 tricubic volume).
  const volumeNormal = Fn(([p, scale]: N[]) => {
    const e = scale.mul(0.012).toVar();
    const xyy = vec3(e, e.negate(), e.negate());
    const yyx = vec3(e.negate(), e.negate(), e);
    const yxy = vec3(e.negate(), e, e.negate());
    const xxx = vec3(e, e, e);
    return normalize(
      xyy.mul(volume(p.add(xyy), scale)).add(yyx.mul(volume(p.add(yyx), scale))).add(yxy.mul(volume(p.add(yxy), scale))).add(xxx.mul(volume(p.add(xxx), scale))),
    );
  }).setLayout({ name: 'icVolumeNormal', type: 'vec3', inputs: [{ name: 'p', type: 'vec3' }, { name: 'scale', type: 'float' }] });

  /** Snap a rasterised surface point onto the exact shape and find its normal. */
  const entry = (pIn: N, size: N, meshNormal: N) => {
    const p = vec3(pIn).toVar();
    const n = vec3(0, 1, 0).toVar();
    const polyhedron = () => {
      // The face this point lies on is the plane it is closest to.
      const best = float(-1e9).toVar();
      eachPlane(size, (plane) => {
        const pl = plane.toVar();
        const sd = dot(pl.xyz, p).sub(pl.w).toVar();
        If(sd.greaterThan(best), () => {
          best.assign(sd);
          n.assign(pl.xyz);
        });
      });
      p.subAssign(n.mul(best));
    };
    if (isMesh(shape)) {
      If(g.meshEntry.greaterThan(0.5), () => {
        // A baked mesh, with only a few objects on show (each large on screen): the
        // point moved onto the volume's surface, with the volume's normal there, so
        // the light enters the same shape it is followed through and leaves. (The
        // mesh and its volume differ by up to about a cell: with the light entering
        // one and leaving the other, the bands where they differ show as jagged dark
        // patches.) The point moves a fraction of a cell, so the normal is found
        // once for the move and once where it lands.
        const toward = volumeNormal(p, size.x).toVar();
        p.subAssign(toward.mul(volume(p, size.x)));
        p.subAssign(toward.mul(volume(p, size.x)));
        n.assign(volumeNormal(p, size.x));
      }).Else(() => {
        // In a pile, where each is small on screen, the cheaper way: the rasterised
        // point and the mesh's own smooth normal.
        n.assign(meshNormal);
      });
    } else if (shape === KIND.torusknot) {
      n.assign(meshNormal);
    } else if (shape === KIND.cone) {
      const k = size.x.div(size.y.mul(2)).toVar();
      const rho = max(length(p.xz), 1e-6).toVar();
      const inv = float(1).div(sqrt(k.mul(k).add(1))).toVar();
      const sBase = p.y.negate().sub(size.y).toVar();
      const sSide = rho.sub(k.mul(size.y.sub(p.y))).mul(inv).toVar();
      If(sBase.greaterThanEqual(sSide), () => {
        n.assign(vec3(0, -1, 0));
        p.assign(vec3(p.x, size.y.negate(), p.z));
      }).Else(() => {
        n.assign(vec3(p.x.div(rho), k, p.z.div(rho)).mul(inv));
        p.subAssign(n.mul(sSide));
      });
    } else if (shape === KIND.torus) {
      const rho = max(length(p.xz), 1e-6).toVar();
      const ring = vec3(p.x.div(rho), 0, p.z.div(rho)).mul(size.x).toVar();
      n.assign(normalize(p.sub(ring)));
      p.assign(ring.add(n.mul(size.y)));
    } else if (shape === KIND.sphere) {
      n.assign(normalize(p));
      p.assign(n.mul(size.x));
    } else if (shape === KIND.cylinder) {
      const radial = length(p.xz).toVar();
      If(abs(p.y).div(size.y).greaterThanEqual(radial.div(size.x)), () => {
        n.assign(vec3(0, sign(p.y), 0));
        p.assign(vec3(p.x, n.y.mul(size.y), p.z));
      }).Else(() => {
        n.assign(vec3(p.x, 0, p.z).div(max(radial, 1e-6)));
        p.assign(vec3(n.x.mul(size.x), p.y, n.z.mul(size.x)));
      });
    } else if (shape === KIND.capsule) {
      const c = vec3(0, clamp(p.y, size.y.negate(), size.y), 0).toVar();
      n.assign(normalize(p.sub(c)));
      p.assign(c.add(n.mul(size.x)));
    } else if (shape === KIND.box) {
      If(bevel.greaterThan(0), polyhedron).Else(() => {
        const a = abs(p).div(size).toVar();
        If(a.x.greaterThanEqual(a.y).and(a.x.greaterThanEqual(a.z)), () => {
          n.assign(vec3(sign(p.x), 0, 0));
          p.assign(vec3(n.x.mul(size.x), p.y, p.z));
        })
          .ElseIf(a.y.greaterThanEqual(a.z), () => {
            n.assign(vec3(0, sign(p.y), 0));
            p.assign(vec3(p.x, n.y.mul(size.y), p.z));
          })
          .Else(() => {
            n.assign(vec3(0, 0, sign(p.z)));
            p.assign(vec3(p.x, p.y, n.z.mul(size.z)));
          });
      });
    } else {
      polyhedron(); // octahedron, prism, hex prism, diamond
    }
    return { pos: p, nrm: n };
  };

  /**
   * From a point on the surface, heading inwards along dir: where the ray reaches
   * the surface again, the outward normal there, and how far it travelled.
   */
  const exit = (pos: N, dir: N, size: N) => {
    const t = float(0).toVar();
    const n = vec3(0, 1, 0).toVar();
    const polyhedron = () => {
      // The nearest face plane the ray is heading towards.
      t.assign(1e9);
      eachPlane(size, (plane) => {
        const pl = plane.toVar();
        const dn = dot(pl.xyz, dir).toVar();
        If(dn.greaterThan(1e-7), () => {
          const tk = pl.w.sub(dot(pl.xyz, pos)).div(dn).toVar();
          If(tk.lessThan(t), () => {
            t.assign(tk);
            n.assign(pl.xyz);
          });
        });
      });
      t.assign(select(t.greaterThan(1e8), 0, max(t, 0)));
    };
    if (shape === KIND.sphere) {
      t.assign(max(dot(pos, dir).mul(-2), 0));
      n.assign(normalize(pos.add(dir.mul(t))));
    } else if (shape === KIND.cylinder || shape === KIND.capsule) {
      // Distance to the round side wall.
      const a = dot(dir.xz, dir.xz).toVar();
      const tSide = float(1e9).toVar();
      If(a.greaterThan(1e-9), () => {
        const b = dot(pos.xz, dir.xz).toVar();
        const c = dot(pos.xz, pos.xz).sub(size.x.mul(size.x));
        tSide.assign(b.negate().add(sqrt(max(b.mul(b).sub(a.mul(c)), 0))).div(a));
      });
      if (shape === KIND.cylinder) {
        const tCap = float(1e9).toVar();
        If(abs(dir.y).greaterThan(1e-7), () => {
          tCap.assign(sign(dir.y).mul(size.y).sub(pos.y).div(dir.y));
        });
        t.assign(max(min(tSide, tCap), 0));
        If(tCap.lessThan(tSide), () => {
          n.assign(vec3(0, sign(dir.y), 0));
        }).Else(() => {
          const q = pos.add(dir.mul(t));
          n.assign(normalize(vec3(q.x, 0, q.z)));
        });
      } else {
        // Capsule: the side wall if the hit is on the middle part, else an end cap.
        const ySide = pos.y.add(dir.y.mul(min(tSide, 1e6))).toVar();
        If(abs(ySide).lessThanEqual(size.y), () => {
          t.assign(max(tSide, 0));
          const q = pos.add(dir.mul(t));
          n.assign(normalize(vec3(q.x, 0, q.z)));
        }).Else(() => {
          const cap = vec3(0, sign(ySide).mul(size.y), 0).toVar();
          const oc = pos.sub(cap).toVar();
          const b = dot(oc, dir).toVar();
          t.assign(max(b.negate().add(sqrt(max(b.mul(b).sub(dot(oc, oc).sub(size.x.mul(size.x))), 0))), 0));
          n.assign(normalize(pos.add(dir.mul(t)).sub(cap)));
        });
      }
    } else if (shape === KIND.cone) {
      // The slanted side is a quadratic in t, the base a plane.
      const k = size.x.div(size.y.mul(2)).toVar();
      const k2 = k.mul(k).toVar();
      const hy = size.y.sub(pos.y).toVar();
      const a = dot(dir.xz, dir.xz).sub(k2.mul(dir.y).mul(dir.y)).toVar();
      const b = dot(pos.xz, dir.xz).add(k2.mul(hy).mul(dir.y)).toVar();
      const c = dot(pos.xz, pos.xz).sub(k2.mul(hy).mul(hy)).toVar();
      const tSide = float(1e9).toVar();
      If(abs(a).greaterThan(1e-7), () => {
        const disc = b.mul(b).sub(a.mul(c)).toVar();
        If(disc.greaterThanEqual(0), () => {
          const sq = sqrt(disc).toVar();
          for (const r of [-1, 1]) {
            const tr = b.negate().add(sq.mul(r)).div(a).toVar();
            const y = pos.y.add(dir.y.mul(tr));
            If(tr.greaterThan(1e-4).and(y.lessThanEqual(size.y)).and(y.greaterThanEqual(size.y.negate().sub(1e-4))), () => {
              tSide.assign(min(tSide, tr));
            });
          }
        });
      }).ElseIf(abs(b).greaterThan(1e-9), () => {
        const tl = c.negate().div(b.mul(2)).toVar();
        const yl = pos.y.add(dir.y.mul(tl));
        If(tl.greaterThan(1e-4).and(yl.lessThanEqual(size.y)).and(yl.greaterThanEqual(size.y.negate().sub(1e-4))), () => {
          tSide.assign(tl);
        });
      });
      const tBase = float(1e9).toVar();
      If(dir.y.lessThan(-1e-7), () => {
        tBase.assign(size.y.negate().sub(pos.y).div(dir.y));
      });
      const nearest = min(tSide, tBase).toVar();
      t.assign(select(nearest.greaterThan(1e8), 0, nearest));
      If(tBase.lessThan(tSide), () => {
        n.assign(vec3(0, -1, 0));
      }).Else(() => {
        const q = pos.add(dir.mul(t)).toVar();
        const rho = max(length(q.xz), 1e-6);
        n.assign(normalize(vec3(q.x.div(rho), k, q.z.div(rho))));
      });
    } else if (shape === KIND.torus) {
      // A quartic, so step along the ray using its exact distance instead.
      const tube = max(size.y, 1e-4).toVar();
      t.assign(tube.mul(0.04));
      Loop({ start: int(0), end: int(32), type: 'int', condition: '<', name: 'ts' }, () => {
        const q = pos.add(dir.mul(t)).toVar();
        const d = size.y.sub(length(vec2(length(q.xz).sub(size.x), q.y))).toVar();
        If(d.lessThan(tube.mul(0.006)), () => {
          Break();
        });
        t.addAssign(max(d, tube.mul(0.015)));
      });
      const q = pos.add(dir.mul(t)).toVar();
      const rho = max(length(q.xz), 1e-6);
      n.assign(normalize(q.sub(vec3(q.x.div(rho), 0, q.z.div(rho)).mul(size.x))));
    } else if (isMesh(shape)) {
      // Baked mesh: step through the distance volume. (Distances are in units of the
      // object's size, so a small one in a pile is stepped through as finely as a
      // large one on its own. A cell of the volume is 2.2 / 128 = 0.017 of it.)
      const s = size.x.toVar();
      t.assign(s.mul(0.004));
      // The ray starts on the mesh, which the volume follows only to within about a
      // cell: where the volume's surface lies a little inside the mesh's, the start
      // is outside it. Step forward until inside, or the exit is found at once, in
      // the wrong place: a dark cut across the glass.
      Loop({ start: int(0), end: int(8), type: 'int', condition: '<', name: 'm0' }, () => {
        const d = volumeFast(pos.add(dir.mul(t)), s).toVar();
        If(d.lessThan(0), () => {
          Break();
        });
        t.addAssign(max(d, s.mul(0.002)));
      });
      // Then to where it leaves. Steps of at least a quarter of a cell: a ray running
      // along the inside of a long body, close to its surface all the way, would
      // otherwise only creep forward and run out of steps somewhere in the middle.
      const tIn = float(t).toVar(); // the last point known to be inside
      Loop({ start: int(0), end: int(48), type: 'int', condition: '<', name: 'm1' }, () => {
        const d = volumeFast(pos.add(dir.mul(t)), s).negate().toVar();
        If(d.lessThan(s.mul(0.0005)), () => {
          Break();
        });
        tIn.assign(t);
        t.addAssign(max(d, s.mul(0.004)));
      });
      // ... and the C^2 tricubic surface pinned down by bisecting across the bracket.
      tIn.assign(max(tIn.sub(s.mul(0.004)), 0));
      t.addAssign(s.mul(0.004));
      Loop({ start: int(0), end: int(4), type: 'int', condition: '<', name: 'm2' }, () => {
        const mid = tIn.add(t).mul(0.5).toVar();
        If(volume(pos.add(dir.mul(mid)), s).lessThan(0), () => {
          tIn.assign(mid);
        }).Else(() => {
          t.assign(mid);
        });
      });
      n.assign(volumeNormal(pos.add(dir.mul(t)), s));
    } else if (shape === KIND.torusknot) {
      // Arbitrary smooth knot mesh: curved thick-shell exit estimate (fast, zero marching).
      t.assign(size.x.mul(0.45));
      n.assign(normalize(dir.add(normalize(pos).mul(0.35))));
    } else if (shape === KIND.box) {
      If(bevel.greaterThan(0), polyhedron).Else(() => {
        const along = (d: N, half: N, p: N) => select(abs(d).greaterThan(1e-7), sign(d).mul(half).sub(p).div(d), 1e9);
        const tx = along(dir.x, size.x, pos.x).toVar();
        const ty = along(dir.y, size.y, pos.y).toVar();
        const tz = along(dir.z, size.z, pos.z).toVar();
        t.assign(max(min(tx, min(ty, tz)), 0));
        If(tx.lessThanEqual(ty).and(tx.lessThanEqual(tz)), () => {
          n.assign(vec3(sign(dir.x), 0, 0));
        })
          .ElseIf(ty.lessThanEqual(tz), () => {
            n.assign(vec3(0, sign(dir.y), 0));
          })
          .Else(() => {
            n.assign(vec3(0, 0, sign(dir.z)));
          });
      });
    } else {
      polyhedron(); // octahedron, prism, hex prism, diamond
    }
    return { pos: pos.add(dir.mul(t)).toVar(), nrm: n, t };
  };

  return { declare, entry, exit };
}

// --------------------------------------------------------------- glass ---

/** What the glass pass reads besides the object: the camera, the room, and the picture behind. */
export interface GlassViewInputs {
  camPos: N;
  viewProj: N;
  /** The picture of what is behind this object (ground plus the glass drawn before it), with distance in alpha. */
  behindTex: N;
  /** Blurred copy of behindTex for frosted glass transmission. */
  behindBlurTex: N;
  /** The lamp's depth map (red: how far toward the lamp, plus 100), and its size in texels. */
  lampDepthTex: N;
  lampDepthSize: number;
}

/**
 * The glass objects: each fragment follows the light into its object and out, in
 * three colour layers when there is dispersion.
 * @param worldPos the surface point (upside down for the mirror image), and its normal
 * @param mirrored the mirror image in the floor, shaded as the object itself
 */
export function glassNode(shape: number, mirrored: boolean, worldPos: N, nrmW: N, g: GlassInputs, s: SceneInputs, v: GlassViewInputs, diamondPlanes?: number[]): N {
  const { center, quat, size, vary, tintOwn, tintAll, cmyAll, glassP, roughness, frostScale } = g;
  const { camPos, viewProj, behindTex, behindBlurTex } = v;
  const { groundOn, scene } = sceneFunctions({ ...s, roughness }, false);
  const { declare, entry, exit } = shapeFunctions(shape, g, diamondPlanes);

  // How much of the lamp reaches point p of an object (1: all of it, 0: none), from
  // the lamp's depth map: in shadow where another object reaches farther toward the
  // lamp than `top`, the farthest this object itself reaches. (An object does not
  // shadow itself: what its own glass does to the light is followed through it.)
  // Four samples soften the edge.
  const lampVisible = (p: N, top: N) => {
    const lit = float(1).toVar();
    const w = p.sub(s.lightC);
    const uv = vec2(dot(w, s.lightR).mul(0.5).add(0.5), float(0.5).sub(dot(w, s.lightU).mul(0.5))).toVar();
    If(uv.x.greaterThan(0).and(uv.y.greaterThan(0)).and(uv.x.lessThan(1)).and(uv.y.lessThan(1)), () => {
      const dims = v.lampDepthSize;
      const at = uv.mul(dims).toVar();
      const count = float(0).toVar();
      for (let k = 0; k < 4; k++) {
        const texel = clamp(ivec2(at.add(vec2(k & 1 ? 0.75 : -0.75, k & 2 ? 0.75 : -0.75))), ivec2(0), ivec2(dims - 1));
        const reach = v.lampDepthTex.load(texel).r.sub(100);
        count.addAssign(select(reach.greaterThan(top), 0, 1));
      }
      lit.assign(count.mul(0.25));
    });
    return lit;
  };

  // What a ray sees after leaving its object through glass. Ground points are read
  // from the picture of what is behind this object at the pixel where that point
  // appears; off screen it falls back to the ground alone. Everything else is the
  // studio. Before that, a refracted ray is followed out in growing steps, looking
  // for other glass in that picture (which also holds how far away each pixel is):
  // when the ray passes just behind something drawn there, that is what it sees. So
  // glass shows through glass along the bent ray, not only straight behind. (The
  // mirror image sees only the studio: the picture behind is the camera's, not its.)
  const screenUv = (ndc: N) => vec2(ndc.x, ndc.y.negate()).mul(0.5).add(0.5);
  const lookThrough = mirrored
    ? (ro: N, rd: N, _reach: N, lampLit: N, cone: N) => scene(ro, rd, lampLit, cone)
    : Fn(([ro, rd, reach, lampLit, cone]: N[]) => {
        const out = vec3(0).toVar();
        const found = bool(false).toVar();
        const toFloor = groundOn.greaterThan(0.5).and(rd.y.lessThan(-1e-5));
        If(glassP.w.greaterThan(0.5), () => {
          const tEnd = float(1e9).toVar(); // no farther than the floor
          If(toFloor, () => {
            tEnd.assign(max(ro.y.negate().div(rd.y), 0));
          });
          const t = reach.mul(0.2).toVar();
          Loop({ start: int(0), end: glassP.w.toInt(), type: 'int', condition: '<', name: 'ls' }, () => {
            If(t.greaterThanEqual(tEnd), () => {
              Break();
            });
            const p = ro.add(rd.mul(t)).toVar();
            const clip = viewProj.mul(vec4(p, 1)).toVar();
            If(clip.w.lessThan(0.01), () => {
              Break();
            });
            const ndc = clip.xy.div(clip.w).toVar();
            If(abs(ndc.x).greaterThan(0.98).or(abs(ndc.y).greaterThan(0.98)), () => {
              Break();
            });
            const suv = screenUv(ndc).toVar();
            const there = behindTex.sample(suv).level(0).toVar();
            // Behind what is drawn there, but not by more than an object's depth: a hit.
            const behind = length(p.sub(camPos)).sub(there.a).toVar();
            If(behind.greaterThan(0).and(behind.lessThan(reach.mul(1.5))), () => {
              out.assign(there.rgb);
              If(roughness.greaterThan(0.001), () => {
                const blurMix = clamp(sqrt(roughness).mul(float(0.65).add(smoothstep(0, reach.mul(2), t).mul(0.35))), 0, 1);
                out.assign(mix(there.rgb, behindBlurTex.sample(suv).level(0).rgb, blurMix));
              });
              found.assign(true);
              Break();
            });
            t.mulAssign(1.6);
          });
        });
        If(found.not().and(toFloor), () => {
          const tg = max(ro.y.negate().div(rd.y), 0);
          const clip = viewProj.mul(vec4(ro.add(rd.mul(tg)), 1)).toVar();
          If(clip.w.greaterThan(0.01), () => {
            const ndc = clip.xy.div(clip.w).toVar();
            If(abs(ndc.x).lessThan(0.98).and(abs(ndc.y).lessThan(0.98)), () => {
              const suv = screenUv(ndc).toVar();
              const sharp = behindTex.sample(suv).level(0).rgb;
              out.assign(sharp);
              If(roughness.greaterThan(0.001), () => {
                const blurMix = clamp(sqrt(roughness).mul(float(0.65).add(smoothstep(0, reach.mul(2), tg).mul(0.35))), 0, 1);
                out.assign(mix(sharp, behindBlurTex.sample(suv).level(0).rgb, blurMix));
              });
              found.assign(true);
            });
          });
        });
        If(found.not(), () => {
          out.assign(scene(ro, rd, lampLit, cone));
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
    const reach = max(size.x, max(size.y, size.z)).toVar(); // about how big this object is (m)
    // A CMY cube: every box when the whole pile is, or just the objects marked as one.
    const cmy = max(cmyAll, vary.z).toVar();
    const cmyBox = shape === KIND.box;
    // In the shadow of other objects? (The mirror image is drawn upside down below
    // the floor: the object itself is above it.) The lamp is dimmed for this object
    // in all it sees of the studio (see env).
    const here = mirrored ? worldPos.mul(vec3(1, -1, 1)) : worldPos;
    const lampLit = lampVisible(here, dot(center, s.lamp).add(length(size).mul(1.25))).toVar();
    const d0 = toLocal(normalize(worldPos.sub(camPos)), quat).toVar();

    // Entry point and normal in the object's own space.
    const { pos: p0, nrm: n0 } = entry(toLocal(worldPos.sub(center), quat), size, toLocal(normalize(nrmW), quat));
    If(dot(d0, n0).greaterThan(0), () => {
      n0.assign(n0.negate());
    });
    // Physical ray-cone angular footprint at entry (in uniform control flow for WGSL).
    // Polyhedra have flat faces (zero curvature); curved shapes measure 1-pixel normal spread.
    const isCurved =
      shape === KIND.sphere ||
      shape === KIND.cylinder ||
      shape === KIND.capsule ||
      shape === KIND.cone ||
      shape === KIND.torus ||
      isMesh(shape) ||
      shape === KIND.torusknot;
    const dnX = dFdx(n0);
    const dnY = dFdy(n0);
    const entryCone = isCurved ? clamp(sqrt(max(dot(dnX, dnX), dot(dnY, dnY))), 0, 0.85).toVar() : float(0).toVar();
    const reflCone = isCurved ? clamp(entryCone.mul(0.6), 0, 0.6).toVar() : float(0).toVar();
    // Organic 3-octave frosted surface grain (ported from glass-effect-webgpu's frost_noise).
    const frostGrain = float(0.5).toVar();
    If(roughness.greaterThan(0.001), () => {
      const fs = max(frostScale, 0.25).toVar();
      const fp = p0.div(max(reach, 0.05)).mul(fs).toVar();
      const fn0 = frostNoise3(fp).toVar();
      frostGrain.assign(fn0);
      const eps = float(0.008).toVar();
      const gx = frostNoise3(fp.add(vec3(eps, 0, 0))).sub(fn0);
      const gy = frostNoise3(fp.add(vec3(0, eps, 0))).sub(fn0);
      const gz = frostNoise3(fp.add(vec3(0, 0, eps))).sub(fn0);
      const perturbed = normalize(n0.add(vec3(gx, gy, gz).mul(roughness.mul(0.45)))).toVar();
      If(dot(d0, perturbed).lessThan(-0.02), () => {
        n0.assign(perturbed);
      });
    });

    // Each object can differ slightly from the settings: vary.x is added to the IOR,
    // vary.y multiplies the dispersion.
    const ior = max(glassP.x.add(vary.x), 1.0001).toVar();
    // How far apart the IOR is for blue and red light (n_F - n_C): 1/12 for each unit
    // of the dispersion setting, far more than real glass shows, for visible fire.
    const disp = glassP.y.mul(vary.y).div(12).toVar();
    // The baked mesh is the one shape that has to be marched, so it gets fewer bounces.
    const maxEvents = isMesh(shape) || shape === KIND.torusknot ? min(glassP.z.toInt(), int(4)) : glassP.z.toInt();
    const layers = select(disp.greaterThan(0), int(SPECTRAL_COUNT), int(1)).toVar();
    const col = vec3(0).toVar();
    // Volumetric Beer-Lambert absorption: `baseTint` is the transmittance across one
    // characteristic object diameter (2 * reach), so sigma = -ln(baseTint) / (2 * reach).
    // Clear glass (white) has sigma = 0. CMY cubes additionally apply dichroic surface
    // filters at entry, internal bounces, and exit.
    const baseTint = mix(vec3(1), tintAll.rgb.mul(tintOwn), tintAll.a).toVar();
    const sigma = log(max(baseTint, vec3(1e-4))).negate().div(max(reach.mul(2), 1e-4)).toVar();
    const entryFace = vec3(1).toVar();
    if (cmyBox)
      If(cmy.greaterThan(0), () => {
        entryFace.assign(cmyTint(n0, cmy, g.cmyOrder));
      });

    // Hoist wavelength-independent outer reflection outside the spectral loop.
    const d0DotN0 = dot(d0, n0).toVar();
    const outerRefl = scene(worldPos.add(toWorld(n0, quat).mul(0.003)), toWorld(reflect(d0, n0), quat), lampLit, reflCone).toVar();

    Loop({ start: int(0), end: layers, type: 'int', condition: '<', name: 'lc' }, ({ lc }: { lc: N }) => {
      // One CIE 1931 spectral band with its own Cauchy IOR (or monochrome when disp == 0).
      const bandOffset = select(layers.equal(1), float(0), spectralCauchy(lc));
      const iorIn = max(ior.add(disp.mul(bandOffset)), 1.0001).toVar();
      const f0 = fresnel(d0DotN0, iorIn).toVar();
      const layer = f0.mul(outerRefl).toVar();
      const thr = float(1).sub(f0).toVar(); // light still inside, weighted by Fresnel transmission
      const dir = refract(d0, n0, float(1).div(iorIn)).toVar();
      const pos = vec3(p0).toVar();
      const cone = entryCone.mul(0.45).toVar();
      // A CMY cube's faces are colour filters: light is filtered by the face it enters
      // through, by each face it bounces off inside, and by the face it leaves
      // through. Volumetric Beer-Lambert absorption is accumulated per segment below.
      const filtered = vec3(entryFace).toVar();
      If(thr.greaterThan(0.004).and(dot(dir, dir).greaterThan(1e-6)), () => {
        Loop({ start: int(1), end: maxEvents, type: 'int', condition: '<', name: 'ev' }, () => {
          // Where the ray meets the inside of the surface again, exactly.
          const hit = exit(pos, dir, size);
          pos.assign(hit.pos);
          const nrm = hit.nrm.toVar();
          if (isMesh(shape)) {
            cone.assign(min(cone.mul(1.75).add(hit.t.div(max(reach, 0.05)).mul(0.06)).add(0.03), 0.85));
          } else if (isCurved) {
            cone.assign(min(cone.mul(1.5).add(0.01), 0.85));
          }
          filtered.mulAssign(exp(sigma.mul(hit.t.negate())));
          const face = vec3(1).toVar();
          if (cmyBox)
            If(cmy.greaterThan(0), () => {
              face.assign(cmyTint(nrm, cmy, g.cmyOrder));
            });
          // Each wavelength channel exits with its own IOR (conservation of wavelength).
          const rf = refract(dir, nrm.negate(), iorIn).toVar();
          const f = fresnel(dot(dir, nrm), float(1).div(iorIn)).toVar();
          If(dot(rf, rf).lessThan(1e-6), () => {
            f.assign(1);
          });
          const outW = thr.mul(float(1).sub(f)).toVar();
          If(outW.greaterThan(0.004), () => {
            const seen = lookThrough(center.add(toWorld(pos.add(nrm.mul(0.003)), quat)), toWorld(rf, quat), reach, lampLit, cone);
            layer.addAssign(filtered.mul(face).mul(outW.mul(seen)));
          });
          thr.mulAssign(f);
          If(thr.lessThan(0.004), () => {
            Break();
          });
          dir.assign(reflect(dir, nrm.negate()));
          filtered.mulAssign(face);
        });
        // Light still bouncing around inside when the bounces run out (common in cones
        // and octahedra) is let out, rather than left as a black patch.
        If(thr.greaterThanEqual(0.004), () => {
          layer.addAssign(filtered.mul(thr.mul(lookThrough(center.add(toWorld(pos, quat)), toWorld(dir, quat), reach, lampLit, cone))));
        });
      });
      const weight = select(layers.equal(1), vec3(1), spectralWeight(lc));
      col.addAssign(layer.mul(weight));
    });
    // Creamy frosted veil + crisp Fresnel rim (inspired by glass-effect-webgpu's frosted look):
    // lifts dark interiors into a soft, lit translucent milkiness modulated by organic frost grain,
    // while keeping a clean specular rim highlight around grazing edges.
    If(roughness.greaterThan(0.001), () => {
      const veilOpacity = roughness.mul(0.42).mul(frostGrain.mul(0.24).add(0.88)).toVar();
      const studioLight = s.skyE.add(s.sunE.mul(mix(0.25, 0.45, lampLit))).toVar();
      const frostTint = baseTint.mul(studioLight).toVar();
      const lifted = col.mul(1.03).add(frostTint.mul(0.22));
      const frosted = mix(mix(col, lifted, veilOpacity), frostTint, veilOpacity.mul(0.55));
      const ndotv = clamp(dot(d0, n0).negate(), 0, 1);
      const rim = float(1).sub(ndotv).toVar();
      const rimGlow = rim.mul(rim).mul(rim).mul(roughness.mul(0.35)).mul(studioLight);
      col.assign(frosted.add(rimGlow));
    });
    if (mirrored) {
      const fade = exp(max(worldPos.y, 0).mul(-3.2)).toVar();
      return vec4(col.mul(fade), fade);
    }
    // In the shadow of other objects, the light the lamp would have sent through it
    // is mostly missing: darker by the lamp's share of all the light in the studio
    // (about a third in the white studio, half under the spotlight), less what the
    // glass in the way lets through.
    const lampShare = dot(s.sunE, vec3(1)).div(max(dot(s.sunE.add(s.skyE), vec3(1)), 1e-4));
    col.mulAssign(mix(float(1).sub(lampShare.mul(0.55)), 1, lampLit));
    // (Alpha: how far away this pixel is, for the finishing passes.)
    return vec4(col, length(worldPos.sub(camPos)));
  })();
}

// ------------------------------------------------------- real caustics ---

/**
 * Real caustics, first half. Objects are drawn from the main lamp's direction, so
 * every pixel here is one ray of its light. Each is followed through its object
 * (in at this point, round the inside for as long as it cannot get out, then out)
 * and on down to the floor. The answer is how far from its straight path the ray
 * lands, in the units of the caustic map (xy), and how much of the light is left
 * (z). The second half moves every one of these rays to where it lands and adds
 * them up, which is bright wherever the glass gathers them together.
 * @param raysAcross how many rays there are each way (the width of this picture)
 */
export function photonNode(shape: number, worldPos: N, nrmW: N, g: GlassInputs, sunDir: N, lightR: N, lightU: N, raysAcross: number, diamondPlanes?: number[]): N {
  const { center, quat, size, vary, tintOwn, tintAll, cmyAll, glassP } = g;
  const { declare, entry, exit } = shapeFunctions(shape, g, diamondPlanes);
  return Fn((builder: N) => {
    declare();
    const cmy = max(cmyAll, vary.z).toVar();
    const d0 = toLocal(sunDir.negate(), quat).toVar();
    const { pos, nrm: n0 } = entry(toLocal(worldPos.sub(center), quat), size, toLocal(normalize(nrmW), quat));
    If(dot(d0, n0).greaterThan(0), () => {
      n0.assign(n0.negate());
    });
    const ior = max(glassP.x.add(vary.x), 1.0001).toVar();
    // Light that only grazes the surface is thrown wide in a thin, faint streak: it
    // is faded out, which keeps the pool of light clean.
    const left = float(1).sub(fresnel(dot(d0, n0), ior)).mul(smoothstep(0.05, 0.4, dot(d0, n0).negate())).toVar();
    // The colour of the glass: the setting for all of it times the object's own,
    // faded toward clear by the setting's alpha.
    const tint = mix(vec3(1), tintAll.rgb.mul(tintOwn), tintAll.a).toVar();
    if (shape === KIND.box)
      If(cmy.greaterThan(0), () => {
        tint.mulAssign(cmyTint(n0, cmy, g.cmyOrder));
      });
    const dir = refract(d0, n0, float(1).div(ior)).toVar();
    const leaving = vec3(0).toVar();
    const escaped = bool(false).toVar();
    If(dot(dir, dir).greaterThan(1e-6), () => {
      Loop({ start: int(0), end: int(8), type: 'int', condition: '<', name: 'ev' }, () => {
        const hit = exit(pos, dir, size);
        pos.assign(hit.pos);
        const nrm = hit.nrm.toVar();
        if (shape === KIND.box)
          If(cmy.greaterThan(0), () => {
            tint.mulAssign(cmyTint(nrm, cmy, g.cmyOrder));
          });
        const through = refract(dir, nrm.negate(), ior).toVar();
        If(dot(through, through).greaterThan(1e-6), () => {
          left.mulAssign(float(1).sub(fresnel(dot(dir, nrm), float(1).div(ior))));
          leaving.assign(through);
          escaped.assign(true);
          Break();
        });
        dir.assign(reflect(dir, nrm.negate())); // cannot get out here: it bounces back inside
      });
    });
    const thrown = vec2(0).toVar(); // how far from its straight path the ray lands
    const light = float(0).toVar(); // and how much light it still carries
    If(escaped, () => {
      const exitAt = center.add(toWorld(pos, quat)).toVar();
      const down = toWorld(leaving, quat).toVar();
      If(down.y.lessThan(-0.02), () => {
        const moved = exitAt.add(down.mul(exitAt.y.negate().div(down.y))).sub(worldPos).toVar();
        thrown.assign(vec2(dot(moved, lightR), dot(moved, lightU)));
        light.assign(left.mul(dot(tint, vec3(1 / 3))));
      });
    });
    // Rays that the glass fans out land far apart, and one ray on its own would show
    // as a bright dot. How far apart is the area a little square of rays covers when
    // it lands, against the area it started with: found from how differently this
    // ray's neighbours are thrown. A fanned-out ray is then spread over that larger
    // area (up to a limit), and dimmed by as much; where neighbours are thrown
    // completely differently (across the edge of a facet) there is next to nothing
    // left. (Down the picture is up the screen in WebGL, so its dFdy is turned round.)
    const across = dFdx(thrown).mul(0.5 * raysAcross).toVar();
    const downward = dFdy(thrown).mul((builder.isFlipY?.() ? -0.5 : 0.5) * raysAcross).toVar();
    const spread = max(abs(across.x.add(1).mul(float(1).sub(downward.y)).add(downward.x.mul(across.y))), 1).toVar();
    // Alpha: above 0 marks the pixel as covered by glass (whether or not its ray
    // reaches the floor); beyond 1, it is how wide (in rays) to draw this one.
    return vec4(thrown, light.div(spread), min(sqrt(spread), 6).add(1));
  })();
}

/**
 * The caustic map: objects drawn from the main lamp's direction, every surface the
 * light crosses multiplying it (with multiplicative blending, which builds up the
 * product over all surfaces). The upper part of an object shades the floor; the
 * part resting on it gathers light there instead, the bright patch at the foot of a
 * cut stone.
 */
export function causticNode(normalW: N, worldPos: N, g: GlassInputs): N {
  const { quat, size, vary, tintOwn, tintAll, cmyAll } = g;
  return Fn(() => {
    // A CMY cube: every box when the whole pile is, or just the objects marked as one.
    const cmy = max(cmyAll, vary.z).toVar();
    const n = normalize(normalW).toVar();
    const height = worldPos.y.div(max(size.x, max(size.y, size.z)).mul(0.9));
    const c = vec3(mix(1.32, 0.79, smoothstep(0, 1, height))).toVar();
    // Coloured glass colours the light that passes through it. Light crosses two
    // surfaces of an object (in and out), so each takes half of the colouring.
    c.mulAssign(sqrt(mix(vec3(1), tintAll.rgb.mul(tintOwn), tintAll.a)));
    // CMY cubes also colour the light, on every face it crosses.
    If(cmy.greaterThan(0), () => {
        const u = quat.xyz.negate();
        const local = n.add(cross(u, cross(u, n).add(n.mul(quat.w))).mul(2));
        c.mulAssign(cmyTint(local, cmy, g.cmyOrder));
    });
    return vec4(c, 1);
  })();
}
