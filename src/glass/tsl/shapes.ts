// Analytic and baked-SDF shape geometry in TSL: snapping a rasterized entry point
// onto the exact surface (`entry`) and finding where an internal ray exits (`exit`).
//
// Shape table (see objects.ts):
//   0 box         half-extents size.xyz (bevelled into 18 planes when bevel > 0)
//                 (the octahedron, prism and hex prism carry a bevel plane per edge too)
//   1 sphere      radius size.x
//   2 cylinder    radius size.x, half-height size.y, axis Y
//   3 capsule     radius size.x, half-length of middle section size.y, axis Y
//   4 cone        base radius size.x, half-height size.y, apex up
//   5 torus       ring radius size.x, tube radius size.y, axis Y
//   6 octahedron  vertices at distance size.x
//   7 prism       triangular: apex at y = size.x, base at y = -size.x / 2, half-length size.z
//   8 hex prism   distance from centre to a side face size.x, half-height size.y
//   9 suzanne     baked 128^3 SDF volume (layer 0); half-width size.x
//  10 diamond     convex polyhedron facets (diamond.ts) of size size.x
//  11 dragon      baked 128^3 SDF volume (layer 1); half-width size.x
//  12 torusknot   baked 128^3 SDF volume (layer 2); half-width size.x (see torusKnot.ts)
/* eslint-disable @typescript-eslint/no-explicit-any -- TSL node types are too loose to spell out */
import * as TSL from 'three/tsl';
import { DIAMOND_PLANES } from '../diamond';
import { KIND } from '../objects';
import { createSdfVolume, SDF_LAYER } from './sdf';

type N = any;
const {
  Break,
  If,
  Loop,
  abs,
  array,
  clamp,
  dot,
  float,
  int,
  length,
  max,
  min,
  normalize,
  select,
  sign,
  sqrt,
  vec2,
  vec3,
  vec4,
} = TSL as N;

/** Shapes whose internal volume is stored as a baked 128^3 signed-distance field. */
export const isBakedMesh = (shape: number): boolean => SDF_LAYER[shape] !== undefined;

/** Shapes with non-zero surface curvature (used for physical ray-cone angular filtering). */
export const isCurvedShape = (shape: number): boolean =>
  shape === KIND.sphere ||
  shape === KIND.cylinder ||
  shape === KIND.capsule ||
  shape === KIND.cone ||
  shape === KIND.torus ||
  isBakedMesh(shape);

const signBit = (k: number) => (k & 1 ? 1 : -1);
const boxExitAlongAxis = (d: N, half: N, p: N) =>
  select(abs(d).greaterThan(1e-7), sign(d).mul(half).sub(p).div(d), 1e9);

/**
 * Bevel planes for the edges between pairs of faces of a convex polyhedron: for faces
 * (n_i, d_i) and (n_j, d_j), the plane along `n_i + n_j` that cuts `bevel` off the edge
 * (as for the box: offset `(d_i + d_j - bevel) / |n_i + n_j|`). Real glass is never cut to a
 * mathematically sharp edge, and these narrow facets are what catch the light along it.
 */
function bevelPlanes(faces: { n: [number, number, number]; d: N }[], edges: [number, number][], bevel: N): N[] {
  return edges.map(([i, j]) => {
    const [a, b] = [faces[i], faces[j]];
    const sum = [a.n[0] + b.n[0], a.n[1] + b.n[1], a.n[2] + b.n[2]];
    const len = Math.hypot(sum[0], sum[1], sum[2]);
    return vec4(sum[0] / len, sum[1] / len, sum[2] / len, a.d.add(b.d).sub(bevel).div(len));
  });
}

/**
 * Outward face planes (xyz = unit normal, w = offset) for analytic polyhedra, plus a bevel
 * plane for every edge, or null for diamonds whose facet planes are stored in a constant
 * TSL array.
 */
function polyhedronPlanes(shape: number, size: N, bevel: N): N[] | null {
  if (shape === KIND.box) {
    const faces = [0, 1, 2, 3, 4, 5].map((k) => {
      const axis = k >> 1;
      const n = [0, 0, 0];
      n[axis] = signBit(k);
      return vec4(n[0], n[1], n[2], [size.x, size.y, size.z][axis]);
    });
    const invSqrt2 = Math.SQRT1_2;
    const bevelEdges = Array.from({ length: 12 }, (_, e) => {
      const a = signBit(e);
      const b = e & 2 ? 1 : -1;
      const pair = e >> 2;
      if (pair === 0) return vec4(a * invSqrt2, b * invSqrt2, 0, size.x.add(size.y).sub(bevel).mul(invSqrt2));
      if (pair === 1) return vec4(0, a * invSqrt2, b * invSqrt2, size.y.add(size.z).sub(bevel).mul(invSqrt2));
      return vec4(a * invSqrt2, 0, b * invSqrt2, size.x.add(size.z).sub(bevel).mul(invSqrt2));
    });
    return [...faces, ...bevelEdges];
  }
  // A face as its unit normal and offset, and the planes for a list of them plus bevels.
  type Face = { n: [number, number, number]; d: N };
  const withBevels = (faces: Face[], edges: [number, number][]) => [
    ...faces.map((f) => vec4(f.n[0], f.n[1], f.n[2], f.d)),
    ...bevelPlanes(faces, edges, bevel),
  ];
  if (shape === KIND.octahedron) {
    const invSqrt3 = 1 / Math.sqrt(3);
    const faces: Face[] = Array.from({ length: 8 }, (_, k) => ({
      n: [signBit(k) * invSqrt3, (k & 2 ? 1 : -1) * invSqrt3, (k & 4 ? 1 : -1) * invSqrt3],
      d: size.x.mul(invSqrt3),
    }));
    // Two faces share an edge where their normals differ in exactly one sign (12 edges).
    const edges: [number, number][] = [];
    for (let i = 0; i < 8; i++) for (const bit of [1, 2, 4]) if (!(i & bit)) edges.push([i, i | bit]);
    return withBevels(faces, edges);
  }
  if (shape === KIND.prism) {
    const half = size.x.mul(0.5);
    const faces: Face[] = [
      { n: [0, -1, 0], d: half },
      { n: [0.8660254, 0.5, 0], d: half },
      { n: [-0.8660254, 0.5, 0], d: half },
      { n: [0, 0, 1], d: size.z },
      { n: [0, 0, -1], d: size.z },
    ];
    // The three long edges, and each of the three sides against both ends (9 edges).
    const edges: [number, number][] = [
      [0, 1],
      [0, 2],
      [1, 2],
      ...[0, 1, 2].flatMap((side): [number, number][] => [
        [side, 3],
        [side, 4],
      ]),
    ];
    return withBevels(faces, edges);
  }
  if (shape === KIND.hexprism) {
    const faces: Face[] = Array.from({ length: 6 }, (_, k) => {
      const angle = ((30 + 60 * k) * Math.PI) / 180;
      return { n: [Math.cos(angle), 0, Math.sin(angle)] as [number, number, number], d: size.x };
    });
    faces.push({ n: [0, 1, 0], d: size.y }, { n: [0, -1, 0], d: size.y });
    // Neighbouring sides, and every side against the top and the bottom (18 edges).
    const edges: [number, number][] = [];
    for (let k = 0; k < 6; k++) edges.push([k, (k + 1) % 6], [k, 6], [k, 7]);
    return withBevels(faces, edges);
  }
  return null;
}

export interface ShapeUniforms {
  bevel: N;
  sdfTex?: N;
  meshEntry?: N;
}

/**
 * Build the shape-specific TSL helpers for one `KIND`:
 * - `declare()`: emits the constant facet array (for diamonds) at the top of the shader.
 * - `entry(pos, size, meshNormal)`: snaps a rasterized surface point onto the exact shape.
 * - `exit(pos, dir, size)`: traces an internal ray to its next surface hit and outward normal.
 */
export function shapeFunctions(shape: number, uniforms: ShapeUniforms, diamondPlanes: number[] = DIAMOND_PLANES) {
  const { bevel, sdfTex } = uniforms;
  const diamondPlaneCount = diamondPlanes.length / 4;
  let diamondFacetArray: N = null;

  const declare = () => {
    if (shape !== KIND.diamond) return;
    const rows = Array.from({ length: diamondPlaneCount }, (_, k) => vec4(...diamondPlanes.slice(k * 4, k * 4 + 4)));
    diamondFacetArray = array(rows).toConst('icDiamond');
  };

  const forEachPlane = (size: N, body: (plane: N) => void) => {
    const planes = polyhedronPlanes(shape, size, bevel);
    if (planes) {
      for (const plane of planes) body(plane);
    } else {
      Loop(
        { start: int(0), end: int(diamondPlaneCount), type: 'int', condition: '<', name: 'kp' },
        ({ kp }: { kp: N }) => {
          const facet = diamondFacetArray.element(kp).toVar();
          body(vec4(facet.xyz, facet.w.mul(size.x)));
        },
      );
    }
  };

  /* Baked 128^3 SDF volume sampling (Suzanne, dragon and torus knot layers; see sdf.ts) */
  const {
    volumeFast,
    volumeFastNormal,
    volumeSmooth,
    volumeSmoothNormal,
    exit: volumeExit,
  } = createSdfVolume(sdfTex, SDF_LAYER[shape] ?? 0);

  /** Snap a rasterized surface point onto the exact shape and compute its outward normal. */
  const entry = (posIn: N, size: N, meshNormal: N) => {
    const p = vec3(posIn).toVar();
    const n = vec3(0, 1, 0).toVar();

    const snapPolyhedron = () => {
      const maxSignedDist = float(-1e9).toVar();
      forEachPlane(size, (plane) => {
        const pl = plane.toVar();
        const sd = dot(pl.xyz, p).sub(pl.w).toVar();
        If(sd.greaterThan(maxSignedDist), () => {
          maxSignedDist.assign(sd);
          n.assign(pl.xyz);
        });
      });
      p.subAssign(n.mul(maxSignedDist));
    };

    if (isBakedMesh(shape)) {
      If(uniforms.meshEntry.greaterThan(0.5), () => {
        // Coarse projection with fast 4-tap normal, refined onto C^2 tricubic isosurface.
        const toward = volumeFastNormal(p, size.x).toVar();
        p.subAssign(toward.mul(volumeFast(p, size.x)));
        p.subAssign(toward.mul(volumeSmooth(p, size.x)));
        n.assign(volumeSmoothNormal(p, size.x));
      }).Else(() => {
        n.assign(meshNormal);
      });
    } else if (shape === KIND.diamond) {
      // diamondGeometry() builds exact flat-shaded facets.
      n.assign(meshNormal);
    } else if (shape === KIND.cone) {
      // Cone surface `x² + z² = slope² * (H - y)²` where `slope = R / (2H)` yields quadratic `a t² + 2b t + c = 0`.
      const slope = size.x.div(size.y.mul(2)).toVar();
      const rho = max(length(p.xz), 1e-6).toVar();
      const invLen = float(1)
        .div(sqrt(slope.mul(slope).add(1)))
        .toVar();
      const baseDist = p.y.negate().sub(size.y).toVar();
      const sideDist = rho
        .sub(slope.mul(size.y.sub(p.y)))
        .mul(invLen)
        .toVar();
      If(baseDist.greaterThanEqual(sideDist), () => {
        n.assign(vec3(0, -1, 0));
        p.assign(vec3(p.x, size.y.negate(), p.z));
      }).Else(() => {
        n.assign(vec3(p.x.div(rho), slope, p.z.div(rho)).mul(invLen));
        p.subAssign(n.mul(sideDist));
      });
    } else if (shape === KIND.torus) {
      const rho = max(length(p.xz), 1e-6).toVar();
      const ringCenter = vec3(p.x.div(rho), 0, p.z.div(rho)).mul(size.x).toVar();
      n.assign(normalize(p.sub(ringCenter)));
      p.assign(ringCenter.add(n.mul(size.y)));
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
      const spine = vec3(0, clamp(p.y, size.y.negate(), size.y), 0).toVar();
      n.assign(normalize(p.sub(spine)));
      p.assign(spine.add(n.mul(size.x)));
    } else if (shape === KIND.box) {
      If(bevel.greaterThan(0), snapPolyhedron).Else(() => {
        const rel = abs(p).div(size).toVar();
        If(rel.x.greaterThanEqual(rel.y).and(rel.x.greaterThanEqual(rel.z)), () => {
          n.assign(vec3(sign(p.x), 0, 0));
          p.assign(vec3(n.x.mul(size.x), p.y, p.z));
        })
          .ElseIf(rel.y.greaterThanEqual(rel.z), () => {
            n.assign(vec3(0, sign(p.y), 0));
            p.assign(vec3(p.x, n.y.mul(size.y), p.z));
          })
          .Else(() => {
            n.assign(vec3(0, 0, sign(p.z)));
            p.assign(vec3(p.x, p.y, n.z.mul(size.z)));
          });
      });
    } else {
      snapPolyhedron(); // octahedron, prism, hex prism
    }
    return { pos: p, nrm: n };
  };

  /**
   * Trace an internal ray from `pos` along `dir` to where it meets the surface again.
   * Returns `{ pos, nrm, t }` (hit position, outward unit normal, and path length).
   */
  const exit = (pos: N, dir: N, size: N) => {
    const t = float(0).toVar();
    const n = vec3(0, 1, 0).toVar();

    const exitPolyhedron = () => {
      // Compare `dist < t * dn` (valid since dn > 0) so we only divide on new minima.
      t.assign(1e9);
      if (shape === KIND.diamond) {
        const unitPos = pos.div(max(size.x, 1e-6)).toVar();
        Loop(
          { start: int(0), end: int(diamondPlaneCount), type: 'int', condition: '<', name: 'kp' },
          ({ kp }: { kp: N }) => {
            const pl = diamondFacetArray.element(kp).toVar();
            const dn = dot(pl.xyz, dir).toVar();
            If(dn.greaterThan(1e-7), () => {
              const dist = pl.w.sub(dot(pl.xyz, unitPos)).toVar();
              If(dist.lessThan(t.mul(dn)), () => {
                t.assign(dist.div(dn));
                n.assign(pl.xyz);
              });
            });
          },
        );
        t.assign(select(t.greaterThan(1e8), 0, max(t.mul(size.x), 0)));
      } else {
        forEachPlane(size, (plane) => {
          const pl = plane.toVar();
          const dn = dot(pl.xyz, dir).toVar();
          If(dn.greaterThan(1e-7), () => {
            const dist = pl.w.sub(dot(pl.xyz, pos)).toVar();
            If(dist.lessThan(t.mul(dn)), () => {
              t.assign(dist.div(dn));
              n.assign(pl.xyz);
            });
          });
        });
        t.assign(select(t.greaterThan(1e8), 0, max(t, 0)));
      }
    };

    if (shape === KIND.sphere) {
      // For `pos` on the sphere `|pos| = R`, `|pos + t * dir|² = R²` simplifies to `t = -2 * (pos · dir)`.
      t.assign(max(dot(pos, dir).mul(-2), 0));
      n.assign(normalize(pos.add(dir.mul(t))));
    } else if (shape === KIND.cylinder || shape === KIND.capsule) {
      // Infinite cylinder intersection in XZ: `a t² + 2b t + c = 0`, taking the forward exit root `(-b + √(b² - ac)) / a`.
      const a = dot(dir.xz, dir.xz).toVar();
      const tSide = float(1e9).toVar();
      If(a.greaterThan(1e-9), () => {
        const b = dot(pos.xz, dir.xz).toVar();
        const c = dot(pos.xz, pos.xz).sub(size.x.mul(size.x));
        tSide.assign(
          b
            .negate()
            .add(sqrt(max(b.mul(b).sub(a.mul(c)), 0)))
            .div(a),
        );
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
      // Cone surface `x² + z² = slope² * (H - y)²` where `slope = R / (2H)` yields quadratic `a t² + 2b t + c = 0`.
      const slope = size.x.div(size.y.mul(2)).toVar();
      const slope2 = slope.mul(slope).toVar();
      const hy = size.y.sub(pos.y).toVar();
      const a = dot(dir.xz, dir.xz).sub(slope2.mul(dir.y).mul(dir.y)).toVar();
      const b = dot(pos.xz, dir.xz).add(slope2.mul(hy).mul(dir.y)).toVar();
      const c = dot(pos.xz, pos.xz).sub(slope2.mul(hy).mul(hy)).toVar();
      const tSide = float(1e9).toVar();
      If(abs(a).greaterThan(1e-7), () => {
        const disc = b.mul(b).sub(a.mul(c)).toVar();
        If(disc.greaterThanEqual(0), () => {
          const root = sqrt(disc).toVar();
          for (const sgn of [-1, 1]) {
            const tr = b.negate().add(root.mul(sgn)).div(a).toVar();
            const y = pos.y.add(dir.y.mul(tr));
            If(
              tr
                .greaterThan(1e-4)
                .and(y.lessThanEqual(size.y))
                .and(y.greaterThanEqual(size.y.negate().sub(1e-4))),
              () => {
                tSide.assign(min(tSide, tr));
              },
            );
          }
        });
      }).ElseIf(abs(b).greaterThan(1e-9), () => {
        const tl = c.negate().div(b.mul(2)).toVar();
        const yl = pos.y.add(dir.y.mul(tl));
        If(
          tl
            .greaterThan(1e-4)
            .and(yl.lessThanEqual(size.y))
            .and(yl.greaterThanEqual(size.y.negate().sub(1e-4))),
          () => {
            tSide.assign(tl);
          },
        );
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
        n.assign(normalize(vec3(q.x.div(rho), slope, q.z.div(rho))));
      });
    } else if (shape === KIND.torus) {
      // Torus ray intersection is quartic, so sphere-trace the exact interior SDF `d = r_tube - |(|q_xz| - R_ring, q_y)|`.
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
    } else if (isBakedMesh(shape)) {
      const hit = volumeExit(pos, dir, size.x);
      t.assign(hit.t);
      n.assign(hit.nrm);
    } else if (shape === KIND.box) {
      If(bevel.greaterThan(0), exitPolyhedron).Else(() => {
        const tx = boxExitAlongAxis(dir.x, size.x, pos.x).toVar();
        const ty = boxExitAlongAxis(dir.y, size.y, pos.y).toVar();
        const tz = boxExitAlongAxis(dir.z, size.z, pos.z).toVar();
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
      exitPolyhedron(); // octahedron, prism, hex prism, diamond
    }
    return { pos: pos.add(dir.mul(t)).toVar(), nrm: n, t };
  };

  return { declare, entry, exit };
}
