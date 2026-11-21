// The finishing passes, in TSL (three.js compiles them to WGSL or GLSL, so they
// run on WebGPU and WebGL alike): the blur that softens shadows and makes the
// glow, contact shadows, and the lens (depth of field, glow and tones).
//
// (Values used again after something they read has changed are made variables
// with toVar(): a TSL expression is worked out where it is used, not where it is
// written.)
/* eslint-disable @typescript-eslint/no-explicit-any -- TSL node types are too loose to spell out */
import * as TSL from 'three/tsl';

type N = any;
// (Untyped: three's TSL typings reject much that compiles, such as a vector in select.)
const {
  Continue,
  Fn,
  If,
  Loop,
  abs,
  clamp,
  cos,
  cross,
  dot,
  exp,
  float,
  fract,
  int,
  length,
  max,
  min,
  mix,
  normalize,
  select,
  sin,
  smoothstep,
  sqrt,
  vec2,
  vec3,
  vec4,
} = TSL as N;

/**
 * Quarter-resolution bloom prefilter: averages a 4×4 block of full-resolution pixels
 * via 4 hardware-bilinear taps at `(±1, ±1) * texelStep`, extracts HDR excess above
 * `cut`, and applies a hue-preserving soft firefly shoulder (`e * cap / (cap + peak)`)
 * so extreme HDRI sun glints (~1000+) produce a smooth halo instead of vertical combing.
 */
export function bloomPrefilterNode(uv: N, texelStep: N, cut: N, cap: N, src: N): N {
  const tap = (dx: number, dy: number) => {
    const raw = src.sample(uv.add(texelStep.mul(vec2(dx, dy)))).level(0).rgb;
    const excess = raw.sub(cut).max(0);
    const peak = max(excess.r, max(excess.g, excess.b));
    // Soft Reinhard-style shoulder: preserves chromatic dispersion hue while bounding HDR peaks.
    return excess.mul(cap.div(cap.add(peak)));
  };
  return Fn(() => {
    const sum = tap(-1, -1).add(tap(1, -1)).add(tap(-1, 1)).add(tap(1, 1));
    return vec4(sum.mul(0.25), 1);
  })();
}

/**
 * A Gaussian blur of `src` along `dir` (in texture coordinates, out to its full
 * length), `reach` bilinear-paired samples each side of the middle (covering
 * 2*reach steps with ~half the texture fetches via hardware linear filtering).
 * Only what is brighter than `cut` is kept (0 keeps everything).
 */
export function blurNode(uv: N, dir: N, cut: N, reach: N, src: N): N {
  return Fn(() => {
    const sum = src.sample(uv).level(0).sub(cut).max(0).toVar();
    const total = float(1).toVar();
    const n = reach.toInt();
    const maxI = reach.mul(2).toVar();
    Loop({ start: int(1), end: n, type: 'int', condition: '<=', name: 'bi' }, ({ bi }: { bi: N }) => {
      const i1 = float(bi).mul(2).sub(1).toVar();
      const i2 = i1.add(1).toVar();
      const x1 = i1.mul(2).div(maxI);
      const x2 = i2.mul(2).div(maxI);
      const w1 = exp(x1.mul(x1).mul(-0.5)).toVar();
      const w2 = exp(x2.mul(x2).mul(-0.5)).toVar();
      const w = w1.add(w2).toVar();
      const delta = dir.mul(i1.mul(w1).add(i2.mul(w2)).div(w.mul(maxI))).toVar();
      const sPos = src.sample(uv.add(delta)).level(0).sub(cut).max(0);
      const sNeg = src.sample(uv.sub(delta)).level(0).sub(cut).max(0);
      sum.addAssign(sPos.add(sNeg).mul(w));
      total.addAssign(w.mul(2));
    });
    return sum.div(total);
  })();
}

/** The camera, as the finishing passes see it. */
export interface ScreenCamera {
  fragCoord: N;
  res: N;
  camPos: N;
  /** xyz: right; w: aspect ratio. */
  camRight: N;
  camUp: N;
  /** xyz: forward; w: tan(half the vertical field of view). */
  camFwd: N;
}

/** A turn (radians) that differs from pixel to pixel, to spread the samples of a spiral. */
function pixelTurn(fragCoord: N): N {
  return fract(fract(dot(fragCoord, vec2(0.06711056, 0.00583715))).mul(52.9829189)).mul(6.2831853);
}

/**
 * Contact shadows (rendered at half resolution for ~4x less texture traffic).
 * Reads full-res depth from `src.a` and writes `(shade, dist, 0, 1)` for
 * depth-aware bilateral upsampling in `lensNode`. p = strength, radius (m), exposure.
 */
export function contactShadeNode(cam: ScreenCamera, p: N, src: N, taps = 8): N {
  const { fragCoord, res, camPos } = cam;
  return Fn(() => {
    const uv = fragCoord.div(res).toVar();
    const dist = src.sample(uv).level(0).a.toVar();
    const shade = float(1).toVar();
    If(p.x.greaterThan(0).and(dist.lessThan(900)), () => {
      // The radius on screen, as a share of its height; skip distant pixels where it is sub-pixel.
      const reach = p.y.div(dist.mul(2).mul(cam.camFwd.w)).toVar();
      If(reach.mul(res.y).greaterThan(0.75), () => {
        const rightScaled = cam.camRight.xyz.mul(cam.camFwd.w.mul(cam.camRight.w)).toVar();
        const upScaled = cam.camUp.mul(cam.camFwd.w).toVar();
        const fastPointAt = (u: N, d: N) => {
          const ndc = u.mul(2).sub(1).mul(vec2(1, -1));
          return camPos.add(normalize(cam.camFwd.xyz.add(rightScaled.mul(ndc.x)).add(upScaled.mul(ndc.y))).mul(d));
        };
        const px = vec2(1).div(res).toVar();
        const at = fastPointAt(uv, dist).toVar();
        // The floor takes the full shadow; glass, being clear, only a little.
        // Check onFloor first so floor pixels skip the 4 neighbor depth fetches & cross product.
        const onFloor = abs(at.y).lessThan(dist.mul(0.002).add(0.02)).toVar();
        const n = vec3(0, 1, 0).toVar();
        If(onFloor.not(), () => {
          // The surface's direction, from its neighbours (on each axis, the one nearer in depth).
          const right = src
            .sample(uv.add(vec2(px.x, 0)))
            .level(0)
            .a.toVar();
          const left = src
            .sample(uv.sub(vec2(px.x, 0)))
            .level(0)
            .a.toVar();
          const below = src
            .sample(uv.add(vec2(0, px.y)))
            .level(0)
            .a.toVar();
          const above = src
            .sample(uv.sub(vec2(0, px.y)))
            .level(0)
            .a.toVar();
          const alongX = fastPointAt(uv.add(vec2(px.x, 0)), right)
            .sub(at)
            .toVar();
          If(abs(left.sub(dist)).lessThan(abs(right.sub(dist))), () => {
            alongX.assign(at.sub(fastPointAt(uv.sub(vec2(px.x, 0)), left)));
          });
          const alongY = fastPointAt(uv.add(vec2(0, px.y)), below)
            .sub(at)
            .toVar();
          If(abs(above.sub(dist)).lessThan(abs(below.sub(dist))), () => {
            alongY.assign(at.sub(fastPointAt(uv.sub(vec2(0, px.y)), above)));
          });
          n.assign(normalize(cross(alongY, alongX)));
          If(dot(n, camPos.sub(at)).lessThan(0), () => {
            n.assign(n.negate());
          });
        });

        // Points round about, in a spiral turned differently at every pixel.
        const turn = pixelTurn(fragCoord).toVar();
        const invAspect = res.y.div(res.x).toVar();
        const crowded = float(0).toVar();
        Loop(taps, ({ i }: { i: N }) => {
          const k = float(i);
          const a = turn.add(k.mul(2.3999632));
          const r = reach.mul(sqrt(k.add(0.5).div(taps)));
          const suv = uv.add(vec2(cos(a).mul(r).mul(invAspect), sin(a).mul(r))).toVar();
          const sd = src.sample(suv).level(0).a.toVar();
          const point = fastPointAt(suv, sd).toVar();
          // The floor never shades the floor (its depth is stored only roughly, so
          // nearby floor would read as tiny bumps, and leave a fine grain over it).
          If(onFloor.and(abs(point.y).lessThan(sd.mul(0.002).add(0.02))), () => {
            Continue();
          });
          // On the floor only what is close above it counts: the spot where an
          // object meets it (its shadow proper comes from the lamp).
          If(onFloor.and(point.y.greaterThan(p.y.mul(0.1))), () => {
            Continue();
          });
          const there = point.sub(at).toVar();
          const far = length(there);
          crowded.addAssign(
            max(dot(n, there).div(far.add(1e-4)).sub(0.1), 0).mul(
              float(1).sub(smoothstep(p.y.mul(0.7), p.y.mul(1.6), far)),
            ),
          );
        });
        shade.assign(float(1).sub(p.x.mul(select(onFloor, 1.0, 0.3)).mul(clamp(crowded.div(taps).mul(2.6), 0, 1))));
      });
    });
    return vec4(shade, dist, 0, 1);
  })();
}

/** How soft a point at that distance is: the radius of its blur, as a share of the height of the picture. */
function softness(dist: N, focus: N, aperture: N): N {
  return min(
    aperture
      .mul(0.05)
      .mul(abs(dist.sub(focus)))
      .div(max(dist, 0.1)),
    0.02,
  );
}

/**
 * Film-like tones: left alone up to fairly bright, then eased toward white
 * instead of clipping, losing a little colour on the way (the Khronos "neutral"
 * curve).
 */
export function tones(cIn: N): N {
  const c = vec3(cIn).toVar();
  const low = min(c.r, min(c.g, c.b)).toVar();
  c.subAssign(select(low.lessThan(0.08), low.sub(low.mul(low).mul(6.25)), 0.04));
  const peak = max(c.r, max(c.g, c.b)).toVar();
  const start = 0.76;
  const d = 1 - start;
  If(peak.greaterThanEqual(start), () => {
    const newPeak = float(1)
      .sub(float(d * d).div(peak.add(d - start)))
      .toVar();
    const grey = float(1).sub(float(1).div(peak.sub(newPeak).mul(0.15).add(1)));
    c.assign(mix(c.mul(newPeak.div(peak)), vec3(newPeak), grey));
  });
  return c;
}

/**
 * The last pass: the camera. Joint-bilateral upsampled contact shadows, exposure,
 * depth of field (what is nearer or farther than the focus goes soft), the glow
 * round the brightest glints, and film-like tones.
 * p = focus distance to fall back on (m), aperture, glow, tones on.
 * finish = exposure, contactShadowsActive (0/1), half-res texel step x, y.
 */
export function lensNode(fragCoord: N, res: N, p: N, finish: N, src: N, glow: N, ao: N): N {
  return Fn(() => {
    const uv = fragCoord.div(res).toVar();
    const here = src.sample(uv).level(0).toVar();
    const col = here.rgb.toVar();
    // Autofocus: on whatever is in the middle of the picture (a few points round
    // the centre, in case one falls in a gap). Only if all of them see the
    // backdrop does it fall back on the distance it was given.
    // Skip autofocus & DoF completely when aperture is 0.
    If(p.y.greaterThan(0.001), () => {
      const focus = float(0).toVar();
      const found = float(0).toVar();
      Loop(5, ({ i }: { i: N }) => {
        const a = float(i).mul(1.2566371);
        const offset = select(i.equal(0), vec2(0), vec2(cos(a), sin(a)).mul(0.03));
        const d = src
          .sample(vec2(0.5).add(offset.mul(vec2(res.y.div(res.x), 1))))
          .level(0)
          .a.toVar();
        If(d.lessThan(900), () => {
          focus.addAssign(d);
          found.addAssign(1);
        });
      });
      focus.assign(select(found.greaterThan(0), focus.div(max(found, 1)), p.x));
      const blur = softness(here.a, focus, p.y).toVar();
      If(blur.mul(res.y).greaterThan(0.75), () => {
        // Gather from a disc as wide as this pixel is soft. A point in front of this
        // one only spills over it as far as it is soft itself.
        const turn = pixelTurn(fragCoord).toVar();
        const sum = here.rgb.toVar();
        const weight = float(1).toVar();
        Loop(16, ({ i }: { i: N }) => {
          const k = float(i);
          const a = turn.add(k.mul(2.3999632));
          const r = blur.mul(sqrt(k.add(0.5).div(16))).toVar();
          const there = src
            .sample(uv.add(vec2(cos(a).mul(r).mul(res.y).div(res.x), sin(a).mul(r))))
            .level(0)
            .toVar();
          const w = float(1).toVar();
          If(there.a.lessThan(here.a.sub(0.05)), () => {
            w.assign(clamp(softness(there.a, focus, p.y).div(max(r, 1e-5)), 0, 1));
          });
          sum.addAssign(there.rgb.mul(w));
          weight.addAssign(w);
        });
        col.assign(sum.div(weight));
      });
    });
    // Joint-bilateral upsample of half-res contact shadows (prevents silhouette bleeding).
    If(finish.y.greaterThan(0.5).and(here.a.lessThan(900)), () => {
      const h = finish.zw.mul(0.5).toVar();
      const s0 = ao
        .sample(uv.add(vec2(h.x.negate(), h.y.negate())))
        .level(0)
        .toVar();
      const s1 = ao
        .sample(uv.add(vec2(h.x, h.y.negate())))
        .level(0)
        .toVar();
      const s2 = ao
        .sample(uv.add(vec2(h.x.negate(), h.y)))
        .level(0)
        .toVar();
      const s3 = ao
        .sample(uv.add(vec2(h.x, h.y)))
        .level(0)
        .toVar();
      const w0 = float(1).div(abs(s0.g.sub(here.a)).mul(6).add(1));
      const w1 = float(1).div(abs(s1.g.sub(here.a)).mul(6).add(1));
      const w2 = float(1).div(abs(s2.g.sub(here.a)).mul(6).add(1));
      const w3 = float(1).div(abs(s3.g.sub(here.a)).mul(6).add(1));
      const shade = s0.r.mul(w0).add(s1.r.mul(w1)).add(s2.r.mul(w2)).add(s3.r.mul(w3)).div(w0.add(w1).add(w2).add(w3));
      col.mulAssign(shade);
    });
    col.mulAssign(finish.x);
    If(p.z.greaterThan(0.001), () => {
      col.addAssign(glow.sample(uv).level(0).rgb.mul(p.z).mul(finish.x));
    });
    If(p.w.greaterThan(0.5), () => {
      col.assign(tones(col));
    });
    return vec4(col, 1);
  })();
}
