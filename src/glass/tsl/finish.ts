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
const { Continue, Fn, If, Loop, abs, clamp, cos, cross, dot, exp, float, fract, length, max, min, mix, normalize, select, sin, smoothstep, sqrt, vec2, vec3, vec4 } = TSL as N;

/**
 * A Gaussian blur of `src` along `dir` (in texture coordinates, out to its full
 * length), `reach` samples each side of the middle. Only what is brighter than
 * `cut` is kept (0 keeps everything).
 */
export function blurNode(uv: N, dir: N, cut: N, reach: N, src: N): N {
  return Fn(() => {
    const sum = vec4(0).toVar();
    const total = float(0).toVar();
    const n = reach.toInt();
    Loop({ start: n.negate(), end: n, type: 'int', condition: '<=' }, ({ i }: { i: N }) => {
      const x = float(i).mul(2).div(reach);
      const w = exp(x.mul(x).mul(-0.5));
      sum.addAssign(src.sample(uv.add(dir.mul(float(i).div(reach)))).level(0).sub(cut).max(0).mul(w));
      total.addAssign(w);
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

/** The point in the world seen at a place on screen, that far from the camera. */
function pointAt(cam: ScreenCamera, uv: N, dist: N): N {
  const ndc = uv.mul(2).sub(1).mul(vec2(1, -1));
  const rd = normalize(
    cam.camFwd.xyz.add(cam.camRight.xyz.mul(ndc.x.mul(cam.camFwd.w).mul(cam.camRight.w))).add(cam.camUp.mul(ndc.y.mul(cam.camFwd.w))),
  );
  return cam.camPos.add(rd.mul(dist));
}

/** A turn (radians) that differs from pixel to pixel, to spread the samples of a spiral. */
function pixelTurn(fragCoord: N): N {
  return fract(fract(dot(fragCoord, vec2(0.06711056, 0.00583715))).mul(52.9829189)).mul(6.2831853);
}

/**
 * Contact shadows. The picture so far carries, in its alpha, how far each pixel is
 * from the camera; from that, each pixel looks at the surfaces around it and
 * darkens where they crowd in over it - under an object where it meets the floor,
 * and in the gaps between objects. p = strength, radius (m), exposure.
 */
export function contactShadeNode(cam: ScreenCamera, p: N, src: N): N {
  const { fragCoord, res, camPos } = cam;
  return Fn(() => {
    const uv = fragCoord.div(res).toVar();
    const here = src.sample(uv).level(0).toVar();
    const dist = here.a;
    const shade = float(1).toVar();
    If(p.x.greaterThan(0).and(dist.lessThan(900)), () => {
      const px = vec2(1).div(res).toVar();
      const at = pointAt(cam, uv, dist).toVar();
      // The surface's direction, from its neighbours (on each axis, the one nearer in depth).
      const right = src.sample(uv.add(vec2(px.x, 0))).level(0).a.toVar();
      const left = src.sample(uv.sub(vec2(px.x, 0))).level(0).a.toVar();
      const below = src.sample(uv.add(vec2(0, px.y))).level(0).a.toVar();
      const above = src.sample(uv.sub(vec2(0, px.y))).level(0).a.toVar();
      const alongX = pointAt(cam, uv.add(vec2(px.x, 0)), right).sub(at).toVar();
      If(abs(left.sub(dist)).lessThan(abs(right.sub(dist))), () => {
        alongX.assign(at.sub(pointAt(cam, uv.sub(vec2(px.x, 0)), left)));
      });
      const alongY = pointAt(cam, uv.add(vec2(0, px.y)), below).sub(at).toVar();
      If(abs(above.sub(dist)).lessThan(abs(below.sub(dist))), () => {
        alongY.assign(at.sub(pointAt(cam, uv.sub(vec2(0, px.y)), above)));
      });
      const n = normalize(cross(alongY, alongX)).toVar();
      If(dot(n, camPos.sub(at)).lessThan(0), () => {
        n.assign(n.negate());
      });
      // The floor takes the full shadow; glass, being clear, only a little.
      const onFloor = abs(at.y).lessThan(dist.mul(0.002).add(0.02)).toVar();
      If(onFloor, () => {
        n.assign(vec3(0, 1, 0));
      });

      // Points round about, in a spiral turned differently at every pixel.
      const reach = p.y.div(dist.mul(2).mul(cam.camFwd.w)).toVar(); // the radius on screen, as a share of its height
      const turn = pixelTurn(fragCoord).toVar();
      const crowded = float(0).toVar();
      Loop(24, ({ i }: { i: N }) => {
        const k = float(i);
        const a = turn.add(k.mul(2.3999632));
        const r = reach.mul(sqrt(k.add(0.5).div(24)));
        const suv = uv.add(vec2(cos(a).mul(r).mul(res.y).div(res.x), sin(a).mul(r))).toVar();
        const sd = src.sample(suv).level(0).a.toVar();
        const point = pointAt(cam, suv, sd).toVar();
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
          max(dot(n, there).div(far.add(1e-4)).sub(0.1), 0).mul(float(1).sub(smoothstep(p.y.mul(0.7), p.y.mul(1.6), far))),
        );
      });
      shade.assign(float(1).sub(p.x.mul(select(onFloor, 1.0, 0.3)).mul(clamp(crowded.div(24).mul(2.6), 0, 1))));
    });
    return vec4(here.rgb.mul(shade).mul(p.z), dist);
  })();
}

/** How soft a point at that distance is: the radius of its blur, as a share of the height of the picture. */
function softness(dist: N, focus: N, aperture: N): N {
  return min(aperture.mul(0.05).mul(abs(dist.sub(focus))).div(max(dist, 0.1)), 0.02);
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
    const newPeak = float(1).sub(float(d * d).div(peak.add(d - start))).toVar();
    const grey = float(1).sub(float(1).div(peak.sub(newPeak).mul(0.15).add(1)));
    c.assign(mix(c.mul(newPeak.div(peak)), vec3(newPeak), grey));
  });
  return c;
}

/**
 * The last pass: the camera. Depth of field (what is nearer or farther than the
 * focus goes soft), the glow round the brightest glints, and film-like tones.
 * p = focus distance to fall back on (m), aperture, glow, tones on.
 */
export function lensNode(fragCoord: N, res: N, p: N, src: N, glow: N): N {
  return Fn(() => {
    const uv = fragCoord.div(res).toVar();
    const here = src.sample(uv).level(0).toVar();
    const col = here.rgb.toVar();
    // Autofocus: on whatever is in the middle of the picture (a few points round
    // the centre, in case one falls in a gap). Only if all of them see the
    // backdrop does it fall back on the distance it was given.
    const focus = float(0).toVar();
    const found = float(0).toVar();
    Loop(5, ({ i }: { i: N }) => {
      const a = float(i).mul(1.2566371);
      const offset = select(i.equal(0), vec2(0), vec2(cos(a), sin(a)).mul(0.03));
      const d = src.sample(vec2(0.5).add(offset.mul(vec2(res.y.div(res.x), 1)))).level(0).a.toVar();
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
      Loop(24, ({ i }: { i: N }) => {
        const k = float(i);
        const a = turn.add(k.mul(2.3999632));
        const r = blur.mul(sqrt(k.add(0.5).div(24))).toVar();
        const there = src.sample(uv.add(vec2(cos(a).mul(r).mul(res.y).div(res.x), sin(a).mul(r)))).level(0).toVar();
        const w = float(1).toVar();
        If(there.a.lessThan(here.a.sub(0.05)), () => {
          w.assign(clamp(softness(there.a, focus, p.y).div(max(r, 1e-5)), 0, 1));
        });
        sum.addAssign(there.rgb.mul(w));
        weight.addAssign(w);
      });
      col.assign(sum.div(weight));
    });
    col.addAssign(glow.sample(uv).level(0).rgb.mul(p.z));
    If(p.w.greaterThan(0.5), () => {
      col.assign(tones(col));
    });
    return vec4(col, 1);
  })();
}
