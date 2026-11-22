// The ground and sky pass, in TSL: the background the glass is drawn over, with
// the floor's grid, its reflections, the spotlight's beam and fog.
/* eslint-disable @typescript-eslint/no-explicit-any -- TSL node types are too loose to spell out */
import * as TSL from 'three/tsl';
import type { ScreenCamera } from './finish';
import { sceneFunctions, type SceneInputs } from './scene';

type N = any;
const { Fn, If, abs, bool, clamp, dot, exp, float, fract, max, min, mix, normalize, select, sqrt, vec2, vec3, vec4 } =
  TSL as N;

/**
 * How far a ray travels inside the cone of the spotlight (from the lamp, spot.y
 * away toward `lamp`, to the middle of the floor, where the beam is spot.x in
 * radius), up to distance tEnd.
 */
function beamFn(spot: N, lamp: N) {
  return Fn(([o, d, tEnd]: N[]) => {
    const axis = lamp.negate();
    const k = spot.x.div(spot.y);
    const cos2 = float(1).div(k.mul(k).add(1)).toVar(); // squared cosine of the half-angle of the cone
    const fromLamp = o.sub(lamp.mul(spot.y)).toVar();
    const dAxis = dot(d, axis).toVar();
    const oAxis = dot(fromLamp, axis).toVar();
    const a = dAxis.mul(dAxis).sub(cos2).toVar();
    const b = dAxis.mul(oAxis).sub(dot(d, fromLamp).mul(cos2)).mul(2).toVar();
    const c = oAxis.mul(oAxis).sub(dot(fromLamp, fromLamp).mul(cos2)).toVar();
    const disc = b.mul(b).sub(a.mul(c).mul(4)).toVar();
    // Where along the ray it may count: in front of the camera, short of tEnd, and
    // on the lamp's side of the cone's tip that the light goes out from (the cone's
    // equation has a mirror half behind the lamp too).
    const lo = float(0).toVar();
    const hi = float(tEnd).toVar();
    const behind = bool(false).toVar();
    If(abs(dAxis).lessThan(1e-6), () => {
      If(oAxis.lessThanEqual(0), () => {
        behind.assign(true);
      });
    })
      .ElseIf(dAxis.greaterThan(0), () => {
        lo.assign(max(lo, oAxis.negate().div(dAxis)));
      })
      .Else(() => {
        hi.assign(min(hi, oAxis.negate().div(dAxis)));
      });
    const out = float(0).toVar();
    If(behind.not().and(hi.greaterThan(lo)), () => {
      // Inside the (double) cone where a t^2 + b t + c >= 0. A ray running more
      // across the beam than the cone opens (a < 0) is inside between the two
      // crossings; one running more along it (a > 0, as when looking down the beam
      // from inside it) is inside before the first crossing and after the second,
      // or all the way if it never crosses.
      If(abs(a).lessThan(1e-7), () => {
        // Parallel to the cone's side: inside on one side of a single crossing.
        const t = c.negate().div(b);
        If(abs(b).lessThan(1e-9), () => {
          out.assign(select(c.greaterThanEqual(0), hi.sub(lo), 0));
        })
          .ElseIf(b.greaterThan(0), () => {
            out.assign(max(hi.sub(max(lo, t)), 0));
          })
          .Else(() => {
            out.assign(max(min(hi, t).sub(lo), 0));
          });
      })
        .ElseIf(disc.lessThanEqual(0), () => {
          out.assign(select(a.greaterThan(0), hi.sub(lo), 0));
        })
        .Else(() => {
          const root = sqrt(disc);
          const q0 = b.negate().sub(root).div(a.mul(2)).toVar();
          const q1 = b.negate().add(root).div(a.mul(2)).toVar();
          const r0 = min(q0, q1).toVar();
          const r1 = max(q0, q1).toVar();
          If(a.lessThan(0), () => {
            out.assign(max(min(hi, r1).sub(max(lo, r0)), 0));
          }).Else(() => {
            out.assign(max(min(hi, r0).sub(lo), 0).add(max(hi.sub(max(lo, r1)), 0)));
          });
        });
    });
    return out;
  }).setLayout({
    name: 'icBeam',
    type: 'float',
    inputs: [
      { name: 'o', type: 'vec3' },
      { name: 'd', type: 'vec3' },
      { name: 'tEnd', type: 'float' },
    ],
  });
}

/**
 * How much of a pixel a grid line covers: the line is `half` wide either side of
 * its middle, the pixel `pixel` wide, and `d` is how far the pixel is from the line.
 */
function gridLine(d: N, half: number, pixel: N): N {
  return clamp(float(half).sub(d).div(pixel).add(0.5), 0, 1).mul(clamp(float(2 * half).div(pixel), 0, 1));
}

/** The ground and sky, with each pixel's distance from the camera in alpha (1000: the backdrop). */
export function groundNode(cam: ScreenCamera, s: SceneInputs, mirror: N): N {
  const { fragCoord, res, camPos, camRight, camUp, camFwd } = cam;
  const { floorP, spot } = s;
  const { dark, groundOn, env, scene } = sceneFunctions(s, true);
  const beam = beamFn(spot, s.lamp);
  return Fn(() => {
    const ndc = fragCoord
      .div(res)
      .add(cam.jitterUv ?? vec2(0))
      .mul(2)
      .sub(1)
      .mul(vec2(1, -1))
      .toVar();
    const rd = normalize(
      camFwd.xyz.add(camRight.xyz.mul(ndc.x.mul(camFwd.w).mul(camRight.w))).add(camUp.mul(ndc.y.mul(camFwd.w))),
    ).toVar();
    const col = scene(camPos, rd, float(1), float(0)).toVar();
    const hasEnvMap = s.envP ? s.envP.x.greaterThan(0.5) : float(0).greaterThan(0.5);
    // The studio backdrop is lit a little unevenly: brighter up and to the left.
    If(hasEnvMap.not(), () => {
      col.mulAssign(ndc.y.mul(0.8).sub(ndc.x.mul(0.5)).mul(0.07).add(1));
    });
    const floorAlpha = s.floorColor ? clamp(s.floorColor.a, 0, 1) : float(1);
    const onFloor = groundOn.greaterThan(0.5).and(rd.y.lessThan(-1e-5)).toVar();
    const floorT = camPos.y.negate().div(min(rd.y, -1e-5)).toVar();
    // An endless grid on the floor: a line every metre, a heavier one every five metres.
    // Controlled independently of floor color/alpha by gridColor (rgb + a).
    If(onFloor.and(floorP.z.greaterThan(0.5)), () => {
      const g = camPos.xz.add(rd.xz.mul(floorT)).toVar();
      // How much floor one pixel covers there (it stretches away from the camera).
      const pixel = floorT.mul(camFwd.w.mul(2).div(res.y)).div(max(rd.y.negate(), 0.02)).toVar();
      const near1 = abs(fract(g.sub(0.5)).sub(0.5)).toVar();
      const near5 = abs(fract(g.div(5).sub(0.5)).sub(0.5))
        .mul(5)
        .toVar();
      const fine = gridLine(min(near1.x, near1.y), 0.009, pixel);
      const heavy = gridLine(min(near5.x, near5.y), 0.02, pixel);
      const lines = min(fine.mul(0.38).add(heavy.mul(0.5)), 0.62).toVar();
      if (s.gridColor) {
        col.assign(mix(col, s.gridColor.rgb, clamp(lines.mul(s.gridColor.a), 0, 1)));
      } else {
        // Dark lines on the white floor, pale ones on the dark one.
        If(dark.greaterThan(0.5), () => {
          col.addAssign(vec3(0.035).mul(lines));
        }).Else(() => {
          col.mulAssign(float(1).sub(lines));
        });
      }
    });
    // A polished floor: the picture of the objects mirrored in it (drawn before this
    // pass), stronger at grazing angles and slightly blurred.
    If(floorP.x.greaterThan(0).and(onFloor), () => {
      const uv = fragCoord.div(res).toVar();
      // The mirror picture's alpha falls with height above the floor: a polished (not
      // mirror) floor shows what touches it clearly, and what is higher up fainter
      // and more blurred. Reuse the 4 probe taps inside the 9-tap disc blur to avoid
      // redundant texture fetches and trig on every floor pixel.
      const m0 = mirror.sample(uv).level(0).toVar();
      const px = vec2(mix(9, 1, clamp(m0.a.mul(1.3), 0, 1)))
        .div(res)
        .toVar();
      const ax = px.mul(0.7).toVar();
      const p1 = mirror
        .sample(uv.add(vec2(ax.x, 0)))
        .level(0)
        .toVar();
      const p2 = mirror
        .sample(uv.sub(vec2(ax.x, 0)))
        .level(0)
        .toVar();
      const p3 = mirror
        .sample(uv.add(vec2(0, ax.y)))
        .level(0)
        .toVar();
      const p4 = mirror
        .sample(uv.sub(vec2(0, ax.y)))
        .level(0)
        .toVar();
      const probe = m0.a.add(p1.a).add(p2.a).add(p3.a).add(p4.a);
      const m = m0.toVar();
      If(probe.greaterThan(1e-4), () => {
        const diag = px.mul(Math.SQRT1_2).toVar();
        m.assign(
          m0
            .add(p1)
            .add(p2)
            .add(p3)
            .add(p4)
            .add(mirror.sample(uv.add(diag)).level(0))
            .add(mirror.sample(uv.sub(diag)).level(0))
            .add(mirror.sample(uv.add(vec2(diag.x, diag.y.negate()))).level(0))
            .add(mirror.sample(uv.add(vec2(diag.x.negate(), diag.y))).level(0))
            .div(9),
        );
      });
      const grazing = float(1)
        .sub(clamp(rd.y.negate(), 0, 1))
        .toVar();
      // (A black floor shows its reflections the more clearly: twice as strong in the dark studio.)
      const k = min(floorP.x.mul(select(dark.greaterThan(0.5), 2, 1)).mul(grazing.mul(grazing).mul(0.65).add(0.35)), 1)
        .mul(m.a)
        .mul(floorAlpha);
      If(m.a.greaterThan(1e-4), () => {
        col.assign(mix(col, m.rgb.div(m.a), k));
      });
    });
    // The beam of the spotlight, made visible by the dust in the air.
    If(spot.x.greaterThan(0), () => {
      const inBeam = beam(camPos, rd, select(onFloor, floorT, 1e4));
      col.addAssign(vec3(0.07, 0.07, 0.075).mul(float(1).sub(exp(inBeam.mul(-0.3)))));
    });
    // Fog: in the procedural studio, the floor fades into a pale haze along the horizon.
    // Disabled when an HDRI envMap is active so photographic sky/buildings stay crisp without vertical smearing.
    If(floorP.w.greaterThan(0).and(hasEnvMap.not()), () => {
      const haze = env(normalize(vec3(rd.x, 0.03, rd.z)), float(1), float(0))
        .mul(1.07)
        .toVar();
      If(onFloor, () => {
        col.assign(mix(col, haze, float(1).sub(exp(floorP.w.mul(-0.05).mul(floorT)))));
      }).Else(() => {
        col.assign(mix(col, haze, exp(max(rd.y, 0).mul(-5).div(floorP.w))));
      });
    });
    return vec4(col, select(onFloor.and(floorAlpha.greaterThan(0.001)), floorT, 1000));
  })();
}
