// The studio and the floor, in TSL: what a ray sees when it misses the glass. Used
// by the ground pass (which looks at them directly) and by the glass (whose rays
// see them after passing through).
/* eslint-disable @typescript-eslint/no-explicit-any -- TSL node types are too loose to spell out */
import * as TSL from 'three/tsl';

type N = any;
const {
  Fn,
  If,
  abs,
  asin,
  atan,
  clamp,
  cos,
  dot,
  float,
  floor,
  length,
  max,
  min,
  mix,
  normalize,
  select,
  sin,
  smoothstep,
  vec2,
  vec3,
} = TSL as N;

/** The uniforms and textures that describe the room, shared by the passes that see it. */
export interface SceneInputs {
  /** 1 if there is a floor, plus 2 for the dark studio. */
  room: N;
  /** Irradiance / pi on the ground from the room all round, and from the main lamp. */
  skyE: N;
  sunE: N;
  albedo: N;
  /** The caustic map: its centre on the ground, and its right and up axes / half-width. */
  lightC: N;
  lightR: N;
  lightU: N;
  /** Floor: how reflective, how dark its shadows (negative: real caustics), grid on, fog. */
  floorP: N;
  /** Floor surface colour (linear rgb) and opacity (a: 0 = transparent, 1 = solid). */
  floorColor?: N;
  /** Floor grid line colour (linear rgb) and opacity (a). */
  gridColor?: N;
  /** Stage spotlight: x the radius of its beam at the floor (0: off), y how far away the lamp is. */
  spot: N;
  /** Toward the main lamp (which is also the spotlight), from the middle of the floor. */
  lamp: N;
  /** The caustic map, and its size in texels. */
  lightTex: N;
  lightTexSize: number;
  /** The caustic map blurred a little and far wider (only for a pass that sees the floor directly). */
  shadeTex?: N;
  shadeFarTex?: N;
  /** Frosted glass roughness (0: crystal clear, 1: fully frosted). */
  roughness?: N;
  /** Optional equirectangular HDRI environment texture node. */
  envTex?: N;
  /** vec2(hasEnvMap, envMapIntensity). */
  envP?: N;
}

/**
 * The studio's functions.
 * @param backdrop looking straight at the background (the ground pass), not
 *   through or off glass: the floor's soft shadows are only drawn then, and the
 *   room is just the plain backdrop the camera sees
 */
export function sceneFunctions(s: SceneInputs, backdrop: boolean) {
  const dark = floor(s.room.div(2));
  const groundOn = s.room.sub(dark.mul(2));
  const spotOn = s.spot.x.greaterThan(0);

  /**
   * A photographer's white studio: a pale room with a few bright lamps, and dark
   * cards for the glass to pick up (they give colourless stones their contrast).
   * `lampLit`: how much of the lamp reaches the object being shaded (1: all; see
   * lampVisible in glass.ts). In the shadow of others, it still sees the lamp in its
   * reflections at about a third (the ones in the way are glass).
   */
  const env = Fn(([d, lampLit, cone]: N[]) => {
    const sampleHdri = (lod: N) => {
      const u = atan(d.z, d.x)
        .mul(1 / (2 * Math.PI))
        .add(0.5);
      const v = asin(clamp(d.y, -1, 1))
        .mul(1 / Math.PI)
        .add(0.5);
      return s.envTex.sample(vec2(u, v)).level(lod).rgb.mul(s.envP.y);
    };
    // As bright as the open ground, so the two meet without a visible horizon.
    // (Under a spotlight the backdrop is outside the pool, so the lamp does not reach it.)
    if (backdrop) {
      const studioBackdrop = s.albedo.mul(s.skyE.add(s.sunE.mul(select(spotOn, 0, 1))));
      if (s.envTex && s.envP) {
        const out = studioBackdrop.toVar();
        If(s.envP.x.greaterThan(0.5), () => {
          out.assign(sampleHdri(float(0)));
        });
        return out;
      }
      return studioBackdrop;
    }
    const lampShare = mix(0.35, 1, lampLit);
    const rough = (s.roughness ?? float(0)).toVar();
    // Analytic angular pre-filtering: combine surface roughness with the ray-cone footprint.
    // Peak lamp gains scale down as lobes widen (conserving energy), while frosted ambient lift
    // applies only to actual surface roughness so clear glass never turns milky.
    const r = max(rough, clamp(cone, 0, 1)).toVar();
    const out = vec3(0).toVar();
    const evalStudio = () => {
      const r2 = r.mul(r).toVar();
      const sharp = float(1).sub(r.mul(0.85)).toVar();
      const az = atan(d.z, d.x).toVar();
      If(dark.greaterThan(0.5), () => {
        // The dark studio: black all round. With the spotlight it is lit by that one
        // lamp and nothing else. Without, by a large soft lamp, a soft one overhead,
        // and two tall strip lights that draw bright lines down the edges of the glass.
        const k = float(0.012).toVar();
        If(spotOn, () => {
          // (The room is not quite black: a dim grey, lighter toward the floor, which is
          // what the dark band low in a glass ball is showing.)
          k.assign(mix(0.11, 0.035, smoothstep(0, 0.6, d.y)));
          const spotGlow = smoothstep(float(0.988).sub(r.mul(0.55)), float(0.996).add(r.mul(0.004)), dot(d, s.lamp));
          k.addAssign(lampShare.mul(40).div(r2.mul(28).add(1)).mul(spotGlow));
          // A jeweller's softboxes, out of the picture: four panels overhead round a
          // dark hole (where the camera looks down through the tent), and a tall strip
          // either side. Only the glass sees them: they make a stone's facets flash.
          const ring = smoothstep(float(0.78).sub(r.mul(0.35)), float(0.86).add(r.mul(0.1)), d.y).mul(
            float(1).sub(smoothstep(0.94, 0.97, d.y).mul(sharp)),
          );
          k.addAssign(
            ring
              .mul(2.6)
              .div(r2.mul(3).add(1))
              .mul(mix(smoothstep(-0.2, 0.3, cos(az.mul(4))), 0.5, r)),
          );
          const tall = smoothstep(0.05, 0.15, d.y)
            .mul(float(1).sub(smoothstep(0.55, 0.7, d.y)))
            .toVar();
          // (Left and right of where the camera starts, not behind the stones, where a
          // table facing up would mirror one straight into the camera as a white blank.)
          const stripLo = float(0.95).sub(r.mul(0.55));
          const stripHi = float(0.985).add(r.mul(0.015));
          const stripGain = float(4).div(r2.mul(5).add(1));
          k.addAssign(tall.mul(stripGain).mul(smoothstep(stripLo, stripHi, cos(az.sub(2.5)))));
          k.addAssign(tall.mul(stripGain).mul(smoothstep(stripLo, stripHi, cos(az.add(0.63)))));
          k.addAssign(rough.mul(0.12));
        }).Else(() => {
          const mainGlow = smoothstep(float(0.84).sub(r.mul(0.55)), float(0.95).add(r.mul(0.05)), dot(d, s.lamp));
          k.addAssign(lampShare.mul(6).div(r2.mul(4).add(1)).mul(mainGlow));
          k.addAssign(
            smoothstep(float(0.8).sub(r.mul(0.45)), 0.97, d.y)
              .mul(1.6)
              .div(r2.mul(2).add(1)),
          );
          const tall = smoothstep(0.02, 0.1, d.y)
            .mul(float(1).sub(smoothstep(0.6, 0.75, d.y)))
            .toVar();
          k.addAssign(
            tall
              .mul(float(5).div(r2.mul(6).add(1)))
              .mul(smoothstep(float(0.985).sub(r.mul(0.6)), 0.996, cos(az.sub(0.5)))),
          );
          k.addAssign(
            tall
              .mul(float(3.5).div(r2.mul(6).add(1)))
              .mul(smoothstep(float(0.99).sub(r.mul(0.6)), 0.998, cos(az.add(1.9)))),
          );
          k.addAssign(rough.mul(0.1));
        });
        out.assign(vec3(k));
      }).Else(() => {
        // A light tent: pale overhead, dark grey toward the floor.
        const c = mix(mix(0.15, 0.38, r), mix(0.72, 0.65, r), smoothstep(-0.1, 0.95, d.y)).toVar();
        // A few grey cards standing round the set, and a small dark one overhead (the
        // camera): they are what the stone's darker facets are looking at.
        const band = smoothstep(0.03, 0.1, d.y).mul(float(1).sub(smoothstep(0.68, 0.82, d.y)));
        c.assign(
          mix(
            c,
            0.04,
            band
              .mul(0.96)
              .mul(sharp)
              .mul(smoothstep(-0.1, 0.15, sin(az.mul(5).add(1)))),
          ),
        );
        c.assign(mix(c, 0.05, smoothstep(0.9, 0.95, d.y).mul(sharp)));
        // Lamps: the main one up and to the left, and two small ones.
        const mainLo = float(0.86).sub(r.mul(0.55));
        const mainHi = float(0.96).add(r.mul(0.04));
        c.addAssign(
          lampShare
            .mul(3)
            .div(r2.mul(3).add(1))
            .mul(smoothstep(mainLo, mainHi, dot(d, s.lamp))),
        );
        const smallLo = float(0.985).sub(r.mul(0.55));
        const smallGain = float(3).div(r2.mul(8).add(1));
        c.addAssign(smoothstep(smallLo, 0.995, dot(d, normalize(vec3(Math.cos(3), 0.6, Math.sin(3))))).mul(smallGain));
        c.addAssign(smoothstep(smallLo, 0.995, dot(d, normalize(vec3(Math.cos(5), 0.9, Math.sin(5))))).mul(smallGain));
        c.addAssign(rough.mul(0.08));
        out.assign(vec3(c));
      });
    };
    if (s.envTex && s.envP) {
      If(s.envP.x.greaterThan(0.5), () => {
        out.assign(sampleHdri(r.mul(5)));
      }).Else(evalStudio);
    } else {
      evalStudio();
    }
    return out;
  }).setLayout({
    name: 'icEnv',
    type: 'vec3',
    inputs: [
      { name: 'd', type: 'vec3' },
      { name: 'lampLit', type: 'float' },
      { name: 'cone', type: 'float' },
    ],
  });

  /** Where a ground point falls in the caustic map. */
  const lightUv = (g: N) => {
    const v = g.sub(s.lightC);
    return vec2(dot(v, s.lightR).mul(0.5).add(0.5), float(0.5).sub(dot(v, s.lightU).mul(0.5)));
  };

  /** How much of the main lamp's light reaches a ground point: single hardware-bilinear lookup (used only for refracted rays). */
  const sunT = Fn(([g]: N[]) => {
    const uv = lightUv(g).toVar();
    const out = vec3(1).toVar();
    If(uv.x.greaterThan(0).and(uv.x.lessThan(1)).and(uv.y.greaterThan(0)).and(uv.y.lessThan(1)), () => {
      out.assign(s.lightTex.sample(uv).level(0).rgb);
    });
    return out;
  }).setLayout({ name: 'icSunT', type: 'vec3', inputs: [{ name: 'g', type: 'vec3' }] });

  /**
   * How much of the spotlight reaches a point of the floor: all of it inside the
   * pool, fading out at its edge. (1 everywhere when there is no spotlight.)
   */
  const spotPool = Fn(([g]: N[]) => {
    const out = float(1).toVar();
    If(spotOn, () => {
      // How far the point is from the axis of the beam, against how wide the beam is
      // there. The beam comes down at an angle, so the pool is an oval.
      const fromLamp = g.sub(s.lamp.mul(s.spot.y)).toVar();
      const along = dot(fromLamp, s.lamp.negate()).toVar();
      out.assign(0);
      If(along.greaterThan(0), () => {
        const across = length(fromLamp.add(s.lamp.mul(along)));
        // It fades out gradually toward its edge, and (the lamp being near) the floor
        // is brighter on the side toward the lamp than on the far side.
        const nearness = s.spot.y.div(max(along, s.spot.y.mul(0.3))).toVar();
        out.assign(
          float(1)
            .sub(smoothstep(0.3, 1, across.div(along).mul(s.spot.y).div(s.spot.x)))
            .mul(nearness)
            .mul(nearness),
        );
      });
    });
    return out;
  }).setLayout({ name: 'icSpotPool', type: 'float', inputs: [{ name: 'g', type: 'vec3' }] });

  const floorAlbedo = s.floorColor ? s.floorColor.rgb : s.albedo;
  const floorAlpha = s.floorColor ? clamp(s.floorColor.a, 0, 1) : float(1);

  /** The floor's colour at a ground point. */
  const groundColor = Fn(([g]: N[]) => {
    // The floor's shadows are blurred: the shadow of the main lamp a little, and a
    // much wider, fainter one where objects keep the light of the room off the floor.
    const pool = spotPool(g).toVar();
    // Blurred only where the floor is seen directly, not through glass.
    if (!backdrop) {
      const direct = clamp(sunT(g), vec3(0), vec3(1)).toVar();
      return floorAlbedo.mul(s.skyE.mul(direct.mul(0.45).add(0.55)).add(s.sunE.mul(direct).mul(pool)));
    }
    const uv = lightUv(g).toVar();
    // x a little blurred (the shadow of the lamp), y far wider (the room's light kept off).
    const shade = vec2(s.shadeTex.sample(uv).level(0).g, s.shadeFarTex.sample(uv).level(0).g).toVar();
    // floorP.y is how dark the shadows are; negative when the near map holds real
    // caustics (the light that actually arrives), which are used as they are.
    const darkness = abs(s.floorP.y).toVar();
    const lamp = min(max(shade.x, 0), 1.7).toVar();
    If(s.floorP.y.lessThan(0), () => {
      // Gathered light is eased off above the level of the open floor, toward three
      // times it at most. Only the top layer of a pile is followed, as if nothing lay
      // beneath it, so taken at face value a pile would light its floor far too much.
      const over = max(shade.x.sub(1), 0).toVar();
      lamp.assign(min(shade.x, 1).add(over.div(over.mul(0.5).add(1))));
    });
    // How dark the shadows are: the share of the lamp's light they hold back (0: none,
    // 1: all of it). Light gathered above the open floor's is left as it is.
    If(lamp.lessThan(1), () => {
      lamp.assign(float(1).sub(darkness.mul(float(1).sub(lamp))));
    });
    const ambient = float(1).sub(darkness.mul(0.8).mul(float(1).sub(min(shade.y, 1))));
    return floorAlbedo.mul(s.skyE.mul(ambient).add(s.sunE.mul(lamp).mul(pool)));
  }).setLayout({ name: 'icGroundColor', type: 'vec3', inputs: [{ name: 'g', type: 'vec3' }] });

  const hasEnvMap = s.envP ? s.envP.x.greaterThan(0.5) : float(0).greaterThan(0.5);

  /** Ground and sky along a ray, ignoring glass. */
  const scene = Fn(([ro, rd, lampLit, cone]: N[]) => {
    const out = env(rd, lampLit, cone).toVar();
    If(groundOn.greaterThan(0.5).and(rd.y.lessThan(-1e-5)).and(floorAlpha.greaterThan(0.001)), () => {
      const t = max(ro.y.negate().div(rd.y), 0).toVar();
      const horizon = env(normalize(vec3(rd.x, 0.03, rd.z)), lampLit, cone);
      const floorCol = mix(
        groundColor(ro.add(rd.mul(t))),
        horizon,
        select(hasEnvMap, float(0), smoothstep(150, 600, t)),
      );
      out.assign(mix(out, floorCol, floorAlpha));
    });
    return out;
  }).setLayout({
    name: 'icScene',
    type: 'vec3',
    inputs: [
      { name: 'ro', type: 'vec3' },
      { name: 'rd', type: 'vec3' },
      { name: 'lampLit', type: 'float' },
      { name: 'cone', type: 'float' },
    ],
  });

  return { dark, groundOn, env, sunT, spotPool, groundColor, scene };
}
