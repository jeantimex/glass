// The studio and the floor, in TSL: what a ray sees when it misses the glass. Used
// by the ground pass (which looks at them directly) and by the glass (whose rays
// see them after passing through).
/* eslint-disable @typescript-eslint/no-explicit-any -- TSL node types are too loose to spell out */
import * as TSL from 'three/tsl';

type N = any;
const { Fn, If, abs, atan, clamp, cos, dot, float, floor, length, max, min, mix, normalize, select, sin, smoothstep, vec2, vec3 } = TSL as N;

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
  const env = Fn(([d, lampLit]: N[]) => {
    // As bright as the open ground, so the two meet without a visible horizon.
    // (Under a spotlight the backdrop is outside the pool, so the lamp does not reach it.)
    if (backdrop) return s.albedo.mul(s.skyE.add(s.sunE.mul(select(spotOn, 0, 1))));
    const lampShare = mix(0.35, 1, lampLit);
    const az = atan(d.z, d.x).toVar();
    const out = vec3(0).toVar();
    If(dark.greaterThan(0.5), () => {
      // The dark studio: black all round. With the spotlight it is lit by that one
      // lamp and nothing else. Without, by a large soft lamp, a soft one overhead,
      // and two tall strip lights that draw bright lines down the edges of the glass.
      const k = float(0.012).toVar();
      If(spotOn, () => {
        // (The room is not quite black: a dim grey, lighter toward the floor, which is
        // what the dark band low in a glass ball is showing.)
        k.assign(mix(0.11, 0.035, smoothstep(0, 0.6, d.y)));
        k.addAssign(lampShare.mul(40).mul(smoothstep(0.988, 0.996, dot(d, s.lamp))));
        // A jeweller's softboxes, out of the picture: four panels overhead round a
        // dark hole (where the camera looks down through the tent), and a tall strip
        // either side. Only the glass sees them: they make a stone's facets flash.
        const ring = smoothstep(0.78, 0.86, d.y).mul(float(1).sub(smoothstep(0.94, 0.97, d.y)));
        k.addAssign(ring.mul(2.6).mul(smoothstep(-0.2, 0.3, cos(az.mul(4)))));
        const tall = smoothstep(0.05, 0.15, d.y).mul(float(1).sub(smoothstep(0.55, 0.7, d.y))).toVar();
        // (Left and right of where the camera starts, not behind the stones, where a
        // table facing up would mirror one straight into the camera as a white blank.)
        k.addAssign(tall.mul(4).mul(smoothstep(0.95, 0.985, cos(az.sub(2.5)))));
        k.addAssign(tall.mul(4).mul(smoothstep(0.95, 0.985, cos(az.add(0.63)))));
      }).Else(() => {
        k.addAssign(lampShare.mul(6).mul(smoothstep(0.84, 0.95, dot(d, s.lamp))));
        k.addAssign(smoothstep(0.8, 0.97, d.y).mul(1.6));
        const tall = smoothstep(0.02, 0.1, d.y).mul(float(1).sub(smoothstep(0.6, 0.75, d.y))).toVar();
        k.addAssign(tall.mul(5).mul(smoothstep(0.985, 0.996, cos(az.sub(0.5)))));
        k.addAssign(tall.mul(3.5).mul(smoothstep(0.99, 0.998, cos(az.add(1.9)))));
      });
      out.assign(vec3(k));
    }).Else(() => {
      // A light tent: pale overhead, dark grey toward the floor.
      const c = mix(0.15, 0.72, smoothstep(-0.1, 0.95, d.y)).toVar();
      // A few grey cards standing round the set, and a small dark one overhead (the
      // camera): they are what the stone's darker facets are looking at.
      const band = smoothstep(0.03, 0.1, d.y).mul(float(1).sub(smoothstep(0.68, 0.82, d.y)));
      c.assign(mix(c, 0.04, band.mul(0.96).mul(smoothstep(-0.1, 0.15, sin(az.mul(5).add(1))))));
      c.assign(mix(c, 0.05, smoothstep(0.9, 0.95, d.y)));
      // Lamps: the main one up and to the left, and two small ones.
      c.addAssign(lampShare.mul(3).mul(smoothstep(0.86, 0.96, dot(d, s.lamp))));
      c.addAssign(smoothstep(0.985, 0.995, dot(d, normalize(vec3(Math.cos(3), 0.6, Math.sin(3))))).mul(3));
      c.addAssign(smoothstep(0.985, 0.995, dot(d, normalize(vec3(Math.cos(5), 0.9, Math.sin(5))))).mul(3));
      out.assign(vec3(c));
    });
    return out;
  }).setLayout({ name: 'icEnv', type: 'vec3', inputs: [{ name: 'd', type: 'vec3' }, { name: 'lampLit', type: 'float' }] });

  /** Where a ground point falls in the caustic map. */
  const lightUv = (g: N) => {
    const v = g.sub(s.lightC);
    return vec2(dot(v, s.lightR).mul(0.5).add(0.5), float(0.5).sub(dot(v, s.lightU).mul(0.5)));
  };

  /** How much of the main lamp's light reaches a ground point: the caustic map, lightly blurred. */
  const sunT = Fn(([g]: N[]) => {
    const uv = lightUv(g).toVar();
    const out = vec3(1).toVar();
    If(uv.x.greaterThan(0).and(uv.x.lessThan(1)).and(uv.y.greaterThan(0)).and(uv.y.lessThan(1)), () => {
      const px = 1 / s.lightTexSize;
      const at = (x: number, y: number) => s.lightTex.sample(uv.add(vec2(x * px, y * px))).level(0).rgb;
      out.assign(at(0, 0).mul(2).add(at(1.5, 0.5)).add(at(-0.5, 1.5)).add(at(-1.5, -0.5)).add(at(0.5, -1.5)).div(6));
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
        out.assign(float(1).sub(smoothstep(0.3, 1, across.div(along).mul(s.spot.y).div(s.spot.x))).mul(nearness).mul(nearness));
      });
    });
    return out;
  }).setLayout({ name: 'icSpotPool', type: 'float', inputs: [{ name: 'g', type: 'vec3' }] });

  /** The floor's colour at a ground point. */
  const groundColor = Fn(([g]: N[]) => {
    // The floor's shadows are blurred: the shadow of the main lamp a little, and a
    // much wider, fainter one where objects keep the light of the room off the floor.
    const direct = clamp(sunT(g), vec3(0), vec3(1)).toVar();
    const pool = spotPool(g).toVar();
    // Blurred only where the floor is seen directly, not through glass.
    if (!backdrop) return s.albedo.mul(s.skyE.mul(direct.mul(0.45).add(0.55)).add(s.sunE.mul(direct).mul(pool)));
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
    return s.albedo.mul(s.skyE.mul(ambient).add(s.sunE.mul(lamp).mul(pool)));
  }).setLayout({ name: 'icGroundColor', type: 'vec3', inputs: [{ name: 'g', type: 'vec3' }] });

  /** Ground and sky along a ray, ignoring glass. */
  const scene = Fn(([ro, rd, lampLit]: N[]) => {
    const out = vec3(0).toVar();
    If(groundOn.greaterThan(0.5).and(rd.y.lessThan(-1e-5)), () => {
      const t = max(ro.y.negate().div(rd.y), 0).toVar();
      const horizon = env(normalize(vec3(rd.x, 0.03, rd.z)), lampLit);
      out.assign(mix(groundColor(ro.add(rd.mul(t))), horizon, smoothstep(150, 600, t)));
    }).Else(() => {
      out.assign(env(rd, lampLit));
    });
    return out;
  }).setLayout({
    name: 'icScene',
    type: 'vec3',
    inputs: [{ name: 'ro', type: 'vec3' }, { name: 'rd', type: 'vec3' }, { name: 'lampLit', type: 'float' }],
  });

  return { dark, groundOn, env, sunT, spotPool, groundColor, scene };
}
