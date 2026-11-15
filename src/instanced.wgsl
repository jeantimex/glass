// Real-time glass for thousands of objects of every shape the app supports.
//
// Nothing is ray-marched and nothing is random. Objects are drawn as instanced
// triangles; each fragment follows the light through its own
// object with exact ray/shape formulas (only the torus and the baked mesh take a
// few steps along the ray), and whatever the light sees
// after leaving is looked up (sky, ground, or the picture of what is behind).
//
// The app builds three programs from this file by keeping only the lines between
// the matching `//#if` and `//#endif` markers:
//   CAUSTIC  objects drawn from the main lamp's direction into the caustic map
//   GROUND   the ground and sky
//   GLASS    the glass objects
//   BLUR     one direction of the blur that softens the shadows (and makes the glow)
//   SHADE    contact shadows, over the finished picture
//   LENS     depth of field, glow and tones: the last pass, onto the screen
//   PHOTON   real caustics: where each ray of the lamp's light lands on the floor
//#if BLUR
// A Gaussian blur along `dir` (in texture coordinates, out to its full length).
// Only what is brighter than `cut` is kept (0 keeps everything).
fn icBlur(uv: vec2f, dir: vec2f, cut: f32, reach: f32, src: texture_2d<f32>, srcSamp: sampler) -> vec4f {
  // `reach` samples each side of the middle: as many as the blur is texels wide.
  var sum = vec4f(0.0);
  var total = 0.0;
  let n = i32(reach);
  for (var i = -n; i <= n; i++) {
    let x = 2.0 * f32(i) / reach;
    let w = exp(-0.5 * x * x);
    sum += w * max(textureSampleLevel(src, srcSamp, uv + dir * (f32(i) / reach), 0.0) - vec4f(cut), vec4f(0.0));
    total += w;
  }
  return sum / total;
}
//#endif
//#if SHADE
// First finishing pass: contact shadows. The picture so far carries, in its alpha,
// how far each pixel is from the camera; from that, each pixel looks at the
// surfaces around it and darkens where they crowd in over it - under an object
// where it meets the floor, and in the gaps between objects. p = strength,
// radius (m), exposure.
fn icShade(fragCoord: vec2f, res: vec2f, camPos: vec3f, camRight: vec4f, camUp: vec3f, camFwd: vec4f, p: vec4f, src: texture_2d<f32>, srcSamp: sampler) -> vec4f {
  gRes = res; gCamPos = camPos; gCamRight = camRight; gCamUp = camUp; gCamFwd = camFwd;
  let uv = fragCoord / res;
  let here = textureSampleLevel(src, srcSamp, uv, 0.0);
  let dist = here.a;
  var shade = 1.0;
  if (p.x > 0.0 && dist < 900.0) {
    let px = 1.0 / res;
    let at = icPointAt(uv, dist);
    // The surface's direction, from its neighbours (on each axis, the one nearer in depth).
    let right = textureSampleLevel(src, srcSamp, uv + vec2f(px.x, 0.0), 0.0).a;
    let left = textureSampleLevel(src, srcSamp, uv - vec2f(px.x, 0.0), 0.0).a;
    let below = textureSampleLevel(src, srcSamp, uv + vec2f(0.0, px.y), 0.0).a;
    let above = textureSampleLevel(src, srcSamp, uv - vec2f(0.0, px.y), 0.0).a;
    var alongX = icPointAt(uv + vec2f(px.x, 0.0), right) - at;
    if (abs(left - dist) < abs(right - dist)) { alongX = at - icPointAt(uv - vec2f(px.x, 0.0), left); }
    var alongY = icPointAt(uv + vec2f(0.0, px.y), below) - at;
    if (abs(above - dist) < abs(below - dist)) { alongY = at - icPointAt(uv - vec2f(0.0, px.y), above); }
    var n = normalize(cross(alongY, alongX));
    if (dot(n, camPos - at) < 0.0) { n = -n; }
    // The floor takes the full shadow; glass, being clear, only a little.
    let onFloor = abs(at.y) < 0.02 + 0.002 * dist;
    if (onFloor) { n = vec3f(0.0, 1.0, 0.0); }

    // Points round about, in a spiral turned differently at every pixel.
    let reach = p.y / (dist * 2.0 * camFwd.w);   // the radius on screen, as a share of its height
    let turn = 6.2831853 * fract(52.9829189 * fract(dot(fragCoord, vec2f(0.06711056, 0.00583715))));
    var crowded = 0.0;
    for (var k = 0; k < 24; k++) {
      let a = turn + f32(k) * 2.3999632;
      let r = reach * sqrt((f32(k) + 0.5) / 24.0);
      let suv = uv + vec2f(cos(a) * r * res.y / res.x, sin(a) * r);
      let sd = textureSampleLevel(src, srcSamp, suv, 0.0).a;
      let point = icPointAt(suv, sd);
      // The floor never shades the floor. (Its depth is stored only roughly, so
      // nearby floor would read as tiny bumps, and leave a fine grain over it.)
      if (onFloor && abs(point.y) < 0.02 + 0.002 * sd) { continue; }
      // On the floor only what is close above it counts: the spot where an object
      // meets it. (Its shadow proper comes from the lamp. Counting all of an object
      // gave a wide, grainy halo round it, the samples falling on it or not at random.)
      if (onFloor && point.y > 0.1 * p.y) { continue; }
      let there = point - at;
      let far = length(there);
      crowded += max(dot(n, there) / (far + 1e-4) - 0.1, 0.0) * (1.0 - smoothstep(0.7 * p.y, 1.6 * p.y, far));
    }
    shade = 1.0 - p.x * select(0.3, 1.0, onFloor) * clamp(crowded / 24.0 * 2.6, 0.0, 1.0);
  }
  return vec4f(here.rgb * shade * p.z, dist);
}

var<private> gRes: vec2f;
var<private> gCamPos: vec3f;
var<private> gCamRight: vec4f;
var<private> gCamUp: vec3f;
var<private> gCamFwd: vec4f;

// The point in the world seen at a place on screen, that far from the camera.
fn icPointAt(uv: vec2f, dist: f32) -> vec3f {
  let ndc = (uv * 2.0 - 1.0) * vec2f(1.0, -1.0);
  let rd = normalize(gCamFwd.xyz + gCamRight.xyz * (ndc.x * gCamFwd.w * gCamRight.w) + gCamUp * (ndc.y * gCamFwd.w));
  return gCamPos + rd * dist;
}
//#endif
//#if LENS
// Last finishing pass: the camera. Depth of field (what is nearer or farther than
// the focus goes soft), the glow round the brightest glints, and film-like tones.
// p = focus distance to fall back on (m), aperture, glow, tones on.
fn icLens(fragCoord: vec2f, res: vec2f, p: vec4f, src: texture_2d<f32>, srcSamp: sampler, glowTex: texture_2d<f32>, glowSamp: sampler) -> vec4f {
  let uv = fragCoord / res;
  let here = textureSampleLevel(src, srcSamp, uv, 0.0);
  var col = here.rgb;
  // Autofocus: on whatever is in the middle of the picture (a few points round the
  // centre, in case one falls in a gap). Only if all of them see the backdrop does
  // it fall back on the distance it was given.
  var focus = 0.0;
  var found = 0.0;
  for (var k = 0; k < 5; k++) {
    let a = f32(k) * 1.2566371;
    let offset = select(vec2f(cos(a), sin(a)) * 0.03, vec2f(0.0), k == 0);
    let d = textureSampleLevel(src, srcSamp, vec2f(0.5) + offset * vec2f(res.y / res.x, 1.0), 0.0).a;
    if (d < 900.0) {
      focus += d;
      found += 1.0;
    }
  }
  focus = select(p.x, focus / max(found, 1.0), found > 0.0);
  let blur = icSoftness(here.a, focus, p.y);
  if (blur * res.y > 0.75) {
    // Gather from a disc as wide as this pixel is soft. A point in front of this
    // one only spills over it as far as it is soft itself.
    let turn = 6.2831853 * fract(52.9829189 * fract(dot(fragCoord, vec2f(0.06711056, 0.00583715))));
    var sum = here.rgb;
    var weight = 1.0;
    for (var k = 0; k < 24; k++) {
      let a = turn + f32(k) * 2.3999632;
      let r = blur * sqrt((f32(k) + 0.5) / 24.0);
      let there = textureSampleLevel(src, srcSamp, uv + vec2f(cos(a) * r * res.y / res.x, sin(a) * r), 0.0);
      var w = 1.0;
      if (there.a < here.a - 0.05) { w = clamp(icSoftness(there.a, focus, p.y) / max(r, 1e-5), 0.0, 1.0); }
      sum += there.rgb * w;
      weight += w;
    }
    col = sum / weight;
  }
  col += textureSampleLevel(glowTex, glowSamp, uv, 0.0).rgb * p.z;
  if (p.w > 0.5) { col = icTones(col); }
  return vec4f(col, 1.0);
}

// How soft a point at that distance is: the radius of its blur, as a share of the
// height of the picture.
fn icSoftness(dist: f32, focus: f32, aperture: f32) -> f32 {
  return min(aperture * 0.05 * abs(dist - focus) / max(dist, 0.1), 0.02);
}

// Film-like tones: left alone up to fairly bright, then eased toward white
// instead of clipping, losing a little colour on the way (the Khronos "neutral"
// curve).
fn icTones(cIn: vec3f) -> vec3f {
  var c = cIn;
  let low = min(c.r, min(c.g, c.b));
  c -= select(0.04, low - 6.25 * low * low, low < 0.08);
  let peak = max(c.r, max(c.g, c.b));
  let start = 0.76;
  if (peak < start) { return c; }
  let d = 1.0 - start;
  let newPeak = 1.0 - d * d / (peak + d - start);
  c *= newPeak / peak;
  let grey = 1.0 - 1.0 / (0.15 * (peak - newPeak) + 1.0);
  return mix(c, vec3f(newPeak), grey);
}
//#endif
//#if PHOTON
// Real caustics, first half. Objects are drawn from the main lamp's direction, so
// every pixel here is one ray of its light. Each is followed through its object
// (in at this point, round the inside for as long as it cannot get out, then out)
// and on down to the floor. The answer is how far from its straight path the ray
// lands, in the units of the caustic map (xy), and how much of the light is left
// (z). The second half moves every one of these rays to where it lands and adds
// them up, which is bright wherever the glass gathers them together.
fn icPhoton(worldPos: vec3f, nrmW: vec3f, center: vec3f, quat: vec4f, size: vec3f, vary: vec3f, tintAll: vec4f, tintOwn: vec3f, sunDir: vec3f, lightR: vec3f, lightU: vec3f, glassP: vec4f, cmyAll: f32, bevel: f32, raysAcross: f32, meshEntry: f32, sdfTex: texture_3d<f32>) -> vec4f {
  // sdfTex is only listed so three.js binds the baked mesh volume (see icGlass).
  // raysAcross: how many rays there are each way (the width of this picture).
  gBevel = bevel;
  gMeshEntry = meshEntry;
  let cmy = max(cmyAll, vary.z);
  let shape = IC_SHAPE;
  let d0 = icToLocal(-sunDir, quat);
  let entry = icEntry(shape, icToLocal(worldPos - center, quat), size, icToLocal(normalize(nrmW), quat));
  var n0 = entry.nrm;
  if (dot(d0, n0) > 0.0) { n0 = -n0; }
  let ior = max(glassP.x + vary.x, 1.02);
  var left = 1.0 - icFresnel(dot(d0, n0), ior);
  // Light that only grazes the surface is thrown wide in a thin, faint streak:
  // it is faded out, which keeps the pool of light clean.
  left *= smoothstep(0.05, 0.4, -dot(d0, n0));
  // The colour of the glass: the setting for all of it times the object's own,
  // faded toward clear by the setting's alpha.
  var tint = mix(vec3f(1.0), tintAll.rgb * tintOwn, tintAll.a);
  if (shape == 0 && cmy > 0.0) { tint *= icCmyTint(n0, cmy); }
  var dir = refract(d0, n0, 1.0 / ior);
  var pos = entry.pos;
  var leaving = vec3f(0.0);
  var escaped = false;
  if (dot(dir, dir) > 1e-6) {
    for (var e = 0; e < 8; e++) {
      let hit = icExit(shape, pos, dir, size);
      pos = hit.pos;
      if (shape == 0 && cmy > 0.0) { tint *= icCmyTint(hit.nrm, cmy); }
      let through = refract(dir, -hit.nrm, ior);
      if (dot(through, through) > 1e-6) {
        left *= 1.0 - icFresnel(dot(dir, hit.nrm), 1.0 / ior);
        leaving = through;
        escaped = true;
        break;
      }
      dir = reflect(dir, -hit.nrm);   // cannot get out here: it bounces back inside
    }
  }
  var thrown = vec2f(0.0);   // how far from its straight path the ray lands
  var light = 0.0;           // and how much light it still carries
  if (escaped) {
    let exitAt = center + icToWorld(pos, quat);
    let down = icToWorld(leaving, quat);
    if (down.y < -0.02) {
      let moved = exitAt + down * (-exitAt.y / down.y) - worldPos;
      thrown = vec2f(dot(moved, lightR), dot(moved, lightU));
      light = left * dot(tint, vec3f(0.333333));
    }
  }
  // Rays that the glass fans out land far apart, and one ray on its own would show
  // as a bright dot. How far apart is the area a little square of rays covers when
  // it lands, against the area it started with: found from how differently this
  // ray's neighbours are thrown. A fanned-out ray is then spread over that larger
  // area (up to a limit), and dimmed by as much; where neighbours are thrown
  // completely differently (across the edge of a facet) there is next to nothing left.
  let across = dpdx(thrown) * (0.5 * raysAcross);
  let downward = dpdy(thrown) * (0.5 * raysAcross);
  let spread = max(abs((1.0 + across.x) * (1.0 - downward.y) + downward.x * across.y), 1.0);
  // Alpha: above 0 marks the pixel as covered by glass (whether or not its ray
  // reaches the floor); beyond 1, it is how wide (in rays) to draw this one.
  return vec4f(thrown, light / spread, 1.0 + min(sqrt(spread), 6.0));
}
//#endif
//#if CAUSTIC
fn icCaustic(normalW: vec3f, worldPos: vec3f, size: vec3f, quat: vec4f, cmyAll: f32, vary: vec3f, tintAll: vec4f, tintOwn: vec3f) -> vec4f {
  // A CMY cube: every box when the whole pile is, or just the objects marked as one.
  let cmy = max(cmyAll, vary.z);
  // Every surface the lamp's light crosses multiplies it; multiplicative blending
  // builds up the product over all surfaces. The upper part of an object shades
  // the floor; the part resting on it gathers light there instead, the bright
  // patch at the foot of a cut stone.
  let n = normalize(normalW);
  let height = worldPos.y / (0.9 * max(size.x, max(size.y, size.z)));
  var c = vec3f(mix(1.32, 0.79, smoothstep(0.0, 1.0, height)));
  // Coloured glass colours the light that passes through it. Light crosses two
  // surfaces of an object (in and out), so each takes half of the colouring.
  c *= sqrt(mix(vec3f(1.0), tintAll.rgb * tintOwn, tintAll.a));
  // CMY cubes also colour the light, on every face it crosses.
  if (cmy > 0.0) {
    let local = n + 2.0 * cross(-quat.xyz, cross(-quat.xyz, n) + quat.w * n);
    c *= icCmyTint(local, cmy);
  }
  return vec4f(c, 1.0);
}
//#endif
//#if GROUND
fn icGround(fragCoord: vec2f, res: vec2f, camPos: vec3f, camRight: vec4f, camUp: vec3f, camFwd: vec4f, room: f32, skyE: vec3f, sunE: vec3f, albedo: vec3f, lightC: vec3f, lightR: vec3f, lightU: vec3f, floorP: vec4f, spot: vec4f, lamp: vec3f, lightTex: texture_2d<f32>, lightSamp: sampler, mirrorTex: texture_2d<f32>, mirrorSamp: sampler, shadeTex: texture_2d<f32>, shadeSamp: sampler, shadeFarTex: texture_2d<f32>, shadeFarSamp: sampler) -> vec4f {
  // shadeTex and shadeFarTex are only listed so three.js binds them; icSoftShade
  // reads them through their global names.
  gFloorP = floorP;
  // room: 1 if there is a floor, plus 2 for the dark studio.
  gDark = floor(room / 2.0); gGroundOn = room - 2.0 * gDark;
  gSpot = spot;
  gLamp = lamp;
  gSkyE = skyE; gSunE = sunE; gAlbedo = albedo;
  gLightC = lightC; gLightR = lightR; gLightU = lightU;
  let ndc = (fragCoord / res * 2.0 - 1.0) * vec2f(1.0, -1.0);
  let rd = normalize(camFwd.xyz + camRight.xyz * (ndc.x * camFwd.w * camRight.w) + camUp * (ndc.y * camFwd.w));
  gBackdrop = true;
  var col = icScene(camPos, rd);
  // The studio backdrop is lit a little unevenly: brighter up and to the left.
  col *= 1.0 + 0.07 * (0.8 * ndc.y - 0.5 * ndc.x);
  let onFloor = gGroundOn > 0.5 && rd.y < -1e-5;
  let floorT = -camPos.y / min(rd.y, -1e-5);
  // An endless grid on the floor: a line every metre, a heavier one every five metres.
  if (onFloor && floorP.z > 0.5) {
    let g = camPos.xz + rd.xz * floorT;
    // How much floor one pixel covers there (it stretches away from the camera).
    let pixel = floorT * (2.0 * camFwd.w / res.y) / max(-rd.y, 0.02);
    let near1 = abs(fract(g - 0.5) - 0.5);
    let near5 = abs(fract(g / 5.0 - 0.5) - 0.5) * 5.0;
    let fine = icGridLine(min(near1.x, near1.y), 0.009, pixel);
    let heavy = icGridLine(min(near5.x, near5.y), 0.02, pixel);
    let lines = min(0.38 * fine + 0.5 * heavy, 0.62);
    // Dark lines on the white floor, pale ones on the dark one.
    if (gDark > 0.5) { col += vec3f(0.035) * lines; } else { col *= 1.0 - lines; }
  }
  // A polished floor: the picture of the objects mirrored in it (drawn before this
  // pass), stronger at grazing angles and slightly blurred.
  if (floorP.x > 0.0 && onFloor) {
    let uv = fragCoord / res;
    // The mirror picture's alpha falls with height above the floor: a polished
    // (not mirror) floor shows what touches it clearly, and what is higher up
    // fainter and more blurred.
    var m = textureSampleLevel(mirrorTex, mirrorSamp, uv, 0.0);
    let px = mix(9.0, 1.0, clamp(m.a * 1.3, 0.0, 1.0)) / res;
    for (var k = 0; k < 12; k++) {
      let a = f32(k) * 2.3999632;
      m += textureSampleLevel(mirrorTex, mirrorSamp, uv + vec2f(cos(a), sin(a)) * sqrt((f32(k) + 0.5) / 12.0) * px, 0.0);
    }
    m /= 13.0;
    let grazing = 1.0 - clamp(-rd.y, 0.0, 1.0);
    // (A black floor shows its reflections the more clearly: twice as strong in the dark studio.)
    let k = min(floorP.x * select(1.0, 2.0, gDark > 0.5) * (0.35 + 0.65 * grazing * grazing), 1.0) * m.a;
    if (m.a > 1e-4) { col = mix(col, m.rgb / m.a, k); }
  }
  // The beam of the spotlight, made visible by the dust in the air.
  if (gSpot.x > 0.0) {
    let inBeam = icBeam(camPos, rd, select(1e4, floorT, onFloor));
    col += vec3f(0.07, 0.07, 0.075) * (1.0 - exp(-0.3 * inBeam));
  }
  // Fog: the floor fades into a pale haze with distance, and the haze hangs along
  // the horizon.
  if (floorP.w > 0.0) {
    let haze = icEnv(normalize(vec3f(rd.x, 0.03, rd.z))) * 1.07;
    if (onFloor) {
      col = mix(col, haze, 1.0 - exp(-0.05 * floorP.w * floorT));
    } else {
      col = mix(col, haze, exp(-max(rd.y, 0.0) * 5.0 / floorP.w));
    }
  }
  // (Alpha: how far away this pixel is, for the finishing passes.)
  return vec4f(col, select(1000.0, floorT, onFloor));
}

// How far a ray travels inside the cone of the spotlight (from the lamp, gSpot.y
// away toward gLamp, to the middle of the floor, where the beam is gSpot.x
// in radius), up to distance tEnd.
fn icBeam(o: vec3f, d: vec3f, tEnd: f32) -> f32 {
  let axis = -gLamp;
  let k = gSpot.x / gSpot.y;
  let cos2 = 1.0 / (1.0 + k * k);         // squared cosine of the half-angle of the cone
  let fromLamp = o - gLamp * gSpot.y;
  let dAxis = dot(d, axis);
  let oAxis = dot(fromLamp, axis);
  let a = dAxis * dAxis - cos2;
  let b = 2.0 * (dAxis * oAxis - dot(d, fromLamp) * cos2);
  let c = oAxis * oAxis - dot(fromLamp, fromLamp) * cos2;
  let disc = b * b - 4.0 * a * c;
  // Where along the ray it may count: in front of the camera, short of tEnd, and
  // on the lamp's side of the cone's tip that the light goes out from (the
  // cone's equation has a mirror half behind the lamp too).
  var lo = 0.0;
  var hi = tEnd;
  if (abs(dAxis) < 1e-6) {
    if (oAxis <= 0.0) { return 0.0; }
  } else if (dAxis > 0.0) {
    lo = max(lo, -oAxis / dAxis);
  } else {
    hi = min(hi, -oAxis / dAxis);
  }
  if (hi <= lo) { return 0.0; }
  // Inside the (double) cone where a t^2 + b t + c >= 0. A ray running more across
  // the beam than the cone opens (a < 0) is inside between the two crossings; one
  // running more along it (a > 0, as when looking down the beam from inside it)
  // is inside before the first crossing and after the second, or all the way if
  // it never crosses. (Leaving those rays out drew a hard-edged circle on the
  // floor round the point straight below a camera inside the beam.)
  if (abs(a) < 1e-7) {
    // Parallel to the cone's side: inside on one side of a single crossing.
    if (abs(b) < 1e-9) { return select(0.0, hi - lo, c >= 0.0); }
    let t = -c / b;
    if (b > 0.0) { return max(hi - max(lo, t), 0.0); }
    return max(min(hi, t) - lo, 0.0);
  }
  if (disc <= 0.0) { return select(0.0, hi - lo, a > 0.0); }
  let root = sqrt(disc);
  let r0 = min((-b - root) / (2.0 * a), (-b + root) / (2.0 * a));
  let r1 = max((-b - root) / (2.0 * a), (-b + root) / (2.0 * a));
  if (a < 0.0) { return max(min(hi, r1) - max(lo, r0), 0.0); }
  return max(min(hi, r0) - lo, 0.0) + max(hi - max(lo, r1), 0.0);
}

// How much of a pixel a grid line covers: the line is `half` wide either side of
// its middle, the pixel `pixel` wide, and `d` is how far the pixel is from the line.
fn icGridLine(d: f32, half: f32, pixel: f32) -> f32 {
  return clamp((half - d) / pixel + 0.5, 0.0, 1.0) * clamp(2.0 * half / pixel, 0.0, 1.0);
}
//#endif
//#if GLASS
fn icGlass(worldPos: vec3f, nrmW: vec3f, center: vec3f, quat: vec4f, size: vec3f, vary: vec3f, tintAll: vec4f, tintOwn: vec3f, camPos: vec3f, glassP: vec4f, viewProj: mat4x4f, room: f32, skyE: vec3f, sunE: vec3f, albedo: vec3f, lightC: vec3f, lightR: vec3f, lightU: vec3f, cmyAll: f32, mirror: f32, bevel: f32, spot: vec4f, lamp: vec3f, lightTex: texture_2d<f32>, lightSamp: sampler, behindTex: texture_2d<f32>, behindSamp: sampler, lampDepthTex: texture_2d<f32>, meshEntry: f32, sdfTex: texture_3d<f32>) -> vec4f {
  // sdfTex is only listed so three.js binds the baked mesh volume; icVolume reads
  // it through its global name (icSdfTex).
  // room: 1 if there is a floor, plus 2 for the dark studio.
  gDark = floor(room / 2.0); gGroundOn = room - 2.0 * gDark;
  gSpot = spot;
  gLamp = lamp;
  gSkyE = skyE; gSunE = sunE; gAlbedo = albedo;
  gLightC = lightC; gLightR = lightR; gLightU = lightU;
  gViewProj = viewProj;
  gEye = camPos;
  gReach = max(size.x, max(size.y, size.z));
  gSteps = glassP.w;
  gMirror = mirror;
  // A CMY cube: every box when the whole pile is, or just the objects marked as one.
  let cmy = max(cmyAll, vary.z);
  gBevel = bevel;
  gMeshEntry = meshEntry;
  // In the shadow of other objects? (The mirror image is drawn upside down below
  // the floor: the object itself is above it.) The lamp is dimmed for this object
  // in all it sees of the studio (see icEnv).
  let here = select(worldPos, worldPos * vec3f(1.0, -1.0, 1.0), mirror > 0.5);
  gLampLit = icLampVisible(here, dot(center, lamp) + 1.25 * length(size));
  let shape = IC_SHAPE;
  let rdW = normalize(worldPos - camPos);
  let d0 = icToLocal(rdW, quat);

  // Entry point and normal in the object's own space.
  let entry = icEntry(shape, icToLocal(worldPos - center, quat), size, icToLocal(normalize(nrmW), quat));
  let p0 = entry.pos;
  var n0 = entry.nrm;
  if (dot(d0, n0) > 0.0) { n0 = -n0; }

  // Each object can differ slightly from the settings: vary.x is added to the IOR,
  // vary.y multiplies the dispersion.
  let ior = max(glassP.x + vary.x, 1.02);
  // How far apart the IOR is for blue and red light (n_F - n_C): 1/12 for each unit of
  // the dispersion setting, far more than real glass shows, for visible fire.
  let disp = glassP.y * vary.y / 12.0;
  // The baked mesh is the one shape that has to be marched, so it gets fewer bounces.
  let maxEvents = select(i32(glassP.z), min(i32(glassP.z), 4), icIsMesh(shape));
  let layers = select(1, 3, disp > 0.0);
  var col = vec3f(0.0);
  // The glass colours light where it enters (white, except for CMY cubes).
  // The colour of the glass: the setting for all of it times the object's own,
  // faded toward clear by the setting's alpha.
  var tint = mix(vec3f(1.0), tintAll.rgb * tintOwn, tintAll.a);
  if (shape == 0 && cmy > 0.0) { tint *= icCmyTint(n0, cmy); }

  for (var c = 0; c < layers; c++) {
    // One colour layer: IOR - 1.5 d, IOR, IOR + d.
    var ch = 1;
    if (layers == 3) { ch = c; }
    let iorIn = max(ior + icIorOffset(disp, ch), 1.0001);
    let f0 = icFresnel(dot(d0, n0), iorIn);
    var layer = f0 * icLook(worldPos + icToWorld(n0, quat) * 0.003, icToWorld(reflect(d0, n0), quat), false);
    var thr = 1.0 - f0;   // light still inside, weighted by the chance of getting here
    var prob = 1.0;
    var dir = refract(d0, n0, 1.0 / iorIn);
    var pos = p0;
    // A CMY cube's faces are colour filters: light is filtered by the face it enters
    // through, by each face it bounces off inside, and by the face it leaves through. Seen through one face, the others mix
    // their colours into it: the cube inside the cube.
    var filtered = tint;
    if (dot(dir, dir) > 1e-6) {
      for (var e = 1; e < maxEvents; e++) {
        // Where the ray meets the inside of the surface again, exactly.
        let hit = icExit(shape, pos, dir, size);
        pos = hit.pos;
        let nrm = hit.nrm;
        var face = vec3f(1.0);
        if (shape == 0 && cmy > 0.0) { face = icCmyTint(nrm, cmy); }
        // The exit glass picks one of the three layer IORs at random in the path
        // tracer; here all three are taken at once, weighted by their chance.
        let fx = icExitFactor(min(8.0 * hit.t, 1.0));
        var reflProb = 0.0;
        for (var k = 0; k < 3; k++) {
          var wk = fx / 3.0;
          if (layers == 1) {
            if (k != 1) { continue; }
            wk = 1.0;
          } else if (k == ch) {
            wk += 1.0 - fx;
          }
          let iorK = max(ior + icIorOffset(disp, k), 1.0001);
          let rf = refract(dir, -nrm, iorK);
          var f = icFresnel(dot(dir, nrm), 1.0 / iorK);
          if (dot(rf, rf) < 1e-6) { f = 1.0; }
          reflProb += wk * f;
          let outW = thr * wk * (1.0 - f);
          if (outW > 0.004) {
            let seen = icLook(center + icToWorld(pos + nrm * 0.003, quat), icToWorld(rf, quat), true);
            layer += filtered * face * min(outW * seen, vec3f(10.0 * prob * wk * (1.0 - f)));
          }
        }
        thr *= reflProb;
        prob *= reflProb;
        if (thr < 0.004) { break; }
        dir = reflect(dir, -nrm);
        filtered *= face;
      }
      // Light still bouncing around inside when the bounces run out (common in
      // cones and octahedra) is let out, rather than left as a black patch.
      if (thr >= 0.004) {
        layer += filtered * min(thr * icLook(center + icToWorld(pos, quat), icToWorld(dir, quat), true), vec3f(10.0 * prob));
      }
    }
    if (layers == 3) { col[c] = layer[c]; } else { col = layer; }
  }
  if (mirror > 0.5) {
    let fade = exp(-3.2 * max(worldPos.y, 0.0));
    return vec4f(col * fade, fade);
  }
  // (Alpha: how far away this pixel is, for the finishing passes.)
  // In the shadow of other objects, the light the lamp would have sent through it
  // is mostly missing: darker by the lamp's share of all the light in the studio
  // (about a third in the white studio, half under the spotlight), less what the
  // glass in the way lets through.
  let lampShare = dot(sunE, vec3f(1.0)) / max(dot(sunE + skyE, vec3f(1.0)), 1e-4);
  col *= mix(1.0 - 0.55 * lampShare, 1.0, gLampLit);
  return vec4f(col, length(worldPos - camPos));
}
//#endif

//#if CAUSTIC GLASS PHOTON
// CMY cube: each pair of opposite faces is a filter that holds back one primary
// (cyan, yellow, magenta). `n` is the face normal in the cube's own space;
// `strength` runs from clear glass (0) to the full filter (1). The filters pass
// their own colours almost untouched, which keeps the cube luminous, and each
// lets a little of the blocked one through, so three in a row stay a deep colour
// instead of going grey.
fn icCmyTint(n: vec3f, strength: f32) -> vec3f {
  let a = abs(n);
  var passed = vec3f(1.0, 0.14, 0.92);
  if (a.x >= a.y && a.x >= a.z) { passed = vec3f(0.1, 0.92, 1.0); }
  else if (a.y >= a.z) { passed = vec3f(1.0, 0.9, 0.2); }
  return mix(vec3f(1.0), passed, strength);
}
//#endif

const IC_PI = 3.14159265359;

//#if GROUND
// The blurred shadow maps at a point of the caustic map: x a little blurred (the
// shadow of the lamp), y blurred far wider (where objects keep the light of the
// room off the floor).
fn icSoftShade(uv: vec2f) -> vec2f {
  return vec2f(textureSampleLevel(icShadeTex, icShadeTex_sampler, uv, 0.0).g, textureSampleLevel(icShadeFarTex, icShadeFarTex_sampler, uv, 0.0).g);
}
//#endif
//#if GLASS
fn icSoftShade(uv: vec2f) -> vec2f {
  return vec2f(1.0);
}
//#endif
//#if GROUND GLASS
var<private> gFloorP: vec4f;   // floor: how reflective, how dark its shadows, grid on, fog
var<private> gGroundOn: f32;   // 1: there is a floor
var<private> gDark: f32;       // 1: the dark studio
var<private> gSpot: vec4f;     // stage spotlight: x the radius of its beam at the floor (0: off), y how far away the lamp is
var<private> gLamp: vec3f;     // toward the main lamp (which is also the spotlight), from the middle of the floor
var<private> gBackdrop: bool;  // looking straight at the background, not through or off glass
var<private> gSkyE: vec3f;     // sky irradiance / pi on the ground
var<private> gSunE: vec3f;     // the main lamp's irradiance / pi on the ground
var<private> gAlbedo: vec3f;
var<private> gLightC: vec3f;   // caustic map: centre on the ground,
var<private> gLightR: vec3f;   //   its right axis / half-width,
var<private> gLightU: vec3f;   //   its up axis / half-width

// A photographer's white studio: a pale room with a few bright lamps, and dark
// cards for the glass to pick up (they give colourless stones their contrast).
// The camera itself only ever sees the plain white backdrop.
var<private> gLampLit: f32 = 1.0;  // how much of the lamp reaches the object being shaded (1: all; see icLampVisible)

// For an object in the shadow of others: how much of the lamp it still sees in its
// reflections (the ones in the way are glass, and let about a third through).
fn icLampShare() -> f32 {
  return mix(0.35, 1.0, gLampLit);
}

fn icEnv(d: vec3f) -> vec3f {
  // As bright as the open ground, so the two meet without a visible horizon.
  // (Under a spotlight the backdrop is outside the pool, so the lamp does not reach it.)
  if (gBackdrop) { return gAlbedo * (gSkyE + gSunE * select(1.0, 0.0, gSpot.x > 0.0)); }
  let az = atan2(d.z, d.x);
  if (gDark > 0.5) {
    // The dark studio: black all round. With the spotlight it is lit by that one
    // lamp and nothing else. Without, by a large soft lamp, a soft one overhead,
    // and two tall strip lights that draw bright lines down the edges of the glass.
    var k = 0.012;
    if (gSpot.x > 0.0) {
      // (The room is not quite black: a dim grey, lighter toward the floor, which is
      // what the dark band low in a glass ball is showing.)
      k = mix(0.11, 0.035, smoothstep(0.0, 0.6, d.y));
      k += 40.0 * icLampShare() * smoothstep(0.988, 0.996, dot(d, gLamp));
      // A jeweller's softboxes, out of the picture: a large one overhead and a tall
      // strip either side. The camera never sees them, only the glass does: they are
      // what makes the facets of a stone flash white against the black.
      // Overhead: four panels round a dark hole (where the camera looks down through
      // the tent), so a stone's table shows a pattern of light and dark, not a blank.
      let ring = smoothstep(0.78, 0.86, d.y) * (1.0 - smoothstep(0.94, 0.97, d.y));
      k += 2.6 * ring * smoothstep(-0.2, 0.3, cos(4.0 * az));
      let tall = smoothstep(0.05, 0.15, d.y) * (1.0 - smoothstep(0.55, 0.7, d.y));
      // (Left and right of where the camera starts, not behind the stones, where a
      // table facing up would mirror one straight into the camera as a white blank.)
      k += 4.0 * tall * smoothstep(0.95, 0.985, cos(az - 2.5));
      k += 4.0 * tall * smoothstep(0.95, 0.985, cos(az + 0.63));
    } else {
      k += 6.0 * icLampShare() * smoothstep(0.84, 0.95, dot(d, gLamp));
      k += 1.6 * smoothstep(0.8, 0.97, d.y);
      let tall = smoothstep(0.02, 0.1, d.y) * (1.0 - smoothstep(0.6, 0.75, d.y));
      k += 5.0 * tall * smoothstep(0.985, 0.996, cos(az - 0.5));
      k += 3.5 * tall * smoothstep(0.99, 0.998, cos(az + 1.9));
    }
    return vec3f(k);
  }
  // A light tent: pale overhead, dark grey toward the floor.
  var c = mix(0.15, 0.72, smoothstep(-0.1, 0.95, d.y));
  // A few grey cards standing round the set, and a small dark one overhead (the
  // camera): they are what the stone's darker facets are looking at.
  let band = smoothstep(0.03, 0.1, d.y) * (1.0 - smoothstep(0.68, 0.82, d.y));
  c = mix(c, 0.04, 0.96 * band * smoothstep(-0.1, 0.15, sin(az * 5.0 + 1.0)));
  c = mix(c, 0.05, smoothstep(0.9, 0.95, d.y));
  // Lamps: the main one up and to the left, and two small ones.
  c += 3.0 * icLampShare() * smoothstep(0.86, 0.96, dot(d, gLamp));
  c += 3.0 * smoothstep(0.985, 0.995, dot(d, normalize(vec3f(cos(3.0), 0.6, sin(3.0)))));
  c += 3.0 * smoothstep(0.985, 0.995, dot(d, normalize(vec3f(cos(5.0), 0.9, sin(5.0)))));
  return vec3f(c);
}

// How much of the main lamp's light reaches a ground point: the caustic map, lightly blurred.
fn icSunT(g: vec3f) -> vec3f {
  let v = g - gLightC;
  let uv = vec2f(0.5 + 0.5 * dot(v, gLightR), 0.5 - 0.5 * dot(v, gLightU));
  if (uv.x <= 0.0 || uv.x >= 1.0 || uv.y <= 0.0 || uv.y >= 1.0) { return vec3f(1.0); }
  let px = 1.0 / vec2f(textureDimensions(icLightTex));
  var sum = textureSampleLevel(icLightTex, icLightTex_sampler, uv, 0.0).rgb * 2.0;
  sum += textureSampleLevel(icLightTex, icLightTex_sampler, uv + vec2f(1.5, 0.5) * px, 0.0).rgb;
  sum += textureSampleLevel(icLightTex, icLightTex_sampler, uv + vec2f(-0.5, 1.5) * px, 0.0).rgb;
  sum += textureSampleLevel(icLightTex, icLightTex_sampler, uv + vec2f(-1.5, -0.5) * px, 0.0).rgb;
  sum += textureSampleLevel(icLightTex, icLightTex_sampler, uv + vec2f(0.5, -1.5) * px, 0.0).rgb;
  return sum / 6.0;
}

fn icGroundColor(g: vec3f) -> vec3f {
  let sun = icSunT(g);
  // The floor's shadows are blurred: the shadow of the main lamp a
  // little, and a much wider, fainter one where objects keep the light of the room
  // off the floor.
  let direct = clamp(sun, vec3f(0.0), vec3f(1.0));
  // Blurred only where the floor is seen directly, not through glass.
  let pool = icSpotPool(g);
  if (!gBackdrop) { return gAlbedo * (gSkyE * (0.55 + 0.45 * direct) + gSunE * direct * pool); }
  let v = g - gLightC;
  let uv = vec2f(0.5 + 0.5 * dot(v, gLightR), 0.5 - 0.5 * dot(v, gLightU));
  let shade = icSoftShade(uv);
  // gFloorP.y is how dark the shadows are; negative when the near map holds real
  // caustics (the light that actually arrives), which are used as they are.
  let darkness = abs(gFloorP.y);
  var lamp = min(max(shade.x, 0.0), 1.7);
  if (gFloorP.y < 0.0) {
    // Gathered light is eased off above the level of the open floor, toward three
    // times it at most. Only the top layer of a pile is followed, as if nothing lay
    // beneath it, so taken at face value a pile would light its floor far too much.
    let over = max(shade.x - 1.0, 0.0);
    lamp = min(shade.x, 1.0) + over / (1.0 + 0.5 * over);
  }
  // How dark the shadows are: the share of the lamp's light they hold back (0: none,
  // 1: all of it). Light gathered above the open floor's is left as it is.
  if (lamp < 1.0) { lamp = 1.0 - darkness * (1.0 - lamp); }
  let ambient = 1.0 - 0.8 * darkness * (1.0 - min(shade.y, 1.0));
  return gAlbedo * (gSkyE * ambient + gSunE * lamp * pool);
}

// How much of the spotlight reaches a point of the floor: all of it inside the
// pool, fading out at its edge. (1 everywhere when there is no spotlight.)
fn icSpotPool(g: vec3f) -> f32 {
  if (gSpot.x <= 0.0) { return 1.0; }
  // How far the point is from the axis of the beam, against how wide the beam is
  // there. The beam comes down at an angle, so the pool is an oval.
  let fromLamp = g - gLamp * gSpot.y;
  let along = dot(fromLamp, -gLamp);
  if (along <= 0.0) { return 0.0; }
  let across = length(fromLamp + gLamp * along);
  // It fades out gradually toward its edge, and (the lamp being near) the floor is
  // brighter on the side toward the lamp than on the far side.
  let nearness = gSpot.y / max(along, 0.3 * gSpot.y);
  return (1.0 - smoothstep(0.3, 1.0, across / along * gSpot.y / gSpot.x)) * nearness * nearness;
}

// Ground and sky along a ray, ignoring glass.
fn icScene(ro: vec3f, rd: vec3f) -> vec3f {
  if (gGroundOn > 0.5 && rd.y < -1e-5) {
    let t = max(-ro.y / rd.y, 0.0);
    let horizon = icEnv(normalize(vec3f(rd.x, 0.03, rd.z)));
    return mix(icGroundColor(ro + rd * t), horizon, smoothstep(150.0, 600.0, t));
  }
  return icEnv(rd);
}
//#endif

//#if GLASS PHOTON
var<private> gViewProj: mat4x4f;
var<private> gEye: vec3f;      // the camera
var<private> gReach: f32;      // about how big this object is (m)
var<private> gSteps: f32;      // how many steps a ray leaving it is followed for, looking for other glass (0: not at all)
var<private> gMirror: f32;     // 1 while drawing the mirror image in the floor
var<private> gBevel: f32;      // how much is cut off the edges of boxes (m)
var<private> gMeshEntry: f32;  // 1: light enters a baked mesh through its volume's surface (see icEntry)

// A point on an object's surface, its outward normal there, and (for exits) how
// far the ray travelled inside the glass to reach it.
struct IcHit {
  pos: vec3f,
  nrm: vec3f,
  t: f32,
}

// Each glass program is built for one shape: the app rewrites this line.
const IC_SHAPE = 0;

// Shapes, in the object's own space (size = per-shape dimensions):
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

// The diamond's facets at size 1 (IC_DIAMOND, IC_DIAMOND_N), from diamond.ts.
/*DIAMOND*/

// The flat faces of the polyhedra as planes: xyz = outward normal, w = offset.
// (A box counts as one once its edges are bevelled: its six faces plus a flat
// cut along each of its twelve edges.)
fn icIsPolyhedron(shape: i32) -> bool {
  return (shape >= 6 && shape <= 8) || shape == 10 || (shape == 0 && gBevel > 0.0);
}

fn icPlaneCount(shape: i32) -> i32 {
  if (shape == 7) { return 5; }
  if (shape == 10) { return IC_DIAMOND_N; }
  if (shape == 0) { return 18; }
  return 8;
}

fn icPlane(shape: i32, k: i32, size: vec3f) -> vec4f {
  if (shape == 0) {
    if (k < 6) {
      var n = vec3f(0.0);
      n[k / 2] = select(-1.0, 1.0, (k & 1) == 1);
      return vec4f(n, size[k / 2]);
    }
    // The cut along an edge: at 45 degrees between the two faces that meet there.
    let e = k - 6;
    let a = select(-1.0, 1.0, (e & 1) == 1);
    let b = select(-1.0, 1.0, (e & 2) == 2);
    let pair = e / 4;
    if (pair == 0) { return vec4f(vec3f(a, b, 0.0) * 0.70710678, (size.x + size.y - gBevel) * 0.70710678); }
    if (pair == 1) { return vec4f(vec3f(0.0, a, b) * 0.70710678, (size.y + size.z - gBevel) * 0.70710678); }
    return vec4f(vec3f(a, 0.0, b) * 0.70710678, (size.x + size.z - gBevel) * 0.70710678);
  }
  if (shape == 10) {
    let facet = IC_DIAMOND[k];
    return vec4f(facet.xyz, facet.w * size.x);
  }
  if (shape == 6) {
    let n = vec3f(select(-1.0, 1.0, (k & 1) != 0), select(-1.0, 1.0, (k & 2) != 0), select(-1.0, 1.0, (k & 4) != 0));
    return vec4f(n * 0.57735027, size.x * 0.57735027);
  }
  if (shape == 7) {
    if (k == 0) { return vec4f(0.0, -1.0, 0.0, size.x * 0.5); }
    if (k == 1) { return vec4f(0.8660254, 0.5, 0.0, size.x * 0.5); }
    if (k == 2) { return vec4f(-0.8660254, 0.5, 0.0, size.x * 0.5); }
    if (k == 3) { return vec4f(0.0, 0.0, 1.0, size.z); }
    return vec4f(0.0, 0.0, -1.0, size.z);
  }
  if (k == 6) { return vec4f(0.0, 1.0, 0.0, size.y); }
  if (k == 7) { return vec4f(0.0, -1.0, 0.0, size.y); }
  let a = (30.0 + 60.0 * f32(k)) * 0.017453293;
  return vec4f(cos(a), 0.0, sin(a), size.x);
}

// The shapes made from meshes, and which of the baked volumes is theirs.
fn icIsMesh(shape: i32) -> bool {
  return shape == 9 || shape == 11;
}

fn icMeshLayer(shape: i32) -> i32 {
  return select(0, 1, shape == 11);
}

// Signed distance to a baked mesh: a 128^3 grid covering a cube of half-width 1.1
// around the mesh, which is scaled to half-width 1. The volumes of the meshes lie
// one after another along z in the same texture; `layer` picks one.
fn icVolume(p: vec3f, scale: f32, layer: i32) -> f32 {
  let q = p / scale;
  let inner = vec3f(1.1 * (1.0 - 1.0 / 128.0));
  let c = clamp(q, -inner, inner);
  let g = (c / 2.2 + 0.5) * 128.0 - 0.5;
  let cell = clamp(vec3i(floor(g)), vec3i(0), vec3i(126));
  // (How far between cells, within this mesh's own volume: worked out before moving
  // to where that volume lies in the texture, or every mesh after the first would
  // not be blended along z at all, and look built of flat slices.)
  let f = clamp(g - vec3f(cell), vec3f(0.0), vec3f(1.0));
  let i0 = cell + vec3i(0, 0, 128 * layer);
  let d000 = textureLoad(icSdfTex, i0, 0).r;
  let d100 = textureLoad(icSdfTex, i0 + vec3i(1, 0, 0), 0).r;
  let d010 = textureLoad(icSdfTex, i0 + vec3i(0, 1, 0), 0).r;
  let d110 = textureLoad(icSdfTex, i0 + vec3i(1, 1, 0), 0).r;
  let d001 = textureLoad(icSdfTex, i0 + vec3i(0, 0, 1), 0).r;
  let d101 = textureLoad(icSdfTex, i0 + vec3i(1, 0, 1), 0).r;
  let d011 = textureLoad(icSdfTex, i0 + vec3i(0, 1, 1), 0).r;
  let d111 = textureLoad(icSdfTex, i0 + vec3i(1, 1, 1), 0).r;
  let d = mix(mix(mix(d000, d100, f.x), mix(d010, d110, f.x), f.y),
              mix(mix(d001, d101, f.x), mix(d011, d111, f.x), f.y), f.z);
  return (d + length(q - c)) * scale;
}

// The outward direction of a baked mesh's surface at p (from its volume), sampled
// almost two cells apart: closer, the slope of the blended grid jumps at every cell
// boundary, and the normals with it, which shows as stripes and ripples.
fn icVolumeNormal(p: vec3f, scale: f32, layer: i32) -> vec3f {
  let e = vec2f(1.0, -1.0) * (0.03 * scale);
  return normalize(e.xyy * icVolume(p + e.xyy, scale, layer) + e.yyx * icVolume(p + e.yyx, scale, layer) +
                   e.yxy * icVolume(p + e.yxy, scale, layer) + e.xxx * icVolume(p + e.xxx, scale, layer));
}

// Snap a rasterised surface point onto the exact shape and find its normal.
fn icEntry(shape: i32, pIn: vec3f, size: vec3f, meshNormal: vec3f) -> IcHit {
  var p = pIn;
  var n: vec3f;
  if (icIsMesh(shape) && gMeshEntry > 0.5) {
    // A baked mesh, with only a few objects on show (each large on screen): the
    // point moved onto the volume's surface, with the volume's normal there, so
    // that the light enters the same shape it is followed through and leaves. (The
    // mesh and its 128^3 volume differ by up to about a cell: with the light
    // entering one and leaving the other, the bands where they differ show as
    // jagged dark patches.) The point moves a fraction of a cell, so the normal is
    // found once for the move and once where it lands.
    let layer = icMeshLayer(shape);
    let toward = icVolumeNormal(p, size.x, layer);
    p -= toward * icVolume(p, size.x, layer);
    p -= toward * icVolume(p, size.x, layer);
    n = icVolumeNormal(p, size.x, layer);
  } else if (icIsMesh(shape)) {
    // In a pile, where each is small on screen, the cheaper way: the rasterised
    // point and the mesh's own smooth normal.
    n = meshNormal;
  } else if (shape == 4) {
    let k = size.x / (2.0 * size.y);
    let rho = max(length(p.xz), 1e-6);
    let inv = 1.0 / sqrt(1.0 + k * k);
    let sBase = -p.y - size.y;
    let sSide = (rho - k * (size.y - p.y)) * inv;
    if (sBase >= sSide) {
      n = vec3f(0.0, -1.0, 0.0);
      p.y = -size.y;
    } else {
      n = vec3f(p.x / rho, k, p.z / rho) * inv;
      p -= n * sSide;
    }
  } else if (shape == 5) {
    let rho = max(length(p.xz), 1e-6);
    let ring = vec3f(p.x / rho, 0.0, p.z / rho) * size.x;
    n = normalize(p - ring);
    p = ring + n * size.y;
  } else if (icIsPolyhedron(shape)) {
    // The face this point lies on is the plane it is closest to.
    var best = -1e9;
    n = vec3f(0.0, 1.0, 0.0);
    for (var k = 0; k < icPlaneCount(shape); k++) {
      let pl = icPlane(shape, k, size);
      let sd = dot(pl.xyz, p) - pl.w;
      if (sd > best) { best = sd; n = pl.xyz; }
    }
    p -= n * best;
  } else if (shape == 1) {
    n = normalize(p);
    p = n * size.x;
  } else if (shape == 2) {
    let radial = length(p.xz);
    if (abs(p.y) / size.y >= radial / size.x) {
      n = vec3f(0.0, sign(p.y), 0.0);
      p.y = n.y * size.y;
    } else {
      n = vec3f(p.x, 0.0, p.z) / max(radial, 1e-6);
      p = vec3f(n.x * size.x, p.y, n.z * size.x);
    }
  } else if (shape == 3) {
    let c = vec3f(0.0, clamp(p.y, -size.y, size.y), 0.0);
    n = normalize(p - c);
    p = c + n * size.x;
  } else {
    let a = abs(p) / size;
    if (a.x >= a.y && a.x >= a.z) { n = vec3f(sign(p.x), 0.0, 0.0); p.x = n.x * size.x; }
    else if (a.y >= a.z) { n = vec3f(0.0, sign(p.y), 0.0); p.y = n.y * size.y; }
    else { n = vec3f(0.0, 0.0, sign(p.z)); p.z = n.z * size.z; }
  }
  return IcHit(p, n, 0.0);
}

// From a point on the surface, heading inwards along dir: where the ray reaches
// the surface again.
fn icExit(shape: i32, pos: vec3f, dir: vec3f, size: vec3f) -> IcHit {
  if (shape == 1) {
    let t = max(-2.0 * dot(pos, dir), 0.0);
    let q = pos + dir * t;
    return IcHit(q, normalize(q), t);
  }
  if (shape == 2 || shape == 3) {
    // Distance to the round side wall.
    let a = dot(dir.xz, dir.xz);
    var tSide = 1e9;
    if (a > 1e-9) {
      let b = dot(pos.xz, dir.xz);
      let c = dot(pos.xz, pos.xz) - size.x * size.x;
      tSide = (-b + sqrt(max(b * b - a * c, 0.0))) / a;
    }
    if (shape == 2) {
      var tCap = 1e9;
      if (abs(dir.y) > 1e-7) { tCap = (sign(dir.y) * size.y - pos.y) / dir.y; }
      let t = max(min(tSide, tCap), 0.0);
      let q = pos + dir * t;
      if (tCap < tSide) { return IcHit(q, vec3f(0.0, sign(dir.y), 0.0), t); }
      return IcHit(q, normalize(vec3f(q.x, 0.0, q.z)), t);
    }
    // Capsule: the side wall if the hit is on the middle part, else an end cap.
    let ySide = pos.y + dir.y * min(tSide, 1e6);
    if (abs(ySide) <= size.y) {
      let t = max(tSide, 0.0);
      let q = pos + dir * t;
      return IcHit(q, normalize(vec3f(q.x, 0.0, q.z)), t);
    }
    let cap = vec3f(0.0, sign(ySide) * size.y, 0.0);
    let oc = pos - cap;
    let b = dot(oc, dir);
    let t = max(-b + sqrt(max(b * b - (dot(oc, oc) - size.x * size.x), 0.0)), 0.0);
    let q = pos + dir * t;
    return IcHit(q, normalize(q - cap), t);
  }
  if (shape == 4) {
    // Cone: the slanted side is a quadratic in t, the base a plane.
    let k = size.x / (2.0 * size.y);
    let k2 = k * k;
    let hy = size.y - pos.y;
    let a = dot(dir.xz, dir.xz) - k2 * dir.y * dir.y;
    let b = dot(pos.xz, dir.xz) + k2 * hy * dir.y;
    let c = dot(pos.xz, pos.xz) - k2 * hy * hy;
    var tSide = 1e9;
    if (abs(a) > 1e-7) {
      let disc = b * b - a * c;
      if (disc >= 0.0) {
        let sq = sqrt(disc);
        for (var r = 0; r < 2; r++) {
          let t = (-b + select(-sq, sq, r == 1)) / a;
          let y = pos.y + dir.y * t;
          if (t > 1e-4 && y <= size.y && y >= -size.y - 1e-4) { tSide = min(tSide, t); }
        }
      }
    } else if (abs(b) > 1e-9) {
      let t = -c / (2.0 * b);
      if (t > 1e-4) { tSide = t; }
    }
    var tBase = 1e9;
    if (dir.y < -1e-7) { tBase = (-size.y - pos.y) / dir.y; }
    let t = select(min(tSide, tBase), 0.0, min(tSide, tBase) > 1e8);
    let q = pos + dir * t;
    if (tBase < tSide) { return IcHit(q, vec3f(0.0, -1.0, 0.0), t); }
    let rho = max(length(q.xz), 1e-6);
    return IcHit(q, normalize(vec3f(q.x / rho, k, q.z / rho)), t);
  }
  if (shape == 5) {
    // Torus: a quartic, so step along the ray using its exact distance instead.
    var t = 0.004;
    for (var i = 0; i < 32; i++) {
      let q = pos + dir * t;
      let d = size.y - length(vec2f(length(q.xz) - size.x, q.y));
      if (d < 0.0006) { break; }
      t += max(d, 0.0015);
    }
    let q = pos + dir * t;
    let rho = max(length(q.xz), 1e-6);
    return IcHit(q, normalize(q - vec3f(q.x / rho, 0.0, q.z / rho) * size.x), t);
  }
  if (icIsPolyhedron(shape)) {
    // Polyhedron: the nearest face plane the ray is heading towards.
    var t = 1e9;
    var n = vec3f(0.0, 1.0, 0.0);
    for (var k = 0; k < icPlaneCount(shape); k++) {
      let pl = icPlane(shape, k, size);
      let dn = dot(pl.xyz, dir);
      if (dn > 1e-7) {
        let tk = (pl.w - dot(pl.xyz, pos)) / dn;
        if (tk < t) { t = tk; n = pl.xyz; }
      }
    }
    t = select(max(t, 0.0), 0.0, t > 1e8);
    return IcHit(pos + dir * t, n, t);
  }
  if (icIsMesh(shape)) {
    // Baked mesh: step through the distance volume.
    let layer = icMeshLayer(shape);
    // (Distances are in units of the object's size, so a small one in a pile is
    // stepped through as finely as a large one on its own. A cell of the volume
    // is 2.2 / 128 = 0.017 of it.)
    let s = size.x;
    var t = 0.004 * s;
    // The ray starts on the mesh, which the volume follows only to within about a
    // cell: where the volume's surface lies a little inside the mesh's, the start
    // is outside it. Step forward until inside, or the exit is found at once, in
    // the wrong place: a dark cut across the glass.
    for (var i = 0; i < 8; i++) {
      let d = icVolume(pos + dir * t, s, layer);
      if (d < 0.0) { break; }
      t += max(d, 0.002 * s);
    }
    // Then to where it leaves. Steps of at least a quarter of a cell: a ray running
    // along the inside of a long body, close to its surface all the way, would
    // otherwise only creep forward and run out of steps somewhere in the middle,
    // which drew contour-like dark lines across the glass.
    var tIn = t;   // the last point known to be inside
    for (var i = 0; i < 48; i++) {
      let d = -icVolume(pos + dir * t, s, layer);
      if (d < 0.0005 * s) { break; }
      tIn = t;
      t += max(d, 0.004 * s);
    }
    // ... and the surface pinned down between the last point inside and the first
    // outside, by halving the gap.
    for (var i = 0; i < 3; i++) {
      let mid = 0.5 * (tIn + t);
      if (icVolume(pos + dir * mid, s, layer) < 0.0) { tIn = mid; } else { t = mid; }
    }
    let q = pos + dir * t;
    return IcHit(q, icVolumeNormal(q, s, layer), t);
  }
  // Box.
  let tx = select(1e9, (sign(dir.x) * size.x - pos.x) / dir.x, abs(dir.x) > 1e-7);
  let ty = select(1e9, (sign(dir.y) * size.y - pos.y) / dir.y, abs(dir.y) > 1e-7);
  let tz = select(1e9, (sign(dir.z) * size.z - pos.z) / dir.z, abs(dir.z) > 1e-7);
  let t = max(min(tx, min(ty, tz)), 0.0);
  let q = pos + dir * t;
  if (tx <= ty && tx <= tz) { return IcHit(q, vec3f(sign(dir.x), 0.0, 0.0), t); }
  if (ty <= tz) { return IcHit(q, vec3f(0.0, sign(dir.y), 0.0), t); }
  return IcHit(q, vec3f(0.0, 0.0, sign(dir.z)), t);
}

fn icToLocal(v: vec3f, q: vec4f) -> vec3f {
  let u = -q.xyz;
  return v + 2.0 * cross(u, cross(u, v) + q.w * v);
}

fn icToWorld(v: vec3f, q: vec4f) -> vec3f {
  return v + 2.0 * cross(q.xyz, cross(q.xyz, v) + q.w * v);
}

// How much the IOR for red (k = 0) and blue (k = 2) light differs from the quoted IOR
// (which is for yellow light, 587.6 nm; green, k = 1, uses it as it is), when blue
// and red are `spread` apart. From Cauchy's law, n = A + B / wavelength^2, at the
// wavelengths glass makers measure at: red 656.3 nm, blue 486.1 nm. Blue bends
// away from yellow more than twice as far as red does.
fn icIorOffset(spread: f32, k: i32) -> f32 {
  if (k == 0) { return -0.301 * spread; }
  if (k == 2) { return 0.699 * spread; }
  return 0.0;
}

// Unpolarised dielectric Fresnel. eta = n_transmitted / n_incident.
fn icFresnel(cosI: f32, eta: f32) -> f32 {
  let c = abs(cosI);
  let g2 = eta * eta - 1.0 + c * c;
  if (g2 <= 0.0) { return 1.0; }
  let g = sqrt(g2);
  let a = (g - c) / (g + c);
  let b = (c * (g + c) - 1.0) / (c * (g - c) + 1.0);
  return 0.5 * a * a * (1.0 + b * b);
}

//#endif
//#if GLASS
// How much of the lamp reaches point p of an object (1: all of it, 0: none), from
// the lamp's depth map: in shadow where another object reaches farther toward the
// lamp than `top`, the farthest this object itself reaches. (An object does not
// shadow itself: what its own glass does to the light is followed through it.)
// Four samples soften the edge.
fn icLampVisible(p: vec3f, top: f32) -> f32 {
  let v = p - gLightC;
  let uv = vec2f(0.5 + 0.5 * dot(v, gLightR), 0.5 - 0.5 * dot(v, gLightU));
  if (any(uv <= vec2f(0.0)) || any(uv >= vec2f(1.0))) { return 1.0; }
  let dims = vec2i(textureDimensions(icLampDepthTex));
  let at = uv * vec2f(dims);
  var lit = 0.0;
  for (var k = 0; k < 4; k++) {
    let o = vec2f(select(-0.75, 0.75, (k & 1) == 1), select(-0.75, 0.75, (k & 2) == 2));
    let texel = clamp(vec2i(at + o), vec2i(0), dims - 1);
    let reach = textureLoad(icLampDepthTex, texel, 0).r - 100.0;
    lit += select(1.0, 0.0, reach > top);
  }
  return lit * 0.25;
}

// Share of a back face that acts as exit glass (a smooth ramp over the distance
// the light travelled inside).
fn icExitFactor(x: f32) -> f32 {
  if (x < 0.9864) {
    let a = x / 0.9864;
    return a * a * a / 6.0;
  }
  if (x < 1.0) {
    let f = (1.0 - x) / 0.0136;
    return (1.0 - f) * (1.0 - f) * (1.0 - f) / 6.0 + f * f * f * 0.5 - f * f + 2.0 / 3.0;
  }
  return 5.0 / 6.0;
}

// What a ray sees after leaving its object. Ground points are read from the
// picture of what is behind this object (ground plus the glass drawn before it)
// at the pixel where that point appears; off screen it falls back to the ground
// alone. Everything else is the studio.
//
// Before that, a refracted ray is followed out in growing steps, looking for other
// glass in that picture (which also holds how far away each pixel is): when the
// ray passes just behind something drawn there, that is what it sees. So glass
// shows through glass along the bent ray, not only straight behind.
fn icLook(ro: vec3f, rd: vec3f, throughGlass: bool) -> vec3f {
  if (throughGlass && gSteps > 0.5 && gMirror < 0.5) {
    var tEnd = 1e9;   // no farther than the floor
    if (gGroundOn > 0.5 && rd.y < -1e-5) { tEnd = max(-ro.y / rd.y, 0.0); }
    var t = 0.2 * gReach;
    for (var i = 0; i < i32(gSteps); i++) {
      if (t >= tEnd) { break; }
      let p = ro + rd * t;
      let clip = gViewProj * vec4f(p, 1.0);
      if (clip.w < 0.01) { break; }
      let ndc = clip.xy / clip.w;
      if (abs(ndc.x) > 0.98 || abs(ndc.y) > 0.98) { break; }
      let there = textureSampleLevel(icBehindTex, icBehindTex_sampler, vec2f(ndc.x, -ndc.y) * 0.5 + 0.5, 0.0);
      // Behind what is drawn there, but not by more than an object's depth: a hit.
      let behind = length(p - gEye) - there.a;
      if (behind > 0.0 && behind < 1.5 * gReach) { return there.rgb; }
      t *= 1.6;
    }
  }
  if (gGroundOn > 0.5 && rd.y < -1e-5) {
    let t = max(-ro.y / rd.y, 0.0);
    let g = ro + rd * t;
    let clip = gViewProj * vec4f(g, 1.0);
    if (throughGlass && gMirror < 0.5 && clip.w > 0.01) {
      let ndc = clip.xy / clip.w;
      if (abs(ndc.x) < 0.98 && abs(ndc.y) < 0.98) {
        return textureSampleLevel(icBehindTex, icBehindTex_sampler, vec2f(ndc.x, -ndc.y) * 0.5 + 0.5, 0.0).rgb;
      }
    }
  }
  return icScene(ro, rd);
}
//#endif
