// Rigid-body physics on the GPU, for piles too large for Box3D.
//
// Every body is solved in parallel, one substep at a time:
//   integrate  move each body ahead by its velocity and gravity (a prediction)
//   clearGrid  empty the grid of cells
//   fillGrid   list each body in the cell its centre is in
//   markActive each moving body marks itself and the bodies it may touch
//   compact    list the marked bodies, and size the solve to them
//   solve      for each listed body, push it out of the ground and of the bodies
//              in the 27 cells around it, then work its velocity out from how far
//              it moved
//
// Only bodies that move, or that something moving may touch, are solved: one at
// rest with nothing moving near it would only be held where it is, so it is left
// where it is. Most of a pile is at rest long before the last of it settles.
//
// Two bodies touch where a sample point of one (its corners and edge middles, or
// the centres of the balls it is made of) is inside the other's exact shape.
//
// Shape ids and `size` are those of instanced.wgsl.

struct Params {
  count: u32,
  h: f32,     // length of a substep (s)
  cell: f32,  // width of a grid cell: at least the largest body's diameter
  pad: f32,
}

@group(0) @binding(0) var<uniform> P: Params;
// Per body: position (w = bounding radius), rotation (quaternion).
@group(0) @binding(1) var<storage, read_write> state: array<vec4f>;
// Per body: velocity (w = substeps spent at rest), angular velocity, predicted position (w = bounding radius),
// predicted rotation, position and rotation before the substep, size (w = shape
// id), inverse inertia (w = inverse mass).
@group(0) @binding(2) var<storage, read_write> work: array<vec4f>;
@group(0) @binding(3) var<storage, read_write> gridCount: array<atomic<u32>>;
@group(0) @binding(4) var<storage, read_write> gridCells: array<u32>;
// Tables of the shapes, in a buffer (a const array looked up with a varying index
// may be copied into every invocation's own memory first).
struct Tables {
  oct8: array<vec4f, 8>,                  // 8 directions round a circle (xy)
  // The shapes made from meshes (Suzanne, then the dragon) collide as the convex
  // shape spanned by their extreme points along 26 directions: those points
  // (xyz; Suzanne's, then the dragon's) and the 26 planes bounding each.
  meshPoints: array<vec4f, MESH_POINTS_N>,
  meshPlanes: array<vec4f, 52>,
}
@group(0) @binding(5) var<uniform> T: Tables;
// The bodies to solve this substep: a mark per body, the list of them, and how
// many. Then the solve's workgroup counts (x, y, z), which it is dispatched with:
// bound only for sizeSolve, as a buffer cannot be written by the pass it sizes.
@group(0) @binding(6) var<storage, read_write> activeMark: array<atomic<u32>>;
@group(0) @binding(7) var<storage, read_write> activeList: array<u32>;
@group(0) @binding(8) var<storage, read_write> activeCount: atomic<u32>;
@group(1) @binding(0) var<storage, read_write> solveSize: array<u32, 3>;

const GRID = vec3u(64u, 32u, 64u);   // cells; the grid repeats beyond that
const SLOTS = 32u;                   // bodies listed per cell
const GRAVITY = vec3f(0.0, -9.81, 0.0);
const FRICTION_STATIC = 0.9;
const FRICTION_SLIDE = 0.7;
// A body slower than this (m/s and rad/s) for this many substeps has come to rest.
const REST_SPEED = 0.16;
const REST_SPIN = 1.2;
const REST_AFTER = 20.0;

/*SUZANNE*/

// ------------------------------------------------------------- rotations ---
fn rot(q: vec4f, v: vec3f) -> vec3f {
  return v + 2.0 * cross(q.xyz, cross(q.xyz, v) + q.w * v);
}

fn rotInv(q: vec4f, v: vec3f) -> vec3f {
  return v + 2.0 * cross(-q.xyz, cross(-q.xyz, v) + q.w * v);
}

fn quatMul(a: vec4f, b: vec4f) -> vec4f {
  return vec4f(a.w * b.xyz + b.w * a.xyz + cross(a.xyz, b.xyz), a.w * b.w - dot(a.xyz, b.xyz));
}

// ----------------------------------------------------------------- shapes ---

// Distance from p (in the shape's own space) to its surface, negative inside.
fn sdf(kind: i32, p: vec3f, s: vec3f) -> f32 {
  switch kind {
    case 0 {
      let q = abs(p) - s;
      return length(max(q, vec3f(0.0))) + min(max(q.x, max(q.y, q.z)), 0.0);
    }
    case 1 { return length(p) - s.x; }
    case 2 {
      let d = abs(vec2f(length(p.xz), p.y)) - vec2f(s.x, s.y);
      return min(max(d.x, d.y), 0.0) + length(max(d, vec2f(0.0)));
    }
    case 3 {
      return length(vec3f(p.x, p.y - clamp(p.y, -s.y, s.y), p.z)) - s.x;
    }
    case 4 {
      // Cone: apex at y = s.y, base of radius s.x at y = -s.y.
      let q = vec2f(s.x, -2.0 * s.y);
      let w = vec2f(length(p.xz), p.y - s.y);
      let a = w - q * clamp(dot(w, q) / dot(q, q), 0.0, 1.0);
      let b = w - q * vec2f(clamp(w.x / q.x, 0.0, 1.0), 1.0);
      let d = min(dot(a, a), dot(b, b));
      let side = max(-(w.x * q.y - w.y * q.x), -(w.y - q.y));
      return sqrt(d) * sign(side);
    }
    case 5 { return length(vec2f(length(p.xz) - s.x, p.y)) - s.y; }
    case 6 { return (abs(p.x) + abs(p.y) + abs(p.z) - s.x) * 0.57735027; }
    case 7 {
      return max(max(-p.y - 0.5 * s.x, 0.8660254 * abs(p.x) + 0.5 * p.y - 0.5 * s.x), abs(p.z) - s.z);
    }
    case 8 {
      let q = abs(p);
      return max(max(q.x * 0.8660254 + q.z * 0.5, q.z) - s.x, q.y - s.y);
    }
    case 9, 11 {
      let m = meshOf(kind);
      var d = -1e9;
      for (var k = 0; k < 26; k++) {
        let pl = T.meshPlanes[m * 26 + k];
        d = max(d, dot(p, pl.xyz) - pl.w * s.x);
      }
      return d;
    }
    default {
      // Diamond, roughly: table, then 8 girdle, crown and pavilion facets.
      var d = p.y - 0.359 * s.x;
      for (var k = 0; k < 8; k++) {
        let c = T.oct8[k].xy.x * p.x + T.oct8[k].xy.y * p.z;
        d = max(d, c - 1.0824 * s.x);
        d = max(d, 0.5664 * c + 0.8241 * p.y - 0.6399 * s.x);
        d = max(d, 0.6528 * c - 0.7576 * p.y - 0.7312 * s.x);
      }
      return d;
    }
  }
}

// The same distance, but only when it is below `limit`: otherwise any value of at
// least `limit`. For the shapes made of planes, one plane that p is far enough
// outside of settles that, without looking at the rest. (Most sample points tested
// against a neighbour are outside it, so this skips most of the work, and finds
// exactly the same contacts.)
fn sdfBelow(kind: i32, p: vec3f, s: vec3f, limit: f32) -> f32 {
  if (kind == 9 || kind == 11) {
    let m = meshOf(kind);
    var d = -1e9;
    for (var k = 0; k < 26; k++) {
      let pl = T.meshPlanes[m * 26 + k];
      d = max(d, dot(p, pl.xyz) - pl.w * s.x);
      if (d >= limit) { return d; }
    }
    return d;
  }
  if (kind == 10) {
    var d = p.y - 0.359 * s.x;
    if (d >= limit) { return d; }
    for (var k = 0; k < 8; k++) {
      let c = T.oct8[k].xy.x * p.x + T.oct8[k].xy.y * p.z;
      d = max(d, c - 1.0824 * s.x);
      d = max(d, 0.5664 * c + 0.8241 * p.y - 0.6399 * s.x);
      d = max(d, 0.6528 * c - 0.7576 * p.y - 0.7312 * s.x);
      if (d >= limit) { return d; }
    }
    return d;
  }
  return sdf(kind, p, s);
}

// Outward direction at p.
fn sdfNormal(kind: i32, p: vec3f, s: vec3f, d: f32) -> vec3f {
  if (kind == 9 || kind == 11) {
    // A mesh shape is bounded by planes: the way out is straight out through the
    // one p is nearest to crossing (exact, and far cheaper than probing round p).
    let m = meshOf(kind);
    var best = -1e9;
    var n = vec3f(0.0, 1.0, 0.0);
    for (var k = 0; k < 26; k++) {
      let pl = T.meshPlanes[m * 26 + k];
      let dk = dot(p, pl.xyz) - pl.w * s.x;
      if (dk > best) {
        best = dk;
        n = pl.xyz;
      }
    }
    return n;
  }
  let e = 0.002;
  let g = vec3f(
    sdf(kind, p + vec3f(e, 0.0, 0.0), s) - d,
    sdf(kind, p + vec3f(0.0, e, 0.0), s) - d,
    sdf(kind, p + vec3f(0.0, 0.0, e), s) - d);
  let l = length(g);
  if (l < 1e-9) { return vec3f(0.0, 1.0, 0.0); }
  return g / l;
}

// Which of the mesh shapes (0: Suzanne, 1: the dragon).
fn meshOf(kind: i32) -> i32 {
  return select(0, 1, kind == 11);
}

fn sampleCount(kind: i32) -> i32 {
  switch kind {
    case 0 { return 20; }
    case 1 { return 1; }
    case 2 { return 24; }
    case 3 { return 3; }
    case 4 { return 17; }
    case 5 { return 12; }
    case 6 { return 18; }
    case 7 { return 15; }
    case 8 { return 18; }
    case 9 { return SUZ_POINTS_N; }
    case 11 { return DRAGON_POINTS_N; }
    default { return 17; }
  }
}

fn sign2(k: i32, bit: u32) -> f32 {
  return f32((k >> bit) & 1) * 2.0 - 1.0;
}

fn prismCorner(v: i32, t: f32) -> vec2f {
  if (v == 0) { return vec2f(0.0, t); }
  if (v == 1) { return vec2f(-0.8660254 * t, -0.5 * t); }
  return vec2f(0.8660254 * t, -0.5 * t);
}

// Sample point k of a shape: xyz in its own space, w the radius of the ball around it.
fn samplePoint(kind: i32, k: i32, s: vec3f) -> vec4f {
  switch kind {
    case 0 {
      if (k < 8) { return vec4f(s * vec3f(sign2(k, 0u), sign2(k, 1u), sign2(k, 2u)), 0.0); }
      let e = k - 8;
      let a = sign2(e, 0u);
      let b = sign2(e, 1u);
      let axis = e / 4;
      if (axis == 0) { return vec4f(s * vec3f(0.0, a, b), 0.0); }
      if (axis == 1) { return vec4f(s * vec3f(a, 0.0, b), 0.0); }
      return vec4f(s * vec3f(a, b, 0.0), 0.0);
    }
    case 1 { return vec4f(0.0, 0.0, 0.0, s.x); }
    case 2 {
      let c = T.oct8[k % 8].xy;
      return vec4f(s.x * c.x, f32(k / 8 - 1) * s.y, s.x * c.y, 0.0);
    }
    case 3 { return vec4f(0.0, f32(k - 1) * s.y, 0.0, s.x); }
    case 4 {
      if (k == 0) { return vec4f(0.0, s.y, 0.0, 0.0); }
      let c = T.oct8[(k - 1) % 8].xy;
      if (k < 9) { return vec4f(s.x * c.x, -s.y, s.x * c.y, 0.0); }
      return vec4f(0.5 * s.x * c.x, 0.0, 0.5 * s.x * c.y, 0.0);
    }
    case 5 {
      let a = f32(k) * 0.52359878;
      return vec4f(s.x * cos(a), 0.0, s.x * sin(a), s.y);
    }
    case 6 {
      if (k < 6) {
        var v = vec3f(0.0);
        v[k / 2] = sign2(k, 0u) * s.x;
        return vec4f(v, 0.0);
      }
      let e = k - 6;
      let a = sign2(e, 0u) * 0.5 * s.x;
      let b = sign2(e, 1u) * 0.5 * s.x;
      let pair = e / 4;
      if (pair == 0) { return vec4f(a, b, 0.0, 0.0); }
      if (pair == 1) { return vec4f(0.0, a, b, 0.0); }
      return vec4f(a, 0.0, b, 0.0);
    }
    case 7 {
      if (k < 6) { return vec4f(prismCorner(k % 3, s.x), sign2(k / 3, 0u) * s.z, 0.0); }
      if (k < 9) { return vec4f(prismCorner(k - 6, s.x), 0.0, 0.0); }
      let e = k - 9;
      let m = 0.5 * (prismCorner(e % 3, s.x) + prismCorner((e + 1) % 3, s.x));
      return vec4f(m, sign2(e / 3, 0u) * s.z, 0.0);
    }
    case 8 {
      let corner = s.x * 1.1547005;
      if (k < 12) {
        let a = f32(k / 2) * 1.04719755;
        return vec4f(corner * cos(a), sign2(k, 0u) * s.y, corner * sin(a), 0.0);
      }
      let a = f32(k - 12) * 1.04719755;
      return vec4f(corner * cos(a), 0.0, corner * sin(a), 0.0);
    }
    case 9 { return vec4f(T.meshPoints[k].xyz * s.x, 0.0); }
    case 11 { return vec4f(T.meshPoints[SUZ_POINTS_N + k].xyz * s.x, 0.0); }
    default {
      if (k == 16) { return vec4f(0.0, -0.966 * s.x, 0.0, 0.0); }
      let c = T.oct8[k % 8].xy;
      let r = select(1.0823922, 0.6061396, k < 8) * s.x;
      return vec4f(r * c.x, select(0.0, 0.359 * s.x, k < 8), r * c.y, 0.0);
    }
  }
}

// ------------------------------------------------------------------- grid ---
fn bucketOf(c: vec3i) -> u32 {
  let w = bitcast<vec3u>(c) & (GRID - vec3u(1u));
  return w.x + GRID.x * (w.y + GRID.y * w.z);
}

@compute @workgroup_size(64)
fn clearGrid(@builtin(global_invocation_id) id: vec3u) {
  if (id.x < GRID.x * GRID.y * GRID.z) { atomicStore(&gridCount[id.x], 0u); }
}

@compute @workgroup_size(64)
fn fillGrid(@builtin(global_invocation_id) id: vec3u) {
  let i = id.x;
  if (i >= P.count) { return; }
  let b = bucketOf(vec3i(floor(work[i * 8u + 2u].xyz / P.cell)));
  let slot = atomicAdd(&gridCount[b], 1u);
  if (slot < SLOTS) { gridCells[b * SLOTS + slot] = i; }
}

// -------------------------------------------------------------- integrate ---
@compute @workgroup_size(64)
fn integrate(@builtin(global_invocation_id) id: vec3u) {
  let i = id.x;
  if (i >= P.count) { return; }
  let b = i * 8u;
  let x = state[i * 2u];
  let q = state[i * 2u + 1u];
  let v = work[b].xyz + GRAVITY * P.h;
  let w = work[b + 1u].xyz;
  work[b + 4u] = x;
  work[b + 5u] = q;
  work[b + 2u] = vec4f(x.xyz + v * P.h, x.w);
  work[b + 3u] = normalize(q + 0.5 * P.h * quatMul(vec4f(w, 0.0), q));
}

// ------------------------------------------------------------------ solve ---
// The body being solved (A), the one it is tested against (B), and the push so far.
var<private> xA: vec3f;
var<private> qA: vec4f;
var<private> prevXA: vec3f;
var<private> prevQA: vec4f;
var<private> invIA: vec3f;
var<private> invMA: f32;
var<private> xB: vec3f;
var<private> qB: vec4f;
var<private> prevXB: vec3f;
var<private> prevQB: vec4f;
var<private> invIB: vec3f;
var<private> invMB: f32;   // 0: B is the ground
var<private> pushX: vec3f;
var<private> pushR: vec3f;
var<private> contacts: f32;

fn invInertiaA(v: vec3f) -> vec3f { return rot(qA, invIA * rotInv(qA, v)); }
fn invInertiaB(v: vec3f) -> vec3f { return rot(qB, invIB * rotInv(qB, v)); }

// How much a push along `n` at arm `r` moves that point, per unit of push.
fn responseA(r: vec3f, n: vec3f) -> f32 {
  let c = cross(r, n);
  return invMA + dot(c, invInertiaA(c));
}

fn responseB(r: vec3f, n: vec3f) -> f32 {
  if (invMB == 0.0) { return 0.0; }
  let c = cross(r, n);
  return invMB + dot(c, invInertiaB(c));
}

// A and B overlap by `depth` at world point c; n points the way A must go.
fn addContact(c: vec3f, n: vec3f, depthIn: f32) {
  let depth = min(depthIn, 0.08);
  let rA = c - xA;
  let rB = c - xB;
  let lambda = depth / (responseA(rA, n) + responseB(rB, n));
  pushX += n * (lambda * invMA);
  pushR += invInertiaA(cross(rA, n * lambda));

  // Friction: how far the two surfaces slid past each other during this substep.
  var slide = (xA - prevXA) + rA - rot(prevQA, rotInv(qA, rA));
  if (invMB != 0.0) { slide -= (xB - prevXB) + rB - rot(prevQB, rotInv(qB, rB)); }
  slide -= n * dot(slide, n);
  let len = length(slide);
  if (len > 1e-7) {
    let t = slide / len;
    var undo = len;                                         // sticking: undo all of it
    if (len > FRICTION_STATIC * depth) { undo = min(FRICTION_SLIDE * depth, len); }
    let lt = undo / (responseA(rA, t) + responseB(rB, t));
    pushX -= t * (lt * invMA);
    pushR -= invInertiaA(cross(rA, t * lt));
  }
  contacts += 1.0;
}

// ------------------------------------------------------- active bodies ---
@compute @workgroup_size(64)
fn clearActive(@builtin(global_invocation_id) id: vec3u) {
  if (id.x < P.count) { atomicStore(&activeMark[id.x], 0u); }
  if (id.x == 0u) { atomicStore(&activeCount, 0u); }
}

// A moving body (not held at rest) marks itself, and every body near enough that
// they may touch: those must be solved too, to be knocked or lose their support.
@compute @workgroup_size(64)
fn markActive(@builtin(global_invocation_id) id: vec3u) {
  let i = id.x;
  if (i >= P.count) { return; }
  let b = i * 8u;
  if (work[b].w > REST_AFTER) { return; }
  atomicStore(&activeMark[i], 1u);
  let predicted = work[b + 2u];
  let here = vec3i(floor(predicted.xyz / P.cell));
  for (var dz = -1; dz <= 1; dz++) {
    for (var dy = -1; dy <= 1; dy++) {
      for (var dx = -1; dx <= 1; dx++) {
        let bucket = bucketOf(here + vec3i(dx, dy, dz));
        let listed = min(atomicLoad(&gridCount[bucket]), SLOTS);
        for (var slot = 0u; slot < listed; slot++) {
          let j = gridCells[bucket * SLOTS + slot];
          let other = work[j * 8u + 2u];
          let apart = other.xyz - predicted.xyz;
          let reach = predicted.w + other.w;
          if (dot(apart, apart) <= reach * reach) { atomicStore(&activeMark[j], 1u); }
        }
      }
    }
  }
}

@compute @workgroup_size(64)
fn compact(@builtin(global_invocation_id) id: vec3u) {
  if (id.x < P.count && atomicLoad(&activeMark[id.x]) != 0u) {
    activeList[atomicAdd(&activeCount, 1u)] = id.x;
  }
}

@compute @workgroup_size(1)
fn sizeSolve() {
  solveSize[0] = atomicLoad(&activeCount); // a workgroup for each
  solveSize[1] = 1u;
  solveSize[2] = 1u;
}

// Each body is solved by a workgroup: its 64 threads share out the bodies it may
// touch (and its points against the ground), then add up what they found. One
// thread alone took as long as the most crowded body needed, however few bodies
// were left to solve.
var<workgroup> sumX: array<vec4f, 64>;   // push (xyz) and contacts (w) found by each thread
var<workgroup> sumR: array<vec4f, 64>;   // turning push (xyz)

@compute @workgroup_size(64)
fn solve(@builtin(workgroup_id) group: vec3u, @builtin(local_invocation_index) t: u32) {
  let i = activeList[group.x];
  let b = i * 8u;
  let predicted = work[b + 2u];
  xA = predicted.xyz;
  let boundA = predicted.w;
  qA = work[b + 3u];
  prevXA = work[b + 4u].xyz;
  prevQA = work[b + 5u];
  let shapeA = work[b + 6u];
  let sizeA = shapeA.xyz;
  let kindA = i32(shapeA.w);
  let countA = sampleCount(kindA);
  invIA = work[b + 7u].xyz;
  invMA = work[b + 7u].w;
  pushX = vec3f(0.0);
  pushR = vec3f(0.0);
  contacts = 0.0;

  // The ground (y = 0).
  if (xA.y < boundA) {
    invMB = 0.0;
    xB = vec3f(0.0);
    for (var k = i32(t); k < countA; k += 64) {
      let s = samplePoint(kindA, k, sizeA);
      let p = xA + rot(qA, s.xyz);
      let depth = s.w - p.y;
      if (depth > 0.0) { addContact(vec3f(p.x, p.y - s.w, p.z), vec3f(0.0, 1.0, 0.0), depth); }
    }
  }

  // The bodies in the 27 cells around this one: every place in those cells' lists
  // (27 x SLOTS of them), shared out among the threads.
  let cell = vec3i(floor(xA / P.cell));
  for (var place = t; place < 27u * SLOTS; place += 64u) {
        let around = i32(place / SLOTS);
        let slot = place % SLOTS;
        let bucket = bucketOf(cell + vec3i(around % 3 - 1, (around / 3) % 3 - 1, around / 9 - 1));
        if (slot >= min(atomicLoad(&gridCount[bucket]), SLOTS)) { continue; }
        {
          let j = gridCells[bucket * SLOTS + slot];
          if (j == i) { continue; }
          let o = j * 8u;
          let other = work[o + 2u];
          let apart = other.xyz - xA;
          let reach = boundA + other.w;
          if (dot(apart, apart) > reach * reach) { continue; }
          xB = other.xyz;
          qB = work[o + 3u];
          prevXB = work[o + 4u].xyz;
          prevQB = work[o + 5u];
          let shapeB = work[o + 6u];
          let sizeB = shapeB.xyz;
          let kindB = i32(shapeB.w);
          invIB = work[o + 7u].xyz;
          invMB = work[o + 7u].w;

          // A's sample points inside B.
          for (var k = 0; k < countA; k++) {
            let s = samplePoint(kindA, k, sizeA);
            let p = xA + rot(qA, s.xyz);
            let fromB = p - xB;
            let reachB = other.w + s.w;
            if (dot(fromB, fromB) > reachB * reachB) { continue; }
            let local = rotInv(qB, fromB);
            let d = sdfBelow(kindB, local, sizeB, s.w);
            if (d < s.w) {
              let n = rot(qB, sdfNormal(kindB, local, sizeB, d));
              addContact(p - n * s.w, n, s.w - d);
            }
          }
          // B's sample points inside A.
          let countB = sampleCount(kindB);
          for (var k = 0; k < countB; k++) {
            let s = samplePoint(kindB, k, sizeB);
            let p = xB + rot(qB, s.xyz);
            let fromA = p - xA;
            let reachA = boundA + s.w;
            if (dot(fromA, fromA) > reachA * reachA) { continue; }
            let local = rotInv(qA, fromA);
            let d = sdfBelow(kindA, local, sizeA, s.w);
            if (d < s.w) {
              let n = -rot(qA, sdfNormal(kindA, local, sizeA, d));
              addContact(p + n * s.w, n, s.w - d);
            }
          }
        }
  }

  // Add up what the threads found; the first one finishes the body off.
  sumX[t] = vec4f(pushX, contacts);
  sumR[t] = vec4f(pushR, 0.0);
  workgroupBarrier();
  for (var half = 32u; half > 0u; half >>= 1u) {
    if (t < half) {
      sumX[t] += sumX[t + half];
      sumR[t] += sumR[t + half];
    }
    workgroupBarrier();
  }
  if (t != 0u) { return; }
  pushX = sumX[0].xyz;
  contacts = sumX[0].w;
  pushR = sumR[0].xyz;

  // Apply the pushes. Each was sized as if it were the only one, so with several
  // their sum overshoots: scale it back.
  var x = xA;
  var q = qA;
  if (contacts > 0.0) {
    let scale = min(1.0, 1.5 / contacts);
    x += pushX * scale;
    q = normalize(q + 0.5 * quatMul(vec4f(pushR * scale, 0.0), q));
  }

  // Velocity is whatever actually happened over the substep.
  var v = (x - prevXA) / P.h;
  let dq = quatMul(q, vec4f(-prevQA.xyz, prevQA.w));
  var w = dq.xyz * (2.0 / P.h) * select(-1.0, 1.0, dq.w >= 0.0);
  // A little drag, and much more of it while touching (so things stop sliding and rolling).
  v *= 1.0 - select(0.08, 1.5, contacts > 0.0) * P.h;
  w *= 1.0 - select(0.6, 4.0, contacts > 0.0) * P.h;
  let speed = length(v);
  if (speed > 30.0) { v *= 30.0 / speed; }
  let spin = length(w);
  if (spin > 40.0) { w *= 40.0 / spin; }

  // Coming to rest: a body that has been touching something and hardly moving for
  // a while is held exactly where it is, until it is knocked or loses its support.
  // (Without this the pushes never quite cancel, and a settled pile keeps stirring.)
  var resting = work[b].w;   // substeps spent barely moving
  if (contacts > 0.0 && dot(v, v) < REST_SPEED * REST_SPEED && dot(w, w) < REST_SPIN * REST_SPIN) {
    resting = min(resting + 1.0, 10000.0);
  } else {
    resting = 0.0;
  }
  if (resting > REST_AFTER) {
    x = prevXA;
    q = prevQA;
    v = vec3f(0.0);
    w = vec3f(0.0);
  }

  state[i * 2u] = vec4f(x, boundA);
  state[i * 2u + 1u] = q;
  work[b] = vec4f(v, resting);
  work[b + 1u] = vec4f(w, 0.0);
}
