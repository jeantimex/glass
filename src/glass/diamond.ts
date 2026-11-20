// Diamond cuts: convex polyhedra defined by their outward facet planes.
// Everything that needs a stone's shape — the drawn mesh, the facet planes the
// shader follows light through, and how the stone rests on the floor — comes
// from here, for a stone of size 1.
import * as THREE from 'three/webgpu';

export type DiamondCut = 'round' | 'princess' | 'emerald' | 'asscher' | 'cushion' | 'radiant';

export interface DiamondCutSpec {
  label: string;
  /** Outward facet planes at size 1: nx, ny, nz, offset (4 numbers per plane). */
  planes: number[];
  planeCount: number;
  /** How far below its centre the resting pavilion facet lies, at size 1. */
  rest: number;
  /** Tilt (radians) around Z so that resting facet lies flat on the floor. */
  lean: number;
}

type V = [number, number, number];

/** Outward plane through three points: [nx, ny, nz, offset] with offset > 0. */
function planeThrough(a: V, b: V, c: V): number[] {
  const u = [b[0] - a[0], b[1] - a[1], b[2] - a[2]];
  const v = [c[0] - a[0], c[1] - a[1], c[2] - a[2]];
  let n = [u[1] * v[2] - u[2] * v[1], u[2] * v[0] - u[0] * v[2], u[0] * v[1] - u[1] * v[0]];
  const length = Math.hypot(n[0], n[1], n[2]);
  n = n.map((x) => x / length);
  let d = n[0] * a[0] + n[1] * a[1] + n[2] * a[2];
  if (d < 0) {
    n = n.map((x) => -x);
    d = -d;
  }
  return [n[0], n[1], n[2], d];
}

/** Outward plane with horizontal direction (hx, hz), vertical slope angle (deg), passing through (px, py, pz). */
function slopedPlane(hx: number, hz: number, angleDeg: number, px: number, py: number, pz: number, upward: boolean): number[] {
  const hLen = Math.hypot(hx, hz);
  const rad = THREE.MathUtils.degToRad(angleDeg);
  const sinA = Math.sin(rad);
  const cosA = Math.cos(rad);
  // Crown facets face outward & upward (ny = +cosA); pavilion facets face outward & downward (ny = -cosA).
  const nx = (hx / hLen) * sinA;
  const ny = upward ? cosA : -cosA;
  const nz = (hz / hLen) * sinA;
  const d = nx * px + ny * py + nz * pz;
  return [nx, ny, nz, d];
}

/** Find a clean resting pavilion facet (facing +X, -Y with nz ≈ 0) for floor placement. */
function restingFacet(planes: number[]): { rest: number; lean: number } {
  const count = planes.length / 4;
  let best = -1;
  let bestScore = -1e9;
  for (let p = 0; p < count; p++) {
    const nx = planes[p * 4];
    const ny = planes[p * 4 + 1];
    const nz = planes[p * 4 + 2];
    if (ny >= -0.25 || nx <= 0.1) continue;
    // Prefer an active pavilion main facet aligned with the +X/-Y plane (|nz| ≈ 0).
    const score = -Math.abs(nz) * 10 - Math.abs(ny + 0.76) - planes[p * 4 + 3] * 0.2;
    if (score > bestScore) {
      bestScore = score;
      best = p;
    }
  }
  if (best < 0) return { rest: 0.7, lean: 0 };
  return {
    rest: planes[best * 4 + 3],
    lean: Math.atan2(planes[best * 4], -planes[best * 4 + 1]),
  };
}

// ------------------------------------------------ 1. Round Brilliant ---
const ROUND_PLANES: number[] = (() => {
  const GIRDLE = 1.0823922;
  const TABLE = 0.56;
  const CROWN = Math.tan(THREE.MathUtils.degToRad(34.5));
  const PAVILION = Math.tan(THREE.MathUtils.degToRad(40.75));
  const HALF_GIRDLE = 0.03;
  const STAR = 0.759;
  const LOWER = 0.25;
  const at = (radius: number, y: number, degrees: number): V => {
    const a = THREE.MathUtils.degToRad(degrees);
    return [radius * Math.cos(a) * GIRDLE, y * GIRDLE, radius * Math.sin(a) * GIRDLE];
  };
  const TABLE_Y = HALF_GIRDLE + (1 - TABLE) * CROWN;
  const COS_HALF = Math.cos(Math.PI / 8);
  const tableCorner = (k: number) => at(TABLE, TABLE_Y, 45 * k);
  const girdleTop = (j: number) => at(1, HALF_GIRDLE, 22.5 * j);
  const girdleBottom = (j: number) => at(1, -HALF_GIRDLE, 22.5 * j);
  const starTip = (k: number) => at(STAR, HALF_GIRDLE + (1 - STAR * COS_HALF) * CROWN, 45 * k + 22.5);
  const lowerTip = (k: number) => at(LOWER, -HALF_GIRDLE - (1 - LOWER * COS_HALF) * PAVILION, 45 * k + 22.5);
  const culet: V = [0, (-HALF_GIRDLE - PAVILION) * GIRDLE, 0];

  const planes: number[] = [0, 1, 0, TABLE_Y * GIRDLE];
  for (let k = 0; k < 8; k++) {
    planes.push(...planeThrough(tableCorner(k), starTip(k), girdleTop(2 * k))); // kite
    planes.push(...planeThrough(tableCorner(k), tableCorner(k + 1), starTip(k))); // star
    planes.push(...planeThrough(starTip(k), girdleTop(2 * k), girdleTop(2 * k + 1))); // upper girdle
    planes.push(...planeThrough(starTip(k), girdleTop(2 * k + 1), girdleTop(2 * k + 2)));
    planes.push(...planeThrough(lowerTip(k), girdleBottom(2 * k), girdleBottom(2 * k + 1))); // lower girdle
    planes.push(...planeThrough(lowerTip(k), girdleBottom(2 * k + 1), girdleBottom(2 * k + 2)));
    planes.push(...planeThrough(girdleBottom(2 * k), culet, lowerTip(k))); // pavilion main
  }
  for (let j = 0; j < 16; j++) planes.push(...planeThrough(girdleTop(j), girdleTop(j + 1), girdleBottom(j))); // girdle
  return planes;
})();

// ------------------------------------------------ 2. Princess Cut ---
const PRINCESS_PLANES: number[] = (() => {
  const W = 0.92;
  const HG = 0.032;
  const TABLE_Y = HG + 0.24;
  const planes: number[] = [0, 1, 0, TABLE_Y]; // table
  // 4 square girdle faces
  planes.push(1, 0, 0, W, -1, 0, 0, W, 0, 0, 1, W, 0, 0, -1, W);
  // 4 main crown bezels & 4 main pavilion facets
  for (const [hx, hz] of [[1, 0], [-1, 0], [0, 1], [0, -1]]) {
    planes.push(...slopedPlane(hx, hz, 35, hx * W, HG, hz * W, true));
    planes.push(...slopedPlane(hx, hz, 44, hx * W, -HG, hz * W, false));
  }
  // 4 diagonal corner crown & pavilion facets (bright corner chevrons)
  for (const sx of [-1, 1]) {
    for (const sz of [-1, 1]) {
      planes.push(...slopedPlane(sx, sz, 29, sx * W * 0.94, HG, sz * W * 0.94, true));
      planes.push(...slopedPlane(sx, sz, 41, sx * W * 0.96, -HG, sz * W * 0.96, false));
      // 8 upper-girdle & 8 lower-girdle chevron facets
      planes.push(...slopedPlane(sx * 2, sz, 41, sx * W, HG, sz * W * 0.55, true));
      planes.push(...slopedPlane(sx, sz * 2, 41, sx * W * 0.55, HG, sz * W, true));
      planes.push(...slopedPlane(sx * 2, sz, 48, sx * W, -HG, sz * W * 0.55, false));
      planes.push(...slopedPlane(sx, sz * 2, 48, sx * W * 0.55, -HG, sz * W, false));
    }
  }
  return planes;
})();

// ---------------------------------------- Helper for step cuts (Emerald & Asscher) ---
function makeStepCut(wx: number, wz: number, clip: number, tableY: number, crownSteps: [number, number][], pavSteps: [number, number][]): number[] {
  const HG = 0.032;
  const planes: number[] = [0, 1, 0, tableY]; // table
  // Distance from origin to the 45-degree clipped corner plane
  const cornerD = (wx + wz - clip) * Math.SQRT1_2;
  // 8 girdle planes (4 sides + 4 clipped corners)
  planes.push(1, 0, 0, wx, -1, 0, 0, wx, 0, 0, 1, wz, 0, 0, -1, wz);
  for (const sx of [-1, 1]) {
    for (const sz of [-1, 1]) {
      planes.push(sx * Math.SQRT1_2, 0, sz * Math.SQRT1_2, cornerD);
    }
  }
  // Crown & pavilion terraced steps
  const addSteps = (steps: [number, number][], upward: boolean) => {
    for (const [angleDeg, yStart] of steps) {
      const y = upward ? HG + yStart : -HG - yStart;
      // Compute inset at this tier from previous slope
      const inset = yStart * 0.85;
      const sxDist = Math.max(wx - inset, 0.05);
      const szDist = Math.max(wz - inset, 0.05);
      const scDist = Math.max(cornerD - inset * 0.95, 0.05);
      planes.push(...slopedPlane(1, 0, angleDeg, sxDist, y, 0, upward));
      planes.push(...slopedPlane(-1, 0, angleDeg, -sxDist, y, 0, upward));
      planes.push(...slopedPlane(0, 1, angleDeg, 0, y, szDist, upward));
      planes.push(...slopedPlane(0, -1, angleDeg, 0, y, -szDist, upward));
      for (const sx of [-1, 1]) {
        for (const sz of [-1, 1]) {
          const cx = sx * scDist * Math.SQRT1_2;
          const cz = sz * scDist * Math.SQRT1_2;
          planes.push(...slopedPlane(sx, sz, angleDeg, cx, y, cz, upward));
        }
      }
    }
  };
  addSteps(crownSteps, true);
  addSteps(pavSteps, false);
  return planes;
}

// ------------------------------------------------ 3. Emerald Cut ---
const EMERALD_PLANES: number[] = makeStepCut(
  0.78, // half-width X
  1.10, // half-length Z (~1.4:1 classic emerald ratio)
  0.34, // clipped corner width
  0.29, // table height
  [[52, 0.0], [38, 0.09], [24, 0.17]], // 3 crown steps
  [[57, 0.0], [45, 0.24], [34, 0.44]], // 3 pavilion steps
);

// ------------------------------------------------ 4. Asscher Cut ---
const ASSCHER_PLANES: number[] = makeStepCut(
  0.90, // square X
  0.90, // square Z
  0.48, // wider Art Deco clipped corners
  0.35, // taller crown, smaller table
  [[55, 0.0], [41, 0.11], [27, 0.21]], // 3 high crown steps
  [[60, 0.0], [48, 0.28], [36, 0.52]], // 3 deep pavilion steps
);

// ------------------------------------------------ 5. Cushion Cut ---
const CUSHION_PLANES: number[] = (() => {
  const W = 0.96;
  const HG = 0.032;
  const TABLE_Y = HG + 0.32;
  const planes: number[] = [0, 1, 0, TABLE_Y]; // table
  // Superellipse-like pillow radius as a function of azimuth angle
  const pillowR = (rad: number) => {
    const c = Math.abs(Math.cos(rad));
    const s = Math.abs(Math.sin(rad));
    // n = 3.2 superellipse gives the classic soft-cornered cushion outline
    return W / Math.pow(Math.pow(c, 3.2) + Math.pow(s, 3.2), 1 / 3.2);
  };
  // 16-sided pillow girdle + upper/lower girdle facets
  for (let j = 0; j < 16; j++) {
    const a = (j * Math.PI) / 8;
    const hx = Math.cos(a);
    const hz = Math.sin(a);
    const r = pillowR(a);
    planes.push(hx, 0, hz, r);
    planes.push(...slopedPlane(hx, hz, 41, hx * r, HG, hz * r, true));
    planes.push(...slopedPlane(hx, hz, 45, hx * r, -HG, hz * r, false));
  }
  // 8 crown mains, 8 star facets, and 8 pavilion mains
  for (let k = 0; k < 8; k++) {
    const a = (k * Math.PI) / 4;
    const hx = Math.cos(a);
    const hz = Math.sin(a);
    const r = pillowR(a);
    planes.push(...slopedPlane(hx, hz, 34, hx * r * 0.99, HG, hz * r * 0.99, true));
    planes.push(...slopedPlane(hx, hz, 40.5, hx * r * 0.99, -HG, hz * r * 0.99, false));
    const aStar = a + Math.PI / 8;
    const sx = Math.cos(aStar);
    const sz = Math.sin(aStar);
    const rs = pillowR(aStar);
    planes.push(...slopedPlane(sx, sz, 21, sx * rs * 0.72, TABLE_Y - 0.04, sz * rs * 0.72, true));
  }
  return planes;
})();

// ------------------------------------------------ 6. Radiant Cut ---
const RADIANT_PLANES: number[] = (() => {
  const WX = 0.80;
  const WZ = 1.06;
  const CLIP = 0.34;
  const HG = 0.032;
  const TABLE_Y = HG + 0.28;
  const cornerD = (WX + WZ - CLIP) * Math.SQRT1_2;
  const planes: number[] = [0, 1, 0, TABLE_Y]; // table
  // 8 clipped-rectangle girdle faces
  const dirs8: [number, number, number][] = [
    [1, 0, WX], [-1, 0, WX], [0, 1, WZ], [0, -1, WZ],
    [Math.SQRT1_2, Math.SQRT1_2, cornerD], [-Math.SQRT1_2, Math.SQRT1_2, cornerD],
    [Math.SQRT1_2, -Math.SQRT1_2, cornerD], [-Math.SQRT1_2, -Math.SQRT1_2, cornerD],
  ];
  for (const [hx, hz, dist] of dirs8) {
    planes.push(hx, 0, hz, dist);
    // Brilliant crown main + steep upper girdle + pavilion main + steep lower girdle
    planes.push(...slopedPlane(hx, hz, 34, hx * dist, HG, hz * dist, true));
    planes.push(...slopedPlane(hx, hz, 43, hx * dist, HG, hz * dist, true));
    planes.push(...slopedPlane(hx, hz, 41, hx * dist, -HG, hz * dist, false));
    planes.push(...slopedPlane(hx, hz, 49, hx * dist, -HG, hz * dist, false));
  }
  // 8 intermediate star & pavilion-chevron facets for blazing Radiant fire
  for (const sx of [-1, 1]) {
    for (const sz of [-1, 1]) {
      for (const [mx, mz] of [[2, 1], [1, 2]]) {
        const hx = sx * mx;
        const hz = sz * mz;
        const len = Math.hypot(hx, hz);
        const nx = hx / len;
        const nz = hz / len;
        // Support distance on the clipped rectangle along (nx, nz)
        const dRect = Math.min(WX / Math.max(Math.abs(nx), 1e-4), WZ / Math.max(Math.abs(nz), 1e-4), cornerD / Math.max((Math.abs(nx) + Math.abs(nz)) * Math.SQRT1_2, 1e-4));
        planes.push(...slopedPlane(nx, nz, 22, nx * dRect * 0.72, TABLE_Y - 0.035, nz * dRect * 0.72, true));
        planes.push(...slopedPlane(nx, nz, 45, nx * dRect * 0.98, -HG, nz * dRect * 0.98, false));
      }
    }
  }
  return planes;
})();

function makeSpec(label: string, planes: number[]): DiamondCutSpec {
  const { rest, lean } = restingFacet(planes);
  return { label, planes, planeCount: planes.length / 4, rest, lean };
}

export const DIAMOND_CUTS: Record<DiamondCut, DiamondCutSpec> = {
  round: makeSpec('Round Brilliant', ROUND_PLANES),
  princess: makeSpec('Princess', PRINCESS_PLANES),
  emerald: makeSpec('Emerald', EMERALD_PLANES),
  asscher: makeSpec('Asscher', ASSCHER_PLANES),
  cushion: makeSpec('Cushion', CUSHION_PLANES),
  radiant: makeSpec('Radiant', RADIANT_PLANES),
};

// Backwards-compatible exports for the default Round Brilliant cut.
export const DIAMOND_PLANES = DIAMOND_CUTS.round.planes;
export const DIAMOND_PLANE_COUNT = DIAMOND_CUTS.round.planeCount;
export const DIAMOND_REST = DIAMOND_CUTS.round.rest;
export const DIAMOND_LEAN = DIAMOND_CUTS.round.lean;

/**
 * Build a watertight flat-shaded BufferGeometry directly from a convex set of
 * outward planes (nx, ny, nz, d) by clipping each facet polygon against every
 * other half-space. Guarantees an exact match with the ray-traced shader planes.
 */
export function diamondGeometry(cut: DiamondCut = 'round'): THREE.BufferGeometry {
  const { planes, planeCount } = DIAMOND_CUTS[cut];
  const positions: number[] = [];
  const normals: number[] = [];

  for (let p = 0; p < planeCount; p++) {
    const nx = planes[p * 4];
    const ny = planes[p * 4 + 1];
    const nz = planes[p * 4 + 2];
    const d = planes[p * 4 + 3];
    const n = new THREE.Vector3(nx, ny, nz);
    // Build a right-handed basis (u, v, n) on plane p.
    const ref = Math.abs(ny) < 0.9 ? new THREE.Vector3(0, 1, 0) : new THREE.Vector3(1, 0, 0);
    const u = ref.clone().cross(n).normalize();
    const v = n.clone().cross(u);
    const c = n.clone().multiplyScalar(d);
    const R = 10;
    let poly: THREE.Vector3[] = [
      c.clone().addScaledVector(u, -R).addScaledVector(v, -R),
      c.clone().addScaledVector(u, R).addScaledVector(v, -R),
      c.clone().addScaledVector(u, R).addScaledVector(v, R),
      c.clone().addScaledVector(u, -R).addScaledVector(v, R),
    ];

    // Clip polygon against every other half-space q: n_q · x <= d_q.
    for (let q = 0; q < planeCount && poly.length >= 3; q++) {
      if (q === p) continue;
      const qx = planes[q * 4];
      const qy = planes[q * 4 + 1];
      const qz = planes[q * 4 + 2];
      const qd = planes[q * 4 + 3];
      const next: THREE.Vector3[] = [];
      for (let i = 0; i < poly.length; i++) {
        const a = poly[i];
        const b = poly[(i + 1) % poly.length];
        const da = qx * a.x + qy * a.y + qz * a.z - qd;
        const db = qx * b.x + qy * b.y + qz * b.z - qd;
        const aIn = da <= 1e-6;
        const bIn = db <= 1e-6;
        if (aIn) next.push(a);
        if (aIn !== bIn) {
          const t = da / (da - db);
          next.push(new THREE.Vector3(a.x + (b.x - a.x) * t, a.y + (b.y - a.y) * t, a.z + (b.z - a.z) * t));
        }
      }
      poly = next;
    }

    // Remove near-coincident vertices produced when multiple planes meet at one corner.
    const clean: THREE.Vector3[] = [];
    for (const vert of poly) {
      const prev = clean[clean.length - 1];
      if (!prev || prev.distanceToSquared(vert) > 1e-10) clean.push(vert);
    }
    if (clean.length >= 2 && clean[0].distanceToSquared(clean[clean.length - 1]) <= 1e-10) clean.pop();
    if (clean.length < 3) continue;
    const ab = new THREE.Vector3();
    const ac = new THREE.Vector3();
    const cross = new THREE.Vector3();
    for (let i = 1; i + 1 < clean.length; i++) {
      ab.subVectors(clean[i], clean[0]);
      ac.subVectors(clean[i + 1], clean[0]);
      if (cross.crossVectors(ab, ac).dot(n) <= 1e-10) continue;
      for (const vert of [clean[0], clean[i], clean[i + 1]]) {
        positions.push(vert.x, vert.y, vert.z);
        normals.push(nx, ny, nz);
      }
    }
  }

  const geometry = new THREE.BufferGeometry();
  geometry.setAttribute('position', new THREE.Float32BufferAttribute(positions, 3));
  geometry.setAttribute('normal', new THREE.Float32BufferAttribute(normals, 3));
  return geometry;
}
