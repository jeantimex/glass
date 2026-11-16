// The diamond: a round brilliant cut (table, 8 star, 8 kite and 16 upper girdle
// facets on the crown; a 16-sided girdle; 16 lower girdle and 8 main facets on the
// pavilion, down to a point). Everything that needs its shape - the drawn mesh,
// the facet planes the shader follows light through, the physics hull - comes
// from here, for a stone of size 1 (its girdle corners are GIRDLE from the axis).
import * as THREE from 'three/webgpu';

/** Girdle radius at size 1 (so the stone is as wide as the simpler cut it replaced). */
const GIRDLE = 1.0823922;
const TABLE = 0.56; // table radius, as a fraction of the girdle's
const CROWN = Math.tan(THREE.MathUtils.degToRad(34.5)); // slope of the kite facets
const PAVILION = Math.tan(THREE.MathUtils.degToRad(40.75)); // slope of the pavilion mains
const HALF_GIRDLE = 0.03; // half the girdle's thickness
const STAR = 0.759; // how far out the star facets reach
const LOWER = 0.25; // how far in the lower girdle facets reach

type V = [number, number, number];
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

/** The plane through three points, facing away from the stone's middle: nx, ny, nz, offset. */
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

/** Facet planes at size 1: outward normal and distance from the centre, 4 numbers each. */
export const DIAMOND_PLANES: number[] = (() => {
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
export const DIAMOND_PLANE_COUNT = DIAMOND_PLANES.length / 4;

/** Every corner of the stone at size 1. */
const CORNERS: V[] = (() => {
  const corners: V[] = [culet];
  for (let k = 0; k < 8; k++) corners.push(tableCorner(k), starTip(k), lowerTip(k));
  for (let j = 0; j < 16; j++) corners.push(girdleTop(j), girdleBottom(j));
  return corners;
})();

// The planes only describe the stone if it bulges outward everywhere.
for (const c of CORNERS) {
  for (let p = 0; p < DIAMOND_PLANE_COUNT; p++) {
    const d = DIAMOND_PLANES[p * 4] * c[0] + DIAMOND_PLANES[p * 4 + 1] * c[1] + DIAMOND_PLANES[p * 4 + 2] * c[2] - DIAMOND_PLANES[p * 4 + 3];
    if (d > 1e-6) console.warn(`diamond: corner ${c} is ${d.toFixed(4)} outside facet ${p}`);
  }
}

/** The facet planes as WGSL constants, for the shaders. */
export function diamondWgsl(prefix: string): string {
  const rows = Array.from({ length: DIAMOND_PLANE_COUNT }, (_, p) => `vec4f(${DIAMOND_PLANES.slice(p * 4, p * 4 + 4).map((x) => x.toFixed(7)).join(', ')})`);
  return `const ${prefix}_N = ${DIAMOND_PLANE_COUNT};\nconst ${prefix} = array<vec4f, ${DIAMOND_PLANE_COUNT}>(\n  ${rows.join(',\n  ')});`;
}

/**
 * A coarser convex shape for collisions (x, y, z, ...): the table's corners, every
 * other girdle corner top and bottom, and the point.
 */
export function diamondPoints(size: number): number[] {
  const points: number[] = [];
  for (let k = 0; k < 8; k++) points.push(...tableCorner(k), ...girdleTop(2 * k), ...girdleBottom(2 * k));
  points.push(...culet);
  return points.map((x) => x * size);
}

/** How far below its centre the facet it rests on is, at size 1 (a pavilion main). */
export const DIAMOND_REST = DIAMOND_PLANES[4 * 7 + 3];
/** That facet's tilt: the stone leans this far (radians) from upright when lying on it. */
export const DIAMOND_LEAN = Math.atan2(DIAMOND_PLANES[4 * 7], -DIAMOND_PLANES[4 * 7 + 1]);

/** The size-1 stone as flat-shaded triangles, one fan per facet. */
export function diamondGeometry(): THREE.BufferGeometry {
  const positions: number[] = [];
  const normals: number[] = [];
  for (let p = 0; p < DIAMOND_PLANE_COUNT; p++) {
    const n = new THREE.Vector3(DIAMOND_PLANES[p * 4], DIAMOND_PLANES[p * 4 + 1], DIAMOND_PLANES[p * 4 + 2]);
    const onFacet = CORNERS.filter((c) => Math.abs(n.x * c[0] + n.y * c[1] + n.z * c[2] - DIAMOND_PLANES[p * 4 + 3]) < 1e-5).map((c) => new THREE.Vector3(...c));
    if (onFacet.length < 3) continue;
    // Order the facet's corners anticlockwise seen from outside.
    const middle = onFacet.reduce((sum, c) => sum.add(c), new THREE.Vector3()).divideScalar(onFacet.length);
    const right = onFacet[0].clone().sub(middle).normalize();
    const up = n.clone().cross(right);
    const angle = (c: THREE.Vector3) => Math.atan2(c.clone().sub(middle).dot(up), c.clone().sub(middle).dot(right));
    onFacet.sort((a, b) => angle(a) - angle(b));
    for (let i = 1; i + 1 < onFacet.length; i++) {
      for (const c of [onFacet[0], onFacet[i], onFacet[i + 1]]) {
        positions.push(c.x, c.y, c.z);
        normals.push(n.x, n.y, n.z);
      }
    }
  }
  const geometry = new THREE.BufferGeometry();
  geometry.setAttribute('position', new THREE.Float32BufferAttribute(positions, 3));
  geometry.setAttribute('normal', new THREE.Float32BufferAttribute(normals, 3));
  return geometry;
}
