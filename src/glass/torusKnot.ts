// The torus knot's exact shape, shared by its rasterized mesh (geometries.ts) and
// the offline bake of its signed-distance volume (scripts/bake-torusknot-sdf.ts),
// so the two always describe the same surface. Kept free of imports so Node can
// run the bake script straight from this TypeScript file.

/** `THREE.TorusKnotGeometry` parameters: knot radius, tube radius, and the (p, q) winding. */
export const TORUS_KNOT = { radius: 0.62, tube: 0.2, p: 2, q: 3 } as const;

/**
 * Point on the knot's centre curve at `u` (0 to 2πp), exactly as
 * `THREE.TorusKnotGeometry` computes it, before the unit transform below.
 */
export function torusKnotCurve(u: number): [number, number, number] {
  const { radius, p, q } = TORUS_KNOT;
  const quOverP = (q / p) * u;
  const cs = Math.cos(quOverP);
  return [
    radius * (2 + cs) * 0.5 * Math.cos(u),
    radius * (2 + cs) * 0.5 * Math.sin(u),
    radius * Math.sin(quOverP) * 0.5,
  ];
}

/**
 * The transform to unit size: `unit = (raw - (0, centerY, 0)) * scale`, which puts
 * the tube's lowest point at y = -1 and its highest at y = 1. Worked out from the
 * exact curve (a tube reaches exactly `tube` beyond its centre curve in every
 * direction), not from the mesh, so every tessellation shares the same frame.
 */
export const UNIT_TORUS_KNOT = (() => {
  const { p, tube } = TORUS_KNOT;
  let minY = Infinity;
  let maxY = -Infinity;
  const steps = 1 << 16;
  for (let i = 0; i < steps; i++) {
    const y = torusKnotCurve((i / steps) * Math.PI * 2 * p)[1];
    minY = Math.min(minY, y);
    maxY = Math.max(maxY, y);
  }
  minY -= tube;
  maxY += tube;
  return { centerY: (minY + maxY) * 0.5, scale: 2 / (maxY - minY) };
})();
