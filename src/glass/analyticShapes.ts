// three.js built-in geometries whose solids have exact formulas in tsl/shapes.ts: from a
// geometry's type and parameters, the shape kind and size MeshGlassMaterial traces exactly
// (no baked volume, no estimate). Anything else (open, partial, or not a built-in solid)
// returns null and the material falls back to its other exits.
import * as THREE from 'three/webgpu';
import { KIND } from './objects';

export interface AnalyticShape {
  /** Shape id in tsl/shapes.ts (`KIND`). */
  kind: number;
  /** Its size in the shapes' conventions (see the table in tsl/shapes.ts). */
  size: [number, number, number];
  /**
   * The shape's frame relative to the geometry: 'y' when its axis is the geometry's y axis,
   * 'z' when three.js builds it around z (the torus), which tsl/shapes.ts has around y.
   */
  axis: 'y' | 'z';
  /** The diamond's facet planes (the cut), for KIND.diamond (default: the round brilliant). */
  diamondPlanes?: number[];
  /**
   * A unit capsule or torus that a scale must not stretch (studio mode): its caps, or its tube,
   * keep their round section, as the renderer draws them. The scale along the geometry's axes
   * is the object's size (capsule: radius, half length; torus: ring radius, tube radius).
   */
  deform?: 'capsule' | 'torus';
}

const FULL_TURN = Math.PI * 2;
const near = (a: number, b: number) => Math.abs(a - b) <= 1e-6 * Math.max(1, Math.abs(a), Math.abs(b));

/** The exact solid a built-in geometry describes, or null if there is none to trace. */
export function analyticShape(geometry: THREE.BufferGeometry | null | undefined): AnalyticShape | null {
  if (!geometry) return null;
  const p = (geometry as { parameters?: Record<string, number | boolean> }).parameters;
  if (!p) return null;
  const num = (name: string, fallback: number) => (typeof p[name] === 'number' ? (p[name] as number) : fallback);
  switch (geometry.type) {
    case 'SphereGeometry': {
      const whole =
        near(num('phiStart', 0), 0) &&
        near(num('phiLength', FULL_TURN), FULL_TURN) &&
        near(num('thetaStart', 0), 0) &&
        near(num('thetaLength', Math.PI), Math.PI);
      return whole ? { kind: KIND.sphere, size: [num('radius', 1), 0, 0], axis: 'y' } : null;
    }
    case 'IcosahedronGeometry': {
      // Subdivided enough to stand for a sphere (as the instanced renderer's spheres do);
      // coarser ones are faceted solids, traced through their volume instead.
      return num('detail', 0) >= 3 ? { kind: KIND.sphere, size: [num('radius', 1), 0, 0], axis: 'y' } : null;
    }
    case 'OctahedronGeometry':
      return num('detail', 0) === 0 ? { kind: KIND.octahedron, size: [num('radius', 1), 0, 0], axis: 'y' } : null;
    case 'BoxGeometry':
      return {
        kind: KIND.box,
        size: [num('width', 1) / 2, num('height', 1) / 2, num('depth', 1) / 2],
        axis: 'y',
      };
    case 'CylinderGeometry':
    case 'ConeGeometry': {
      if (p.openEnded === true || !near(num('thetaLength', FULL_TURN), FULL_TURN)) return null;
      const top = geometry.type === 'ConeGeometry' ? 0 : num('radiusTop', 1);
      const bottom = geometry.type === 'ConeGeometry' ? num('radius', 1) : num('radiusBottom', 1);
      const halfHeight = num('height', 1) / 2;
      if (near(top, bottom)) return { kind: KIND.cylinder, size: [bottom, halfHeight, bottom], axis: 'y' };
      if (top === 0) return { kind: KIND.cone, size: [bottom, halfHeight, bottom], axis: 'y' };
      return null; // a truncated cone: no exact formula yet
    }
    case 'CapsuleGeometry':
      // `height` is the length of the straight middle section.
      return {
        kind: KIND.capsule,
        size: [num('radius', 1), num('height', 1) / 2, num('radius', 1)],
        axis: 'y',
      };
    case 'TorusGeometry':
      return near(num('arc', FULL_TURN), FULL_TURN)
        ? { kind: KIND.torus, size: [num('radius', 1), num('tube', 0.4), num('tube', 0.4)], axis: 'z' }
        : null;
    default:
      return null;
  }
}
