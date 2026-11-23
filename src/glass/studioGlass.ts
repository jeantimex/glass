// MeshGlassMaterial's studio mode: glass on a standard three.js mesh shaded with the instanced
// renderer's own glassNode, its floor and room, through the studio inputs that renderer
// exposes (`createInstancedGlass(...).studio`). Same shader, same inputs: the glass matches
// the renderer's pixel for pixel. The mesh's frame comes from its object (and instance)
// matrices: rotation, translation and a scale along the geometry's own axes, folded into the
// shape's size. A per-axis scale must keep the shape what it is: any for a box, equal across
// for a cylinder or cone, uniform for the rest (a stretched sphere is no longer a sphere).
/* eslint-disable @typescript-eslint/no-explicit-any -- TSL node types are too loose to spell out */
import * as THREE from 'three/webgpu';
import * as TSL from 'three/tsl';
import type { AnalyticShape } from './analyticShapes';
import { glassNode, type GlassFrame } from './tsl/glass';

const { abs, length, normalLocal, normalWorld, normalize, positionWorld, uniform, vec3, vec4 } = TSL as any;

/** What the renderer exposes as `studio`: the very nodes its own glass shading reads. */
export interface GlassStudio {
  glassInputs: Record<string, any>;
  sceneInputs: any;
  viewInputs: any;
  farScene: THREE.Scene;
  nearScene: THREE.Scene;
  /**
   * The renderer's own geometry for a shape kind (unit size, as finely divided as its close-ups;
   * the diamond in `cut`), declaring its solid in `userData.glassShape`: scale it to size.
   */
  shapeGeometry: (kind: number, cut?: string) => THREE.BufferGeometry;
  /** Build an object's programs in its pass, without blocking (see the renderer). */
  compile: (object: THREE.Object3D) => Promise<void>;
  /** The view of the glass's mirror image in the floor (the camera mirrored). */
  mirrorViewInputs: any;
  /** The renderer's other passes over the glass, with the scenes they draw. */
  passScenes: { mirror: THREE.Scene; light: THREE.Scene; lampDepth: THREE.Scene; photon: THREE.Scene };
  passNodes: {
    /** Depth along the lamp (shadows on the glass). */
    lampDepth: () => any;
    /** The light map's multiplicative filter (shadows and tint on the floor). */
    caustic: (normal: any, worldPos: any, inputs: any) => any;
    /** The real caustics' rays, for this wavelength band. */
    photon: (kind: number, worldPos: any, normal: any, inputs: any, diamondPlanes?: number[]) => any;
  };
}

/**
 * The renderer's per-object inputs for one standard mesh: `toWorld` takes the geometry's space
 * to the world (object times instance matrix), `toGeometry` back; `tint` is the mesh's own
 * colour; `vary` its own glass, as the renderer's objects carry it (added to the IOR,
 * multiplying the dispersion, CMY strength). Also the surface normal as the renderer computes
 * it (the geometry's, through the inverse scale, turned to the world).
 */
export function studioShapeInputs(
  studio: GlassStudio,
  shape: AnalyticShape,
  toWorld: any,
  toGeometry: any,
  tint: any,
  vary: any,
): { inputs: any; normal: any } {
  // The shape's own frame may be turned from the geometry's (three.js builds the torus around z).
  const toShape = (v: any) => (shape.axis === 'z' ? vec3(v.x, v.z.negate(), v.y) : v);
  const fromShape = (v: any) => (shape.axis === 'z' ? vec3(v.x, v.z, v.y.negate()) : v);
  // The scale along each of the geometry's axes, folded into the shape's size (in metres, the
  // shape's own space is then a rigid turn and move of the world's).
  const scale = vec3(
    length(toWorld.mul(vec4(1, 0, 0, 0)).xyz),
    length(toWorld.mul(vec4(0, 1, 0, 0)).xyz),
    length(toWorld.mul(vec4(0, 0, 1, 0)).xyz),
  );
  // (Plain expressions, not variables: the tracer uses them inside shader functions of its own.)
  const center = toWorld.mul(vec4(0, 0, 0, 1)).xyz;
  const frame: GlassFrame = {
    center,
    toLocalPoint: (p: any) => toShape(toGeometry.mul(vec4(p, 1)).xyz.mul(scale)),
    toLocalDir: (v: any) => toShape(toGeometry.mul(vec4(v, 0)).xyz.mul(scale)),
    toWorldPoint: (p: any) => toWorld.mul(vec4(fromShape(p).div(scale), 1)).xyz,
    toWorldDir: (v: any) => toWorld.mul(vec4(fromShape(v).div(scale), 0)).xyz,
  };
  const size = uniform(new THREE.Vector3(...shape.size)).mul(abs(toShape(scale)));
  const inputs = {
    ...studio.glassInputs,
    center,
    quat: vec4(0, 0, 0, 1),
    size,
    vary,
    tintOwn: tint,
    frame,
  };
  const normal = toWorld.mul(vec4(normalize(normalLocal.div(scale)).div(scale), 0)).xyz;
  return { inputs, normal };
}

/**
 * The renderer's glass shading for one standard mesh (see `studioShapeInputs`). `mirrored`: its
 * mirror image in the floor, drawn with the object's matrix mirrored (`toWorld` and
 * `toGeometry` are then the object's own, unmirrored) and shaded as the object itself, seen
 * from the mirrored camera, as the renderer draws it.
 */
export function studioGlassNode(
  studio: GlassStudio,
  shape: AnalyticShape,
  toWorld: any,
  toGeometry: any,
  tint: any,
  vary: any,
  mirrored = false,
): any {
  const { inputs, normal } = studioShapeInputs(studio, shape, toWorld, toGeometry, tint, vary);
  const upsideDown = vec3(1, -1, 1);
  return glassNode(
    shape.kind,
    mirrored,
    mirrored ? positionWorld.mul(upsideDown) : positionWorld,
    mirrored ? normal : normalWorld,
    inputs,
    studio.sceneInputs,
    mirrored ? studio.mirrorViewInputs : studio.viewInputs,
    shape.diamondPlanes,
  );
}
