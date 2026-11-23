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
import { KIND } from './objects';
import { UNIT_TUBE } from './geometries';
import { glassNode, type GlassFrame } from './tsl/glass';

const {
  abs,
  length,
  normalGeometry,
  normalWorld,
  normalize,
  positionGeometry,
  positionWorld,
  uniform,
  vec2,
  vec3,
  vec4,
} = TSL as any;

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
  shapeGeometry: (kind: number, cut?: string, coarse?: boolean) => THREE.BufferGeometry;
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
  const sized = uniform(new THREE.Vector3(...shape.size)).mul(abs(toShape(scale)));
  // A sphere or an octahedron is given by its radius alone: all three, as the renderer's own
  // objects carry it (its lamp test reads the size's length).
  const size = shape.kind === KIND.sphere || shape.kind === KIND.octahedron ? vec3(sized.x) : sized;
  const inputs = {
    ...studio.glassInputs,
    center,
    quat: vec4(0, 0, 0, 1),
    size,
    vary,
    tintOwn: tint,
    frame,
  };
  // (A deformed capsule or torus keeps the unit shape's normals, turned.)
  const normal = shape.deform
    ? toWorld.mul(vec4(normalGeometry.div(scale), 0)).xyz
    : toWorld.mul(vec4(normalize(normalGeometry.div(scale)).div(scale), 0)).xyz;
  return { inputs, normal };
}

/**
 * The position (in the object's space, as a material's `positionNode`) a deformed unit capsule
 * or torus (`shape.deform`) is drawn at, so that `toWorld` (whose scale along the geometry's
 * axes is the object's size) takes it to the renderer's: the capsule's caps moved to the ends
 * of its straight part, the torus's ring and tube scaled each to its own radius. Null for every
 * other shape (drawn as it is). `instanceMatrix`: an InstancedMesh's (three.js applies a
 * `positionNode` after its instancing, so the instance's transform is applied here).
 */
export function studioDeformedPosition(shape: AnalyticShape, toWorld: any, instanceMatrix: any = null): any {
  if (!shape.deform) return null;
  const s = vec3(
    length(toWorld.mul(vec4(1, 0, 0, 0)).xyz),
    length(toWorld.mul(vec4(0, 1, 0, 0)).xyz),
    length(toWorld.mul(vec4(0, 0, 1, 0)).xyz),
  );
  const p = positionGeometry;
  let local: any;
  if (shape.deform === 'capsule') {
    const capY = p.y.clamp(-1, 1);
    local = vec3(p.x.mul(s.x), p.y.sub(capY).mul(s.x).add(capY.mul(s.y)), p.z.mul(s.x)).div(s);
  } else {
    const ringDir = vec2(p.x, p.z).normalize();
    const ringPoint = vec3(ringDir.x, 0, ringDir.y);
    local = ringPoint
      .mul(s.x)
      .add(p.sub(ringPoint).mul(s.y.div(UNIT_TUBE)))
      .div(s);
  }
  return instanceMatrix ? instanceMatrix.mul(vec4(local, 1)).xyz : local;
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
    mirrored || shape.deform ? normal : normalWorld,
    inputs,
    studio.sceneInputs,
    mirrored ? studio.mirrorViewInputs : studio.viewInputs,
    shape.diamondPlanes,
  );
}
