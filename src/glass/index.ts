// The glass renderer: everything needed to draw glass objects and the floor they
// stand on. Whatever places or moves the objects writes their positions and
// rotations into GlassObjects (the demo sets them out in a grid: src/demo/grid.ts).
export { KIND, MAX_OBJECTS, createGlassObjects, type GlassObjects } from './objects';
export { createInstancedGlass, type InstancedLighting } from './renderer';
export { DIAMOND_CUTS, DIAMOND_LEAN, DIAMOND_REST, diamondGeometry, type DiamondCut } from './diamond';
export { GlassMaterial, type GlassMaterialParameters, type GlassShape } from './material';
export { CMY_FACE_ORDERS, type CmyFaceOrder } from './tsl/cmy';
