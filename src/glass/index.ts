// The glass renderer: everything needed to draw glass objects and the floor they
// stand on. It knows nothing about physics: whatever moves the objects writes
// their positions and rotations into GlassObjects (see src/physics for two ways).
export { KIND, MAX_OBJECTS, createGlassObjects, type GlassObjects } from './objects';
export { createInstancedGlass, type InstancedLighting } from './renderer';
export { DIAMOND_LEAN, DIAMOND_REST, diamondPoints } from './diamond';
