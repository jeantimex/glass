// What the demo needs from a physics engine. Each engine (box3d.ts on the CPU,
// gpu.ts on the GPU) is an adapter: it spawns the objects into the shared
// GlassObjects arrays, then writes their positions and rotations there as they
// fall. The glass renderer only reads those arrays and never sees the engine.

export interface Pile {
  /**
   * Fill the objects with a fresh pile, ready to drop.
   * @param count  how many objects
   * @param only   shape ids to draw from; null for the mix of every shape
   * @param cubic  make boxes cubes (equal sides)
   * @param adjust called once they are spawned, to change any of them by hand
   */
  reset(count: number, only?: number[] | null, cubic?: boolean, adjust?: () => void): void;
  /** Advance the simulation; returns how many objects moved. */
  step(): number;
  /** Half-width of the square the objects were dropped over (m). */
  footprint(): number;
  /** A short description for the stats panel (threads, GPU...). */
  label(): string;
}
