// Automatic glass volumes (MeshGlassMaterial's `volume: 'auto'`, the default): the first time a
// geometry with no exact formula is drawn in glass, it is baked once in a worker
// (bakeGlassVolumeAsync) and the volume kept for every material and mesh that uses it. Until
// it is ready the mesh is drawn with the thickness estimate; then the waiting materials are
// told to rebuild with exact exits.
import type * as THREE from 'three/webgpu';
import type { GlassVolume } from './material';
import { bakeGlassVolumeAsync } from './volume';

interface Entry {
  volume: GlassVolume | null;
  /** Called once when the bake finishes (materials asking to be rebuilt). */
  waiting: Set<() => void>;
  done: Promise<void>;
}

const entries = new WeakMap<THREE.BufferGeometry, Entry>();
const pending = new Set<Promise<void>>();

/**
 * The geometry's glass volume if it has been baked; otherwise null, starting its bake if this
 * is the first ask, and calling `onReady` once when it is ready.
 */
export function autoGlassVolume(geometry: THREE.BufferGeometry, onReady: () => void): GlassVolume | null {
  let entry = entries.get(geometry);
  if (!entry) {
    const created: Entry = { volume: null, waiting: new Set(), done: Promise.resolve() };
    created.done = bakeGlassVolumeAsync(geometry)
      .then((volume) => {
        created.volume = volume;
        geometry.addEventListener('dispose', () => {
          volume.texture.dispose();
          entries.delete(geometry);
        });
      })
      .catch((error) => {
        // Leave the geometry on the estimate (a mesh the bake cannot make solid).
        console.warn('MeshGlassMaterial: could not bake a glass volume for this geometry', error);
      })
      .finally(() => {
        pending.delete(created.done);
        const callbacks = [...created.waiting];
        created.waiting.clear();
        for (const callback of callbacks) callback();
      });
    pending.add(created.done);
    entries.set(geometry, created);
    entry = created;
  }
  if (!entry.volume) entry.waiting.add(onReady);
  return entry.volume;
}

/** Resolves when every automatic volume bake started so far has finished. */
export async function autoGlassVolumesSettled(): Promise<void> {
  while (pending.size > 0) await Promise.all(pending);
}
