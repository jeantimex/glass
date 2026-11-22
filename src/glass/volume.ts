// Glass volumes for any geometry, in the browser: bake a THREE.BufferGeometry into the signed-
// distance volume MeshGlassMaterial's `volume` option follows light through (exact exits on
// every internal bounce). See src/glass/bake.ts for what the bake does to make a mesh solid.
import * as THREE from 'three/webgpu';
import { bakeGlassVolume, VOLUME_SIZE, type BakeOptions, type BakeResult } from './bake.ts';
import type { GlassVolume } from './material';

/** A baked volume's data as a 128^3 HalfFloat 3D texture, as the glass shaders sample it. */
export function createVolumeTexture(data: Uint16Array, layers = 1): THREE.Data3DTexture {
  const texture = new THREE.Data3DTexture(data, VOLUME_SIZE, VOLUME_SIZE, VOLUME_SIZE * layers);
  texture.format = THREE.RedFormat;
  texture.type = THREE.HalfFloatType;
  texture.minFilter = texture.magFilter = THREE.LinearFilter;
  texture.wrapS = texture.wrapT = texture.wrapR = THREE.ClampToEdgeWrapping;
  texture.needsUpdate = true;
  return texture;
}

/** A bake's result as a `GlassVolume` for MeshGlassMaterial. */
export function toGlassVolume(result: Pick<BakeResult, 'data' | 'center' | 'halfWidth'>): GlassVolume {
  return { texture: createVolumeTexture(result.data), center: result.center, halfWidth: result.halfWidth };
}

/** The geometry's positions and triangle indices, as the bake takes them. */
function meshArrays(geometry: THREE.BufferGeometry): { positions: Float32Array; indices: Uint32Array | null } {
  const attribute = geometry.getAttribute('position');
  const positions = new Float32Array(attribute.count * 3);
  for (let i = 0; i < attribute.count; i++) {
    positions[i * 3] = attribute.getX(i);
    positions[i * 3 + 1] = attribute.getY(i);
    positions[i * 3 + 2] = attribute.getZ(i);
  }
  return { positions, indices: geometry.index ? Uint32Array.from(geometry.index.array) : null };
}

/**
 * Bake `geometry`'s glass volume on this thread (a few seconds for a big mesh; prefer
 * `bakeGlassVolumeAsync` in a page).
 */
export function createGlassVolume(geometry: THREE.BufferGeometry, options: BakeOptions = {}): GlassVolume {
  const { positions, indices } = meshArrays(geometry);
  return toGlassVolume(bakeGlassVolume(positions, indices, options));
}

/**
 * Bake `geometry`'s glass volume in a Web Worker, so the page stays responsive (falls back to
 * this thread where workers are unavailable).
 *
 * ```ts
 * const geometry = new TeapotGeometry(1, 10);
 * const glass = new MeshGlassMaterial({ volume: await bakeGlassVolumeAsync(geometry) });
 * scene.add(new THREE.Mesh(geometry, glass));
 * ```
 */
export async function bakeGlassVolumeAsync(
  geometry: THREE.BufferGeometry,
  options: BakeOptions = {},
): Promise<GlassVolume> {
  if (typeof Worker === 'undefined') return createGlassVolume(geometry, options);
  const { positions, indices } = meshArrays(geometry);
  const worker = new Worker(new URL('./bakeWorker.ts', import.meta.url), { type: 'module' });
  try {
    const result = await new Promise<BakeResult>((resolve, reject) => {
      worker.addEventListener('message', (event: MessageEvent<BakeResult>) => resolve(event.data), { once: true });
      worker.addEventListener('error', (event) => reject(new Error(event.message || 'glass volume bake failed')), {
        once: true,
      });
      const transfer: Transferable[] = [positions.buffer];
      if (indices) transfer.push(indices.buffer);
      worker.postMessage({ positions, indices, options }, transfer);
    });
    return toGlassVolume(result);
  } finally {
    worker.terminate();
  }
}
