// Loading-screen progress indicator and binary mesh / 128^3 SDF volume loaders.
import * as THREE from 'three/webgpu';

// In layer order (see SDF_LAYER in src/glass/tsl/sdf.ts).
const MESH_SDF_URLS = ['suzanne_sdf_128.bin', 'dragon_sdf_128.bin', 'torusknot_sdf_128.bin'];
const MESH_SDF_SIZE = 128;

export interface LoadingProgress {
  stage(fraction: number, message: string): void;
  fetch(url: string): Promise<Response & { buffer: () => Promise<ArrayBuffer> }>;
  done(): void;
}

/** Display a full-screen fatal error and throw. */
export function showFatalError(message: string): never {
  document.getElementById('loading')?.remove();
  const el = document.getElementById('error')!;
  el.textContent = message;
  el.style.display = 'flex';
  throw new Error(message);
}

const formatMegabytes = (bytes: number) => (bytes / 1048576).toFixed(1);

/** Wire up the DOM loading bar & progress tracker for binary model downloads. */
export function createLoadingScreen(): LoadingProgress {
  const root = document.getElementById('loading')!;
  const fill = document.getElementById('loading-fill')!;
  const text = document.getElementById('loading-text')!;
  const files = new Map<string, { loaded: number; total: number | null; done: boolean }>();
  const DOWNLOAD_SHARE = 0.85;

  const setProgress = (fraction: number, message: string) => {
    fill.style.width = `${(Math.min(Math.max(fraction, 0), 1) * 100).toFixed(1)}%`;
    text.textContent = message;
  };

  const updateDownloads = () => {
    const entries = [...files.values()];
    const loaded = entries.reduce((sum, f) => sum + f.loaded, 0);
    const allKnown = entries.every((f) => f.total !== null);
    const total = entries.reduce((sum, f) => sum + (f.total ?? 0), 0);
    const fraction =
      allKnown && total > 0 ? loaded / total : entries.filter((f) => f.done).length / Math.max(entries.length, 1);
    setProgress(
      fraction * DOWNLOAD_SHARE,
      allKnown
        ? `Downloading models… ${formatMegabytes(loaded)} / ${formatMegabytes(total)} MB`
        : `Downloading models… ${formatMegabytes(loaded)} MB`,
    );
  };

  return {
    stage: setProgress,
    async fetch(url: string) {
      const res = await fetch(url);
      const lengthHeader = res.headers.get('content-length');
      const total = lengthHeader && !res.headers.get('content-encoding') ? Number(lengthHeader) : null;
      const entry = { loaded: 0, total, done: false };
      files.set(url, entry);

      const buffer = async () => {
        if (!res.body) return res.arrayBuffer();
        const chunks: Uint8Array[] = [];
        const reader = res.body.getReader();
        for (;;) {
          const { done, value } = await reader.read();
          if (done) break;
          chunks.push(value);
          entry.loaded += value.byteLength;
          updateDownloads();
        }
        entry.done = true;
        updateDownloads();
        const out = new Uint8Array(entry.loaded);
        let offset = 0;
        for (const chunk of chunks) {
          out.set(chunk, offset);
          offset += chunk.byteLength;
        }
        return out.buffer;
      };
      return Object.assign(res, { buffer });
    },
    done() {
      setProgress(1, 'Ready');
      root.classList.add('done');
      setTimeout(() => root.remove(), 500);
    },
  };
}

/** Load the Suzanne, dragon and torus knot 128^3 HalfFloat SDF volumes stacked along Z in one 3D texture. */
export async function loadMeshVolumes(loading: LoadingProgress): Promise<THREE.Data3DTexture> {
  const size = MESH_SDF_SIZE;
  const stacked = new Uint16Array(size * size * size * MESH_SDF_URLS.length);
  await Promise.all(
    MESH_SDF_URLS.map(async (url, layer) => {
      const res = await loading.fetch(url);
      if (!res.ok) showFatalError(`Could not load the mesh volume (${url}).`);
      stacked.set(new Uint16Array(await res.buffer()), layer * size * size * size);
    }),
  );
  const texture = new THREE.Data3DTexture(stacked, size, size, size * MESH_SDF_URLS.length);
  texture.format = THREE.RedFormat;
  texture.type = THREE.HalfFloatType;
  texture.minFilter = texture.magFilter = THREE.LinearFilter;
  texture.wrapS = texture.wrapT = texture.wrapR = THREE.ClampToEdgeWrapping;
  texture.needsUpdate = true;
  return texture;
}

/**
 * Load a binary mesh asset (`uint32[3]` header: vertexCount, indexCount, hullCount;
 * followed by `Float32` positions, `Float32` normals, and `Uint32` triangle indices).
 */
export async function loadBinaryMesh(loading: LoadingProgress, url: string): Promise<THREE.BufferGeometry> {
  const res = await loading.fetch(url);
  if (!res.ok) showFatalError(`Could not load the mesh (${url}).`);
  const buffer = await res.buffer();
  const [vertexCount, indexCount] = new Uint32Array(buffer, 0, 2);
  let offset = 12;
  const readSlice = <T>(factory: (start: number, count: number) => T, count: number): T => {
    const slice = factory(offset, count);
    offset += count * 4;
    return slice;
  };
  const positions = readSlice((o, n) => new Float32Array(buffer.slice(o, o + n * 4)), vertexCount * 3);
  const normals = readSlice((o, n) => new Float32Array(buffer.slice(o, o + n * 4)), vertexCount * 3);
  const indices = readSlice((o, n) => new Uint32Array(buffer.slice(o, o + n * 4)), indexCount);

  const geometry = new THREE.BufferGeometry();
  geometry.setAttribute('position', new THREE.BufferAttribute(positions, 3));
  geometry.setAttribute('normal', new THREE.BufferAttribute(normals, 3));
  geometry.setIndex(new THREE.BufferAttribute(indices, 1));
  return geometry;
}
