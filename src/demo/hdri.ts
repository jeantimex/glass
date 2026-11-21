// Equirectangular HDRI environments for the demo:
// - Real CC0 photographic HDRIs streamed on demand from Poly Haven (`HDRLoader`, 1k HalfFloat).
// - Procedural HalfFloat fallback generator for offline / zero-network environments.
import * as THREE from 'three/webgpu';
import { HDRLoader } from 'three/addons/loaders/HDRLoader.js';

export type PolyHavenSlug =
  | 'santos_beach_pavillion'
  | 'kloofendal_48d_partly_cloudy_puresky'
  | 'venice_sunset'
  | 'belfast_sunset_puresky'
  | 'studio_small_09'
  | 'rogland_clear_night';

export type HdriPreset = 'none' | PolyHavenSlug;

/** Display labels -> preset ids for the `Lighting -> Environment` dropdown. */
export const HDRI_PRESET_LABELS: Record<string, HdriPreset> = {
  'Studio (procedural)': 'none',
  'Santos Beach Pavilion (Beach)': 'santos_beach_pavillion',
  'Partly Cloudy Sky (Clouds)': 'kloofendal_48d_partly_cloudy_puresky',
  'Venice Sunset (Ocean)': 'venice_sunset',
  'Sunset Clouds': 'belfast_sunset_puresky',
  'Photo Studio (HDRI)': 'studio_small_09',
  'Clear Night Sky': 'rogland_clear_night',
};

export type HdriResolution = '1k' | '2k' | '4k';

/** Official Poly Haven CC0 CDN URL for an equirectangular `.hdr` environment. */
export function polyHavenHdriUrl(slug: PolyHavenSlug, resolution: HdriResolution = '4k'): string {
  return `https://dl.polyhaven.org/file/ph-assets/HDRIs/hdr/${resolution}/${slug}_${resolution}.hdr`;
}

/** Configure an equirectangular HDRI texture for mipmapped hardware trilinear sampling. */
export function configureEquirectHdri(tex: THREE.DataTexture): THREE.DataTexture {
  tex.mapping = THREE.EquirectangularReflectionMapping;
  tex.wrapS = THREE.RepeatWrapping;
  tex.wrapT = THREE.ClampToEdgeWrapping;
  tex.minFilter = THREE.LinearMipmapLinearFilter;
  tex.magFilter = THREE.LinearFilter;
  tex.generateMipmaps = true;
  tex.needsUpdate = true;
  return tex;
}

const hdrLoader = new HDRLoader();
const loadedCache = new Map<HdriPreset, THREE.DataTexture | null>([['none', null]]);
const pendingLoads = new Map<PolyHavenSlug, Promise<THREE.DataTexture | null>>();

/** Return an already-loaded HDRI texture synchronously, or `null` if not yet loaded / `'none'`. */
export function getCachedHdri(preset: HdriPreset): THREE.DataTexture | null {
  return loadedCache.get(preset) ?? null;
}

/**
 * Load a Poly Haven CC0 equirectangular `.hdr` on demand (cached & deduplicated across calls).
 * Defaults to `4k` on desktop for crisp backgrounds, or `2k` when requested (e.g. mobile).
 * Falls back to a procedural sky/studio HDRI if the network request fails.
 */
export async function loadHdriPreset(
  preset: HdriPreset,
  resolution: HdriResolution = '4k',
): Promise<THREE.DataTexture | null> {
  if (preset === 'none') return null;
  if (loadedCache.has(preset)) return loadedCache.get(preset) ?? null;

  const existing = pendingLoads.get(preset);
  if (existing) return existing;

  const promise = hdrLoader
    .loadAsync(polyHavenHdriUrl(preset, resolution))
    .then((tex) => {
      configureEquirectHdri(tex);
      loadedCache.set(preset, tex);
      pendingLoads.delete(preset);
      return tex;
    })
    .catch((err) => {
      console.warn(`Failed to load Poly Haven HDRI (${preset}), falling back to procedural HDRI:`, err);
      const fallback = createProceduralHdri(preset === 'studio_small_09' ? 'studio' : 'daylight');
      loadedCache.set(preset, fallback);
      pendingLoads.delete(preset);
      return fallback;
    });

  pendingLoads.set(preset, promise);
  return promise;
}

/* Offline procedural HDRI fallback generator */

export type ProceduralHdriKind = 'studio' | 'daylight';

const clamp01 = (x: number) => (x < 0 ? 0 : x > 1 ? 1 : x);
const smoothstep = (e0: number, e1: number, x: number) => {
  const t = clamp01((x - e0) / (e1 - e0));
  return t * t * (3 - 2 * t);
};
const lerp = (a: number, b: number, t: number) => a + (b - a) * t;

function sampleProceduralHdri(kind: ProceduralHdriKind, dx: number, dy: number, dz: number): [number, number, number] {
  if (kind === 'studio') {
    const up = smoothstep(-0.2, 0.9, dy);
    let r = lerp(0.06, 0.28, up);
    let g = lerp(0.065, 0.3, up);
    let b = lerp(0.075, 0.34, up);
    const key = smoothstep(0.86, 0.95, dx * -0.58 + dy * 0.68 + dz * 0.45) * 8.0;
    const rim = smoothstep(0.9, 0.97, dx * 0.78 + dy * 0.45 - dz * 0.43) * 6.5;
    const ring = smoothstep(0.82, 0.88, dy) * (1 - smoothstep(0.93, 0.97, dy)) * 3.5;
    r += key * 1.08 + rim * 0.82 + ring;
    g += key * 0.98 + rim * 0.95 + ring;
    b += key * 0.86 + rim * 1.15 + ring;
    return [r, g, b];
  }
  const skyT = smoothstep(0, 0.85, dy);
  const horizonHaze = Math.exp(-Math.abs(dy) * 9.0);
  let r = lerp(0.65, 0.16, skyT) + horizonHaze * 0.45;
  let g = lerp(0.76, 0.38, skyT) + horizonHaze * 0.45;
  let b = lerp(0.92, 0.85, skyT) + horizonHaze * 0.4;
  if (dy < 0) {
    const groundFade = smoothstep(0, -0.25, dy);
    r = lerp(r, 0.22, groundFade);
    g = lerp(g, 0.23, groundFade);
    b = lerp(b, 0.24, groundFade);
  }
  const sunDot = dx * -0.45 + dy * 0.78 + dz * 0.43;
  const sun = Math.pow(Math.max(sunDot, 0), 16) * 2.5 + smoothstep(0.995, 0.999, sunDot) * 24.0;
  return [r + sun * 1.04, g + sun * 0.98, b + sun * 0.88];
}

/** Generate an offline mipmapped equirectangular `HalfFloatType` HDRI `DataTexture`. */
export function createProceduralHdri(kind: ProceduralHdriKind, width = 512, height = 256): THREE.DataTexture {
  const data = new Uint16Array(width * height * 4);
  const toHalf = THREE.DataUtils.toHalfFloat;
  const oneHalf = toHalf(1);

  for (let y = 0; y < height; y++) {
    const v = (y + 0.5) / height;
    const theta = (v - 0.5) * Math.PI;
    const cosTheta = Math.cos(theta);
    const dy = Math.sin(theta);
    for (let x = 0; x < width; x++) {
      const u = (x + 0.5) / width;
      const phi = (u - 0.5) * 2 * Math.PI;
      const dx = cosTheta * Math.cos(phi);
      const dz = cosTheta * Math.sin(phi);
      const [r, g, b] = sampleProceduralHdri(kind, dx, dy, dz);
      const offset = (y * width + x) * 4;
      data[offset] = toHalf(r);
      data[offset + 1] = toHalf(g);
      data[offset + 2] = toHalf(b);
      data[offset + 3] = oneHalf;
    }
  }

  return configureEquirectHdri(new THREE.DataTexture(data, width, height, THREE.RGBAFormat, THREE.HalfFloatType));
}
