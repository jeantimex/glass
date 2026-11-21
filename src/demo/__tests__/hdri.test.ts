import { describe, expect, it } from 'vitest';
import * as THREE from 'three/webgpu';
import {
  configureEquirectHdri,
  createProceduralHdri,
  getCachedHdri,
  HDRI_PRESET_LABELS,
  loadHdriPreset,
  polyHavenHdriUrl,
  type PolyHavenSlug,
} from '../hdri';

describe('Poly Haven CC0 HDRI loader & procedural fallback (hdri.ts)', () => {
  it('exposes Santos Beach Pavilion, Clouds, Ocean, Studio, and Night presets with valid Poly Haven URLs', () => {
    const presets = Object.values(HDRI_PRESET_LABELS);
    expect(presets).toContain('none');
    expect(presets).toContain('santos_beach_pavillion');
    expect(presets).toContain('kloofendal_48d_partly_cloudy_puresky');
    expect(presets).toContain('venice_sunset');
    expect(presets).toContain('studio_small_09');

    for (const preset of presets) {
      if (preset === 'none') continue;
      const url4k = polyHavenHdriUrl(preset as PolyHavenSlug);
      expect(url4k).toBe(`https://dl.polyhaven.org/file/ph-assets/HDRIs/hdr/4k/${preset}_4k.hdr`);
      const url2k = polyHavenHdriUrl(preset as PolyHavenSlug, '2k');
      expect(url2k).toBe(`https://dl.polyhaven.org/file/ph-assets/HDRIs/hdr/2k/${preset}_2k.hdr`);
    }
  });

  it('returns null for "none" synchronously and asynchronously', async () => {
    expect(getCachedHdri('none')).toBeNull();
    expect(await loadHdriPreset('none')).toBeNull();
  });

  it('configures equirectangular HDRIs with trilinear mipmaps and Repeat/Clamp wrapping', () => {
    const raw = new THREE.DataTexture(new Uint16Array(4), 1, 1, THREE.RGBAFormat, THREE.HalfFloatType);
    const configured = configureEquirectHdri(raw);
    expect(configured.mapping).toBe(THREE.EquirectangularReflectionMapping);
    expect(configured.wrapS).toBe(THREE.RepeatWrapping);
    expect(configured.wrapT).toBe(THREE.ClampToEdgeWrapping);
    expect(configured.minFilter).toBe(THREE.LinearMipmapLinearFilter);
    expect(configured.magFilter).toBe(THREE.LinearFilter);
    expect(configured.generateMipmaps).toBe(true);
  });

  it('generates valid HalfFloat RGBA fallback HDRIs with true HDR peaks (> 2.0)', () => {
    for (const kind of ['studio', 'daylight'] as const) {
      const tex = createProceduralHdri(kind, 64, 32);
      expect(tex.type).toBe(THREE.HalfFloatType);
      expect(tex.format).toBe(THREE.RGBAFormat);
      const data = tex.image.data as Uint16Array;
      expect(data.length).toBe(64 * 32 * 4);
      let maxRadiance = 0;
      for (let i = 0; i < data.length; i += 4) {
        const r = THREE.DataUtils.fromHalfFloat(data[i]);
        const g = THREE.DataUtils.fromHalfFloat(data[i + 1]);
        const b = THREE.DataUtils.fromHalfFloat(data[i + 2]);
        const a = THREE.DataUtils.fromHalfFloat(data[i + 3]);
        expect(Number.isFinite(r) && r >= 0).toBe(true);
        expect(Number.isFinite(g) && g >= 0).toBe(true);
        expect(Number.isFinite(b) && b >= 0).toBe(true);
        expect(a).toBeCloseTo(1, 3);
        maxRadiance = Math.max(maxRadiance, r, g, b);
      }
      expect(maxRadiance).toBeGreaterThan(2);
    }
  });
});
