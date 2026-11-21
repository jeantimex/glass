import { describe, expect, it, vi } from 'vitest';
import * as THREE from 'three/webgpu';
import { GlassBackfacePass } from '../backfacePass';
import { MeshGlassMaterial } from '../material';

describe('GlassBackfacePass', () => {
  it('initializes with a half-float BackSide material and clamped resolution scale', () => {
    const pass = new GlassBackfacePass({ resolutionScale: 0.5 });
    expect(pass.resolutionScale).toBe(0.5);
    expect(pass.material.side).toBe(THREE.BackSide);
    expect(pass.target.texture.type).toBe(THREE.HalfFloatType);
    expect(pass.texture).toBe(pass.target.texture);

    pass.setSize(1000, 600);
    expect(pass.target.width).toBe(500);
    expect(pass.target.height).toBe(300);
    pass.dispose();
  });

  it('hides non-glass meshes during the back-face pass and restores scene & renderer state', () => {
    const pass = new GlassBackfacePass({ resolutionScale: 1 });
    const scene = new THREE.Scene();
    const glassMesh = new THREE.Mesh(new THREE.BoxGeometry(1, 1, 1), new MeshGlassMaterial());
    const opaqueMesh = new THREE.Mesh(new THREE.BoxGeometry(1, 1, 1), new THREE.MeshBasicMaterial());
    const hiddenOpaque = new THREE.Mesh(new THREE.BoxGeometry(1, 1, 1), new THREE.MeshBasicMaterial());
    hiddenOpaque.visible = false;
    scene.add(glassMesh, opaqueMesh, hiddenOpaque);

    let opaqueVisibleDuringRender = true;
    let glassVisibleDuringRender = false;
    let overrideDuringRender: THREE.Material | null = null;

    const fakeRenderer = {
      autoClear: true,
      getDrawingBufferSize: (v: THREE.Vector2) => v.set(640, 480),
      getRenderTarget: () => null,
      setRenderTarget: vi.fn(),
      getClearColor: (c: THREE.Color) => c.set(0x112233),
      getClearAlpha: () => 1,
      setClearColor: vi.fn(),
      render: vi.fn(() => {
        opaqueVisibleDuringRender = opaqueMesh.visible;
        glassVisibleDuringRender = glassMesh.visible;
        overrideDuringRender = scene.overrideMaterial;
      }),
    } as unknown as THREE.WebGPURenderer;

    const camera = new THREE.PerspectiveCamera();
    pass.render(fakeRenderer, scene, camera);

    // During pass: only glass meshes visible, overrideMaterial = pass.material
    expect(opaqueVisibleDuringRender).toBe(false);
    expect(glassVisibleDuringRender).toBe(true);
    expect(overrideDuringRender).toBe(pass.material);

    // After pass: everything restored cleanly
    expect(opaqueMesh.visible).toBe(true);
    expect(hiddenOpaque.visible).toBe(false);
    expect(scene.overrideMaterial).toBeNull();
    expect(pass.target.width).toBe(640);
    expect(pass.target.height).toBe(480);

    pass.dispose();
  });
});
