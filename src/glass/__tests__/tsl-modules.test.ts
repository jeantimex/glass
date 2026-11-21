import { describe, expect, it } from 'vitest';
import * as THREE from 'three/webgpu';
import { Fn, float, texture, texture3D, uniform, uv, vec2, vec3 } from 'three/tsl';
import { KIND } from '../objects';
import { blurNode, contactShadeNode, lensNode, tones } from '../tsl/finish';
import { applyFrostVeil, frostNoise3, perturbFrostNormal } from '../tsl/frost';
import { groundNode } from '../tsl/ground';
import { sceneFunctions } from '../tsl/scene';
import { createSdfVolume } from '../tsl/sdf';
import { isBakedMesh, isCurvedShape, shapeFunctions } from '../tsl/shapes';
import { makeStudioEnv } from '../tsl/studioEnv';

describe('modular TSL shader units', () => {
  const dummyTex = new THREE.DataTexture(new Uint8Array([255, 255, 255, 255]), 1, 1);
  const dummy3D = new THREE.Data3DTexture(new Float32Array(8), 2, 2, 2);

  it('classifies baked meshes and curved vs flat-faceted shapes accurately', () => {
    expect(isBakedMesh(KIND.suzanne)).toBe(true);
    expect(isBakedMesh(KIND.dragon)).toBe(true);
    expect(isBakedMesh(KIND.diamond)).toBe(false);
    expect(isBakedMesh(KIND.sphere)).toBe(false);

    for (const curved of [
      KIND.sphere,
      KIND.cylinder,
      KIND.capsule,
      KIND.cone,
      KIND.torus,
      KIND.suzanne,
      KIND.dragon,
      KIND.torusknot,
    ]) {
      expect(isCurvedShape(curved)).toBe(true);
    }
    for (const flat of [KIND.box, KIND.octahedron, KIND.prism, KIND.hexprism, KIND.diamond]) {
      expect(isCurvedShape(flat)).toBe(false);
    }
  });

  it('builds frost noise, normal perturbation, and frost veil nodes (frost.ts)', () => {
    expect(frostNoise3(vec3(0.5, 1.0, -0.3))).toBeTruthy();
    const frostFn = Fn(() => {
      const normal = vec3(0, 1, 0).toVar();
      const grain = float(0.5).toVar();
      const color = vec3(0.5, 0.5, 0.5).toVar();
      perturbFrostNormal(normal, vec3(0, -1, 0), vec3(0, 0.5, 0), float(1), float(0.3), float(4), grain);
      applyFrostVeil(color, vec3(0, -1, 0), normal, vec3(1, 1, 1), vec3(0.7, 0.7, 0.7), float(0.3), grain);
      return color;
    })();
    expect(frostFn).toBeTruthy();
  });

  it('builds fast and tricubic SDF volume samplers (sdf.ts)', () => {
    const sdf = createSdfVolume(texture3D(dummy3D), 1);
    expect(sdf.volumeFast(vec3(0, 0, 0), float(1))).toBeTruthy();
    expect(sdf.volumeFastNormal(vec3(0, 0, 0), float(1))).toBeTruthy();
    expect(sdf.volumeSmooth(vec3(0, 0, 0), float(1))).toBeTruthy();
    expect(sdf.volumeSmoothNormal(vec3(0, 0, 0), float(1))).toBeTruthy();
  });

  it('builds shapeFunctions (declare, entry, exit) for all 13 KINDs (shapes.ts)', () => {
    const uniforms = { bevel: uniform(0), sdfTex: texture3D(dummy3D), meshEntry: uniform(1) };
    for (const kind of Object.values(KIND)) {
      const fns = shapeFunctions(kind, uniforms);
      const traced = Fn(() => {
        fns.declare();
        const entered = fns.entry(vec3(0, 0.5, 0), vec3(0.5, 0.5, 0.5), vec3(0, 1, 0));
        const exited = fns.exit(entered.pos, vec3(0, -1, 0), vec3(0.5, 0.5, 0.5));
        return exited.pos;
      })();
      expect(traced).toBeTruthy();
    }
  });

  it('builds standalone studioEnv, sceneFunctions, groundNode, and finishing passes', () => {
    const studio = makeStudioEnv(
      uniform(0),
      uniform(1),
      uniform(new THREE.Color(0xeeeeee)),
      uniform(new THREE.Vector3(0, 1, 0)),
      uniform(0.2),
    );
    expect(studio.env(vec3(0, 1, 0))).toBeTruthy();
    expect(studio.sampleScene(vec3(0, 1, 0), vec3(0, -1, 0))).toBeTruthy();

    const sInputs = {
      room: uniform(1),
      skyE: uniform(new THREE.Vector3(0.3, 0.3, 0.3)),
      sunE: uniform(new THREE.Vector3(0.5, 0.5, 0.5)),
      albedo: uniform(new THREE.Vector3(0.9, 0.9, 0.9)),
      lightC: uniform(new THREE.Vector3()),
      lightR: uniform(new THREE.Vector3(1, 0, 0)),
      lightU: uniform(new THREE.Vector3(0, 0, 1)),
      floorP: uniform(new THREE.Vector4(0.4, -0.8, 1, 0.15)),
      floorColor: uniform(new THREE.Vector4(0.9, 0.9, 0.9, 0.8)),
      gridColor: uniform(new THREE.Vector4(0.1, 0.1, 0.1, 0.6)),
      spot: uniform(new THREE.Vector4(0, 0, 0, 0)),
      lamp: uniform(new THREE.Vector3(0, 1, 0)),
      lightTex: texture(dummyTex),
      lightTexSize: 1024,
      shadeTex: texture(dummyTex),
      shadeFarTex: texture(dummyTex),
    };
    const cam = {
      fragCoord: vec2(10, 10),
      res: uniform(new THREE.Vector2(800, 600)),
      camPos: uniform(new THREE.Vector3(0, 2, 5)),
      camRight: uniform(new THREE.Vector4(1, 0, 0, 1.33)),
      camUp: uniform(new THREE.Vector3(0, 1, 0)),
      camFwd: uniform(new THREE.Vector4(0, 0, -1, 0.25)),
    };

    const backdropScene = sceneFunctions(sInputs, true);
    expect(backdropScene.scene(vec3(0, 2, 0), vec3(0, -1, 0), float(1), float(0))).toBeTruthy();
    expect(groundNode(cam, sInputs, texture(dummyTex))).toBeTruthy();

    expect(blurNode(uv(), uniform(new THREE.Vector2(0.01, 0)), uniform(0), uniform(6), texture(dummyTex))).toBeTruthy();
    expect(contactShadeNode(cam, uniform(new THREE.Vector4(0.7, 0.3, 1, 0)), texture(dummyTex))).toBeTruthy();
    expect(Fn(() => tones(vec3(0.5, 1.2, 2.0)))()).toBeTruthy();
    expect(
      lensNode(
        vec2(10, 10),
        uniform(new THREE.Vector2(800, 600)),
        uniform(new THREE.Vector4(5, 0, 0.35, 1)),
        uniform(new THREE.Vector4(1, 1, 0.002, 0.002)),
        texture(dummyTex),
        texture(dummyTex),
        texture(dummyTex),
      ),
    ).toBeTruthy();
  });
});
