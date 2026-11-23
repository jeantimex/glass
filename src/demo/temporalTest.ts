// Test page (temporal-test.html): strongly dispersive glass (a sphere, an octahedron, a torus)
// rendered three ways, through the same render pipeline:
//   ?mode=fixed      the default material: fixed, evenly spaced wavelengths, one frame
//   ?mode=traa       `temporal: true` under three.js's TRAA, rendered until it settles: the
//                    wavelengths move every frame and TRAA averages them
//   ?mode=reference  fixed wavelengths, 64 of them: close to a continuous spectrum
//   ?samples=N       wavelengths per pixel and frame for fixed and traa (default: automatic)
//   ?temporal=0      with mode=traa: TRAA alone, on fixed wavelengths (to tell TRAA's own
//                    filtering from the moving wavelengths)
// TRAA should converge on the reference (smooth dispersion) with far fewer samples per frame.
import * as THREE from 'three/webgpu';
import { mrt, output, pass, velocity } from 'three/tsl';
import { traa } from 'three/addons/tsl/display/TRAANode.js';
import { MeshGlassMaterial } from '../glass';

const query = new URLSearchParams(location.search);
const mode = (['fixed', 'traa', 'reference'] as const).find((m) => m === query.get('mode')) ?? 'fixed';
/** Frames TRAA is given to settle (its jitter cycle is 32 frames). */
const SETTLE_FRAMES = 96;

async function main() {
  const canvas = document.getElementById('view') as HTMLCanvasElement;
  // MSAA off for every mode (TRAA needs it off), so the three differ only in their wavelengths.
  const renderer = new THREE.WebGPURenderer({
    canvas,
    antialias: false,
    forceWebGL: query.get('renderer') === 'webgl',
  });
  renderer.toneMapping = THREE.ACESFilmicToneMapping;
  await renderer.init();

  const scene = new THREE.Scene();
  scene.background = new THREE.Color(0x2a2f38);
  scene.add(new THREE.HemisphereLight(0xdde6ff, 0x30281f, 1.2));
  const sun = new THREE.DirectionalLight(0xffffff, 2.2);
  sun.position.set(-3, 5, 4);
  scene.add(sun);
  const checker = new THREE.DataTexture(
    Uint8Array.from({ length: 64 * 4 }, (_, i) => {
      const p = i >> 2;
      return i % 4 === 3 ? 255 : ((p % 8) + (p >> 3)) % 2 ? 230 : 40;
    }),
    8,
    8,
  );
  checker.magFilter = THREE.NearestFilter;
  checker.wrapS = checker.wrapT = THREE.RepeatWrapping;
  checker.repeat.set(6, 3);
  checker.colorSpace = THREE.SRGBColorSpace;
  checker.needsUpdate = true;
  const floor = new THREE.Mesh(new THREE.PlaneGeometry(8, 4), new THREE.MeshStandardMaterial({ map: checker }));
  floor.rotation.x = -Math.PI / 2;
  floor.position.y = -0.6;
  scene.add(floor);
  [0xffffff, 0xe0453a, 0x2f8fd8, 0xffffff].forEach((color, i) => {
    const bar = new THREE.Mesh(new THREE.BoxGeometry(0.25, 1.8, 0.1), new THREE.MeshStandardMaterial({ color }));
    bar.position.set((i - 1.5) * 1.2, 0.3, -1.2);
    scene.add(bar);
  });

  const material = () =>
    new MeshGlassMaterial({
      ior: 1.52,
      dispersion: 3,
      transmission: 1,
      thickness: 0.9,
      temporal: mode === 'traa' && query.get('temporal') !== '0',
      spectralSamples: mode === 'reference' ? 64 : Number(query.get('samples') ?? 0),
    });
  const pieces: [THREE.BufferGeometry, number, number][] = [
    [new THREE.SphereGeometry(0.5, 64, 32), -1.4, 0],
    [new THREE.OctahedronGeometry(0.6), 0, 0.3],
    [new THREE.TorusGeometry(0.36, 0.15, 48, 128), 1.4, 0.6],
  ];
  for (const [geometry, x, turn] of pieces) {
    const mesh = new THREE.Mesh(geometry, material());
    mesh.position.x = x;
    mesh.rotation.set(turn, turn * 1.3, 0);
    scene.add(mesh);
  }

  const camera = new THREE.PerspectiveCamera(30, 1, 0.1, 100);
  camera.position.set(0, 1.0, 5.6);
  camera.lookAt(0, 0, 0);
  renderer.setSize(window.innerWidth, window.innerHeight, false);
  camera.aspect = window.innerWidth / window.innerHeight;
  camera.updateProjectionMatrix();

  // The same pipeline for every mode; only TRAA's mode averages frames.
  const pipeline = new THREE.RenderPipeline(renderer);
  const scenePass = pass(scene, camera);
  if (mode === 'traa') {
    scenePass.setMRT(mrt({ output, velocity }));
    pipeline.outputNode = traa(
      scenePass.getTextureNode('output'),
      scenePass.getTextureNode('depth'),
      scenePass.getTextureNode('velocity'),
      camera,
    );
  } else {
    pipeline.outputNode = scenePass;
  }

  let frames = 0;
  const needed = mode === 'traa' ? SETTLE_FRAMES : 3;
  renderer.setAnimationLoop(() => {
    pipeline.render();
    if (++frames === needed) {
      // Stop here, so the picture is the same however long a screenshot takes.
      renderer.setAnimationLoop(null);
      document.documentElement.dataset.glassReady = 'true';
    }
  });
}

main().catch((err) => {
  console.error(err);
  document.body.textContent = `Failed: ${err instanceof Error ? err.message : String(err)}`;
});
