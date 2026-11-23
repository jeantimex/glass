// Test page (layers-test.html): glass seen through glass with GlassLayers. Glass spheres in a
// front row, a row of glass boxes behind them, colour bars behind both.
//   ?layers=0     plain renderer.render(): the front glass sees opaque objects only
//   ?back=0       no back row (the front glass sees straight to the bars)
import * as THREE from 'three/webgpu';
import { GlassLayers, MeshGlassMaterial } from '../glass';

const query = new URLSearchParams(location.search);
const useLayers = query.get('layers') !== '0';
const withBack = query.get('back') !== '0';

/** A glass material of its own for each piece (so each row gets its own layer). */
const glass = () => new MeshGlassMaterial({ ior: 1.52, dispersion: 0.5, transmission: 1, thickness: 0.7 });

async function main() {
  const canvas = document.getElementById('view') as HTMLCanvasElement;
  const renderer = new THREE.WebGPURenderer({ canvas, antialias: true, forceWebGL: query.get('renderer') === 'webgl' });
  renderer.toneMapping = THREE.ACESFilmicToneMapping;
  await renderer.init();

  const scene = new THREE.Scene();
  scene.background = new THREE.Color(0x2a2f38);
  scene.add(new THREE.HemisphereLight(0xdde6ff, 0x30281f, 1.2));
  const sun = new THREE.DirectionalLight(0xffffff, 2.2);
  sun.position.set(-3, 5, 4);
  scene.add(sun);
  [0xe0453a, 0xf2b134, 0x3fae5c, 0x2f8fd8, 0x8c4fd1, 0xffffff].forEach((color, i) => {
    const bar = new THREE.Mesh(new THREE.BoxGeometry(0.4, 2.2, 0.1), new THREE.MeshStandardMaterial({ color }));
    bar.position.set((i - 2.5) * 0.9, 0.2, -2);
    scene.add(bar);
  });

  // Each piece its own material, so the front row (nearer) gets the layer and the back row not.
  for (let i = 0; i < 4; i++) {
    const front = new THREE.Mesh(new THREE.SphereGeometry(0.38, 64, 32), glass());
    front.position.set((i - 1.5) * 1.05, 0.15, 0.8);
    scene.add(front);
    if (withBack) {
      const back = new THREE.Mesh(new THREE.BoxGeometry(0.55, 0.9, 0.55), glass());
      back.position.set((i - 1.5) * 1.05 + 0.25, 0.2, -0.6);
      back.rotation.y = 0.5;
      scene.add(back);
    }
  }

  const camera = new THREE.PerspectiveCamera(32, 1, 0.1, 100);
  camera.position.set(0, 0.6, 5);
  camera.lookAt(0, 0.1, 0);
  renderer.setSize(window.innerWidth, window.innerHeight, false);
  camera.aspect = window.innerWidth / window.innerHeight;
  camera.updateProjectionMatrix();

  const layers = useLayers ? new GlassLayers() : null;
  let frames = 0;
  renderer.setAnimationLoop(() => {
    if (layers) layers.render(renderer, scene, camera);
    else renderer.render(scene, camera);
    if (++frames === 4) {
      renderer.setAnimationLoop(null);
      document.documentElement.dataset.glassReady = 'true';
    }
  });
}

main().catch((err) => {
  console.error(err);
  document.body.textContent = `Failed: ${err instanceof Error ? err.message : String(err)}`;
});
