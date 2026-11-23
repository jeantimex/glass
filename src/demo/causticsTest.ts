// Test page (caustics-test.html): GlassCaustics in a plain three.js scene. Glass pieces (a
// sphere, a box, a teapot with an automatic volume) on a MeshStandardMaterial floor, lit by
// one DirectionalLight in a dark room.
//   ?caustics=0   no GlassCaustics: the floor lit as if the glass were not there
//   ?ior=1.0001   glass that bends nothing: the caustics must leave the floor as it is
//   ?glass=0      no glass objects drawn (they still feed GlassCaustics)
//   ?dispersion=N dispersion of the glass (default 1)
import * as THREE from 'three/webgpu';
import { TeapotGeometry } from 'three/addons/geometries/TeapotGeometry.js';
import { autoGlassVolumesSettled, GlassCaustics, MeshGlassMaterial } from '../glass';

const query = new URLSearchParams(location.search);
const withCaustics = query.get('caustics') !== '0';
const ior = Number(query.get('ior') ?? 1.52);
const dispersion = Number(query.get('dispersion') ?? 1);
const drawGlass = query.get('glass') !== '0';

/** The glass of every piece. */
const glass = (color = 0xffffff) => new MeshGlassMaterial({ ior, dispersion, color, thickness: 0.8 });

async function main() {
  const canvas = document.getElementById('view') as HTMLCanvasElement;
  const renderer = new THREE.WebGPURenderer({ canvas, antialias: true, forceWebGL: query.get('renderer') === 'webgl' });
  renderer.toneMapping = THREE.ACESFilmicToneMapping;
  await renderer.init();

  const scene = new THREE.Scene();
  scene.background = new THREE.Color(0x14161b);
  scene.add(new THREE.HemisphereLight(0x8a93a8, 0x1a1712, 0.35));
  const sun = new THREE.DirectionalLight(0xffffff, 3);
  sun.position.set(-2.5, 6, 2);
  sun.target.position.set(0, 0, 0);
  scene.add(sun, sun.target);

  const floor = new THREE.Mesh(
    new THREE.PlaneGeometry(10, 10),
    new THREE.MeshStandardNodeMaterial({ color: 0xbfbfbf, roughness: 0.9 }),
  );
  floor.rotation.x = -Math.PI / 2;
  scene.add(floor);

  const pieces: [THREE.BufferGeometry, number, number, number][] = [
    [new THREE.SphereGeometry(0.45, 64, 32), -1.4, 0.45, 0],
    [new THREE.BoxGeometry(0.7, 0.7, 0.7), 0, 0.35, 0.5],
    [new TeapotGeometry(0.35, 8), 1.4, 0.35 * 0.78, -0.4],
  ];
  for (const [geometry, x, y, turn] of pieces) {
    const mesh = new THREE.Mesh(geometry, glass());
    mesh.position.set(x, y, 0);
    mesh.rotation.y = turn;
    if (!drawGlass) mesh.layers.set(1); // still in the scene (for GlassCaustics), not drawn
    scene.add(mesh);
  }

  const camera = new THREE.PerspectiveCamera(32, 1, 0.1, 100);
  camera.position.set(0, 3.2, 5.4);
  camera.lookAt(0, 0.1, 0);
  renderer.setSize(window.innerWidth, window.innerHeight, false);
  camera.aspect = window.innerWidth / window.innerHeight;
  camera.updateProjectionMatrix();

  const caustics = withCaustics ? new GlassCaustics({ light: sun, extent: 3, softness: 0.04 }) : null;
  caustics?.attach();

  let frames = 0;
  let settled = false;
  renderer.setAnimationLoop(() => {
    caustics?.update(renderer, scene);
    renderer.render(scene, camera);
    frames++;
    if (frames === 2) {
      void autoGlassVolumesSettled().then(() => {
        settled = true;
        frames = 2;
      });
    }
    if (settled && frames === 6) {
      renderer.setAnimationLoop(null);
      document.documentElement.dataset.glassReady = 'true';
    }
  });
}

main().catch((err) => {
  console.error(err);
  document.body.textContent = `Failed: ${err instanceof Error ? err.message : String(err)}`;
});
