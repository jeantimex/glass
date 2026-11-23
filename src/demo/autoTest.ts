// Test page (auto-test.html): geometries with no exact formula (a torus knot, the Utah teapot,
// a lathed vase) in MeshGlassMaterial.
//   ?exits=auto      the default material: each geometry's volume is baked automatically
//   ?exits=baked     volumes baked explicitly from the same geometries (bakeGlassVolumeAsync)
//   ?instanced=1     each drawn as a one-instance InstancedMesh instead of a Mesh
// The page is ready once every automatic bake has finished and the glass has been redrawn.
import * as THREE from 'three/webgpu';
import { TeapotGeometry } from 'three/addons/geometries/TeapotGeometry.js';
import { autoGlassVolumesSettled, bakeGlassVolumeAsync, MeshGlassMaterial } from '../glass';

const query = new URLSearchParams(location.search);
const exits = query.get('exits') === 'baked' ? 'baked' : 'auto';
const instanced = query.get('instanced') === '1';

/** A vase turned on a lathe: an open profile (its rim and foot are capped by the bake). */
const vase = new THREE.LatheGeometry(
  [
    [0.0, -0.5],
    [0.3, -0.5],
    [0.36, -0.3],
    [0.22, 0.0],
    [0.16, 0.25],
    [0.24, 0.5],
  ].map(([x, y]) => new THREE.Vector2(x, y)),
  64,
);

const SOLIDS: { geometry: THREE.BufferGeometry; x: number; y: number; turn: number; scale: number }[] = [
  { geometry: new THREE.TorusKnotGeometry(0.4, 0.13, 160, 24), x: -1.6, y: 0, turn: 0.4, scale: 1 },
  { geometry: new TeapotGeometry(0.4, 8), x: 0, y: -0.1, turn: -0.6, scale: 1 },
  { geometry: vase, x: 1.6, y: -0.05, turn: 0, scale: 1 },
];

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
  const checker = new THREE.DataTexture(
    Uint8Array.from({ length: 64 * 4 }, (_, i) => {
      const p = i >> 2;
      return i % 4 === 3 ? 255 : ((p % 8) + (p >> 3)) % 2 ? 205 : 90;
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
  [0xe0453a, 0xf2b134, 0x3fae5c, 0x2f8fd8].forEach((color, i) => {
    const bar = new THREE.Mesh(new THREE.BoxGeometry(0.3, 1.8, 0.1), new THREE.MeshStandardMaterial({ color }));
    bar.position.set((i - 1.5) * 1.3, 0.3, -1.4);
    scene.add(bar);
  });

  const camera = new THREE.PerspectiveCamera(30, 1, 0.1, 100);
  camera.position.set(0, 1.4, 6.4);
  camera.lookAt(0, -0.05, 0);

  const baked = exits === 'baked' ? await Promise.all(SOLIDS.map((s) => bakeGlassVolumeAsync(s.geometry))) : [];
  const matrix = new THREE.Matrix4();
  SOLIDS.forEach((solid, i) => {
    const material = new MeshGlassMaterial({
      ior: 1.52,
      dispersion: 1,
      transmission: 1,
      thickness: 0.6,
      ...(exits === 'baked' ? { volume: baked[i] } : {}),
    });
    matrix.compose(
      new THREE.Vector3(solid.x, solid.y, 0),
      new THREE.Quaternion().setFromAxisAngle(new THREE.Vector3(0, 1, 0), solid.turn),
      new THREE.Vector3(solid.scale, solid.scale, solid.scale),
    );
    if (instanced) {
      const mesh = new THREE.InstancedMesh(solid.geometry, material, 1);
      mesh.setMatrixAt(0, matrix);
      scene.add(mesh);
    } else {
      const mesh = new THREE.Mesh(solid.geometry, material);
      mesh.applyMatrix4(matrix);
      scene.add(mesh);
    }
  });

  renderer.setSize(window.innerWidth, window.innerHeight, false);
  camera.aspect = window.innerWidth / window.innerHeight;
  camera.updateProjectionMatrix();
  let frames = 0;
  let settled = false;
  renderer.setAnimationLoop(() => {
    renderer.render(scene, camera);
    frames++;
    // The first frames start the automatic bakes; once they are done the glass is redrawn.
    if (frames === 2) {
      void autoGlassVolumesSettled().then(() => {
        settled = true;
        frames = 2;
      });
    }
    if (settled && frames === 5) document.documentElement.dataset.glassReady = 'true';
  });
}

main().catch((err) => {
  console.error(err);
  document.body.textContent = `Failed: ${err instanceof Error ? err.message : String(err)}`;
});
