// Test page (shapes-test.html): three.js built-in solids with MeshGlassMaterial, in a row.
//   ?exits=exact     traced exactly from the geometries' parameters (the default)
//   ?exits=volume    traced through volumes baked from the same geometries (an independent
//                    route to the same answer, rounded at voxel scale)
//   ?instanced=1     each drawn as a one-instance InstancedMesh instead of a Mesh
// Each is turned and scaled (uniformly), and the camera looks down a little, so every shape's
// frame is exercised. Clear glass over a checkered floor, refracting it (transmission).
import * as THREE from 'three/webgpu';
import { bakeGlassVolumeAsync, MeshGlassMaterial } from '../glass';

const query = new URLSearchParams(location.search);
const exits = query.get('exits') === 'volume' ? 'volume' : 'exact';
const instanced = query.get('instanced') === '1';

/** The solids, and each one's position, turn (radians about y, then x) and uniform scale. */
const SOLIDS: { geometry: THREE.BufferGeometry; x: number; turn: [number, number]; scale: number }[] = [
  { geometry: new THREE.SphereGeometry(0.5, 64, 32), x: -3.3, turn: [0, 0], scale: 1 },
  { geometry: new THREE.BoxGeometry(0.8, 0.8, 0.8), x: -2.2, turn: [0.6, 0.35], scale: 1 },
  { geometry: new THREE.CylinderGeometry(0.4, 0.4, 0.9, 96), x: -1.1, turn: [0.3, 0.4], scale: 1 },
  { geometry: new THREE.ConeGeometry(0.45, 0.9, 96), x: 0, turn: [0.2, 0.3], scale: 1 },
  { geometry: new THREE.CapsuleGeometry(0.3, 0.5, 16, 64), x: 1.1, turn: [0, 0.5], scale: 1 },
  { geometry: new THREE.TorusGeometry(0.36, 0.14, 48, 128), x: 2.2, turn: [0.5, 0.6], scale: 1 },
  { geometry: new THREE.OctahedronGeometry(1), x: 3.3, turn: [0.4, 0.2], scale: 0.5 },
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
  checker.repeat.set(8, 4);
  checker.colorSpace = THREE.SRGBColorSpace;
  checker.needsUpdate = true;
  const floor = new THREE.Mesh(new THREE.PlaneGeometry(10, 5), new THREE.MeshStandardMaterial({ map: checker }));
  floor.rotation.x = -Math.PI / 2;
  floor.position.y = -0.6;
  scene.add(floor);
  // Colour bars behind, so refraction and internal reflection show plainly.
  [0xe0453a, 0xf2b134, 0x3fae5c, 0x2f8fd8, 0x8c4fd1].forEach((color, i) => {
    const bar = new THREE.Mesh(new THREE.BoxGeometry(0.35, 2, 0.1), new THREE.MeshStandardMaterial({ color }));
    bar.position.set((i - 2) * 1.6, 0.3, -1.6);
    scene.add(bar);
  });

  const camera = new THREE.PerspectiveCamera(30, 1, 0.1, 100);
  camera.position.set(0, 1.6, 8.2);
  camera.lookAt(0, -0.05, 0);

  const volumes = exits === 'volume' ? await Promise.all(SOLIDS.map((s) => bakeGlassVolumeAsync(s.geometry))) : [];
  const matrix = new THREE.Matrix4();
  SOLIDS.forEach((solid, i) => {
    const material = new MeshGlassMaterial({
      ior: 1.52,
      dispersion: 1,
      transmission: 1,
      thickness: 0.8,
      ...(exits === 'volume' ? { volume: volumes[i] } : {}),
    });
    matrix.compose(
      new THREE.Vector3(solid.x, 0, 0),
      new THREE.Quaternion().setFromEuler(new THREE.Euler(solid.turn[1], solid.turn[0], 0, 'YXZ')),
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
  renderer.setAnimationLoop(() => {
    renderer.render(scene, camera);
    if (++frames === 3) document.documentElement.dataset.glassReady = 'true';
  });
}

main().catch((err) => {
  console.error(err);
  document.body.textContent = `Failed: ${err instanceof Error ? err.message : String(err)}`;
});
