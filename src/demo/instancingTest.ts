// Test page (instancing.html): the same glass objects drawn as separate THREE.Mesh objects
// (?mode=meshes) or as one THREE.InstancedMesh (?mode=instanced), which must look the same.
// Three Suzannes with a baked volume, each with its own position, rotation, scale and colour
// (per-mesh material colours, or InstancedMesh.setColorAt), over a checkered floor to refract.
import * as THREE from 'three/webgpu';
import { MeshGlassMaterial } from '../glass';

const query = new URLSearchParams(location.search);
const mode = query.get('mode') === 'instanced' ? 'instanced' : 'meshes';

async function loadMesh(url: string): Promise<THREE.BufferGeometry> {
  const buffer = await (await fetch(url)).arrayBuffer();
  const [vertexCount, indexCount] = new Uint32Array(buffer, 0, 2);
  const geometry = new THREE.BufferGeometry();
  geometry.setAttribute('position', new THREE.BufferAttribute(new Float32Array(buffer, 12, vertexCount * 3), 3));
  geometry.setIndex(new THREE.BufferAttribute(new Uint32Array(buffer, 12 + vertexCount * 24, indexCount), 1));
  geometry.computeVertexNormals();
  return geometry;
}

async function loadVolume(url: string): Promise<THREE.Data3DTexture> {
  const volume = new THREE.Data3DTexture(new Uint16Array(await (await fetch(url)).arrayBuffer()), 128, 128, 128);
  volume.format = THREE.RedFormat;
  volume.type = THREE.HalfFloatType;
  volume.minFilter = volume.magFilter = THREE.LinearFilter;
  volume.needsUpdate = true;
  return volume;
}

/** Position, turn (radians about y, then x) and uniform scale of each object, and its colour. */
const OBJECTS = [
  { position: [-1.5, 0.55, 0], turn: [0.5, 0], scale: 0.55, color: 0xffffff },
  { position: [0, 0.7, -0.3], turn: [-0.4, 0.3], scale: 0.7, color: 0x99ddff },
  { position: [1.5, 0.45, 0.2], turn: [2.6, -0.2], scale: 0.45, color: 0xffcc88 },
] as const;

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
  checker.repeat.set(5, 5);
  checker.colorSpace = THREE.SRGBColorSpace;
  checker.needsUpdate = true;
  const floor = new THREE.Mesh(new THREE.PlaneGeometry(10, 10), new THREE.MeshStandardMaterial({ map: checker }));
  floor.rotation.x = -Math.PI / 2;
  scene.add(floor);

  const camera = new THREE.PerspectiveCamera(32, 1, 0.1, 100);
  camera.position.set(0, 1.5, 5.2);
  camera.lookAt(0, 0.5, 0);

  const [geometry, texture] = await Promise.all([loadMesh('suzanne_mesh.bin'), loadVolume('suzanne_sdf_128.bin')]);
  const glass = (color: number) =>
    new MeshGlassMaterial({ ior: 1.52, dispersion: 1, transmission: 1, thickness: 1, color, volume: { texture } });
  const matrix = new THREE.Matrix4();
  const matrixOf = (o: (typeof OBJECTS)[number]) =>
    matrix.compose(
      new THREE.Vector3(...o.position),
      new THREE.Quaternion().setFromEuler(new THREE.Euler(o.turn[1], o.turn[0], 0, 'YXZ')),
      new THREE.Vector3(o.scale, o.scale, o.scale),
    );

  if (mode === 'instanced') {
    // One material (white), the colours per instance.
    const instanced = new THREE.InstancedMesh(geometry, glass(0xffffff), OBJECTS.length);
    OBJECTS.forEach((o, i) => {
      instanced.setMatrixAt(i, matrixOf(o));
      instanced.setColorAt(i, new THREE.Color(o.color));
    });
    scene.add(instanced);
  } else {
    for (const o of OBJECTS) {
      const mesh = new THREE.Mesh(geometry, glass(o.color));
      mesh.applyMatrix4(matrixOf(o));
      scene.add(mesh);
    }
  }

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
