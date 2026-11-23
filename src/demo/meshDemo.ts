// MeshGlassMaterial on ordinary THREE.Mesh objects in an ordinary three.js scene (no
// instanced renderer): lit opaque props behind the glass to refract, and four glass pieces
// using each way the material finds its exits. Opened as /mesh.html.
//   ?renderer=webgl  use WebGL 2 even where WebGPU is available
//   ?transmission=0  environment only (no scene refraction)
//   ?taa=1           three.js TRAA, with the glass's `temporal` option (smooth dispersion and edges)
import * as THREE from 'three/webgpu';
import { OrbitControls } from 'three/addons/controls/OrbitControls.js';
import { traa } from 'three/addons/tsl/display/TRAANode.js';
import { mrt, output, pass, velocity } from 'three/tsl';
import { TeapotGeometry } from 'three/addons/geometries/TeapotGeometry.js';
import { bakeGlassVolumeAsync, GlassBackfacePass, MeshGlassMaterial } from '../glass';

const query = new URLSearchParams(location.search);
const useTaa = query.get('taa') === '1';

/** A baked mesh from public/: [vertexCount, indexCount, pad] + positions + normals + indices. */
async function loadMesh(url: string): Promise<THREE.BufferGeometry> {
  const buffer = await (await fetch(url)).arrayBuffer();
  const [vertexCount, indexCount] = new Uint32Array(buffer, 0, 2);
  const geometry = new THREE.BufferGeometry();
  geometry.setAttribute('position', new THREE.BufferAttribute(new Float32Array(buffer, 12, vertexCount * 3), 3));
  geometry.setAttribute(
    'normal',
    new THREE.BufferAttribute(new Float32Array(buffer, 12 + vertexCount * 12, vertexCount * 3), 3),
  );
  geometry.setIndex(new THREE.BufferAttribute(new Uint32Array(buffer, 12 + vertexCount * 24, indexCount), 1));
  return geometry;
}

/** A baked 128^3 HalfFloat signed-distance volume from public/. */
async function loadVolume(url: string): Promise<THREE.Data3DTexture> {
  const data = new Uint16Array(await (await fetch(url)).arrayBuffer());
  const volume = new THREE.Data3DTexture(data, 128, 128, 128);
  volume.format = THREE.RedFormat;
  volume.type = THREE.HalfFloatType;
  volume.minFilter = volume.magFilter = THREE.LinearFilter;
  volume.wrapS = volume.wrapT = volume.wrapR = THREE.ClampToEdgeWrapping;
  volume.needsUpdate = true;
  return volume;
}

async function main() {
  const canvas = document.getElementById('view') as HTMLCanvasElement;
  const renderer = new THREE.WebGPURenderer({
    canvas,
    // (TRAA needs MSAA off.)
    antialias: !useTaa,
    forceWebGL: query.get('renderer') === 'webgl',
  });
  renderer.setPixelRatio(Math.min(window.devicePixelRatio, 2));
  renderer.toneMapping = THREE.ACESFilmicToneMapping;
  await renderer.init();

  const scene = new THREE.Scene();
  scene.background = new THREE.Color(0x2a2f38);
  const camera = new THREE.PerspectiveCamera(32, 1, 0.1, 100);
  camera.position.set(0, 1.6, 6.2);
  const orbit = new OrbitControls(camera, canvas);
  orbit.target.set(0, 0.55, 0);
  orbit.enableDamping = true;

  /* Opaque props to refract: a checkered floor, a back wall with colour bars, lights. */
  scene.add(new THREE.HemisphereLight(0xdde6ff, 0x30281f, 1.2));
  const sun = new THREE.DirectionalLight(0xffffff, 2.2);
  sun.position.set(-3, 5, 4);
  scene.add(sun);
  const checker = (() => {
    const size = 8;
    const data = new Uint8Array(size * size * 4);
    for (let y = 0; y < size; y++)
      for (let x = 0; x < size; x++)
        data.set((x + y) % 2 ? [205, 205, 205, 255] : [90, 90, 96, 255], (y * size + x) * 4);
    const tex = new THREE.DataTexture(data, size, size);
    tex.magFilter = THREE.NearestFilter;
    tex.wrapS = tex.wrapT = THREE.RepeatWrapping;
    tex.repeat.set(6, 6);
    tex.colorSpace = THREE.SRGBColorSpace;
    tex.needsUpdate = true;
    return tex;
  })();
  const floor = new THREE.Mesh(new THREE.PlaneGeometry(14, 14), new THREE.MeshStandardMaterial({ map: checker }));
  floor.rotation.x = -Math.PI / 2;
  scene.add(floor);
  const barColors = [0xe0453a, 0xf2b134, 0x3fae5c, 0x2f8fd8, 0x8c4fd1];
  barColors.forEach((color, i) => {
    const bar = new THREE.Mesh(new THREE.BoxGeometry(0.5, 2.4, 0.1), new THREE.MeshStandardMaterial({ color }));
    bar.position.set((i - 2) * 1.3, 1.2, -2.2);
    scene.add(bar);
  });

  /* Glass: one material setup, each way of finding exits. */
  const transmission = query.has('transmission') ? Number(query.get('transmission')) : 1;
  const base = { ior: 1.52, dispersion: 1, transmission, temporal: useTaa };
  // The Utah teapot as three.js builds it (open, unwelded patches; a separate lid, spout and
  // handle) baked into a solid glass volume right here, in a worker.
  const teapotGeometry = new TeapotGeometry(1, 10);
  const [suzanneGeometry, dragonGeometry, suzanneVolume, dragonVolume, teapotVolume] = await Promise.all([
    loadMesh('suzanne_mesh.bin'),
    loadMesh('dragon_mesh.bin'),
    loadVolume('suzanne_sdf_128.bin'),
    loadVolume('dragon_sdf_128.bin'),
    bakeGlassVolumeAsync(teapotGeometry),
  ]);
  suzanneGeometry.computeVertexNormals();
  const pieces: { mesh: THREE.Mesh; spin: number }[] = [];
  const place = (geometry: THREE.BufferGeometry, material: MeshGlassMaterial, x: number, y: number, scale: number) => {
    const mesh = new THREE.Mesh(geometry, material);
    mesh.position.set(x, y, 0);
    mesh.scale.setScalar(scale);
    scene.add(mesh);
    pieces.push({ mesh, spin: 0 });
    return mesh;
  };
  // A three.js built-in solid: traced exactly from the geometry's parameters (`exactShapes`).
  place(new THREE.IcosahedronGeometry(1, 12), new MeshGlassMaterial({ ...base, thickness: 1.1 }), -2.4, 0.55, 0.55);
  // First exits read from the back-face map (exact along the view ray), then estimated.
  const backfacePass = new GlassBackfacePass({ resolutionScale: 1 });
  place(
    new THREE.TorusKnotGeometry(0.62, 0.2, 220, 36),
    new MeshGlassMaterial({ ...base, thickness: 0.3, backfaceMap: backfacePass.texture }),
    -0.8,
    0.62,
    0.58,
  );
  // Exact exits through the baked volumes (mesh local coordinates are the volumes').
  place(
    suzanneGeometry,
    new MeshGlassMaterial({ ...base, thickness: 1, volume: { texture: suzanneVolume } }),
    0.8,
    0.6,
    0.6,
  ).rotation.y = -0.35;
  place(
    dragonGeometry,
    new MeshGlassMaterial({ ...base, thickness: 0.8, volume: { texture: dragonVolume } }),
    2.4,
    0.62,
    0.7,
  ).rotation.y = -0.6;
  // Exact exits through the volume baked at load time (bakeGlassVolumeAsync), in front.
  teapotGeometry.computeBoundingBox();
  const teapotScale = 0.36;
  const teapot = place(
    teapotGeometry,
    new MeshGlassMaterial({ ...base, thickness: 0.6, volume: teapotVolume }),
    0,
    -teapotGeometry.boundingBox!.min.y * teapotScale,
    teapotScale,
  );
  teapot.position.z = 1.4;
  teapot.rotation.y = -0.5;

  const resize = () => {
    renderer.setSize(window.innerWidth, window.innerHeight, false);
    camera.aspect = window.innerWidth / window.innerHeight;
    camera.updateProjectionMatrix();
  };
  window.addEventListener('resize', resize);
  resize();

  // With ?taa=1: the scene through three.js's TRAA, which averages frames (camera jitter for
  // smooth edges; the glass's moving wavelengths for smooth dispersion).
  let pipeline: THREE.RenderPipeline | null = null;
  if (useTaa) {
    pipeline = new THREE.RenderPipeline(renderer);
    const scenePass = pass(scene, camera);
    scenePass.setMRT(mrt({ output, velocity }));
    pipeline.outputNode = traa(
      scenePass.getTextureNode('output'),
      scenePass.getTextureNode('depth'),
      scenePass.getTextureNode('velocity'),
      camera,
    );
  }

  let frames = 0;
  renderer.setAnimationLoop(() => {
    orbit.update();
    backfacePass.render(renderer, scene, camera);
    if (pipeline) pipeline.render();
    else renderer.render(scene, camera);
    if (++frames === 3) document.documentElement.dataset.glassReady = 'true';
  });
}

main().catch((err) => {
  console.error(err);
  document.body.textContent = `Failed to start: ${err instanceof Error ? err.message : String(err)}`;
});
