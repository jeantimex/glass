// Rigid-body physics on the GPU (WebGPU compute), for piles too large for Box3D.
// See gpu.wgsl for how it works. Like box3d.ts it owns no rendering: it
// writes positions and rotations into the shared GlassObjects arrays, a frame or
// two after the GPU has worked them out.
import { KIND, MAX_OBJECTS, type GlassObjects } from '../glass';
import type { Pile } from './pile';
import { spawnObjects, meshExtremes } from './spawn';
import source from './gpu.wgsl?raw';

const GRID_CELLS = 64 * 32 * 64;
const SLOTS = 32;
/**
 * Longest substep (s): shorter makes deep piles firmer, and costs GPU time. A frame
 * is cut into as many substeps as that takes, up to MAX_SUBSTEPS (?substeps= fixes it).
 */
const SUBSTEP = 1 / 240;
const MAX_SUBSTEPS = 4;
const FIXED_SUBSTEPS = Math.min(Math.max(Number(new URLSearchParams(location.search).get('substeps')) || 0, 0), 16);
const WORKGROUP = 64;
// WebGPU's flag values (the TypeScript library has the types but not these constants).
const GPUBufferUsage = { MAP_READ: 1, COPY_SRC: 4, COPY_DST: 8, UNIFORM: 64, STORAGE: 128 };
const GPUShaderStage = { COMPUTE: 4 };
const GPUMapMode = { READ: 1 };

/** Half-extents of a box about as big as the shape, for its mass and how it spins. */
function roughBox(kind: number, x: number, y: number, z: number): [number, number, number] {
  switch (kind) {
    case KIND.box:
      return [x, y, z];
    case KIND.sphere:
      return [x * 0.8, x * 0.8, x * 0.8];
    case KIND.cylinder:
    case KIND.hexprism:
      return [x, y, x];
    case KIND.capsule:
      return [x, y + x, x];
    case KIND.cone:
      return [x * 0.7, y, x * 0.7];
    case KIND.torus:
      return [x + y, y, x + y];
    case KIND.octahedron:
      return [x * 0.6, x * 0.6, x * 0.6];
    case KIND.prism:
      return [x * 0.7, x * 0.7, z];
    case KIND.suzanne:
      return [x, x * 0.7, x * 0.6];
    case KIND.dragon:
      return [x, x * 0.7, x * 0.45];
    default:
      return [x, x * 0.6, x]; // diamond
  }
}

/**
 * @param hulls points (x, y, z, ...) of the convex hulls of the shapes made from
 *   meshes, at half-width 1
 */
export function createGpuPile(device: GPUDevice, objects: GlassObjects, hulls: { suzanne: Float32Array; dragon: Float32Array }) {
  // The shapes made from meshes collide as the convex shape spanned by their
  // extreme points.
  const suzanne = meshExtremes(hulls.suzanne);
  const dragon = meshExtremes(hulls.dragon);
  const reachOf = (points: number[]) =>
    Math.sqrt(Math.max(...Array.from({ length: points.length / 3 }, (_, k) => points[k * 3] ** 2 + points[k * 3 + 1] ** 2 + points[k * 3 + 2] ** 2)));
  const suzanneReach = reachOf(suzanne.points);
  const dragonReach = reachOf(dragon.points);
  const suzannePoints = suzanne.points.length / 3;
  const dragonPoints = dragon.points.length / 3;
  const meshTables = [
    `const SUZ_POINTS_N = ${suzannePoints};`,
    `const DRAGON_POINTS_N = ${dragonPoints};`,
    `const MESH_POINTS_N = ${suzannePoints + dragonPoints};`,
  ].join('\n');
  // The tables themselves go in a uniform buffer (see Tables in gpu.wgsl):
  // 8 directions, the meshes' points, then 26 planes for each mesh.
  const tableData = new Float32Array((8 + suzannePoints + dragonPoints + 52) * 4);
  for (let k = 0; k < 8; k++) tableData.set([Math.cos((k * Math.PI) / 4), Math.sin((k * Math.PI) / 4), 0, 0], k * 4);
  [...suzanne.points, ...dragon.points].forEach((v, k) => (tableData[(8 + Math.floor(k / 3)) * 4 + (k % 3)] = v));
  tableData.set([...suzanne.planes, ...dragon.planes], (8 + suzannePoints + dragonPoints) * 4);
  const module = device.createShaderModule({ code: source.replace('/*SUZANNE*/', meshTables) });
  module.getCompilationInfo().then((info) => {
    for (const m of info.messages) if (m.type === 'error') console.error(`gpu.wgsl:${m.lineNum}: ${m.message}`);
  });

  const storage = GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_DST;
  const params = device.createBuffer({ size: 16, usage: GPUBufferUsage.UNIFORM | GPUBufferUsage.COPY_DST });
  const state = device.createBuffer({ size: MAX_OBJECTS * 2 * 16, usage: storage | GPUBufferUsage.COPY_SRC });
  const work = device.createBuffer({ size: MAX_OBJECTS * 8 * 16, usage: storage });
  const gridCount = device.createBuffer({ size: GRID_CELLS * 4, usage: storage });
  const gridCells = device.createBuffer({ size: GRID_CELLS * SLOTS * 4, usage: storage });
  const tables = device.createBuffer({ size: tableData.byteLength, usage: GPUBufferUsage.UNIFORM | GPUBufferUsage.COPY_DST });
  // The bodies to solve each substep (see markActive in gpu.wgsl).
  const activeMark = device.createBuffer({ size: MAX_OBJECTS * 4, usage: storage });
  const activeList = device.createBuffer({ size: MAX_OBJECTS * 4, usage: storage });
  const activeCount = device.createBuffer({ size: 4, usage: storage });
  const solveSize = device.createBuffer({ size: 12, usage: storage | 0x0100 /* INDIRECT */ });
  device.queue.writeBuffer(tables, 0, tableData);

  const layout = device.createBindGroupLayout({
    entries: [
      { binding: 0, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'uniform' } },
      ...[1, 2, 3, 4].map((binding) => ({ binding, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'storage' as const } })),
      { binding: 5, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'uniform' } },
      ...[6, 7, 8].map((binding) => ({ binding, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'storage' as const } })),
    ],
  });
  const bindGroup = device.createBindGroup({
    layout,
    entries: [params, state, work, gridCount, gridCells, tables, activeMark, activeList, activeCount].map((buffer, binding) => ({ binding, resource: { buffer } })),
  });
  const pipelineLayout = device.createPipelineLayout({ bindGroupLayouts: [layout] });
  const pipeline = (entryPoint: string) => device.createComputePipeline({ layout: pipelineLayout, compute: { module, entryPoint } });
  // sizeSolve alone also writes the solve's workgroup counts.
  const sizeLayout = device.createBindGroupLayout({ entries: [{ binding: 0, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'storage' } }] });
  const sizeBindGroup = device.createBindGroup({ layout: sizeLayout, entries: [{ binding: 0, resource: { buffer: solveSize } }] });
  const sizePipelineLayout = device.createPipelineLayout({ bindGroupLayouts: [layout, sizeLayout] });
  const integrate = pipeline('integrate');
  const clearGrid = pipeline('clearGrid');
  const fillGrid = pipeline('fillGrid');
  const solve = pipeline('solve');
  const clearActive = pipeline('clearActive');
  const markActive = pipeline('markActive');
  const compact = pipeline('compact');
  const sizeSolve = device.createComputePipeline({ layout: sizePipelineLayout, compute: { module, entryPoint: 'sizeSolve' } });

  let count = 0;
  let cell = 1;
  let footprint = 2;
  let lastStep = 0;
  /** Bumped by reset, so results of the previous pile still on their way are dropped. */
  let generation = 0;
  /** Set when positions arrived from the GPU since the last step(). */
  let arrived = 0;
  let arrivedVisibly = 0; // of those, how many by enough to see (over a millimetre)
  // A pile that has come to rest is no longer worked on: since when hardly anything
  // has moved, and whether it has settled for good (until it is dropped again).
  let quietSince = 0;
  let settled = false;
  // Timing on the GPU (where the work is: the CPU only hands it over).
  const timing = device.features.has('timestamp-query')
    ? {
        queries: device.createQuerySet({ type: 'timestamp', count: 2 }),
        resolved: device.createBuffer({ size: 16, usage: 0x0200 /* QUERY_RESOLVE */ | GPUBufferUsage.COPY_SRC }),
        read: device.createBuffer({ size: 16, usage: GPUBufferUsage.MAP_READ | GPUBufferUsage.COPY_DST }),
        frame: 0,
        busy: false,
        ms: 0,
      }
    : null;
  const freeReadbacks: GPUBuffer[] = [];
  let readbacksInFlight = 0;
  const paramData = new ArrayBuffer(16);
  const paramU32 = new Uint32Array(paramData);
  const paramF32 = new Float32Array(paramData);

  function reset(wanted: number, only: number[] | null = null, cubic = false, adjust?: () => void) {
    footprint = spawnObjects(objects, wanted, only, cubic, adjust);
    count = objects.count;
    generation += 1;
    lastStep = 0;
    quietSince = 0;
    settled = false;

    const stateData = new Float32Array(count * 8);
    const workData = new Float32Array(count * 32);
    let largest = 0;
    for (let i = 0; i < count; i++) {
      const kind = objects.kind[i];
      const [sx, sy, sz] = objects.size.subarray(i * 3, i * 3 + 3);
      // Radius of a ball round the centre that holds the whole shape.
      let bound: number;
      if (kind === KIND.box) bound = Math.hypot(sx, sy, sz);
      else if (kind === KIND.sphere || kind === KIND.octahedron) bound = sx;
      else if (kind === KIND.cylinder || kind === KIND.cone) bound = Math.hypot(sx, sy);
      else if (kind === KIND.capsule || kind === KIND.torus) bound = sx + sy;
      else if (kind === KIND.prism) bound = Math.hypot(sx, sz);
      else if (kind === KIND.hexprism) bound = Math.hypot(sx * 1.1547, sy);
      else if (kind === KIND.suzanne) bound = suzanneReach * sx;
      else if (kind === KIND.dragon) bound = dragonReach * sx;
      else bound = 1.0824 * sx;
      bound += 0.01;
      largest = Math.max(largest, bound);

      const [ex, ey, ez] = roughBox(kind, sx, sy, sz);
      const mass = 8 * ex * ey * ez;
      stateData.set(objects.center.subarray(i * 3, i * 3 + 3), i * 8);
      stateData[i * 8 + 3] = bound;
      stateData.set(objects.quat.subarray(i * 4, i * 4 + 4), i * 8 + 4);
      const w = i * 32;
      workData.set([sx, sy, sz, kind], w + 24);
      workData.set([3 / (mass * (ey * ey + ez * ez)), 3 / (mass * (ex * ex + ez * ez)), 3 / (mass * (ex * ex + ey * ey)), 1 / mass], w + 28);
    }
    cell = 2 * largest;
    device.queue.writeBuffer(state, 0, stateData);
    device.queue.writeBuffer(work, 0, workData);
  }

  /** Fetch the positions the GPU has just worked out; they arrive a frame or two later. */
  function readBack() {
    if (readbacksInFlight >= 3) return;
    const bytes = count * 32;
    const buffer = freeReadbacks.pop() ?? device.createBuffer({ size: MAX_OBJECTS * 32, usage: GPUBufferUsage.MAP_READ | GPUBufferUsage.COPY_DST });
    const encoder = device.createCommandEncoder();
    encoder.copyBufferToBuffer(state, 0, buffer, 0, bytes);
    device.queue.submit([encoder.finish()]);
    const forGeneration = generation;
    readbacksInFlight += 1;
    buffer
      .mapAsync(GPUMapMode.READ, 0, bytes)
      .then(() => {
        if (forGeneration === generation) {
          const data = new Float32Array(buffer.getMappedRange(0, bytes));
          const n = bytes / 32;
          let moved = 0;
          let movedVisibly = 0;
          for (let i = 0; i < n; i++) {
            const s = i * 8;
            const c = i * 3;
            const q = i * 4;
            const change = Math.abs(data[s] - objects.center[c]) + Math.abs(data[s + 1] - objects.center[c + 1]) + Math.abs(data[s + 2] - objects.center[c + 2]) + Math.abs(data[s + 4] - objects.quat[q]) + Math.abs(data[s + 5] - objects.quat[q + 1]);
            if (change > 2e-4) moved += 1;
            if (change > 1e-3) movedVisibly += 1;
            objects.center[c] = data[s];
            objects.center[c + 1] = data[s + 1];
            objects.center[c + 2] = data[s + 2];
            objects.quat[q] = data[s + 4];
            objects.quat[q + 1] = data[s + 5];
            objects.quat[q + 2] = data[s + 6];
            objects.quat[q + 3] = data[s + 7];
          }
          arrived += moved;
          arrivedVisibly += movedVisibly;
        }
        buffer.unmap();
        freeReadbacks.push(buffer);
      })
      .catch(() => buffer.destroy())
      .finally(() => (readbacksInFlight -= 1));
  }

  /** Advance by the time since the last call. Returns how many objects moved. */
  function step(): number {
    if (!count || settled) return 0;
    const now = performance.now();
    // Real time, but never more than 1/30 s at once (after a pause, or a slow frame).
    const dt = lastStep ? Math.min(Math.max((now - lastStep) / 1000, 1 / 240), 1 / 30) : 1 / 60;
    lastStep = now;
    paramU32[0] = count;
    const substeps = FIXED_SUBSTEPS || Math.min(Math.max(Math.round(dt / SUBSTEP), 1), MAX_SUBSTEPS);
    paramF32[1] = dt / substeps;
    paramF32[2] = cell;
    device.queue.writeBuffer(params, 0, paramData);

    const bodyGroups = Math.ceil(count / WORKGROUP);
    const encoder = device.createCommandEncoder();
    // Every so often, the time the GPU takes over it, if it can tell.
    const timeIt = timing !== null && !timing.busy && ++timing.frame % 4 === 0;
    const pass = encoder.beginComputePass(
      timeIt ? { timestampWrites: { querySet: timing!.queries, beginningOfPassWriteIndex: 0, endOfPassWriteIndex: 1 } } : undefined,
    );
    pass.setBindGroup(0, bindGroup);
    for (let s = 0; s < substeps; s++) {
      pass.setPipeline(integrate);
      pass.dispatchWorkgroups(bodyGroups);
      pass.setPipeline(clearGrid);
      pass.dispatchWorkgroups(GRID_CELLS / WORKGROUP);
      pass.setPipeline(fillGrid);
      pass.dispatchWorkgroups(bodyGroups);
      // Solve only the bodies that move, or that something moving may touch.
      pass.setPipeline(clearActive);
      pass.dispatchWorkgroups(bodyGroups);
      pass.setPipeline(markActive);
      pass.dispatchWorkgroups(bodyGroups);
      pass.setPipeline(compact);
      pass.dispatchWorkgroups(bodyGroups);
      pass.setPipeline(sizeSolve);
      pass.setBindGroup(1, sizeBindGroup);
      pass.dispatchWorkgroups(1);
      pass.setPipeline(solve);
      pass.dispatchWorkgroupsIndirect(solveSize, 0);
    }
    pass.end();
    if (timeIt) {
      encoder.resolveQuerySet(timing!.queries, 0, 2, timing!.resolved, 0);
      encoder.copyBufferToBuffer(timing!.resolved, 0, timing!.read, 0, 16);
    }
    device.queue.submit([encoder.finish()]);
    if (timeIt) {
      const t = timing!;
      t.busy = true;
      t.read
        .mapAsync(GPUMapMode.READ)
        .then(() => {
          const [start, end] = new BigUint64Array(t.read.getMappedRange());
          const ms = Number(end - start) / 1e6;
          t.ms = t.ms ? t.ms + (ms - t.ms) * 0.5 : ms; // a little smoothed
          t.read.unmap();
        })
        .catch(() => {})
        .finally(() => (t.busy = false));
    }
    readBack();

    const moved = arrived;
    const movedVisibly = arrivedVisibly;
    arrived = arrivedVisibly = 0;
    // Settled: for a second and a half, no more than one in 400 has moved by as much
    // as a millimetre. (A few never quite stop: they rock or tremble on the spot for
    // ever.) Held exactly where they are, they look no different, the trembling
    // stops, and the GPU is free for the picture.
    if (movedVisibly > Math.max(2, count / 400)) quietSince = 0;
    else if (!quietSince) quietSince = now;
    else if (now - quietSince > 1500) settled = true;
    return moved;
  }

  /**
   * Let go of the objects: positions still on their way back from the GPU are
   * dropped when they arrive. Without this they would land on whatever has taken
   * the place of this pile in the shared arrays (a single object, say).
   */
  function release() {
    generation += 1;
    count = 0;
  }

  /** How long the GPU took over a step lately (ms), or null if it cannot tell (or nothing is moving). */
  const gpuMs = () => (timing && !settled && count ? timing.ms : null);
  const pile: Pile = { reset, step, footprint: () => footprint, label: () => 'GPU' };
  return { ...pile, release, gpuMs };
}
