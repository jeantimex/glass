// Web Worker for bakeGlassVolumeAsync (volume.ts): bakes one mesh off the main thread.
import { bakeGlassVolume, type BakeOptions } from './bake.ts';

interface BakeRequest {
  positions: Float32Array;
  indices: Uint32Array | null;
  options: BakeOptions;
}

self.addEventListener('message', (event: MessageEvent<BakeRequest>) => {
  const { positions, indices, options } = event.data;
  const result = bakeGlassVolume(positions, indices, options);
  (self as unknown as Worker).postMessage(result, [result.data.buffer]);
});
