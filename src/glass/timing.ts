// Per-pass GPU timestamp profiler for WebGPURenderer (when `trackTimestamp: true`).
import * as THREE from 'three/webgpu';

export interface PassTimer {
  /** Begin a new frame's timing marks. */
  beginFrame(): void;
  /** Label subsequent draw calls in the current frame with `passName`. */
  mark(passName: string): void;
  /** Record the current frame's marks for asynchronous timestamp resolution. */
  endFrame(): void;
  /** Resolve GPU time (ms) per named pass for the most recent completed frame. */
  passTimes(): Promise<Record<string, number> | null>;
}

/** Create a per-pass GPU timestamp profiler bound to `renderer` (no-op when `trackTimestamp` is off). */
export function createPassTimer(renderer: THREE.WebGPURenderer): PassTimer {
  const enabled = (renderer.backend as unknown as { trackTimestamp?: boolean }).trackTimestamp === true;
  let currentMarks: [number, string][] | null = null;
  const markedFrames = new Map<number, [number, string][]>();

  return {
    beginFrame() {
      currentMarks = enabled ? [] : null;
    },
    mark(passName: string) {
      if (currentMarks) currentMarks.push([renderer.info.render.frameCalls + 1, passName]);
    },
    endFrame() {
      if (!currentMarks) return;
      markedFrames.set(renderer.info.frame, currentMarks);
      for (const frame of markedFrames.keys()) {
        if (frame < renderer.info.frame - 30) markedFrames.delete(frame);
      }
    },
    async passTimes(): Promise<Record<string, number> | null> {
      if (!enabled) return null;
      await renderer.resolveTimestampsAsync(THREE.TimestampQuery.RENDER);
      const pool = (
        renderer.backend as unknown as {
          timestampQueryPool: Record<string, { timestamps: Map<string, number> } | null>;
        }
      ).timestampQueryPool.render;
      if (!pool) return null;

      const frames = new Map<number, Map<number, number>>();
      for (const [uid, ms] of pool.timestamps) {
        const match = uid.match(/^r:(\d+):.*:f(\d+)$/);
        if (!match) continue;
        const frame = Number(match[2]);
        if (!frames.has(frame)) frames.set(frame, new Map());
        frames.get(frame)!.set(Number(match[1]), ms);
      }

      const latestFrame = [...frames.keys()].filter((f) => markedFrames.has(f)).sort((a, b) => b - a)[0];
      if (latestFrame === undefined) return null;

      const frameMarks = markedFrames.get(latestFrame)!;
      const totals: Record<string, number> = {};
      for (const [callIndex, ms] of frames.get(latestFrame)!) {
        let label = 'Other';
        for (const [fromCall, passName] of frameMarks) {
          if (callIndex >= fromCall) label = passName;
        }
        totals[label] = (totals[label] ?? 0) + ms;
      }
      return totals;
    },
  };
}
