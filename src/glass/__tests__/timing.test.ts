import { describe, expect, it } from 'vitest';
import type * as THREE from 'three/webgpu';
import { createPassTimer } from '../timing';

describe('createPassTimer (timing.ts)', () => {
  it('returns null when GPU timestamp tracking is disabled', async () => {
    const fakeRenderer = {
      backend: { trackTimestamp: false },
      info: { frame: 1, render: { frameCalls: 0 } },
    } as unknown as THREE.WebGPURenderer;

    const timer = createPassTimer(fakeRenderer);
    timer.beginFrame();
    timer.mark('Glass');
    timer.endFrame();
    expect(await timer.passTimes()).toBeNull();
  });

  it('aggregates GPU pass durations by frame call ranges', async () => {
    const timestamps = new Map<string, number>([
      ['r:1:pass:f10', 0.4],
      ['r:2:pass:f10', 1.2],
      ['r:3:pass:f10', 0.8],
      ['r:4:pass:f10', 0.6],
    ]);
    const fakeRenderer = {
      backend: {
        trackTimestamp: true,
        timestampQueryPool: { render: { timestamps } },
      },
      info: { frame: 10, render: { frameCalls: 0 } },
      resolveTimestampsAsync: async () => {},
    } as unknown as THREE.WebGPURenderer;

    const timer = createPassTimer(fakeRenderer);
    timer.beginFrame();
    timer.mark('Ground'); // call 1
    fakeRenderer.info.render.frameCalls = 1;
    timer.mark('Glass'); // calls 2, 3
    fakeRenderer.info.render.frameCalls = 3;
    timer.mark('Finishing'); // call 4
    timer.endFrame();

    const result = await timer.passTimes();
    expect(result).not.toBeNull();
    expect(result!.Ground).toBeCloseTo(0.4, 6);
    expect(result!.Glass).toBeCloseTo(2.0, 6);
    expect(result!.Finishing).toBeCloseTo(0.6, 6);
  });
});
