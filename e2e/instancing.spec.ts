import { expect, test, type Page } from '@playwright/test';
import { PNG } from 'pngjs';

// MeshGlassMaterial on an InstancedMesh must look exactly like the same objects drawn as
// separate meshes (instancing.html): exits through a baked volume follow each instance's
// matrix, and per-instance colours (setColorAt) tint like per-mesh material colours.

async function render(page: Page, mode: 'meshes' | 'instanced', renderer = ''): Promise<Buffer> {
  await page.setViewportSize({ width: 900, height: 540 });
  await page.goto(`/instancing.html?mode=${mode}${renderer}`);
  await page.waitForFunction(() => document.documentElement.dataset.glassReady === 'true', null, {
    timeout: 60_000,
  });
  await page.waitForTimeout(300);
  return page.locator('#view').screenshot();
}

/** Share of pixels whose largest channel difference exceeds `threshold` (0-255). */
function differingShare(a: Buffer, b: Buffer, threshold: number): number {
  const pa = PNG.sync.read(a);
  const pb = PNG.sync.read(b);
  expect([pa.width, pa.height]).toEqual([pb.width, pb.height]);
  let differing = 0;
  for (let i = 0; i < pa.data.length; i += 4) {
    const d = Math.max(
      Math.abs(pa.data[i] - pb.data[i]),
      Math.abs(pa.data[i + 1] - pb.data[i + 1]),
      Math.abs(pa.data[i + 2] - pb.data[i + 2]),
    );
    if (d > threshold) differing++;
  }
  return differing / (pa.width * pa.height);
}

test('MeshGlassMaterial on an InstancedMesh matches the same objects as separate meshes', async ({ page }) => {
  const meshes = await render(page, 'meshes');
  const instanced = await render(page, 'instanced');
  // Same optics; only floating-point rounding differs between the two shader programs.
  expect(differingShare(meshes, instanced, 8)).toBeLessThan(0.001);
  expect(instanced).toMatchSnapshot('instancing-glass.png');
});
