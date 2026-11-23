import { expect, test, type Page } from '@playwright/test';
import { PNG } from 'pngjs';

// GlassCaustics in a plain three.js scene (caustics-test.html): a sphere, a box and a teapot
// (automatic volume) on a MeshStandardMaterial floor under one DirectionalLight.

async function render(page: Page, query: string): Promise<Buffer> {
  await page.setViewportSize({ width: 1000, height: 560 });
  await page.goto(`/caustics-test.html?${query}`);
  await page.waitForFunction(() => document.documentElement.dataset.glassReady === 'true', null, {
    timeout: 90_000,
  });
  return page.locator('#view').screenshot();
}

/** Share of pixels whose brightness differs by more than `threshold` (0-255). */
function differingShare(a: Buffer, b: Buffer, threshold: number): number {
  const pa = PNG.sync.read(a);
  const pb = PNG.sync.read(b);
  let differing = 0;
  for (let i = 0; i < pa.data.length; i += 4) {
    const la = (pa.data[i] + pa.data[i + 1] + pa.data[i + 2]) / 3;
    const lb = (pb.data[i] + pb.data[i + 1] + pb.data[i + 2]) / 3;
    if (Math.abs(la - lb) > threshold) differing++;
  }
  return differing / (pa.width * pa.height);
}

test.describe.configure({ mode: 'serial' });

test('glass that bends nothing (IOR 1) leaves the floor as if it were not there', async ({ page }) => {
  // The glass itself is not drawn, so only the floor's lighting is compared.
  const caustics = await render(page, 'ior=1.0001&dispersion=0&glass=0');
  const none = await render(page, 'caustics=0&glass=0');
  // Measured 8 pixels in 560,000 (an edge of the teapot's spout the volume rounds away).
  expect(differingShare(caustics, none, 6)).toBeLessThan(0.0005);
});

test('real glass casts shadows and caustics on the floor', async ({ page }) => {
  const caustics = await render(page, 'glass=0');
  const none = await render(page, 'caustics=0&glass=0');
  expect(differingShare(caustics, none, 6)).toBeGreaterThan(0.01);
  expect(await render(page, '')).toMatchSnapshot('glass-caustics.png');
});
