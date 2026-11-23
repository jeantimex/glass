import { expect, test, type Page } from '@playwright/test';
import { PNG } from 'pngjs';

// GlassLayers (layers-test.html): a front row of glass spheres before a row of glass boxes,
// colour bars behind both. With layers, the front glass sees the glass behind it (through the
// instanced renderer's own screen-space march); without glass behind, layers change little.

async function render(page: Page, query: string): Promise<PNG> {
  await page.setViewportSize({ width: 1000, height: 460 });
  await page.goto(`/layers-test.html?${query}`);
  await page.waitForFunction(() => document.documentElement.dataset.glassReady === 'true', null, {
    timeout: 60_000,
  });
  return PNG.sync.read(await page.locator('#view').screenshot());
}

/** Share of pixels whose largest channel difference exceeds 8 (0-255). */
function differingShare(a: PNG, b: PNG): number {
  let differing = 0;
  for (let i = 0; i < a.data.length; i += 4) {
    const d = Math.max(
      Math.abs(a.data[i] - b.data[i]),
      Math.abs(a.data[i + 1] - b.data[i + 1]),
      Math.abs(a.data[i + 2] - b.data[i + 2]),
    );
    if (d > 8) differing++;
  }
  return differing / (a.width * a.height);
}

test.describe.configure({ mode: 'serial' });

test('with layers, glass shows the glass behind it', async ({ page }) => {
  const layered = await render(page, 'layers=1');
  const plain = await render(page, 'layers=0');
  // Measured 5.2% of pixels change (the boxes seen refracted in the spheres).
  expect(differingShare(layered, plain)).toBeGreaterThan(0.03);
});

test('with no glass behind, layers change little', async ({ page }) => {
  const layered = await render(page, 'layers=1&back=0');
  const plain = await render(page, 'layers=0&back=0');
  // Measured 0.61%: neighbouring spheres seen at the edges (real glass through glass), and a
  // known difference along the bars' edges between the layer and three.js's own opaque viewport
  // texture in the far fallback (independent of the march; see ROADMAP).
  expect(differingShare(layered, plain)).toBeLessThan(0.01);
});

test('glass seen through glass renders as expected', async ({ page }) => {
  await page.setViewportSize({ width: 1000, height: 460 });
  await page.goto('/layers-test.html?layers=1');
  await page.waitForFunction(() => document.documentElement.dataset.glassReady === 'true', null, {
    timeout: 60_000,
  });
  await expect(page.locator('#view')).toHaveScreenshot('glass-layers.png');
});
