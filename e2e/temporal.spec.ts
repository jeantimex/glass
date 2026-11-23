import { expect, test, type Page } from '@playwright/test';
import { PNG } from 'pngjs';

// MeshGlassMaterial's `temporal` option under three.js's TRAA (temporal-test.html): with the
// wavelengths moving every frame, TRAA's average gets closer to a continuous spectrum than
// the same number of fixed wavelengths does. Everything goes through TRAA, so its own
// filtering is the same in each render and only the wavelengths differ; the reference is
// 64 fixed wavelengths. Few wavelengths per frame (3) make the difference plain.

const VIEWPORT = { width: 1000, height: 420 };
/** The three glass pieces on screen: x0, x1, y0, y1. */
const PIECES = {
  sphere: [230, 380, 135, 285],
  octahedron: [415, 580, 125, 285],
  torus: [635, 750, 140, 280],
} as const;

async function render(page: Page, query: string): Promise<PNG> {
  await page.setViewportSize(VIEWPORT);
  await page.goto(`/temporal-test.html?${query}`);
  await page.waitForFunction(() => document.documentElement.dataset.glassReady === 'true', null, {
    timeout: 90_000,
  });
  return PNG.sync.read(await page.locator('#view').screenshot());
}

/** Mean difference (0-255, largest channel) over a box, compared in 8x8-pixel blocks. */
function meanBlockDifference(a: PNG, b: PNG, [x0, x1, y0, y1]: readonly number[]): number {
  let sum = 0;
  let blocks = 0;
  for (let by = y0; by + 8 <= y1; by += 8) {
    for (let bx = x0; bx + 8 <= x1; bx += 8) {
      const ma = [0, 0, 0];
      const mb = [0, 0, 0];
      for (let y = by; y < by + 8; y++) {
        for (let x = bx; x < bx + 8; x++) {
          const i = (y * a.width + x) * 4;
          for (let c = 0; c < 3; c++) {
            ma[c] += a.data[i + c] / 64;
            mb[c] += b.data[i + c] / 64;
          }
        }
      }
      sum += Math.max(...[0, 1, 2].map((c) => Math.abs(ma[c] - mb[c])));
      blocks++;
    }
  }
  return sum / blocks;
}

test.describe.configure({ mode: 'serial' });

test('temporal wavelengths under TRAA converge closer to a continuous spectrum than fixed ones', async ({ page }) => {
  const reference = await render(page, 'mode=traa&samples=64&temporal=0');
  const moving = await render(page, 'mode=traa&samples=3');
  const fixed = await render(page, 'mode=traa&samples=3&temporal=0');
  let movingTotal = 0;
  let fixedTotal = 0;
  for (const [name, box] of Object.entries(PIECES)) {
    const m = meanBlockDifference(moving, reference, box);
    const f = meanBlockDifference(fixed, reference, box);
    // Measured: sphere 2.3 / 3.7, octahedron 5.4 / 6.4, torus 1.2 / 1.5 (moving / fixed).
    expect(m, `${name}: moving wavelengths should not be further from the reference`).toBeLessThanOrEqual(f * 1.05);
    movingTotal += m;
    fixedTotal += f;
  }
  expect(movingTotal).toBeLessThan(fixedTotal * 0.9);
});

test('the temporal material under TRAA renders as expected', async ({ page }) => {
  await page.setViewportSize(VIEWPORT);
  await page.goto('/temporal-test.html?mode=traa');
  await page.waitForFunction(() => document.documentElement.dataset.glassReady === 'true', null, {
    timeout: 90_000,
  });
  await expect(page.locator('#view')).toHaveScreenshot('temporal-traa.png');
});
