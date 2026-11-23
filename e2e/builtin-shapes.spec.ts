import { expect, test, type Page } from '@playwright/test';
import { PNG } from 'pngjs';

// MeshGlassMaterial on three.js built-in solids (shapes-test.html), traced exactly from the
// geometries' parameters:
// - against volumes baked from the same geometries, an independent route to the same answer
//   (rounded at voxel scale): each shape must agree, which catches a wrong frame (e.g. the
//   torus, which three.js builds around z) or a wrong size convention;
// - as a one-instance InstancedMesh against a Mesh: identical.

const VIEWPORT = { width: 1400, height: 420 };
/** Where each solid is on screen (fixed camera and viewport): its centre x, and the rows it spans. */
const SHAPES = [
  { name: 'sphere', x: 388 },
  { name: 'box', x: 495 },
  { name: 'cylinder', x: 597 },
  { name: 'cone', x: 700 },
  { name: 'capsule', x: 803 },
  { name: 'torus', x: 906 },
  { name: 'octahedron', x: 1010 },
];
const ROWS = [145, 275];
const HALF_WIDTH = 52;

async function render(page: Page, query: string): Promise<PNG> {
  await page.setViewportSize(VIEWPORT);
  await page.goto(`/shapes-test.html?${query}`);
  await page.waitForFunction(() => document.documentElement.dataset.glassReady === 'true', null, {
    timeout: 60_000,
  });
  await page.waitForTimeout(300);
  return PNG.sync.read(await page.locator('#view').screenshot());
}

/**
 * Mean difference (0-255, largest channel) between two renders over a box, compared at a
 * quarter of the resolution (4x4 blocks averaged), so a volume's voxel-scale speckle does not
 * count, only differences in what the glass shows.
 */
function meanBlockDifference(a: PNG, b: PNG, x0: number, x1: number, y0: number, y1: number): number {
  let sum = 0;
  let blocks = 0;
  for (let by = y0; by + 4 <= y1; by += 4) {
    for (let bx = x0; bx + 4 <= x1; bx += 4) {
      const ma = [0, 0, 0];
      const mb = [0, 0, 0];
      for (let y = by; y < by + 4; y++) {
        for (let x = bx; x < bx + 4; x++) {
          const i = (y * a.width + x) * 4;
          for (let c = 0; c < 3; c++) {
            ma[c] += a.data[i + c] / 16;
            mb[c] += b.data[i + c] / 16;
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

test('built-in solids traced exactly agree with volumes baked from the same geometries', async ({ page }) => {
  const exact = await render(page, 'exits=exact');
  const volume = await render(page, 'exits=volume');
  for (const shape of SHAPES) {
    const d = meanBlockDifference(exact, volume, shape.x - HALF_WIDTH, shape.x + HALF_WIDTH, ROWS[0], ROWS[1]);
    // Correct exits differ from the voxel-rounded volume by under 6; a wrong frame by over 20.
    expect(d, `${shape.name}: mean difference from its baked volume`).toBeLessThan(10);
  }
});

test('a built-in solid on an InstancedMesh matches the same solid as a Mesh', async ({ page }) => {
  const meshes = await render(page, 'exits=exact');
  const instanced = await render(page, 'exits=exact&instanced=1');
  let differing = 0;
  for (let i = 0; i < meshes.data.length; i += 4) {
    const d = Math.max(
      Math.abs(meshes.data[i] - instanced.data[i]),
      Math.abs(meshes.data[i + 1] - instanced.data[i + 1]),
      Math.abs(meshes.data[i + 2] - instanced.data[i + 2]),
    );
    if (d > 8) differing++;
  }
  expect(differing / (meshes.width * meshes.height)).toBeLessThan(0.001);
});

test('built-in solids render as expected', async ({ page }) => {
  await page.setViewportSize(VIEWPORT);
  await page.goto('/shapes-test.html?exits=exact');
  await page.waitForFunction(() => document.documentElement.dataset.glassReady === 'true', null, {
    timeout: 60_000,
  });
  await page.waitForTimeout(300);
  await expect(page.locator('#view')).toHaveScreenshot('builtin-shapes-exact.png');
});
