import { expect, test, type Page } from '@playwright/test';
import { PNG } from 'pngjs';

// MeshGlassMaterial's default `volume: 'auto'` (auto-test.html): geometries with no exact
// formula (a torus knot, the Utah teapot, a lathed vase) get volumes baked automatically, and
// must render exactly as with volumes baked explicitly from the same geometries, on a Mesh and
// on an InstancedMesh.

async function render(page: Page, query: string): Promise<Buffer> {
  await page.setViewportSize({ width: 1000, height: 420 });
  await page.goto(`/auto-test.html?${query}`);
  await page.waitForFunction(() => document.documentElement.dataset.glassReady === 'true', null, {
    timeout: 90_000,
  });
  await page.waitForTimeout(300);
  return page.locator('#view').screenshot();
}

/** Share of pixels whose largest channel difference exceeds `threshold` (0-255). */
function differingShare(a: Buffer, b: Buffer, threshold: number): number {
  const pa = PNG.sync.read(a);
  const pb = PNG.sync.read(b);
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

test.describe.configure({ mode: 'serial' });

test('automatic volumes render exactly like volumes baked explicitly', async ({ page }) => {
  const auto = await render(page, 'exits=auto');
  const baked = await render(page, 'exits=baked');
  expect(differingShare(auto, baked, 8)).toBeLessThan(0.001);
  expect(auto).toMatchSnapshot('auto-volume.png');
});

test('automatic volumes on an InstancedMesh match the same on a Mesh', async ({ page }) => {
  const meshes = await render(page, 'exits=auto');
  const instanced = await render(page, 'exits=auto&instanced=1');
  expect(differingShare(meshes, instanced, 8)).toBeLessThan(0.001);
});
