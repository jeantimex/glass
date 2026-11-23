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

test('the same pieces as InstancedMesh objects cast the same caustics', async ({ page }) => {
  const meshes = await render(page, 'glass=0');
  const instanced = await render(page, 'glass=0&instanced=1');
  expect(differingShare(meshes, instanced, 6)).toBeLessThan(0.0005);
});

/**
 * Energy balance read back from the light map itself (light-space, 1 = the lamp's light
 * unchanged): over the whole map, clear glass can only lose light (Fresnel reflection sends
 * some back up, total internal reflection traps a little), never make it. Returns the mean of
 * (map - 1) per texel the glass covers, as seen from the lamp.
 */
async function lightBalance(page: Page, query: string): Promise<number> {
  await render(page, `${query}&debug=1`);
  return page.evaluate(async () => {
    const w = window as unknown as {
      __caustics: { target: { width: number; height: number }; photonMap: unknown };
      __renderer: { readRenderTargetPixelsAsync: (...a: unknown[]) => Promise<ArrayLike<number>> };
    };
    // (Defined here: this function runs in the page, which cannot see the test file's scope.)
    // oxlint-disable-next-line unicorn/consistent-function-scoping
    const half = (h: number) => {
      const e = (h >> 10) & 31;
      const m = h & 1023;
      const v = e === 0 ? 2 ** -14 * (m / 1024) : 2 ** (e - 15) * (1 + m / 1024);
      return h & 0x8000 ? -v : v;
    };
    const read = async (target: unknown) => {
      const t = target as { width: number; height: number };
      const raw = await w.__renderer.readRenderTargetPixelsAsync(target, 0, 0, t.width, t.height);
      return raw instanceof Uint16Array ? Array.from(raw, half) : Array.from(raw);
    };
    const light = await read(w.__caustics.target);
    const photons = await read(w.__caustics.photonMap);
    let excess = 0;
    let covered = 0;
    for (let i = 0; i < light.length; i += 4) {
      excess += (light[i] + light[i + 1] + light[i + 2]) / 3 - 1;
      if (photons[i + 3] > 0.5) covered++;
    }
    return excess / Math.max(covered, 1);
  });
}

test('light is conserved: none made, and a glass ball passes what Fresnel predicts', async ({ page }) => {
  // IOR 1: nothing reflected or bent, so the light over the map is unchanged.
  expect(Math.abs(await lightBalance(page, 'ior=1.0001&dispersion=0&glass=0'))).toBeLessThan(0.01);
  // A crown-glass ball (n = 1.52) under parallel light: averaged over its disc, light refracted
  // in and straight out keeps 83.0% (exact Fresnel at both surfaces; the photon pass does not
  // follow light reflected inside), so the balance is -0.170. Measured -0.196 (texel rounding
  // at the disc's edge).
  const ball = await lightBalance(page, 'dispersion=0&glass=0&only=0');
  expect(Math.abs(ball - -0.17)).toBeLessThan(0.05);
  // All three pieces together: light lost (more through the box and teapot, whose sides send
  // light up by total internal reflection off their bases), never made.
  expect(await lightBalance(page, 'dispersion=0&glass=0')).toBeLessThan(-0.02);
});
