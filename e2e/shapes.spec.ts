import { expect, test, type Page } from '@playwright/test';

const SHAPES = [
  { name: 'Diamond', slug: 'diamond' },
  { name: 'Sphere', slug: 'sphere' },
  { name: 'Cube', slug: 'cube' },
  { name: 'CMY cube', slug: 'cmy-cube' },
  { name: 'Box', slug: 'box' },
  { name: 'Cylinder', slug: 'cylinder' },
  { name: 'Capsule', slug: 'capsule' },
  { name: 'Cone', slug: 'cone' },
  { name: 'Torus', slug: 'torus' },
  { name: 'Torus knot', slug: 'torus-knot' },
  { name: 'Octahedron', slug: 'octahedron' },
  { name: 'Prism', slug: 'prism' },
  { name: 'Hex prism', slug: 'hex-prism' },
  { name: 'Suzanne', slug: 'suzanne' },
  { name: 'Dragon', slug: 'dragon' },
] as const;

interface GlassTestHarness {
  setScene(opts: { shape: string; dark?: boolean; hdri?: string; ground?: boolean }): Promise<void>;
}

test.describe.configure({ mode: 'serial' });

let page: Page;

test.beforeAll(async ({ browser }) => {
  page = await browser.newPage({ viewport: { width: 800, height: 600 } });
  await page.goto('/?shape=diamond&ui=0');
  await page.waitForFunction(() => document.documentElement.dataset.glassReady === 'true', {
    timeout: 45_000,
  });
});

test.afterAll(async () => {
  await page.close();
});

for (const { name, slug } of SHAPES) {
  test(`${name} renders default, dark studio, and ocean HDRI (no ground)`, async () => {
    const canvas = page.locator('#view');

    // 1. Default studio lighting with ground
    await page.evaluate(
      (shapeName) =>
        (window as unknown as { __glassTest: GlassTestHarness }).__glassTest.setScene({
          shape: shapeName,
          dark: false,
          hdri: 'none',
          ground: true,
        }),
      name,
    );
    await expect(canvas).toHaveScreenshot(`${slug}-default.png`);

    // 2. Dark studio lighting with ground
    await page.evaluate(
      (shapeName) =>
        (window as unknown as { __glassTest: GlassTestHarness }).__glassTest.setScene({
          shape: shapeName,
          dark: true,
          hdri: 'none',
          ground: true,
        }),
      name,
    );
    await expect(canvas).toHaveScreenshot(`${slug}-dark.png`);

    // 3. Ocean HDRI (Venice Sunset) without ground rendering
    await page.evaluate(
      (shapeName) =>
        (window as unknown as { __glassTest: GlassTestHarness }).__glassTest.setScene({
          shape: shapeName,
          dark: false,
          hdri: 'venice_sunset',
          ground: false,
        }),
      name,
    );
    await expect(canvas).toHaveScreenshot(`${slug}-ocean-hdri.png`);
  });
}
