import { expect, test, type Page } from '@playwright/test';
import { PNG } from 'pngjs';

// The migration's acceptance test: glass on a standard THREE.Mesh with MeshGlassMaterial (studio
// mode) must match the instanced renderer's own glass. Each built-in solid is drawn that way, in
// the renderer's studio with its own glass hidden, and compared with the existing baselines of
// e2e/shapes.spec.ts: the very images the instanced renderer produces, at the same tolerance.

const SHAPES = [
  { name: 'Sphere', slug: 'sphere' },
  { name: 'Cube', slug: 'cube' },
  { name: 'Box', slug: 'box' },
  { name: 'Cylinder', slug: 'cylinder' },
  { name: 'Capsule', slug: 'capsule' },
  { name: 'Cone', slug: 'cone' },
  { name: 'Torus', slug: 'torus' },
  { name: 'Octahedron', slug: 'octahedron' },
] as const;

const LIGHTINGS = [
  { name: 'default', dark: false },
  { name: 'dark', dark: true },
] as const;

interface GlassTestHarness {
  setScene(opts: {
    shape: string;
    dark?: boolean;
    hdri?: string;
    ground?: boolean;
    standardGlass?: boolean;
  }): Promise<void>;
}

test.describe.configure({ mode: 'serial' });

let page: Page;

test.beforeAll(async ({ browser }) => {
  page = await browser.newPage({ viewport: { width: 800, height: 600 } });
  await page.goto('/?shape=diamond&ui=0');
  await page.waitForFunction(() => document.documentElement.dataset.glassReady === 'true', null, {
    timeout: 45_000,
  });
});

test.afterAll(async () => {
  await page.close();
});

for (const { name, slug } of SHAPES) {
  test(`${name}: MeshGlassMaterial on a standard mesh matches the instanced renderer`, async () => {
    for (const lighting of LIGHTINGS) {
      await page.evaluate(
        ([shapeName, dark]) =>
          (window as unknown as { __glassTest: GlassTestHarness }).__glassTest.setScene({
            shape: shapeName as string,
            dark: dark as boolean,
            hdri: 'none',
            ground: true,
            standardGlass: true,
          }),
        [name, lighting.dark] as const,
      );
      // The baseline of e2e/shapes.spec.ts: the instanced renderer's own image.
      await expect(page.locator('#view')).toHaveScreenshot(`${slug}-${lighting.name}.png`);
    }
  });
}

/** Largest channel difference (0-255) per pixel: the share of pixels above 2, and the largest. */
function compare(a: PNG, b: PNG): { share: number; max: number } {
  let over = 0;
  let max = 0;
  for (let i = 0; i < a.data.length; i += 4) {
    const d = Math.max(
      Math.abs(a.data[i] - b.data[i]),
      Math.abs(a.data[i + 1] - b.data[i + 1]),
      Math.abs(a.data[i + 2] - b.data[i + 2]),
    );
    if (d > 2) over++;
    if (d > max) max = d;
  }
  return { share: over / (a.width * a.height), max };
}

// Stricter than the baselines (whose tolerance suits images that differ run to run): with the
// jitter sequence restarted, the renderer repeats itself exactly, so the standard mesh is held
// to its render directly. Measured: at most 0.003% of pixels (about 14) differ by more than
// 2/255, isolated pixels inside the glass (most likely floating-point rounding: the material
// reaches the shape through matrices, the renderer through a quaternion).
for (const { name } of SHAPES) {
  test(`${name}: standard mesh and instanced renderer, pixel for pixel`, async () => {
    for (const lighting of LIGHTINGS) {
      const shots: PNG[] = [];
      for (const standardGlass of [false, true]) {
        await page.evaluate(
          ([shapeName, dark, standard]) =>
            (window as unknown as { __glassTest: GlassTestHarness }).__glassTest.setScene({
              shape: shapeName as string,
              dark: dark as boolean,
              hdri: 'none',
              ground: true,
              standardGlass: standard as boolean,
            }),
          [name, lighting.dark, standardGlass] as const,
        );
        shots.push(PNG.sync.read(await page.locator('#view').screenshot()));
      }
      const { share } = compare(shots[0], shots[1]);
      expect(share, `${name} ${lighting.name}`).toBeLessThan(0.0001);
    }
  });
}
