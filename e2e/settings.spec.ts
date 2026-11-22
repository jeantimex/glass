import { expect, test, type Page } from '@playwright/test';

// Glass settings beyond the defaults, on a representative set of shapes (a lens, flat
// parallel faces, a prism, a curved tube and a baked mesh): no and strong dispersion, low and
// high IOR, light and heavy frost, tinted glass, real caustics off, and surface imperfections.
// Each is drawn where it shows best: dispersion and caustics in the dark studio, frost, tint
// and refraction strength in the white one.
const SHAPES = [
  { name: 'Sphere', slug: 'sphere' },
  { name: 'Cube', slug: 'cube' },
  { name: 'Prism', slug: 'prism' },
  { name: 'Torus knot', slug: 'torus-knot' },
  { name: 'Suzanne', slug: 'suzanne' },
] as const;

interface Variant {
  name: string;
  dark: boolean;
  glass?: Record<string, unknown>;
  settings?: Record<string, unknown>;
}

const VARIANTS: Variant[] = [
  { name: 'dispersion-0', dark: true, glass: { dispersion: 0 } },
  { name: 'dispersion-3', dark: true, glass: { dispersion: 3 } },
  { name: 'ior-1.33', dark: false, glass: { ior: 1.33 } },
  { name: 'ior-2.0', dark: false, glass: { ior: 2.0 } },
  { name: 'frosted-light', dark: false, glass: { roughness: 0.2 } },
  { name: 'frosted-heavy', dark: false, glass: { roughness: 0.6, frostScale: 8 } },
  { name: 'tinted-amber', dark: false, glass: { glassColor: { r: 1, g: 0.55, b: 0.15 }, glassAlpha: 1 } },
  { name: 'caustics-off', dark: true, settings: { realCaustics: false } },
  { name: 'imperfections', dark: false, settings: { imperfections: 0.6 } },
];

interface GlassTestHarness {
  setScene(opts: {
    shape: string;
    dark?: boolean;
    hdri?: string;
    ground?: boolean;
    glass?: Record<string, unknown>;
    settings?: Record<string, unknown>;
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
  test(`${name} with different glass settings`, async () => {
    const canvas = page.locator('#view');
    for (const variant of VARIANTS) {
      await page.evaluate(
        ([shapeName, v]) =>
          (window as unknown as { __glassTest: GlassTestHarness }).__glassTest.setScene({
            shape: shapeName as string,
            dark: (v as Variant).dark,
            hdri: 'none',
            ground: true,
            glass: (v as Variant).glass,
            settings: (v as Variant).settings,
          }),
        [name, variant] as const,
      );
      await expect(canvas).toHaveScreenshot(['settings', `${slug}-${variant.name}.png`]);
    }
  });
}
