import { expect, test, type Page } from '@playwright/test';

// Every shape from six camera angles in both studios (the demo's white one with its floor
// grid, and the dark one), with the shape turned so its faces lie along the world axes
// (turn 0): faces square on to the camera from the front, right, back and left, a 45-degree
// corner view from above, and straight down. Square-on views are
// where total internal reflection, face filters (the CMY cube) and bevels show most plainly,
// and back and top views catch shading that only one side of a shape exercised.
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

/** Camera directions (degrees): azimuth 0 from +z (in front), 90 from +x; elevation above the floor. */
const VIEWS = [
  { name: 'front', azimuth: 0, elevation: 10 },
  { name: 'right', azimuth: 90, elevation: 10 },
  { name: 'back', azimuth: 180, elevation: 10 },
  { name: 'left', azimuth: 270, elevation: 10 },
  { name: 'corner', azimuth: 45, elevation: 30 },
  { name: 'top', azimuth: 0, elevation: 89 },
] as const;

/** The demo's two studios: the white light tent with its floor grid, and the dark one. */
const LIGHTINGS = [
  { name: 'studio', dark: false },
  { name: 'dark', dark: true },
] as const;

interface GlassTestHarness {
  setScene(opts: {
    shape: string;
    dark?: boolean;
    hdri?: string;
    ground?: boolean;
    turn?: number;
    view?: { azimuth: number; elevation: number };
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
  test(`${name} from six angles in both studios`, async () => {
    const canvas = page.locator('#view');
    for (const lighting of LIGHTINGS) {
      for (const view of VIEWS) {
        await page.evaluate(
          ([shapeName, dark, azimuth, elevation]) =>
            (window as unknown as { __glassTest: GlassTestHarness }).__glassTest.setScene({
              shape: shapeName as string,
              dark: dark as boolean,
              hdri: 'none',
              ground: true,
              turn: 0,
              view: { azimuth: azimuth as number, elevation: elevation as number },
            }),
          [name, lighting.dark, view.azimuth, view.elevation] as const,
        );
        await expect(canvas).toHaveScreenshot(['views', `${slug}-${view.name}-${lighting.name}.png`]);
      }
    }
  });
}
