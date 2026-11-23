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
  { name: 'CMY cube', slug: 'cmy-cube' },
  { name: 'Prism', slug: 'prism' },
  { name: 'Hex prism', slug: 'hex-prism' },
  { name: 'Diamond', slug: 'diamond' },
  { name: 'Torus knot', slug: 'torus-knot' },
  { name: 'Suzanne', slug: 'suzanne' },
  { name: 'Dragon', slug: 'dragon' },
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
    count?: number;
    offset?: number;
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

/** Shapes traced through a baked volume (their rays take many steps: see below). */
const VOLUME_SHAPES = new Set<string>(['Torus knot', 'Suzanne', 'Dragon']);

/** One settled render of the harness's scene. */
async function shot(opts: Parameters<GlassTestHarness['setScene']>[0]): Promise<PNG> {
  await page.evaluate((o) => (window as unknown as { __glassTest: GlassTestHarness }).__glassTest.setScene(o), opts);
  return PNG.sync.read(await page.locator('#view').screenshot());
}

// Stricter than the baselines (whose tolerance suits images that differ run to run): with the
// jitter sequence restarted, the renderer repeats itself exactly, so the standard mesh is held
// to its render directly. Measured: at most 0.003% of pixels (about 14) differ by more than
// 2/255 on the shapes traced by formula or facet planes (most likely floating-point rounding:
// the material reaches the shape through matrices, the renderer through a quaternion).
// Shapes traced through a baked volume amplify the last bits of where a ray starts: moving the
// renderer's own object by a micrometre changes 0.05-1% of their pixels. Bit-exact output from
// different (equivalent) arithmetic is not possible there, so they are held to that floor,
// measured in the same run: the standard mesh may differ from the renderer no more than the
// renderer differs from itself when the object moves 1 micrometre.
for (const { name } of SHAPES) {
  test(`${name}: standard mesh and instanced renderer, pixel for pixel`, async () => {
    for (const lighting of LIGHTINGS) {
      const scene = { shape: name, dark: lighting.dark, hdri: 'none', ground: true };
      const own = await shot({ ...scene, standardGlass: false });
      const standard = await shot({ ...scene, standardGlass: true });
      const { share, max } = compare(own, standard);
      let limit = 0.0001;
      if (VOLUME_SHAPES.has(name)) {
        const moved = await shot({ ...scene, standardGlass: false, offset: 1e-6 });
        limit = Math.max(limit, compare(own, moved).share);
      }
      console.log(
        `parity ${name} ${lighting.name}: ${(share * 100).toFixed(4)}% over 2/255 ` +
          `(limit ${(limit * 100).toFixed(4)}%), largest ${max}`,
      );
      expect.soft(share, `${name} ${lighting.name}`).toBeLessThan(limit);
    }
  });
}

// Several objects of each solid (the demo's seeded layouts: their own sizes, turns, IOR and
// dispersion; 16 in random colours): instanced standard meshes, each in the half of the studio
// the renderer draws it in (`inNearHalf`), against the renderer's own render, held as strictly.
// Measured: at most 0.0023% of pixels differ by more than 2/255 (16 octahedra, dark).
for (const { name } of SHAPES) {
  test(`${name}: many, instanced standard meshes and instanced renderer, pixel for pixel`, async () => {
    for (const [count, colorize] of [
      [9, false],
      [16, true],
    ] as const) {
      for (const lighting of LIGHTINGS) {
        const shots: PNG[] = [];
        for (const standardGlass of [false, true]) {
          await page.evaluate(
            ([shapeName, dark, standard, n, colour]) =>
              (window as unknown as { __glassTest: GlassTestHarness }).__glassTest.setScene({
                shape: shapeName as string,
                dark: dark as boolean,
                hdri: 'none',
                ground: true,
                standardGlass: standard as boolean,
                count: n as number,
                settings: { colorize: colour as boolean },
              }),
            [name, lighting.dark, standardGlass, count, colorize] as const,
          );
          shots.push(PNG.sync.read(await page.locator('#view').screenshot()));
        }
        const { share, max } = compare(shots[0], shots[1]);
        let limit = 0.0001;
        if (VOLUME_SHAPES.has(name)) {
          const moved = await shot({
            shape: name,
            dark: lighting.dark,
            hdri: 'none',
            ground: true,
            count,
            settings: { colorize },
            standardGlass: false,
            offset: 1e-6,
          });
          limit = Math.max(limit, compare(shots[0], moved).share);
        }
        console.log(
          `parity ${name} x${count} ${lighting.name}: ${(share * 100).toFixed(4)}% over 2/255 ` +
            `(limit ${(limit * 100).toFixed(4)}%), largest ${max}`,
        );
        expect.soft(share, `${name} x${count} ${lighting.name}`).toBeLessThan(limit);
      }
    }
  });
}
