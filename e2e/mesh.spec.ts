import { expect, test } from '@playwright/test';

// MeshGlassMaterial on plain THREE.Mesh objects (mesh.html): one exit source per piece
// (thickness estimate, back-face map, two baked volumes). Spin-free and still, so it is
// deterministic.
test('MeshGlassMaterial renders on plain meshes (estimate, back-face map, baked volumes)', async ({ page }) => {
  await page.setViewportSize({ width: 1200, height: 700 });
  await page.goto('/mesh.html');
  await page.waitForFunction(() => document.documentElement.dataset.glassReady === 'true', null, {
    timeout: 60_000,
  });
  await page.waitForTimeout(500);
  await expect(page.locator('#view')).toHaveScreenshot('mesh-glass-material.png');
});
