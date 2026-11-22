import { defineConfig } from 'vite';

export default defineConfig(({ command }) => ({
  // GitHub Pages serves the site from https://jeantimex.github.io/glass/.
  base: command === 'build' ? '/glass/' : '/',
  build: {
    rollupOptions: {
      // Two pages: the instanced studio demo, and MeshGlassMaterial on plain meshes.
      input: { main: 'index.html', mesh: 'mesh.html' },
    },
  },
  test: {
    exclude: ['e2e/**', 'node_modules/**', 'dist/**'],
  },
}));
