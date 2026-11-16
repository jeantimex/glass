import { defineConfig } from 'vite';

export default defineConfig(({ command }) => ({
  // GitHub Pages serves the site from https://jeantimex.github.io/glass/.
  base: command === 'build' ? '/glass/' : '/',
}));
