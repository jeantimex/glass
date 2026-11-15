import { defineConfig } from 'vite';

// Cross-origin isolation lets the physics engine use threads (SharedArrayBuffer).
// A production host needs to send the same two headers to get that speed-up.
const isolation = {
  'Cross-Origin-Opener-Policy': 'same-origin',
  'Cross-Origin-Embedder-Policy': 'require-corp',
};

export default defineConfig({
  // The physics engine's worker threads are ES modules (they use top-level await).
  worker: { format: 'es' },
  server: { headers: isolation },
  preview: { headers: isolation },
});
