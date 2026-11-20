# Glass Material (WIP)

Real-time physical glass rendering in the browser with **Three.js** and **TSL** (compiles to **WebGPU / WGSL**, with automatic fallback to **WebGL 2 / GLSL**): exact dielectric Fresnel, Cauchy chromatic dispersion, Beer–Lambert volumetric absorption, frosted glass (毛玻璃), 6 diamond cuts, caustics, and instanced rendering of thousands of glass objects.

---

## Features

* **Standalone [`GlassMaterial`](src/glass/material.ts)** (`extends THREE.MeshBasicNodeMaterial`) — drop directly onto any `THREE.Mesh` / `THREE.BufferGeometry`.
* **Exact Optical Math** — unpolarized Cook–Torrance dielectric Fresnel, conserved-wavelength Cauchy dispersion (Fraunhofer C/d/F lines), Beer–Lambert volumetric absorption, and Total Internal Reflection (TIR).
* **Frosted Glass (毛玻璃)** — noise-free analytic environment softening, 3-octave organic micro-particle grain (`frostScale`), creamy translucent frost veil, and crisp Fresnel rim.
* **6 Diamond Cuts** — `round` (Round Brilliant), `princess`, `emerald`, `asscher`, `cushion`, and `radiant`, built as exact watertight convex polyhedra matching the ray-traced facet planes.
* **Instanced Studio Renderer** — [`createInstancedGlass`](src/glass/renderer.ts) for rendering up to 20,000 glass objects with real photon-traced caustics, floor reflections, contact shadows, depth of field, and bloom.

---

## Quick Start

### 1. Frosted / Colored Glass on Any Three.js Mesh

```ts
import * as THREE from 'three/webgpu';
import { GlassMaterial } from './src/glass';

const scene = new THREE.Scene();
const camera = new THREE.PerspectiveCamera(45, window.innerWidth / window.innerHeight, 0.1, 100);
camera.position.set(0, 1.5, 4);

const renderer = new THREE.WebGPURenderer({ antialias: true });
renderer.setSize(window.innerWidth, window.innerHeight);
document.body.appendChild(renderer.domElement);

const glass = new GlassMaterial({
  ior: 1.52,          // Optical crown glass
  dispersion: 0.6,    // Chromatic dispersion
  roughness: 0.25,    // Frosted glass (0 = clear, 1 = fully frosted)
  frostScale: 5,      // Higher = finer sandblasted micro-particles
  color: '#b8e0ff',   // Beer-Lambert volumetric tint
  thickness: 1.2,     // Approximate thickness in world units
});

const knot = new THREE.Mesh(new THREE.TorusKnotGeometry(0.8, 0.28, 180, 36), glass);
scene.add(knot);

await renderer.init();
renderer.setAnimationLoop(() => {
  knot.rotation.y += 0.008;
  renderer.render(scene, camera);
});
```

### 2. Exact Multi-Bounce Diamond with Cut Switching

```ts
import * as THREE from 'three/webgpu';
import { GlassMaterial, diamondGeometry, DIAMOND_CUTS, type DiamondCut } from './src/glass';

const cut: DiamondCut = 'princess';

const diamondMat = new GlassMaterial({
  shape: 'diamond',
  diamondCut: cut,
  ior: 2.42,          // Real diamond IOR
  dispersion: 0.4,    // Spectral fire
  bounces: 6,         // Internal reflections
  dark: true,         // Dark studio with overhead softboxes & spotlight
});

const diamond = new THREE.Mesh(diamondGeometry(cut), diamondMat);
// Rest the stone flush on the floor (y = 0)
diamond.rotation.z = -DIAMOND_CUTS[cut].lean;
diamond.position.y = DIAMOND_CUTS[cut].rest;
scene.add(diamond);

// Switch cuts dynamically at runtime:
function setCut(nextCut: DiamondCut) {
  diamond.geometry.dispose();
  diamond.geometry = diamondGeometry(nextCut);
  diamondMat.diamondCut = nextCut;
  diamond.rotation.z = -DIAMOND_CUTS[nextCut].lean;
  diamond.position.y = DIAMOND_CUTS[nextCut].rest;
}
```

### 3. Live Reactive Updates (Zero Shader Recompilation)

Every optical property is backed by a GPU uniform—mutating properties updates the shader immediately:

```ts
glass.ior = 1.66;              // Lead crystal
glass.dispersion = 1.2;        // Stronger prismatic fire
glass.roughness = 0.4;         // Frosted glass
glass.frostScale = 8;          // Ultra-fine micro-particles
glass.color.set('#ff88aa');    // Rose quartz tint
glass.colorAlpha = 0.8;        // Blend toward clear
glass.dark = true;             // Switch to dark studio lighting
```

### 4. Refracting a Live Scene Background

Pass a rendered background texture (and optional blurred copy for frosted glass) with `setBackground`:

```ts
// Render your opaque background scene to a RenderTarget, then bind it:
glass.setBackground(backgroundTarget.texture, blurredBackgroundTarget.texture);

// Revert to the built-in procedural studio environment:
glass.setBackground(null);
```

### 5. High-Scale Instanced Renderer (`createInstancedGlass`)

For scenes with hundreds or thousands of glass objects, real caustics, floor reflections, and contact shadows, pass a `GlassMaterial` directly to `instanced.render()`:

```ts
import { GlassMaterial, KIND, createGlassObjects, createInstancedGlass } from './src/glass';

const objects = createGlassObjects();
objects.count = 1;
objects.kind[0] = KIND.sphere;
objects.center.set([0, 0.6, 0], 0);
objects.quat.set([0, 0, 0, 1], 0);
objects.size.set([0.6, 0.6, 0.6], 0);

const glass = new GlassMaterial({ ior: 1.52, dispersion: 1, roughness: 0.15 });
const instanced = createInstancedGlass(renderer, camera, objects, meshes);

instanced.render(glass, lighting, false);
```

---

## API Reference

### `new GlassMaterial(params?: GlassMaterialParameters)`

Extends `THREE.MeshBasicNodeMaterial`.

#### Constructor Parameters & Live Properties

| Property | Type | Default | Description |
| :--- | :--- | :--- | :--- |
| `ior` | `number` | `1.52` | Index of refraction (`1.0` – `2.5`). Examples: `1.47` borosilicate, `1.49` acrylic, `1.52` crown glass, `1.62` flint glass, `2.42` diamond. |
| `dispersion` | `number` | `0.5` | Chromatic dispersion strength (`0` disables RGB splitting; `0.4` realistic diamond fire; `1.0+` vivid prism fire). |
| `roughness` | `number` | `0` | Frosted glass amount (`0` = crystal clear, `1` = fully frosted 毛玻璃). |
| `frostScale` | `number` | `4` | Frosted grain density (`0.5` = coarse etched frost, `4`–`12` = fine sandblasted micro-particles). |
| `color` | `THREE.ColorRepresentation` | `0xffffff` | Volumetric glass tint color (Beer–Lambert transmittance across one characteristic diameter). |
| `colorAlpha` | `number` | `1` | Blend strength of `color` (`0` = clear regardless of `color`, `1` = full tint). |
| `thickness` | `number` | `1` | Characteristic thickness in world units used for `'mesh'` mode refraction & absorption. |
| `bounces` | `number` | `6` | Maximum internal reflection/refraction events for analytic shapes. |
| `shape` | [`GlassShape`](#glassshape) | `'mesh'` | `'mesh'` for any `THREE.BufferGeometry`, or an analytic shape for exact multi-bounce internal raytracing. *(Read-only after construction.)* |
| `shapeSize` | `[number, number, number] \| THREE.Vector3` | `[1, 1, 1]` | Local-space size parameters for analytic shapes (radius, half-extents, etc.). |
| `diamondCut` | [`DiamondCut`](#diamondcut) | `'round'` | Facet cut when `shape === 'diamond'`. Updating this property on a diamond material rebuilds the facet shader automatically. |
| `bevel` | `number` | `0` | Edge bevel width in local units when `shape === 'box'`. |
| `cmy` | `number` | `0` | CMY dichroic face-filter strength (`0`–`1`) when `shape === 'box'`. |
| `dark` | `boolean` | `false` | Switches the procedural studio lighting from bright light-tent (`false`) to dark jeweller’s studio (`true`). |
| `ground` | `boolean` | `true` | Includes a procedural ground plane in reflections and refractions. |
| `groundColor` | `THREE.ColorRepresentation` | `0xeeeeee` | Albedo color of the procedural ground plane. |
| `lampDirection` | `THREE.Vector3 \| [number, number, number]` | `[-0.6, 0.7, 0.4]` | Direction vector toward the main studio lamp (automatically normalized). |
| `exposure` | `number` | `1` | Linear exposure multiplier before tone mapping. |
| `filmic` | `boolean` | `true` | Applies the Khronos Neutral filmic tone-mapping curve. |
| `backgroundTexture` | `THREE.Texture \| null` | `null` | Optional screen-space background texture to refract through the glass. |
| `backgroundBlurTexture` | `THREE.Texture \| null` | `null` | Optional blurred copy of `backgroundTexture` used when `roughness > 0`. |

#### Methods

* **`setBackground(sharp: THREE.Texture | null, blurred?: THREE.Texture | null): void`**
  Binds or clears the screen-space refraction textures at runtime.

---

### Types & Helpers

#### `GlassShape`
```ts
type GlassShape =
  | 'mesh' | 'box' | 'sphere' | 'cylinder' | 'capsule'
  | 'cone' | 'torus' | 'octahedron' | 'prism' | 'hexprism' | 'diamond';
```

#### `DiamondCut`
```ts
type DiamondCut = 'round' | 'princess' | 'emerald' | 'asscher' | 'cushion' | 'radiant';
```

#### `diamondGeometry(cut?: DiamondCut): THREE.BufferGeometry`
Builds a watertight, flat-shaded `BufferGeometry` of size 1 for the given diamond cut directly from its convex facet planes.

#### `DIAMOND_CUTS: Record<DiamondCut, DiamondCutSpec>`
Metadata for each cut: `{ label, planes, planeCount, rest, lean }`, where `rest` is the resting center height above `y = 0` and `lean` is the tilt angle (radians around Z) to sit flat on a pavilion facet.

---

## Development & Testing

```bash
npm install          # Install dependencies
npm run dev          # Start Vite dev server (open ?shape=diamond or ?shape=torus-knot&count=27&layout=grid)
npm test             # Run Vitest unit test suite
npm run typecheck    # TypeScript check (tsc --noEmit)
npm run build        # Production build
npm run deploy       # Build and deploy demo to GitHub Pages
```

---

## Credits

* **Dragon model:** the Stanford dragon, © 1996 Stanford University Computer Graphics Laboratory ([Stanford 3D Scanning Repository](https://graphics.stanford.edu/data/3Dscanrep/)), converted and cleaned up by Morgan McGuire (© 2017), from the [Khronos glTF sample models](https://github.com/KhronosGroup/glTF-Sample-Assets/tree/main/Models/DragonAttenuation). Free for non-commercial use with credit to the Stanford Computer Graphics Laboratory; not to be used in a product for sale without their permission.
* **Suzanne:** Blender’s test mesh.
