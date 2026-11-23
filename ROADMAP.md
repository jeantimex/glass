# Roadmap

## Goal: `MeshGlassMaterial` as a drop-in for `MeshPhysicalMaterial`

Developers should get studio-quality glass by putting `MeshGlassMaterial` on their own meshes,
including `InstancedMesh` and `BatchedMesh`, rendered with a plain `renderer.render(scene, camera)`,
without `createInstancedGlass()`:

```ts
const glass = new MeshGlassMaterial({ ior: 1.52, dispersion: 1, transmission: 1 });
scene.add(new THREE.InstancedMesh(anyGeometry, glass, 100));
```

### Where things stand

`MeshGlassMaterial` already works on `Mesh`, `InstancedMesh` and `BatchedMesh` with
`MeshPhysicalMaterial`-style parameters, per-instance colours feed its absorption, and it
shades with the same tracer as the instanced renderer (`src/glass/tsl/trace.ts`). Compared with
`createInstancedGlass()`, it still lacks:

| Feature                     | `createInstancedGlass()`                  | `MeshGlassMaterial` today                                                    |
| :-------------------------- | :---------------------------------------- | :--------------------------------------------------------------------------- |
| Exact internal light paths  | Analytic exits for its 13 built-in shapes | Exact on any mesh: built-in solids by formula, others by an automatic volume |
| Glass seen through glass    | Yes (screen space)                        | No: transmission sees opaque objects only                                    |
| Caustics on the floor       | Yes (photon traced, per wavelength band)  | No (a scene-level effect, like shadows)                                      |
| Anti-aliasing               | Built in (TAA)                            | None: up to the app's render pipeline                                        |
| Contact shadows, bloom, DoF | Built in                                  | Up to the app's post-processing                                              |

### Plan

1. **Exact exits on any mesh, with no setup.**
   - ✅ Instancing for `volume` exits: rays follow each instance's matrix as well as the
     object's world matrix (`e2e/instancing.spec.ts` checks an `InstancedMesh` renders exactly
     like the same objects as separate meshes).
   - ✅ Analytic exits for three.js built-in geometries (`SphereGeometry`, `BoxGeometry`,
     `CylinderGeometry`, `ConeGeometry`, `CapsuleGeometry`, `TorusGeometry`,
     `OctahedronGeometry`, `IcosahedronGeometry` of detail 3 and up as a sphere): their parameters
     are read into the exact formulas of `src/glass/tsl/shapes.ts` (no bake, sharper than a voxel
     volume), on `Mesh` and `InstancedMesh`, under any scale. `e2e/builtin-shapes.spec.ts`
     checks each against a volume baked from the same geometry.
   - ✅ `volume: 'auto'` (the default): any other geometry (with no `backfaceMap`) is baked once
     in a worker and cached per geometry; it renders with the thickness estimate until the volume
     is ready, then with exact exits. `autoGlassVolumesSettled()` waits for pending bakes.
     `e2e/auto-volume.spec.ts` checks automatic volumes render exactly like explicit ones.
2. ✅ **Temporal-friendly option.** `temporal: true` moves the wavelengths on every frame (the
   same step for every pixel), so three.js's TRAA averages them into a continuous spectrum;
   `spectralSamples` sets how many per frame. `e2e/temporal.spec.ts` checks TRAA's average gets
   closer to a 64-wavelength reference than the same number of fixed wavelengths. `mesh.html?taa=1`
   shows it.
3. ✅ **Caustics as an add-on.** `GlassCaustics` (`src/glass/glassCaustics.ts`) traces a
   `DirectionalLight` through every `MeshGlassMaterial` mesh, on `Mesh` and `InstancedMesh` (same
   exits as the material, per wavelength band), onto a floor plane, and folds the result into the
   light (`attach()`), so any lit floor material gets caustics with plain `renderer.render()`.
   `e2e/caustics.spec.ts` checks energy: glass at IOR 1 leaves the floor unchanged, a glass ball
   passes what Fresnel predicts (balance -0.196 against -0.170 exact), and no light is made.
4. ✅ **Glass seen through glass as an add-on.** `GlassLayers` (`src/glass/glassLayers.ts`)
   renders the farther half of the glass (with the opaque scene) into a layer, with each pixel's
   distance in alpha, and the nearer glass looks into it (`behindMap`) with the renderer's own
   march (`tsl/lookThrough.ts`). `e2e/layers.spec.ts` checks glass behind glass shows, and little
   changes with none behind. Known: beyond the march's reach the material samples the layer where
   transmission sampled three.js's opaque viewport texture, and the two differ along some edges
   (0.6% of pixels in the test; to be resolved by the studio environment hook, which replaces it).
5. **Material parity with the instanced renderer.** Features `createInstancedGlass()` has that
   `MeshGlassMaterial` does not yet:
   - CMY dichroic face filters (the CMY cube) and edge bevels on faceted shapes.
   - The diamond's exact facet tracing (`DiamondMesh` has it; the material on
     `diamondGeometry()` falls back to an automatic volume).
   - Exact exits on `BatchedMesh` (per-instance matrices from its matrices texture).
   - Non-uniform scale with a `volume` (exact shapes already handle it).
6. **The studio on standard meshes.** Rebuild the demo with plain three.js objects: the floor
   with its grid, reflections, contact shadows, spotlight and glow, using the add-ons above and
   three.js post-processing.
7. **Parity, then retirement.** Run the existing screenshot suites against the rebuilt demo and
   compare with the current baselines; only then retire `createInstancedGlass()`. Until then it
   stays, unchanged, as the reference.

After that, `createInstancedGlass()` becomes an optional "studio in a box", which could later be
rebuilt from the same pieces. Extending it with a registry of new shape kinds is no longer
needed for this goal.

## Acceptance: a 100% visual match

The migration is done when the demo, rebuilt on standard three.js meshes with
`MeshGlassMaterial`, passes the existing screenshot suites against today's baselines (the images
`createInstancedGlass()` renders now). That needs the same shading code, not similar code:

- the glass seen through glass by the renderer's own screen-space march (`tsl/lookThrough.ts`,
  shared since step 4);
- the same glass shader: in the rebuilt studio, `MeshGlassMaterial` should shade with the
  renderer's own `glassNode` (`tsl/glass.ts`), fed the studio's floor and room (`tsl/scene.ts`),
  rather than a port of it. Every per-pixel detail (outer reflections routed like refraction,
  ray-cone filtering, shadows between objects, frost blur, CMY faces, bevels, the diamond's facet
  tracing) then matches by construction. `glassNode` takes each object's frame as a pluggable
  `GlassFrame` (done), so object and instance matrices can drive it instead of the renderer's
  per-instance attributes. Done for the built-in solids: `createInstancedGlass(...).studio`
  exposes the renderer's inputs, `MeshGlassMaterial({ studio })` shades with `glassNode` through
  the mesh's matrices (`studioGlass.ts`), and `e2e/parity.spec.ts` draws each solid as a
  standard mesh in the renderer's studio: it passes the shape suite's baselines, and against
  the renderer's own render (jitter sequence restarted, so both repeat exactly) at most 0.003%
  of pixels differ by more than 2/255. Next: the custom meshes (volumes), the diamond, CMY and
  bevels, instancing in studio mode, shadows and caustics from standard meshes, then the
  studio drawn entirely from standard meshes.

## Known physical limits

- **Caustics only follow the topmost object.** In piles and stacks, lower objects cast no
  shadow or caustic of their own. Fix: trace light through several layers of objects.
- **Glass seen through glass is screen space.** Objects hidden behind others or off screen
  cannot be seen through glass.
- **Baked volumes are 128³.** The finest creases of detailed meshes (the dragon) are rounded at
  voxel scale. Fix: higher resolution, or exact triangle tracing (a BVH on WebGPU).
- **TAA smears glass a little while moving fast.** Refracted content has no valid motion
  vector; inherent to reprojection.

## Testing

- `e2e/shapes.spec.ts`: every shape in three lighting setups.
- `e2e/views.spec.ts`: every shape from six camera angles (faces square on, at 45°, from
  behind, from above) in both studios: the white one with its floor grid, and the dark one.
- `e2e/settings.spec.ts`: five representative shapes with no and strong dispersion, low and
  high IOR, light and heavy frost, tinted glass, real caustics off, and imperfections.
- `e2e/mesh.spec.ts`: `MeshGlassMaterial` on plain meshes (`mesh.html`).
- `e2e/instancing.spec.ts`: `MeshGlassMaterial` on an `InstancedMesh` matches separate meshes (`instancing.html`).
- `e2e/temporal.spec.ts`: `temporal` wavelengths under three.js TRAA converge closer to a continuous spectrum than fixed ones (`temporal-test.html`).
- `e2e/layers.spec.ts`: `GlassLayers` shows glass behind glass (`layers-test.html`).
- `e2e/caustics.spec.ts`: `GlassCaustics` conserves light (IOR 1 unchanged, a ball matches Fresnel), casts caustics, and matches on `InstancedMesh` (`caustics-test.html`).
- `e2e/auto-volume.spec.ts`: automatic volumes render exactly like explicit ones, on `Mesh` and `InstancedMesh` (`auto-test.html`).
- `e2e/builtin-shapes.spec.ts`: built-in solids traced exactly agree with volumes baked from the same geometries, and on an `InstancedMesh` match a `Mesh` (`shapes-test.html`).

Run `npm run test:screenshots`; after an intended visual change, regenerate with
`npm run test:screenshots:update` and look through the new images before committing.
