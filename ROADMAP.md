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

| Feature                     | `createInstancedGlass()`                  | `MeshGlassMaterial` today                              |
| :-------------------------- | :---------------------------------------- | :----------------------------------------------------- |
| Exact internal light paths  | Analytic exits for its 13 built-in shapes | Only with a baked `volume`, and not on `InstancedMesh` |
| Glass seen through glass    | Yes (screen space)                        | No: transmission sees opaque objects only              |
| Caustics on the floor       | Yes (photon traced, per wavelength band)  | No (a scene-level effect, like shadows)                |
| Anti-aliasing               | Built in (TAA)                            | None: up to the app's render pipeline                  |
| Contact shadows, bloom, DoF | Built in                                  | Up to the app's post-processing                        |

### Plan

1. **Exact exits on any mesh, with no setup.**
   - Instancing for `volume` exits: map rays through each instance's matrix as well as the
     object's world matrix (what blocks `InstancedMesh` today).
   - `volume: 'auto'` (the default): bake each geometry once in a worker
     (`bakeGlassVolumeAsync`) and cache the result per geometry. Until it is ready the mesh
     renders with the thickness estimate, then switches to exact exits. This geometry→volume
     cache is the "shape registry" this goal needs.
   - Analytic exits for three.js built-in geometries (`SphereGeometry`, `BoxGeometry`,
     `CylinderGeometry`, `CapsuleGeometry`, `ConeGeometry`, `TorusGeometry`, …): read their
     parameters and use the exact formulas from `src/glass/tsl/shapes.ts` (no bake, sharper
     than a voxel volume).
2. **Temporal-friendly option.** `temporal: true` jitters the wavelengths per frame, so the
   material pairs with three.js's own TRAA post-processing (smooth dispersion, no speckle).
3. **Caustics as an add-on.** A `GlassCaustics` helper usable with plain `renderer.render()`:
   given the lamp, it traces caustics from any `MeshGlassMaterial` meshes into a texture that a
   floor material uses (reusing the photon pass from the instanced renderer).
4. **Glass seen through glass as an add-on.** A pass, like `GlassBackfacePass`, that captures
   the glass layers so glass refracts the glass behind it.

After that, `createInstancedGlass()` becomes an optional "studio in a box", which could later be
rebuilt from the same pieces. Extending it with a registry of new shape kinds is no longer
needed for this goal.

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

Run `npm run test:screenshots`; after an intended visual change, regenerate with
`npm run test:screenshots:update` and look through the new images before committing.
