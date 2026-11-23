// Real caustics, second half, shared by the instanced renderer and GlassCaustics: given the
// photon map (where each ray of the lamp's light lands on the floor, from photonNode in
// tsl/caustics.ts), draw the light that meets no glass (`openQuad`), and splat every ray that
// went through glass over the patch of floor it lights (`landScene`, drawn with the lamp's
// camera). Also the separable Gaussian blur both use to soften them by the lamp's size.
/* eslint-disable @typescript-eslint/no-explicit-any -- TSL node types are too loose to spell out */
import * as THREE from 'three/webgpu';
import * as TSL from 'three/tsl';
import { blurNode } from './tsl/finish';

const {
  abs,
  float,
  instanceIndex,
  length,
  max,
  positionGeometry,
  select,
  texture,
  uniform,
  uv,
  varying,
  vec2,
  vec3,
  vec4,
} = TSL as any;

/**
 * The open-light quad and the splat scene for a photon map of `photonMapSize`^2 rays.
 * `uPhotonBand`: the wavelength band being gathered (yzw its share of white light).
 */
export function createCausticSplat(photonMap: THREE.RenderTarget, photonMapSize: number, uPhotonBand: any) {
  // The second half of real caustics. First the light that reaches the floor
  // without meeting any glass ...
  const photonNode = texture(photonMap.texture);
  const openMaterial = new THREE.NodeMaterial();
  openMaterial.fragmentNode = vec4(vec3(float(1).sub(photonNode.a.min(1))), 1);
  openMaterial.depthTest = openMaterial.depthWrite = false;
  const openQuad = new THREE.QuadMesh(openMaterial);
  // ... then every ray that went through glass, spread over the patch of floor it
  // lights and added on. Each ray stands for one texel of the lamp's view; the glass
  // bends that texel into a parallelogram on the floor, spanned by where the rays on
  // either side of it land. Its light is shared out over that area, so the patches of
  // neighbouring rays tile the floor without gaps or overlaps (a fixed square per ray
  // left a grid of gaps wherever the glass stretched the light one way).
  const rayIndex = float(instanceIndex);
  const rayRow = rayIndex.div(photonMapSize).floor();
  const rayTexel = vec2(rayIndex.sub(rayRow.mul(photonMapSize)), rayRow);
  const texelStep = 1 / photonMapSize;
  /** Where the ray of the texel `offset` texels away lands (lamp clip units), and whether it does. */
  const landingOf = (offset: [number, number]) => {
    const at = rayTexel.add(vec2(offset[0], offset[1])).add(0.5).mul(texelStep);
    const r = texture(photonMap.texture, at.clamp(float(0), float(1))).level(float(0));
    const inside = at.x.greaterThan(0).and(at.x.lessThan(1)).and(at.y.greaterThan(0)).and(at.y.lessThan(1));
    return {
      pos: vec2(at.x.mul(2).sub(1), float(1).sub(at.y.mul(2))).add(r.xy),
      lands: inside.and(r.z.greaterThan(1e-5)),
      light: r.z,
    };
  };
  const self = landingOf([0, 0]);
  /** Footprint edge along one axis of the lamp map: central difference, one-sided at an edge, else one texel. */
  const footprintEdge = (axis: [number, number]) => {
    const ahead = landingOf(axis);
    const behind = landingOf([-axis[0], -axis[1]]);
    // A neighbour that lands far away went round the other side of the object (or
    // through another surface): not part of the same smooth bundle of rays.
    const reach = 24 * 2 * texelStep;
    const aheadOk = ahead.lands.and(ahead.pos.sub(self.pos).length().lessThan(reach));
    const behindOk = behind.lands.and(behind.pos.sub(self.pos).length().lessThan(reach));
    const oneTexel = vec2(axis[0] * 2 * texelStep, -axis[1] * 2 * texelStep);
    return select(
      aheadOk.and(behindOk),
      ahead.pos.sub(behind.pos).mul(0.5),
      select(aheadOk, ahead.pos.sub(self.pos), select(behindOk, self.pos.sub(behind.pos), oneTexel)),
    );
  };
  const edgeX = footprintEdge([1, 0]).toVar();
  const edgeY = footprintEdge([0, 1]).toVar();
  // One clip unit of the lamp map is photonMapSize / 2 texels.
  const texelsPerUnit = photonMapSize / 2;
  // Each ray's light is spread by a bilinear "hat" over its own footprint: twice as wide
  // along each footprint edge and falling linearly to 0, so the hats of neighbouring rays
  // (one footprint apart) add up to a perfectly smooth spread, whether the footprint is
  // small, large, or long and thin. (Hard-edged shapes snapped to whole texels, leaving a
  // grid of squares; one square per ray turned long, thin footprints into blocks.) Edges
  // shorter than a texel are stretched to one, so no light falls between texel centres.
  // eslint-disable-next-line @typescript-eslint/no-explicit-any
  const atLeastOneTexel = (edge: any, fallback: any) => {
    const texels = length(edge).mul(texelsPerUnit);
    return select(texels.lessThan(1e-4), fallback, edge.mul(max(float(1).div(max(texels, 1e-4)), 1)));
  };
  /* eslint-disable @typescript-eslint/no-explicit-any -- select() loses the vector types */
  const hatX: any = atLeastOneTexel(edgeX, vec2(2 * texelStep, 0)).toVar();
  const hatY: any = atLeastOneTexel(edgeY, vec2(0, -2 * texelStep)).toVar();
  const hatArea = abs(hatX.x.mul(hatY.y).sub(hatX.y.mul(hatY.x))).mul(texelsPerUnit * texelsPerUnit);
  // Edges (nearly) along one line, where a fold of the caustic squeezes the footprint flat:
  // a one-texel hat instead.
  const flat = hatArea.lessThan(0.5);
  const corner = vec2(positionGeometry.x, positionGeometry.y);
  const spanX: any = select(flat, vec2(2 * texelStep, 0), hatX);
  const spanY: any = select(flat, vec2(0, -2 * texelStep), hatY);
  /* eslint-enable @typescript-eslint/no-explicit-any */
  const landPos = self.pos.add(spanX.mul(corner.x.mul(2)).add(spanY.mul(corner.y.mul(2))));
  // The hat over (-1..1)^2 of the footprint integrates to the footprint's area.
  const landLight = self.light.div(select(flat, float(1), hatArea));
  // Where in the hat this corner is (-1 to 1 along each edge).
  const tentAt = varying(corner.mul(2));
  const tentWeight = float(1)
    .sub(abs(tentAt.x))
    .mul(float(1).sub(abs(tentAt.y)))
    .max(0);
  const landMaterial = new THREE.NodeMaterial();
  // Cull empty photon rays (no glass hit or zero carried light) offscreen before rasterization.
  landMaterial.vertexNode = vec4(select(self.lands, landPos.x, 2), select(self.lands, landPos.y, 2), 0, 1);
  landMaterial.fragmentNode = vec4(uPhotonBand.yzw.mul(varying(landLight).mul(tentWeight)), 1);
  // A footprint's edges can wind either way (the lamp map's rows run down, clip space up,
  // and a caustic can fold the light over): draw both sides, or many would be culled.
  landMaterial.side = THREE.DoubleSide;
  landMaterial.transparent = true;
  landMaterial.blending = THREE.AdditiveBlending;
  landMaterial.depthTest = landMaterial.depthWrite = false;
  const landGeometry = new THREE.InstancedBufferGeometry();
  const square = new THREE.PlaneGeometry(1, 1);
  landGeometry.index = square.index;
  landGeometry.setAttribute('position', square.getAttribute('position'));
  landGeometry.instanceCount = photonMapSize * photonMapSize;
  const landMesh = new THREE.Mesh(landGeometry, landMaterial);
  landMesh.frustumCulled = false;
  const landScene = new THREE.Scene();
  landScene.add(landMesh);
  return { openQuad, landScene, landMaterial };
}

/**
 * A separable Gaussian blur through a scratch target: `blur(from, to, radiusX, radiusY,
 * scratch, cut, maxPasses)` blurs `from` into `to` by `radiusX` and `radiusY` (texture
 * coordinates; 2 sigma), keeping only what is brighter than `cut`.
 */
export function createBlur(renderer: THREE.WebGPURenderer, defaultScratch: THREE.RenderTarget) {
  const uBlurCut = uniform(0);
  const uBlurReach = uniform(6);
  const uBlurDir = uniform(new THREE.Vector2());
  const blurSource = texture(defaultScratch.texture);
  const blurMaterial = new THREE.NodeMaterial();
  blurMaterial.fragmentNode = blurNode(uv(), uBlurDir, uBlurCut, uBlurReach, blurSource);
  blurMaterial.depthTest = blurMaterial.depthWrite = false;
  const blurQuad = new THREE.QuadMesh(blurMaterial);
  /**
   * Blur `from` into `to` (through a scratch target) by `radiusX` and `radiusY`, in
   * texture coordinates, keeping only what is brighter than `cut`.
   */
  const blur = (
    from: THREE.Texture,
    to: THREE.RenderTarget,
    radiusX: number,
    radiusY = radiusX,
    scratch = defaultScratch,
    cut = 0,
    maxPasses = 4,
  ) => {
    // Each bilinear-paired tap covers 2 texels (up to 12 pairs = 25 fetches per pass).
    const texels = Math.max(radiusX * scratch.width, radiusY * scratch.height);
    const passes = Math.min(Math.max(Math.ceil((texels / 22) ** 2), 1), maxPasses);
    const share = 1 / Math.sqrt(passes);
    // Taps cover the kernel out to 3 sigma (1.5 times `radius`, which is 2 sigma), about a
    // texel apart: farther apart, sharp content (caustic splats) breaks up into a grid.
    uBlurReach.value = Math.min(Math.max(Math.ceil((texels * share * 1.5) / 2.2), 2), 12);
    for (let pass = 0; pass < passes; pass++) {
      blurSource.value = pass === 0 ? from : to.texture;
      uBlurCut.value = pass === 0 ? cut : 0;
      uBlurDir.value.set(radiusX * share, 0);
      renderer.setRenderTarget(scratch);
      blurQuad.render(renderer);
      blurSource.value = scratch.texture;
      uBlurCut.value = 0;
      uBlurDir.value.set(0, radiusY * share);
      renderer.setRenderTarget(to);
      blurQuad.render(renderer);
    }
  };

  return { blur, uBlurCut };
}
