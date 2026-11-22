// Demo presets, default studio/glass settings, and URL query-string helpers.
import { KIND, type CmyFaceOrder, type DiamondCut } from '../glass';
import type { GridLayout } from './grid';
import type { HdriPreset } from './hdri';

export type RGB = { r: number; g: number; b: number };

/** Return a fresh copy of the default lighting, floor, camera, and glass demo settings. */
export function createDefaultSettings() {
  return {
    // Lighting: bright light-tent, dark jeweller's studio, or equirectangular HDRI.
    hdri: 'none' as HdriPreset,
    hdriIntensity: 1,
    dark: false,
    spotlight: true,
    lamp: 1,
    lampHeight: 45,
    lampTurn: 150,
    ground: true,
    groundColor: { r: 0xee / 255, g: 0xee / 255, b: 0xee / 255 } as RGB,
    exposure: 1,

    // Floor surface, reflections, shadows, grid, and fog.
    floorColor: { r: 0xee / 255, g: 0xee / 255, b: 0xee / 255 } as RGB,
    floorAlpha: 1,
    floorReflection: 0.4,
    shadowSoftness: 0.15,
    shadowDarkness: 0.85,
    objectShadows: true,
    floorGrid: true,
    floorGridColor: { r: 0x33 / 255, g: 0x33 / 255, b: 0x33 / 255 } as RGB,
    floorGridAlpha: 1,
    fog: 0.15,

    // Caustics & glass-through-glass transmission.
    realCaustics: true,
    seeThrough: true,
    causticSoftness: 0.3,
    causticStrength: 0.3,

    // Camera finishing.
    contactShadows: 0.7,
    depthOfField: 0,
    glow: 0.35,
    filmic: true,

    // Glass material properties (for grids; single showcase shapes keep their own copies).
    ior: 1.52,
    dispersion: 1,
    roughness: 0,
    frostScale: 4,
    // Faint waviness of the glass surfaces, as in cast glass (0: perfect, optical glass).
    imperfections: 0,
    // Edge bevel of faceted shapes in grids (m): their objects are about a third the size.
    bevel: 0.003,
    diamondCut: 'round' as DiamondCut,
    cmyOrder: 'cym' as CmyFaceOrder,
    glassColor: { r: 1, g: 1, b: 1 } as RGB,
    glassAlpha: 1,
    colorize: false,
    wireframe: false,
    wireColor: { r: 0xff / 255, g: 0xaa / 255, b: 0x40 / 255 } as RGB,
    wireAlpha: 1,
  };
}

export type DemoSettings = ReturnType<typeof createDefaultSettings>;

export interface ShapePreset {
  /** Shape KIND ids to include (null = random mix of all shapes). */
  kinds: number[] | null;
  cubic?: boolean;
  cmy?: boolean;
  ior?: number;
  dispersion?: number;
  /** Optional default lighting/camera overrides when viewed as a single object. */
  look?: Partial<DemoSettings>;
  /** Initial camera [posX, posY, posZ, targetX, targetY, targetZ] for single showcase mode. */
  camera?: [number, number, number, number, number, number];
}

// Every glass shape is the same glass (crown glass, IOR 1.52) seen in the same studio,
// so they can be compared side by side; only the diamond is a different material.
export const SHAPES: Record<string, ShapePreset> = {
  'Mixed shapes': { kinds: null, ior: 1.52 },
  Diamond: { kinds: [KIND.diamond], ior: 2.42, dispersion: 0.4, camera: [3.0, 1.2, 4.1, 0.35, 0.4, 0] },
  Sphere: { kinds: [KIND.sphere], ior: 1.52 },
  Cube: { kinds: [KIND.box], cubic: true, ior: 1.52 },
  'CMY cube': { kinds: [KIND.box], cubic: true, cmy: true, ior: 1.52 },
  Box: { kinds: [KIND.box], ior: 1.52 },
  Cylinder: { kinds: [KIND.cylinder], ior: 1.52 },
  Capsule: { kinds: [KIND.capsule], ior: 1.52 },
  Cone: { kinds: [KIND.cone], ior: 1.52 },
  Torus: { kinds: [KIND.torus], ior: 1.52 },
  'Torus knot': { kinds: [KIND.torusknot], ior: 1.52 },
  Octahedron: { kinds: [KIND.octahedron], ior: 1.52 },
  Prism: { kinds: [KIND.prism], ior: 1.52 },
  'Hex prism': { kinds: [KIND.hexprism], ior: 1.52 },
  Suzanne: { kinds: [KIND.suzanne], ior: 1.52 },
  Dragon: { kinds: [KIND.dragon], ior: 1.52 },
};

export const SINGLE_SHAPE_SPECS: Record<number, { labels: string[]; measures: number[]; turn: number }> = {
  [KIND.box]: { labels: ['Half width', 'Half height', 'Half depth'], measures: [0.6, 0.6, 0.6], turn: -15 },
  [KIND.sphere]: { labels: ['Radius'], measures: [0.6], turn: 0 },
  [KIND.cylinder]: { labels: ['Radius', 'Half height'], measures: [0.5, 0.6], turn: 0 },
  [KIND.capsule]: { labels: ['Radius', 'Half length'], measures: [0.4, 0.4], turn: 0 },
  [KIND.cone]: { labels: ['Radius', 'Half height'], measures: [0.55, 0.6], turn: 0 },
  [KIND.torus]: { labels: ['Ring radius', 'Tube radius'], measures: [0.62, 0.22], turn: 0 },
  [KIND.octahedron]: { labels: ['Size'], measures: [0.85], turn: 20 },
  [KIND.prism]: { labels: ['Size', 'Half length'], measures: [0.65, 0.6], turn: 55 },
  [KIND.hexprism]: { labels: ['Radius', 'Half height'], measures: [0.5, 0.5], turn: -15 },
  [KIND.suzanne]: { labels: ['Size'], measures: [0.8], turn: -20 },
  [KIND.diamond]: { labels: ['Size'], measures: [0.6], turn: 240 },
  [KIND.dragon]: { labels: ['Size'], measures: [0.9], turn: -30 },
  [KIND.torusknot]: { labels: ['Size'], measures: [0.75], turn: -20 },
};

export const MAX_COUNT = 15000;

/** Convert sRGB [0..1] component to linear light. */
export const srgbToLinear = (c: number) => (c <= 0.04045 ? c / 12.92 : Math.pow((c + 0.055) / 1.055, 2.4));

/** Format a shape display name as a URL slug (e.g. "CMY cube" -> "cmy-cube"). */
export const shapeSlug = (name: string) => name.toLowerCase().replace(/[^a-z0-9]+/g, '-');

const normalizeName = (text: string) => text.toLowerCase().replace(/[^a-z0-9]+/g, '');

function matchShapeName(text: string): string | undefined {
  const wanted = normalizeName(text);
  if (!wanted) return undefined;
  const matches = Object.keys(SHAPES).filter(
    (name) => wanted.startsWith(normalizeName(name)) || normalizeName(name).startsWith(wanted),
  );
  return matches.sort((a, b) => b.length - a.length)[0];
}

/** Parse initial `{ shape, count, layout }` from the page's URL query string. */
export function parseInitialView(query: URLSearchParams): { shape: string; count: number; layout: GridLayout } {
  const legacyDemo = normalizeName(query.get('demo') ?? '');
  const rawCount = query.has('count') ? Number(query.get('count')) : legacyDemo.startsWith('pile') ? 2000 : 1;
  return {
    shape: matchShapeName(query.get('shape') ?? legacyDemo.replace(/^(single|pile)/, '')) ?? 'Diamond',
    count: Math.min(Math.max(Math.round(rawCount) || 1, 1), MAX_COUNT),
    layout: (query.get('layout') === 'grid' ? 'grid' : 'flat') as GridLayout,
  };
}
