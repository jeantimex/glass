// Bake Suzanne's 128^3 signed-distance volume to public/suzanne_sdf_128.bin.
//
//   node scripts/bake-suzanne-sdf.ts [fillet]
//
// Blender's Suzanne is made for opaque rendering: an open head shell with a hole for each
// eye, plugged by an open half-sphere eye cup. The general bake (src/glass/bake.ts) makes her
// the solid a cast glass figurine would be: each piece Loop-subdivided (3 levels, as the
// smooth model shows), every hole capped, and the head and eyes merged. The holes are a
// little smaller than the eyes, which bulge out through them, so the merge keeps each hole's
// rim as a crease around its eye, softened by `fillet` (volume units; default 0.01, about
// half a voxel, so it does not stair-step; 0 keeps it crisp).
//
// The mesh is already unit size (half-width 1), so it is baked in its own coordinates.
import { readFileSync, writeFileSync } from 'node:fs';
import { fileURLToPath } from 'node:url';
import { bakeGlassVolume, VOLUME_EXTENT, VOLUME_SIZE } from '../src/glass/bake.ts';

const fillet = Math.max(Number(process.argv[2] ?? 0.01) || 0, 0);
const meshUrl = new URL('../public/suzanne_mesh.bin', import.meta.url);
const outUrl = new URL('../public/suzanne_sdf_128.bin', import.meta.url);

/* The mesh: [vertexCount, indexCount, pad] + positions + normals + indices. */
const bytes = readFileSync(meshUrl);
const buffer = bytes.buffer.slice(bytes.byteOffset, bytes.byteOffset + bytes.byteLength);
const [vertexCount, indexCount] = new Uint32Array(buffer, 0, 2);
const positions = new Float32Array(buffer, 12, vertexCount * 3);
const indices = new Uint32Array(buffer, 12 + vertexCount * 24, indexCount);

const started = Date.now();
const { data, stats } = bakeGlassVolume(positions, indices, {
  subdivisions: 3,
  fillet,
  center: [0, 0, 0],
  halfWidth: 1,
});
writeFileSync(outUrl, new Uint8Array(data.buffer));
console.log(
  `wrote ${fileURLToPath(outUrl)} in ${((Date.now() - started) / 1000).toFixed(1)} s: ${stats.pieces} solids ` +
    `(${stats.holesCapped} holes capped, ${stats.triangles} triangles, ${stats.degenerateColumns} degenerate columns), ` +
    `fillet ${fillet} (${(fillet / (VOLUME_EXTENT / VOLUME_SIZE)).toFixed(1)} voxels)`,
);
