// CMY dichroic cube face filters in TSL.
// Each pair of opposite cube faces (±X, ±Y, ±Z) acts as a subtractive colour filter
// (Cyan, Yellow, or Magenta) that can be permuted across axes via `CmyFaceOrder`.
/* eslint-disable @typescript-eslint/no-explicit-any -- TSL node types are too loose to spell out */
import * as TSL from 'three/tsl';

type N = any;
const { abs, mix, select, vec3 } = TSL as N;

export type CmyFaceOrder = 'cym' | 'myc' | 'ycm' | 'mcy' | 'cmy' | 'ymc';

/** Axis colour assignments (0 = Cyan, 1 = Yellow, 2 = Magenta) for [X, Y, Z] faces. */
export const CMY_FACE_ORDERS: Record<CmyFaceOrder, readonly [number, number, number]> = {
  cym: [0, 1, 2],
  myc: [2, 1, 0],
  ycm: [1, 0, 2],
  mcy: [2, 0, 1],
  cmy: [0, 2, 1],
  ymc: [1, 2, 0],
};

/** Linear RGB transmittance of one dichroic filter (0 = Cyan, 1 = Yellow, 2 = Magenta). */
export const cmyChannelColor = (colorId: N): N =>
  select(
    colorId.lessThan(0.5),
    vec3(0.1, 0.92, 1), // Cyan: blocks red, passes green & blue
    select(colorId.lessThan(1.5), vec3(1, 0.9, 0.2), vec3(1, 0.14, 0.92)), // Yellow : Magenta
  );

/**
 * CMY cube face transmittance: given `localNormal` in the cube's local space, selects the
 * dominant axis (±X, ±Y, or ±Z) and blends from clear (0) to full dichroic filter (`strength`).
 * `order` maps [X, Y, Z] to [0..2] (defaults to X=Cyan, Y=Yellow, Z=Magenta).
 */
export function cmyTint(localNormal: N, strength: N, order?: N): N {
  const absN = abs(localNormal).toVar();
  const xColor = order ? cmyChannelColor(order.x) : vec3(0.1, 0.92, 1);
  const yColor = order ? cmyChannelColor(order.y) : vec3(1, 0.9, 0.2);
  const zColor = order ? cmyChannelColor(order.z) : vec3(1, 0.14, 0.92);
  const faceFilter = select(
    absN.x.greaterThanEqual(absN.y).and(absN.x.greaterThanEqual(absN.z)),
    xColor,
    select(absN.y.greaterThanEqual(absN.z), yColor, zColor),
  );
  return mix(vec3(1), faceFilter, strength);
}
