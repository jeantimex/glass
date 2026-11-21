// Instanced geometry batch for one shape `KIND`: manages per-instance center,
// quaternion, size, optical variation, and linear tint buffers with partial GPU uploads.
import * as THREE from 'three/webgpu';
import { MAX_OBJECTS, type GlassObjects } from './objects';

export class InstanceBatch {
  readonly geometry = new THREE.InstancedBufferGeometry();
  readonly center: THREE.InstancedBufferAttribute;
  readonly quat: THREE.InstancedBufferAttribute;
  readonly size: THREE.InstancedBufferAttribute;
  readonly vary: THREE.InstancedBufferAttribute;
  readonly tint: THREE.InstancedBufferAttribute;
  count = 0;

  /** Number of instances uploaded on the previous commit (avoids uploading empty batches). */
  private uploadedCount = 0;

  /** @param capacity Maximum number of instances this batch can hold. */
  constructor(baseGeometry: THREE.BufferGeometry, capacity = MAX_OBJECTS) {
    this.center = new THREE.InstancedBufferAttribute(new Float32Array(capacity * 3), 3);
    this.quat = new THREE.InstancedBufferAttribute(new Float32Array(capacity * 4), 4);
    this.size = new THREE.InstancedBufferAttribute(new Float32Array(capacity * 3), 3);
    this.vary = new THREE.InstancedBufferAttribute(new Float32Array(capacity * 3), 3);
    this.tint = new THREE.InstancedBufferAttribute(new Float32Array(capacity * 3), 3);

    this.geometry.index = baseGeometry.index;
    this.geometry.setAttribute('position', baseGeometry.getAttribute('position'));
    this.geometry.setAttribute('normal', baseGeometry.getAttribute('normal'));

    // Kept as static buffers on purpose: Three.js re-uploads dynamic buffers in full on every
    // render pass; instead we upload only the active slice inside `commit()`.
    this.geometry.setAttribute('instCenter', this.center);
    this.geometry.setAttribute('instQuat', this.quat);
    this.geometry.setAttribute('instSize', this.size);
    this.geometry.setAttribute('instVary', this.vary);
    this.geometry.setAttribute('instTint', this.tint);
    this.geometry.instanceCount = 0;
  }

  /** Append object `index` from `objects` into this batch. */
  push(objects: GlassObjects, index: number): void {
    const slot = this.count++;
    const center = this.center.array;
    const quat = this.quat.array;
    const size = this.size.array;
    const vary = this.vary.array;
    const tint = this.tint.array;

    const dst3 = slot * 3;
    const src3 = index * 3;
    center[dst3] = objects.center[src3];
    center[dst3 + 1] = objects.center[src3 + 1];
    center[dst3 + 2] = objects.center[src3 + 2];

    const dst4 = slot * 4;
    const src4 = index * 4;
    quat[dst4] = objects.quat[src4];
    quat[dst4 + 1] = objects.quat[src4 + 1];
    quat[dst4 + 2] = objects.quat[src4 + 2];
    quat[dst4 + 3] = objects.quat[src4 + 3];

    size[dst3] = objects.size[src3];
    size[dst3 + 1] = objects.size[src3 + 1];
    size[dst3 + 2] = objects.size[src3 + 2];

    vary[dst3] = objects.vary[src3];
    vary[dst3 + 1] = objects.vary[src3 + 1];
    vary[dst3 + 2] = objects.vary[src3 + 2];

    tint[dst3] = objects.tint[src3];
    tint[dst3 + 1] = objects.tint[src3 + 1];
    tint[dst3 + 2] = objects.tint[src3 + 2];
  }

  /** Upload the active range of instance attributes to the GPU. */
  commit(): void {
    this.geometry.instanceCount = this.count;
    if (!this.count && !this.uploadedCount) return;
    this.uploadedCount = this.count;
    for (const attr of [this.center, this.quat, this.size, this.vary, this.tint]) {
      attr.clearUpdateRanges();
      attr.addUpdateRange(0, Math.max(this.count, 1) * attr.itemSize);
      attr.needsUpdate = true;
    }
  }
}
