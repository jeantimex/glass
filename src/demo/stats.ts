// Corner HUD overlay showing FPS, triangle count, and per-pass GPU timings.

export interface StatsPanelOptions {
  isVisible: () => boolean;
  getStatus: () => { fps: string; triangles: string };
  shouldPollGpu: () => boolean;
  readPassTimes: () => Promise<Record<string, number> | null>;
}

const PASS_ORDER = [
  'Shadow map',
  'Caustics',
  'Reflection',
  'Ground',
  'Glass',
  'Far glass',
  'Near glass',
  'Finishing',
  'Other',
];

/** Mount the bottom-left stats overlay and begin polling GPU pass timings every 500 ms. */
export function mountStatsPanel(opts: StatsPanelOptions): HTMLElement {
  const panel = document.createElement('div');
  panel.id = 'stats';
  panel.style.cssText =
    'position:fixed;left:12px;bottom:12px;display:grid;grid-template-columns:auto auto;column-gap:16px;padding:8px 10px;font:12px/1.5 ui-monospace,monospace;font-variant-numeric:tabular-nums;color:#eee;background:rgba(20,20,20,0.8);border-radius:6px;pointer-events:none;z-index:10';
  panel.style.display = opts.isVisible() ? 'grid' : 'none';
  document.body.append(panel);

  const renderRows = (rows: ([string, string] | [string])[]) => {
    panel.replaceChildren(
      ...rows.flatMap((row) => {
        const label = document.createElement('span');
        label.textContent = row[0];
        if (row.length === 1) {
          label.style.cssText = 'grid-column:1 / -1;margin-top:6px;color:#aaa';
          return [label];
        }
        const value = document.createElement('span');
        value.textContent = row[1];
        value.style.textAlign = 'right';
        return [label, value];
      }),
    );
  };

  const recentTimes = new Map<string, { ms: number; at: number }>();

  const poll = async () => {
    const now = performance.now();
    if (opts.shouldPollGpu()) {
      for (const [name, ms] of Object.entries((await opts.readPassTimes()) ?? {})) {
        recentTimes.set(name, { ms, at: now });
      }
    }
    panel.style.display = opts.isVisible() ? 'grid' : 'none';
    const recent = PASS_ORDER.filter((name) => recentTimes.has(name) && now - recentTimes.get(name)!.at < 2000).map(
      (name) => [name, recentTimes.get(name)!.ms] as const,
    );
    const totalMs = recent.reduce((sum, [, ms]) => sum + ms, 0);
    const { fps, triangles } = opts.getStatus();
    renderRows([
      ['FPS', fps],
      ['Triangles', triangles],
      ['GPU time per frame'],
      ...recent.map(([name, ms]) => [name, `${ms.toFixed(2)} ms`] as [string, string]),
      ['Drawing', `${totalMs.toFixed(2)} ms`],
    ]);
    setTimeout(poll, 500);
  };

  poll();
  return panel;
}
