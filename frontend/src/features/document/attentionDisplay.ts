import type { DisplaySettings } from '../../api/client';
import type { Attention, Generation } from './GenerationPanel';

export function attentionWeightLabel(value: number): string {
  return value !== 0 && Math.abs(value) < 1e-6 ? value.toExponential(3) : value.toFixed(6);
}

/**
 * A weight for reading at a glance: at most three decimals, "< 0.001" when smaller and "≈ 0" at
 * floating-point noise, so no scientific notation. Details › Display switches to the exact label.
 */
export function roundedWeightLabel(value: number): string {
  const size = Math.abs(value);
  if (size < 1e-9) return value === 0 ? '0' : '≈ 0';
  if (size < 0.0005) return value > 0 ? '< 0.001' : '> −0.001';
  return String(Number(value.toFixed(3))).replace('-', '−');
}

/** A share as a percentage (one decimal below 10%): a tiny positive share reads "<0.1%", never 0%. */
export function sharePercent(x: number): string {
  return x > 0 && x < .001 ? '<0.1%' : `${(x * 100).toFixed(x < .1 ? 1 : 0)}%`;
}

/** A large ratio in words: 9,593,546 reads "9.6 million". */
export function roundedTimesLabel(times: number): string {
  for (const [size, name] of [[1e12, 'trillion'], [1e9, 'billion'], [1e6, 'million']] as const)
    if (times >= size) return `${Number((times / size).toFixed(1))} ${name}`;
  return times >= 10 ? Math.round(times).toLocaleString() : times.toFixed(1);
}

// The color scale never changes stored weights or allocation. Sharing the image
// maximum across all supplied pages keeps page-to-page comparisons meaningful.
export function imageAttentionMaximum(capture?: Generation, row?: Attention): number {
  if (!capture || !row) return 1;
  return capture.positions.reduce((maximum, position) =>
    position.role === 'image' && position.bbox && position.index < row.target_count
      ? Math.max(maximum, row.values[position.index]) : maximum, 0);
}

export const ATTENTION_FILL = '#d99a36';

// One intensity mapping for page patches and tokens; `floor` is where the scale starts.
export function attentionAlpha(value: number, maximum: number, opacity: number, floor = 0): number {
  return maximum > floor ? Math.max(0, Math.min(opacity, (value - floor) / (maximum - floor) * opacity)) : 0;
}

/** Position of a value on the colour scale, 0 at the floor and 1 at the maximum. */
export function scalePosition(value: number, maximum: number, floor = 0): number {
  return maximum > floor ? Math.min(1, Math.max(0, (value - floor) / (maximum - floor))) : 0;
}

/** The scale floor for shown values: 0, or their minimum when the range starts at the page minimum. */
export function rangeFloor(values: number[], mode: DisplaySettings['attention_range_floor'] | undefined): number {
  if (mode !== 'minimum' || !values.length) return 0;
  return Math.min(...values);
}

/** How the page shows attention: hidden, opaque heatmap, amber overlay, or the colour map blended over the page. */
export type AttentionMode = 'original' | 'heatmap' | 'overlay' | 'blend';
/** The display modes in the order Display settings and the comparison maps' toolbar offer them. */
export const ATTENTION_MODES: {id: AttentionMode; label: string}[] = [
  {id: 'original', label: 'Original image'}, {id: 'heatmap', label: 'Heatmap'}, {id: 'overlay', label: 'Overlay'}, {id: 'blend', label: 'Blend'}];

// Investigations saved before the display toolbar hid attention with the view-state overlay flag.
export function attentionMode(settings: DisplaySettings, overlay?: boolean | null): AttentionMode {
  return overlay === false ? 'original' : settings.attention_display_mode ?? 'overlay';
}

type RGB = [number, number, number];
export type Colormap = NonNullable<DisplaySettings['attention_colormap']>;
// Perceptually ordered colour maps, sampled from matplotlib's. Inferno keeps the exact stops shared
// with the attention-research explorer and visual_doc_retrieval; the others use nine even stops.
const EVEN = [0, .125, .25, .375, .5, .625, .75, .875, 1];
const MAPS: Record<Colormap, {stops: number[]; colours: RGB[]}> = {
  inferno: {stops: [0, .13, .25, .38, .5, .63, .75, .88, 1],
    colours: [[0, 0, 4], [31, 12, 72], [85, 15, 109], [136, 34, 106], [186, 54, 85], [227, 89, 51], [249, 140, 10], [249, 201, 50], [252, 255, 164]]},
  magma: {stops: EVEN, colours: [[0, 0, 4], [28, 16, 68], [79, 18, 123], [129, 37, 129], [181, 54, 122], [229, 80, 100], [251, 135, 97], [254, 194, 135], [252, 253, 191]]},
  viridis: {stops: EVEN, colours: [[68, 1, 84], [70, 50, 127], [54, 92, 141], [39, 127, 142], [31, 161, 135], [74, 194, 109], [159, 218, 58], [234, 229, 26], [253, 231, 37]]},
  plasma: {stops: EVEN, colours: [[13, 8, 135], [84, 2, 163], [139, 10, 165], [185, 50, 137], [219, 92, 104], [244, 136, 73], [254, 188, 43], [244, 237, 39], [240, 249, 33]]},
};
export const COLORMAPS: {id: Colormap; label: string}[] = [
  {id: 'inferno', label: 'Inferno'}, {id: 'magma', label: 'Magma'}, {id: 'viridis', label: 'Viridis'}, {id: 'plasma', label: 'Plasma'}];

export function heatColor(t: number, map: Colormap = 'inferno'): RGB {
  const {stops, colours} = MAPS[map] ?? MAPS.inferno;
  const x = Math.min(1, Math.max(0, t));
  const i = stops.findIndex(stop => x <= stop);
  if (i <= 0) return colours[0];
  const t0 = stops[i - 1], t1 = stops[i], c0 = colours[i - 1], c1 = colours[i];
  const f = (x - t0) / (t1 - t0);
  return c0.map((c, k) => Math.round(c + f * (c1[k] - c))) as RGB;
}

/** CSS gradient of a colour map, low to high. */
export function mapGradient(map: Colormap = 'inferno'): string {
  const {stops, colours} = MAPS[map] ?? MAPS.inferno;
  return `linear-gradient(to right, ${colours.map(([r, g, b], i) => `rgb(${r},${g},${b}) ${stops[i] * 100}%`).join(', ')})`;
}

export function rgb([r, g, b]: RGB): string {
  return `rgb(${r},${g},${b})`;
}

export type PatchGrid = {x1: number; y1: number; x2: number; y2: number; columns: number[]; rows: number[]};

/**
 * Column and row edges when the patch boxes tile a regular grid, which lets
 * smoothing interpolate one pixel per patch. Undefined for any other layout.
 */
export function patchGrid(boxes: number[][]): PatchGrid | undefined {
  const columns = [...new Set(boxes.map(b => b[0]))].sort((a, b) => a - b);
  const rows = [...new Set(boxes.map(b => b[1]))].sort((a, b) => a - b);
  if (!boxes.length || columns.length * rows.length !== boxes.length) return undefined;
  return {x1: columns[0], y1: rows[0], x2: Math.max(...boxes.map(b => b[2])), y2: Math.max(...boxes.map(b => b[3])), columns, rows};
}
