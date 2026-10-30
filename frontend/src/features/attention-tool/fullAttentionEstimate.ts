import type { components } from '../../api/generated/schema';

type Model = components['schemas']['ModelRegistration'];

/** Storage estimate for "Save full attention matrices" (an upper bound); a page of unknown size counts as A4. */
export const UNKNOWN_PAGE_POSITIONS = 494;
const FIXED_TEXT_TOKENS = 20;
const EXTRA_PAGE_TOKENS = 7;
const CHARS_PER_TOKEN = 4;

export type FullAttentionGeometry = {
  layers: number;
  heads: number;
  /** Source-image pixels per merged image position, per side (patch size × spatial merge). */
  cell: number;
  /** Most merged image positions per page after resizing. */
  maxPagePositions: number;
  /** Most answer tokens whose decode rows a full-attention capture keeps (T); 0 when the catalogue does not say. */
  decodeSteps: number;
};

/** Most answer tokens a full-attention capture keeps rows for; falls back to `max_new_tokens`, else undefined. */
export function fullAttentionSteps(model?: Model): number | undefined {
  const limits = model?.capabilities.limits;
  const steps = limits?.full_attention_captured_steps ?? limits?.max_new_tokens;
  return typeof steps === 'number' && steps > 0 ? steps : undefined;
}

/** Decoder layers, query heads and patch geometry from the model catalogue, when it declares them. */
export function modelGeometry(model?: Model): FullAttentionGeometry | undefined {
  const components = model?.architecture.components ?? [];
  const text = components.find(c => c.name === 'Language'), vision = components.find(c => c.name === 'Vision');
  const maxPagePositions = model?.capabilities.limits?.image_positions_per_page;
  if (!text?.layers || !text.query_heads || !vision?.patch_size || !maxPagePositions) return undefined;
  return {layers: text.layers, heads: text.query_heads, cell: vision.patch_size * (vision.spatial_merge_size ?? 1), maxPagePositions,
    decodeSteps: fullAttentionSteps(model) ?? 0};
}

/** Merged image positions for one page of the given pixel size (Qwen-VL smart resize). */
export function pageImagePositions(width: number, height: number, cell: number, maxPositions: number) {
  if (!(width > 0 && height > 0)) return UNKNOWN_PAGE_POSITIONS;
  const grid = pageImageGrid(width, height, cell, maxPositions);
  return grid.rows * grid.columns;
}

/** The merged image grid (rows × columns) of one page after the Qwen-VL smart resize. */
export function pageImageGrid(width: number, height: number, cell: number, maxPositions: number) {
  const maxPixels = maxPositions * cell * cell, minPixels = 4 * cell * cell;
  let h = Math.max(cell, Math.round(height / cell) * cell), w = Math.max(cell, Math.round(width / cell) * cell);
  if (h * w > maxPixels) {
    const beta = Math.sqrt(height * width / maxPixels);
    h = Math.max(cell, Math.floor(height / beta / cell) * cell);
    w = Math.max(cell, Math.floor(width / beta / cell) * cell);
  } else if (h * w < minPixels) {
    const beta = Math.sqrt(minPixels / (height * width));
    h = Math.ceil(height * beta / cell) * cell;
    w = Math.ceil(width * beta / cell) * cell;
  }
  return {rows: h / cell, columns: w / cell};
}

export type PageSize = {width: number; height: number} | undefined;

/** Estimated prefill positions n for the supplied pages (in order) and prompt text. */
export function estimatePrefillPositions(pages: PageSize[], geometry: Pick<FullAttentionGeometry, 'cell' | 'maxPagePositions'>, text = '') {
  if (!pages.length) return 0;
  const image = pages.reduce((n, page) => n + (page ? pageImagePositions(page.width, page.height, geometry.cell, geometry.maxPagePositions) : UNKNOWN_PAGE_POSITIONS), 0);
  return image + FIXED_TEXT_TOKENS + EXTRA_PAGE_TOKENS * (pages.length - 1) + Math.ceil(text.length / CHARS_PER_TOKEN);
}

/** Stored bytes of the layer arrays: layers × heads × n(n+1)/2 × 2 (BF16 lower triangle). */
export function fullAttentionBytes(layers: number, heads: number, positions: number) {
  return layers * heads * (positions * (positions + 1) / 2) * 2;
}

/**
 * Stored bytes of T decode rows: layers × heads × Σ_{k=0}^{T−1} (n+k) × 4 (float32). Step k is the row of position
 * n+k−1 over its n+k causal targets, and Σ_{k=0}^{T−1} (n+k) = T·n + T(T−1)/2.
 */
export function decodeRowBytes(layers: number, heads: number, positions: number, steps: number) {
  if (!(steps > 0)) return 0;
  return layers * heads * (steps * positions + steps * (steps - 1) / 2) * 4;
}

/** Decimal gigabytes with two significant figures, e.g. "0.19 GB", "1.3 GB", "12 GB". */
export function formatGigabytes(bytes: number) {
  const gb = bytes / 1e9;
  if (gb > 0 && gb < 0.01) return '< 0.01 GB';
  return `${gb >= 10 ? Math.round(gb) : Number(gb.toPrecision(2))} GB`;
}

export type FullAttentionEstimate = {
  positions: number;
  /** The prefill triangles plus `steps` decode rows: an upper bound, since an answer may stop sooner. */
  bytes: number;
  /** Answer tokens whose decode rows are counted (T). */
  steps: number;
  label: string;
};

/** The whole estimate, or undefined when the model's dimensions or the pages are not known yet. */
export function estimateFullAttention(geometry: FullAttentionGeometry | undefined, pages: PageSize[], text = ''): FullAttentionEstimate | undefined {
  if (!geometry || !pages.length) return undefined;
  const positions = estimatePrefillPositions(pages, geometry, text);
  const steps = geometry.decodeSteps;
  const bytes = fullAttentionBytes(geometry.layers, geometry.heads, positions) + decodeRowBytes(geometry.layers, geometry.heads, positions, steps);
  return {positions, bytes, steps, label: `${steps > 0 ? 'up to ' : ''}about ${formatGigabytes(bytes)}`};
}

/** What the option enables, shared by setup and the acquire drawer. */
export const FULL_ATTENTION_ENABLES = 'Enables rollout, attention flow and graphs, attention structure, and image-patch sources.';
/** What the option keeps that an ordinary capture does not, shared by setup and the acquire drawer. */
export const FULL_ATTENTION_KEEPS = "Keeps every layer's complete prefill attention and every answer token's attention row, so every answer token can be selected (ordinary captures keep the first prediction steps only).";
