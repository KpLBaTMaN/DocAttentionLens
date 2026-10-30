import { describe, expect, it } from 'vitest';
import type { components } from '../src/api/generated/schema';
import { UNKNOWN_PAGE_POSITIONS, decodeRowBytes, estimateFullAttention, estimatePrefillPositions, formatGigabytes, fullAttentionBytes, fullAttentionSteps,
  modelGeometry, pageImagePositions } from '../src/features/attention-tool/fullAttentionEstimate';

type Model = components['schemas']['ModelRegistration'];
// Shaped like GET /api/v1/setup/models, trimmed to the fields the estimate reads.
// Limits as the catalogue serves them for the Qwen generators: 8 retained steps on ordinary captures, every answer token
// (up to 64) with full attention.
const LIMITS = {image_positions_per_page: 512, captured_steps: 8, max_new_tokens: 64, full_attention_captured_steps: 64};
const model = (layers: number, heads: number, patch: number, limits: Record<string, number> = LIMITS): Model => ({
  registration_id: 'm', role: 'generation', name: 'Model', model_id: 'm', revision: 'r', validation: '',
  architecture: {model_type: 'qwen', architectures: [], components: [
    {name: 'Language', layers, query_heads: heads, kv_heads: 2, hidden_size: 1, patch_size: null, spatial_merge_size: null, image_size: null},
    {name: 'Vision', layers: 32, query_heads: 16, kv_heads: null, hidden_size: 1, patch_size: patch, spatial_merge_size: 2, image_size: null}],
  embedding_dim: null, dtype: 'bfloat16'},
  capabilities: {limits},
} as unknown as Model);
const qwen25 = modelGeometry(model(36, 16, 14))!, qwen3 = modelGeometry(model(36, 32, 16))!;
const SYSTEM = 'Answer the question using only the supplied document page images. If the images do not provide enough evidence, say so. Give a concise answer.';

describe('full attention storage estimate', () => {
  it('reads layers, heads and the merged cell from the model catalogue', () => {
    expect(qwen25).toEqual({layers: 36, heads: 16, cell: 28, maxPagePositions: 512, decodeSteps: 64});
    expect(qwen3.cell).toBe(32);
    expect(modelGeometry(undefined)).toBeUndefined();
    const undeclared = model(36, 16, 14);
    undeclared.architecture.components = [];
    expect(modelGeometry(undeclared)).toBeUndefined();
  });

  it('matches the image positions of saved captures (Qwen smart resize)', () => {
    // Saved captures: 2480 × 3509 → 532 × 728 (494 positions); 2475 × 3225 → 532 × 700 (475); Qwen3-VL 2480 × 3509 → 608 × 832 (494).
    expect(pageImagePositions(2480, 3509, 28, 512)).toBe(494);
    expect(pageImagePositions(2475, 3225, 28, 512)).toBe(475);
    expect(pageImagePositions(2480, 3509, 32, 512)).toBe(494);
    // Small images are kept at their size (rounded to whole cells), never below four cells.
    expect(pageImagePositions(280, 280, 28, 512)).toBe(100);
    expect(pageImagePositions(20, 20, 28, 512)).toBe(4);
    expect(pageImagePositions(0, 0, 28, 512)).toBe(UNKNOWN_PAGE_POSITIONS);
  });

  it('adds template and text tokens per page and per character', () => {
    const page = {width: 2480, height: 3509};
    // 494 image positions + 20 fixed + ceil(8 / 4) text tokens.
    expect(estimatePrefillPositions([page], qwen25, 'abcdefgh')).toBe(516);
    // A second page adds its image positions and 7 template tokens; an unknown page counts 494.
    expect(estimatePrefillPositions([page, undefined], qwen25, 'abcdefgh')).toBe(516 + 494 + 7);
    expect(estimatePrefillPositions([], qwen25)).toBe(0);
  });

  it('computes layers × heads × n(n+1)/2 × 2 bytes', () => {
    expect(fullAttentionBytes(1, 1, 1)).toBe(2);
    expect(fullAttentionBytes(36, 16, 569)).toBe(36 * 16 * 569 * 570);
    expect(fullAttentionBytes(2, 4, 10)).toBe(2 * 4 * 55 * 2);
  });

  it('reads T from full_attention_captured_steps, else max_new_tokens (an older catalogue), else counts no decode rows', () => {
    expect(fullAttentionSteps(model(36, 16, 14))).toBe(64);
    expect(fullAttentionSteps(model(36, 16, 14, {image_positions_per_page: 512, max_new_tokens: 32}))).toBe(32);
    expect(fullAttentionSteps(model(36, 16, 14, {image_positions_per_page: 512, max_new_tokens: 64, full_attention_captured_steps: 48}))).toBe(48);
    expect(fullAttentionSteps(model(36, 16, 14, {image_positions_per_page: 512}))).toBeUndefined();
    expect(fullAttentionSteps(undefined)).toBeUndefined();
    expect(modelGeometry(model(36, 16, 14, {image_positions_per_page: 512}))?.decodeSteps).toBe(0);
  });

  it('adds the decode rows: layers × heads × Σ_{k=0}^{T−1} (n+k) × 4 bytes', () => {
    // Step k is the row of position n+k−1 over its n+k causal targets, stored as float32.
    expect(decodeRowBytes(1, 1, 10, 1)).toBe(10 * 4);
    expect(decodeRowBytes(1, 1, 10, 3)).toBe((10 + 11 + 12) * 4);
    expect(decodeRowBytes(2, 4, 10, 0)).toBe(0);
    let rows = 0;
    for (let k = 0; k < 64; k++) rows += 569 + k;
    expect(decodeRowBytes(36, 16, 569, 64)).toBe(36 * 16 * rows * 4);
  });

  it('labels the size in decimal gigabytes with two significant figures', () => {
    expect(formatGigabytes(192_099_456)).toBe('0.19 GB');
    expect(formatGigabytes(1_337_000_000)).toBe('1.3 GB');
    expect(formatGigabytes(12_400_000_000)).toBe('12 GB');
    expect(formatGigabytes(1_000_000)).toBe('< 0.01 GB');
  });

  it('gives the proposal’s prefill figures (about 0.2 GB for one page with Qwen2.5-VL-3B, about 1.4 GB for two with Qwen3-VL-8B) plus up to 64 decode rows', () => {
    const a4 = {width: 2480, height: 3509};
    const question = 'What is the total amount shown in the table of the second section of this report?';
    const one = estimateFullAttention(qwen25, [a4], SYSTEM + question)!;
    // 0.19 GB of prefill triangles and 0.089 GB for 64 decode rows: an upper bound, since the answer may stop sooner.
    expect(one.steps).toBe(64);
    expect(one.bytes).toBe(fullAttentionBytes(36, 16, one.positions) + decodeRowBytes(36, 16, one.positions, 64));
    expect(one.label).toBe('up to about 0.28 GB');
    // A catalogue that declares no answer-token limit counts the prefill triangles only.
    const prefillOnly = estimateFullAttention(modelGeometry(model(36, 16, 14, {image_positions_per_page: 512})), [a4], SYSTEM + question)!;
    expect(prefillOnly.label).toBe('about 0.19 GB');
    // A saved one-page capture with this instruction had 569 prefill positions; the estimate stays close and above.
    expect(one.positions).toBeGreaterThanOrEqual(569);
    expect(one.positions).toBeLessThan(569 + 20);
    const two = estimateFullAttention(qwen3, [a4, a4], SYSTEM + question)!;
    const triangles = fullAttentionBytes(36, 32, two.positions);
    expect(triangles).toBeGreaterThan(1.2e9);
    expect(triangles).toBeLessThan(1.5e9);
    expect(two.label).toBe('up to about 1.6 GB');
    expect(estimateFullAttention(undefined, [a4])).toBeUndefined();
    expect(estimateFullAttention(qwen25, [])).toBeUndefined();
  });
});
