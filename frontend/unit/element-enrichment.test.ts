import { describe, expect, it } from 'vitest';
import { enrichmentPosition, pageEnrichment } from '../src/features/document/elementEnrichment';
import { PAGE_ID, PATCH_COUNT, TABLE_ID, TEXT_ID, attentionRow, retrieverOwnership, retrieverPositions, retrieverRow } from '../fixtures/attention';

// Fixture page: 12 patches; the table owns patches 0–1, the text 6, 7, 10 and 11, the background the other six.
// Expected values come from the patch values and ownership directly, not from app code.
const positions = retrieverPositions.map((p, i) => ({...p, page_id: PAGE_ID, ownership: retrieverOwnership[i]}));
const image = Array.from({length: PATCH_COUNT}, (_, k) => 0.01 * (k + 1));
const sum = (values: number[], ks: number[]) => ks.reduce((s, k) => s + values[k], 0);
const all = [...Array(PATCH_COUNT).keys()];

describe('element Enrichment', () => {
  it("divides each bucket's share of image attention by its share of image positions", () => {
    const e = pageEnrichment(retrieverRow(image), positions, PAGE_ID);
    const I = sum(image, all);
    expect(e.get(TABLE_ID)!.areaShare).toBeCloseTo(2 / 12, 12);
    expect(e.get(TABLE_ID)!.imageShare).toBeCloseTo(sum(image, [0, 1]) / I, 12);
    expect(e.get(TABLE_ID)!.value).toBeCloseTo((sum(image, [0, 1]) / I) / (2 / 12), 12);
    expect(e.get(TEXT_ID)!.value).toBeCloseTo((sum(image, [6, 7, 10, 11]) / I) / (4 / 12), 12);
    expect(e.get('background')!.areaShare).toBeCloseTo(6 / 12, 12);
    expect(e.get(TABLE_ID)!.ceiling).toBeCloseTo(6, 12);
    // Prompt and special tokens own no image area; the image buckets partition both denominators.
    expect(e.has('non_page')).toBe(false);
    expect([...e.values()].reduce((s, x) => s + x.areaShare, 0)).toBeCloseTo(1, 12);
    expect([...e.values()].reduce((s, x) => s + x.imageShare, 0)).toBeCloseTo(1, 12);
  });

  it('measures only the image positions a causal source reaches', () => {
    const reached = image.map((v, k) => k <= 6 ? v : 0);
    const e = pageEnrichment(retrieverRow(reached, {source_index: 6}), positions, PAGE_ID, {source: 6, reaches: i => i <= 6, seen: 7, total: PATCH_COUNT});
    expect(e.get(TABLE_ID)!.areaShare).toBeCloseTo(2 / 7, 12);
    expect(e.get(TEXT_ID)!.areaShare).toBeCloseTo(1 / 7, 12);
    expect(e.get(TABLE_ID)!.value).toBeCloseTo((sum(image, [0, 1]) / sum(image, [0, 1, 2, 3, 4, 5, 6])) / (2 / 7), 12);
  });

  it('flags a bucket that owns under one image position', () => {
    const tiny = 'element_fixture_tiny';
    const split = positions.map(p => p.index === 4 ? {...p, ownership: {[tiny]: .25, background: .75}} : p);
    const row = attentionRow({positions: split, patchIndex: k => k, image, targetCount: split.length, capture_id: 'retrieval_fixture', source_index: 0});
    const e = pageEnrichment(row, split, PAGE_ID);
    expect(e.get(tiny)!.positions).toBeCloseTo(.25, 12);
    expect(e.get(tiny)!.belowOnePosition).toBe(true);
    expect(e.get(tiny)!.value).toBeCloseTo((.25 * image[4] / sum(image, all)) / (.25 / 12), 12);
    expect(e.get(TABLE_ID)!.belowOnePosition).toBe(false);
  });

  it('places values on a log bar from ¼× to 4×, with 1× in the middle', () => {
    expect(enrichmentPosition(1)).toBe(50);
    expect(enrichmentPosition(2)).toBe(75);
    expect(enrichmentPosition(4)).toBe(100);
    expect(enrichmentPosition(.25)).toBe(0);
    expect(enrichmentPosition(8)).toBe(100);
    expect(enrichmentPosition(0)).toBe(0);
  });
});
