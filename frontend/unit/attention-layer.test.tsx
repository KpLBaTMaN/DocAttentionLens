// @vitest-environment happy-dom
import { afterEach, describe, expect, it } from 'vitest';
import { cleanup, render } from '@testing-library/react';
import { AttentionLayer, type MapPosition } from '../src/features/document/AttentionLayer';
import type { AttentionMode } from '../src/features/document/attentionDisplay';
import { HOT, OVERLAY_RGB, PATCH_COUNT, SOURCE_OUTLINE, generationPositions, generatorPatchIndex, generatorRow, inferno,
  overlayOpacity, patchBox, patchValues, retrieverPositions, retrieverRow, type Row } from '../fixtures/attention';

// Known-answer page: 4 × 3 patches of 100 px. Each test states the picture it expects
// from the fixture's numbers; the app's own colour and opacity helpers are not reused.

afterEach(cleanup);

const pagePositions: MapPosition[] = generationPositions.filter(p => p.page_id);
const hex = `#${OVERLAY_RGB.map(c => c.toString(16).padStart(2, '0')).join('')}`;

function draw(row: Row, {positions = pagePositions, mode = 'overlay' as AttentionMode, maximum = HOT, opacity = .65, outlineOnly = false, showPatches = false} = {}) {
  const view = render(<svg><AttentionLayer positions={positions} row={row} maximum={maximum} mode={mode} opacity={opacity} smooth={false} showPatches={showPatches} outlineOnly={outlineOnly}/></svg>);
  const patches = [...view.container.querySelectorAll('[data-testid="attention-patch"]')].map(r => ({
    index: Number(r.getAttribute('data-position-index')),
    box: ['x', 'y', 'width', 'height'].map(a => Number(r.getAttribute(a))),
    fill: r.getAttribute('fill'), opacity: Number(r.getAttribute('fill-opacity')), stroke: r.getAttribute('stroke')}));
  return {view, patches};
}

describe('patch placement', () => {
  it('draws one patch per mapped image position, at that position’s own source box', () => {
    const {patches} = draw(generatorRow(patchValues(5)));
    expect(patches.map(p => p.index)).toEqual(Array.from({length: PATCH_COUNT}, (_, k) => generatorPatchIndex(k)));
    patches.forEach((p, k) => {
      const [x1, y1, x2, y2] = patchBox(k);
      expect(p.box).toEqual([x1, y1, x2 - x1, y2 - y1]);
    });
  });

  it('puts the hot weight on the hot patch and nowhere else', () => {
    for (const hot of [0, 5, 11]) {
      const {patches} = draw(generatorRow(patchValues(hot)));
      const brightest = patches.reduce((a, b) => b.opacity > a.opacity ? b : a);
      expect(brightest.index).toBe(generatorPatchIndex(hot));
      expect(patches.filter(p => p.opacity === .65)).toHaveLength(1);
      cleanup();
    }
  });

  it('skips targets outside the causal row and positions without a page box', () => {
    // Question token at position 9 sees targets 0–9: patches 0–6 only.
    const row = {...generatorRow(patchValues(2)), target_count: 10};
    const {patches} = draw(row, {positions: generationPositions});
    expect(patches.map(p => p.index)).toEqual([3, 4, 5, 6, 7, 8, 9]);
  });
});

describe('overlay and heatmap intensity', () => {
  it('overlay opacity is weight ÷ display maximum × opacity, in the amber fill', () => {
    const image = patchValues(5);
    const {patches} = draw(generatorRow(image));
    patches.forEach((p, k) => {
      expect(p.fill).toBe(hex);
      expect(p.opacity).toBeCloseTo(overlayOpacity(image[k], HOT), 12);
    });
  });

  it('values above a fixed maximum saturate at the chosen opacity', () => {
    const {patches} = draw(generatorRow(patchValues(5)), {maximum: HOT / 2, opacity: .4});
    expect(patches[5].opacity).toBe(.4);
    expect(patches[0].opacity).toBeCloseTo(.004 / (HOT / 2) * .4, 12);
  });

  it('heatmap paints every patch opaque with its inferno colour', () => {
    const image = patchValues(5);
    const {patches} = draw(generatorRow(image), {mode: 'heatmap'});
    patches.forEach((p, k) => {
      expect(p.fill).toBe(`rgb(${inferno(image[k] / HOT).join(',')})`);
      expect(p.opacity).toBe(1);
    });
    expect(patches[5].fill).toBe('rgb(252,255,164)');
  });

  it('Original draws no patches at all', () => {
    const {patches, view} = draw(generatorRow(patchValues(5)), {mode: 'original'});
    expect(patches).toEqual([]);
    expect(view.container.querySelector('[data-testid="attention-layer"]')?.getAttribute('data-mode')).toBe('original');
  });

  it('a zero display maximum draws nothing visible rather than dividing by zero', () => {
    const {patches} = draw(generatorRow(patchValues(5)), {maximum: 0});
    expect(patches.every(p => p.opacity === 0)).toBe(true);
  });
});

describe('source and selection marks', () => {
  it('outlines the retriever source patch and no other', () => {
    const {patches} = draw(retrieverRow(patchValues(5), {source_index: 7}), {positions: retrieverPositions});
    expect(patches.filter(p => p.stroke === SOURCE_OUTLINE).map(p => p.index)).toEqual([7]);
    expect(patches.filter(p => p.stroke !== SOURCE_OUTLINE).every(p => p.stroke === 'none')).toBe(true);
  });

  it('outline-only marks the selected position without painting any weight', () => {
    const {patches} = draw(retrieverRow(patchValues(5), {source_index: 3}), {positions: retrieverPositions, outlineOnly: true});
    expect(patches.every(p => p.opacity === 0)).toBe(true);
    expect(patches.filter(p => p.stroke === SOURCE_OUTLINE).map(p => p.index)).toEqual([3]);
  });

  it('the patch grid traces every drawn patch boundary', () => {
    const {view} = draw(generatorRow(patchValues(5)), {showPatches: true});
    const d = view.container.querySelector('[data-testid="attention-patch-grid"]')!.getAttribute('d')!;
    expect(d.match(/M/g)).toHaveLength(PATCH_COUNT);
    expect(d).toContain('M100 200H200V300H100Z');
  });
});
