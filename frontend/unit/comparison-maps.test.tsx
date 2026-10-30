// @vitest-environment happy-dom
import { afterEach, describe, expect, it } from 'vitest';
import { cleanup, render } from '@testing-library/react';
import { ComparisonMaps } from '../src/features/document/ComparisonMaps';
import type { AttentionMode } from '../src/features/document/attentionDisplay';
import type { DisplaySettings } from '../src/api/client';
import { HOT, PAGE, PAGE_ID, generationPositions, generatorPatchIndex, generatorRow, inferno, overlayOpacity, patchBox, patchValues } from '../fixtures/attention';

afterEach(cleanup);

// Three heads, each hot on a different patch, drawn as separate maps on one shared scale.
const heads = [0, 1, 2].map(head => ({label: `Head ${head}`, row: generatorRow(patchValues([1, 6, 11][head]), {head})}));
const positions = generationPositions.filter(p => p.page_id === PAGE_ID);
const page = {width: PAGE.width, height: PAGE.height, image_url: '/page.png', page_number: 1};

function maps(mode: AttentionMode, scale = HOT) {
  const view = render(<ComparisonMaps comparisons={heads} positions={positions} page={page} scale={scale} mode={mode}
    settings={{attention_opacity: .65, attention_scale_mode: 'auto'} as DisplaySettings} onView={() => {}}/>);
  return [...view.container.querySelectorAll('figure')].map(figure => ({
    caption: figure.querySelector('figcaption')!.textContent,
    patches: [...figure.querySelectorAll('[data-testid="comparison-patch"]')].map(r => ({index: Number(r.getAttribute('data-position-index')),
      fill: r.getAttribute('fill'), opacity: Number(r.getAttribute('fill-opacity')), box: ['x', 'y', 'width', 'height'].map(a => Number(r.getAttribute(a)))}))}));
}

describe('comparison maps', () => {
  it('draws one map per comparison row, each from its own row on the shared scale', () => {
    const figures = maps('overlay');
    expect(figures.map(f => f.caption?.split(' · ')[0])).toEqual(['Head 0', 'Head 1', 'Head 2']);
    figures.forEach((figure, h) => {
      const image = patchValues([1, 6, 11][h]);
      expect(figure.patches).toHaveLength(12);
      figure.patches.forEach((p, k) => {
        expect(p.index).toBe(generatorPatchIndex(k));
        const [x1, y1, x2, y2] = patchBox(k);
        expect(p.box).toEqual([x1, y1, x2 - x1, y2 - y1]);
        expect(p.opacity).toBeCloseTo(overlayOpacity(image[k], HOT), 12);
      });
      const brightest = figure.patches.reduce((a, b) => b.opacity > a.opacity ? b : a);
      expect(brightest.index).toBe(generatorPatchIndex([1, 6, 11][h]));
    });
  });

  it('heatmap maps follow the page heatmap colours', () => {
    const figures = maps('heatmap');
    figures.forEach((figure, h) => figure.patches.forEach((p, k) =>
      expect(p.fill).toBe(`rgb(${inferno(patchValues([1, 6, 11][h])[k] / HOT).join(',')})`)));
  });

  it('a larger shared scale dims every map equally', () => {
    const figures = maps('overlay', HOT * 2);
    expect(figures.map(f => Math.max(...f.patches.map(p => p.opacity)))).toEqual([.325, .325, .325]);
  });
});
