// @vitest-environment happy-dom
import { afterEach, describe, expect, it } from 'vitest';
import { cleanup, fireEvent, render } from '@testing-library/react';
import { AttentionLayer } from '../src/features/document/AttentionLayer';
import { ComparisonMaps } from '../src/features/document/ComparisonMaps';
import { PageCanvas, type ElementRegions } from '../src/features/document/PageCanvas';
import { rampStyle } from '../src/features/document/ScaleKey';
import { ATTENTION_FILL, COLORMAPS, heatColor, mapGradient, type AttentionMode, type Colormap } from '../src/features/document/attentionDisplay';
import type { DisplaySettings } from '../src/api/client';
import { HOT, PAGE, PAGE_ID, TABLE_ID, TEXT_ID, elementMasses, elementRecords, generationPositions, generatorPatchIndex, generatorRow, pageRecord, patchValues } from '../fixtures/attention';

// The colour map recolours Heatmap, Blend, element fills, comparison grids and the key; Overlay stays
// amber whichever map is chosen. Each map's top and middle colours are copied from matplotlib here,
// independently of attentionDisplay.ts, so a map that silently falls back to inferno fails.

afterEach(cleanup);

const MAPS: Colormap[] = ['inferno', 'magma', 'viridis', 'plasma'];
const TOP: Record<Colormap, number[]> = {inferno: [252, 255, 164], magma: [252, 253, 191], viridis: [253, 231, 37], plasma: [240, 249, 33]};
const MIDDLE: Record<Colormap, number[]> = {inferno: [186, 54, 85], magma: [181, 54, 122], viridis: [31, 161, 135], plasma: [219, 92, 104]};
const css = (colour: number[]) => `rgb(${colour.join(',')})`;

const row = generatorRow(patchValues(6));
const positions = generationPositions.filter(p => p.page_id === PAGE_ID);
const HOT_INDEX = generatorPatchIndex(6);

describe('colour map choices', () => {
  it('offers exactly the four maps, each with its own colours', () => {
    expect(COLORMAPS.map(c => c.id)).toEqual(MAPS);
    for (const map of MAPS) {
      expect(heatColor(1, map), map).toEqual(TOP[map]);
      expect(heatColor(.5, map), map).toEqual(MIDDLE[map]);
    }
    // No two maps share a top or middle colour, so a choice is always visible on the page.
    for (const colours of [TOP, MIDDLE]) expect(new Set(MAPS.map(m => css(colours[m]))).size).toBe(MAPS.length);
    expect(new Set(MAPS.map(m => mapGradient(m))).size).toBe(MAPS.length);
  });
});

describe('page patches', () => {
  const layer = (mode: AttentionMode, colormap: Colormap) => <svg><AttentionLayer positions={positions} row={row} maximum={HOT} mode={mode}
    opacity={.65} smooth={false} showPatches={false} colormap={colormap}/></svg>;
  const fills = (container: HTMLElement) => new Map([...container.querySelectorAll('[data-testid="attention-patch"]')].map(r =>
    [Number(r.getAttribute('data-position-index')), r.getAttribute('fill')]));

  for (const mode of ['heatmap', 'blend'] as const) it(`${mode} paints the hot patch in each chosen map and changes when the map changes`, () => {
    const view = render(layer(mode, 'inferno'));
    let before = fills(view.container);
    for (const map of MAPS) {
      view.rerender(layer(mode, map));
      const drawn = fills(view.container);
      expect(view.container.querySelector('[data-testid="attention-layer"]')!.getAttribute('data-colormap')).toBe(map);
      expect(drawn.get(HOT_INDEX), `${mode} · ${map}`).toBe(css(TOP[map]));
      // Every patch follows the map, not only the hot one.
      for (const [index, fill] of drawn) expect(fill).toBe(css(heatColor(row.values[index] / HOT, map)));
      if (map !== 'inferno') expect([...drawn.values()]).not.toEqual([...before.values()]);
      before = drawn;
    }
  });

  it('Overlay stays amber, with the same opacities, whichever map is chosen', () => {
    const view = render(layer('overlay', 'inferno'));
    const opacities = () => [...view.container.querySelectorAll('[data-testid="attention-patch"]')].map(r => r.getAttribute('fill-opacity'));
    const amber = opacities();
    for (const map of MAPS) {
      view.rerender(layer('overlay', map));
      expect(new Set(fills(view.container).values()), map).toEqual(new Set([ATTENTION_FILL]));
      expect(opacities(), map).toEqual(amber);
      expect(view.container.querySelector('[data-testid="attention-layer"]')!.hasAttribute('data-colormap')).toBe(false);
    }
  });
});

describe('element fills', () => {
  const masses = elementMasses(row);
  const maximum = Math.max(...Object.values(masses));
  const regions = (mode: AttentionMode): ElementRegions => ({masses, maximum, labelled: new Set([TABLE_ID, TEXT_ID]), mode, opacity: .65});
  const settings = (colormap: Colormap) => ({zoom: 1, center_x: null, center_y: null, show_boxes: true, show_support: false, show_reading_order: true,
    show_geometry_flags: true, attention_opacity: .65, attention_smooth: false, attention_show_patches: false, attention_colormap: colormap} as DisplaySettings);
  const canvas = (mode: AttentionMode, colormap: Colormap) => <PageCanvas page={pageRecord} elements={elementRecords} settings={settings(colormap)} selected={null}
    source={null} target={null} support={new Set([TABLE_ID])} onView={() => {}} scaleMax={HOT} attentionMode={mode} regions={regions(mode)}/>;
  const textFill = (container: HTMLElement) => container.querySelector(`[data-testid="element-region"][data-element-id="${TEXT_ID}"]`)!.getAttribute('fill');

  for (const mode of ['heatmap', 'blend'] as const) it(`${mode} element fills follow the chosen map`, () => {
    const view = render(canvas(mode, 'inferno'));
    fireEvent.load(view.container.querySelector('image')!);
    for (const map of MAPS) {
      view.rerender(canvas(mode, map));
      // The text element holds the element maximum, so it takes the map's top colour.
      expect(textFill(view.container), `${mode} · ${map}`).toBe(css(TOP[map]));
    }
  });

  it('Overlay element fills stay amber whichever map is chosen', () => {
    const view = render(canvas('overlay', 'inferno'));
    fireEvent.load(view.container.querySelector('image')!);
    for (const map of MAPS) {
      view.rerender(canvas('overlay', map));
      expect(textFill(view.container), map).toBe(ATTENTION_FILL);
    }
  });
});

describe('colour keys', () => {
  it('Heatmap and Blend keys draw the chosen map; the amber key ignores it', () => {
    for (const map of MAPS) {
      expect(String(rampStyle({kind: 'heatmap', opacity: .65, colormap: map}).background)).toBe(mapGradient(map));
      expect(String(rampStyle({kind: 'blend', opacity: .65, colormap: map}).background)).toContain(mapGradient(map));
      expect(rampStyle({kind: 'amber', opacity: .65, colormap: map})).toEqual(rampStyle({kind: 'amber', opacity: .65}));
    }
  });
});

describe('comparison grids', () => {
  const heads = [0, 1].map(head => ({label: `Head ${head}`, row: generatorRow(patchValues([6, 11][head]), {head})}));
  const page = {width: PAGE.width, height: PAGE.height, image_url: '/page.png', page_number: 1};
  const grids = (mode: AttentionMode, colormap: Colormap) => {
    const view = render(<ComparisonMaps comparisons={heads} positions={positions} page={page} scale={HOT} mode={mode}
      settings={{attention_opacity: .65, attention_scale_mode: 'auto', attention_colormap: colormap} as DisplaySettings} onView={() => {}}/>);
    const hot = view.container.querySelector(`figure [data-testid="comparison-patch"][data-position-index="${HOT_INDEX}"]`)!.getAttribute('fill');
    cleanup();
    return hot;
  };

  it('Heatmap grids follow the chosen map and Overlay grids stay amber', () => {
    for (const map of MAPS) {
      expect(grids('heatmap', map), map).toBe(css(TOP[map]));
      expect(grids('overlay', map), map).toBe(ATTENTION_FILL);
    }
  });
});
