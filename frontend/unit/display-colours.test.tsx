// @vitest-environment happy-dom
import { afterEach, describe, expect, it } from 'vitest';
import { cleanup, render } from '@testing-library/react';
import { AttentionLayer } from '../src/features/document/AttentionLayer';
import { attentionAlpha, heatColor, mapGradient, rangeFloor, scalePosition } from '../src/features/document/attentionDisplay';
import { HOT, PAGE_ID, generationPositions, generatorPatchIndex, generatorRow, patchValues } from '../fixtures/attention';

afterEach(cleanup);

describe('colour maps', () => {
  it('keeps the exact inferno stops and adds the other maps at their ends', () => {
    expect(heatColor(0)).toEqual([0, 0, 4]);
    expect(heatColor(.5)).toEqual([186, 54, 85]);
    expect(heatColor(1)).toEqual([252, 255, 164]);
    expect(heatColor(0, 'viridis')).toEqual([68, 1, 84]);
    expect(heatColor(1, 'viridis')).toEqual([253, 231, 37]);
    expect(heatColor(1, 'plasma')).toEqual([240, 249, 33]);
    expect(heatColor(1, 'magma')).toEqual([252, 253, 191]);
    expect(mapGradient('inferno')).toContain('rgb(186,54,85) 50%');
  });

  it('starts the scale at zero by default, or at the shown minimum when stretched', () => {
    expect(rangeFloor([.2, .05, .4], 'zero')).toBe(0);
    expect(rangeFloor([.2, .05, .4], 'minimum')).toBe(.05);
    expect(rangeFloor([], 'minimum')).toBe(0);
    expect(scalePosition(.05, .4, .05)).toBe(0);
    expect(scalePosition(.4, .4, .05)).toBe(1);
    expect(attentionAlpha(.225, .4, .5, .05)).toBeCloseTo(.25, 12);
  });
});

describe('Blend', () => {
  const row = generatorRow(patchValues(6));
  const positions = generationPositions.filter(p => p.page_id === PAGE_ID);
  const patches = (props: Partial<Parameters<typeof AttentionLayer>[0]>) => {
    const view = render(<svg><AttentionLayer positions={positions} row={row} maximum={HOT} mode="blend" opacity={.45} smooth={false} showPatches={false} {...props}/></svg>);
    return new Map([...view.container.querySelectorAll('[data-testid="attention-patch"]')].map(r =>
      [Number(r.getAttribute('data-position-index')), {fill: r.getAttribute('fill'), opacity: Number(r.getAttribute('fill-opacity'))}]));
  };

  it('colours every patch with the map at one fixed opacity', () => {
    const drawn = patches({colormap: 'viridis'});
    const hot = drawn.get(generatorPatchIndex(6))!, weak = drawn.get(generatorPatchIndex(0))!;
    expect(hot).toEqual({fill: `rgb(${heatColor(1, 'viridis').join(',')})`, opacity: .45});
    expect(weak.opacity).toBe(.45);
    expect(weak.fill).toBe(`rgb(${heatColor(row.values[generatorPatchIndex(0)] / HOT, 'viridis').join(',')})`);
  });

  it('stretched from the page minimum, the weakest patch takes the darkest colour', () => {
    const values = positions.filter(p => p.bbox).map(p => row.values[p.index]);
    const weakest = positions.find(p => p.bbox && row.values[p.index] === Math.min(...values))!;
    expect(patches({floor: 'minimum'}).get(weakest.index)!.fill).toBe(`rgb(${heatColor(0).join(',')})`);
    expect(patches({floor: 'zero'}).get(weakest.index)!.fill).not.toBe(`rgb(${heatColor(0).join(',')})`);
  });
});
