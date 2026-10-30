// @vitest-environment happy-dom
import { afterEach, describe, expect, it, vi } from 'vitest';
import { cleanup, fireEvent, render, within } from '@testing-library/react';
import type { Element } from '../src/api/client';
import { ElementProfile, formatProfile, layerMeans, nearTie, profileValue, rankCells, type ElementProfileData, type ProfileStatus } from '../src/features/document/ElementProfile';

// Known-answer profile: 2 layers × 4 heads in 2 key/value groups, two elements. Values are fixture numbers.
afterEach(cleanup);

const profile: ElementProfileData = {
  policy: 'element_layer_head_mass_v1', layer_count: 2, head_count: 4, kv_head_count: 2, deepstack_layers: [1], image_positions: 10,
  image_mass: [[.5, .5, 0, .5], [.5, .4, .5, .5]], significant_digits: 7, unmeasured_element_ids: ['element_gap'],
  denominator: 'original captured domain',
  elements: [
    {element_id: 'element_a', page_id: 'page_a', label: 'table · 1', reading_order: 1, area_share: .1, mass: [[.1, .05, .02, .01], [.2, .1, .02, .199]]},
    {element_id: 'element_b', page_id: 'page_a', label: 'text · 2', reading_order: 2, area_share: .5, mass: [[.3, .01, .01, .01], [.05, .05, .05, .05]]},
  ],
};
const a = profile.elements[0];

function element(id: string, page = 'page_a', flags: string[] = []): Element {
  return {element_id: id, page_id: page, native_id: id, element_type: 'table', geometry_flags: flags} as unknown as Element;
}

function show(status: ProfileStatus, {id = 'element_a', linked = true, layer = 0, head = 0, metric = 'mass' as const, page = 'page_a', flags = [] as string[]} = {}) {
  const onShow = vi.fn(), onElement = vi.fn(), onMetric = vi.fn();
  const view = render(<ElementProfile status={status} element={element(id, page, flags)} suppliedPageIds={['page_a']}
    source={{label: 'Step 0 → “£”', captureId: 'generation_x', step: 0}} layer={layer} head={head} linked={linked}
    metric={metric} onMetric={onMetric} onShow={onShow} onElement={onElement}/>);
  return {view, onShow, onElement, onMetric};
}

const ready: ProfileStatus = {kind: 'ready', profile, analysisId: 'analysis_1', notices: ['Decoder layers 1: image keys and values already include DeepStack vision features.']};

describe('profile values', () => {
  it('derives image share and enrichment per cell, and refuses a zero denominator', () => {
    expect(profileValue(profile, a, 'mass', 0, 0)).toBe(.1);
    expect(profileValue(profile, a, 'image', 0, 0)).toBeCloseTo(.2);
    expect(profileValue(profile, a, 'enrichment', 0, 0)).toBeCloseTo(2);
    expect(profileValue(profile, a, 'image', 0, 2)).toBeNull();
    expect(profileValue(profile, {...a, area_share: null}, 'enrichment', 0, 0)).toBeNull();
  });

  it('takes the metric per cell before the layer mean, and leaves a layer with an unavailable head unavailable', () => {
    expect(layerMeans(profile, a, 'mass')).toEqual([expect.closeTo(.045), expect.closeTo(.12975)]);
    const image = layerMeans(profile, a, 'image');
    expect(image[0]).toBeNull();
    expect(image[1]).toBeCloseTo((.4 + .25 + .04 + .398) / 4);
  });

  it('ranks available cells with ties in layer then head order and flags near-ties', () => {
    const ranked = rankCells(profile, a, 'mass');
    expect(ranked.slice(0, 3).map(c => [c.layer, c.head])).toEqual([[1, 0], [1, 3], [0, 0]]);
    expect(nearTie(.2, .199)).toBe(true);
    expect(nearTie(.2, .1)).toBe(false);
    expect(rankCells(profile, a, 'image').some(c => c.layer === 0 && c.head === 2)).toBe(false);
  });

  it('formats percentages, ratios and unavailable values', () => {
    expect(formatProfile(.0795, 'mass')).toBe('7.95%');
    expect(formatProfile(3.4, 'enrichment')).toBe('3.40×');
    expect(formatProfile(null, 'image')).toBe('Unavailable');
  });
});

describe('Details › Element profile', () => {
  it('draws one cell per layer and head, outlines the scrubber cell and groups key/value heads', () => {
    const {view} = show(ready, {layer: 1, head: 3});
    const cells = view.container.querySelectorAll('[role=gridcell]');
    expect(cells).toHaveLength(8);
    const current = view.container.querySelector('.is-current')!;
    expect([current.getAttribute('data-layer'), current.getAttribute('data-head')]).toEqual(['1', '3']);
    expect(view.container.querySelectorAll('.docattentionlens-profile-cell.is-group-start')).toHaveLength(2);
    expect(view.getByText('1†')).toBeTruthy();
  });

  it('reads the focused cell exactly: mass, image share, enrichment, element rank and the top element', () => {
    const {view, onElement} = show(ready);
    const readout = view.getByTestId('element-profile-readout');
    expect(view.getByTestId('profile-mass').textContent).toBe('10.00% of the row');
    expect(readout.textContent).toContain('20.00% of image attention');
    expect(readout.textContent).toContain('2.00×');
    expect(view.getByTestId('profile-rank').textContent).toContain('2 of 2 mapped elements by mass in this head');
    fireEvent.click(view.getByRole('button', {name: 'text · 2'}));
    expect(onElement).toHaveBeenCalledWith('element_b');
  });

  it('moves a keyboard cursor without changing the page, and Enter or a click sets the scrubber', () => {
    const {view, onShow} = show(ready);
    const grid = view.getByTestId('element-profile-grid');
    fireEvent.keyDown(grid, {key: 'ArrowDown'});
    fireEvent.keyDown(grid, {key: 'ArrowRight'});
    expect(onShow).not.toHaveBeenCalled();
    expect(grid.getAttribute('aria-activedescendant')).toMatch(/-l1-h1$/);
    fireEvent.keyDown(grid, {key: 'Enter'});
    expect(onShow).toHaveBeenCalledWith(1, 1);
    fireEvent.click(view.container.querySelector('[data-layer="0"][data-head="2"]')!);
    expect(onShow).toHaveBeenLastCalledWith(0, 2);
  });

  it('only reads a cell when the page method has its own axes; Show on page is explicit', () => {
    const {view, onShow} = show(ready, {linked: false});
    fireEvent.click(view.container.querySelector('[data-layer="1"][data-head="0"]')!);
    expect(onShow).not.toHaveBeenCalled();
    expect(view.getByText(/chooses its own layers or heads/)).toBeTruthy();
    fireEvent.click(view.getByRole('button', {name: 'Show on page'}));
    expect(onShow).toHaveBeenCalledWith(1, 0);
  });

  it('says the grid is raw attention, which does not decompose a combined page method', () => {
    const linked = show(ready).view;
    expect(linked.container.querySelector('.docattentionlens-raw-tag')!.textContent).toBe('raw attention');
    expect(linked.container.textContent).not.toContain('does not decompose');
    cleanup();
    expect(show(ready, {linked: false}).view.getByText(/does not decompose the combined method shown on the page/)).toBeTruthy();
  });

  it('lists the strongest heads with a near-tie mark and hatches unavailable cells', () => {
    const {view} = show(ready, {metric: 'image' as never});
    expect(view.container.querySelector('[data-layer="0"][data-head="2"]')!.classList.contains('is-unavailable')).toBe(true);
    const mass = show(ready).view;
    const top = mass.getAllByTestId('profile-top-heads').at(-1)!;
    expect(top.textContent).toContain('L1 · H0');
    expect(top.textContent).toContain('≈ next');
  });

  it('states missing, omitted, loading and unavailable profiles in words, never as zero', () => {
    const text = (status: ProfileStatus, options = {}, query = (c: ReturnType<typeof within>) => c.getByRole('status')) =>
      query(within(show(status, options).view.container)).textContent;
    const gap = (c: ReturnType<typeof within>) => c.getByTestId('element-profile-gap');
    expect(text(ready, {id: 'element_gap'}, gap)).toContain('unmeasured, not zero');
    expect(text(ready, {id: 'element_z', page: 'page_b'}, gap)).toContain('not supplied');
    expect(text(ready, {id: 'element_f', flags: ['out_of_bounds']}, gap)).toContain('Unreviewed geometry');
    expect(text({kind: 'loading'})).toContain('Calculating');
    expect(text({kind: 'unavailable', reason: 'Needs a validated document ownership mapping.'})).toContain('ownership');
  });
});
