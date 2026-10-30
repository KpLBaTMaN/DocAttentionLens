// @vitest-environment happy-dom
import { afterEach, describe, expect, it, vi } from 'vitest';
import { cleanup, fireEvent, render, screen } from '@testing-library/react';
import { LayerScrubber, indexAt, keyIndex, type LayerScrubberProps } from '../src/features/document/LayerScrubber';

afterEach(() => {cleanup(); vi.restoreAllMocks();});

/** A 360 px track starting at x = 40: with 36 layers each layer is 10 px wide. */
function sizeTrack() {
  const track = screen.getByTestId('layer-scrubber-track');
  vi.spyOn(track, 'getBoundingClientRect').mockReturnValue({left: 40, width: 360, top: 0, height: 26, right: 400, bottom: 26, x: 40, y: 0, toJSON: () => ({})} as DOMRect);
  return track;
}
const xOf = (layer: number) => 40 + layer * 10 + 5;

function setup(props: Partial<LayerScrubberProps> = {}) {
  const onLayer = vi.fn(), onHead = vi.fn();
  const view = render(<LayerScrubber layer={12} layers={36} onLayer={onLayer} head={3} heads={16} onHead={onHead} {...props}/>);
  return {view, onLayer, onHead};
}

describe('pure helpers', () => {
  it('maps pointer positions to layers, clamped to the track', () => {
    const rect = {left: 40, width: 360};
    expect(indexAt(45, rect, 36)).toBe(0);
    expect(indexAt(xOf(20), rect, 36)).toBe(20);
    expect(indexAt(10, rect, 36)).toBe(0);
    expect(indexAt(1000, rect, 36)).toBe(35);
    expect(indexAt(50, {left: 0, width: 0}, 36)).toBeUndefined();
  });
  it('moves by one, by a page and to the ends', () => {
    expect(keyIndex('ArrowRight', 12, 36)).toBe(13);
    expect(keyIndex('ArrowDown', 0, 36)).toBe(0);
    expect(keyIndex('PageUp', 12, 36)).toBe(16);
    expect(keyIndex('PageDown', 2, 36)).toBe(0);
    expect(keyIndex('Home', 12, 36)).toBe(0);
    expect(keyIndex('End', 12, 36)).toBe(35);
    expect(keyIndex('a', 12, 36)).toBeUndefined();
  });
});

describe('layer track', () => {
  it('is an accessible slider with a compatible numeric select', () => {
    setup();
    const slider = screen.getByRole('slider', {name: 'Attention layer slider'});
    expect(slider.getAttribute('aria-valuenow')).toBe('12');
    expect(slider.getAttribute('aria-valuemax')).toBe('35');
    expect((screen.getByLabelText('Attention layer') as HTMLSelectElement).value).toBe('12');
    expect(screen.getByText('/ 35')).toBeTruthy();
    expect(slider.querySelectorAll('.docattentionlens-scrubber-bar')).toHaveLength(36);
    expect(slider.querySelector('.is-current')).toBe(slider.querySelectorAll('.docattentionlens-scrubber-bar')[12]);
    expect(slider.querySelectorAll('.is-before')).toHaveLength(12);
  });
  it('sets the layer on click and previews a drag until release', () => {
    const {onLayer} = setup();
    const track = sizeTrack();
    fireEvent.pointerDown(track, {clientX: xOf(20), pointerId: 1, button: 0});
    fireEvent.pointerUp(track, {clientX: xOf(20), pointerId: 1});
    expect(onLayer).toHaveBeenLastCalledWith(20);
    onLayer.mockClear();
    fireEvent.pointerDown(track, {clientX: xOf(2), pointerId: 1, button: 0});
    fireEvent.pointerMove(track, {clientX: xOf(30), pointerId: 1});
    expect(onLayer).not.toHaveBeenCalled();
    expect(track.getAttribute('aria-valuenow')).toBe('30');
    fireEvent.pointerUp(track, {clientX: xOf(31), pointerId: 1});
    expect(onLayer).toHaveBeenCalledTimes(1);
    expect(onLayer).toHaveBeenLastCalledWith(31);
  });
  it('commits keys immediately and the select directly', () => {
    const {onLayer} = setup();
    const slider = screen.getByRole('slider', {name: 'Attention layer slider'});
    fireEvent.keyDown(slider, {key: 'ArrowRight'});
    expect(onLayer).toHaveBeenLastCalledWith(13);
    fireEvent.keyDown(slider, {key: 'ArrowLeft'});
    expect(onLayer).toHaveBeenLastCalledWith(11);
    fireEvent.keyDown(slider, {key: 'End'});
    expect(onLayer).toHaveBeenLastCalledWith(35);
    fireEvent.keyDown(slider, {key: 'Home'});
    expect(onLayer).toHaveBeenLastCalledWith(0);
    fireEvent.keyDown(slider, {key: 'PageUp'});
    expect(onLayer).toHaveBeenLastCalledWith(16);
    fireEvent.change(screen.getByLabelText('Attention layer'), {target: {value: '7'}});
    expect(onLayer).toHaveBeenLastCalledWith(7);
  });
  it('draws optional bar heights', () => {
    setup({layers: 3, layer: 0, bars: [0, .5, 1]});
    const bars = [...document.querySelectorAll<HTMLElement>('.docattentionlens-scrubber-bar')];
    expect(bars.map(b => b.style.height)).toEqual(['20%', '60%', '100%']);
  });
});

describe('heads', () => {
  it('selects a head from a cell, the select, or arrow keys', () => {
    const {onHead} = setup();
    const group = screen.getByRole('radiogroup', {name: 'Attention head slider'});
    expect(screen.getAllByRole('radio')).toHaveLength(16);
    expect(screen.getByRole('radio', {name: 'Head 3'}).getAttribute('aria-checked')).toBe('true');
    fireEvent.click(screen.getByRole('radio', {name: 'Head 9'}));
    expect(onHead).toHaveBeenLastCalledWith(9);
    fireEvent.change(screen.getByLabelText('Attention head'), {target: {value: '5'}});
    expect(onHead).toHaveBeenLastCalledWith(5);
    fireEvent.keyDown(group, {key: 'ArrowRight'});
    expect(onHead).toHaveBeenLastCalledWith(4);
    fireEvent.keyDown(screen.getByRole('radio', {name: 'Head 3'}), {key: 'End'});
    expect(onHead).toHaveBeenLastCalledWith(15);
  });
  it('toggles the All cell and leaves it when a head is chosen', () => {
    const onChange = vi.fn();
    const first = setup({allHeads: {active: false, onChange}});
    fireEvent.click(screen.getByRole('button', {name: 'All'}));
    expect(onChange).toHaveBeenLastCalledWith(true);
    first.view.unmount();
    const onAll = vi.fn();
    const {onHead} = setup({allHeads: {active: true, onChange: onAll, label: 'Mean'}});
    expect(screen.getByRole('button', {name: 'Mean'}).getAttribute('aria-pressed')).toBe('true');
    expect(screen.getAllByRole('radio').every(r => r.getAttribute('aria-checked') === 'false')).toBe(true);
    expect((screen.getByLabelText('Attention head') as HTMLSelectElement).value).toBe('all');
    fireEvent.click(screen.getByRole('radio', {name: 'Head 3'}));
    expect(onAll).toHaveBeenLastCalledWith(false);
    expect(onHead).toHaveBeenLastCalledWith(3);
    fireEvent.click(screen.getByRole('button', {name: 'Mean'}));
    expect(onAll).toHaveBeenLastCalledWith(false);
  });
  it('collapses many heads to a select and a slider', () => {
    vi.useFakeTimers();
    try {
      const {onHead} = setup({heads: 40});
      expect(screen.queryAllByRole('radio')).toHaveLength(0);
      const slider = screen.getByLabelText('Attention head slider') as HTMLInputElement;
      expect(slider.type).toBe('range');
      fireEvent.change(slider, {target: {value: '30'}});
      fireEvent.keyUp(slider, {key: 'ArrowRight'});
      expect(onHead).toHaveBeenLastCalledWith(30);
    } finally {vi.useRealTimers();}
  });
});

describe('rollout range', () => {
  const range = (over: Partial<NonNullable<LayerScrubberProps['range']>> = {}) => {
    const onChange = vi.fn();
    setup({range: {from: 10, to: 20, onChange, ...over}, showHead: false});
    return onChange;
  };
  it('shows two handles and the composed interval', () => {
    range();
    const group = screen.getByRole('group', {name: 'Rollout layer interval'});
    expect(group.textContent).toContain('Layers 10–20 · 11 layers composed');
    expect(screen.getByRole('slider', {name: 'Start layer slider'}).getAttribute('aria-valuenow')).toBe('10');
    expect(screen.getByRole('slider', {name: 'End layer slider'}).getAttribute('aria-valuenow')).toBe('20');
    expect(screen.queryByLabelText('Attention layer')).toBeNull();
    expect(document.querySelectorAll('.docattentionlens-scrubber-bar.is-in-range')).toHaveLength(9);
  });
  it('keeps a valid interval when one handle passes the other', () => {
    const onChange = range();
    fireEvent.keyDown(screen.getByRole('slider', {name: 'Start layer slider'}), {key: 'End'});
    expect(onChange).toHaveBeenLastCalledWith(35, 35);
    fireEvent.keyDown(screen.getByRole('slider', {name: 'End layer slider'}), {key: 'Home'});
    expect(onChange).toHaveBeenLastCalledWith(0, 0);
    fireEvent.change(screen.getByLabelText('End layer'), {target: {value: '8'}});
    expect(onChange).toHaveBeenLastCalledWith(8, 8);
    fireEvent.change(screen.getByLabelText('Start layer'), {target: {value: '25'}});
    expect(onChange).toHaveBeenLastCalledWith(25, 25);
  });
  it('drags the nearest handle on the track', () => {
    const onChange = range();
    const track = sizeTrack();
    fireEvent.pointerDown(track, {clientX: xOf(8), pointerId: 1, button: 0});
    fireEvent.pointerUp(track, {clientX: xOf(8), pointerId: 1});
    expect(onChange).toHaveBeenLastCalledWith(8, 20);
    fireEvent.pointerDown(track, {clientX: xOf(18), pointerId: 1, button: 0});
    fireEvent.pointerMove(track, {clientX: xOf(30), pointerId: 1});
    fireEvent.pointerUp(track, {clientX: xOf(30), pointerId: 1});
    expect(onChange).toHaveBeenLastCalledWith(10, 30);
  });
  it('full rollout fixes the start at layer 0 and offers a range', () => {
    const onChooseRange = vi.fn();
    const onChange = range({from: 0, to: 4, fixedStart: true, onChooseRange});
    expect(screen.queryByLabelText('Start layer')).toBeNull();
    expect(screen.queryByRole('slider', {name: 'Start layer slider'})).toBeNull();
    fireEvent.change(screen.getByLabelText('End layer'), {target: {value: '6'}});
    expect(onChange).toHaveBeenLastCalledWith(0, 6);
    fireEvent.click(screen.getByRole('button', {name: 'Choose a layer range'}));
    expect(onChooseRange).toHaveBeenCalled();
  });
  it('reports an invalid saved interval and repairs it only on request', () => {
    const onChange = range({from: 127, to: 20});
    expect(screen.getByRole('alert').textContent).toContain('saved interval 127–20');
    expect(onChange).not.toHaveBeenCalled();
    fireEvent.click(screen.getByRole('button', {name: 'Use layers 20–20'}));
    expect(onChange).toHaveBeenLastCalledWith(20, 20);
  });
});

describe('hidden and disabled parts', () => {
  it('hides the layer or head and the matching hint', () => {
    const first = setup({showLayer: false});
    expect(screen.queryByRole('slider', {name: 'Attention layer slider'})).toBeNull();
    expect(screen.getByLabelText('Attention head')).toBeTruthy();
    expect(document.querySelector('.docattentionlens-scrubber-hint')?.textContent).toBe('{ } head');
    first.view.unmount();
    setup({showHead: false});
    expect(screen.queryByLabelText('Attention head')).toBeNull();
    expect(screen.getByLabelText('Attention layer')).toBeTruthy();
    expect(document.querySelector('.docattentionlens-scrubber-hint')?.textContent).toBe('[ ] layer');
  });
  it('renders nothing when there is nothing to choose', () => {
    const {view} = setup({showLayer: false, showHead: false});
    expect(view.container.innerHTML).toBe('');
  });
  it('shows the reason and ignores input when disabled', () => {
    const {onLayer, onHead} = setup({disabled: 'Choose a saved capture first.'});
    expect(screen.getByRole('status').textContent).toBe('Choose a saved capture first.');
    expect((screen.getByLabelText('Attention layer') as HTMLSelectElement).disabled).toBe(true);
    fireEvent.keyDown(screen.getByRole('slider', {name: 'Attention layer slider'}), {key: 'ArrowRight'});
    const track = sizeTrack();
    fireEvent.pointerDown(track, {clientX: xOf(3), pointerId: 1, button: 0});
    fireEvent.pointerUp(track, {clientX: xOf(3), pointerId: 1});
    fireEvent.keyDown(screen.getByRole('radiogroup', {name: 'Attention head slider'}), {key: 'ArrowRight'});
    expect(onLayer).not.toHaveBeenCalled();
    expect(onHead).not.toHaveBeenCalled();
  });
});
