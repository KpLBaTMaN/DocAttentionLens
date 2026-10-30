// @vitest-environment happy-dom
import { afterEach, describe, expect, it, vi } from 'vitest';
import { cleanup, fireEvent, render } from '@testing-library/react';
import { MarkingCanvas, type Tool } from '../src/features/marking/MarkingCanvas';
import type { Mark, MarkingPage } from '../src/features/marking/markingModel';

// The canvas starts at 600 × 680. A 400 × 300 page fits at scale 1.5, centred vertically: offset (0, 115),
// so display (x, y) is page (x / 1.5, (y − 115) / 1.5).
afterEach(cleanup);
const PAGE:MarkingPage = {page_id:'p1', page_number:1, width:400, height:300, image_url:'/p1.png', text_layer:false};
const MARKS:Mark[] = [
  {key:'big', page_id:'p1', element_type:'table', bbox:{x1:20, y1:20, x2:220, y2:120}, reading_order:1, text:null, text_source:'none'},
  {key:'small', page_id:'p1', element_type:'caption', bbox:{x1:40, y1:40, x2:80, y2:60}, reading_order:2, text:null, text_source:'none'},
];
const at = (x:number, y:number) => ({clientX:x * 1.5, clientY:y * 1.5 + 115, pointerId:1, button:0});

function canvas(tool:Tool, selected:string|null = null) {
  const calls = {select:vi.fn(), draw:vi.fn(), geometry:vi.fn(), order:vi.fn()};
  const view = render(<MarkingCanvas page={PAGE} marks={MARKS} selected={selected} tool={tool} hideLabels={false}
    onSelect={calls.select} onDraw={calls.draw} onGeometry={calls.geometry} onOrderClick={calls.order}/>);
  const svg = view.container.querySelector('svg')!;
  const drag = (from:[number, number], to:[number, number], extra = {}) => {
    fireEvent.pointerDown(svg, at(...from));
    fireEvent.pointerMove(svg, at(...to));
    fireEvent.pointerUp(svg, {...at(...to), ...extra});
  };
  const click = (x:number, y:number) => {fireEvent.pointerDown(svg, at(x, y)); fireEvent.pointerUp(svg, at(x, y));};
  return {view, svg, calls, drag, click};
}

describe('marking canvas', () => {
  it('draws boxes in page pixels and reports Alt as keep-as-drawn', () => {
    const {calls, drag} = canvas('box');
    drag([250, 150], [300, 200]);
    drag([300, 200], [250, 150], {altKey:true});
    expect(calls.draw.mock.calls).toEqual([[{x1:250, y1:150, x2:300, y2:200}, false], [{x1:250, y1:150, x2:300, y2:200}, true]]);
  });

  it('clamps a box to the page and ignores a click without a drag', () => {
    const {calls, drag, click} = canvas('box');
    drag([390, 290], [450, 400]);
    expect(calls.draw.mock.calls[0][0]).toEqual({x1:390, y1:290, x2:400, y2:300});
    click(50, 50);
    expect(calls.draw).toHaveBeenCalledTimes(1);
    expect(calls.select).toHaveBeenLastCalledWith('small');
  });

  it('selects the smallest box under the pointer and moves it within the page', () => {
    const {calls, drag, click} = canvas('select');
    click(50, 50);
    expect(calls.select).toHaveBeenLastCalledWith('small');
    click(300, 250);
    expect(calls.select).toHaveBeenLastCalledWith(null);
    drag([100, 100], [140, 90]);
    expect(calls.geometry).toHaveBeenCalledWith('big', {x1:60, y1:10, x2:260, y2:110});
  });

  it('resizes the selected box from a handle', () => {
    const {view, calls, drag} = canvas('select', 'big');
    expect(view.container.querySelectorAll('[data-handle]')).toHaveLength(8);
    drag([220, 120], [250, 140]);
    expect(calls.geometry).toHaveBeenCalledWith('big', {x1:20, y1:20, x2:250, y2:140});
  });

  it('numbers boxes in the order they are clicked', () => {
    const {calls, click} = canvas('order');
    click(150, 100);
    click(60, 50);
    expect(calls.order.mock.calls).toEqual([['big'], ['small']]);
  });

  it('labels boxes beside them and cycles with Tab', () => {
    const {view, calls} = canvas('select', 'big');
    expect([...view.container.querySelectorAll('.docattentionlens-mark-chip text')].map(t => t.textContent)).toEqual(['1 table', '2 caption']);
    fireEvent.keyDown(view.container.querySelector('.docattentionlens-marking-canvas')!, {key:'Tab'});
    expect(calls.select).toHaveBeenLastCalledWith('small');
  });
});
