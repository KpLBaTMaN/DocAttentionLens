// @vitest-environment happy-dom
import { afterEach, describe, expect, it } from 'vitest';
import { cleanup, fireEvent, render } from '@testing-library/react';
import { ElementSummary, type SourceReach } from '../src/features/document/ElementSummary';
import { PAGE_ID, PATCH_COUNT, TABLE_ID, TEXT_ID, elementRecords, patchValues, retrieverOwnership, retrieverPositions, retrieverRow } from '../fixtures/attention';

afterEach(cleanup);

// A causal source at patch 6 (row 1, column 2) reaches patches 0–6. The table (patches 0, 1) is
// fully reached, the text (6, 7, 10, 11) only through patch 6, and a heading on patches 8 and 9
// lies entirely after the source, so it has no allocation.
const LATE_ID = 'element_fixture_late';
const late = {...elementRecords[0], key: LATE_ID, element_id: LATE_ID, native_id: 3, element_type: 'heading', element_category: 'HEADING'};
const positions = retrieverPositions.map((p, i) => ({...p, page_id: PAGE_ID,
  ownership: p.index === 8 || p.index === 9 ? {[LATE_ID]: 1} : retrieverOwnership[i]}));
const row = retrieverRow(patchValues(6), {source_index: 6});
const reach = (onWholePage?: () => void): SourceReach => ({source: 6, reaches: i => i <= 6, seen: 7, total: PATCH_COUNT, onWholePage});

function summary(sourceReach?: SourceReach) {
  return render(<ElementSummary row={row} positions={positions} elements={[...elementRecords, late]} pageId={PAGE_ID} pageSupplied
    onSelect={() => {}} onPage={() => {}} pageLabel={() => 'page 1'} documentAware reach={sourceReach}/>).container;
}
const cells = (view: HTMLElement, id: string) => [...view.querySelectorAll(`[data-element-id="${id}"] td`)].map(c => c.textContent);

describe('element summary for a causal source', () => {
  it('states how much of the page the source reaches, with a way to the whole page', () => {
    let used = 0;
    const view = summary(reach(() => used++));
    expect(view.querySelector('[data-testid="source-reach"]')!.textContent).toContain('Source 6 sees 7 of 12 page patches');
    fireEvent.click([...view.querySelectorAll('button')].find(b => b.textContent === 'Use the last token (whole page)')!);
    expect(used).toBe(1);
  });

  it('marks an element measured through only some of its patches', () => {
    const view = summary(reach());
    expect(cells(view, TEXT_ID)[0]).toContain('partial');
    expect(cells(view, TEXT_ID)[3]).toBe('1 of 4');
    expect(cells(view, TABLE_ID)[0]).not.toContain('partial');
    expect(cells(view, TABLE_ID)[3]).toBe('2');
  });

  it('lists unreachable elements once, behind one summary line', () => {
    const view = summary(reach());
    const listed = view.querySelector('[data-testid="out-of-reach"]')!;
    expect(listed.querySelector('summary')!.textContent).toBe('1 element lies after source 6 and cannot receive its attention: not measured, not zero');
    expect([...listed.querySelectorAll('li')].map(li => li.textContent)).toEqual(['heading · element 3']);
    expect(view.textContent).not.toContain('outside this row');
  });

  it('without a causal reach, nothing is called partial and every owned patch counts', () => {
    const view = summary();
    expect(view.querySelector('[data-testid="source-reach"]')).toBeNull();
    expect(view.querySelectorAll('[data-testid="partial-tag"]')).toHaveLength(0);
    expect(cells(view, TEXT_ID)[3]).toBe('4');
    expect(view.querySelector('[data-testid="out-of-reach"] summary')!.textContent).toContain('after the source');
  });
});
