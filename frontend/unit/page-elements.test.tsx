// @vitest-environment happy-dom
import { afterEach, beforeEach, describe, expect, it } from 'vitest';
import { cleanup, fireEvent, render } from '@testing-library/react';
import { PageElements } from '../src/features/document/PageElements';
import { shareLabel, type SourceReach } from '../src/features/document/ElementSummary';
import { PAGE_ID, PATCH_COUNT, TABLE_ID, TEXT_ID, attentionRow, elementRecords, patchValues, retrieverOwnership, retrieverPositions, retrieverRow } from '../fixtures/attention';

afterEach(cleanup);
beforeEach(() => {try {localStorage.clear();} catch {/* Order preference is optional. */}});

// A causal source at patch 6 reaches patches 0–6: the table (0, 1) fully, the text (6, 7, 10, 11)
// through patch 6 only, and a heading on patches 8 and 9 not at all, so it has no allocation.
const LATE_ID = 'element_fixture_late';
const late = {...elementRecords[0], key: LATE_ID, element_id: LATE_ID, native_id: 3, element_type: 'heading', element_category: 'HEADING', reading_order: 3};
const positions = retrieverPositions.map((p, i) => ({...p, page_id: PAGE_ID,
  ownership: p.index === 8 || p.index === 9 ? {[LATE_ID]: 1} : retrieverOwnership[i]}));
const row = retrieverRow(patchValues(6), {source_index: 6});
const reach: SourceReach = {source: 6, reaches: i => i <= 6, seen: 7, total: PATCH_COUNT, onWholePage: () => {}};
const mass = (id: string) => row.allocations.find(a => a.element_id === id)!.mass;
const share = (id: string) => shareLabel(mass(id), row.total_mass);
const imageShare = (id: string) => shareLabel(mass(id), row.image_mass);

function list(props: Partial<Parameters<typeof PageElements>[0]> = {}) {
  return render(<PageElements elements={[...elementRecords, late]} onSelect={() => {}} row={row} positions={positions} pageId={PAGE_ID} reach={reach} {...props}/>).container;
}
const native = (id: string) => [...elementRecords, late].find(e => e.element_id === id)!.native_id;
const item = (view: HTMLElement, id: string) => view.querySelector(`[data-native-id="${native(id)}"]`)!;
const order = (view: HTMLElement) => [...view.querySelectorAll('[data-testid="outline-element"]')].map(b => [...elementRecords, late].find(e => String(e.native_id) === b.getAttribute('data-native-id'))!.element_id);

describe('page rail element values', () => {
  it('gives each reached element its share, keeps unreached ones in place, and marks partial coverage', () => {
    const view = list();
    expect(order(view)).toEqual([TABLE_ID, TEXT_ID, LATE_ID]);
    expect(item(view, TABLE_ID).querySelector('[data-testid="element-value"]')!.textContent).toBe(imageShare(TABLE_ID));
    expect(item(view, TEXT_ID).textContent).toContain('partial 1/4');
    expect(item(view, TABLE_ID).textContent).not.toContain('partial');
    expect(item(view, LATE_ID).querySelector('[data-testid="element-value"]')!.textContent).toBe('not reached');
    expect(item(view, LATE_ID).classList.contains('is-unreached')).toBe(true);
    expect(item(view, LATE_ID).textContent).not.toContain('partial');
    expect(view.querySelector('[data-testid="rail-reach"]')!.textContent).toContain('Source 6 sees 7 of 12 patches');
    expect(view.querySelector('[data-testid="outside-elements"]')!.textContent).toContain('background');
  });

  it('shows the share of image attention by default and the share of the whole row on request, remembering the choice', () => {
    const view = list();
    const value = () => item(view, TABLE_ID).querySelector('[data-testid="element-value"]')!;
    expect(row.image_mass).toBeLessThan(row.total_mass);
    expect(value().getAttribute('title')).toContain(`${imageShare(TABLE_ID)} of the row's image attention`);
    expect(value().getAttribute('title')).toContain(`${share(TABLE_ID)} of the row`);
    expect(view.textContent).toContain("share of the row's image attention");
    fireEvent.click([...view.querySelectorAll('button')].find(b => b.textContent === 'Row share')!);
    expect(value().textContent).toBe(share(TABLE_ID));
    expect(view.textContent).toContain('share of the shown row');
    cleanup();
    expect(item(list(), TABLE_ID).querySelector('[data-testid="element-value"]')!.textContent).toBe(share(TABLE_ID));
  });

  it('orders by mass on request, with unreached elements last', () => {
    const view = list();
    fireEvent.click([...view.querySelectorAll('button')].find(b => b.textContent === 'Mass')!);
    const masses = new Map(row.allocations.filter(a => a.element_id).map(a => [a.element_id!, a.mass]));
    expect(order(view)).toEqual([...[TABLE_ID, TEXT_ID].sort((a, b) => masses.get(b)! - masses.get(a)!), LATE_ID]);
  });

  it('opens the full table from the list', () => {
    let opened = 0;
    const view = list({onDetails: () => opened++});
    fireEvent.click([...view.querySelectorAll('button')].find(b => b.textContent === 'Patch contributions and full table ↓')!);
    expect(opened).toBe(1);
  });

  it('divides the image share by the share of image positions, with the ratio and ceiling on the selected element', () => {
    const view = list({selectedId: TABLE_ID});
    const button = () => [...view.querySelectorAll('button')].find(b => b.textContent === 'Enrichment')!;
    fireEvent.click(button());
    expect(button().getAttribute('aria-pressed')).toBe('true');
    // Source 6 reaches patches 0–6: the table owns two of the seven measured image positions.
    const expected = (mass(TABLE_ID) / row.image_mass) / (2 / 7);
    expect(item(view, TABLE_ID).querySelector('[data-testid="element-value"]')!.textContent).toBe(`${expected.toFixed(2)}×`);
    expect(item(view, TABLE_ID).querySelector('[data-testid="enrichment-bar"]')).not.toBeNull();
    expect(item(view, TABLE_ID).querySelector('[data-testid="enrichment-ratio"]')!.textContent)
      .toContain(`${imageShare(TABLE_ID)} of image attention ÷ ${shareLabel(2, 7)} of image positions · at most ${(7 / 2).toFixed(2)}×`);
    expect(item(view, TEXT_ID).querySelector('[data-testid="enrichment-ratio"]')).toBeNull();
    expect(item(view, LATE_ID).querySelector('[data-testid="element-value"]')!.textContent).toBe('not reached');
    expect(view.textContent).toContain('1× is in proportion to area');
    expect(view.querySelector('[data-testid="outside-elements"]')!.textContent).toContain('prompt and special tokens: not image positions');
    cleanup();
    expect(item(list(), TABLE_ID).querySelector('[data-testid="element-value"]')!.textContent).toBe(`${expected.toFixed(2)}×`);
  });

  it('marks an element that owns under one image position', () => {
    const tiny = {...elementRecords[0], key: 'element_fixture_tiny', element_id: 'element_fixture_tiny', native_id: 4, element_type: 'page number', element_category: 'TEXT', reading_order: 4};
    const split = positions.map(p => p.index === 4 ? {...p, ownership: {[tiny.element_id]: .25, background: .75}} : p);
    const splitRow = attentionRow({positions: split, patchIndex: k => k, image: patchValues(6), targetCount: split.length, capture_id: 'retrieval_fixture', source_index: 6});
    const view = render(<PageElements elements={[...elementRecords, tiny]} onSelect={() => {}} row={splitRow} positions={split} pageId={PAGE_ID}
      reach={reach} selectedId={tiny.element_id}/>).container;
    fireEvent.click([...view.querySelectorAll('button')].find(b => b.textContent === 'Enrichment')!);
    const node = view.querySelector('[data-native-id="4"]')!;
    expect(node.querySelector('[data-testid="fragment-tag"]')!.textContent).toBe('under 1 position');
    expect(node.classList.contains('is-fragment')).toBe(true);
    expect(node.querySelector('[data-testid="enrichment-ratio"]')!.textContent).toContain('covers 0.25 of one position');
    expect(item(view, TABLE_ID).querySelector('[data-testid="fragment-tag"]')).toBeNull();
  });

  it('without element values it is the plain reading-order list', () => {
    const view = list({row: undefined});
    expect(view.querySelectorAll('[data-testid="element-value"]')).toHaveLength(0);
    expect(view.querySelector('[role="group"][aria-label="Element order"]')).toBeNull();
    expect(view.querySelector('[data-testid="rail-reach"]')).toBeNull();
    expect(order(view)).toEqual([TABLE_ID, TEXT_ID, LATE_ID]);
  });
});
