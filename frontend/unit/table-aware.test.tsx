// @vitest-environment happy-dom
import { cleanup, fireEvent, render } from '@testing-library/react';
import { afterEach, describe, expect, it, vi } from 'vitest';
import { TableHeatmap, type TableAllocation } from '../src/features/tables/TableHeatmap';
import { TableComparisons, type TableRelated } from '../src/features/tables/TableComparisons';
import type { Table, TableGrounding } from '../src/features/tables/tableGeometry';

afterEach(cleanup);

const box = (x1: number, y1: number, x2: number, y2: number) => ({x1, y1, x2, y2});
/** A 2 × 2 table whose first row is one merged header cell spanning both columns. */
function table(): Table {
  const cell = (id: string, rows: number[], columns: number[], text: string) => ({native_id: id, row_ids: [], column_ids: [], row_indices: rows, column_indices: columns,
    bbox: box(columns[0] * 10, rows[0] * 10, (columns.at(-1)! + 1) * 10, (rows.at(-1)! + 1) * 10), crop_bbox: box(0, 0, 1, 1), pdf_bboxes: [], text,
    text_method: 'fixture', text_status: 'unreviewed', pdf_text: null, roles: [{role: rows[0] === 0 ? 'column_header' : 'data', status: 'proposed', method: 'fixture'}], native_flags: []});
  return {key: 't', table_id: 't', native_id: 'pt', document_id: 'd', page_id: 'p', ref: {dataset_id: 'd', dataset_version: 'v', domain: 'pubtables_1m', document_id: 'doc'},
    title: 'Fixture', revision: 'v1', table_sha256: 's', grounding_revision: 'v1', grounding_sha256: 'g', geometry_frame: 'page_image_pixels',
    bbox: box(0, 0, 20, 20), wrap_bbox: box(0, 0, 20, 24), crop_width: 1, crop_height: 1, crop_sha256: 'c', crop_to_page: [], registration_correlation: .9,
    geometry_verification: 'fixture', row_count: 2, column_count: 2, rows: [], columns: [],
    cells: [cell('head', [0], [0, 1], 'Group'), cell('a', [1], [0], '4'), cell('b', [1], [1], '1')],
    quality_status: 'needs_review', publishable: false, quality_issues: [], source: {}};
}
const band = (id: string, index: number, mass: number) => ({band_id: id, index, mass, area_share: .25, enrichment: mass * 4});
const allocation: TableAllocation = {table_id: 't', native_table_id: 'pt', policy: 'table_cell_partition_v1', selection: 'step 0 · layer 3 · head 1',
  denominator: 'original captured domain', total_mass: 1, image_mass: .8, table_mass: .6,
  cells: [
    {cell_id: 'head', row_indices: [0], column_indices: [0, 1], mass: .1, area_share: .5, enrichment: .25, isolable: true, touching_positions: 2, largest_position_share: 1},
    {cell_id: 'a', row_indices: [1], column_indices: [0], mass: null, area_share: .25, enrichment: null, isolable: false, touching_positions: 0, largest_position_share: 0},
    {cell_id: 'b', row_indices: [1], column_indices: [1], mass: .5, area_share: .25, enrichment: 2.5, isolable: false, touching_positions: 2, largest_position_share: .4}],
  rows: [band('r0', 0, .1), band('r1', 1, .5)], columns: [band('c0', 0, .05), band('c1', 1, .55)],
  regions: [{region: 'table', label: 'Table cells', mass: .6, area_share: .5, enrichment: 1.5}, {region: 'non_image:query', label: 'Question tokens', mass: .2, area_share: null, enrichment: null}],
  resolution: {image_positions: 4, positions_on_table: 4, cell_count: 3, isolable_cells: 1, rows_per_position: 1, columns_per_position: 1}};
const grounding = {table_id: 't', answer_cell_ids: ['b'], answer_row_ids: [], answer_column_ids: [], row_header_path: [], column_header_path: ['head'],
  support_cell_ids: []} as unknown as TableGrounding;

describe('table heatmap', () => {
  it('keeps merged spans, marks shared and unmeasured cells and outlines the lookup path on request', () => {
    const onSelect = vi.fn();
    const view = render(<TableHeatmap table={table()} allocation={allocation} grounding={grounding} showGrounding onSelect={onSelect}/>);
    const head = view.container.querySelector('[data-cell-id="head"]') as HTMLElement;
    expect(head.style.gridColumn).toBe('1 / 3');
    expect(view.container.querySelector('[data-cell-id="a"]')!.className).toContain('is-unmeasured');
    const answer = view.container.querySelector('[data-cell-id="b"]')!;
    expect(answer.className).toContain('is-shared');
    expect(answer.className).toContain('is-answer');
    expect(head.className).toContain('is-column_path');
    expect(view.getByText(/1 of 3 isolable/)).toBeTruthy();
    fireEvent.click(answer);
    expect(onSelect).toHaveBeenCalledWith('b');
  });

  it('shows the selected cell and hides grounding unless asked', () => {
    const view = render(<TableHeatmap table={table()} allocation={allocation} grounding={grounding} showGrounding={false} selected="b" onSelect={() => {}}/>);
    expect(view.container.querySelector('[data-cell-id="b"]')!.className).not.toContain('is-answer');
    expect(view.getByTestId('table-cell-detail').textContent).toContain('shared with neighbours');
    fireEvent.click(view.getByRole('radio', {name: 'Enrichment'}));
    expect(view.getByText(/enrichment maximum/)).toBeTruthy();
  });
});

const related: TableRelated = {table_id: 't', baseline_capture_ids: ['g2', 'g3'], matched_capture_ids: ['g2'], items: [
  {capture_id: 'g2', case_id: 'c2', question_id: 'q2', query: 'Same row question', answer: '1', created_at: '', answer_row: 1, answer_column: 0, relation: 'same_row', binding_status: 'text_matched_not_human_approved', comparable: true},
  {capture_id: 'g3', case_id: 'c3', question_id: 'q3', query: 'Other question', answer: '2', created_at: '', answer_row: 0, answer_column: 0, relation: 'other', binding_status: 'text_matched_not_human_approved', comparable: true},
  {capture_id: 'g4', case_id: 'c4', question_id: 'q4', query: 'Other model', answer: '2', created_at: '', answer_row: 0, answer_column: 0, relation: 'other', binding_status: 'text_matched_not_human_approved', comparable: false, reason: 'Different model or revision'}]};

describe('table comparisons', () => {
  it('offers the question-swap baseline only for lookup-path allocation', () => {
    const onBaseline = vi.fn();
    const view = render(<TableComparisons methodId="table.lookup_path" related={related} useBaseline onBaseline={onBaseline} onPartner={() => {}}/>);
    fireEvent.click(view.getByRole('checkbox'));
    expect(onBaseline).toHaveBeenCalledWith(false);
    expect(view.getByText(/2 saved captures/)).toBeTruthy();
    expect(view.getByText(/1 other capture is not comparable/)).toBeTruthy();
    expect(render(<TableComparisons methodId="table.cell_allocation" related={related} useBaseline onBaseline={() => {}} onPartner={() => {}}/>).container.textContent).toBe('');
  });

  it('lists matched questions for the contrast and explains when there are none', () => {
    const view = render(<TableComparisons methodId="table.matched_contrast" related={related} useBaseline onBaseline={() => {}} onPartner={() => {}}/>);
    expect((view.getByLabelText('Matched question capture') as HTMLSelectElement).value).toBe('g2');
    const none = render(<TableComparisons methodId="table.matched_contrast" related={{...related, matched_capture_ids: []}} useBaseline onBaseline={() => {}} onPartner={() => {}}/>);
    expect(none.container.textContent).toContain('No saved capture asks a question');
  });
});
