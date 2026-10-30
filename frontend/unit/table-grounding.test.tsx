// @vitest-environment happy-dom
import { render } from '@testing-library/react';
import { describe, expect, it } from 'vitest';
import { TableOverlay } from '../src/features/tables/TableOverlay';
import { cellText, groundingRoles, tableResolution, type Table, type TableGrounding } from '../src/features/tables/tableGeometry';

const box = (x1: number, y1: number, x2: number, y2: number) => ({x1, y1, x2, y2});
/** A 2 × 2 table at (100, 100)–(300, 200) on a 560 × 560 page: 20 × 20 merged positions of 28 px at 512. */
function table(): Table {
  const cell = (id: string, r: number, c: number, text: string | null, pdf: string | null) => ({native_id: id, row_ids: [`row-000${r + 1}`], column_ids: [`column-000${c + 1}`],
    row_indices: [r], column_indices: [c], bbox: box(100 + 100 * c, 100 + 50 * r, 200 + 100 * c, 150 + 50 * r), crop_bbox: box(0, 0, 1, 1), pdf_bboxes: [],
    text, text_method: 'assistant_visual_read', text_status: 'unreviewed', pdf_text: pdf, roles: [], native_flags: []});
  return {key: 't', table_id: 't', native_id: 'pt1m-table', document_id: 'd', page_id: 'p', ref: {dataset_id: 'd', dataset_version: 'v', domain: 'pubtables_1m', document_id: 'doc'},
    title: 'Fixture', revision: 'v0001', table_sha256: 's', grounding_revision: 'v0001', grounding_sha256: 'g', geometry_frame: 'page_image_pixels',
    bbox: box(100, 100, 300, 200), wrap_bbox: box(90, 80, 300, 210), crop_width: 1, crop_height: 1, crop_sha256: 'c', crop_to_page: [], registration_correlation: .9,
    geometry_verification: 'fixture', row_count: 2, column_count: 2,
    rows: [{native_id: 'row-0001', index: 0, bbox: box(100, 100, 300, 150), crop_bbox: box(0, 0, 1, 1)}, {native_id: 'row-0002', index: 1, bbox: box(100, 150, 300, 200), crop_bbox: box(0, 0, 1, 1)}],
    columns: [], cells: [cell('cell-1', 0, 0, 'Group', 'Group'), cell('cell-2', 0, 1, null, '12.5'), cell('cell-3', 1, 0, null, null), cell('cell-4', 1, 1, '7', '7')],
    quality_status: 'needs_review', publishable: false, quality_issues: [], source: {}};
}
const grounding: TableGrounding = {table_id: 't', native_table_id: 'pt1m-table', table_revision: 'v0001', table_sha256: 's', answer_cell_ids: ['cell-4'],
  answer_row_ids: ['row-0002'], answer_column_ids: ['column-0002'], row_header_path: ['cell-3'], column_header_path: ['cell-2'], support_cell_ids: ['cell-1'],
  accepted_answers: ['7'], value_type: 'integer', unit: null, rationale: 'fixture', binding_status: 'text_matched_not_human_approved', pdf_answer_text: '7',
  authored_input: 'table_crop', review_status: 'unreviewed', support_role: 'evaluation_annotation'};

describe('table geometry', () => {
  it('estimates the image grid and which cells have a position of their own', () => {
    const coarse = tableResolution({width: 560, height: 560}, table(), 28, 512);
    expect([coarse.rows, coarse.columns, coarse.positions]).toEqual([20, 20, 400]);
    // Columns 100–300 cover positions 3–10 (x 84–308); rows 100–200 cover 3–7 (y 84–224).
    expect(coarse.onTable).toBe(8 * 5);
    expect(coarse.isolable).toBe(4);
    expect(coarse.rowsPerPosition).toBeCloseTo(28 / 50);
    // A page twice the size keeps 512 positions, so each covers more of the table.
    const large = tableResolution({width: 1120, height: 1120}, {...table(), bbox: box(200, 200, 600, 400),
      rows: table().rows.map(r => ({...r, bbox: box(200, 200 + 100 * r.index, 600, 300 + 100 * r.index)})),
      cells: table().cells.map(c => ({...c, bbox: box(c.bbox.x1 * 2, c.bbox.y1 * 2, c.bbox.x2 * 2, c.bbox.y2 * 2)}))}, 28, 512);
    expect(large.positions).toBeLessThanOrEqual(512);
  });

  it('uses PDF text only where the annotation has no transcription, and says so', () => {
    const [header, number, empty, answer] = table().cells;
    expect(cellText(header)).toEqual({text: 'Group', source: 'transcribed'});
    expect(cellText(number)).toEqual({text: '12.5', source: 'pdf'});
    expect(cellText(empty)).toEqual({text: '', source: 'none'});
    expect(cellText(answer).source).toBe('transcribed');
  });

  it('numbers the lookup path and puts the answer last', () => {
    const roles = groundingRoles(grounding);
    expect(roles.get('cell-4')).toEqual({kind: 'answer', label: 'Answer'});
    expect(roles.get('cell-3')).toEqual({kind: 'row_path', label: 'R1'});
    expect(roles.get('cell-2')).toEqual({kind: 'column_path', label: 'C1'});
    expect(roles.get('cell-1')).toEqual({kind: 'support', label: 'S'});
    expect(groundingRoles(undefined).size).toBe(0);
  });
});

describe('TableOverlay', () => {
  const draw = (showGrid: boolean, showGrounding: boolean) => render(<svg><TableOverlay tables={[table()]} grounding={grounding} showGrid={showGrid} showGrounding={showGrounding} scale={1}/></svg>).container;

  it('draws the table, its caption area and cell grid at page coordinates', () => {
    const view = draw(true, false);
    expect(view.querySelector('.docattentionlens-table-outline')?.getAttribute('width')).toBe('200');
    expect(view.querySelector('.docattentionlens-table-wrap')?.getAttribute('y')).toBe('80');
    expect(view.querySelectorAll('.docattentionlens-table-cell')).toHaveLength(4);
    expect(view.querySelector('[data-cell-id="cell-2"] title')?.textContent).toContain('12.5 (PDF text)');
    expect(view.querySelectorAll('[data-testid="grounding-cell"]')).toHaveLength(0);
  });

  it('shows the evaluation-only grounding only when asked, with the answer drawn last', () => {
    const view = draw(false, true);
    expect(view.querySelectorAll('.docattentionlens-table-cell')).toHaveLength(0);
    const roles = [...view.querySelectorAll('[data-testid="grounding-cell"]')].map(e => e.getAttribute('data-role'));
    expect(roles).toEqual(['support', 'column_path', 'row_path', 'answer']);
    expect(view.querySelector('.docattentionlens-grounding-label.is-row_path')?.textContent).toBe('R1');
  });
});
