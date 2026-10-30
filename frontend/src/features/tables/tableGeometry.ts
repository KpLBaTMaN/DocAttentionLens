import type { components } from '../../api/generated/schema';
import { pageImageGrid } from '../attention-tool/fullAttentionEstimate';

export type Table = components['schemas']['TableRecord'];
export type TableCell = components['schemas']['TableCellRecord'];
export type TableGrounding = components['schemas']['TableGrounding'];
type Box = {x1: number; y1: number; x2: number; y2: number};

/** Datasets without a structure kind predate table grounding and annotate layout elements. */
export const isTableGrounding = (kind?: string | null) => kind === 'table_grounding';

/** How finely a planned input resolves a table: an estimate before the run. */
export type TableResolution = {rows: number; columns: number; positions: number; onTable: number; isolable: number; cells: number; rowsPerPosition: number};

const overlap = (a: Box, b: Box) => Math.max(0, Math.min(a.x2, b.x2) - Math.max(a.x1, b.x1)) * Math.max(0, Math.min(a.y2, b.y2) - Math.max(a.y1, b.y1));

export function tableResolution(page: {width: number; height: number}, table: Pick<Table, 'bbox' | 'cells' | 'rows'>, cell: number, maxPositions: number): TableResolution {
  const {rows, columns} = pageImageGrid(page.width, page.height, cell, maxPositions);
  const width = page.width / columns, height = page.height / rows, area = width * height;
  const span = (box: Box) => ({c1: Math.max(0, Math.floor(box.x1 / width)), c2: Math.min(columns - 1, Math.ceil(box.x2 / width) - 1),
    r1: Math.max(0, Math.floor(box.y1 / height)), r2: Math.min(rows - 1, Math.ceil(box.y2 / height) - 1)});
  const position = (r: number, c: number): Box => ({x1: c * width, y1: r * height, x2: (c + 1) * width, y2: (r + 1) * height});
  const t = span(table.bbox);
  let onTable = 0;
  for (let r = t.r1; r <= t.r2; r++) for (let c = t.c1; c <= t.c2; c++) if (overlap(position(r, c), table.bbox) > 0) onTable++;
  const isolable = table.cells.filter(cellRecord => {
    const s = span(cellRecord.bbox);
    for (let r = s.r1; r <= s.r2; r++) for (let c = s.c1; c <= s.c2; c++) if (overlap(position(r, c), cellRecord.bbox) >= .5 * area) return true;
    return false;
  }).length;
  const heights = table.rows.map(r => r.bbox.y2 - r.bbox.y1).sort((a, b) => a - b);
  const medianRow = heights[Math.floor(heights.length / 2)] ?? height;
  return {rows, columns, positions: rows * columns, onTable, isolable, cells: table.cells.length, rowsPerPosition: height / medianRow};
}

/** A cell's text: the annotation's transcription when present, otherwise the PDF's own text layer, labelled. */
export function cellText(cell: Pick<TableCell, 'text' | 'pdf_text'>): {text: string; source: 'transcribed' | 'pdf' | 'none'} {
  if (cell.text) return {text: cell.text, source: 'transcribed'};
  if (cell.pdf_text) return {text: cell.pdf_text, source: 'pdf'};
  return {text: '', source: 'none'};
}

export type GroundingRole = {kind: 'answer' | 'row_path' | 'column_path' | 'support'; label: string};

/** The question's lookup path by cell: answer cell, numbered row and column header paths, support cells. */
export function groundingRoles(grounding?: TableGrounding | null): Map<string, GroundingRole> {
  const roles = new Map<string, GroundingRole>();
  if (!grounding) return roles;
  grounding.support_cell_ids.forEach(id => roles.set(id, {kind: 'support', label: 'S'}));
  grounding.column_header_path.forEach((id, i) => roles.set(id, {kind: 'column_path', label: `C${i + 1}`}));
  grounding.row_header_path.forEach((id, i) => roles.set(id, {kind: 'row_path', label: `R${i + 1}`}));
  grounding.answer_cell_ids.forEach(id => roles.set(id, {kind: 'answer', label: 'Answer'}));
  return roles;
}
