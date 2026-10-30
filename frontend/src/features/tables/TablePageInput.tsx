import { useState } from 'react';
import type { Page } from '../../api/client';
import { pageImageGrid } from '../attention-tool/fullAttentionEstimate';
import { TableOverlay } from './TableOverlay';
import { tableResolution, type Table, type TableGrounding } from './tableGeometry';

export type ImagePositions = 512 | 1024;
export const RESOLUTIONS: {id: ImagePositions; label: string; text: string}[] = [
  {id: 512, label: 'Standard · 512', text: 'The usual plan for every page.'},
  {id: 1024, label: 'Table detail · 1,024', text: 'Finer image positions; offered for a single supplied page.'},
];

/** Readable summary of how a planned input resolves the table (an estimate before the run). */
export function resolutionLabel(page: Pick<Page, 'width' | 'height'>, table: Table, cell: number, positions: ImagePositions) {
  const r = tableResolution(page, table, cell, positions);
  return {...r, text: `${r.positions.toLocaleString()} image positions · ${r.onTable} on the table · ${r.isolable} of ${r.cells} cells have a position of their own (${Math.round(100 * r.isolable / r.cells)}%) · about ${r.rowsPerPosition.toFixed(1)} table rows per position`};
}

/**
 * Setup's page input for a table grounding question: the full page the model will read, with the table,
 * its caption area, the cell grid and (evaluation only) the lookup path, plus the resolution plan.
 */
export function TablePageInput({page, tables, grounding, cell, positions, onPositions, sourceLabel}: {
  page: Page; tables: Table[]; grounding?: TableGrounding | null;
  /** Source pixels per merged image position of the chosen model; unknown models give no estimate. */
  cell?: number; positions: ImagePositions; onPositions: (positions: ImagePositions) => void; sourceLabel: string;
}) {
  const [showPositions, setShowPositions] = useState(true);
  const table = tables.find(t => t.table_id === grounding?.table_id) ?? tables[0];
  const grid = cell ? pageImageGrid(page.width, page.height, cell, positions) : undefined;
  const readout = cell && table ? resolutionLabel(page, table, cell, positions) : undefined;
  const scale = 420 / page.width;
  return <div className="docattentionlens-table-input" data-testid="table-page-input">
    <div className="docattentionlens-table-input-controls">
      <div><span className="docattentionlens-kicker">Model input</span><p><strong>Full page</strong> · {sourceLabel}</p>
        <p className="docattentionlens-muted">The question was written against the cropped table. Here the model reads the whole page, so the rest of the page competes for attention.</p></div>
      <div><span className="docattentionlens-kicker">Resolution</span>
        <div className="docattentionlens-actions" role="group" aria-label="Image resolution">{RESOLUTIONS.map(r =>
          <button type="button" key={r.id} className="docattentionlens-button" aria-pressed={positions === r.id} title={r.text} onClick={() => onPositions(r.id)}>{r.label}</button>)}</div>
        {readout ? <p className="docattentionlens-table-readout" data-testid="resolution-readout">{readout.text}. <span className="docattentionlens-muted">Estimate before the run.</span></p>
          : <p className="docattentionlens-muted">This model does not declare its image geometry, so no resolution estimate is shown.</p>}
        {readout && readout.isolable / readout.cells < .5 && <p className="docattentionlens-table-flag">Most cells share every image position with their neighbours at this resolution, so cell-level attention will blur across rows.</p>}
        <label className="docattentionlens-checkbox"><input type="checkbox" checked={showPositions} onChange={e => setShowPositions(e.target.checked)}/>Show image positions on the page</label>
      </div>
    </div>
    <svg className="docattentionlens-table-input-page" viewBox={`0 0 ${page.width} ${page.height}`} role="img" aria-label={`${sourceLabel} with the table outlined`}>
      <image href={page.image_url} width={page.width} height={page.height}/>
      {showPositions && grid && <g className="docattentionlens-position-grid" data-testid="position-grid">
        {Array.from({length: grid.columns - 1}, (_, i) => <line key={`c${i}`} x1={(i + 1) * page.width / grid.columns} x2={(i + 1) * page.width / grid.columns} y1={0} y2={page.height} vectorEffect="non-scaling-stroke"/>)}
        {Array.from({length: grid.rows - 1}, (_, i) => <line key={`r${i}`} y1={(i + 1) * page.height / grid.rows} y2={(i + 1) * page.height / grid.rows} x1={0} x2={page.width} vectorEffect="non-scaling-stroke"/>)}
      </g>}
      <TableOverlay tables={tables} grounding={grounding} showGrid showGrounding scale={scale}/>
    </svg>
    <p className="docattentionlens-muted">Gold: answer cell · dashed: the row (R) and column (C) header paths · dotted: support cells. These guide the investigation and evaluation; they are never given to the model.</p>
  </div>;
}
