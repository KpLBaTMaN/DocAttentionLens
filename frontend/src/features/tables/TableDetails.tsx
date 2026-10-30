import type { DisplaySettings } from '../../api/client';
import { Message } from '../../app/App';
import { cellText, type Table, type TableCell, type TableGrounding } from './tableGeometry';
import { TableHeatmap, type TableAllocation } from './TableHeatmap';

const position = (cell: TableCell) => `row ${cell.row_indices.map(i => i + 1).join('–')}, column ${cell.column_indices.map(i => i + 1).join('–')}`;

function CellName({cell}: {cell?: TableCell}) {
  if (!cell) return <span className="docattentionlens-muted">missing cell</span>;
  const {text, source} = cellText(cell);
  return <span data-cell-id={cell.native_id}>{text ? source === 'pdf' ? <em title="Text from the PDF's text layer; the annotation has no transcription">{text}</em> : text : <span className="docattentionlens-muted">no text</span>}</span>;
}

/**
 * Details › Table for a table grounding case: the table on this page, how its geometry was verified,
 * and the question's lookup path. The path is an evaluation annotation and never reaches the model.
 */
export function TableDetails({tables, grounding, error, settings, onView, allocation, selectedCell, onSelectCell}: {
  tables?: Table[]; grounding?: TableGrounding | null; error?: string; settings: DisplaySettings; onView: (settings: DisplaySettings) => void;
  /** The shown Table-aware result's aggregation; General results have none. */
  allocation?: TableAllocation; selectedCell?: string; onSelectCell?: (cellId?: string) => void;
}) {
  if (error) return <Message error>{error}</Message>;
  if (!tables) return <Message>Loading table structure…</Message>;
  if (!tables.length) return <p className="docattentionlens-muted">No table is located on this page.</p>;
  return <>{tables.map(table => {
    const cells = new Map(table.cells.map(c => [c.native_id, c]));
    const ground = grounding?.table_id === table.table_id ? grounding : undefined;
    const path = (ids: string[]) => ids.length ? ids.map((id, i) => <span key={id}>{i ? ' → ' : ''}<CellName cell={cells.get(id)}/></span>) : <span className="docattentionlens-muted">none</span>;
    const answer = ground ? cells.get(ground.answer_cell_ids[0]) : undefined;
    return <section key={table.key} className="docattentionlens-details-section docattentionlens-table-details" aria-label={`Table ${table.title}`} data-testid="table-details">
      <h3>{table.title}</h3>
      <p className="docattentionlens-muted">{table.row_count} rows × {table.column_count} columns · {table.cells.length} cells · revision {table.revision}</p>
      {allocation?.table_id === table.table_id
        ? <TableHeatmap table={table} allocation={allocation} grounding={ground} showGrounding={settings.show_support ?? false}
            colormap={settings.attention_colormap ?? 'inferno'} selected={selectedCell} onSelect={id => onSelectCell?.(id)}/>
        : <p className="docattentionlens-muted" data-testid="table-heatmap-hint">Choose a Table-aware method (Analysis › Table-aware) to see attention on each cell, row and column.</p>}
      <div className="docattentionlens-table-toggles" role="group" aria-label="Table overlays">
        <label className="docattentionlens-checkbox"><input type="checkbox" checked={settings.show_boxes ?? true} onChange={e => onView({...settings, show_boxes: e.target.checked})}/>Cell grid on the page</label>
        {ground && <label className="docattentionlens-checkbox"><input type="checkbox" checked={settings.show_support ?? false} onChange={e => onView({...settings, show_support: e.target.checked})}/>Question grounding on the page</label>}
      </div>
      {ground && <>
        <h4>Question grounding <small>evaluation only · never given to the model</small></h4>
        {ground.binding_status === 'needs_source_qa_review' && <p className="docattentionlens-table-flag" role="note" data-testid="binding-flag">Answer/cell mismatch: the saved answer “{ground.accepted_answers[0]}” differs from the text in its PDF cell (“{ground.pdf_answer_text ?? 'none'}”). Excluded from grounding comparisons by default.</p>}
        <dl className="docattentionlens-table-grounding" data-testid="table-grounding">
          <dt>Answer cell</dt><dd><CellName cell={answer}/>{answer && <small>{position(answer)}</small>}</dd>
          <dt>Row header path</dt><dd>{path(ground.row_header_path)}</dd>
          <dt>Column header path</dt><dd>{path(ground.column_header_path)}</dd>
          <dt>Support cells</dt><dd>{path(ground.support_cell_ids)}</dd>
          <dt>Accepted answers</dt><dd>{ground.accepted_answers.join(' · ')}{ground.unit ? ` (${ground.unit})` : ''} <small>{ground.value_type}</small></dd>
          <dt>Rationale</dt><dd>{ground.rationale}</dd>
        </dl>
      </>}
      <details><summary>Geometry and text sources</summary>
        <p>Crop-to-page image registration correlation: {table.registration_correlation.toFixed(3)}.</p>
        <p>Cells are located on this page through the dataset's verified crop-to-page transform. The logical cell boxes divide the table without gaps; the original PDF cell boxes are kept for reference.</p>
        <p>Cell text is the annotation's transcription. Where none was made, the PDF's own text layer is shown in <em>italics</em>.</p>
        {grounding && ground && <p>The question was written against the cropped table image; this case gives the model the full page.</p>}
      </details>
    </section>;
  })}</>;
}
