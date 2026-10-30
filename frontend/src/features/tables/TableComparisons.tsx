import type { components } from '../../api/generated/schema';

export type TableRelated = components['schemas']['TableRelatedPage'];
type Related = components['schemas']['TableRelatedCapture'];

const where = (item: Related) => `row ${(item.answer_row ?? -1) + 1}, column ${(item.answer_column ?? -1) + 1}`;

/** The saved captures a table-aware comparison reads; their IDs are recorded with the analysis. */
export function TableComparisons({methodId, related, error, useBaseline, onBaseline, partner, onPartner}: {
  methodId?: string; related?: TableRelated; error?: string; useBaseline: boolean; onBaseline: (use: boolean) => void;
  partner?: string; onPartner: (captureId: string) => void;
}) {
  if (methodId !== 'table.lookup_path' && methodId !== 'table.matched_contrast') return null;
  if (error) return <p className="docattentionlens-table-comparisons" role="alert">Related captures could not be loaded: {error}</p>;
  if (!related) return <p className="docattentionlens-table-comparisons">Finding saved captures of other questions on this table…</p>;
  const byId = new Map(related.items.map(i => [i.capture_id, i]));
  const unusable = related.items.filter(i => !i.comparable && i.relation !== 'same_question');
  if (methodId === 'table.lookup_path') {
    const baseline = related.baseline_capture_ids.map(id => byId.get(id)!).filter(Boolean);
    return <section className="docattentionlens-table-comparisons" aria-label="Question-swap baseline" data-testid="table-baseline">
      <h4>Question-swap baseline</h4>
      {baseline.length ? <>
        <label className="docattentionlens-checkbox"><input type="checkbox" checked={useBaseline} onChange={e => onBaseline(e.target.checked)}/>
          Compare with {baseline.length} saved capture{baseline.length === 1 ? '' : 's'} of other questions on this table</label>
        <ul>{baseline.slice(0, 6).map(i => <li key={i.capture_id}>{where(i)} · “{i.query.slice(0, 70)}{i.query.length > 70 ? '…' : ''}”</li>)}
          {baseline.length > 6 && <li>and {baseline.length - 6} more</li>}</ul>
      </> : <p>No other question on this table has a comparable saved capture. Capture other questions from this table (same model and resolution) to add a baseline.</p>}
      {!!unusable.length && <p className="docattentionlens-muted">{unusable.length} other capture{unusable.length === 1 ? ' is' : 's are'} not comparable (different model, axes or resolution).</p>}
    </section>;
  }
  const matched = related.matched_capture_ids.map(id => byId.get(id)!).filter(Boolean);
  return <section className="docattentionlens-table-comparisons" aria-label="Matched question" data-testid="table-partner">
    <h4>Matched question</h4>
    {matched.length ? <label>Compare with<select aria-label="Matched question capture" value={partner ?? matched[0].capture_id} onChange={e => onPartner(e.target.value)}>
      {matched.map(i => <option key={i.capture_id} value={i.capture_id}>{i.relation === 'same_row' ? 'Same row' : 'Same column'} · {where(i)} · {i.query.slice(0, 60)}</option>)}
    </select></label>
      : <p>No saved capture asks a question that shares this answer's row or column. Capture one (same model and resolution) to compare.</p>}
  </section>;
}
