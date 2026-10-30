import { useEffect, useRef } from 'react';
import type { Element } from '../../api/client';
import type { Attention } from './GenerationPanel';
import { attentionWeightLabel } from './attentionDisplay';
import { bucketLabel, contributions, quantityLabel } from './perspective';
import { elementQuantityLabel } from './documentGrouping';

type Position = {index: number; role?:string; page_id?: string | null; ownership?: Record<string, number> | null};
const SHOWN_PATCHES = 12;

/** What one source of a causal encoding can attend to: itself and earlier active positions. */
export type SourceReach = {source: number; reaches: (index: number) => boolean; seen: number; total: number; onWholePage?: () => void};

/** A share of the row's total; a tiny measured share is never printed as 0%: only an exact zero is. */
export function shareLabel(mass: number, total: number) {
  if (!(total > 0)) return '—';
  const x = mass / total;
  return x === 0 ? '0%' : x < 1e-4 ? '<0.01%' : `${(x * 100).toFixed(x < .1 ? 2 : 1)}%`;
}

/** Each element's mass and measured patches on one page; the page rail and element summary share it. */
export function elementCoverage(row: Attention, positions: Position[], pageId: string, reach?: SourceReach) {
  const masses = new Map(row.allocations.filter(a => a.page_id === pageId && a.element_id).map(a => [a.element_id!, a.mass]));
  const reached = new Map<string, number>(), owned = new Map<string, number>();
  for (const p of positions) if (p.page_id === pageId) for (const bucket of Object.keys(p.ownership ?? {})) if (bucket.startsWith('element_')) {
    owned.set(bucket, (owned.get(bucket) ?? 0) + 1);
    if (p.index < row.target_count && (reach?.reaches(p.index) ?? true)) reached.set(bucket, (reached.get(bucket) ?? 0) + 1);
  }
  return {masses, reached, owned,
    /** Measured, but through only some of its patches: only a causal reach can cut an element. */
    partial: (id: string) => !!reach && masses.has(id) && (reached.get(id) ?? 0) < (owned.get(id) ?? 0)};
}

/** Element attention mass for the displayed page; missing structure is stated as missing. */
export function ElementSummary({row, positions, elements, pageId, pageSupplied, selectedId, onSelect, onPage, pageLabel, unavailable, provenance, documentAware, support, methodName, onShowPatches, reach}: {
  row?: Attention; positions: Position[]; elements: Element[]; pageId: string; pageSupplied: boolean;
  selectedId?: string | null; onSelect: (element: Element) => void; onPage: (pageId: string) => void;
  pageLabel: (pageId: string) => string; unavailable?: string; provenance?: string; documentAware: boolean;
  /** Designated supporting elements: an evaluation annotation shown for orientation, never a calculation input here. */
  support?: Set<string>;
  methodName?: string; onShowPatches?: () => void;
  /** A causal source's reach; without it every captured position below the row's target count is reachable. */
  reach?: SourceReach;
}) {
  const selectedRow = useRef<HTMLTableRowElement>(null);
  useEffect(() => {selectedRow.current?.scrollIntoView?.({block: 'nearest'});}, [selectedId, row]);
  if (unavailable) return <section className="docattentionlens-element-summary" aria-label="Element summary" data-testid="element-summary">
    <p role="status" data-testid="element-summary-unavailable">{unavailable}</p>{provenance && <p className="docattentionlens-structure-provenance">{provenance}</p>}</section>;
  if (!row) return <p>No attention row for this selection: the method reports a summary without one row (see Method result), or no capture is selected.</p>;
  // Only head envelopes (max/min over heads) are envelopes; other non-weight rows keep their own quantity name.
  const envelope = row.quantity.includes('envelope');
  const positionName=positions.some(p=>p.role==='document_text')?'Tokens':'Patches';
  const quantity = elementQuantityLabel(row.quantity);
  const share = (mass: number) => shareLabel(mass, row.total_mass);
  const pageAllocations = row.allocations.filter(a => a.page_id === pageId);
  const {masses, reached: patchCounts, owned: ownedCounts, partial} = elementCoverage(row, positions, pageId, reach);
  const pageElements = elements.filter(e => e.page_id === pageId);
  const mapped = pageElements.filter(e => masses.has(e.element_id)).sort((a, b) => masses.get(b.element_id)! - masses.get(a.element_id)!);
  const flagged = pageElements.filter(e => !masses.has(e.element_id) && e.geometry_flags.length);
  // Elements that own patches on this page but have no mass lie outside the row's support (after a causal
  // source): one fact for all of them, so they are listed once behind a summary line.
  const outOfReach = pageElements.filter(e => !masses.has(e.element_id) && !e.geometry_flags.length && ownedCounts.has(e.element_id));
  const unmapped = pageElements.filter(e => !masses.has(e.element_id) && !e.geometry_flags.length && !ownedCounts.has(e.element_id));
  const residuals = pageAllocations.filter(a => !a.element_id);
  const otherPages = [...row.allocations.filter(a => a.page_id && a.page_id !== pageId).reduce((m, a) => m.set(a.page_id!, (m.get(a.page_id!) ?? 0) + a.mass), new Map<string, number>())];
  const offPage = row.allocations.filter(a => !a.page_id);
  const label = (e: Element) => `${e.element_type} · element ${e.native_id}`;
  const detail = (e: Element) => `Element ${e.native_id} · ${e.element_type} · ${e.element_category}${e.reading_order !== null ? ` · reading order ${e.reading_order}` : ''}`;
  const selected = mapped.find(e => e.element_id === selectedId);
  const parts = selected ? contributions(positions.filter(p => p.page_id === pageId), row.values, row.target_count, selected.element_id) : [];
  return <section className="docattentionlens-element-summary" aria-label="Element summary" data-testid="element-summary">
    <h3>{quantity} · {pageLabel(pageId)}</h3>
    <p>Sum of {quantityLabel(row.quantity).toLowerCase()} × ownership for each element{envelope ? '; envelope values need not sum to one' : ''}.
      {methodName && <> Source analysis: {methodName === 'Element attention aggregation' ? 'Raw attention' : methodName}.</>} Select an element to inspect its contributing {positionName.toLowerCase()}.{documentAware ? '' : ' Document-aware draws these regions on the page.'}</p>
    {provenance && <p className="docattentionlens-structure-provenance">{provenance}</p>}
    {reach && reach.seen < reach.total && <p className="docattentionlens-source-reach" data-testid="source-reach">
      Source {reach.source} sees {reach.seen} of {reach.total} page patches: this encoder is causal, so a source reaches only earlier positions (patches above it and to its left).
      {reach.onWholePage && <> <button className="docattentionlens-link-button" onClick={reach.onWholePage}>Use the last token (whole page)</button></>}</p>}
    {!pageSupplied ? <p role="status">This page has no captured attention in the selected row, so its elements have no measured value.</p> : <>
      <table className="docattentionlens-element-table"><thead><tr><th>Element</th><th title={quantity}>Mass</th><th title="Share of the selected row's total mass">Share</th><th title="Captured patches that the element owns, fully or partly">{positionName}</th></tr></thead><tbody>
        {mapped.map(e => {const isSelected = e.element_id === selectedId; return <tr key={e.element_id} ref={isSelected ? selectedRow : undefined} aria-selected={isSelected} data-testid="element-summary-row" data-element-id={e.element_id} className={isSelected ? 'is-selected' : ''}>
          <td><button className="docattentionlens-row-button" aria-pressed={isSelected} title={detail(e)} onClick={() => onSelect(e)}>{label(e)}</button>{support?.has(e.element_id) && <span className="docattentionlens-support-tag" data-testid="support-tag" title="Designated supporting element: an evaluation annotation that may omit other relevant regions; never supplied to the model">support</span>}
            {partial(e.element_id) && <span className="docattentionlens-partial-tag" data-testid="partial-tag" title={`Only ${patchCounts.get(e.element_id) ?? 0} of its ${ownedCounts.get(e.element_id)} ${positionName.toLowerCase()} come before source ${reach!.source}; its mass covers those only`}>partial</span>}</td>
          <td>{attentionWeightLabel(masses.get(e.element_id)!)}</td><td>{share(masses.get(e.element_id)!)}</td>
          <td>{partial(e.element_id) ? `${patchCounts.get(e.element_id) ?? 0} of ${ownedCounts.get(e.element_id)}` : patchCounts.get(e.element_id) ?? 0}</td></tr>;})}
        {!mapped.length && <tr><td colSpan={4}>No annotated element on this page is mapped to captured positions.</td></tr>}
      </tbody></table>
      {selected && <div className="docattentionlens-element-contributions" data-testid="element-contributions">
        <h4>Contributing {positionName.toLowerCase()} · {detail(selected)}</h4>
        <p>{parts.length} {positionName.toLowerCase()} · Σ weight × ownership = {attentionWeightLabel(parts.reduce((s, p) => s + p.contribution, 0))} (element mass {attentionWeightLabel(masses.get(selected.element_id)!)}).</p>
        <table><thead><tr><th>Patch</th><th title="Patch attention weight">Weight</th><th title="Share of the patch owned by this element">Owned</th><th title="Weight × owned share; these sum to the element mass">Contribution</th></tr></thead><tbody>
          {parts.slice(0, SHOWN_PATCHES).map(p => <tr key={p.index} data-testid="element-contribution"><td>{p.index}</td><td>{attentionWeightLabel(p.weight)}</td><td>{(p.fraction * 100).toFixed(1)}%</td><td>{attentionWeightLabel(p.contribution)}</td></tr>)}
        </tbody></table>
        {parts.length > SHOWN_PATCHES && <p>{parts.length - SHOWN_PATCHES} smaller contributions not listed; all are drawn on the page.</p>}
      </div>}
      {(!!flagged.length || !!unmapped.length || !!outOfReach.length) && <><h4>Elements without a measured value</h4>
        {!!outOfReach.length && <details className="docattentionlens-out-of-reach" data-testid="out-of-reach">
          <summary>{outOfReach.length} {outOfReach.length === 1 ? 'element lies' : 'elements lie'} after {reach ? `source ${reach.source}` : 'the source'} and cannot receive its attention: not measured, not zero</summary>
          <ul className="docattentionlens-missing-elements">{outOfReach.map(e => <li key={e.element_id}><button className="docattentionlens-link-button" onClick={() => onSelect(e)}>{label(e)}</button></li>)}</ul>
        </details>}
        {(!!flagged.length || !!unmapped.length) && <ul className="docattentionlens-missing-elements">
          {flagged.map(e => <li key={e.element_id}><button className="docattentionlens-link-button" onClick={() => onSelect(e)}>{label(e)}</button> · geometry flagged; its patches are held in Unreviewed element geometry</li>)}
          {unmapped.map(e => <li key={e.element_id}><button className="docattentionlens-link-button" onClick={() => onSelect(e)}>{label(e)}</button> · no mapped positions (missing, not zero)</li>)}
        </ul>}</>}
    </>}
    <h4>Attention outside annotated elements{pageSupplied ? '' : ' on this page'}</h4>
    {onShowPatches&&<button className="docattentionlens-link-button" onClick={onShowPatches}>Show attention patches including background</button>}
    <table><tbody>
      {residuals.map(a => <tr key={a.bucket}><th>{bucketLabel(a.bucket)}</th><td>{attentionWeightLabel(a.mass)}</td><td>{share(a.mass)}</td></tr>)}
      {otherPages.map(([id, mass]) => <tr key={id}><th><button className="docattentionlens-link-button" onClick={() => onPage(id)}>{pageLabel(id)}</button> · elements and background</th><td>{attentionWeightLabel(mass)}</td><td>{share(mass)}</td></tr>)}
      {offPage.map(a => <tr key={a.bucket}><th>{bucketLabel(a.bucket)}</th><td>{attentionWeightLabel(a.mass)}</td><td>{share(a.mass)}</td></tr>)}
    </tbody></table>
  </section>;
}
