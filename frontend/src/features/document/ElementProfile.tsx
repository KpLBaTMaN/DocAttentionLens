import { useEffect, useId, useMemo, useState, type CSSProperties, type KeyboardEvent } from 'react';
import type { Element } from '../../api/client';
import type { components } from '../../api/generated/schema';
import { heatColor, mapGradient, rgb, type Colormap } from './attentionDisplay';
import { downloadBlob } from './receiptExport';

export type ElementProfileData = components['schemas']['ElementProfile'];
type ProfileElement = components['schemas']['ProfileElement'];
export type ProfileMetric = 'mass' | 'image' | 'enrichment';

export const PROFILE_METRICS: {id: ProfileMetric; label: string; title: string}[] = [
  {id: 'mass', label: 'Mass', title: "Original-domain attention mass: the element's share of the whole source row"},
  {id: 'image', label: 'Image share', title: "The element's share of the row's image attention (every supplied page, background included)"},
  {id: 'enrichment', label: 'Enrichment', title: "Image share ÷ the element's share of image positions; 1 is attention in proportion to area"},
];
export const isProfileMetric = (value: unknown): value is ProfileMetric => PROFILE_METRICS.some(m => m.id === value);

/** One cell under a metric; null when its denominator (image mass or area share) is zero or absent. */
export function profileValue(profile: ElementProfileData, element: ProfileElement, metric: ProfileMetric, layer: number, head: number): number | null {
  const mass = element.mass[layer][head];
  if (metric === 'mass') return mass;
  const image = profile.image_mass[layer][head];
  if (!image) return null;
  return metric === 'image' ? mass / image : element.area_share ? mass / image / element.area_share : null;
}

/** Arithmetic mean over heads of each layer's cell values (metric first, then mean); null if any head is unavailable. */
export function layerMeans(profile: ElementProfileData, element: ProfileElement, metric: ProfileMetric): (number | null)[] {
  return Array.from({length: profile.layer_count}, (_, l) => {
    const values = Array.from({length: profile.head_count}, (_, h) => profileValue(profile, element, metric, l, h));
    return values.some(v => v === null) ? null : (values as number[]).reduce((a, b) => a + b, 0) / values.length;
  });
}

/** Cells in descending order; ties keep architecture order (layer, then head). Unavailable cells are never ranked. */
export function rankCells(profile: ElementProfileData, element: ProfileElement, metric: ProfileMetric) {
  const cells: {layer: number; head: number; value: number}[] = [];
  for (let layer = 0; layer < profile.layer_count; layer++)
    for (let head = 0; head < profile.head_count; head++) {
      const value = profileValue(profile, element, metric, layer, head);
      if (value !== null) cells.push({layer, head, value});
    }
  return cells.sort((a, b) => b.value - a.value || a.layer - b.layer || a.head - b.head);
}

/** Two ranked values this close are within the capture's BF16 rounding, so their order is not meaningful. */
export const nearTie = (value: number, next?: number) => next !== undefined && value > 0 && (value - next) / value < .01;

export function formatProfile(value: number | null | undefined, metric: ProfileMetric) {
  if (value === null || value === undefined) return 'Unavailable';
  if (metric === 'enrichment') return value !== 0 && value < .01 ? `${value.toExponential(1)}×` : `${value.toFixed(2)}×`;
  const percent = value * 100;
  return percent === 0 ? '0%' : percent < .01 ? `${percent.toExponential(1)}%` : `${percent.toFixed(2)}%`;
}

export type ProfileStatus =
  | {kind: 'unavailable'; reason: string}
  | {kind: 'loading'}
  | {kind: 'error'; message: string}
  | {kind: 'ready'; profile: ElementProfileData; analysisId: string; notices: string[]};

export type ProfileSource = {label: string; captureId: string; step?: number; sourceIndex?: number};

/** Why the selected element has no profile values, in the words of the element list. */
function elementGap(profile: ElementProfileData, element: Element, suppliedPageIds: string[]) {
  if (!suppliedPageIds.includes(element.page_id)) return 'Unavailable: this page was not supplied to the model, so no attention was captured for it.';
  if (element.geometry_flags.length) return 'Unreviewed geometry: its allocation is held in the residual, not measured per element.';
  if (profile.unmeasured_element_ids?.includes(element.element_id)) return 'No mapped valid position: unmeasured, not zero.';
  return 'No mapped positions in this capture: missing, not zero.';
}

function exportProfile(kind: 'csv' | 'json', profile: ElementProfileData, element: ProfileElement, source: ProfileSource, analysisId: string) {
  const prefix = `element-profile-${element.element_id}-${source.captureId}`;
  const cells = Array.from({length: profile.layer_count}, (_, layer) => Array.from({length: profile.head_count}, (_, head) => ({layer, head,
    mass: element.mass[layer][head], image_mass: profile.image_mass[layer][head],
    image_share: profileValue(profile, element, 'image', layer, head), enrichment: profileValue(profile, element, 'enrichment', layer, head)}))).flat();
  if (kind === 'json') {
    downloadBlob(new Blob([JSON.stringify({method_id: 'attention.element_profile', analysis_id: analysisId, capture_id: source.captureId,
      source: {label: source.label, step: source.step ?? null, source_index: source.sourceIndex ?? null},
      element: {element_id: element.element_id, page_id: element.page_id, label: element.label, area_share: element.area_share},
      axes: {layers: profile.layer_count, query_heads: profile.head_count, kv_heads: profile.kv_head_count ?? null, deepstack_layers: profile.deepstack_layers ?? []},
      image_positions: profile.image_positions, policy: profile.policy, denominator: profile.denominator, significant_digits: profile.significant_digits,
      metrics: Object.fromEntries(PROFILE_METRICS.map(m => [m.id, m.title])), cells}, null, 2)], {type: 'application/json'}), `${prefix}.json`);
    return;
  }
  const header = [`# Element attention profile · ${element.label} (${element.element_id}, ${element.page_id})`,
    `# Source: ${source.label} · capture ${source.captureId} · analysis ${analysisId}`,
    `# mass: original captured domain; image_share = mass / image_mass; enrichment = image_share / area_share (${element.area_share ?? 'unavailable'})`,
    `# Values rounded to ${profile.significant_digits} significant digits; empty cells are unavailable, not zero`];
  const value = (v: number | null) => v === null ? '' : String(v);
  const lines = cells.map(c => [c.layer, c.head, c.mass, c.image_mass, value(c.image_share), value(c.enrichment)].join(','));
  downloadBlob(new Blob([[...header, 'layer,head,mass,image_mass,image_share,enrichment', ...lines].join('\n') + '\n'], {type: 'text/csv'}), `${prefix}.csv`);
}

const ProfileHeading = () => <div className="docattentionlens-profile-head"><h3>Heads and layers</h3><span>Element attention profile</span>
  <em className="docattentionlens-raw-tag" title="Each cell is one layer and head of raw attention, whatever method the page shows">raw attention</em></div>;

/**
 * The Heads and layers section when no profile can be shown (no element selected, a retrieval view, a column view):
 * why, and the next step, so the feature is found where it lives.
 */
export function ProfileHint({text, action}: {text: string; action?: {label: string; run: () => void}}) {
  return <section className="docattentionlens-element-profile" aria-label="Element attention profile" data-testid="element-profile-hint">
    <ProfileHeading/><p className="docattentionlens-profile-note">{text}</p>
    {action && <button type="button" className="docattentionlens-button" onClick={action.run}>{action.label}</button>}
  </section>;
}

/** Details › Element: every captured layer and head's attention to the selected element, on one scale. */
export function ElementProfile({status, element, suppliedPageIds, source, layer, head, linked, metric, onMetric, onShow, onElement, colormap = 'inferno'}: {
  status: ProfileStatus; element: Element; suppliedPageIds: string[]; source: ProfileSource;
  /** The scrubber's layer and head, outlined in the grid; head -1 while All heads shows the head mean (no single cell is shown). */
  layer: number; head: number;
  /** Whether the page method takes one layer and head from the scrubber. */
  linked: boolean;
  metric: ProfileMetric; onMetric: (metric: ProfileMetric) => void;
  onShow: (layer: number, head: number) => void;
  onElement: (elementId: string) => void;
  colormap?: Colormap;
}) {
  const id = useId();
  const [cursor, setCursor] = useState<[number, number]>([layer, head]);
  // The cursor follows the scrubber, so the readout describes what the page shows.
  useEffect(() => setCursor([layer, head]), [layer, head]);
  const ready = status.kind === 'ready' ? status : undefined;
  const data = ready?.profile.elements.find(e => e.element_id === element.element_id);
  const ranked = useMemo(() => ready && data ? rankCells(ready.profile, data, metric) : [], [ready, data, metric]);
  const means = useMemo(() => ready && data ? layerMeans(ready.profile, data, metric) : [], [ready, data, metric]);
  const heading = <ProfileHeading/>;
  if (status.kind !== 'ready') return <section className="docattentionlens-element-profile" aria-label="Element attention profile" data-testid="element-profile">{heading}
    <p className="docattentionlens-profile-note" role={status.kind === 'error' ? 'alert' : 'status'}>{status.kind === 'loading' ? 'Calculating every layer and head from the saved capture…'
      : status.kind === 'error' ? status.message : status.reason}</p></section>;
  const profile = status.profile;
  if (!data) return <section className="docattentionlens-element-profile" aria-label="Element attention profile" data-testid="element-profile">{heading}
    <p className="docattentionlens-profile-note" role="status" data-testid="element-profile-gap">{elementGap(profile, element, suppliedPageIds)}</p></section>;
  const layers = profile.layer_count, heads = profile.head_count;
  const maximum = Math.max(ranked[0]?.value ?? 0, 1e-12);
  const group = profile.kv_head_count && profile.kv_head_count < heads && heads % profile.kv_head_count === 0 ? heads / profile.kv_head_count : 0;
  const labelEvery = heads > 16 ? 4 : heads > 8 ? 2 : 1;
  const deepstack = new Set(profile.deepstack_layers ?? []);
  const [cl, ch] = [Math.max(0, Math.min(cursor[0], layers - 1)), Math.max(0, Math.min(cursor[1], heads - 1))];
  const cellId = (l: number, h: number) => `${id}-l${l}-h${h}`;
  const commit = (l: number, h: number) => {setCursor([l, h]); if (linked) onShow(l, h);};
  const onKey = (e: KeyboardEvent<HTMLDivElement>) => {
    const moves: Record<string, [number, number]> = {ArrowUp: [-1, 0], ArrowDown: [1, 0], ArrowLeft: [0, -1], ArrowRight: [0, 1], PageUp: [-5, 0], PageDown: [5, 0]};
    let move: ((at: [number, number]) => [number, number]) | undefined;
    if (moves[e.key]) move = ([l, h]) => [l + moves[e.key][0], h + moves[e.key][1]];
    else if (e.key === 'Home') move = ([l]) => [e.ctrlKey ? 0 : l, 0];
    else if (e.key === 'End') move = ([l]) => [e.ctrlKey ? layers - 1 : l, heads - 1];
    else if (e.key === 'Enter' || e.key === ' ') {e.preventDefault(); commit(cl, ch); return;}
    if (!move) return;
    e.preventDefault();
    // Functional, so keys pressed faster than a render still compose.
    setCursor(at => {const [l, h] = move!([Math.max(0, Math.min(at[0], layers - 1)), Math.max(0, Math.min(at[1], heads - 1))]);
      return [Math.max(0, Math.min(layers - 1, l)), Math.max(0, Math.min(heads - 1, h))];});
  };
  const value = (l: number, h: number) => profileValue(profile, data, metric, l, h);
  const layerMax = (l: number) => {
    let best = -1, at = 0;
    for (let h = 0; h < heads; h++) {const v = value(l, h); if (v !== null && v > best) {best = v; at = h;}}
    return best < 0 ? undefined : {value: best, head: at};
  };
  const topLayers = means.map((m, l) => ({layer: l, value: m})).filter((m): m is {layer: number; value: number} => m.value !== null)
    .sort((a, b) => b.value - a.value || a.layer - b.layer).slice(0, 3);
  // The focused cell's place among every mapped element in that head's row, by mass.
  const rowMasses = profile.elements.map(e => ({element: e, mass: e.mass[cl][ch]})).sort((a, b) => b.mass - a.mass || (a.element.element_id < b.element.element_id ? -1 : 1));
  const rank = rowMasses.findIndex(r => r.element.element_id === data.element_id) + 1;
  const top = rowMasses[0];
  const shown = linked && cl === layer && ch === head;
  const cursorMax = layerMax(cl);
  const metricLabel = PROFILE_METRICS.find(m => m.id === metric)!.label;
  const unitScale = metric === 'enrichment' && maximum > 1 ? `${(1 / maximum * 100).toFixed(1)}%` : undefined;
  return <section className="docattentionlens-element-profile" aria-label="Element attention profile" data-testid="element-profile">
    {heading}
    <p className="docattentionlens-profile-note">{source.label} · raw attention at every captured layer and query head{linked ? '' : '; it does not decompose the combined method shown on the page'}. Choose a cell to {linked ? 'show that head on the page' : 'read it'}.</p>
    <div className="docattentionlens-segmented" role="radiogroup" aria-label="Profile value" onKeyDown={event => {
      if (!['ArrowLeft', 'ArrowRight', 'ArrowUp', 'ArrowDown'].includes(event.key)) return;
      event.preventDefault();
      const at = PROFILE_METRICS.findIndex(m => m.id === metric), by = ['ArrowRight', 'ArrowDown'].includes(event.key) ? 1 : -1;
      const next = PROFILE_METRICS[(at + by + PROFILE_METRICS.length) % PROFILE_METRICS.length];
      onMetric(next.id);
      requestAnimationFrame(() => document.getElementById(`${id}-metric-${next.id}`)?.focus());
    }}>
      {PROFILE_METRICS.map(m => <button key={m.id} id={`${id}-metric-${m.id}`} type="button" role="radio" aria-checked={metric === m.id} tabIndex={metric === m.id ? 0 : -1}
        title={m.title} onClick={() => onMetric(m.id)}>{m.label}</button>)}
    </div>
    <div className="docattentionlens-profile-atlas" style={{'--docattentionlens-profile-heads': heads, '--docattentionlens-profile-row': layers > 24 ? '11px' : '15px'} as CSSProperties}>
      <div className="docattentionlens-profile-columns" aria-hidden="true"><span>L\H</span>
        {Array.from({length: heads}, (_, h) => <span key={h} className={group && h && h % group === 0 ? 'is-group-start' : undefined}>{h % labelEvery === 0 ? h : ''}</span>)}
        <span title="Layer mean (bar) and strongest head (tick)">Mean</span></div>
      <div className="docattentionlens-profile-grid" role="grid" tabIndex={0} aria-label={`${metricLabel} for ${data.label} by layer (rows) and head (columns)`}
        aria-activedescendant={cellId(cl, ch)} aria-describedby={`${id}-readout`} onKeyDown={onKey} data-testid="element-profile-grid">
        {Array.from({length: layers}, (_, l) => {
          const mean = means[l], peak = layerMax(l);
          return <div key={l} role="row" className={`docattentionlens-profile-row${l === cl ? ' is-cursor-row' : ''}`}>
            <span role="rowheader" className={deepstack.has(l) ? 'is-deepstack' : undefined} title={deepstack.has(l) ? 'DeepStack: image keys and values here include vision features' : undefined}>{l}{deepstack.has(l) ? '†' : ''}</span>
            {Array.from({length: heads}, (_, h) => {
              const v = value(l, h);
              const classes = ['docattentionlens-profile-cell', group && h && h % group === 0 ? 'is-group-start' : '', v === null ? 'is-unavailable' : '',
                linked && l === layer && h === head ? 'is-current' : '', l === cl && h === ch ? 'is-cursor' : ''].filter(Boolean).join(' ');
              return <div key={h} id={cellId(l, h)} role="gridcell" className={classes} aria-selected={linked && l === layer && h === head}
                aria-label={`Layer ${l}, head ${h}: ${formatProfile(v, metric)}`} title={`Layer ${l} · head ${h}: ${formatProfile(v, metric)}`}
                data-layer={l} data-head={h} style={v === null ? undefined : {background: rgb(heatColor(v / maximum, colormap))}}
                onClick={() => commit(l, h)}/>;
            })}
            <span className="docattentionlens-profile-margin" title={`Layer ${l}: mean ${formatProfile(mean, metric)}${peak ? ` · max ${formatProfile(peak.value, metric)} (head ${peak.head})` : ''}`}>
              <i style={{width: `${(mean ?? 0) / maximum * 100}%`}}/>{peak && <b style={{left: `${Math.min(100, peak.value / maximum * 100)}%`}}/>}</span>
          </div>;
        })}
      </div>
    </div>
    <div className="docattentionlens-profile-key"><span>0</span><i style={{background: mapGradient(colormap)}}>{unitScale && <b style={{left: unitScale}} title="1× · attention in proportion to area"/>}</i>
      <span>{formatProfile(ranked[0]?.value ?? 0, metric)}</span></div>
    <p className="docattentionlens-profile-note">Common scale: 0 to this element's largest cell. Margin: layer mean (bar) and strongest head (tick).{group ? ` Thick column rules separate the ${profile.kv_head_count} key/value head groups.` : ''}{deepstack.size ? ' † DeepStack layers.' : ''}{unitScale ? ' The key marks 1×.' : ''}</p>
    <dl id={`${id}-readout`} className="docattentionlens-profile-readout" aria-live="polite" data-testid="element-profile-readout">
      <dt>Cell</dt><dd>Layer {cl} · head {ch}{shown ? <small> · shown on the page</small>
        : <> <button type="button" className="docattentionlens-link-button" onClick={() => onShow(cl, ch)}
          title={linked ? 'Set the layer scrubber to this layer and head' : 'Switch the page to Element attention aggregation at this layer and head'}>Show on page</button></>}</dd>
      <dt>Mass</dt><dd data-testid="profile-mass">{formatProfile(profileValue(profile, data, 'mass', cl, ch), 'mass')} of the row</dd>
      <dt>Image share</dt><dd>{formatProfile(profileValue(profile, data, 'image', cl, ch), 'image')} of image attention</dd>
      <dt>Enrichment</dt><dd>{formatProfile(profileValue(profile, data, 'enrichment', cl, ch), 'enrichment')} <small>area {data.area_share === null || data.area_share === undefined ? 'unavailable' : formatProfile(data.area_share, 'mass')} of image positions</small></dd>
      <dt>Element rank</dt><dd data-testid="profile-rank">{rank} of {profile.elements.length} mapped elements by mass in this head{top && top.element.element_id !== data.element_id
        ? <> · top: <button type="button" className="docattentionlens-link-button" onClick={() => onElement(top.element.element_id)}>{top.element.label}</button> {formatProfile(top.mass, 'mass')}</> : ' · the top element'}</dd>
      <dt>Layer {cl}</dt><dd>mean {formatProfile(means[cl], metric)}{cursorMax ? ` · max ${formatProfile(cursorMax.value, metric)} (head ${cursorMax.head})` : ''}</dd>
    </dl>
    {!linked && <p className="docattentionlens-profile-note">The page method chooses its own layers or heads, so choosing a cell only updates this readout.</p>}
    <div className="docattentionlens-profile-ranks">
      <div><h4>Strongest heads · {metricLabel}</h4><ol data-testid="profile-top-heads">{ranked.slice(0, 5).map((c, i) => <li key={`${c.layer}/${c.head}`}>
        <button type="button" className="docattentionlens-link-button" onClick={() => commit(c.layer, c.head)}>L{c.layer} · H{c.head}</button> {formatProfile(c.value, metric)}
        {nearTie(c.value, ranked[i + 1]?.value) && <small title="Within 1% of the next value: comparable to BF16 rounding of the captured weights, so treat the order as tied"> ≈ next</small>}</li>)}</ol></div>
      <div><h4>Strongest layer means</h4><ol>{topLayers.map(m => <li key={m.layer}>
        <button type="button" className="docattentionlens-link-button" onClick={() => {const peak = layerMax(m.layer); commit(m.layer, peak?.head ?? ch);}}>Layer {m.layer}</button> {formatProfile(m.value, metric)}</li>)}</ol></div>
    </div>
    <p className="docattentionlens-profile-note">The layer scrubber's bars show this element's layer means. Each cell is a separate allocation of one source row; it describes attention, not causal influence.{metric === 'mass' ? ' Mass favours larger elements; Enrichment adjusts for area.' : ''}</p>
    {status.notices.filter(n => n.includes('DeepStack') || n.includes('unmeasured')).map((n, i) => <p key={i} className="docattentionlens-profile-note">{n}</p>)}
    <div className="docattentionlens-actions"><button type="button" className="docattentionlens-quiet" onClick={() => exportProfile('csv', profile, data, source, status.analysisId)}>Export CSV</button>
      <button type="button" className="docattentionlens-quiet" onClick={() => exportProfile('json', profile, data, source, status.analysisId)}>Export JSON</button></div>
  </section>;
}
