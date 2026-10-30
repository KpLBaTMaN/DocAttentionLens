import { useCallback, useId, useState } from 'react';
import { Link, useParams, useSearchParams } from 'react-router-dom';
import { api, unwrap, type Method } from '../../api/client';
import { Heading, Message } from '../../app/App';
import { useRemote } from '../../app/remote';
import { perspectiveLabel } from '../document/perspective';
import { FACETS, NO_FACETS, facetOptions, groupMethods, type FacetId, type FacetSelection } from './methodFacets';
import '../../styles/methods.css';

/** How a calculation relates to the literature its references cite; the note names the departures. */
const LITERATURE_LABELS = {published_method: 'Published method', standard_summary: 'Standard summary', adaptation: 'Adaptation'} as const;

const detailUrl = (m: Method) => `/toolbox/${encodeURIComponent(m.method_id)}?variant=${encodeURIComponent(m.variant_id)}`;

function MethodDetail({method, loading, error, investigationUrl}: {method?: Method; loading: boolean; error?: string; investigationUrl?: string}) {
  return <><Heading kicker="Methods / Method detail" title={method?.name ?? 'Method detail'}/><div className="docattentionlens-content docattentionlens-methods-detail">
    <nav className="docattentionlens-methods-detail-bar" aria-label="Method detail navigation"><Link to="/toolbox">← All methods</Link>{investigationUrl&&<Link className="docattentionlens-button" to={investigationUrl}>Return to investigation</Link>}</nav>
    {error ? <Message error>{error}</Message> : loading ? <Message>Loading descriptor…</Message> : !method ? <Message error>Method not found.</Message> : <>
      <div className="docattentionlens-detail-lead"><span className="docattentionlens-badge">{method.status}</span><h2>What it does</h2><p>{method.summary}</p>
        {method.literature_relation&&<p data-testid="literature-relation"><span className="docattentionlens-badge">{LITERATURE_LABELS[method.literature_relation.kind]}</span> {method.literature_relation.note}</p>}</div>
      <div className="docattentionlens-detail-grid"><section className="docattentionlens-section"><h2>Quantity and support</h2><dl><dt>Analysis perspective</dt><dd>{perspectiveLabel(method.perspective)}{method.perspective==='document_aware'?' · uses annotated document structure':' · uses model positions only'}</dd><dt>Evidence kind</dt><dd>{method.evidence_kind}</dd><dt>Quantity</dt><dd>{method.quantity}</dd><dt>Units</dt><dd>{method.output.units}</dd><dt>Support</dt><dd>{method.output.support}</dd><dt>Normalization</dt><dd>{method.output.normalization}</dd><dt>Signed values</dt><dd>{method.output.signed ? 'Yes' : 'No'}</dd></dl></section>
        <section className="docattentionlens-section"><h2>Requirements</h2><ul>{method.requires.map(r => <li key={r}>{r}</li>)}</ul><p>Execution: {method.execution_requirement}</p><p>Output target: {method.target_requirement}</p>{method.status==='planned'?<><button className="docattentionlens-button" disabled aria-describedby="planned-reason">Run calculation</button><p id="planned-reason">Planned variant: no implementation or validation evidence is available.</p></>:<><Link className="docattentionlens-button docattentionlens-primary" to="/attention-tool">Open document workspace</Link><p>{method.family==='attention_interventions'?'Open a case, then ⋯ › Experiments › Attention interventions to prepare, run and inspect a controlled model experiment.':method.requires.includes('full_square_attention_matrices')?'Select a saved retriever encoding with complete language-attention matrices. Retained generator rows cannot support this calculation.':method.evidence_kind.startsWith('attention_')||method.family==='head_and_representation'?'Select compatible saved attention in the workspace. Availability follows the captured source rows, geometry and annotations.':'Select the method’s compatible capture or experiment in the document workspace.'}</p></>}</section>
        <section className="docattentionlens-section"><h2>Limitations</h2><ul>{method.limitations.map(r => <li key={r}>{r}</li>)}</ul><h3>Comparison constraints</h3><ul>{method.comparison_constraints.map(r => <li key={r}>{r}</li>)}</ul></section></div>
      <details className="docattentionlens-section"><summary>Definition, version, parameters and references</summary><p>{method.definition}</p><p>Method {method.method_id} · Descriptor {method.descriptor_version} · Variant {method.variant_id} · Implementation {method.implementation_version ?? 'unavailable'}</p><pre>{JSON.stringify(method.parameters_schema, null, 2)}</pre><ul>{method.source_references.map(r => <li key={r}>{r.startsWith('https://') ? <a href={r} target="_blank" rel="noreferrer">{r}</a> : <span>{r}</span>}</li>)}</ul><p>Validation evidence: {method.validation_evidence.length ? method.validation_evidence.join(', ') : 'None'}</p></details>
    </>}</div></>;
}

function Facets({items, search, selection, toggle, clear, active}: {items: Method[]; search: string; selection: FacetSelection; toggle: (facet: FacetId, value: string) => void; clear: () => void; active: number}) {
  const [open, setOpen] = useState(false);
  const panelId = useId();
  return <aside className="docattentionlens-methods-facets" aria-label="Method filters">
    <button type="button" className="docattentionlens-button docattentionlens-methods-filters-toggle" aria-expanded={open} aria-controls={panelId} onClick={() => setOpen(o => !o)}>Filters{active ? ` · ${active} active` : ''}</button>
    <div id={panelId} className="docattentionlens-methods-facet-panel" data-open={open || undefined}>
      {FACETS.map(facet => <fieldset key={facet.id} className="docattentionlens-methods-facet"><legend>{facet.label}</legend>
        {facetOptions(items, facet.id, search, selection).map(option => { const checked = selection[facet.id].includes(option.value); return <label key={option.value} className={option.count || checked ? undefined : 'is-empty'}>
          <input type="checkbox" checked={checked} disabled={!option.count && !checked} onChange={() => toggle(facet.id, option.value)}/>
          <span>{option.label}</span><span className="docattentionlens-methods-count">{option.count}</span></label>; })}
      </fieldset>)}
      <button type="button" className="docattentionlens-methods-quiet" disabled={!active && !search} onClick={clear}>Clear filters</button>
    </div>
  </aside>;
}

export function Toolbox() {
  const { methodId } = useParams();
  const [params] = useSearchParams();
  const returnTo=params.get('returnTo');
  const investigationUrl=returnTo&&/^\/attention-tool(?:\/[^?]*)?(?:\?.*)?$/.test(returnTo)?returnTo:undefined;
  const catalogue = useRemote(useCallback(async () => unwrap(await api.GET('/api/v1/methods')), []));
  const [filter, setFilter] = useState('');
  const [selection, setSelection] = useState<FacetSelection>(NO_FACETS);
  if (methodId) {
    const method = catalogue.data?.items.find(m => m.method_id === methodId&&(!params.get('variant')||m.variant_id===params.get('variant')));
    return <MethodDetail method={method} loading={!catalogue.data} error={catalogue.error} investigationUrl={investigationUrl}/>;
  }
  const items = catalogue.data?.items ?? [];
  const active = Object.values(selection).reduce((n, values) => n + values.length, 0);
  const toggle = (facet: FacetId, value: string) => setSelection(s => ({...s, [facet]: s[facet].includes(value) ? s[facet].filter(v => v !== value) : [...s[facet], value]}));
  const clear = () => { setSelection(NO_FACETS); setFilter(''); };
  const {count, groups} = groupMethods(items, filter, selection);
  return <><Heading kicker="Attention inspection" title="Methods"><span className="docattentionlens-badge">{catalogue.data?.items.length ?? '…'} method variants</span></Heading><div className="docattentionlens-content docattentionlens-methods">
    <div className="docattentionlens-methods-intro">
      <p className="docattentionlens-methods-lead">Browse the calculations available for inspecting captured attention, with what each one reports and needs.</p>
      <details className="docattentionlens-methods-about"><summary>About methods</summary><p>Explore methods for inspecting attention, aggregating it over document elements and comparing heads. General methods analyse model positions; document-aware methods aggregate or compare the same attention over annotated document regions. Retrieval scoring, output attribution and layout effects are labelled separately. Each variant lists its implementation status, required evidence and limitations. Availability is checked against the selected capture in the Attention Tool.</p></details>
    </div>
    <div className="docattentionlens-methods-layout">
      <Facets items={items} search={filter} selection={selection} toggle={toggle} clear={clear} active={active}/>
      <div className="docattentionlens-methods-search">
        <label className="docattentionlens-search">Filter methods<input value={filter} onChange={e => setFilter(e.target.value)} placeholder="Method, operation or evidence kind"/></label>
        {catalogue.data&&<p className="docattentionlens-methods-tally" aria-live="polite">{count === items.length ? `${count} methods` : `${count} of ${items.length} methods`}</p>}
      </div>
      <div className="docattentionlens-methods-results">
        {catalogue.error ? <Message error>{catalogue.error}</Message> : !catalogue.data ? <Message>Loading method catalogue…</Message> : !count ? <div className="docattentionlens-methods-none"><Message>No methods match these filters.</Message><button type="button" className="docattentionlens-methods-quiet" onClick={clear}>Clear filters</button></div> :
          groups.map(group => <section className="docattentionlens-methods-family" key={group.family} aria-label={group.label}>
            <h2>{group.label}<span className="docattentionlens-methods-count">{group.methods.length}</span></h2>
            <div className="docattentionlens-method-list">{group.methods.map(m => <Link className={`docattentionlens-method-row${m.status === 'planned' ? ' is-planned' : ''}`} to={detailUrl(m)} key={`${m.method_id}:${m.variant_id}`}>
              <div className="docattentionlens-method-row-main"><h3>{m.name}</h3><p>{m.summary}</p></div>
              <span className="docattentionlens-kicker">{perspectiveLabel(m.perspective)}</span>
              <span className="docattentionlens-badge">{m.status}</span>
            </Link>)}</div>
          </section>)}
      </div>
    </div>
  </div></>;
}
