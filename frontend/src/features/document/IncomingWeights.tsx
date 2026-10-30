import type { components } from '../../api/generated/schema';

export function IncomingWeights({column,error}:{column?:components['schemas']['ReceivedAttention'];error?:string}) {
  return <section aria-label="Incoming attention values" data-testid="incoming-evidence"><h2>Incoming attention · raw weights</h2>{error?<p role="alert">{error}</p>:column?<>
    <p>{column.normalization}. {column.sources.length} captured source rows · layer {column.layer} · head {column.head} · target {column.target_index}.</p>
    <div className="docattentionlens-method-table"><table><caption>Weight each captured source gives this target</caption><thead><tr><th>Source</th><th>Kind</th><th>Weight</th></tr></thead><tbody>{column.sources.map(s=><tr key={s.source_index}><td>{s.source_index} · {s.label}</td><td>{s.kind}</td><td>{s.value.toPrecision(6)}</td></tr>)}</tbody></table></div>
  </>:<p>Select a token or image patch to inspect incoming weights.</p>}</section>;
}
