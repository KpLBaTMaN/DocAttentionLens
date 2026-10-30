import { useEffect, useState } from 'react';
import { Link } from 'react-router-dom';
import { api, errorMessage, unwrap, type Case, type Document, type DisplaySettings, type Element } from '../../api/client';
import type { components } from '../../api/generated/schema';
import { Message } from '../../app/App';
import { Busy } from '../../app/Busy';
import type { useRetrievalAcquisition, View, UpdateView } from './useAcquisition';
import { SignalDisplay } from './SignalDisplay';

export type Capture = components['schemas']['RetrievalCapture'];
export type Row = components['schemas']['SimilarityRow'];
type Session = ReturnType<typeof useRetrievalAcquisition>;

/** MaxSim's source: the query position whose similarity to every patch is shown. In Details › Method. */
export function SimilaritySource({capture,state,update}:{capture?:Capture;state:View;update:UpdateView}) {
  return <div className="docattentionlens-similarity-controls" aria-label="Retrieval selection" role="group">
    {capture?<label>Query position<select aria-label="Overlay query position" value={state.query_position??0} onChange={e=>update(s=>({...s,query_position:Number(e.target.value)}))}>{capture.query_positions.map(p=>
      <option key={p.index} value={p.index} disabled={!p.eligible}>{p.index} · {p.piece} · {p.role}</option>)}</select></label>:<p>No retrieval capture selected.</p>}
  </div>;
}

/** MaxSim similarity display, shown above the MaxSim ledgers in Details › Result. */
export function SimilarityControls({capture,state,settings,update,onView}:{capture?:Capture;state:View;settings:DisplaySettings;update:UpdateView;onView:(next:DisplaySettings)=>void}) {
  return <div className="docattentionlens-similarity-controls"><SignalDisplay enabled={!!capture} state={state} settings={settings} update={update} onView={onView}/></div>;
}

export function RetrievalPanel({ record, document, state, capture, session }: { record: Case; document: Document; state: View; capture?: Capture; session: Session }) {
  const model = session.plan.data?.model_name ?? 'the retriever';
  return <section aria-label="Retrieval acquisition">
    {session.captureAttention&&<><p>Saves Question, Page and the diagnostic Page + question attention. Retrieval scores use the separate Question and Page readings. The diagnostic reading is bounded to 1,100 positions without truncation.</p>
      <label className="docattentionlens-checkbox"><input type="checkbox" checked={session.captureJointAnswer} disabled={!record.reference_answer.trim()} onChange={e=>session.setCaptureJointAnswer(e.target.checked)}/>Also Page + question with the reference answer: “{record.reference_answer.trim()||'none for this case'}”</label>
      <p>Adds a second Page + question pass whose prompt ends with “Answer: {record.reference_answer.trim()}”. The answer comes from the case annotations and is shown wherever this reading is used. It never enters the Question or Page readings or the retrieval score.</p></>}
    <h2>Retrieval evidence</h2><p>{session.captureAttention?`${model} internal language attention and signed similarity.`:`Signed ${model} similarity.`} Query and page vectors are encoded separately.</p>
    <label>Saved capture<select aria-label="Saved retrieval capture" value={state.retrieval_capture_id ?? ''} onChange={e => session.choose(session.records.data?.items.find(c => c.capture_id === e.target.value))}>
      <option value="">Choose saved evidence</option>{capture && !session.records.data?.items.some(c => c.capture_id === capture.capture_id) && <option value={capture.capture_id}>Saved selection · {capture.capture_id.slice(-8)}</option>}
      {session.records.data?.items.map(c => <option key={c.capture_id} value={c.capture_id}>{new Date(c.created_at).toLocaleString()} · {c.pages.length} pages · {c.capture_id.slice(-8)}</option>)}
    </select></label>
    {session.records.data?.next_cursor && <button className="docattentionlens-button" onClick={() => session.setCursor(session.records.data?.next_cursor ?? undefined)}>Older captures</button>}{session.cursor && <button className="docattentionlens-button" onClick={() => session.setCursor(undefined)}>Latest captures</button>}
    {session.records.error && <Message error>{session.records.error}</Message>}
    <details open><summary>Choose candidate pages and run {model}</summary>
      <p>{session.plan.data?.reason ?? session.plan.error ?? 'Checking GPU worker…'}</p><p>Compare up to eight explicit case pages. Evaluation labels and OCR are excluded.</p>
      <div className="docattentionlens-candidate-picker">{record.page_ids.map(id => <label className="docattentionlens-checkbox" key={id}><input type="checkbox" checked={session.candidates.includes(id)} disabled={!session.candidates.includes(id) && session.candidates.length >= 8} onChange={e => session.setCandidates(old => e.target.checked ? [...old, id] : old.filter(p => p !== id))}/>Page {document.page_ids.indexOf(id) + 1}</label>)}</div>
      {session.captureAttention&&<label>Document attention page<select aria-label="Acquire document attention page" value={session.candidates.includes(session.attentionPage)?session.attentionPage:session.candidates[0]??''} onChange={e=>session.setAttentionPage(e.target.value)}>{session.candidates.map(id=><option key={id} value={id}>Page {document.page_ids.indexOf(id)+1}</option>)}</select></label>}
      <button className="docattentionlens-button docattentionlens-primary" disabled={!(session.captureAttention?session.plan.data?.attention_available:session.plan.data?.available) || !session.candidates.length || session.busy || !!session.jobId} onClick={() => void session.run()}>{(session.busy || !!session.jobId) && <Busy/>}Run {session.plan.data?.model_name ?? 'retriever'} on {session.candidates.length} {session.candidates.length === 1 ? 'page' : 'pages'}</button>
      <details><summary>Execution budget</summary><p>{session.plan.data?.execution} · {(session.plan.data?.required_free_gpu_mib??8192)/1024} GiB free GPU memory required. About {((session.plan.data?.estimated_retained_bytes_per_page ?? 700000) / 1e6).toFixed(1)} MB vectors per page.{session.captureAttention?` Attention retains Question, Page and the diagnostic Page + question reading${session.jointAnswer?' (also with the reference answer)':''}, up to ${session.plan.data?Math.round(session.plan.data.maximum_retained_attention_bytes/1e6)+' MB':'the checked capture budget'}; free artifact storage is checked before launch.`:''}</p><p>A new run re-encodes inputs; fixed-vector diagnostics reuse the capture.</p></details>
    </details>
  </section>;
}

export function RetrievalDetails({ capture, row, state, elements, update, view }: { capture?: Capture; row?: Row; state: View; elements: Element[]; update: UpdateView; view: 'winners' | 'contributions' }) {
  const current = capture?.pages.find(p => p.page_id === state.page_id);
  const labels = new Map(elements.map(e => [e.element_id, `Element ${e.native_id} · ${e.element_type}`]));
  const [removal, setRemoval] = useState<components['schemas']['RemovalResult']>();
  const [busy, setBusy] = useState(false);
  const [error, setError] = useState<string>();
  useEffect(() => { setRemoval(undefined); setError(undefined); }, [capture?.capture_id, state.page_id, state.query_position]);
  async function removeWinner() {
    if (!capture || !row) return;
    setBusy(true); setError(undefined);
    try { setRemoval(unwrap(await api.POST('/api/v1/retrieval/captures/{capture_id}/fixed-vector-removal', { params: { path: { capture_id: capture.capture_id } }, body: { page_id: row.page_id, removed_position_indices: [row.winner_index] } }))); }
    catch (reason) { setError(errorMessage(reason)); } finally { setBusy(false); }
  }
  if (!capture) return <p>Choose or acquire retrieval evidence to inspect a ledger.</p>;
  if (!current) return <p>This page is outside the captured candidate set. Select a retrieved page.</p>;
  return <div className="docattentionlens-ledger-content">
    {view === 'winners' ? <>
      <div className="docattentionlens-detail-summary"><p>Each eligible query position contributes its signed maximum, including prefixes and augmentation.</p><button className="docattentionlens-button" disabled={busy || !row} onClick={() => void removeWinner()}>Rescore without this winner vector</button></div>
      <table><thead><tr><th>Query position</th><th>Winner position · role</th><th>Contribution</th></tr></thead><tbody>{capture.query_positions.filter(p => p.eligible).map((p, i) => <tr key={p.index} aria-selected={p.index === state.query_position}><td><button className="docattentionlens-button" aria-pressed={p.index === state.query_position} onClick={() => update(s => ({ ...s, query_position: p.index }))}>{p.index} · {p.piece}</button></td><td>{current.winner_indices[i]} · {current.positions[current.winner_indices[i]].role}</td><td>{current.query_maxima[i].toFixed(6)}</td></tr>)}</tbody><tfoot><tr><th colSpan={2}>Full score</th><td>{current.score.toFixed(6)}</td></tr></tfoot></table>
    </> : <>
      <p>Full-query signed MaxSim contributions · exact cell partition, equal overlap sharing. Flagged geometry stays in residuals.</p>
      <table><thead><tr><th>Element / residual</th><th>Allocated score</th></tr></thead><tbody>{Object.entries(current.contributions).sort((a,b) => b[1] - a[1]).map(([id, value]) => <tr key={id} aria-selected={id === state.selected_element_id}><td>{labels.has(id) ? <button className="docattentionlens-button" aria-pressed={id === state.selected_element_id} onClick={() => update(s => ({ ...s, selected_element_id: id }))}>{labels.get(id)}</button> : id.replaceAll('_', ' ')}</td><td>{value.toFixed(6)}</td></tr>)}</tbody><tfoot><tr><th>Total allocated</th><td data-testid="allocated-total">{Object.values(current.contributions).reduce((a,b) => a+b, 0).toFixed(6)}</td></tr></tfoot></table>
    </>}
    {removal && <Message>Frozen-vector score {removal.score.toFixed(6)} · change {removal.score_delta.toFixed(6)}. {removal.interpretation} Diagnostic is temporary; baseline capture remains saved.</Message>}{error && <Message error>{error}</Message>}
    <Link to="/toolbox/retrieval.maxsim">Method details</Link>
  </div>;
}
