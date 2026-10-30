import type { Case } from '../../api/client';
import type { components } from '../../api/generated/schema';
import { Message } from '../../app/App';
import { Busy } from '../../app/Busy';
import { Link } from 'react-router-dom';
import type { Capture } from './RetrievalPanel';
import type { useGenerationAcquisition, View, UpdateView } from './useAcquisition';
import { FULL_ATTENTION_ENABLES, FULL_ATTENTION_KEEPS, fullAttentionSteps } from '../attention-tool/fullAttentionEstimate';
import { answerRowCoverage } from '../attention-tool/ModelCapabilities';

export type Generation = components['schemas']['GenerationCapture'];
export type Attention = components['schemas']['AttentionRow'];
type Session = ReturnType<typeof useGenerationAcquisition>;

export function GenerationPanel({record,retrieval,capture,state,session,update}:{record:Case;retrieval?:Capture;capture?:Generation;state:View;session:Session;update:UpdateView}) {
  return <section aria-label="Generator acquisition"><h2>Generator evidence</h2><p>Quick acquisition uses the pinned Qwen2.5-VL-3B generator with its default instruction. For model selection and a complete context review, <Link to={`/attention-tool?case=${record.case_id}&configure=1`}>configure a new investigation</Link>.</p>
    <label>Saved answer and attention<select aria-label="Saved generator capture" value={state.generation_capture_id??''} onChange={e=>{const c=session.records.data?.items.find(c=>c.capture_id===e.target.value);if(c)session.choose(c);else update(s=>({...s,generation_capture_id:null}));}}><option value="">Choose saved generator evidence</option>{capture&&!session.records.data?.items.some(c=>c.capture_id===capture.capture_id)&&<option value={capture.capture_id}>Saved selection · {capture.capture_id.slice(-8)}</option>}{session.records.data?.items.map(c=><option key={c.capture_id} value={c.capture_id}>{new Date(c.created_at).toLocaleString()} · {c.context.pages.length} supplied pages · {c.capture_id.slice(-8)}</option>)}</select></label>
    {session.records.data?.next_cursor&&<button className="docattentionlens-button" onClick={()=>session.setCursor(session.records.data?.next_cursor??undefined)}>Older answers</button>}{session.cursor&&<button className="docattentionlens-button" onClick={()=>session.setCursor(undefined)}>Latest answers</button>}{session.records.error&&<Message error>{session.records.error}</Message>}
    <details open><summary>Choose context pages and generate an answer</summary>
      {!retrieval?<Message>Supply document pages directly or choose retrieved context in <Link to={`/attention-tool?case=${record.case_id}&configure=1`}>investigation setup</Link>.</Message>:<>
        <p>Choose up to two pages. Selection order is the supplied image order.</p>
        <div className="docattentionlens-candidate-picker">{[...retrieval.pages].sort((a,b)=>b.score-a.score).map(p=><label className="docattentionlens-checkbox" key={p.page_id}><input aria-label={`Supply page ${p.page_number}`} type="checkbox" checked={session.selected.includes(p.page_id)} disabled={!session.selected.includes(p.page_id)&&session.selected.length>=2} onChange={e=>session.setSelected(s=>e.target.checked?[...s,p.page_id]:s.filter(id=>id!==p.page_id))}/>Page {p.page_number} · score {p.score.toFixed(4)}</label>)}</div>
        <p>Draft context order: {session.selected.map(id=>retrieval.pages.find(p=>p.page_id===id)?.page_number).join(' → ')||'No pages'}. {record.query}</p>
        <label className="docattentionlens-checkbox docattentionlens-setup-option"><input type="checkbox" checked={session.fullAttention} onChange={e=>session.setFullAttention(e.target.checked)}/><span><strong>Save full attention matrices</strong> (optional)<small>{FULL_ATTENTION_ENABLES} {FULL_ATTENTION_KEEPS} Estimated extra storage: {session.fullAttentionEstimate?`${session.fullAttentionEstimate.label} for ${session.model?.name??'this model'} with ${session.selected.length||1} page${(session.selected.length||1)===1?'':'s'}`:'checking model dimensions…'}.</small></span></label>
        <button className="docattentionlens-button docattentionlens-primary" disabled={!session.plan.data?.available||!session.selected.length||session.busy||!!session.jobId} onClick={()=>void session.run()}>{(session.busy||!!session.jobId)&&<Busy/>}Generate answer and capture attention</button>
      </>}
      <p>{session.plan.error??session.plan.data?.reason??'Checking generator availability…'}</p><p data-testid="generator-limits">{generatorLimits(session)}</p>
      <details><summary>Execution budget</summary><p>{session.plan.data?.execution} · 12 GiB free GPU memory required.</p><p>Evaluation labels and OCR are excluded. Vision-tower attention is not captured; earlier prefill rows are kept only with full attention matrices.</p></details>
    </details>
  </section>;
}

/**
 * The quick generator's limits, read from the model catalogue (or the run plan): output length, which answer tokens
 * keep their attention row (the first `captured_steps`, or every answer token with full attention), layers and heads.
 */
function generatorLimits(session: Session) {
  const plan = session.plan.data;
  const limits: Record<string, number | undefined> = {captured_steps: plan?.captured_steps, max_new_tokens: plan?.max_new_tokens,
    full_attention_captured_steps: plan?.full_attention_captured_steps, ...session.model?.capabilities.limits};
  const language = session.model?.architecture.components?.find(c => c.name === 'Language');
  const layers = language?.layers ?? plan?.layers, heads = language?.query_heads ?? plan?.query_heads;
  const rows = answerRowCoverage(limits, session.fullAttention, (session.model && fullAttentionSteps(session.model)) ?? limits.full_attention_captured_steps ?? limits.max_new_tokens);
  const parts = [limits.max_new_tokens && `Up to ${limits.max_new_tokens} output tokens`, rows && (limits.max_new_tokens ? rows[0].toLowerCase() + rows.slice(1) : rows),
    layers && heads && `${layers} layers, ${heads} query heads`].filter(Boolean);
  const pagePositions = limits.image_positions_per_page;
  return `${parts.length ? `${parts.join('; ')}. ` : ''}${pagePositions ? `Each page resizes to at most ${pagePositions} merged image positions. ` : ''}Over-budget prompts are rejected.`;
}

/** Full answer and its prediction tokens for Details › Result; selecting a token selects that prediction step. */
export function AnswerTokens({capture,state,update}:{capture?:Generation;state:View;update:UpdateView}) {
  if(!capture)return <p>Open saved evidence or generate an answer using chosen context pages.</p>;
  return <section className="docattentionlens-answer-tokens" aria-label="Generated output"><h3>Full answer</h3><p className="docattentionlens-full-answer">{capture.answer||'(No visible output text)'}</p><h3>Captured prediction tokens</h3><div className="docattentionlens-token-strip">{capture.steps.map(step=><button className="docattentionlens-button" key={step.step} aria-label={`Inspect prediction ${step.step}: ${step.predicted_piece}`} aria-pressed={state.attention_step===step.step} onClick={()=>update(s=>({...s,attention_step:step.step}))}><small>{step.step}</small>{step.predicted_piece}</button>)}</div>
    <p>{capture.steps.length} / {capture.output_token_ids.length} prediction steps captured{capture.steps.length<capture.output_token_ids.length?' · later rows unavailable': ' · all output steps retained'}.</p></section>;
}

/** Complete causal target row for Details › Result. Element allocations use the linked ElementSummary. */
export function GenerationDetails({capture,row}:{capture?:Generation;row?:Attention}) {
  if(!capture||!row) return <p>No attention row for this selection. Numerical and comparison results appear with the document; open saved evidence if no capture is selected.</p>;
  return <div className="docattentionlens-ledger-content"><p>Complete causal target row · {row.target_count} positions · {row.quantity.replaceAll('_',' ')}.</p><table><thead><tr><th>Position</th><th>Token</th><th>Role</th><th>Weight</th></tr></thead><tbody>{capture.positions.slice(0,row.target_count).map(p=><tr key={p.index}><td>{p.index}</td><td>{p.piece}</td><td>{p.role}</td><td>{row.values[p.index].toFixed(8)}</td></tr>)}</tbody></table></div>;
}
