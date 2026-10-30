import { useCallback, useRef } from 'react';
import { Link } from 'react-router-dom';
import { api, unwrap } from '../../api/client';
import { useRemote } from '../../app/remote';
import type { Generation } from './GenerationPanel';
import type { Capture } from './RetrievalPanel';
import { useMenuDismiss } from './WorkspaceControls';
import { formatGigabytes } from '../attention-tool/fullAttentionEstimate';

type Props = {
  /** The shown evidence: generator attention, or the retriever's encodings and scores. */
  isGeneration: boolean; generation?: Generation; retrieval?: Capture;
  captures?: Generation[]; captureError?: string;
  onCapture: (capture?: Generation) => void;
  /** Opens the acquisition drawer for the shown evidence. */
  onAcquire?: () => void;
  /** Investigation setup for a new capture of this case. */
  configureHref?: string;
  /** Details › Run: the same chooser as a section instead of the top bar's chip. */
  inline?: boolean;
};

const plural = (n: number, word: string) => `${n} ${word}${n === 1 ? '' : 's'}`;

/**
 * Which answer tokens of a full-attention capture have an attention row: every one when a step was kept for each
 * output token (generation schema 0.8.0 keeps them all, up to 64), else the first N (0.7.0 kept the first eight).
 */
export function answerTokenCoverage(generation: Pick<Generation, 'steps' | 'output_token_ids'>) {
  return generation.steps.length >= generation.output_token_ids.length ? 'every answer token' : `first ${plural(generation.steps.length, 'answer token')}`;
}

/** The model chip in the top bar: the analysed role, model and capture, with capture switching. */
export function CaptureMenu({isGeneration, generation, retrieval, captures, captureError, onCapture, onAcquire, configureHref, inline = false}: Props) {
  const models = useRemote(useCallback(async () => unwrap(await api.GET('/api/v1/setup/models')), []));
  const provenance = isGeneration ? generation?.provenance : retrieval?.provenance;
  const modelId = typeof provenance?.model_id === 'string' ? provenance.model_id : '';
  const modelName = models.data?.find(m => m.model_id === modelId)?.name ?? (modelId.split('/').pop() || undefined);
  const captureId = isGeneration ? generation?.capture_id : retrieval?.capture_id;
  const label = modelName ? `${modelName} · ${captureId?.slice(-8)}` : isGeneration ? 'No attention capture' : 'No retrieval capture';
  const encodings = retrieval?.attention_encodings ?? [];
  const role = isGeneration ? 'Generation' : 'Retrieval';
  const roleDetail = isGeneration
    ? generation && `answers the question from ${plural(generation.context.pages.length, generation.context.input_source === 'supplied_pages' ? 'supplied page' : 'retrieved page')}`
    : retrieval && `ranks ${plural(retrieval.pages.length, 'candidate page')} by signed MaxSim`;
  const roleRow = <><dt>Role</dt><dd data-testid="capture-role">{role}{roleDetail && ` · ${roleDetail}`}</dd></>;
  const menu = useRef<HTMLDetailsElement>(null);
  const dismiss = useMenuDismiss(menu);
  const content = <>
      {isGeneration ? <>
        {generation && <dl>{roleRow}<dt>Model</dt><dd>{modelId}</dd>
          <dt>Coverage</dt><dd data-testid="capture-coverage">{generation.full_attention ? `Attention rows for ${generation.steps.length < generation.output_token_ids.length ? 'the ' : ''}${answerTokenCoverage(generation)} (${generation.steps.length} of ${generation.output_token_ids.length} captured)` : `${generation.steps.length} captured predictions`}{generation.source_rows?.length ? ` · ${generation.source_rows.length} question-token rows` : ''} · {generation.layer_count} layers · {generation.query_head_count} query heads</dd>
          {generation.full_attention && <><dt>Full attention</dt><dd data-testid="full-attention-coverage">Full attention · {formatGigabytes(generation.full_attention.bytes)} stored · {answerTokenCoverage(generation)} · complete attention rows for the first {generation.full_attention.square_limit} positions ({generation.full_attention.position_count} prefill)</dd></>}</dl>}
        <label>{inline ? 'Attention capture' : 'Saved attention capture'}<select aria-label={inline ? 'Choose attention capture' : 'Saved attention capture'} value={generation?.capture_id ?? ''} onChange={e => onCapture(captures?.find(c => c.capture_id === e.target.value) ?? (generation?.capture_id === e.target.value ? generation : undefined))}>
          <option value="">Choose saved attention evidence</option>
          {generation && !captures?.some(c => c.capture_id === generation.capture_id) && <option value={generation.capture_id}>Saved selection · {generation.capture_id.slice(-8)}</option>}
          {captures?.map(c => <option key={c.capture_id} value={c.capture_id}>{new Date(c.created_at).toLocaleString()} · {plural(c.context.pages.length, 'supplied page')}{c.full_attention ? ' · full attention' : ''} · {c.capture_id.slice(-8)}</option>)}
        </select></label>
        {captureError && <p role="alert">{captureError}</p>}
      </> : retrieval ? <dl>{roleRow}<dt>Model</dt><dd>{modelId} · revision {String(retrieval.provenance.model_revision)}</dd>
        <dt>Capture</dt><dd>{retrieval.capture_id}</dd>
        <dt>Coverage</dt><dd>{encodings.length ? `${encodings.length} saved encodings · ${encodings[0].layer_count} layers · ${encodings[0].query_head_count} query heads · vision tower excluded. Query and document encodings are separate.` : 'Embeddings and MaxSim scores only; no internal retriever attention.'}</dd></dl>
        : <p>No retrieval capture selected.</p>}
      {(onAcquire || configureHref) && <div className="docattentionlens-capture-actions">
        {onAcquire && <button type="button" className="docattentionlens-button" onClick={() => {if (menu.current) menu.current.open = false; onAcquire();}}>Capture more evidence</button>}
        {configureHref && <Link className="docattentionlens-button" to={configureHref}>Configure new capture</Link>}
      </div>}
  </>;
  if (inline) return <section className="docattentionlens-run-capture" aria-label="Model and capture" data-testid="run-capture">{content}</section>;
  return <details ref={menu} className="docattentionlens-capture-menu" data-testid="capture-menu" {...dismiss}>
    <summary aria-label={`Model and capture: ${role} · ${label}`} title="Role, model and capture"><span className="docattentionlens-capture-role">{role}</span><span className="docattentionlens-capture-label">{label}</span></summary>
    <div>{content}</div>
  </details>;
}
