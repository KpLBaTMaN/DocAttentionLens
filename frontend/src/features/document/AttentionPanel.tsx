import { type DisplaySettings } from '../../api/client';
import type { Generation } from './GenerationPanel';
import type { Capture } from './RetrievalPanel';
import type { RetrieverAttention } from './useRetrieverAttention';
import type { View } from './useAcquisition';
import { MethodParameters } from './MethodParameters';
import { displayPiece } from './TokenStrip';
import { defaultSource, lastAnswerToken, lastQuestionWord } from './useRetrieverAttention';
import { EncodingReadings } from './EncodingReadings';
import { provenanceLabel } from './perspective';
import { attentionInputLabel } from './documentGrouping';
import type { MethodSelection } from './useMethodSelection';

type SourceProps = {
  /** Method selection shared with the method bar and Details › Method. */
  selection: MethodSelection;
  generation?: Generation; retrieval?: Capture; retriever: RetrieverAttention; state: View;
  onOpenEvidence: (signal: 'attention' | 'retrieval') => void;
  /** Opens acquisition for the Page + question pass with the reference answer. */
  onAcquireAnswer: () => void;
  /** Shows the full prompt in the source line above the page. */
  onShowPrompt?: () => void;
  /** Opens Details › Run, where saved captures are chosen. */
  onChooseCapture: () => void;
  /** Choosing a generator source shows its row, as clicking its token does. */
  onSource?: (change: (state: View) => View) => void;
};

/**
 * Settings that change the selected calculation (`MethodParameters`), for the shown capture. Used in the
 * Details panel and in the method picker's preview of the current method.
 */
export function MethodSettings({selection, generation, state}: {selection: MethodSelection; generation?: Generation; state: View}) {
  const {selected, isRetrieval, encoding, partialRollout, updateAttention} = selection;
  if (!selected?.available) return null;
  return isRetrieval
    ? encoding ? <MethodParameters method={selected.method} hiddenParameters={partialRollout ? ['start_layer'] : []} domain="encoding" layerCount={encoding.layer_count} headCount={encoding.query_head_count} maxTargets={encoding.attention_mask.filter((active, i) => active && (!encoding.causal || i <= (state.retrieval_source_index ?? 0))).length}
      parameters={state.retrieval_attention_parameters ?? {}} onChange={next => updateAttention(s => ({...s, retrieval_attention_parameters: next(s.retrieval_attention_parameters ?? {}), retrieval_comparison_index: 0}))}/> : null
    : generation ? <MethodParameters method={selected.method} hiddenParameters={partialRollout ? ['start_layer'] : []} layerCount={generation.layer_count} headCount={generation.query_head_count} maxTargets={generation.source_rows?.find(r => r.source_index === state.attention_source_index)?.target_count ?? generation.steps.find(s => s.step === (state.attention_step ?? 0))?.target_count}
      parameters={state.attention_parameters ?? {}} onChange={next => updateAttention(s => ({...s, attention_parameters: next(s.attention_parameters ?? {})}))}/> : null;
}

/** What the calculation reads, at the top of Details › Method: the source row or the retriever reading. */
export function AttentionSource({selection, generation, retrieval, retriever, state, onOpenEvidence, onAcquireAnswer, onShowPrompt, onChooseCapture, onSource}: SourceProps) {
  const {isRetrieval, encodings, encoding, updateAttention} = selection;
  if (!isRetrieval) {
    // One source list: a prediction step's row, or a captured question-token row, which takes precedence.
    const sourceValue = state.attention_source_index != null && generation?.source_rows?.some(r => r.source_index === state.attention_source_index)
      ? `token:${state.attention_source_index}` : `step:${state.attention_step ?? 0}`;
    const chooseSource = (value: string) => {const [kind, index] = value.split(':'); const n = Number(index);
      (onSource ?? updateAttention)(s => kind === 'token' ? {...s, attention_source_index: n} : {...s, attention_step: n, attention_source_index: null});};
    return <section className="docattentionlens-attention-selection" aria-label="Attention capture selection">
      {generation ? <label>Source<select aria-label="Attention source" value={sourceValue} onChange={e => chooseSource(e.target.value)}>
          <optgroup label="Answer predictions">{generation.steps.map(step => <option key={step.step} value={`step:${step.step}`}>Step {step.step} → “{displayPiece(step.predicted_piece)}” · {step.pass_kind}</option>)}</optgroup>
          {!!generation.source_rows?.length && <optgroup label="Question tokens">{generation.source_rows.map(r => <option key={r.source_index} value={`token:${r.source_index}`}>“{displayPiece(r.piece)}” · position {r.source_index}</option>)}</optgroup>}
        </select></label>
      : <div className="docattentionlens-empty-capture"><p>Choose a saved attention capture to explore prediction steps, layers and heads. Retrieval captures store similarity scores; attention weights come from a generator capture.</p>
        <div className="docattentionlens-actions"><button className="docattentionlens-button" onClick={onChooseCapture}>Choose capture</button><button className="docattentionlens-button" onClick={() => onOpenEvidence('attention')}>Acquire attention</button></div></div>}
    </section>;
  }
  // A retriever has no prediction steps: a reading (encoding) and its source position select the attention row.
  // Changing reading keeps layer and head, and starts on that reading's default source.
  const chooseReading = (next: NonNullable<typeof encoding>) => updateAttention(s => ({...s, retrieval_encoding_id: next.encoding_id,
    retrieval_source_index: defaultSource(next), retrieval_comparison_index: 0, attention_target_index: null, attention_direction: 'row',
    ...(next.page_id ? {page_id: next.page_id} : {})}));
  // Source options in plain words, grouped as the source line is: question words, reference answer, page patches, prompt.
  const textRole = (role: string) => role === 'query_text' || role === 'answer_text';
  const sourceGroups: [string, (p: {role: string; bbox?: number[] | null}) => boolean][] = [
    ['Question words', p => p.role === 'query_text'], ['Reference answer', p => p.role === 'answer_text'],
    ['Page patches', p => !!p.bbox], ['Prompt', p => !textRole(p.role) && !p.bbox]];
  // "Last answer token" reads the position that predicts the answer's final token.
  const answerTarget = encoding?.pass_kind === 'joint' ? lastAnswerToken(encoding) : undefined;
  const questionWord = encoding?.pass_kind === 'joint' ? lastQuestionWord(encoding) : undefined;
  const sourceLabel = (p: {index: number; piece: string; role: string; bbox?: number[] | null}) =>
    p.bbox ? `Patch ${p.index}` : `${p.role === 'query_text' ? 'Question word' : p.role === 'answer_text' ? 'Answer' : 'Prompt'} · “${displayPiece(p.piece)}” (${p.index})`
      + (p.index === answerTarget?.source ? ` · predicts last answer token “${displayPiece(answerTarget.target.piece ?? '')}”` : '');
  const chooseSourcePosition = (index: number) => updateAttention(s => ({...s, retrieval_source_index: index}));
  return <section className="docattentionlens-attention-selection" aria-label="Attention capture selection">
    <EncodingReadings encodings={encodings} selected={encoding} pageId={state.page_id}
      onAcquireAnswer={onAcquireAnswer} onChoose={chooseReading} onShowPrompt={onShowPrompt}/>
    {retrieval && encodings.length ? <>
      <label>Source<select aria-label="Attention source position" value={state.retrieval_source_index ?? 0} onChange={e => chooseSourcePosition(Number(e.target.value))}>
        {sourceGroups.map(([group, member]) => {
          const options = (retriever.positions ?? []).filter(member);
          return !!options.length && <optgroup key={group} label={group}>
            {options.map(p => <option key={p.index} value={p.index} disabled={!p.eligible}>{sourceLabel(p)}</option>)}</optgroup>;})}
      </select></label>
      {answerTarget && <div className="docattentionlens-source-chips" role="group" aria-label="Source shortcuts">
        {questionWord !== undefined && <button type="button" className="docattentionlens-chip" aria-pressed={state.retrieval_source_index === questionWord} onClick={() => chooseSourcePosition(questionWord)}>Last question word</button>}
        <button type="button" className="docattentionlens-chip" aria-pressed={state.retrieval_source_index === answerTarget.source} title={`The position whose output predicts the answer's final token, “${displayPiece(answerTarget.target.piece ?? '')}”`}
          onClick={() => chooseSourcePosition(answerTarget.source)}>Last answer token</button></div>}
    </> : <div className="docattentionlens-empty-capture"><p>{retrieval ? 'This capture stores embeddings and MaxSim scores only. Saved retriever encodings enable heads, layers and source positions.' : 'No retrieval capture selected.'}</p>
      <button className="docattentionlens-button" onClick={() => onOpenEvidence('retrieval')}>Acquire retriever attention</button></div>}
  </section>;
}

/** Whether the chosen method can run on this capture, in Details › Method above the method list. */
export function MethodReadiness({selection, isGeneration, generation, preparedError}: {selection: MethodSelection; isGeneration: boolean; generation?: Generation; preparedError?: string}) {
  const {isRetrieval, catalogueError, selected, perspective, grouped, input, status, structure, updateAttention} = selection;
  // A generation investigation showing its supporting retrieval scores returns to the generator's attention.
  const showAttention = !isRetrieval && !isGeneration && !!generation;
  const ready = selected?.available && !status && perspective !== 'document_aware';
  if (!grouped && !catalogueError && !(isRetrieval && preparedError) && (!selected || ready) && !showAttention) return null;
  return <div className="docattentionlens-method-readiness">
    {grouped && <p className="docattentionlens-grouping-note" data-testid="document-grouping">{input ? attentionInputLabel(input.method) : 'Attention'} → Element totals</p>}
    {catalogueError && <p role="alert">{catalogueError}</p>}
    {isRetrieval && preparedError && <p role="alert">Document structure could not be loaded: {preparedError}</p>}
    {selected && <>
      {status ? <p role="status" className={`docattentionlens-structure-status is-${status.state}`} data-testid="structure-status">{status.text}</p>
        : !selected.available && <p role="status">{selected.reasons.join(' ')}</p>}
      {perspective === 'document_aware' && <p className="docattentionlens-structure-provenance" data-testid="structure-provenance">{provenanceLabel(structure, preparedError)}</p>}</>}
    {showAttention && <div className="docattentionlens-attention-source"><p>Viewing retrieval similarity. Attention controls use saved generator weights.</p><button className="docattentionlens-button" onClick={() => updateAttention(s => s)}>Show captured attention</button></div>}
  </div>;
}

/** Details › Method: the method's weights on patches and tokens (None) or added up per document element (Element totals). */
export function MethodGrouping({selection}: {selection: MethodSelection}) {
  const elementTotals = selection.elementAnalysis;
  return <div className="docattentionlens-method-grouping" role="radiogroup" aria-label="Grouping"
      onKeyDown={e=>{if(e.key!=='ArrowLeft'&&e.key!=='ArrowRight')return;e.preventDefault();
        if(selection.grouped||elementTotals?.available)selection.choosePerspective(selection.grouped?'general':'document_aware');}}>
      <span className="docattentionlens-method-grouping-label">Grouping</span>
      <button type="button" role="radio" aria-checked={!selection.grouped} tabIndex={selection.grouped?-1:0} title="Show the method's weights on patches and tokens"
        onClick={()=>{if(selection.grouped)selection.choosePerspective('general');}}>None</button>
      <button type="button" role="radio" aria-checked={selection.grouped} tabIndex={selection.grouped?0:-1} disabled={!elementTotals?.available}
        title={elementTotals?.available?"Add up the method's weights per document element":elementTotals?selection.reason(elementTotals):'Element totals are not offered for this capture'}
        onClick={()=>{if(!selection.grouped)selection.choosePerspective('document_aware');}}>Element totals</button>
    </div>;
}

/** Which grouped comparison the Result tab reads, while the result is grouped into element totals. */
export function GroupedComparison({selection, comparisons, state, settings, onView}: {
  selection: MethodSelection; comparisons?: {label: string}[]; state: View; settings: DisplaySettings; onView: (next: DisplaySettings) => void;
}) {
  const {isRetrieval, grouped, updateAttention} = selection;
  if (!grouped || !comparisons?.length) return null;
  return <label className="docattentionlens-grouped-comparison">Grouped comparison<select aria-label="Grouped comparison"
    value={isRetrieval ? Math.min(state.retrieval_comparison_index ?? 0, comparisons.length - 1) : Math.min(settings.attention_grouped_comparison_index ?? 0, comparisons.length - 1)}
    onChange={e => isRetrieval ? updateAttention(s => ({...s, retrieval_comparison_index: Number(e.target.value)}))
      : onView({...settings, attention_grouped_comparison_index: Number(e.target.value)})}>
    {comparisons.map((c, i) => <option key={i} value={i}>{c.label}</option>)}</select></label>;
}
