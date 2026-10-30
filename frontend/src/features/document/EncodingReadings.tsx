import { useId } from 'react';
import type { components } from '../../api/generated/schema';

type Encoding = components['schemas']['EncodingAttention'];

const READINGS: {kind: Encoding['pass_kind']; label: string; description: string}[] = [
  {kind: 'query', label: 'Question', description: 'The question read on its own. Question and template tokens attend to each other.'},
  {kind: 'document', label: 'Page', description: 'The page read on its own, as for retrieval. Patches attend to patches and the prompt.'},
  {kind: 'joint', label: 'Page + question', description: 'Page and question read together, so question words can attend to the page. Excluded from the retrieval score.'},
];
const WITH_ANSWER = 'Page, question and the reference answer read together. The answer is a case annotation added to the prompt; this diagnostic reading is excluded from the retrieval score.';

/** The capture's encoding of a reading: a page reading prefers the page on screen, a joint reading its question-only pass. */
function encodingFor(encodings: Encoding[], kind: Encoding['pass_kind'], pageId?: string | null) {
  const same = encodings.filter(e => e.pass_kind === kind).sort((a, b) => Number(!!a.answer_source) - Number(!!b.answer_source));
  return same.find(e => kind === 'query' || e.page_id === pageId) ?? same[0];
}

/** What the retriever read: Question, Page, or the diagnostic Page + question pass. */
export function EncodingReadings({encodings, selected, pageId, onChoose, onAcquireAnswer, onShowPrompt}: {
  encodings: Encoding[]; selected?: Encoding; pageId?: string | null;
  onChoose: (encoding: Encoding) => void;
  /** Opens acquisition with the Page + question pass that carries the reference answer. */
  onAcquireAnswer: () => void;
  /** Opens the token row above the page on its Prompt view. */
  onShowPrompt?: () => void;
}) {
  const hint = useId();
  const readings = READINGS.map(reading => ({...reading, encoding: encodingFor(encodings, reading.kind, pageId)}));
  const joint = selected?.pass_kind === 'joint';
  const current = readings.find(r => r.kind === selected?.pass_kind);
  const variants = joint ? encodings.filter(e => e.pass_kind === 'joint' && e.page_id === selected.page_id) : [];
  const plain = variants.find(e => !e.answer_source), answered = variants.find(e => !!e.answer_source);
  return <fieldset className="docattentionlens-readings" aria-describedby={hint}>
    <legend>What the model read</legend>
    <div className="docattentionlens-reading-buttons">{readings.map(r =>
      <button key={r.kind} type="button" className="docattentionlens-reading-button" disabled={!r.encoding}
        title={r.encoding ? undefined : `${r.label} is not saved in this older capture; new attention captures include it`}
        aria-pressed={!!r.encoding && selected?.pass_kind === r.kind} onClick={() => r.encoding && selected?.pass_kind !== r.kind && onChoose(r.encoding)}>{r.label}</button>)}
    </div>
    {joint && (plain || answered) && <div className="docattentionlens-prompt-variant">
      <span>Prompt</span>
      <div className="docattentionlens-reading-buttons is-pair" role="group" aria-label="Prompt">
        <button type="button" className="docattentionlens-reading-button" disabled={!plain} aria-pressed={!!plain && selected?.encoding_id === plain.encoding_id} onClick={() => plain && selected?.encoding_id !== plain.encoding_id && onChoose(plain)}>Question only</button>
        <button type="button" className="docattentionlens-reading-button" disabled={!answered} aria-pressed={!!answered && selected?.encoding_id === answered.encoding_id} onClick={() => answered && selected?.encoding_id !== answered.encoding_id && onChoose(answered)}>Question + reference answer</button>
      </div></div>}
    <p id={hint} className={`docattentionlens-reading-note ${joint ? 'is-diagnostic' : ''}`}>
      {joint && <span className="docattentionlens-diagnostic-tag">Diagnostic</span>}
      {(selected?.answer_source ? WITH_ANSWER : current?.description) ?? 'Capture attention to explore how the model read these inputs.'}</p>
    {selected && onShowPrompt && <p className="docattentionlens-reading-prompt"><button type="button" className="docattentionlens-link-button" onClick={onShowPrompt}>Show prompt above the page</button></p>}
    {joint && !answered && <p className="docattentionlens-reading-missing">Question + reference answer isn’t in this capture ·{' '}
      <button type="button" className="docattentionlens-link-button" aria-label="Capture Page + question attention with the reference answer" onClick={onAcquireAnswer}>Capture it</button></p>}
  </fieldset>;
}
