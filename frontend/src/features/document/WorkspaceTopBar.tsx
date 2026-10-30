import { useContext, type ReactNode } from 'react';
import { ThemeContext } from '../../app/App';
import { Link } from 'react-router-dom';
import { ChevronDown, More } from '../../app/icons';
import { MenuButton, MenuGroup, MenuItem, PopoverButton } from './WorkspaceMenus';
import type { components } from '../../api/generated/schema';
import { INPUT_GROUNDING, type GroundingKind } from '../inputs/grounding';

/** A table grounding case: its accepted answers, the server's exact-match check and source. */
export type TableCaseDetails = {acceptedAnswers: string[]; check?: components['schemas']['AnswerCheck']; source: string};

type Props = {
  documentName: string; question: string;
  /** Case details shown in the breadcrumb's popover. */
  details: {answer?: string; answerNote?: string; referenceAnswer: string; wording: string; modelLine: string; grounding?: GroundingKind; table?: TableCaseDetails};
  /** Set while a generation investigation shows its supporting retrieval scores: returns to generator attention. */
  onBackToGeneration?: () => void;
  /** The model and capture chip (`CaptureMenu`), which names the analysed role. On phones it sits in the case popover instead. */
  modelChip: ReactNode; wide: boolean;
  dirty: boolean; revision?: number; saving: boolean; canSave: boolean; onSave: () => void;
  onNotes: () => void; onDisplay: () => void; onStudies: () => void; onInterventions: () => void; onExperiment: () => void; onProvenance: () => void;
  onPalette: () => void; onShortcuts: () => void;
};

/**
 * The workspace's part of the app top bar: the case (document / question, with its details), the model
 * chip (role, model and capture), save state, Save and the investigation menu.
 */
export function WorkspaceTopBar({documentName, question, details, onBackToGeneration, modelChip, wide, dirty, revision, saving, canSave, onSave,
  onNotes, onDisplay, onStudies, onInterventions, onExperiment, onProvenance, onPalette, onShortcuts}: Props) {
  const {theme, setTheme} = useContext(ThemeContext);
  const state = revision === undefined ? 'New' : `Revision ${revision}`;
  const table = details.table;
  const grounding = details.grounding ? INPUT_GROUNDING[details.grounding] : undefined;
  return <div className="docattentionlens-ws-bar">
    <PopoverButton className="docattentionlens-crumb" buttonClassName="docattentionlens-quiet docattentionlens-crumb-button" panelLabel="Case details" title={`${documentName} / ${question}`}
      buttonContent={<><span className="docattentionlens-crumb-doc">{documentName}</span><span className="docattentionlens-crumb-sep" aria-hidden="true">/</span>
        <span className="docattentionlens-crumb-question" data-testid="frozen-question" title={question}>{question}</span><ChevronDown size={14}/></>}>
      {close => <div className="docattentionlens-case-details">
        <dl>
          <dt>Question</dt><dd>{question}</dd>
          {details.answer !== undefined && <><dt>Model answer</dt><dd><span data-testid="generated-answer">{details.answer || '(No visible output text)'}</span><small>{details.answerNote}</small>
            {table?.check && <small data-testid="answer-check">{table.check.verdict === 'correct' ? `Matches the accepted answer “${table.check.matched_answer}”` : 'Does not match an accepted answer'} · exact match after normalising case and spacing; agreement, not semantic correctness</small>}</dd></>}
          {table ? <><dt>Accepted answers</dt><dd><span data-testid="reference-answer">{table.acceptedAnswers.join(' · ')}</span><small>From the table question annotations · evaluation only</small></dd></>
            : <><dt>Reference answer</dt><dd><span data-testid="reference-answer">{details.referenceAnswer || 'None for this case'}</span><small>From the case annotations</small></dd></>}
          <dt>Document</dt><dd>{documentName}{table && <small>{table.source}</small>}</dd>
          {grounding && <><dt>Grounding</dt><dd data-testid="case-grounding">{grounding.label}<small>{grounding.description}</small></dd></>}
          <dt>Query</dt><dd>{details.wording}</dd>
          <dt>Model</dt><dd>{details.modelLine}</dd>
        </dl>
        {!wide && <div className="docattentionlens-case-model">{modelChip}</div>}
        <div className="docattentionlens-actions"><Link className="docattentionlens-button" to="/artifacts/inputs" onClick={() => close(false)}>Choose another case</Link></div>
      </div>}
    </PopoverButton>
    {wide && <div className="docattentionlens-model-chip">{modelChip}</div>}
    {onBackToGeneration && <div className="docattentionlens-evidence-pill" role="status">
      <span className="docattentionlens-evidence-pill-label">Supporting retrieval scores</span>
      <button type="button" className="docattentionlens-quiet" onClick={onBackToGeneration}>← Back to generation</button></div>}
    <span className="docattentionlens-ws-spacer"/>
    <span className="docattentionlens-save-state" data-dirty={dirty} title={dirty ? 'Unsaved changes' : 'All changes saved'}>
      {dirty && <span className="docattentionlens-unsaved-dot" aria-hidden="true"/>}<span className="docattentionlens-save-text">{state}{dirty ? ' · unsaved' : ''}</span></span>
    <button type="button" className="docattentionlens-button docattentionlens-primary docattentionlens-save" aria-label="Save investigation" title="Save investigation (Ctrl+S)" disabled={saving || !canSave} onClick={onSave}>{saving ? 'Saving…' : 'Save'}</button>
    <MenuButton label="Investigation menu" buttonClassName="docattentionlens-icon-button" buttonContent={<More/>} testId="investigation-menu">
      {close => <>
        <MenuItem onSelect={() => {close(false); onNotes();}}>Rename and notes</MenuItem>
        <MenuItem onSelect={() => {close(false); onDisplay();}}>Display settings</MenuItem>
        <MenuGroup label="Experiments">
          <MenuItem onSelect={() => {close(false); onStudies();}}>Document studies</MenuItem>
          <MenuItem onSelect={() => {close(false); onInterventions();}}>Attention interventions</MenuItem>
          <MenuItem onSelect={() => {close(false); onExperiment();}}>Layout experiment</MenuItem>
        </MenuGroup>
        <MenuItem onSelect={() => {close(false); onProvenance();}}>Provenance and receipt</MenuItem>
        <Link role="menuitem" tabIndex={-1} className="docattentionlens-menu-item" to="/artifacts/investigations" onClick={() => close(false)}>Saved investigations</Link>
        <MenuGroup label="Appearance">
          <MenuItem checked={theme !== 'light'} onSelect={() => {close(); setTheme('dark');}}>Charcoal</MenuItem>
          <MenuItem checked={theme === 'light'} onSelect={() => {close(); setTheme('light');}}>White</MenuItem>
        </MenuGroup>
        <MenuItem onSelect={() => {close(false); onPalette();}}>Command palette<kbd className="docattentionlens-menu-key" aria-hidden="true">Ctrl K</kbd></MenuItem>
        <MenuItem onSelect={() => {close(false); onShortcuts();}}>Keyboard shortcuts<kbd className="docattentionlens-menu-key" aria-hidden="true">?</kbd></MenuItem>
      </>}
    </MenuButton>
  </div>;
}
