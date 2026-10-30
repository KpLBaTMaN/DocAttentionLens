import { useEffect, useRef, type ReactNode } from 'react';
import { attentionWeightLabel } from './attentionDisplay';
import { More, X } from '../../app/icons';
import { PopoverButton } from './WorkspaceMenus';

/** The source line above the page: the tokens that can be a source, then the page chip. */
export function SourceLine({children, column, prompt, fullPrompt, onFullPrompt, notes = [], selectedKey}: {
  /** The row's token groups and chips. */
  children?: ReactNode;
  /** Column mode: what is named ("What attends to patch 37") and how to return to the row. */
  column?: {label: string; onClear: () => void; clearLabel?: string};
  /** Full prompt (PromptTokens), when the shown reading has one; it is given the line's controls for its header. */
  prompt?: (controls: ReactNode) => ReactNode;
  fullPrompt: boolean; onFullPrompt: (shown: boolean) => void;
  /** Explanatory notes for the row: how to choose a source, and the token scale. */
  notes?: string[];
  /** Changes when the selected source changes, so the row scrolls it into view. */
  selectedKey?: string | number;
}) {
  const row = useRef<HTMLDivElement>(null);
  const showingPrompt = fullPrompt && !!prompt;
  // The row scrolls sideways; keep the selected token in view without scrolling the page.
  useEffect(() => {
    const box = row.current;
    if (!box || showingPrompt) return;
    const reveal = () => {
      const chip = box.querySelector<HTMLElement>('.docattentionlens-token[aria-pressed="true"]');
      if (!chip) return;
      const outer = box.getBoundingClientRect(), own = chip.getBoundingClientRect();
      if (own.left < outer.left) box.scrollLeft -= outer.left - own.left + 24;
      else if (own.right > outer.right) box.scrollLeft += own.right - outer.right + 24;
    };
    reveal();
    const observer = new ResizeObserver(reveal);
    observer.observe(box);
    return () => observer.disconnect();
  }, [selectedKey, showingPrompt]);
  if (!children && !prompt && !column) return null;
  const controls = <>
    {prompt && <span className="docattentionlens-source-view" role="group" aria-label="Source line view">
      <button type="button" className="docattentionlens-button" aria-pressed={!showingPrompt} title="The tokens that can be a source, then the page" onClick={() => onFullPrompt(false)}>Source tokens</button>
      <button type="button" className="docattentionlens-button" aria-pressed={showingPrompt} title="Every token in the order the model read it, template tokens included (P)" onClick={() => onFullPrompt(true)}>Full prompt</button>
    </span>}
    <PopoverButton label="Source line options" title="Source line options" buttonClassName="docattentionlens-icon-button docattentionlens-source-more" align="end"
      panelLabel="Source line options" className="docattentionlens-source-options" buttonContent={<More/>}>
      {() => <div className="docattentionlens-source-options-panel">
        {notes.map((note, i) => <p key={i} className="docattentionlens-source-note">{note}</p>)}
        {prompt && <p className="docattentionlens-source-hint">The full prompt shows every token in the order the model read it, template tokens included. Drag its bottom edge to resize it.</p>}
        <p className="docattentionlens-source-hint">Keys: , and . choose the previous or next source token{prompt ? '; P shows or hides the full prompt' : ''}.</p>
      </div>}
    </PopoverButton>
  </>;
  return <div className="docattentionlens-source-line" data-view={showingPrompt ? 'prompt' : 'row'} data-column={!!column}>
    {column && <span className="docattentionlens-source-column" data-testid="source-column">
      <span>{column.label}</span>
      <button type="button" className="docattentionlens-icon-button" aria-label={column.clearLabel ?? 'Clear patch selection'} title={column.clearLabel ?? 'Return to the selected token'} onClick={column.onClear}><X size={14}/></button>
    </span>}
    {showingPrompt ? <div className="docattentionlens-source-prompt">{prompt(controls)}</div>
      : <><div ref={row} className="docattentionlens-source-row" role="group" aria-label="Source tokens">{children}</div>{controls}</>}
  </div>;
}

/** A labelled part of the source row (Question, Answer, Query…). */
export function SourceSegment({children, arrow}: {children: ReactNode; arrow?: boolean}) {
  return <>{arrow && <span className="docattentionlens-source-arrow" aria-hidden="true">→</span>}<div className="docattentionlens-source-segment">{children}</div></>;
}

/** The page image as one chip, with the summed weight its patches receive in the shown row. */
export function ImageChip({patches, weight, label = 'Page image', signed = false}: {patches: number; weight?: number; label?: string; signed?: boolean}) {
  if (!patches) return null;
  return <span className="docattentionlens-token is-special docattentionlens-image-chip" data-testid="page-image-chip"
    title={weight === undefined ? `${label}: ${patches} patches · click a patch on the page to choose it` : `Summed ${signed ? 'similarity' : 'attention weight'} on the ${patches} page-image patches`}>
    <span aria-hidden="true">▣ </span>{label} · {patches} patches{weight !== undefined && ` · Σ ${attentionWeightLabel(weight)}`}</span>;
}

/** The selected source when it is not a token in the row: a template token or a page patch. */
export function SourceChip({children}: {children: ReactNode}) {
  return <span className="docattentionlens-source-chip" data-testid="source-chip">{children}</span>;
}
