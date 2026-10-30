import { useState, type ReactNode } from 'react';
import { ChevronLeft, ChevronRight, Info, Maximize, Minimize, Minus, PanelRight, Plus, Settings } from '../../app/icons';

/**
 * Which of a method's maps (one per layer, head or condition) the page draws, in the method sentence.
 * Every map shares one page scale; Details › Result shows them side by side.
 */
export function ComparisonChoice({labels, index, onChange}: {labels: string[]; index: number; onChange: (index: number) => void}) {
  if (labels.length < 2) return null;
  const at = Math.max(0, Math.min(index, labels.length - 1));
  return <span className="docattentionlens-sentence-comparison" role="group" aria-label="Map shown on the page">
    <button type="button" className="docattentionlens-icon-button" aria-label="Previous map" disabled={at === 0} onClick={() => onChange(at - 1)}><ChevronLeft size={14}/></button>
    <select aria-label="Shown comparison" value={at} onChange={e => onChange(Number(e.target.value))}>
      {labels.map((label, i) => <option key={i} value={i}>{label}</option>)}</select>
    <span className="docattentionlens-sentence-count">of {labels.length}</span>
    <button type="button" className="docattentionlens-icon-button" aria-label="Next map" disabled={at === labels.length - 1} onClick={() => onChange(at + 1)}><ChevronRight size={14}/></button>
  </span>;
}

/**
 * Instruments that float over the page canvas. The layer covers the canvas without taking pointer
 * events; its top and bottom rows stay pinned in view while a fitted-width page scrolls.
 */
export function CanvasInstruments({topLeft, topRight, bottomLeft, bottomRight}: {topLeft?: ReactNode; topRight?: ReactNode; bottomLeft?: ReactNode; bottomRight?: ReactNode}) {
  return <div className="docattentionlens-canvas-instruments">
    <div className="docattentionlens-instruments-row is-top"><div className="docattentionlens-instruments-start">{topLeft}</div><div className="docattentionlens-instruments-end">{topRight}</div></div>
    <div className="docattentionlens-instruments-row is-bottom"><div className="docattentionlens-instruments-start">{bottomLeft}</div><div className="docattentionlens-instruments-end">{bottomRight}</div></div>
  </div>;
}

/** The one-line method sentence: what the page shows and the method that computes it. */
export function MethodSentence({lead, method, extra, detail, detailText, testId}: {
  lead?: ReactNode; method?: ReactNode; extra?: ReactNode;
  /** The exact row (keeps its test ids); read by screen readers, not shown. */
  detail?: ReactNode;
  /** The same as plain text, for the tooltip. */
  detailText?: string; testId?: string;
}) {
  return <div className="docattentionlens-method-sentence docattentionlens-instrument" role="group" aria-label="Shown on the page" data-testid={testId} title={detailText || undefined}>
    <p className="docattentionlens-sentence-main">{lead && <span className="docattentionlens-sentence-lead">{lead}</span>}
      {lead && method && <span className="docattentionlens-sentence-sep" aria-hidden="true">·</span>}{method}{extra}
      {detail && <span className="docattentionlens-sr-only docattentionlens-sentence-detail"> · {detail}</span>}</p>
  </div>;
}

/** The Details toggle on the canvas's top-right corner. */
/** The Details toggle: on the page's top-right corner, or at the end of the method bar in the three-column layout (`inBar`). */
export function DetailsToggle({open, onToggle, controls, inBar = false}: {open: boolean; onToggle: () => void; controls?: string; inBar?: boolean}) {
  return <button type="button" className={`docattentionlens-button ${inBar ? '' : 'docattentionlens-instrument '}docattentionlens-details-toggle`} aria-expanded={open} aria-controls={controls}
    title={open ? `Close Details (D)${inBar ? ' and show the selected element' : ''}` : `Open Details: ${inBar ? 'result' : 'element, result'}, display, run and notes (D)`} onClick={onToggle}>
    <PanelRight/><span>Details</span></button>;
}

/** The one-line colour legend at the canvas's bottom-left. */
export function CanvasLegend({ramp, bound, range, share, lowMass, notes, settingsLabel, settingsText, settingsOpen, onSettings, disabled}: {
  /** PageKey (floor, ramp, bound, hover readout), or a plain ramp. */
  ramp?: ReactNode;
  /** The bound in words when `ramp` is a plain ramp without its ends. */
  bound?: ReactNode;
  /** What the colour is and what sets the bound (`display-scale` / `element-scale`), visually hidden. */
  range?: ReactNode;
  /** The image share. */
  share?: ReactNode;
  /** Why the image share is small; expands on click. */
  lowMass?: string;
  notes?: ReactNode;
  /** Accessible name ("Display settings · Overlay · Image contrast") and the shown text ("Overlay"), which it contains. */
  settingsLabel: string; settingsText: string;
  /** Whether Details is open on Display. */
  settingsOpen: boolean; onSettings: () => void;
  /** Nothing on the page takes display choices (for example entropy, which is not an attention weight). */
  disabled?: boolean;
}) {
  const [why, setWhy] = useState(false);
  return <div className="docattentionlens-canvas-legend docattentionlens-instrument" role="group" aria-label="Colour legend">
    <div className="docattentionlens-legend-main">
      {ramp}
      {bound && <span className="docattentionlens-legend-bound">{bound}</span>}
      {range && <span className="docattentionlens-hidden-copy">{range}</span>}
      {share && <span className="docattentionlens-legend-share">{share}</span>}
      {lowMass && <button type="button" className="docattentionlens-icon-button docattentionlens-legend-why" aria-expanded={why} aria-label="Why the image share is small" title="Why the image share is small" onClick={() => setWhy(v => !v)}><Info size={14}/></button>}
      <button type="button" className="docattentionlens-quiet docattentionlens-legend-settings" aria-label={settingsLabel} title="Display settings in Details: colour mode, opacity, scale, colour map and overlays (V cycles the colour mode)"
        aria-expanded={settingsOpen} aria-controls="workspace-details" onClick={onSettings} disabled={disabled}><Settings size={14}/><span>{settingsText}</span></button>
    </div>
    {lowMass && why && <p className="docattentionlens-legend-note" data-testid="low-mass-note">{lowMass}</p>}
    {notes}
  </div>;
}

export type FitMode = 'page' | 'width' | 'manual';

/**
 * Zoom pill at the canvas's bottom-right: zoom out, the fit (click toggles Fit page and Fit width; a
 * custom zoom returns to Fit page), zoom in, image focus and, with a selected element, Focus element.
 */
export function ZoomPill({fit, zoom, onZoomOut, onZoomIn, onFit, imageFocus, onImageFocus, onFocusElement}: {
  fit: FitMode; zoom: number; onZoomOut: () => void; onZoomIn: () => void; onFit: (next: 'page' | 'width') => void;
  imageFocus: boolean; onImageFocus: () => void; onFocusElement?: () => void;
}) {
  const label = fit === 'page' ? 'Fit page' : fit === 'width' ? 'Fit width' : `${Math.round(zoom * 100)}%`;
  return <div className="docattentionlens-zoom-pill docattentionlens-instrument" role="group" aria-label="Page view">
    {onFocusElement && <button type="button" className="docattentionlens-quiet docattentionlens-focus-element" title="Zoom to the selected element" onClick={onFocusElement}>Focus element</button>}
    <button type="button" className="docattentionlens-icon-button" aria-label="Zoom out" title="Zoom out" onClick={onZoomOut}><Minus/></button>
    <button type="button" className="docattentionlens-quiet docattentionlens-fit-toggle" data-fit={fit} aria-label={`Canvas sizing: ${fit === 'manual' ? `custom zoom ${label}` : label}`}
      title={fit === 'page' ? 'Fit page · click for Fit width (the page uses the column width and scrolls)' : fit === 'width' ? 'Fit width · click for Fit page' : 'Custom zoom · click to fit the page'}
      onClick={() => onFit(fit === 'page' ? 'width' : 'page')}>{label}</button>
    <button type="button" className="docattentionlens-icon-button" aria-label="Zoom in" title="Zoom in" onClick={onZoomIn}><Plus/></button>
    <button type="button" className="docattentionlens-icon-button docattentionlens-image-focus" aria-pressed={imageFocus} aria-label="Image focus"
      title={imageFocus ? 'Leave image focus (F)' : 'Image focus: the page with its instruments only (F)'} onClick={onImageFocus}>{imageFocus ? <Minimize/> : <Maximize/>}</button>
  </div>;
}
