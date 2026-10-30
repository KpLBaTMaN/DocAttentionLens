import { useRef, type ReactNode, type RefObject } from 'react';
import { ChevronDown, ChevronLeft, ChevronRight, Sliders, X } from '../../app/icons';
import { MenuButton, MenuItem, useDismiss } from './WorkspaceMenus';

export type MethodStep = {
  key: string;
  /** What the step is ("Attention", "Grouped into", "Reading"). */
  kicker: string;
  label: string;
  /** Tooltip and accessible description: what clicking does. */
  title: string;
  onClick: () => void;
  /** Set on steps that open the method panel: whether it is open. */
  expanded?: boolean;
  /** A short menu of choices opened from the step instead of `onClick` (Grouping in the three-column layout). */
  menu?: {label: string; items: MethodStepChoice[]};
};

export type MethodStepChoice = {label: string; checked: boolean; disabled?: boolean; title?: string; onSelect: () => void};

/** The method panel of the three-column layout, opened under the method bar. */
export type MethodPanel = {open: boolean; onClose: (refocus: boolean) => void; label: string; content: ReactNode; keepOpen?: RefObject<HTMLElement | null>[]};

const stepFace = (step: MethodStep) => <>
  <span className="docattentionlens-method-step-kicker" aria-hidden="true">{step.kicker}</span>
  <span className="docattentionlens-method-step-label" aria-hidden="true">{step.label}<ChevronDown size={12}/></span>
</>;

/** The method bar under the source line: how the page's colour is computed, as labelled steps. */
export function MethodBar({steps, testId, data, onPrev, onNext, prevTitle, nextTitle, onBrowse, browsing, browseTitle, extra, end, panel}: {
  steps: MethodStep[];
  /** `method-choice`: the shown method, with `data` as its `data-*` attributes. */
  testId?: string; data?: Record<string, string | undefined>;
  onPrev?: () => void; onNext?: () => void; prevTitle?: string; nextTitle?: string;
  /** Opens the method list with the search focused (M). Absent when there is no method choice (MaxSim). */
  onBrowse?: () => void;
  /** Whether the method list is open. */
  browsing?: boolean;
  browseTitle?: string;
  extra?: ReactNode;
  /** After Browse methods: the Details toggle in the three-column layout. */
  end?: ReactNode;
  panel?: MethodPanel;
}) {
  const bar = useRef<HTMLDivElement>(null), sheet = useRef<HTMLDivElement>(null);
  useDismiss(!!panel?.open, refocus => panel?.onClose(refocus), [bar, sheet, ...(panel?.keepOpen ?? [])]);
  const attributes = Object.fromEntries(Object.entries(data ?? {}).filter(([, v]) => v !== undefined).map(([k, v]) => [`data-${k}`, v]));
  const controls = (step: MethodStep) => step.expanded === undefined ? 'workspace-details' : step.expanded ? 'method-panel' : undefined;
  return <div ref={bar} className="docattentionlens-method-bar" role="group" aria-label="Method" data-testid={testId} {...attributes}>
    <span className="docattentionlens-method-bar-kicker" aria-hidden="true">Method</span>
    <div className="docattentionlens-method-steps">
      {steps.map((step, i) => <span key={step.key} className="docattentionlens-method-step-wrap">
        {i > 0 && <span className="docattentionlens-method-arrow" aria-hidden="true">→</span>}
        {step.menu
          ? <MenuButton label={`${step.kicker}: ${step.label}`} menuLabel={step.menu.label} title={step.title} align="start" buttonClassName="docattentionlens-method-step"
              buttonData={{'data-step': step.key}} buttonContent={stepFace(step)}>
              {close => step.menu!.items.map(item => <MenuItem key={item.label} checked={item.checked} disabled={item.disabled} title={item.title}
                onSelect={() => {close(); item.onSelect();}}>{item.label}</MenuItem>)}
            </MenuButton>
          : <button type="button" className="docattentionlens-method-step" data-step={step.key} title={step.title}
              aria-label={`${step.kicker}: ${step.label}`} aria-controls={controls(step)} aria-expanded={step.expanded}
              aria-haspopup={step.expanded === undefined ? undefined : 'dialog'} onClick={step.onClick}>{stepFace(step)}</button>}
      </span>)}
    </div>
    {extra}
    <span className="docattentionlens-method-bar-end">
      {(onPrev || onNext) && <span className="docattentionlens-method-cycle" role="group" aria-label="Step through methods">
        <button type="button" className="docattentionlens-icon-button" aria-label="Previous method" title={prevTitle ?? 'Previous method'} disabled={!onPrev} onClick={onPrev}><ChevronLeft size={14}/></button>
        <button type="button" className="docattentionlens-icon-button" aria-label="Next method" title={nextTitle ?? 'Next method'} disabled={!onNext} onClick={onNext}><ChevronRight size={14}/></button>
      </span>}
      {onBrowse && <button type="button" className="docattentionlens-button docattentionlens-method-browse" aria-pressed={browsing} aria-controls={panel ? (panel.open ? 'method-panel' : undefined) : 'workspace-details'}
        title={browseTitle ?? 'All methods, their descriptions and settings in Details › Method (M)'} onClick={onBrowse}>
        <Sliders size={14}/><span>Browse methods</span><kbd aria-hidden="true">M</kbd></button>}
      {end}
    </span>
    {panel?.open && <div ref={sheet} id="method-panel" role="dialog" aria-label={panel.label} className="docattentionlens-method-panel">
      <div className="docattentionlens-panel-title"><h2>Method</h2>
        <button type="button" className="docattentionlens-icon-button" aria-label="Close the method panel" title="Close (Esc)" onClick={() => panel.onClose(true)}><X/></button></div>
      {panel.content}
    </div>}
  </div>;
}
