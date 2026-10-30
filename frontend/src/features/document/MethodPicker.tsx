import { useCallback, useEffect, useId, useLayoutEffect, useMemo, useRef, useState,
  type CSSProperties, type KeyboardEvent, type ReactNode, type Ref, type RefObject } from 'react';
import type { Method } from '../../api/client';
import { PERSPECTIVES, methodKey, perspectiveLabel, type Perspective, type Role } from './perspective';
import { familyLabel, filterChoices, groupByFamily, groupingInputChoices, methodControls, shortReason,
  type MethodChoice } from './methodChoices';
import '../../styles/method-picker.css';

/** Below this width the picker is a full-width bottom sheet with the preview under the list. */
export const PICKER_SHEET_WIDTH = 900;
const GAP = 6, MARGIN = 8, WIDTH = 760, HEIGHT = 540;

export type MethodSelectContext<T extends MethodChoice = MethodChoice> = {
  choice: T;
  /** True when a General method was chosen from "grouped into elements" in the Document-aware view. */
  asGroupingInput: boolean;
};

export type MethodReadiness = {state: 'ready' | 'unavailable' | 'planned'; text: string};

export type MethodPickerProps<T extends MethodChoice = MethodChoice> = {
  open: boolean;
  /** Called on Esc, outside click and after a selection. Focus returns to the anchor except on outside click. */
  onClose: () => void;
  /** The trigger; the popover opens below it, flipping above or clamping inside the viewport. Not used when docked. */
  anchorRef?: RefObject<HTMLElement | null>;
  /** Choices for the active model role (see `methodChoices.ts`). */
  choices: T[];
  /** `methodKey` of the method shown on the page. */
  currentKey?: string;
  /** `methodKey` of the General input while a document-aware view groups it into elements. */
  currentInputKey?: string;
  /** Perspective the list opens on; the picker's General | Document-aware filter starts here. */
  perspective: Perspective;
  /** Called when the filter changes. The picker filters its own list whether or not the parent follows. */
  onPerspective?: (perspective: Perspective) => void;
  onSelect: (method: Method, context: MethodSelectContext<T>) => void;
  /** Parameters for the previewed method (for example `MethodParameters`); render nothing when not applicable. */
  renderSettings?: (method: Method, choice: T) => ReactNode;
  /** Link to the method's full description, e.g. `/toolbox/:methodId?returnTo=`. */
  catalogueHref?: (method: Method) => string;
  /** Readiness for this capture, e.g. `structureStatus` for document-aware methods. Falls back to reasons. */
  readiness?: (choice: T) => MethodReadiness | undefined;
  /** Role and encoding count for the short unavailable reason (`shortReason`). */
  role?: Role;
  encodingCount?: number;
  /** Perspectives this case has structure for; a table grounding case has no layout elements, so no Document-aware. */
  perspectives?: Perspective[];
  /** In the Document-aware view, also list General methods that can be grouped into elements. */
  groupingInputs?: boolean;
  loading?: boolean;
  error?: string;
  /** Keep the picker open after choosing (defaults to closing). */
  keepOpenOnSelect?: boolean;
  /** A note under the method list, e.g. a full-attention version of this run that unlocks more methods. */
  notice?: ReactNode;
  /**
   * In flow (Details › Method) instead of a popover: no positioning, scrim, focus trap or outside-click close;
   * it stays open after a choice and Esc only clears the search.
   */
  docked?: boolean;
};

type OptionRow<T> = {kind: 'option'; id: string; key: string; choice: T; asGroupingInput: boolean; current: boolean};
type FamilyRow = {kind: 'family'; id: string; key: string; label: string; count: number; expanded: boolean; groupId: string};
type Section<T> = {key: string; label: string; header: FamilyRow; options: OptionRow<T>[]};

const domId = (s: string) => s.replace(/[^\w-]/g, '_');
const focusable = (root: HTMLElement) => [...root.querySelectorAll<HTMLElement>(
  'a[href], button:not([disabled]), input:not([disabled]), select:not([disabled]), textarea:not([disabled]), [tabindex]:not([tabindex="-1"])')]
  .filter(el => !el.closest('[hidden]'));
const isTyping = (target: EventTarget | null) => target instanceof HTMLElement
  && (target.tagName === 'INPUT' || target.tagName === 'TEXTAREA' || target.tagName === 'SELECT' || target.isContentEditable);

function useViewportWidth(active: boolean) {
  const [width, setWidth] = useState(() => typeof window === 'undefined' ? WIDTH * 2 : window.innerWidth);
  useEffect(() => {
    if (!active) return;
    const update = () => setWidth(window.innerWidth);
    update();
    window.addEventListener('resize', update);
    return () => window.removeEventListener('resize', update);
  }, [active]);
  return width;
}

/** Popover position below the anchor; flips above when there is more room there, and stays inside the viewport. */
export function popoverPosition(anchor: {top: number; bottom: number; left: number}, viewport: {width: number; height: number}): CSSProperties {
  const width = Math.min(WIDTH, viewport.width - 2 * MARGIN);
  const left = Math.max(MARGIN, Math.min(anchor.left, viewport.width - width - MARGIN));
  const below = viewport.height - anchor.bottom - GAP - MARGIN, above = anchor.top - GAP - MARGIN;
  const fits = 320;
  if (below >= fits) return {top: anchor.bottom + GAP, left, width, height: Math.min(HEIGHT, below)};
  if (above >= fits && above > below) {const height = Math.min(HEIGHT, above); return {top: anchor.top - GAP - height, left, width, height};}
  // Neither side has room: use the viewport's height and clamp inside it.
  const height = Math.min(HEIGHT, viewport.height - 2 * MARGIN);
  return {top: Math.max(MARGIN, Math.min(anchor.bottom + GAP, viewport.height - height - MARGIN)), left, width, height};
}

const Chevron = ({open}: {open?: boolean}) => <svg className="docattentionlens-mp-chevron" width="10" height="10" viewBox="0 0 10 10" aria-hidden="true"
  style={open === undefined ? undefined : {transform: open ? 'rotate(0deg)' : 'rotate(-90deg)'}}>
  <path d="M2 3.5 5 6.5 8 3.5" fill="none" stroke="currentColor" strokeWidth="1.4"/></svg>;
const SearchIcon = () => <svg width="12" height="12" viewBox="0 0 12 12" aria-hidden="true">
  <circle cx="5" cy="5" r="3.6" fill="none" stroke="currentColor" strokeWidth="1.3"/><path d="M7.7 7.7 11 11" stroke="currentColor" strokeWidth="1.3"/></svg>;

/** A method name that opens the picker as a popover. */
export function MethodTrigger({label, open, onClick, disabled, className, title, controls, ref}: {
  label: string; open: boolean; onClick: () => void; disabled?: boolean; className?: string; title?: string;
  /** Id of the open picker dialog, for `aria-controls`. */
  controls?: string; ref?: Ref<HTMLButtonElement>;
}) {
  return <button ref={ref} type="button" className={`docattentionlens-mp-trigger${className ? ` ${className}` : ''}`} aria-haspopup="dialog" aria-expanded={open}
    aria-controls={open ? controls : undefined} disabled={disabled} title={title ?? 'Choose analysis method (M)'} onClick={onClick}>
    <span className="docattentionlens-mp-trigger-label">{label}</span><span className="docattentionlens-mp-sr">, choose analysis method</span><Chevron/>
  </button>;
}

/**
 * Search-first method chooser: perspective filter, families, and a preview of the highlighted method.
 * A popover (or bottom sheet) by default; `docked` renders it in flow with the preview under the list.
 */
export function MethodPicker<T extends MethodChoice = MethodChoice>(props: MethodPickerProps<T>) {
  return props.open ? <OpenPicker {...props}/> : null;
}

function OpenPicker<T extends MethodChoice>({onClose, anchorRef, choices, currentKey, currentInputKey, perspective, onPerspective, onSelect,
  renderSettings, catalogueHref, readiness, role = 'generation', encodingCount, groupingInputs = false, loading, error, keepOpenOnSelect, notice, perspectives,
  docked = false}: MethodPickerProps<T>) {
  // Table-aware needs a table grounding case, so it is offered only when the caller lists it.
  const offered = PERSPECTIVES.filter(p => perspectives ? perspectives.includes(p.id) : p.id !== 'table_aware');
  const base = domId(useId());
  const dialogRef = useRef<HTMLDivElement>(null);
  const searchRef = useRef<HTMLInputElement>(null);
  const listRef = useRef<HTMLDivElement>(null);
  const [query, setQuery] = useState('');
  const [view, setView] = useState<Perspective>(perspective);
  const [overrides, setOverrides] = useState<Record<string, boolean>>({});
  const [activeId, setActiveId] = useState<string>();
  const lastPreview = useRef<OptionRow<T>>(undefined);
  const viewportWidth = useViewportWidth(!docked);
  const sheet = !docked && viewportWidth < PICKER_SHEET_WIDTH;
  const [position, setPosition] = useState<CSSProperties>();

  useEffect(() => setView(perspective), [perspective]);
  // A popover takes focus when it opens; a docked picker is focused by whoever opens its tab.
  useEffect(() => {if (!docked) searchRef.current?.focus();}, []); // eslint-disable-line react-hooks/exhaustive-deps

  const place = useCallback(() => {
    const anchor = anchorRef?.current?.getBoundingClientRect();
    if (!anchor || sheet || docked) {setPosition(undefined); return;}
    setPosition(popoverPosition(anchor, {width: window.innerWidth, height: window.innerHeight}));
  }, [anchorRef, sheet, docked]);
  useLayoutEffect(() => {
    place();
    window.addEventListener('resize', place);
    window.addEventListener('scroll', place, true);
    return () => {window.removeEventListener('resize', place); window.removeEventListener('scroll', place, true);};
  }, [place]);

  const close = useCallback((refocus = true) => {
    onClose();
    if (refocus) anchorRef?.current?.focus();
  }, [onClose, anchorRef]);

  // Outside clicks close; Esc and "/" still work if focus has left the dialog.
  useEffect(() => {
    if (docked) return;
    const onPointer = (e: Event) => {
      const target = e.target as Node | null;
      if (target && !dialogRef.current?.contains(target) && !anchorRef?.current?.contains(target)) close(false);
    };
    const onKey = (e: globalThis.KeyboardEvent) => {
      if (dialogRef.current?.contains(e.target as Node)) return;
      if (e.key === 'Escape') {e.preventDefault(); close();}
      else if (e.key === '/' && !isTyping(e.target)) {e.preventDefault(); searchRef.current?.focus();}
    };
    document.addEventListener('pointerdown', onPointer);
    document.addEventListener('keydown', onKey);
    return () => {document.removeEventListener('pointerdown', onPointer); document.removeEventListener('keydown', onKey);};
  }, [close, anchorRef, docked]);

  const searching = query.trim().length > 0;
  const sections = useMemo<Section<T>[]>(() => {
    const option = (section: string, choice: T, asGroupingInput = false): OptionRow<T> => {
      const key = methodKey(choice.method);
      return {kind: 'option', id: `${base}-o-${section}-${domId(key)}`, key, choice, asGroupingInput,
        current: asGroupingInput ? key === currentInputKey : key === currentKey};
    };
    const out: Section<T>[] = [];
    const family = (key: string, label: string, members: T[], asGroupingInput = false) => {
      const options = members.map(c => option(key, c, asGroupingInput));
      const expanded = searching || (overrides[key] ?? options.some(o => o.current));
      out.push({key, label, options: expanded ? options : [],
        header: {kind: 'family', id: `${base}-f-${domId(key)}`, key, label, count: members.length, expanded, groupId: `${base}-g-${domId(key)}`}});
    };
    for (const group of groupByFamily(filterChoices(choices, query, view))) family(`family-${group.family}`, group.label, group.choices);
    if (groupingInputs && view === 'document_aware') {
      const inputs = filterChoices(groupingInputChoices(choices), query, 'general');
      if (inputs.length) family('grouped-inputs', 'General attention · grouped into elements', inputs, true);
    }
    return out;
  }, [base, choices, currentKey, currentInputKey, groupingInputs, overrides, query, searching, view]);

  // Keyboard order: each family header (when not searching), then its visible options.
  const rows = useMemo(() => sections.flatMap(s => [...(searching ? [] : [s.header]), ...s.options]), [sections, searching]);
  const options = rows.filter((r): r is OptionRow<T> => r.kind === 'option');
  const fallback = options.find(o => o.current && o.choice.available) ?? options.find(o => o.choice.available) ?? rows[0];
  const active = rows.find(r => r.id === activeId) ?? fallback;
  if (active?.kind === 'option') lastPreview.current = active;
  const preview = active?.kind === 'option' ? active
    : lastPreview.current && rows.some(r => r.id === lastPreview.current?.id) ? lastPreview.current : options[0];

  useEffect(() => {
    if (active) document.getElementById(active.id)?.scrollIntoView?.({block: 'nearest'});
  }, [active?.id]); // eslint-disable-line react-hooks/exhaustive-deps

  const toggle = (row: FamilyRow) => {setOverrides(o => ({...o, [row.key]: !row.expanded})); setActiveId(row.id);};
  const choose = (row: OptionRow<T>) => {
    if (!row.choice.available || row.choice.method.status === 'planned') return;
    onSelect(row.choice.method, {choice: row.choice, asGroupingInput: row.asGroupingInput});
    if (!keepOpenOnSelect && !docked) close();
  };
  const changeView = (next: Perspective) => {
    if (next === view) return;
    setView(next); setActiveId(undefined); onPerspective?.(next);
  };
  const move = (delta: number) => {
    if (!rows.length) return;
    const index = active ? rows.indexOf(active) : -1;
    setActiveId(rows[Math.max(0, Math.min(rows.length - 1, index + delta))].id);
  };

  const onKeyDown = (e: KeyboardEvent<HTMLDivElement>) => {
    const inList = e.target === searchRef.current || listRef.current?.contains(e.target as Node);
    if (e.key === 'Escape') {
      if (docked && !query) return;
      e.preventDefault(); e.stopPropagation();
      if (docked) {setQuery(''); setActiveId(undefined);} else close();
      return;
    }
    if (e.key === 'Tab' && dialogRef.current && !docked) {
      const items = focusable(dialogRef.current);
      if (!items.length) return;
      const first = items[0], last = items[items.length - 1];
      if (e.shiftKey && (document.activeElement === first || !dialogRef.current.contains(document.activeElement))) {e.preventDefault(); last.focus();}
      else if (!e.shiftKey && document.activeElement === last) {e.preventDefault(); first.focus();}
      return;
    }
    if (e.key === '/' && !isTyping(e.target)) {e.preventDefault(); searchRef.current?.focus(); return;}
    if (!inList) return;
    if (e.key === 'ArrowDown' || e.key === 'ArrowUp') {e.preventDefault(); move(e.key === 'ArrowDown' ? 1 : -1);}
    else if ((e.key === 'Home' || e.key === 'End') && e.target !== searchRef.current) {e.preventDefault(); move(e.key === 'Home' ? -rows.length : rows.length);}
    else if (e.key === 'Enter' && active) {e.preventDefault(); if (active.kind === 'option') choose(active); else toggle(active);}
  };

  const reasonText = (choice: T) => shortReason(choice, {role, encodingCount});
  const listId = `${base}-list`;
  const renderOption = (row: OptionRow<T>) => {
    const {choice} = row, disabled = !choice.available || choice.method.status === 'planned';
    return <div key={row.id} id={row.id} role="option" aria-selected={row.current} aria-disabled={disabled || undefined}
      data-method-key={row.key} data-grouping-input={row.asGroupingInput || undefined}
      data-badge={row.current ? row.asGroupingInput ? 'Current input' : 'Current' : undefined}
      className={`docattentionlens-mp-option${row.current ? ' is-current' : ''}${row.id === active?.id ? ' is-active' : ''}${disabled ? ' is-disabled' : ''}`}
      onMouseDown={e => e.preventDefault()} onMouseMove={() => row.id !== active?.id && setActiveId(row.id)} onClick={() => choose(row)}>
      <span className="docattentionlens-mp-option-name">{choice.method.name}</span>
      {disabled && <span className="docattentionlens-mp-option-reason" aria-hidden="true">{reasonText(choice)}</span>}
      {disabled && <span className="docattentionlens-mp-sr">{`. Unavailable: ${choice.method.status === 'planned' ? 'Planned, not implemented yet.' : choice.reasons.join(' ') || reasonText(choice)}`}</span>}
    </div>;
  };

  const viewIndex = offered.findIndex(p => p.id === view);
  const dialogClass = `docattentionlens-mp${sheet ? ' is-sheet' : ''}${docked ? ' is-docked' : ''}`;
  return <>
    {sheet && <div className="docattentionlens-mp-scrim" aria-hidden="true" onClick={() => close(false)}/>}
    <div ref={dialogRef} id={`${base}-dialog`} role={docked ? 'region' : 'dialog'} aria-label="Choose analysis method" aria-modal={sheet || undefined}
      className={dialogClass} style={position} onKeyDown={onKeyDown} data-testid={docked ? 'method-browser' : undefined}>
      <div className="docattentionlens-mp-list-column">
        <label className="docattentionlens-mp-search">
          <SearchIcon/><span className="docattentionlens-mp-sr">Find a method</span>
          <input ref={searchRef} type="search" role="combobox" aria-expanded="true" aria-controls={listId} aria-autocomplete="list"
            aria-activedescendant={active?.id} value={query} placeholder="Search methods  /" autoComplete="off" spellCheck={false}
            onChange={e => {setQuery(e.target.value); setActiveId(undefined);}}/>
        </label>
        {offered.length > 1 && <div className="docattentionlens-mp-perspective" role="radiogroup" aria-label="Analysis perspective"
          onKeyDown={e => {if (e.key === 'ArrowLeft' || e.key === 'ArrowRight') {e.preventDefault();
            const next = offered[(viewIndex + (e.key === 'ArrowRight' ? 1 : offered.length - 1)) % offered.length];
            changeView(next.id); (e.currentTarget.querySelector(`[data-perspective="${next.id}"]`) as HTMLElement | null)?.focus();}}}>
          {offered.map(p => <button key={p.id} type="button" role="radio" aria-checked={view === p.id} data-perspective={p.id}
            tabIndex={view === p.id ? 0 : -1} title={p.description} onClick={() => changeView(p.id)}>{p.label}</button>)}
        </div>}
        <div ref={listRef} id={listId} role="listbox" aria-label={`${perspectiveLabel(view)} analysis methods`} tabIndex={-1}
          aria-activedescendant={active?.id} className="docattentionlens-mp-list">
          {error ? <p className="docattentionlens-mp-note" role="alert">{error}</p>
            : loading && !choices.length ? <p className="docattentionlens-mp-note" role="status">Loading method catalogue…</p>
            : !sections.length ? <p className="docattentionlens-mp-note" role="status">
              {searching ? `No ${perspectiveLabel(view)} methods match “${query.trim()}”.` : `No ${perspectiveLabel(view)} methods for this capture.`}</p>
            : sections.map(s => <div key={s.key} role="presentation" className="docattentionlens-mp-section">
                <div role="presentation" className="docattentionlens-mp-family-head">
                  <button type="button" id={s.header.id} tabIndex={-1} aria-expanded={s.header.expanded} aria-controls={s.header.groupId}
                    className={`docattentionlens-mp-family${s.header.id === active?.id ? ' is-active' : ''}`} disabled={searching}
                    onMouseDown={e => e.preventDefault()} onClick={() => toggle(s.header)}>
                    <Chevron open={s.header.expanded}/><span className="docattentionlens-mp-family-label">{s.label}</span>
                    <span className="docattentionlens-mp-count" aria-label={`${s.header.count} methods`}>{s.header.count}</span>
                  </button>
                </div>
                {s.header.expanded && <div role="group" id={s.header.groupId} aria-labelledby={s.header.id}>{s.options.map(renderOption)}</div>}
              </div>)}
        </div>
        {notice && <div className="docattentionlens-mp-notice">{notice}</div>}
      </div>
      <Preview row={preview} reasonText={reasonText} readiness={readiness} renderSettings={renderSettings} catalogueHref={catalogueHref} onChoose={choose}/>
    </div>
  </>;
}

function Preview<T extends MethodChoice>({row, reasonText, readiness, renderSettings, catalogueHref, onChoose}: {
  row?: OptionRow<T>; reasonText: (choice: T) => string; readiness?: (choice: T) => MethodReadiness | undefined;
  renderSettings?: (method: Method, choice: T) => ReactNode; catalogueHref?: (method: Method) => string; onChoose: (row: OptionRow<T>) => void;
}) {
  if (!row) return <section className="docattentionlens-mp-preview" aria-label="Method preview"><p className="docattentionlens-mp-note">Highlight a method to see what it reports.</p></section>;
  const {method, available, reasons} = row.choice;
  const planned = method.status === 'planned';
  const status = readiness?.(row.choice);
  const state = status?.state ?? (planned ? 'planned' : available ? 'ready' : 'unavailable');
  const {axes, settings} = methodControls(method);
  const href = catalogueHref?.(method);
  const human = (s: string) => s.replaceAll('_', ' ');
  return <section className="docattentionlens-mp-preview" aria-label={`Preview: ${method.name}`} data-testid="method-preview">
    <p className="docattentionlens-mp-kicker">{perspectiveLabel(method.perspective)} · {familyLabel(method.family)}{planned ? ' · Planned' : ''}</p>
    <h3 className="docattentionlens-mp-title">{method.name}</h3>
    <p className="docattentionlens-mp-summary">{method.summary}</p>
    <dl className="docattentionlens-mp-facts">
      <dt>Reports</dt><dd>{human(method.evidence_kind)} · {human(method.quantity)}</dd>
      <dt>Units</dt><dd>{method.output.units}{method.output.signed ? ' · signed' : ''}</dd>
      <dt>Support</dt><dd>{method.output.support}</dd>
      <dt>Normalization</dt><dd>{method.output.normalization}</dd>
      <dt>Controls</dt><dd>{axes.length ? axes.join(' · ') : 'No layer or head controls'}{settings.length ? ` · settings: ${settings.join(', ')}` : ''}</dd>
    </dl>
    <div className={`docattentionlens-mp-readiness is-${state}`} data-testid="method-readiness">
      <p><strong>{state === 'ready' ? 'Ready' : state === 'planned' ? 'Planned' : 'Unavailable'}</strong>
        {' · '}{status ? status.text.replace(/^(Ready|Unavailable|Planned):\s*/, '') : planned ? 'Not implemented yet.' : available ? 'Computed from this capture.' : reasonText(row.choice)}</p>
      {!available && !planned && !!reasons.length && <ul>{reasons.map(r => <li key={r}>{r}</li>)}</ul>}
    </div>
    {renderSettings && available && <div className="docattentionlens-mp-settings">{renderSettings(method, row.choice)}</div>}
    <div className="docattentionlens-mp-preview-actions">
      <button type="button" className="docattentionlens-mp-use" disabled={!available || planned || row.current} onClick={() => onChoose(row)}>
        {row.current ? 'Current method' : `Use ${method.name}`}</button>
      {href && <a href={href} target="_blank" rel="noreferrer">Full description in Methods ↗</a>}
    </div>
  </section>;
}
