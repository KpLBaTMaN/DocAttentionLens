import type { KeyboardEvent, ReactNode, Ref } from 'react';
import { Maximize, Minimize, X } from '../../app/icons';

export type DetailsTab = 'method' | 'element' | 'result' | 'display' | 'run' | 'notes';

export const DETAILS_TABS: {id: DetailsTab; label: string; title: string}[] = [
  {id: 'method', label: 'Method', title: 'Choose the analysis method and its settings (M)'},
  {id: 'element', label: 'Element', title: 'Selected element, all page elements and the element summary'},
  {id: 'result', label: 'Result', title: "The method's own output and its settings"},
  {id: 'display', label: 'Display', title: 'Colour mode, opacity, colour map, scale and overlays, with the colour scales and row totals'},
  {id: 'run', label: 'Run', title: 'Capture, pages, provenance and research receipt'},
  {id: 'notes', label: 'Notes', title: 'Investigation name and notes'},
];

export const isDetailsTab = (value: unknown): value is DetailsTab => DETAILS_TABS.some(t => t.id === value);

/** The Details drawer of the three-column layout: the method and the elements have their own places there. */
export const DRAWER_TABS: DetailsTab[] = ['result', 'display', 'run', 'notes'];

/** Arrow keys, Home and End move between the offered tabs and select them (automatic activation). */
function tabKeys(e: KeyboardEvent<HTMLDivElement>, tab: DetailsTab, onTab: (tab: DetailsTab) => void, offered: {id: DetailsTab}[]) {
  const at = offered.findIndex(t => t.id === tab);
  const next = e.key === 'ArrowRight' ? at + 1 : e.key === 'ArrowLeft' ? at - 1 : e.key === 'Home' ? 0 : e.key === 'End' ? offered.length - 1 : undefined;
  if (next === undefined) return;
  e.preventDefault();
  const id = offered[(next + offered.length) % offered.length].id;
  onTab(id);
  requestAnimationFrame(() => document.getElementById(`details-tab-${id}`)?.focus());
}

/** The Details panel: tabs on the right, a bottom sheet on phones, a drawer in the three-column layout. */
export function DetailsPanel({tab, onTab, label, panelRef, content, hidden, wideTable, onWideTable, onClose, sheet, labels, tabs, closeTitle}: {
  tab: DetailsTab; onTab: (tab: DetailsTab) => void;
  /** Tab names and descriptions for this case, such as Table instead of Element for a table grounding case. */
  labels?: Partial<Record<DetailsTab, {label: string; title: string}>>;
  /** The tabs offered, in their usual order; every tab by default. */
  tabs?: DetailsTab[];
  /** Tooltip of the close button, when closing does more than hide the panel. */
  closeTitle?: string;
  /** Accessible name of the panel ("Attention panel" or "Retrieval panel"). */
  label: string;
  panelRef?: Ref<HTMLElement>;
  content: Record<DetailsTab, ReactNode>;
  hidden: boolean;
  wideTable: boolean; onWideTable: (wide: boolean) => void;
  onClose: () => void;
  /** Phones: the panel is a bottom sheet over the bottom bar. */
  sheet?: boolean;
}) {
  const wideOffered = !sheet && (tab === 'result' || tab === 'element');
  const offered = tabs ? DETAILS_TABS.filter(t => tabs.includes(t.id)) : DETAILS_TABS;
  return <aside ref={panelRef} id="workspace-details" tabIndex={-1} className="docattentionlens-analysis-inspector" aria-label={label} hidden={hidden}
    data-wide={wideOffered && wideTable} data-tab={tab}
    onKeyDown={e => {if (e.key === 'Escape' && wideTable && !e.defaultPrevented) {e.preventDefault(); onWideTable(false);}}}>
    <div className="docattentionlens-panel-title"><h2>Details</h2>
      {wideOffered && <button type="button" className="docattentionlens-quiet docattentionlens-details-wide" aria-pressed={wideTable}
        title={wideTable ? 'Return Details to its column (Esc)' : 'Widen Details over the page for large tables'}
        onClick={() => onWideTable(!wideTable)}>{wideTable ? <Minimize size={14}/> : <Maximize size={14}/>}<span>{wideTable ? 'Close wide table' : 'Open as wide table'}</span></button>}
      <button type="button" className="docattentionlens-icon-button docattentionlens-details-close" aria-label="Close details" title={closeTitle ?? (sheet ? 'Close Details (Esc)' : 'Close Details (D)')} onClick={onClose}><X/></button>
    </div>
    <div className="docattentionlens-details-tabs" role="tablist" aria-label="Details sections" onKeyDown={e => tabKeys(e, tab, onTab, offered)}>
      {offered.map(t => <button key={t.id} type="button" role="tab" id={`details-tab-${t.id}`} aria-controls="details-tabpanel" aria-selected={tab === t.id}
        tabIndex={tab === t.id ? 0 : -1} title={labels?.[t.id]?.title ?? t.title} onClick={() => onTab(t.id)}>{labels?.[t.id]?.label ?? t.label}</button>)}
    </div>
    {/* Named "Details: …" rather than by its tab, so the panel's name never collides with a field's label ("Notes"). */}
    <div id="details-tabpanel" role="tabpanel" aria-label={`Details: ${labels?.[tab]?.label ?? DETAILS_TABS.find(t => t.id === tab)?.label}`} className="docattentionlens-details-body" data-tab={tab}>{content[tab]}</div>
  </aside>;
}
