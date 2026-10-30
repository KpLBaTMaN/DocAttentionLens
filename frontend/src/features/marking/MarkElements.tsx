import { useEffect, useRef, useState, type DragEvent } from 'react';
import { Link, useNavigate, useParams, useSearchParams } from 'react-router-dom';
import { api, ApiError, errorMessage, unwrap, type Document } from '../../api/client';
import { Message } from '../../app/App';
import { Busy } from '../../app/Busy';
import { adoptMarkedDocument } from '../attention-tool/InvestigationSetup';
import { MarkingCanvas, type Tool } from './MarkingCanvas';
import {
  ELEMENT_TYPES, GRID, addMark, automaticOrder, checks, commit, moveInOrder, newKey, pageMarks, positionsCovered, redo, removeMark,
  setReviewed, startHistory, typeInfo, undo, updateMark, type ElementType, type History, type MarkBox, type Marking, type MarkingDraft,
} from './markingModel';
import '../../styles/marking.css';

type SaveState = 'saved'|'unsaved'|'saving'|'failed'|'conflict';
const SAVE_LABEL:Record<SaveState, string> = {saved:'Draft saved', unsaved:'Unsaved changes', saving:'Saving…', failed:'Not saved', conflict:'Changed in another window'};
const SHORTCUTS:[string, string][] = [
  ['B', 'Box: drag to draw (hold Alt to keep the box exactly as drawn)'], ['V', 'Select: click a box; drag to move; drag a handle to resize'],
  ['O', 'Order: click boxes in reading order'], ['1–9, 0', 'Box: type of the next box. Select: type of the selected box'],
  ['Tab', 'Next box under the pointer, or next in reading order'], ['Arrows', 'Nudge the selected box by 1 px (Shift: 10 px)'],
  ['Alt + ↑ / ↓', 'Move the selected box earlier or later in reading order'], ['Delete', 'Delete the selected box'],
  ['Enter', 'Page reviewed; go to the next page to review'], ['[ / ]', 'Previous or next page'], ['H (hold)', 'Hide labels'],
  ['Space + drag, wheel', 'Pan and zoom'], ['Ctrl + Z / Ctrl + Shift + Z', 'Undo and redo'], ['Esc', 'Stop ordering or clear the selection'], ['?', 'These shortcuts'],
];

/**
 * Mark the elements of an uploaded document: boxes, types and reading order on each page. The draft saves as you work;
 * Done saves a fixed version that investigations use, and returns to setup when setup opened this screen.
 */
export function MarkElements() {
  const {documentId = ''} = useParams();
  const [search] = useSearchParams();
  const fromSetup = search.get('return') === 'setup';
  const navigate = useNavigate();
  const [draft, setDraft] = useState<MarkingDraft>();
  const [loadError, setLoadError] = useState<string>();
  const [history, setHistory] = useState<History>();
  const [pageId, setPageId] = useState<string>();
  const [selected, setSelected] = useState<string|null>(null);
  const [tool, setTool] = useState<Tool>('box');
  const [nextType, setNextType] = useState<ElementType>('text');
  const [orderCursor, setOrderCursor] = useState(1);
  const [pending, setPending] = useState<MarkBox>();
  const [hideLabels, setHideLabels] = useState(false);
  const [help, setHelp] = useState(false);
  const [showChecks, setShowChecks] = useState(false);
  const [saveState, setSaveState] = useState<SaveState>('saved');
  const [message, setMessage] = useState<string>();
  const [freezing, setFreezing] = useState(false);
  const [finished, setFinished] = useState<Document>();
  const [dragging, setDragging] = useState<string>();
  const revision = useRef(0);
  const latest = useRef<Marking>(undefined);
  const dirty = useRef(false);
  const queue = useRef<Promise<void>>(Promise.resolve());
  const timer = useRef<ReturnType<typeof setTimeout>>(undefined);

  useEffect(() => {
    let alive = true;
    api.GET('/api/v1/documents/{document_id}/marking', {params:{path:{document_id:documentId}}}).then(unwrap).then(loaded => {
      if (!alive) return;
      revision.current = loaded.revision;
      setDraft(loaded);
      const present = {marks:loaded.elements, reviewed:loaded.reviewed_page_ids, manual:loaded.manual_order_page_ids};
      latest.current = present;
      setHistory(startHistory(present));
      setPageId((loaded.pages.find(p => !loaded.reviewed_page_ids.includes(p.page_id)) ?? loaded.pages[0])?.page_id);
    }).catch(reason => {if (alive) setLoadError(errorMessage(reason));});
    return () => {alive = false;};
  }, [documentId]);

  // Each change saves shortly after the last edit; saves run one at a time against the latest revision.
  async function save() {
    if (!dirty.current || !draft || !latest.current) return;
    dirty.current = false;
    const state = latest.current;
    setSaveState('saving');
    try {
      const saved = unwrap(await api.PUT('/api/v1/documents/{document_id}/marking', {params:{path:{document_id:draft.document_id}},
        body:{expected_revision:revision.current, elements:state.marks, reviewed_page_ids:state.reviewed, manual_order_page_ids:state.manual}}));
      revision.current = saved.revision;
      setSaveState(dirty.current ? 'unsaved' : 'saved');
    } catch (reason) {
      const conflict = reason instanceof ApiError && reason.code === 'revision_conflict';
      setSaveState(conflict ? 'conflict' : 'failed');
      setMessage(errorMessage(reason));
      if (!conflict) dirty.current = true;
    }
  }
  const flush = () => {clearTimeout(timer.current); queue.current = queue.current.then(save); return queue.current;};
  useEffect(() => {
    if (!history || history.present === latest.current) return;
    latest.current = history.present;
    dirty.current = true;
    if (saveState !== 'conflict') setSaveState('unsaved');
    clearTimeout(timer.current);
    timer.current = setTimeout(() => void flush(), 700);
  }, [history?.present]);
  useEffect(() => {
    const warn = (event:BeforeUnloadEvent) => {if (dirty.current || saveState === 'saving') {event.preventDefault(); event.returnValue = '';}};
    window.addEventListener('beforeunload', warn);
    return () => window.removeEventListener('beforeunload', warn);
  }, [saveState]);

  const present = history?.present;
  const pages = draft?.pages ?? [];
  const page = pages.find(p => p.page_id === pageId);
  const marks = present && page ? pageMarks(present.marks, page.page_id) : [];
  const mark = marks.find(m => m.key === selected);
  const allChecks = present ? checks(present.marks, pages) : [];
  const pageChecks = allChecks.filter(c => c.pageId === pageId && c.keys.length);
  const blocking = allChecks.filter(c => c.level === 'blocking');
  const change = (next:(state:Marking)=>Marking) => setHistory(h => h && commit(h, next(h.present)));
  // Text read back from the PDF follows the box without its own undo step.
  const patchText = (key:string, text:string|null) => setHistory(h => h && {...h, present:updateMark(h.present, key, {text})});

  function choosePage(id:string|undefined) {
    if (!id || id === pageId) return;
    setPageId(id); setSelected(null); setOrderCursor(1);
  }
  /** Retype the selected box; the type is also kept for the next box drawn. */
  function retype(type:ElementType) {
    setNextType(type);
    if (mark) change(s => updateMark(s, mark.key, {element_type:type}));
  }
  function chooseTool(next:Tool) {
    setTool(next); setOrderCursor(1);
  }
  async function draw(box:MarkBox, keepAsDrawn:boolean) {
    if (!page || !draft) return;
    setPending(box);
    let fitted = {bbox:box, text:null as string|null, text_source:'none' as 'none'|'pdf_text_layer'};
    try {
      fitted = unwrap(await api.POST('/api/v1/documents/{document_id}/marking/fit', {params:{path:{document_id:draft.document_id}},
        body:{page_id:page.page_id, bbox:box, tighten:!keepAsDrawn}}));
    } catch (reason) { setMessage(`The box was kept as drawn: ${errorMessage(reason)}`); }
    const key = newKey();
    change(s => addMark(s, {key, page_id:page.page_id, element_type:nextType, bbox:fitted.bbox, text:fitted.text, text_source:fitted.text_source}));
    setSelected(key);
    setPending(undefined);
  }
  async function reshape(key:string, box:MarkBox) {
    if (!page || !draft) return;
    change(s => updateMark(s, key, {bbox:box}));
    const current = present?.marks.find(m => m.key === key);
    if (!page.text_layer || current?.text_source === 'typed') return;
    try {
      const read = unwrap(await api.POST('/api/v1/documents/{document_id}/marking/fit', {params:{path:{document_id:draft.document_id}},
        body:{page_id:page.page_id, bbox:box, tighten:false}}));
      patchText(key, read.text);
    } catch { /* The previous text stays; it can be corrected by hand. */ }
  }
  function orderClick(key:string) {
    change(s => moveInOrder(s, key, orderCursor - 1));
    setSelected(key);
    setOrderCursor(c => Math.min(c + 1, marks.length));
  }
  function reviewAndContinue() {
    if (!page || !present) return;
    change(s => setReviewed(s, page.page_id, true));
    const index = pages.indexOf(page);
    const reviewed = new Set([...present.reviewed, page.page_id]);
    choosePage([...pages.slice(index + 1), ...pages.slice(0, index)].find(p => !reviewed.has(p.page_id))?.page_id);
  }
  function step(offset:number) {
    if (!page) return;
    choosePage(pages[Math.max(0, Math.min(pages.length - 1, pages.indexOf(page) + offset))]?.page_id);
  }

  useEffect(() => {
    function keydown(event:KeyboardEvent) {
      const target = event.target as HTMLElement;
      if (target.closest?.('input, textarea, select, [contenteditable="true"]') || !present) return;
      if ((event.key === 'Enter' || event.key === ' ') && target.closest?.('button, a, summary')) return;
      const key = event.key, lower = key.toLowerCase();
      if (event.ctrlKey || event.metaKey) {
        if (lower === 'z') {event.preventDefault(); setHistory(h => h && (event.shiftKey ? redo(h) : undo(h)));}
        if (lower === 'y') {event.preventDefault(); setHistory(h => h && redo(h));}
        return;
      }
      if (help) {if (key === 'Escape' || key === '?') setHelp(false); return;}
      if (event.altKey && mark && (key === 'ArrowUp' || key === 'ArrowDown')) {
        event.preventDefault();
        change(s => moveInOrder(s, mark.key, mark.reading_order - 1 + (key === 'ArrowUp' ? -1 : 1)));
        return;
      }
      if (event.altKey) return;
      const type = ELEMENT_TYPES.find(t => t.key === key);
      // Box: a number chooses what is drawn next. Select: it retypes the selected box.
      if (type) {if (tool === 'select') retype(type.type); else setNextType(type.type); return;}
      if (lower === 'b' || lower === 'v' || lower === 'o') chooseTool(lower === 'b' ? 'box' : lower === 'v' ? 'select' : 'order');
      else if (lower === 'h' && !event.repeat) setHideLabels(true);
      else if (key === '?') setHelp(true);
      else if (key === '[' || key === ']') step(key === '[' ? -1 : 1);
      else if (key === 'Enter') {event.preventDefault(); reviewAndContinue();}
      else if (key === 'Escape') {if (tool === 'order') chooseTool('select'); else setSelected(null);}
      else if ((key === 'Delete' || key === 'Backspace') && mark) {event.preventDefault(); change(s => removeMark(s, mark.key)); setSelected(null);}
      else if (key.startsWith('Arrow') && mark && page) {
        event.preventDefault();
        const d = event.shiftKey ? 10 : 1, b = mark.bbox, w = b.x2 - b.x1, h = b.y2 - b.y1;
        const x = Math.max(0, Math.min(page.width - w, b.x1 + (key === 'ArrowLeft' ? -d : key === 'ArrowRight' ? d : 0)));
        const y = Math.max(0, Math.min(page.height - h, b.y1 + (key === 'ArrowUp' ? -d : key === 'ArrowDown' ? d : 0)));
        void reshape(mark.key, {x1:x, y1:y, x2:x + w, y2:y + h});
      }
    }
    const keyup = (event:KeyboardEvent) => {if (event.key.toLowerCase() === 'h') setHideLabels(false);};
    window.addEventListener('keydown', keydown); window.addEventListener('keyup', keyup);
    return () => {window.removeEventListener('keydown', keydown); window.removeEventListener('keyup', keyup);};
  });

  async function done() {
    if (!draft || !present) return;
    if (blocking.length || !present.marks.length) {setShowChecks(true); return;}
    setFreezing(true); setMessage(undefined);
    try {
      await flush();
      if (dirty.current) throw new Error('The latest changes could not be saved. Try again.');
      const result = unwrap(await api.POST('/api/v1/documents/{document_id}/marking/freeze', {params:{path:{document_id:draft.document_id}}, body:{expected_revision:revision.current}}));
      revision.current = result.draft.revision;
      setDraft(result.draft);
      if (fromSetup) {adoptMarkedDocument(result.document.document_id); navigate('/attention-tool');}
      else setFinished(result.document);
    } catch (reason) {
      setMessage(errorMessage(reason));
      if (reason instanceof ApiError && reason.code === 'revision_conflict') setSaveState('conflict');
    } finally { setFreezing(false); }
  }

  function dropOn(event:DragEvent, index:number) {
    event.preventDefault();
    if (dragging) change(s => moveInOrder(s, dragging, index));
    setDragging(undefined);
  }

  if (loadError) return <div className="docattentionlens-marking-message"><Message error>{loadError}</Message><Link to="/attention-tool">Back to Investigate</Link></div>;
  if (!draft || !present || !page) return <div className="docattentionlens-marking-message"><Message>Loading the document’s pages…</Message></div>;
  if (finished) return <div className="docattentionlens-marking-message docattentionlens-marking-finished">
    <h1>Elements saved</h1>
    <p>{finished.element_count} elements on {finished.total_pages} pages of {finished.filename} are saved as version {finished.annotation_version.slice(0, 8)}. Investigations of this version read attention by element.</p>
    <div className="docattentionlens-actions"><button className="docattentionlens-button docattentionlens-primary" onClick={() => {adoptMarkedDocument(finished.document_id); navigate('/attention-tool');}}>Start an investigation with these pages</button>
      <button className="docattentionlens-button" onClick={() => setFinished(undefined)}>Keep editing</button><Link className="docattentionlens-quiet" to="/artifacts/inputs">Library</Link></div>
  </div>;

  const reviewedCount = pages.filter(p => present.reviewed.includes(p.page_id)).length;
  const back = fromSetup ? '/attention-tool' : '/artifacts/inputs';
  const toolButton = (value:Tool, label:string, key:string) => <button type="button" className="docattentionlens-button" aria-pressed={tool === value} onClick={() => chooseTool(value)}>{label}<kbd>{key}</kbd></button>;
  const manual = present.manual.includes(page.page_id);
  return <div className="docattentionlens-marking">
    <div className="docattentionlens-marking-narrow"><Message>Marking elements needs a window at least 900 px wide.</Message></div>
    <header className="docattentionlens-marking-bar">
      <strong className="docattentionlens-marking-title">Mark elements</strong>
      <span className="docattentionlens-marking-context">{draft.filename} · page {page.page_number} of {pages.length}</span>
      <div className="docattentionlens-marking-tools" role="group" aria-label="Tool">{toolButton('box', 'Box', 'B')}{toolButton('select', 'Select', 'V')}{toolButton('order', 'Order', 'O')}</div>
      <label className="docattentionlens-marking-next">New boxes<select value={nextType} onChange={e => setNextType(e.target.value as ElementType)}>
        {ELEMENT_TYPES.map(t => <option key={t.type} value={t.type}>{t.label}</option>)}</select></label>
      <button type="button" className="docattentionlens-button" aria-label="Undo" title="Undo (Ctrl+Z)" disabled={!history?.past.length} onClick={() => setHistory(h => h && undo(h))}>↶</button>
      <button type="button" className="docattentionlens-button" aria-label="Redo" title="Redo (Ctrl+Shift+Z)" disabled={!history?.future.length} onClick={() => setHistory(h => h && redo(h))}>↷</button>
      <button type="button" className="docattentionlens-quiet" onClick={() => setHelp(true)}>Shortcuts</button>
      <span className="docattentionlens-marking-progress" role="status" data-save={saveState}>{reviewedCount} of {pages.length} pages reviewed · {present.marks.length} elements · {saveState === 'saving' && <Busy/>}{SAVE_LABEL[saveState]}</span>
      <Link className="docattentionlens-quiet" to={back} onClick={() => void flush()}>{fromSetup ? 'Back to setup' : 'Library'}</Link>
      <button type="button" className="docattentionlens-button docattentionlens-primary" disabled={freezing || saveState === 'conflict'} onClick={() => void done()}>{freezing && <Busy/>}Done</button>
    </header>
    {saveState === 'conflict' && <div className="docattentionlens-marking-alert"><Message error>These elements were changed in another window, so this window stopped saving. <button className="docattentionlens-link-button" onClick={() => window.location.reload()}>Reload the latest elements</button></Message></div>}
    {message && saveState !== 'conflict' && <div className="docattentionlens-marking-alert"><Message error>{message} <button className="docattentionlens-link-button" onClick={() => setMessage(undefined)}>Dismiss</button></Message></div>}
    <div className="docattentionlens-marking-body">
      <nav className="docattentionlens-marking-pages" aria-label="Pages">
        <h2>Pages</h2>
        <ol>{pages.map(p => {
          const count = present.marks.filter(m => m.page_id === p.page_id).length, reviewed = present.reviewed.includes(p.page_id);
          return <li key={p.page_id}><button type="button" aria-current={p.page_id === page.page_id ? 'page' : undefined} onClick={() => choosePage(p.page_id)}>
            <img src={p.image_url} alt="" loading="lazy" style={{aspectRatio:`${p.width} / ${p.height}`}}/>
            <span>Page {p.page_number}</span>
            <small className={reviewed ? 'is-reviewed' : ''}>{reviewed ? `✓ Reviewed · ${count} element${count === 1 ? '' : 's'}` : count ? `${count} element${count === 1 ? '' : 's'}` : 'Not started'}</small>
          </button></li>;})}</ol>
      </nav>
      <section className="docattentionlens-marking-stage" aria-label="Page">
        <MarkingCanvas key={page.page_id} page={page} marks={marks} selected={selected} tool={tool} hideLabels={hideLabels} pending={pending}
          onSelect={setSelected} onDraw={(box, keep) => void draw(box, keep)} onGeometry={(key, box) => void reshape(key, box)} onOrderClick={orderClick}/>
        <footer className="docattentionlens-marking-hint">
          {tool === 'order' ? <span>Click boxes in reading order: the next click becomes number {orderCursor}. <kbd>Esc</kbd> stops.</span>
            : tool === 'box' ? <span>Drag to draw a <strong>{typeInfo(nextType).label.toLowerCase()}</strong> box · <kbd>1</kbd>–<kbd>9</kbd> type of the next box · <kbd>Alt</kbd> keep box as drawn · <kbd>V</kbd> select to edit · <kbd>?</kbd> all shortcuts</span>
            : <span>Drag a box to move it, or a handle to resize · <kbd>1</kbd>–<kbd>9</kbd> retype the selected box · <kbd>Tab</kbd> next box · <kbd>?</kbd> all shortcuts</span>}
          <button type="button" className="docattentionlens-button" onClick={reviewAndContinue}>{present.reviewed.includes(page.page_id) ? 'Next page to review' : 'Page reviewed, next page'}<kbd>Enter</kbd></button>
        </footer>
      </section>
      <aside className="docattentionlens-marking-side" aria-label="Element">
        {mark ? <section className="docattentionlens-marking-section">
          <h2>Element {mark.reading_order} · {typeInfo(mark.element_type).label}</h2>
          <div className="docattentionlens-marking-types" role="radiogroup" aria-label="Element type">{ELEMENT_TYPES.map(t =>
            <button type="button" key={t.type} role="radio" aria-checked={mark.element_type === t.type} onClick={() => retype(t.type)}>
              <kbd>{t.key ?? ''}</kbd>{t.label}</button>)}</div>
          <p className="docattentionlens-marking-definition">{typeInfo(mark.element_type).definition}</p>
          <label className="docattentionlens-marking-text">Text{mark.text_source === 'pdf_text_layer' ? ' · from the PDF text layer' : mark.text_source === 'typed' ? ' · typed' : page.text_layer ? '' : ' · this page is an image, so there is no text layer'}
            <textarea key={mark.key + (mark.text ?? '')} rows={4} defaultValue={mark.text ?? ''} placeholder="No text"
              onBlur={e => {const text = e.target.value; if (text !== (mark.text ?? '')) change(s => updateMark(s, mark.key, {text, text_source:'typed'}));}}/></label>
          <p className="docattentionlens-muted">{Math.round(mark.bbox.x2 - mark.bbox.x1)} × {Math.round(mark.bbox.y2 - mark.bbox.y1)} px · about {Math.max(0, positionsCovered(mark.bbox, page)).toFixed(1)} of {GRID * GRID} input positions on a {GRID} × {GRID} grid</p>
          {pageChecks.filter(c => c.keys.includes(mark.key)).map(c => <p key={c.message} className={`docattentionlens-marking-check is-${c.level}`}>{c.message}</p>)}
          <button type="button" className="docattentionlens-button" onClick={() => {change(s => removeMark(s, mark.key)); setSelected(null);}}>Delete element</button>
        </section> : <section className="docattentionlens-marking-section docattentionlens-marking-guide">
          <h2>How to mark this page</h2>
          <ol><li>Choose a type with <kbd>1</kbd>–<kbd>9</kbd> or the list above the page.</li><li>Drag a box around one element. It tightens to the content inside it.</li><li>Press <kbd>Enter</kbd> when the page is done.</li></ol>
          <p>One box per paragraph, heading, list item, table or figure. Keep boxes tight and avoid overlaps: overlapping areas are split equally between elements. Page headers, footers and page numbers are elements too.</p>
        </section>}
        <section className="docattentionlens-marking-section">
          <h2>Reading order</h2>
          <p className="docattentionlens-muted">{manual ? <>Set by hand. <button type="button" className="docattentionlens-link-button" onClick={() => change(s => automaticOrder(s, page.page_id))}>Use automatic order</button></>
            : 'Automatic: top to bottom, column by column. Drag to change it, or use Order and click boxes in sequence.'}</p>
          {marks.length ? <ol className="docattentionlens-marking-order">{marks.map((m, i) =>
            <li key={m.key} draggable onDragStart={() => setDragging(m.key)} onDragEnd={() => setDragging(undefined)} onDragOver={e => e.preventDefault()} onDrop={e => dropOn(e, i)}
              className={`${m.key === selected ? 'is-selected' : ''}${m.key === dragging ? ' is-dragging' : ''}`}>
              <button type="button" onClick={() => setSelected(m.key)} aria-pressed={m.key === selected}>
                <span aria-hidden="true" className="docattentionlens-marking-grip">⋮⋮</span><b>{m.reading_order}</b>{typeInfo(m.element_type).label}<small>{m.text?.split('\n')[0].slice(0, 40)}</small></button></li>)}</ol>
            : <p>No elements on this page yet.</p>}
        </section>
        <section className="docattentionlens-marking-section">
          <h2>Checks</h2>
          {!present.marks.length ? <p>Mark at least one element before choosing Done.</p>
            : (showChecks ? allChecks : allChecks.filter(c => c.pageId === page.page_id)).length === 0 ? <p>No problems {showChecks ? 'in this document' : 'on this page'}.</p>
            : <ul className="docattentionlens-marking-checks">{(showChecks ? allChecks : allChecks.filter(c => c.pageId === page.page_id)).map(c =>
              <li key={c.message} className={`is-${c.level}`}><button type="button" className="docattentionlens-link-button" onClick={() => {choosePage(c.pageId); setSelected(c.keys[0] ?? null);}}>{c.message}</button></li>)}</ul>}
          {blocking.length > 0 && <p className="docattentionlens-marking-check is-blocking">{blocking.length} problem{blocking.length === 1 ? '' : 's'} must be fixed before Done.</p>}
          <button type="button" className="docattentionlens-link-button" onClick={() => setShowChecks(v => !v)}>{showChecks ? 'Show this page only' : 'Check every page'}</button>
        </section>
      </aside>
    </div>
    {help && <div className="docattentionlens-marking-help" role="dialog" aria-modal="true" aria-label="Shortcuts" onClick={() => setHelp(false)}>
      <div onClick={e => e.stopPropagation()}><h2>Shortcuts</h2><dl>{SHORTCUTS.map(([k, v]) => <div key={k}><dt><kbd>{k}</kbd></dt><dd>{v}</dd></div>)}</dl>
        <button type="button" className="docattentionlens-button" autoFocus onClick={() => setHelp(false)}>Close</button></div>
    </div>}
  </div>;
}
