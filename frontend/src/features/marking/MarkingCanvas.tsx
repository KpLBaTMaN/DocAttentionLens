import { useEffect, useRef, useState, type KeyboardEvent, type PointerEvent } from 'react';
import { Message } from '../../app/App';
import { IDENTITY_VIEW, boxToRect, fitContain, hitTest, toSource, viewTransform, zoomAt, type Point, type View } from '../../geometry/transform';
import { normalise, typeInfo, type Mark, type MarkBox, type MarkingPage } from './markingModel';

export type Tool = 'select'|'box'|'order';
type Handle = 'nw'|'n'|'ne'|'e'|'se'|'s'|'sw'|'w';
const HANDLES:Handle[] = ['nw', 'n', 'ne', 'e', 'se', 's', 'sw', 'w'];
type Gesture = {at:Point; moved:boolean} & ({kind:'draw'; start:Point} | {kind:'click'} | {kind:'pan'; view:View; deselect:boolean}
  | {kind:'move'; key:string; origin:MarkBox} | {kind:'resize'; key:string; handle:Handle; origin:MarkBox});

function handlePoint(rect:{left:number;top:number;width:number;height:number}, handle:Handle):Point {
  return {x:rect.left + (handle.includes('w') ? 0 : handle.includes('e') ? rect.width : rect.width / 2),
    y:rect.top + (handle.includes('n') ? 0 : handle.includes('s') ? rect.height : rect.height / 2)};
}

/**
 * The page with its marked boxes. Box draws a new element; Select moves and resizes; Order numbers boxes in the
 * order they are clicked. Drawing and editing happen in page-image pixels through the shared display transform.
 */
export function MarkingCanvas({page, marks, selected, tool, hideLabels, pending, onSelect, onDraw, onGeometry, onOrderClick}:{
  page:MarkingPage; marks:Mark[]; selected:string|null; tool:Tool; hideLabels:boolean;
  /** A drawn box waiting for its fitted geometry. */
  pending?:MarkBox;
  onSelect:(key:string|null)=>void; onDraw:(box:MarkBox, keepAsDrawn:boolean)=>void;
  onGeometry:(key:string, box:MarkBox)=>void; onOrderClick:(key:string)=>void;
}) {
  const container = useRef<HTMLDivElement>(null);
  // A starting size until the first measurement, as PageCanvas uses.
  const [size, setSize] = useState({width:600, height:680});
  const [view, setView] = useState<View>(IDENTITY_VIEW);
  const [live, setLive] = useState<{key?:string; box:MarkBox}>();
  const gesture = useRef<Gesture>(undefined);
  const pointer = useRef<Point>(undefined);
  const space = useRef(false);
  useEffect(() => {
    const observer = new ResizeObserver(entries => {const box = entries[0].contentRect; if (box.width > 0 && box.height > 0) setSize({width:box.width, height:box.height});});
    if (container.current) observer.observe(container.current);
    return () => observer.disconnect();
  }, []);
  // Space held pans with the primary button, as in drawing tools.
  useEffect(() => {
    const typing = (event:globalThis.KeyboardEvent) => !!(event.target as HTMLElement).closest?.('input, textarea, select, button');
    const down = (event:globalThis.KeyboardEvent) => {if (event.key === ' ' && !typing(event)) {space.current = true; event.preventDefault();}};
    const up = (event:globalThis.KeyboardEvent) => {if (event.key === ' ') space.current = false;};
    window.addEventListener('keydown', down); window.addEventListener('keyup', up);
    return () => {window.removeEventListener('keydown', down); window.removeEventListener('keyup', up);};
  }, []);
  const fit = fitContain(page, size);
  const transform = fit ? viewTransform(fit, size, view) : undefined;
  // The wheel zooms about the pointer; React's wheel listener is passive, so this one is attached directly.
  const latest = useRef({fit, size});
  latest.current = {fit, size};
  useEffect(() => {
    const element = container.current;
    if (!element) return;
    const wheel = (event:WheelEvent) => {
      const {fit:current, size:viewport} = latest.current;
      if (!current) return;
      event.preventDefault();
      const rect = element.getBoundingClientRect();
      setView(v => zoomAt(current, viewport, v, event.deltaY < 0 ? 1.15 : 1 / 1.15, {x:event.clientX - rect.left, y:event.clientY - rect.top}));
    };
    element.addEventListener('wheel', wheel, {passive:false});
    return () => element.removeEventListener('wheel', wheel);
  }, []);
  if (!fit || !transform) return <div className="docattentionlens-marking-canvas" ref={container}><Message>Waiting for the page viewport…</Message></div>;
  const t = transform;
  const local = (event:PointerEvent<SVGSVGElement>):Point => {const r = event.currentTarget.getBoundingClientRect(); return {x:event.clientX - r.left, y:event.clientY - r.top};};
  const onPage = (p:Point):Point => {const s = toSource(t, p); return {x:Math.max(0, Math.min(page.width, s.x)), y:Math.max(0, Math.min(page.height, s.y))};};
  const hits = (p:Point) => hitTest(t, p, marks.map(m => ({item:m.key, box:m.bbox})));
  const current = marks.find(m => m.key === selected);
  const handleAt = (p:Point) => {
    if (!current) return undefined;
    const rect = boxToRect(t, current.bbox);
    return HANDLES.find(h => {const q = handlePoint(rect, h); return Math.abs(q.x - p.x) <= 7 && Math.abs(q.y - p.y) <= 7;});
  };

  function down(event:PointerEvent<SVGSVGElement>) {
    const at = local(event);
    if (event.button === 1 || (event.button === 0 && space.current)) gesture.current = {kind:'pan', at, moved:false, view, deselect:false};
    else if (event.button !== 0) return;
    else {
      const handle = handleAt(at);
      if (handle && current) gesture.current = {kind:'resize', at, moved:false, key:current.key, handle, origin:current.bbox};
      else if (tool === 'box') gesture.current = {kind:'draw', at, moved:false, start:onPage(at)};
      else if (tool === 'order') gesture.current = {kind:'click', at, moved:false};
      else {
        const key = hits(at)[0];
        if (key) onSelect(key);
        const mark = marks.find(m => m.key === key);
        gesture.current = mark ? {kind:'move', at, moved:false, key:mark.key, origin:mark.bbox} : {kind:'pan', at, moved:false, view, deselect:true};
      }
    }
    event.currentTarget.setPointerCapture?.(event.pointerId);
  }

  function move(event:PointerEvent<SVGSVGElement>) {
    const at = local(event);
    pointer.current = at;
    const g = gesture.current;
    if (!g) return;
    if (!g.moved && Math.hypot(at.x - g.at.x, at.y - g.at.y) > 4) g.moved = true;
    if (!g.moved) return;
    if (g.kind === 'draw') setLive({box:normalise({x1:g.start.x, y1:g.start.y, x2:onPage(at).x, y2:onPage(at).y}, page)});
    else if (g.kind === 'pan') setView({...g.view, panX:g.view.panX + at.x - g.at.x, panY:g.view.panY + at.y - g.at.y});
    else if (g.kind === 'move') {
      const o = g.origin, w = o.x2 - o.x1, h = o.y2 - o.y1;
      const x = Math.max(0, Math.min(page.width - w, o.x1 + (at.x - g.at.x) / t.scale)), y = Math.max(0, Math.min(page.height - h, o.y1 + (at.y - g.at.y) / t.scale));
      setLive({key:g.key, box:normalise({x1:x, y1:y, x2:x + w, y2:y + h}, page)});
    } else if (g.kind === 'resize') {
      const p = onPage(at), b = {...g.origin};
      if (g.handle.includes('w')) b.x1 = p.x; if (g.handle.includes('e')) b.x2 = p.x;
      if (g.handle.includes('n')) b.y1 = p.y; if (g.handle.includes('s')) b.y2 = p.y;
      setLive({key:g.key, box:normalise(b, page)});
    }
  }

  function up(event:PointerEvent<SVGSVGElement>) {
    const g = gesture.current, result = live;
    gesture.current = undefined;
    setLive(undefined);
    if (!g) return;
    if (!g.moved) {
      if (g.kind === 'resize' || (g.kind === 'pan' && !g.deselect)) return;
      const key = hits(local(event))[0] ?? null;
      if (tool === 'order') {if (key) onOrderClick(key);}
      else onSelect(key);
      return;
    }
    if (!result) return;
    const {x1, y1, x2, y2} = result.box;
    if (g.kind === 'draw' && x2 - x1 >= 2 && y2 - y1 >= 2) onDraw(result.box, event.altKey);
    else if ((g.kind === 'move' || g.kind === 'resize') && x2 - x1 >= 2 && y2 - y1 >= 2) onGeometry(g.key, result.box);
  }

  /** Tab moves through the boxes under the pointer (smallest first), or through the page in reading order. */
  function cycle(event:KeyboardEvent<HTMLDivElement>) {
    if (event.key !== 'Tab' || event.ctrlKey || event.metaKey || event.altKey || !marks.length) return;
    const under = pointer.current ? hits(pointer.current) : [];
    const list = under.length > 1 ? under : marks.map(m => m.key);
    const index = selected ? list.indexOf(selected) : -1;
    event.preventDefault();
    onSelect(list[(index + (event.shiftKey ? -1 : 1) + list.length) % list.length]);
  }

  const zoomBy = (factor:number) => setView(v => zoomAt(fit, size, v, factor, {x:size.width / 2, y:size.height / 2}));
  const shown = (m:Mark) => live?.key === m.key ? live.box : m.bbox;
  const selectedRect = current ? boxToRect(t, shown(current)) : undefined;
  return <div className={`docattentionlens-marking-canvas is-${tool}`} ref={container} tabIndex={0} onKeyDown={cycle}
    aria-label={`Page ${page.page_number}. ${tool === 'box' ? 'Drag to draw a box.' : tool === 'order' ? 'Click boxes in reading order.' : 'Click a box to select it; drag to move.'} Tab moves between boxes.`}>
    <svg width="100%" height="100%" onPointerDown={down} onPointerMove={move} onPointerUp={up}
      onPointerCancel={() => {gesture.current = undefined; setLive(undefined);}} onPointerLeave={() => {pointer.current = undefined;}}>
      <g transform={`translate(${t.offsetX} ${t.offsetY}) scale(${t.scale})`}>
        <image href={page.image_url} width={page.width} height={page.height} preserveAspectRatio="none"/>
        {marks.map(m => {const b = shown(m); return <rect key={m.key} data-mark-key={m.key} className={`docattentionlens-mark${m.key === selected ? ' is-selected' : ''}`}
          x={b.x1} y={b.y1} width={b.x2 - b.x1} height={b.y2 - b.y1} vectorEffect="non-scaling-stroke"><title>{m.reading_order} · {typeInfo(m.element_type).label}</title></rect>;})}
        {pending && <rect className="docattentionlens-mark is-pending" x={pending.x1} y={pending.y1} width={pending.x2 - pending.x1} height={pending.y2 - pending.y1} vectorEffect="non-scaling-stroke"/>}
        {live && !live.key && <rect className="docattentionlens-mark is-drawing" x={live.box.x1} y={live.box.y1} width={live.box.x2 - live.box.x1} height={live.box.y2 - live.box.y1} vectorEffect="non-scaling-stroke"/>}
      </g>
      {!hideLabels && marks.map(m => {
        // Labels sit in the margin left of the box when there is room, so they do not cover its text.
        const rect = boxToRect(t, shown(m)), label = `${m.reading_order} ${typeInfo(m.element_type).label.toLowerCase()}`;
        const width = label.length * 6.6 + 10, x = rect.left - width - 3 >= 0 ? rect.left - width - 3 : rect.left;
        return <g key={m.key} className={`docattentionlens-mark-chip${m.key === selected ? ' is-selected' : ''}`} pointerEvents="none">
          <rect x={x} y={rect.top} width={width} height={16}/><text x={x + 5} y={rect.top + 12}>{label}</text></g>;
      })}
      {selectedRect && HANDLES.map(h => {const q = handlePoint(selectedRect, h); return <rect key={h} className={`docattentionlens-mark-handle is-${h}`} data-handle={h} x={q.x - 4} y={q.y - 4} width={8} height={8}/>;})}
    </svg>
    <div className="docattentionlens-marking-zoom" role="group" aria-label="Zoom">
      <button type="button" className="docattentionlens-quiet" aria-label="Zoom out" onClick={() => zoomBy(1 / 1.25)}>−</button>
      <button type="button" className="docattentionlens-quiet" title="Fit the page" onClick={() => setView(IDENTITY_VIEW)}>{view === IDENTITY_VIEW ? 'Fit page' : 'Fit'} · {Math.round(t.scale * 100)}%</button>
      <button type="button" className="docattentionlens-quiet" aria-label="Zoom in" onClick={() => zoomBy(1.25)}>+</button>
    </div>
  </div>;
}
