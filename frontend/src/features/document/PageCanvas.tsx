import { useEffect, useRef, useState } from 'react';
import { type DisplaySettings, type Element, type Page } from '../../api/client';
import { Message } from '../../app/App';
import { clampZoom, fitContain, hitTest, viewFromCentre, viewTransform } from '../../geometry/transform';
import type { Capture, Row } from './RetrievalPanel';
import type { Generation, Attention } from './GenerationPanel';
import { ATTENTION_FILL, attentionAlpha, heatColor, rangeFloor, rgb, scalePosition, type AttentionMode } from './attentionDisplay';
import { AttentionLayer } from './AttentionLayer';
import { TableOverlay } from '../tables/TableOverlay';
import type { Table, TableGrounding } from '../tables/tableGeometry';

type Props = {page: Page; elements: Element[]; settings: DisplaySettings;
  selected: string|null; source: string|null; target: string|null; support: Set<string>;
  onView: (settings: DisplaySettings) => void;
  evidence?:{positions:Capture['pages'][number]['positions'];row:Row};
  attention?:{positions:Generation['positions'];row:Attention;outlineOnly?:boolean}; attentionMode?:AttentionMode;
  imageUrl?:string; inputLabel?:string; scaleMax?:number;
  focusAction?:{id:number;box:Element['display_bbox']};
  /** A click (not a drag) chooses the patch under the pointer. */
  pick?:{positions:{index:number;bbox:number[]}[];onPick:(index:number)=>void};
  /** Without patch picking, a click chooses the smallest element under the pointer. */
  onElement?:(element:Element)=>void;
  /** Document-aware analysis: element attention mass drawn over element regions, with boundaries and labels. */
  regions?:ElementRegions;
  /** The coloured patch, or element when elements carry the colour, under a pointer that is not dragging. */
  onHover?:(hit:PageHit|undefined)=>void;
  /** Tables located on this page: their outline and cell grid (Layout boxes) and the question's grounding (Designated support). */
  tables?:{items:Table[];grounding?:TableGrounding|null;highlight?:string}};

export type PageHit = {kind:'patch';index:number} | {kind:'element';id:string;label:string};

/** `owned`: elements that own captured patches on this page, so a missing mass means outside the row's support, not unmapped. */
export type ElementRegions = {masses:Record<string,number>; maximum:number; labelled:Set<string>; mode:AttentionMode; opacity:number; owned?:Set<string>; quantity?:string};

export function PageCanvas({page, elements, settings, selected, source, target, support, onView, evidence, attention, attentionMode='overlay', imageUrl, inputLabel, scaleMax, focusAction, pick, onElement, regions, onHover, tables}: Props) {
  const container = useRef<HTMLDivElement>(null);
  const [size, setSize] = useState({width:600, height:680});
  // Load state belongs to one image URL. A reset effect raced cached images, whose load
  // event can fire before the effect, and left overlays hidden behind "Loading" for good.
  const href = imageUrl ?? page.image_url;
  const [loadedHref, setLoadedHref] = useState<string>();
  const [failedHref, setFailedHref] = useState<string>();
  const loaded = loadedHref === href, failed = failedHref === href;
  const drag = useRef<{x:number;y:number;cx:number;cy:number;moved:boolean}|undefined>(undefined);
  const hovered = useRef('');
  useEffect(() => {
    const observer = new ResizeObserver(entries => {const box = entries[0].contentRect; if(box.width>0&&box.height>0) setSize({width:box.width,height:box.height});});
    if (container.current) observer.observe(container.current);
    return () => observer.disconnect();
  }, []);
  const fit = fitContain(page, size);
  const latestView=useRef({settings,onView});
  latestView.current={settings,onView};
  const lastFocus=useRef(focusAction?.id);
  useEffect(()=>{
    if(!fit||!focusAction||lastFocus.current===focusAction.id||!focusAction.box)return;
    lastFocus.current=focusAction.id;
    const b=focusAction.box;
    const scale=Math.min(size.width*.85/(b.x2-b.x1),size.height*.85/(b.y2-b.y1));
    const {settings:s,onView:update}=latestView.current;
    update({...s,canvas_fit:'manual',zoom:clampZoom(scale/fit.scale),center_x:(b.x1+b.x2)/2,center_y:(b.y1+b.y2)/2});
  },[focusAction,size.width,size.height,page.width,page.height]);
  useEffect(()=>{
    if(!fit||settings.canvas_fit!=='width')return;
    const zoom=clampZoom(size.width/(page.width*fit.scale));
    if(Math.abs(zoom-(settings.zoom??1))>.0001)latestView.current.onView({...latestView.current.settings,zoom});
  },[settings.canvas_fit,size.width,size.height,page.width,page.height]);
  if(!fit) return <div className="docattentionlens-canvas" ref={container}><Message>Waiting for the page viewport…</Message></div>;
  const center = {x:settings.center_x ?? page.width/2, y:settings.center_y ?? page.height/2};
  const view = viewFromCentre(fit, size, settings.zoom ?? 1, center);
  const transform = viewTransform(fit, size, view);
  const colormap=settings.attention_colormap??'inferno';
  const regionFloor=regions?rangeFloor(Object.values(regions.masses),settings.attention_range_floor):0;
  const similarityScale=scaleMax??(evidence?Math.max(...evidence.row.similarities.map(v=>Math.abs(v??0)),1e-8):1);
  function point(event: React.PointerEvent<SVGSVGElement>) {
    const rect = event.currentTarget.getBoundingClientRect();
    return {x:event.clientX-rect.left,y:event.clientY-rect.top};
  }
  // Only what carries the colour is read: patches when they are drawn, otherwise element fills.
  function hover(at:{x:number;y:number}|undefined) {
    if(!onHover) return;
    let hit:PageHit|undefined;
    if(at&&loaded&&attention&&!attention.outlineOnly&&attentionMode!=='original'){
      const index=hitTest(transform,at,attention.positions.filter(p=>p.bbox&&p.index<attention.row.target_count).map(p=>({item:p.index,box:{x1:p.bbox![0],y1:p.bbox![1],x2:p.bbox![2],y2:p.bbox![3]}})))[0];
      if(index!==undefined) hit={kind:'patch',index};
    } else if(at&&loaded&&regions&&regions.mode!=='original'){
      const element=hitTest(transform,at,elements.filter(e=>e.display_bbox&&e.element_id!==selected&&regions.masses[e.element_id]!==undefined).map(e=>({item:e,box:e.display_bbox!})))[0];
      if(element) hit={kind:'element',id:element.element_id,label:`${element.element_type} ${element.native_id}`};
    }
    const key=hit?(hit.kind==='patch'?`p${hit.index}`:`e${hit.id}`):'';
    if(key===hovered.current) return;
    hovered.current=key;
    onHover(hit);
  }
  return <div className={`docattentionlens-canvas ${attention||evidence||regions?'has-quantitative-overlay':''} ${regions?'has-regions':''} ${pick?'is-pickable':onElement?'is-element-pickable':''}`} ref={container} data-testid="page-canvas">
    {failed ? <Message error>Original page image is unavailable or differs from this saved version. Recheck the source asset before continuing.</Message> : <>
      {!loaded && <span className="docattentionlens-canvas-loading" role="status">Loading original page…</span>}
      <svg width="100%" height="100%" viewBox={`0 0 ${size.width} ${size.height}`} aria-label={`${inputLabel??'Original page'} ${page.page_number} with layout overlays`}
        onPointerDown={event => {if(event.button !== 0) return; const p=point(event); drag.current={...p,cx:center.x,cy:center.y,moved:false}; event.currentTarget.setPointerCapture(event.pointerId);}}
        onPointerMove={event => {const active=drag.current; if(!active) {hover(point(event)); return;} const p=point(event); const dx=p.x-active.x, dy=p.y-active.y; if(Math.hypot(dx,dy)>4) active.moved=true; if(active.moved) onView({...settings,canvas_fit:'manual',center_x:Math.max(0,Math.min(page.width,active.cx-dx/transform.scale)),center_y:Math.max(0,Math.min(page.height,active.cy-dy/transform.scale))});}}
        onPointerUp={event => {const active=drag.current; drag.current=undefined; if(!active || active.moved || !loaded) return;
          if(pick){const patch=hitTest(transform,point(event),pick.positions.map(p=>({item:p.index,box:{x1:p.bbox[0],y1:p.bbox[1],x2:p.bbox[2],y2:p.bbox[3]}})))[0]; if(patch!==undefined) pick.onPick(patch); return;}
          if(onElement){const element=hitTest(transform,point(event),elements.filter(e=>e.display_bbox).map(e=>({item:e,box:e.display_bbox!})))[0]; if(element) onElement(element);}}}
        onPointerCancel={() => {drag.current=undefined;}} onPointerLeave={() => hover(undefined)}>
        <g transform={`translate(${transform.offsetX} ${transform.offsetY}) scale(${transform.scale})`}>
          <image href={href} width={page.width} height={page.height} preserveAspectRatio="none" onLoad={() => setLoadedHref(href)} onError={() => setFailedHref(href)}/>
          {loaded&&evidence&&evidence.positions.map(position=>{
            if(!position.bbox||!position.eligible) return null;
            const [x1,y1,x2,y2]=position.bbox;
            const value=evidence.row.similarities[position.index];
            if(value===null) return null;
            const winner=position.index===evidence.row.winner_index;
            const opacity=settings.similarity_opacity??.65;
            return <rect key={`patch-${position.index}`} data-testid="similarity-patch" data-position-index={position.index} x={x1} y={y1} width={x2-x1} height={y2-y1} fill={value>=0?'#b080d1':'#247ed5'} fillOpacity={Math.min(opacity,Math.abs(value)/similarityScale*opacity)} stroke={winner?'#ffc400':'none'} strokeWidth={winner?3:0} vectorEffect="non-scaling-stroke" pointerEvents="none"><title>Patch {position.index}: {value.toFixed(6)}{winner?' · winner':''}</title></rect>;
          })}
          {/* Element mass uses the patch palette with its own maximum; the selected element shows its patches instead. */}
          {loaded&&regions&&regions.mode!=='original'&&elements.map(element=>{
            const box=element.display_bbox, mass=regions.masses[element.element_id];
            if(!box||mass===undefined||element.element_id===selected) return null;
            const t=scalePosition(mass,regions.maximum,regionFloor);
            return <rect key={`region-${element.key}`} data-testid="element-region" data-element-id={element.element_id} data-mass={mass} x={box.x1} y={box.y1} width={box.x2-box.x1} height={box.y2-box.y1}
              fill={regions.mode==='heatmap'||regions.mode==='blend'?rgb(heatColor(t,colormap)):ATTENTION_FILL}
              fillOpacity={regions.mode==='heatmap'?.75:regions.mode==='blend'?regions.opacity:attentionAlpha(mass,regions.maximum,regions.opacity,regionFloor)} pointerEvents="none"/>;
          })}
          {loaded&&attention&&<AttentionLayer positions={attention.positions} row={attention.row} outlineOnly={attention.outlineOnly} maximum={scaleMax??attention.row.display_max}
            mode={attentionMode} opacity={settings.attention_opacity??.65} smooth={settings.attention_smooth??false} showPatches={settings.attention_show_patches??false}
            colormap={colormap} floor={settings.attention_range_floor}/>}
          {loaded && elements.map(element => {const box=element.display_bbox; if(!box) return null;
            const isSelected=element.element_id===selected, isSource=element.element_id===source, isTarget=element.element_id===target;
            const isSupport=(settings.show_support ?? false)&&support.has(element.element_id);
            const flagged=(settings.show_geometry_flags ?? true)&&element.geometry_flags.length>0;
            // Boundaries are the analysis in document-aware mode, so every annotated element is drawn.
            if(!regions&&!(settings.show_boxes ?? true)&&!isSelected&&!isSource&&!isTarget&&!isSupport&&!flagged) return null;
            const mass=regions?.masses[element.element_id];
            const unmapped=!!regions&&mass===undefined;
            const missing=regions?.owned?.has(element.element_id)?' · outside this row’s support':' · no mapped patches';
            const label=`Element ${element.native_id} · ${element.element_type}${regions?(unmapped?missing:` · ${(regions.quantity??'Element attention mass').toLowerCase()} ${mass!.toPrecision(4)}`):''}`;
            return <g key={element.key}>
              <rect data-element-id={element.element_id} className={`docattentionlens-box ${isSelected?'is-selected':''} ${isSource?'is-source':''} ${isTarget?'is-target':''} ${isSupport?'is-support':''} ${flagged?'is-flagged':''} ${unmapped?'is-unmapped':''}`}
                x={box.x1} y={box.y1} width={box.x2-box.x1} height={box.y2-box.y1} vectorEffect="non-scaling-stroke" aria-label={label}><title>{label}{flagged?' · source geometry flagged':''}</title></rect>
              {regions&&(regions.labelled.has(element.element_id)||isSelected)&&<text className="docattentionlens-region-label" data-testid="element-region-label" x={box.x1+4/transform.scale} y={box.y1+14/transform.scale} fontSize={12/transform.scale}>{element.element_type}{unmapped?missing:` · ${mass!.toPrecision(3)}`}</text>}
              {!regions&&(settings.show_reading_order ?? true)&&element.reading_order!==null && <text className="docattentionlens-box-order" x={box.x1+8/transform.scale} y={box.y1+18/transform.scale} fontSize={13/transform.scale}>{element.reading_order}</text>}
              {(isSource||isTarget) && <text className="docattentionlens-box-role" x={box.x1+8/transform.scale} y={box.y2-8/transform.scale} fontSize={14/transform.scale}>{isSource?'SOURCE':''}{isSource&&isTarget?' / ':''}{isTarget?'TARGET':''}</text>}
            </g>;})}
          {loaded&&!!tables?.items.length&&<TableOverlay tables={tables.items} grounding={tables.grounding} showGrid={settings.show_boxes ?? true} showGrounding={settings.show_support ?? false} scale={transform.scale} highlight={tables.highlight}/>}
        </g>
      </svg>
    </>}
  </div>;
}
