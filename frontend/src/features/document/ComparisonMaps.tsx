import { useEffect, useRef, useState } from 'react';
import type { DisplaySettings } from '../../api/client';
import type { components } from '../../api/generated/schema';
import { AttentionLayer, type MapPosition } from './AttentionLayer';
import { attentionWeightLabel, mapGradient, type AttentionMode } from './attentionDisplay';
import { Drawer } from './WorkspaceControls';
import { AttentionToolbar } from './AttentionToolbar';
import { downloadBlob,exportPageFigure,type ResearchReceipt } from './receiptExport';

type Comparison = components['schemas']['Calculation']['comparisons'];
type Page = {width:number;height:number;image_url:string;page_number:number;imageLabel?:string};
type Props = {comparisons:NonNullable<Comparison>;positions:MapPosition[];page:Page;
  scale:number;mode:AttentionMode;settings:DisplaySettings;onView:(settings:DisplaySettings)=>void;onMode?:(mode:AttentionMode)=>void;encoding?:boolean;excludedSource?:number;receipt?:ResearchReceipt};

/** One compatible target domain, one scale, one viewport for all comparison maps. */
export function ComparisonMaps({comparisons,positions,page,scale,mode,settings,onView,onMode,encoding,excludedSource,receipt}:Props) {
  const [expanded,setExpanded]=useState(false);
  const [start,setStart]=useState(0);
  const [exportError,setExportError]=useState<string>();
  const [exporting,setExporting]=useState(false);
  const identity=comparisons.map(c=>c.label).join('|');
  useEffect(()=>setStart(0),[identity]);
  const drag=useRef<{x:number;y:number;cx:number;cy:number;scale:number}|undefined>(undefined);
  const count=expanded?(settings.comparison_columns??2):8;
  const zoom=expanded?(settings.comparison_zoom??1):1;
  const width=page.width/zoom,height=page.height/zoom;
  const cx=Math.max(width/2,Math.min(page.width-width/2,settings.comparison_center_x??page.width/2));
  const cy=Math.max(height/2,Math.min(page.height-height/2,settings.comparison_center_y??page.height/2));
  const viewBox=expanded?`${cx-width/2} ${cy-height/2} ${width} ${height}`:`0 0 ${page.width} ${page.height}`;
  const pan=(dx:number,dy:number)=>onView({...settings,comparison_center_x:Math.max(width/2,Math.min(page.width-width/2,cx+dx*width*.1)),comparison_center_y:Math.max(height/2,Math.min(page.height-height/2,cy+dy*height*.1))});
  const maps=<>
    <p className="docattentionlens-comparison-scale" data-testid="comparison-scale">{page.imageLabel??'Original document page'} · Shared page scale 0 → {attentionWeightLabel(scale)} · {mode==='heatmap'?'dark → bright':mode==='blend'?'colour map blended over the page':mode==='original'?'attention hidden':'amber intensity'}{settings.attention_range_floor==='minimum'&&mode!=='original'?' · each map stretched from its minimum':''} · {settings.attention_scale_mode==='auto'?`image maximum across all comparison rows${excludedSource!==undefined?`, excluding source patch ${excludedSource}`:''}`:settings.attention_scale_mode==='fixed'?'fixed bound; saturated values clipped visually':'maximum over all comparison targets'}. {settings.attention_smooth?'Smoothing interpolates the display only.':'Captured patch boundaries retained.'}</p>
    {(mode==='heatmap'||mode==='blend')&&<span className="docattentionlens-comparison-colour-key" style={{background:mapGradient(settings.attention_colormap??'inferno')}} aria-hidden="true"/>}
    <div className={`docattentionlens-comparison-grid${expanded?' is-expanded':''}${expanded&&count===4?' is-four':''}`} data-testid={expanded?'expanded-comparisons':'compact-comparisons'} style={expanded?{gridTemplateColumns:`repeat(${count===4?2:count},minmax(0,1fr))`}:undefined}>
      {comparisons.slice(start,start+count).map(c=><figure key={c.label}><figcaption>{c.label} · image mass {c.row.image_mass.toFixed(6)} ({c.row.total_mass>0?(100*c.row.image_mass/c.row.total_mass).toFixed(1)+'%':'unavailable'}) · total {c.row.total_mass.toFixed(6)}</figcaption>
        <svg role="img" aria-label={`${c.label} attention on page ${page.page_number}`} viewBox={viewBox}
          onPointerDown={e=>{if(!expanded||e.button!==0||zoom===1)return;const box=e.currentTarget.getBoundingClientRect();drag.current={x:e.clientX,y:e.clientY,cx,cy,scale:Math.min(box.width/width,box.height/height)};e.currentTarget.setPointerCapture(e.pointerId);}}
          onPointerMove={e=>{const d=drag.current;if(!d)return;onView({...settings,comparison_center_x:Math.max(width/2,Math.min(page.width-width/2,d.cx-(e.clientX-d.x)/d.scale)),comparison_center_y:Math.max(height/2,Math.min(page.height-height/2,d.cy-(e.clientY-d.y)/d.scale))});}}
          onPointerUp={()=>{drag.current=undefined;}} onPointerCancel={()=>{drag.current=undefined;}}>
          <image href={page.image_url} width={page.width} height={page.height}/>
          <AttentionLayer positions={positions} row={c.row} maximum={scale} mode={mode} opacity={settings.attention_opacity??.65} smooth={settings.attention_smooth??false} showPatches={settings.attention_show_patches??false}
            colormap={settings.attention_colormap??'inferno'} floor={settings.attention_range_floor} patchTestId="comparison-patch"/>
        </svg></figure>)}
    </div>
    {comparisons.length>count&&<div className="docattentionlens-actions"><button className="docattentionlens-button" disabled={start===0} onClick={()=>setStart(s=>Math.max(0,s-count))}>Previous comparisons</button><span aria-live="polite">{start+1}–{Math.min(start+count,comparisons.length)} / {comparisons.length}</span><button className="docattentionlens-button" disabled={start+count>=comparisons.length} onClick={()=>setStart(s=>s+count)}>Next comparisons</button></div>}
  </>;
  return <><button className="docattentionlens-button" onClick={()=>{setStart(0);setExpanded(true);}}>Expand comparisons</button>{!expanded&&maps}
    {expanded&&<Drawer wide title="Compare attention maps" onClose={()=>{setExpanded(false);setStart(0);}}>
      <AttentionToolbar mode={mode} settings={settings} enabled encoding={encoding} onView={onView} onMode={onMode??(next=>onView({...settings,attention_display_mode:next}))}/>
      <div className="docattentionlens-comparison-controls"><label>Maps per view<select aria-label="Maps per view" value={settings.comparison_columns??2} onChange={e=>{setStart(0);onView({...settings,comparison_columns:Number(e.target.value) as 2|4});}}><option value={2}>2</option><option value={4}>4</option></select></label>
        <label>Linked zoom<select aria-label="Linked comparison zoom" value={zoom} onChange={e=>onView({...settings,comparison_zoom:Number(e.target.value)})}>{[1,1.5,2,3,4].map(z=><option key={z} value={z}>{z}×</option>)}</select></label>
        <div role="group" aria-label="Pan comparison maps">{([{label:'left',dx:-1,dy:0,glyph:'←'},{label:'right',dx:1,dy:0,glyph:'→'},{label:'up',dx:0,dy:-1,glyph:'↑'},{label:'down',dx:0,dy:1,glyph:'↓'}]).map(d=><button key={d.label} className="docattentionlens-button" disabled={zoom===1} aria-label={`Pan comparisons ${d.label}`} onClick={()=>pan(d.dx,d.dy)}>{d.glyph}</button>)}</div>
        <button className="docattentionlens-button" onClick={()=>onView({...settings,comparison_zoom:1,comparison_center_x:null,comparison_center_y:null})}>Reset comparison view</button><span>Drag a zoomed map or use the pan arrows. Previous and Next select comparison rows.</span></div>
      {receipt&&<button className="docattentionlens-button" disabled={exporting} onClick={async()=>{setExporting(true);setExportError(undefined);try{
        const svgs=[...document.querySelectorAll<SVGSVGElement>('[data-testid="expanded-comparisons"] svg')];
        const exported={...receipt,presentation:{...receipt.presentation,figure_input:page.imageLabel??'Original document page',visible_comparisons:comparisons.slice(start,start+count).map(c=>c.label)}};
        downloadBlob(await exportPageFigure(exported,svgs,`Page ${page.page_number} · ${receipt.case.question} · compatible comparison rows`,2),`attention-${receipt.evidence.capture_id}-comparisons.svg`);
      }catch(reason){setExportError(reason instanceof Error?reason.message:'Comparison export failed.');}finally{setExporting(false);}}}>{exporting?'Preparing comparison figure…':'Export comparison figure (SVG)'}</button>}{exportError&&<p role="alert">{exportError}</p>}
      {maps}
    </Drawer>}
  </>;
}
