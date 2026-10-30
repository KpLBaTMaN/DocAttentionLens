import { useState, type KeyboardEvent } from 'react';
import type { components } from '../../api/generated/schema';
import { AttentionGraphView } from './AttentionGraphView';

type Calculation=components['schemas']['Calculation'];
type Plot=components['schemas']['ResearchPlot'];
type Table=components['schemas']['ResultTable'];
export const researchNumber=(v:number|null|undefined)=>v==null?'Unavailable':v!==0&&Math.abs(v)<1e-6?v.toExponential(2):v.toLocaleString(undefined,{maximumFractionDigits:6});
const number=researchNumber;

export function ResearchTable({table,onElement}:{table:Table;onElement?:(id:string)=>void}) {
  if(!table.labels.length)return <p className="docattentionlens-research-notice">{table.title||'Analysis values'}: no rows for this selection.</p>;
  return <div className="docattentionlens-table-scroll"><table className="docattentionlens-research-table"><caption>{table.title || 'Analysis values'} · {table.units}</caption><thead><tr><th>Group</th>{table.columns.map(c=><th key={c}>{c}</th>)}</tr></thead>
    <tbody>{table.labels.slice(0,200).map((label,i)=><tr key={`${label}/${i}`}><th>{table.element_ids?.[i]&&onElement?<button className="docattentionlens-link-button" onClick={()=>onElement(table.element_ids![i]!)}>{label}</button>:label}</th>{table.values[i].map((v,j)=><td key={j}>{number(v)}</td>)}</tr>)}</tbody></table>{table.labels.length>200&&<p>Showing 200 of {table.labels.length} rows. The saved result and JSON export retain every row.</p>}</div>;
}

/** Categorical colours for plot groups (clusters, series); the first is the attention accent. */
const GROUP_COLOURS=['var(--docattentionlens-accent)','#247ed5','#9a6fd1','#2a9d8f','#c4572e','#6c8a1f','#b8860b','#d1497a'];

function ResearchChart({plot,onElement}:{plot:Plot;onElement?:(id:string)=>void}) {
  const [selected,setSelected]=useState<number>();
  if(!plot.points.length)return null;
  const xs=plot.points.map(p=>p.x),ys=plot.points.map(p=>p.y);
  const minX=plot.kind==='grid'?0:Math.min(0,...xs),maxX=plot.kind==='grid'?1:Math.max(...xs);
  const minY=plot.kind==='grid'?0:Math.min(0,...ys),maxY=plot.kind==='grid'?1:Math.max(...ys);
  const x=(v:number)=>70+(v-minX)/(maxX-minX||1)*580;
  const y=(v:number)=>plot.kind==='grid'?35+(v-minY)/(maxY-minY||1)*255:290-(v-minY)/(maxY-minY||1)*255;
  const groups=[...new Set(plot.points.map(p=>p.group??''))];
  const maxValue=Math.max(...plot.points.map(p=>Math.abs(p.value??0)),1e-12);
  // Signed grids (output attribution) use a diverging scale; dense grids fill their cells.
  const signed=plot.kind==='grid'&&plot.points.some(p=>(p.value??0)<0);
  const columns=new Set(xs).size,rows=new Set(ys).size;
  const cellWidth=plot.kind==='grid'&&columns>12?580/columns:18,cellHeight=plot.kind==='grid'&&rows>12?255/rows:18;
  const colour=(group?:string|null)=>GROUP_COLOURS[Math.max(0,groups.indexOf(group??''))%GROUP_COLOURS.length];
  const select=(i:number)=>{setSelected(i);const point=plot.points[i];if(point.element_id)onElement?.(point.element_id);};
  return <figure className="docattentionlens-research-chart"><figcaption><strong>{plot.title}</strong></figcaption>
    <svg viewBox="0 0 720 345" role="img" aria-label={plot.title}><title>{plot.title}</title>
      <path d="M70 35V290H650" fill="none" stroke="var(--docattentionlens-line)"/>
      <text x="360" y="335" textAnchor="middle">{plot.x_label}</text><text x="15" y="165" textAnchor="middle" transform="rotate(-90 15 165)">{plot.y_label}</text>
      {plot.kind!=='grid'&&<><text x="70" y="310">{number(minX)}</text><text x="650" y="310" textAnchor="end">{number(maxX)}</text><text x="65" y="290" textAnchor="end">{number(minY)}</text><text x="65" y="40" textAnchor="end">{number(maxY)}</text></>}
      {plot.kind==='line'&&groups.map((group,i)=><polyline key={group} points={plot.points.filter(p=>p.group===group).sort((a,b)=>a.x-b.x).map(p=>`${x(p.x)},${y(p.y)}`).join(' ')} fill="none" stroke={colour(group)} strokeDasharray={i?`${2+i*2} ${2+i}`:undefined} opacity={Math.max(.3,1-i*.1)} strokeWidth="2"/>)}
      {plot.points.map((p,i)=>{
        const common={onClick:()=>select(i),onKeyDown:(e:KeyboardEvent<SVGElement>)=>{if(e.key==='Enter'||e.key===' '){e.preventDefault();select(i);}},tabIndex:0,role:'button','aria-label':`${p.label}: ${number(p.value??p.y)}`};
        return plot.kind==='grid'?<rect key={i} {...common} x={x(p.x)-cellWidth/2} y={y(p.y)-cellHeight/2} width={cellWidth} height={cellHeight} fill={(p.value??0)<0?'#247ed5':'var(--docattentionlens-accent)'} fillOpacity={.08+.92*Math.abs(p.value??0)/maxValue}><title>{p.label}: {number(p.value)}</title></rect>
          :plot.kind==='bar'||plot.kind==='ribbon'?<rect key={i} {...common} x={x(p.x)-8} y={Math.min(y(0),y(p.y))} width="16" height={Math.max(1,Math.abs(y(0)-y(p.y)))} fill={colour(p.group)}><title>{p.label}: {number(p.y)}</title></rect>
          :<circle key={i} {...common} cx={x(p.x)} cy={y(p.y)} r={selected===i?6:4} fill={colour(p.group)}><title>{p.label}: {number(p.x)}, {number(p.y)}</title></circle>;
      })}
    </svg>
    {plot.kind==='grid'&&<p>{signed?'Red: positive · blue: negative · ':''}Opacity ∝ |value| / {number(maxValue)}. Select a cell to read its label and value.</p>}
    {groups.some(Boolean)&&<p className="docattentionlens-chart-legend">{groups.map((g,i)=><span key={g}><span className="docattentionlens-chart-swatch" style={{background:colour(g)}} aria-hidden="true"/>{g}{plot.kind==='line'&&i?` (dash ${i})`:''}</span>)}</p>}
    {selected!==undefined&&plot.points[selected]&&<p aria-live="polite">{plot.points[selected].label} · x {number(plot.points[selected].x)} · y {number(plot.points[selected].y)}{plot.points[selected].value!=null?` · value ${number(plot.points[selected].value)}`:''}</p>}
  </figure>;
}

export function ResearchResults({result,onElement,primaryTable=false}:{result:Calculation;onElement?:(id:string)=>void;primaryTable?:boolean}) {
  const edges=result.edges??[];
  return <div className="docattentionlens-research-results">
    {(result.notices??[]).map((notice,i)=><p key={i} className="docattentionlens-research-notice">{notice}</p>)}
    {result.graph&&<AttentionGraphView graph={result.graph}/>}
    {(result.plots??[]).map((plot,i)=><ResearchChart key={`${plot.title}/${i}`} plot={plot} onElement={onElement}/>)}
    {primaryTable&&result.table&&<ResearchTable table={result.table} onElement={onElement}/>}
    {(result.tables??[]).map((table,i)=><ResearchTable key={`${table.title}/${i}`} table={table} onElement={onElement}/>)}
    {!!edges.length&&<details><summary>Linked signals · {edges.length} connections</summary>{(['retrieval','attention','section'] as const).map(signal=>{
      const selected=edges.filter(e=>e.signal===signal);if(!selected.length)return null;
      return <div key={signal}><h3>{signal==='retrieval'?'Query → element · signed retrieval score':signal==='attention'?'Element → answer prediction · attention mass':'Inferred section relationships'}</h3><table><thead><tr><th>Source</th><th>Target</th><th>Value</th><th>Units</th></tr></thead><tbody>{selected.slice(0,120).map((e,i)=><tr key={i}><td>{e.source}</td><td>{e.target}</td><td>{number(e.value)}</td><td>{e.units}</td></tr>)}</tbody></table>{selected.length>120&&<p>Showing 120 connections; the saved JSON contains all {selected.length}.</p>}</div>;
    })}</details>}
  </div>;
}
