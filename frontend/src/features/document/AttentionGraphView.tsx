import { useId, useState } from 'react';
import type { components } from '../../api/generated/schema';
import { researchNumber } from './ResearchResults';

type Graph=components['schemas']['AttentionGraph'];

/** Display a declared edge subset; the method result preserves all retained edges. */
export function AttentionGraphView({graph}:{graph:Graph}) {
  const marker=useId().replaceAll(':','');
  const [selected,setSelected]=useState<number>();
  const edges=graph.edges.slice(0,160);
  const ids=new Set(edges.flatMap(e=>[e.source,e.target]));
  const nodes=graph.nodes.filter(n=>ids.has(n.node_id));
  const byId=new Map(nodes.map(n=>[n.node_id,n]));
  if(!edges.length)return <p>No edges meet the declared graph threshold.</p>;
  const low=Math.min(...nodes.map(n=>n.level)),high=Math.max(...nodes.map(n=>n.level));
  const positionMax=Math.max(...nodes.map(n=>n.position),1),maximum=Math.max(...edges.map(e=>e.value),1e-12);
  const x=(level:number)=>70+(level-low)/(high-low||1)*570;
  const y=(position:number)=>35+position/positionMax*250;
  const current=selected!==undefined?edges[selected]:undefined;
  return <figure className="docattentionlens-research-chart" aria-label="Attention graph">
    <figcaption><strong>{graph.title}</strong> · {graph.units}</figcaption>
    <p>{edges.length} displayed edges · {graph.edges.length} retained · {graph.total_edges} calculation/threshold edges. Lines point from source level to target level. Image nodes are circles; non-image nodes are squares.</p>
    <svg viewBox="0 0 720 345" role="img" aria-label={graph.title}>
      <title>{graph.title}</title><desc>Attention allocation or residual capacity edges over model positions and attention depth. Line width is relative to the largest displayed edge.</desc>
      <defs><marker id={marker} viewBox="0 0 10 10" refX="9" refY="5" markerWidth="5" markerHeight="5" orient="auto-start-reverse"><path d="M0 0L10 5L0 10z" fill="var(--docattentionlens-accent)"/></marker></defs>
      <path d="M70 35V290H650" fill="none" stroke="var(--docattentionlens-line)"/>
      <text x="360" y="335" textAnchor="middle">Attention level</text><text x="15" y="165" textAnchor="middle" transform="rotate(-90 15 165)">Encoded position</text>
      <text x="70" y="310">{low}</text><text x="650" y="310" textAnchor="end">{high}</text><text x="65" y="40" textAnchor="end">0</text><text x="65" y="290" textAnchor="end">{positionMax}</text>
      {edges.map((e,i)=>{const a=byId.get(e.source)!,b=byId.get(e.target)!;return <line key={i} x1={x(a.level)} y1={y(a.position)} x2={x(b.level)} y2={y(b.position)} stroke="var(--docattentionlens-accent)" opacity={selected===i?1:.28} strokeWidth={.5+2.5*Math.sqrt(e.value/maximum)} markerEnd={`url(#${marker})`}
        tabIndex={0} role="button" aria-label={`${e.source} to ${e.target}: ${researchNumber(e.value)}`} onClick={()=>setSelected(i)} onKeyDown={event=>{if(event.key==='Enter'||event.key===' '){event.preventDefault();setSelected(i);}}}><title>{e.source} → {e.target} · layer {e.layer}{e.head!=null?` · head ${e.head}`:''} · {researchNumber(e.value)}</title></line>;})}
      {nodes.map(n=>n.role==='image'?<circle key={n.node_id} cx={x(n.level)} cy={y(n.position)} r="3" fill="var(--docattentionlens-ink)"><title>{n.node_id} · {n.label}</title></circle>:<rect key={n.node_id} x={x(n.level)-2.5} y={y(n.position)-2.5} width="5" height="5" fill="var(--docattentionlens-ink)"><title>{n.node_id} · {n.label}</title></rect>)}
    </svg>
    {current&&<p aria-live="polite">{current.source} → {current.target} · layer {current.layer}{current.head!=null?` · head ${current.head}`:''} · {researchNumber(current.value)} {graph.units}</p>}
  </figure>;
}
