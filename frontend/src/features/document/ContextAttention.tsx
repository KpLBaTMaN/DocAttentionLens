import type { Generation, Attention } from './GenerationPanel';
import { attentionWeightLabel } from './attentionDisplay';

/** Details › Run in a generation investigation: supplied pages with their share of the selected row, and omitted pages. */
export function ContextAttention({capture,row,pageId,pageLabel,onPage}:{capture?:Generation;row?:Attention;pageId:string;pageLabel:(id:string)=>string;onPage:(id:string)=>void}) {
  if(!capture)return <p>Open a generator capture to inspect attention to its supplied context.</p>;
  const title=`${capture.context.input_source==='supplied_pages'?'Supplied':'Retrieved'} context`;
  const mass=(id:string)=>row?capture.positions.filter(p=>p.page_id===id&&p.index<row.target_count).reduce((total,p)=>total+row.values[p.index],0):undefined;
  return <section className="docattentionlens-context-budget" aria-label="Generator attention to context"><h3>{title}</h3>
    <p>{row?`Generator attention from the selected row. ${row.quantity==='post_softmax_attention_weight'?'Original-domain attention mass':'Head envelope values; totals need not equal one'}.`:'Page allocations appear with methods that produce attention weight rows.'}</p>
    <ol className="docattentionlens-page-list">
      {capture.context.pages.map((p,i)=>{const value=mass(p.page_id);return <li key={p.page_id}><button data-page-id={p.page_id} aria-pressed={pageId===p.page_id} onClick={()=>onPage(p.page_id)}><strong>{i+1} · Page {p.page_number}</strong><span>{value===undefined?`Supplied · ${p.resized_width} × ${p.resized_height}`:<>Attention <b data-testid="context-page-mass">{attentionWeightLabel(value)}</b></>}</span></button></li>;})}
      {capture.context.omitted_page_ids.map(id=><li key={id}><button data-page-id={id} aria-pressed={pageId===id} onClick={()=>onPage(id)}><strong>{pageLabel(id)}</strong><span>Retrieved · omitted from context</span></button></li>)}
    </ol>
    {row&&<p className="docattentionlens-context-rest">Question, prompt and preceding output <strong>{attentionWeightLabel(row.total_mass-row.image_mass)}</strong></p>}
    <small>{capture.context.omitted_page_ids.length} retrieved pages omitted; omitted pages have no captured image attention.</small>
  </section>;
}
