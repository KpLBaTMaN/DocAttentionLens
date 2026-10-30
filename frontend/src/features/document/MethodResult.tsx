import type { DisplaySettings, Page } from '../../api/client';
import { Busy } from '../../app/Busy';
import type { components } from '../../api/generated/schema';
import type { Generation } from './GenerationPanel';
import type { View, UpdateView } from './useAcquisition';
import type { RetrieverAttention } from './useRetrieverAttention';
import { type AttentionMode } from './attentionDisplay';
import { ComparisonMaps } from './ComparisonMaps';
import type { ResearchReceipt } from './receiptExport';
import { ResearchResults, ResearchTable, researchNumber } from './ResearchResults';

export type Analysis = components['schemas']['AnalysisResult'];
type Calculation = components['schemas']['Calculation'];

/** Statistics and summary tables, formatted identically for every attention source. */
const LONG_TABLE = 20;

function ResultValues({result,onElement}:{result:Calculation;onElement?:(id:string)=>void}) {
  return <>
    {!!result.statistics?.length&&<dl className="docattentionlens-statistics">{result.statistics.map(s=><div key={s.name}><dt>{s.name}</dt><dd>{researchNumber(s.value)} <small>{s.units}</small></dd></div>)}</dl>}
    {/* Long tables fold away so the grids and charts that follow stay near the heading. */}
    {result.table&&((result.graph&&/edge/i.test(result.table.title))||result.table.labels.length>LONG_TABLE?<details><summary>{result.table.title||'Result table'} · {result.table.labels.length} rows</summary><ResearchTable table={result.table} onElement={onElement}/></details>:<ResearchTable table={result.table} onElement={onElement}/>) }
  </>;
}

/** Comparison maps use the page's display mode and colour mapping, so a head grid reads like the page above it. */
export function MethodResult({result,generation,pageId,scale,mode,settings,onView,onMode,methodName,receipt,onElement}:{result:Analysis;generation:Generation;pageId:string;scale:number;mode:AttentionMode;settings:DisplaySettings;onView:(settings:DisplaySettings)=>void;onMode?:(mode:AttentionMode)=>void;methodName?:string;receipt?:ResearchReceipt;onElement?:(id:string)=>void}) {
  const comparisons=result.comparisons??[];
  const page=generation.context.pages.find(p=>p.page_id===pageId);
  const positions=generation.positions.filter(p=>p.page_id===pageId&&p.bbox);
  return <section className="docattentionlens-method-result" aria-label="Method result" data-testid="method-result" data-analysis-id={result.analysis_id}>
    <h2>{methodName??result.quantity.replaceAll('_',' ')} <small>· {result.units}</small></h2>
    <p>{result.normalization}</p>
    {generation.full_attention&&generation.positions[result.source_index]?.role==='image'&&<p className="docattentionlens-muted" data-testid="causal-scope-note">Causal scope: image patches precede the question, so a patch row attends only to earlier template tokens and patches, never to the question or the answer.{result.method_id==='attention.vision_to_text'?' Vision → text therefore measures attention to the preceding template text.':''}</p>}
    <ResultValues result={result} onElement={onElement}/>
    <ResearchResults result={result} onElement={onElement}/>
    {!!comparisons.length&&(page?<ComparisonMaps comparisons={comparisons} positions={positions} page={{width:page.source_width,height:page.source_height,image_url:page.preview_url,page_number:page.page_number,imageLabel:'Consumed model input · rounded BF16 display in source coordinates'}} scale={scale} mode={mode} settings={settings} onView={onView} onMode={onMode} receipt={receipt}/>:<p>Page was not supplied to this generator.</p>)}
    <details><summary>Result provenance and support</summary><p>{result.valid_target_indices.length} valid targets · {result.source_indices?.length?`captured source population: ${result.source_indices.length} rows; selected input anchor ${result.source_index}`:`source ${result.source_index} predicts ${result.predicted_index}`}. {result.coverage.selection as string}</p><pre>{JSON.stringify({analysis_id:result.analysis_id,method:result.method_id,variant:result.variant_id,implementation:result.implementation_version,parameters:result.parameters,capture_sha256:result.capture_sha256,array_sha256:result.array_sha256},null,2)}</pre></details>
  </section>;
}

/** Retriever encoding result for Details › Result. Comparison rows drive the page overlay because retrieval pages carry no preview image. */
export function RetrieverResult({attention,state,update,methodName,page,settings,onView,onMode,scale,mode,receipt}:{attention:RetrieverAttention;state:View;update:UpdateView;methodName?:string;page?:Page;settings:DisplaySettings;onView:(settings:DisplaySettings)=>void;onMode?:(mode:AttentionMode)=>void;scale:number;mode:AttentionMode;receipt?:ResearchReceipt}) {
  const result=attention.result;
  if(!result)return null;
  const comparisons=result.comparisons??[];
  return <section className="docattentionlens-method-result" aria-label="Method result" data-testid="method-result">
    <h2>{methodName??attention.method?.name??'Retriever attention'} <small>· {attention.encoding?.pass_kind==='query'?'query encoding':attention.encoding?.pass_kind==='joint'?'diagnostic joint page + question':'document encoding'}</small></h2>
    <p>{result.normalization}. Attention describes allocation; it does not establish causation.</p>
    <ResultValues result={result}/>
    <ResearchResults result={result}/>
    {!!comparisons.length&&<label>Comparison row<select aria-label="Comparison row" value={Math.min(state.retrieval_comparison_index??0,comparisons.length-1)} onChange={e=>update(s=>({...s,retrieval_comparison_index:Number(e.target.value)}))}>{comparisons.map((c,i)=><option key={c.label} value={i}>{c.label}</option>)}</select></label>}
    {!!comparisons.length&&page&&attention.encoding?.pass_kind!=='query'&&attention.page?.page_id===page.page_id&&<ComparisonMaps comparisons={comparisons} positions={attention.positions??[]} page={page} scale={scale} mode={mode} settings={settings} onView={onView} onMode={onMode} encoding receipt={receipt} excludedSource={attention.positions?.find(p=>p.index===attention.row?.source_index&&p.bbox)?.index}/>}
  </section>;
}

/** Complete retriever encoding row, the dock view matching generation's causal row. Element allocations use ElementSummary. */
export function RetrieverRowDetails({attention}:{attention:RetrieverAttention}) {
  const row=attention.row;
  if(!row)return <p>{attention.unavailable??(attention.result?'The selected method reports a summary rather than one encoding row. See Method result.':<>{attention.loading&&<Busy/>}Calculating selected method from saved evidence…</>)}</p>;
  return <div className="docattentionlens-method-table"><table><caption>Complete encoding row · including non-page targets</caption><thead><tr><th>Target</th><th>Weight</th></tr></thead><tbody>{attention.positions?.map(p=><tr key={p.index}><td>{p.index} · {p.piece} · {p.role}</td><td>{row.values[p.index]?.toPrecision(6)}</td></tr>)}</tbody></table></div>;
}
