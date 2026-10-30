import { useCallback, useEffect, useState } from 'react';
import { api, unwrap, errorMessage, type Case, type Element, type Method } from '../../api/client';
import { useRemote } from '../../app/remote';
import { Busy } from '../../app/Busy';
import type { components } from '../../api/generated/schema';
import type { Generation } from './GenerationPanel';
import { MethodParameters } from './MethodParameters';
import { ResearchResults, researchNumber } from './ResearchResults';
import { downloadBlob } from './receiptExport';

type Analysis=components['schemas']['AnalysisResult'];
type Study=components['schemas']['DocumentStudy'];
type Job=components['schemas']['JobDetail'];
const EXPERIMENTS=['document.image_text_contrast','document.element_masking','document.element_steering','attribution.integrated_gradients'];
const AXES=['step','layer','head'] as const;

/** Explicit multi-input and new-run controls stay separate from ordinary saved-row selection. */
export function DocumentStudies({record,generation,elements,onElement}:{record:Case;generation?:Generation;elements:Element[];onElement:(id:string)=>void}) {
  const catalogue=useRemote(useCallback(async()=>unwrap(await api.GET('/api/v1/methods')),[]));
  const [primary,setPrimary]=useState(generation?.capture_id??'');
  const [moreCaptures,setMoreCaptures]=useState<Generation[]>([]);
  const captures=useRemote(useCallback(async()=>unwrap(await api.GET('/api/v1/generation/captures',{params:{query:{limit:25}}})),[]));
  const selectedCapture=useRemote(useCallback(async()=>primary?unwrap(await api.GET('/api/v1/generation/captures/{capture_id}',{params:{path:{capture_id:primary}}})):undefined,[primary]));
  const inputPages=selectedCapture.data?.context.selected_page_ids.join(',')??'';
  const inputElements=useRemote(useCallback(async()=>inputPages?(await Promise.all(inputPages.split(',').map(async page_id=>unwrap(await api.GET('/api/v1/pages/{page_id}/elements',{params:{path:{page_id}}}))))).flatMap(p=>p.items):[],[inputPages]));
  const [methodId,setMethodId]=useState('attention.element_type_profile');
  const method=catalogue.data?.items.find(m=>m.method_id===methodId);
  const scientificProperties=method?.parameters_schema.properties as Record<string,unknown>|undefined;
  const [parameters,setParameters]=useState<Record<string,unknown>>({step:0,layer:0,head:0});
  const [samples,setSamples]=useState<string[]>([]),[comparison,setComparison]=useState(''),[experimentId,setExperimentId]=useState('');
  const [result,setResult]=useState<Analysis>(),[study,setStudy]=useState<Study>(),[job,setJob]=useState<Job>(),[busy,setBusy]=useState(false),[error,setError]=useState<string>();
  const [refresh,setRefresh]=useState(0);
  const studies=useRemote(useCallback(async()=>unwrap(await api.GET('/api/v1/document-studies',{params:{query:{case_id:record.case_id}}})),[record.case_id,refresh]));
  const layouts=useRemote(useCallback(async()=>unwrap(await api.GET('/api/v1/experiments',{params:{query:{case_id:record.case_id,limit:25}}})),[record.case_id]));
  const saved=useRemote(useCallback(async()=>unwrap(await api.GET('/api/v1/methods/saved-analyses',{params:{query:{generation_capture_id:primary||undefined}}})),[primary,refresh]));
  const allCaptures=[...new Map([...(generation?[generation]:[]),...(captures.data?.items??[]),...moreCaptures].map(c=>[c.capture_id,c])).values()];
  const methods=(catalogue.data?.items??[]).filter(m=>m.status!=='planned'&&(m.perspective==='document_aware'||m.method_id==='attention.modality_budget_depth'||m.method_id==='attribution.integrated_gradients'));
  const isExperiment=EXPERIMENTS.includes(methodId);
  const needsSample=method?.requires.includes('corpus_sample_manifest');
  const needsComparison=method?.requires.includes('paired_capture');
  const running=!!job&&['queued','running','cancellation_requested'].includes(job.state);
  const clear=()=>{setResult(undefined);setStudy(undefined);setError(undefined);};
  const choose=(m:Method)=>{clear();setMethodId(m.method_id);const props=m.parameters_schema.properties as Record<string,{default?:unknown}>;
    // Only declared axes are sent: a method without layer or head (such as integrated gradients) rejects them.
    setParameters({...Object.fromEntries(Object.entries(props??{}).filter(([,s])=>s.default!==undefined).map(([key,s])=>[key,s.default])),...Object.fromEntries(AXES.filter(key=>props?.[key]).map(key=>[key,0]))});};
  async function calculate() {
    if(!method||!primary)return;setBusy(true);setError(undefined);
    try {setResult(unwrap(await api.POST('/api/v1/methods/analyses',{body:{generation_capture_id:primary,method_id:method.method_id,variant_id:method.variant_id,
      parameters,...(needsSample?{sample_capture_ids:[primary,...samples.filter(i=>i!==primary)]}:{}),...(needsComparison&&comparison?{comparison_capture_id:comparison}:{}),...(needsComparison&&experimentId?{experiment_result_id:experimentId}:{})}})));setRefresh(v=>v+1);}
    catch(e){setError(errorMessage(e));}finally{setBusy(false);}
  }
  async function prepare() {
    if(!method||!primary)return;setBusy(true);setError(undefined);setResult(undefined);
    try{setStudy(unwrap(await api.POST('/api/v1/document-studies',{body:{generation_capture_id:primary,method_id:method.method_id,variant_id:method.variant_id,parameters}})));setRefresh(v=>v+1);}
    catch(e){setError(errorMessage(e));}finally{setBusy(false);}
  }
  async function run() {
    if(!study)return;setBusy(true);setError(undefined);
    try{const accepted=unwrap(await api.POST('/api/v1/document-studies/{study_id}/runs',{params:{path:{study_id:study.study_id}},body:{idempotency_key:`document-study-${study.study_id}`}}));
      setJob(unwrap(await api.GET('/api/v1/jobs/{job_id}',{params:{path:{job_id:accepted.job_id}}})));}
    catch(e){setError(errorMessage(e));}finally{setBusy(false);}
  }
  async function inspect(completed:Study) {
    setStudy(completed);
    if(completed.kind!=='result'||!completed.outcomes?.length)return;
    const descriptor=catalogue.data?.items.find(m=>m.method_id===completed.method_id);
    if(!descriptor)return;
    setResult(unwrap(await api.POST('/api/v1/methods/analyses',{body:{generation_capture_id:completed.outcomes[0].capture_id,method_id:descriptor.method_id,
      variant_id:descriptor.variant_id,parameters:completed.parameters,experiment_result_id:completed.study_id}})));
  }
  useEffect(()=>{
    if(!running||!job)return;
    let alive=true,polling=false;
    // The completed study loads before the job state is set: a terminal job ends this effect,
    // and setting it first cancelled the load, so a finished study never appeared.
    const timer=window.setInterval(async()=>{
      if(polling)return;polling=true;
      try{const next=unwrap(await api.GET('/api/v1/jobs/{job_id}',{params:{path:{job_id:job.job_id}}}));if(!alive)return;setError(undefined);
        if(next.state==='completed'&&next.output_resource_id){window.clearInterval(timer);
          try{const completed=unwrap(await api.GET('/api/v1/document-studies/{study_id}',{params:{path:{study_id:next.output_resource_id}}}));if(alive)await inspect(completed);}
          catch(e){if(alive)setError(errorMessage(e));}
          setRefresh(v=>v+1);}
        else if(next.state==='failed'||next.state==='cancelled')setError(next.error_message??'Experiment cancelled.');
        if(alive)setJob(next);
      }catch(e){if(alive)setError(errorMessage(e));}finally{polling=false;}
    },2000);
    return()=>{alive=false;window.clearInterval(timer);};
  // The selected job is immutable; each run retains its own method and parameters.
  // eslint-disable-next-line react-hooks/exhaustive-deps
  },[job?.job_id,running]);
  async function loadMore() {
    const before=moreCaptures.at(-1)?.capture_id??captures.data?.next_cursor;
    if(!before)return;
    try{const page=unwrap(await api.GET('/api/v1/generation/captures',{params:{query:{limit:25,before}}}));setMoreCaptures(v=>[...v,...page.items]);}catch(e){setError(errorMessage(e));}
  }
  const resultMethod=catalogue.data?.items.find(m=>m.method_id===result?.method_id);
  const toggles=(parameters.element_ids??[]) as string[];
  return <section className="docattentionlens-document-studies" aria-label="Document studies">
    <p>Calculate document analyses from saved captures, choose a corpus sample, or prepare a controlled model experiment. New model runs start only with Run prepared experiment.</p>
    <label>Document study method<select aria-label="Document study method" value={methodId} disabled={running} onChange={e=>{const m=methods.find(m=>m.method_id===e.target.value);if(m)choose(m);}}>{methods.map(m=><option key={m.method_id} value={m.method_id}>{m.name}</option>)}</select></label>
    <label>Primary saved capture<select aria-label="Primary saved capture" value={primary} disabled={running} onChange={e=>{clear();setPrimary(e.target.value);}}><option value="">Choose saved generation</option>{allCaptures.map(c=><option key={c.capture_id} value={c.capture_id}>{c.context.query.slice(0,90)} · {String(c.provenance.model_id)} · {c.created_at}</option>)}</select></label>
    {captures.data?.next_cursor&&<button className="docattentionlens-button" onClick={()=>void loadMore()}>Load more saved captures</button>}
    {method&&<p>{method.summary}</p>}
    {method&&selectedCapture.data&&<>
      <div className="docattentionlens-study-axes">{AXES.filter(key=>scientificProperties?.[key]!==undefined&&!(key==='head'&&scientificProperties?.heads)&&!(key==='layer'&&scientificProperties?.layers)).map(key=><label key={key}>{key==='step'?'Prediction step':key==='layer'?'Decoder layer':'Query head'}<input type="number" aria-label={`Study ${key}`} min={0} max={key==='step'?Math.max(...selectedCapture.data!.steps.map(s=>s.step)):key==='layer'?selectedCapture.data!.layer_count-1:selectedCapture.data!.query_head_count-1} value={Number(parameters[key]??0)} onChange={e=>{if(Number.isInteger(e.target.valueAsNumber))setParameters(v=>({...v,[key]:e.target.valueAsNumber}));}}/></label>)}</div>
      <MethodParameters method={method} layerCount={selectedCapture.data.layer_count} headCount={selectedCapture.data.query_head_count} parameters={parameters} onChange={setParameters}/>
    </>}
    {needsSample&&<fieldset><legend>Frozen saved-capture sample · primary capture included</legend>{allCaptures.filter(c=>c.capture_id!==primary).map(c=><label key={c.capture_id} className="docattentionlens-checkbox"><input type="checkbox" checked={samples.includes(c.capture_id)} disabled={samples.length>=63&&!samples.includes(c.capture_id)} onChange={e=>setSamples(v=>e.target.checked?[...v,c.capture_id]:v.filter(i=>i!==c.capture_id))}/>{c.context.query.slice(0,100)} · {String(c.provenance.model_id)}</label>)}</fieldset>}
    {needsComparison&&<><label>Saved layout result<select aria-label="Saved layout result" value={experimentId} onChange={e=>{clear();setExperimentId(e.target.value);}}><option value="">Same-case comparison without layout transformation</option>{(layouts.data?.items??[]).filter(e=>e.record_kind==='experiment_result').map(e=><option key={e.experiment_id} value={e.experiment_id}>{e.name}</option>)}</select></label>
      {experimentId?<div className="docattentionlens-study-axes">{(['baseline','comparison'] as const).map(kind=><label key={kind}>{kind==='baseline'?'Baseline arm':'Comparison arm'}<select aria-label={`${kind} arm`} value={kind==='baseline'?primary:comparison} onChange={e=>{clear();kind==='baseline'?setPrimary(e.target.value):setComparison(e.target.value);}}><option value="">Choose arm</option>{layouts.data?.items.find(e=>e.experiment_id===experimentId)?.outcomes?.filter(o=>o.generation).map(o=>{const gen=o.generation as {capture_id:string};return <option key={String(o.arm)} value={gen.capture_id}>{String(o.arm)}</option>;})}</select></label>)}</div>
        :<label>Comparison saved capture<select aria-label="Comparison saved capture" value={comparison} onChange={e=>{clear();setComparison(e.target.value);}}><option value="">Choose comparison</option>{allCaptures.filter(c=>c.capture_id!==primary).map(c=><option key={c.capture_id} value={c.capture_id}>{c.context.query.slice(0,90)} · {c.created_at}</option>)}</select></label>}
    </>}
    {(methodId==='document.element_masking'||methodId==='document.element_steering')&&<fieldset><legend>Elements to {methodId==='document.element_masking'?'mask':'steer'} · current case</legend>{(inputElements.data??elements).filter(e=>selectedCapture.data?.context.selected_page_ids.includes(e.page_id)).map(e=><label className="docattentionlens-checkbox" key={e.element_id}><input type="checkbox" checked={toggles.includes(e.element_id)} disabled={!!e.geometry_flags.length} onChange={event=>setParameters(p=>({...p,element_ids:event.target.checked?[...toggles,e.element_id]:toggles.filter(i=>i!==e.element_id)}))}/>{e.element_type} · element {e.native_id} · order {e.reading_order??'unrecorded'}</label>)}</fieldset>}
    <button className="docattentionlens-button docattentionlens-primary" disabled={busy||running||!method||!primary} onClick={()=>void(isExperiment?prepare():calculate())}>{busy?<><Busy/>Working…</>:isExperiment?'Prepare experiment and preview arms':'Calculate from saved captures'}</button>
    {error&&<p role="alert">{error}</p>}{catalogue.error&&<p role="alert">{catalogue.error}</p>}
    {/* A failed capture list otherwise looks like a sample holding only the current capture. */}
    {captures.error&&<p role="alert">Saved captures could not be listed: {captures.error}</p>}{selectedCapture.error&&<p role="alert">{selectedCapture.error}</p>}
    {study&&<section><h3>{study.kind==='prepared'?'Prepared experiment':'Completed experiment'} · {study.method_id}</h3><p>Fixed evidence · {study.arms.length} arms · model revision {study.model_revision}</p>{(study.provenance.limitations as string[]|undefined)?.map(v=><p key={v}>{v}</p>)}
      <div className="docattentionlens-study-arm-previews">{study.arms.map(arm=><details key={arm.name}><summary>{arm.name} · {arm.representation}{arm.strength!=null?` · strength ${arm.strength}`:''}</summary>{study.provenance.page_ids instanceof Array&&study.provenance.page_ids.map(pid=>arm.representation==='image'?<img key={String(pid)} alt={`${arm.name} page preview`} src={`/api/v1/document-studies/${study.study_id}/arms/${encodeURIComponent(arm.name)}/pages/${String(pid)}`}/>:<a key={String(pid)} href={`/api/v1/document-studies/${study.study_id}/arms/${encodeURIComponent(arm.name)}/pages/${String(pid)}`} target="_blank" rel="noreferrer">Inspect exact {arm.representation} input</a>)}</details>)}</div>
      {study.kind==='prepared'&&<button className="docattentionlens-button docattentionlens-primary" disabled={busy||running} onClick={()=>void run()}>{running&&<Busy/>}Run prepared experiment</button>}
      {!!study.outcomes?.length&&<table><thead><tr><th>Arm</th><th>Observed answer</th><th>Input positions</th><th>Generated tokens</th></tr></thead><tbody>{study.outcomes.map(o=><tr key={o.name}><td>{o.name}</td><td>{o.answer}</td><td>{o.input_positions}</td><td>{o.output_token_ids.length}</td></tr>)}</tbody></table>}
      {study.method_id==='attribution.integrated_gradients'&&!!study.provenance.target&&<p>Target: log-probability of prediction step {String((study.provenance.target as {step:number}).step)} · {String((study.provenance.path as {baseline:string;steps:number}).baseline)} baseline · {String((study.provenance.path as {baseline:string;steps:number}).steps)} midpoint Riemann steps. Gradients use the unchanged source inputs; no new capture is created.</p>}
    </section>}
    {job&&<p role="status">{running&&<Busy/>}Experiment {job.state}{job.error_message?` · ${job.error_message}`:''}{running&&<button className="docattentionlens-button" onClick={()=>void api.POST('/api/v1/jobs/{job_id}/cancel',{params:{path:{job_id:job.job_id}}})}>Cancel experiment</button>}</p>}
    {result&&<section data-testid="document-study-result"><h3>{resultMethod?.name??result.method_id}</h3><p>{result.normalization}</p>{!!result.statistics?.length&&<dl>{result.statistics.map(s=><div key={s.name}><dt>{s.name}</dt><dd>{researchNumber(s.value)} {s.units}</dd></div>)}</dl>}<ResearchResults result={result} onElement={onElement} primaryTable/>
      <button className="docattentionlens-button" onClick={()=>downloadBlob(new Blob([JSON.stringify(result,null,2)],{type:'application/json'}),'document-analysis.json')}>Export result and input manifest</button><details><summary>Saved result and input identities</summary><pre>{JSON.stringify({analysis_id:result.analysis_id,input_manifest:result.input_manifest,parameters:result.parameters},null,2)}</pre></details></section>}
    {!!saved.data?.items.length&&<details><summary>Saved analyses · {saved.data.items.length}</summary>{saved.data.items.map(a=><button key={a.analysis_id} className="docattentionlens-button" onClick={()=>{setResult(a);setMethodId(a.method_id);setParameters(a.parameters);}}>{catalogue.data?.items.find(m=>m.method_id===a.method_id)?.name??a.method_id}</button>)}</details>}
    {!!studies.data?.items.length&&<details><summary>Saved document experiments · {studies.data.items.length}</summary>{studies.data.items.map(s=><button key={s.study_id} className="docattentionlens-button" onClick={()=>void inspect(s).catch(e=>setError(errorMessage(e)))}>{s.method_id} · {s.kind} · {s.created_at}</button>)}</details>}
  </section>;
}
