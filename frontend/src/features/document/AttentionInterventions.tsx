import { useCallback, useEffect, useState } from 'react';
import { api, unwrap, errorMessage, type Case } from '../../api/client';
import type { components } from '../../api/generated/schema';
import { useRemote } from '../../app/remote';
import { Busy } from '../../app/Busy';
import type { Generation } from './GenerationPanel';
import type { Capture } from './RetrievalPanel';
import { MethodParameters } from './MethodParameters';
import { ResearchResults } from './ResearchResults';
import { downloadBlob } from './receiptExport';

type Experiment=components['schemas']['AttentionExperiment'];
type Job=components['schemas']['JobDetail'];

export function AttentionInterventions({record,generation,retrieval,role,pageId,layer,head,sourceIndex,targetIndex}:{
  record:Case;generation?:Generation;retrieval?:Capture;role:'generation'|'retrieval';pageId:string;layer:number;head:number;sourceIndex?:number;targetIndex?:number|null;
}) {
  const catalogue=useRemote(useCallback(async()=>unwrap(await api.GET('/api/v1/methods')),[]));
  const methods=(catalogue.data?.items??[]).filter(m=>m.family==='attention_interventions');
  const [methodId,setMethodId]=useState('attention.head_ablation');
  const method=methods.find(m=>m.method_id===methodId);
  const [selectedRole,setSelectedRole]=useState(role);
  const capture=selectedRole==='generation'?generation:retrieval;
  const [parameters,setParameters]=useState<Record<string,unknown>>({layer,head});
  const [selectedPage,setSelectedPage]=useState(pageId);
  const [experiment,setExperiment]=useState<Experiment>();
  const [job,setJob]=useState<Job>();
  const [busy,setBusy]=useState(false),[error,setError]=useState<string>(),[revision,setRevision]=useState(0);
  const running=!!job&&['queued','running','cancellation_requested'].includes(job.state);
  const records=useRemote(useCallback(async()=>unwrap(await api.GET('/api/v1/attention-interventions',{params:{query:{case_id:record.case_id}}})),[record.case_id,revision]));
  const pages=selectedRole==='generation'?generation?.context.selected_page_ids??[]:retrieval?.pages.map(p=>p.page_id)??[];
  const layerCount=selectedRole==='generation'?generation?.layer_count??1:retrieval?.attention_encodings?.[0]?.layer_count??128;
  const headCount=selectedRole==='generation'?generation?.query_head_count??1:retrieval?.attention_encodings?.[0]?.query_head_count??128;
  const properties=method?.parameters_schema.properties as Record<string,{default?:unknown}>|undefined;
  const clear=()=>{setExperiment(undefined);setJob(undefined);setError(undefined);};
  useEffect(()=>{
    if(!method)return;
    const props=method.parameters_schema.properties as Record<string,{default?:unknown}>;
    const values:Record<string,unknown>=Object.fromEntries(Object.entries(props).filter(([,s])=>s.default!==undefined).map(([k,s])=>[k,s.default]));
    if(props.layer)values.layer=Math.min(layer,layerCount-1);
    if(props.head)values.head=Math.min(head,headCount-1);
    if(props.source_index)values.source_index=sourceIndex??(selectedRole==='generation'?Math.max(0,(generation?.context.input_token_ids.length??1)-1):0);
    if(props.target_index)values.target_index=targetIndex??0;
    setParameters(values);
  // A chosen method starts from the workspace axes, then retains its explicit draft.
  // eslint-disable-next-line react-hooks/exhaustive-deps
  },[methodId,method?.variant_id,selectedRole]);
  useEffect(()=>{
    if(!running||!job)return;
    let alive=true,polling=false;
    const poll=window.setInterval(async()=>{
      // One poll at a time: a slow response must not land after a newer one.
      if(polling)return;polling=true;
      try {
        const next=unwrap(await api.GET('/api/v1/jobs/{job_id}',{params:{path:{job_id:job.job_id}}}));
        if(!alive)return;
        setError(undefined);
        if(next.state==='completed'&&next.output_resource_id){
          const saved=unwrap(await api.GET('/api/v1/attention-interventions/{experiment_id}',{params:{path:{experiment_id:next.output_resource_id}}}));
          if(alive){setExperiment(saved);setRevision(v=>v+1);}
        }else if(next.state==='failed'||next.state==='cancelled')setError(next.error_message??'Experiment cancelled.');
        if(alive)setJob(next);
      }catch(e){if(alive)setError(errorMessage(e));}finally{polling=false;}
    },2000);
    return()=>{alive=false;window.clearInterval(poll);};
  },[running,job?.job_id]);
  async function prepare() {
    if(!capture||!method)return;
    setBusy(true);setError(undefined);
    try {
      const page=pages.includes(selectedPage)?selectedPage:pages[0];
      setExperiment(unwrap(await api.POST('/api/v1/attention-interventions',{body:{capture_id:capture.capture_id,method_id:method.method_id,variant_id:method.variant_id,page_id:page,parameters}})));
      setRevision(v=>v+1);
    }catch(e){setError(errorMessage(e));}finally{setBusy(false);}
  }
  async function run() {
    if(!experiment)return;
    setBusy(true);setError(undefined);
    try {
      const accepted=unwrap(await api.POST('/api/v1/attention-interventions/{experiment_id}/runs',{params:{path:{experiment_id:experiment.experiment_id}},body:{idempotency_key:crypto.randomUUID()}}));
      const next=unwrap(await api.GET('/api/v1/jobs/{job_id}',{params:{path:{job_id:accepted.job_id}}}));
      if(next.state==='completed'&&next.output_resource_id){
        setExperiment(unwrap(await api.GET('/api/v1/attention-interventions/{experiment_id}',{params:{path:{experiment_id:next.output_resource_id}}})));
        setRevision(v=>v+1);
      }
      setJob(next);
    }catch(e){setError(errorMessage(e));}finally{setBusy(false);}
  }
  return <section aria-label="Attention interventions">
    <h2>Direct attention interventions</h2>
    <p>Compare the unchanged model inputs under a fresh baseline and one internal attention intervention. Head/layer ablation changes context outputs; edge, modality and region masking remove attention before softmax. Preparation checks the frozen input without running the model.</p>
    <fieldset disabled={busy||running}>
      <label>Model role<select aria-label="Intervention model role" value={selectedRole} onChange={e=>{clear();setSelectedRole(e.target.value as typeof selectedRole);}}><option value="generation">Generation</option><option value="retrieval">Retrieval</option></select></label>
      <p>{capture?`Source capture: ${capture.capture_id}`:'Select or acquire a capture for this model role first.'}</p>
      <label>Intervention method<select aria-label="Intervention method" value={methodId} onChange={e=>{clear();setMethodId(e.target.value);}}>{methods.map(m=><option key={m.method_id} value={m.method_id}>{m.name}</option>)}</select></label>
      {method&&<p>{method.summary}</p>}
      <label>Intervention page<select aria-label="Intervention page" value={pages.includes(selectedPage)?selectedPage:pages[0]??''} onChange={e=>{clear();setSelectedPage(e.target.value);}}>{pages.map((id,i)=><option key={id} value={id}>Page {record.page_ids.indexOf(id)+1||i+1}</option>)}</select></label>
      <label>Layer<input aria-label="Intervention layer" type="number" min={0} max={layerCount-1} value={Number(parameters.layer??0)} onChange={e=>{const v=e.target.valueAsNumber;if(Number.isInteger(v)){clear();setParameters(p=>({...p,layer:v}));}}}/></label>
      {properties?.head&&<label>Head<input aria-label="Intervention head" type="number" min={0} max={headCount-1} value={Number(parameters.head??0)} onChange={e=>{const v=e.target.valueAsNumber;if(Number.isInteger(v)){clear();setParameters(p=>({...p,head:v}));}}}/></label>}
      {method&&<MethodParameters method={method} layerCount={layerCount} headCount={headCount} parameters={parameters} onChange={next=>{clear();setParameters(next);}}/>}
      <button className="docattentionlens-button" disabled={!capture||!method} onClick={()=>void prepare()}>Prepare attention intervention</button>
    </fieldset>
    {experiment&&<section aria-label="Attention experiment plan">
      <h3>{experiment.kind==='prepared'?'Prepared experiment':'Saved experiment result'}</h3>
      <p>{methods.find(m=>m.method_id===experiment.method_id)?.name??experiment.method_id} · {experiment.role} · unchanged input pages {experiment.page_ids.length}</p>
      <p>Source population: {String(experiment.provenance.source_population)}</p>
      {experiment.kind==='prepared'&&<button className="docattentionlens-button docattentionlens-primary" disabled={busy||running} onClick={()=>void run()}>{running&&<Busy/>}Run attention intervention on GPU</button>}
      {experiment.result&&<ResearchResults result={experiment.result} primaryTable/>}
      <details><summary>Exact parameters and provenance</summary><pre>{JSON.stringify({parameters:experiment.parameters,intervention:experiment.intervention,source_capture_sha256:experiment.source_capture_sha256,model_revision:experiment.model_revision,outcomes:experiment.outcomes},null,2)}</pre></details>
      <button className="docattentionlens-button" onClick={()=>downloadBlob(new Blob([JSON.stringify(experiment,null,2)],{type:'application/json'}),experiment.experiment_id+'.json')}>Export attention experiment</button>
    </section>}
    {job&&<p aria-live="polite">{running&&<Busy/>}Experiment job: {job.state}</p>}
    {running&&job&&<button className="docattentionlens-button" onClick={async()=>{try{unwrap(await api.POST('/api/v1/jobs/{job_id}/cancel',{params:{path:{job_id:job.job_id}}}));}catch(e){setError(errorMessage(e));}}}>Cancel attention intervention</button>}
    {error&&<p role="alert">{error}</p>}
    {!!records.data?.items.length&&<details><summary>Saved attention experiments</summary><ul>{records.data.items.map(x=><li key={x.experiment_id}><button className="docattentionlens-link-button" disabled={running} onClick={()=>setExperiment(x)}>{x.kind} · {methods.find(m=>m.method_id===x.method_id)?.name??x.method_id} · {new Date(x.created_at).toLocaleString()}</button></li>)}</ul></details>}
  </section>;
}
