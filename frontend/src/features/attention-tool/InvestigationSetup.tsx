import { useCallback, useEffect, useRef, useState } from 'react';
import { Link, useNavigate } from 'react-router-dom';
import { api, errorMessage, unwrap, type Case } from '../../api/client';
import { Message } from '../../app/App';
import { ChevronRight, Plus } from '../../app/icons';
import { useRemote } from '../../app/remote';
import type { components } from '../../api/generated/schema';
import { ModelCapabilities, captureSummary } from './ModelCapabilities';
import { QuestionPicker } from './QuestionPicker';
import { INPUT_GROUNDING } from '../inputs/grounding';
import { FULL_ATTENTION_ENABLES, FULL_ATTENTION_KEEPS, estimateFullAttention, modelGeometry } from './fullAttentionEstimate';
import { TablePageInput, resolutionLabel, type ImagePositions } from '../tables/TablePageInput';
import { MarkingCard } from '../marking/MarkingCard';
// The generator's default instruction, the one quick acquisition always uses.
import { QUICK_SYSTEM as SYSTEM } from '../document/useAcquisition';
import '../../styles/setup.css';

type Focus = 'generation'|'context'|'both';
type ModelRole = 'generation'|'retrieval';
type Draft = {step:number;model:string;role:ModelRole;focus:Focus;mode:'dataset'|'upload';caseId?:string;documentId?:string;prompt:string;candidates:string[];attentionPage?:string;supplied:string[];retrievalId?:string;system:string;fullAttention:boolean;jobId?:string;jobKind?:'retrieval'|'generation';requestKey?:string;
  /** Merged image positions for a single supplied page; unset uses the default for the case (1,024 for tables). */
  imageTokens?:ImagePositions};
// Generation supplies the chosen pages directly unless the researcher asks a retriever to rank them first.
const INITIAL:Draft = {step:0,model:'builtin',role:'generation',focus:'generation',mode:'dataset',prompt:'',candidates:[],supplied:[],system:SYSTEM,fullAttention:false};
const STEPS = ['Model','Question and pages','Review'];
const TITLES = ['Choose a model','Choose a question and pages','Review and run'];
const DRAFT_KEY = 'docattentionlens-investigation-setup-v2';
const LEGACY_DRAFT_KEY = 'docattentionlens-investigation-setup-v1';
const OPEN_KEY = 'docattentionlens-investigation-setup-open';
function storedDraft():Draft|null {
  const saved=JSON.parse(localStorage.getItem(DRAFT_KEY)??'null') as Draft|null;
  if(saved)return saved;
  const legacy=JSON.parse(localStorage.getItem(LEGACY_DRAFT_KEY)??'null') as Draft|null;
  // Version 1 had four steps: Model, Focus, Inputs and Review. Focus is now part of the first step.
  return legacy&&typeof legacy.step==='number'?{...legacy,step:[0,0,1,2][legacy.step]}:null;
}
function readDraft(caseId?:string):Draft {
  try {
    const saved=storedDraft();
    if(saved&&[0,1,2].includes(saved.step)&&['generation','context','both'].includes(saved.focus)&&Array.isArray(saved.supplied)&&Array.isArray(saved.candidates)&&(!caseId||saved.caseId===caseId))return {...INITIAL,...saved,fullAttention:saved.fullAttention===true};
  }catch{/* Setup remains usable without browser storage. */}
  return {...INITIAL,caseId,step:caseId?1:0};
}
function saveDraft(draft:Draft) {
  localStorage.setItem(DRAFT_KEY,JSON.stringify(draft));
}
/** A reload in the same tab returns to the open setup; a new visit starts on the entry screen. */
function setupWasOpen() {
  try{return sessionStorage.getItem(OPEN_KEY)==='1';}catch{return false;}
}
function rememberSetupOpen(open:boolean) {
  try{if(open)sessionStorage.setItem(OPEN_KEY,'1');else sessionStorage.removeItem(OPEN_KEY);}catch{/* A reload then returns to the entry screen. */}
}
/** After elements are marked, setup continues on question and pages with the marked version of the document. */
export function adoptMarkedDocument(documentId:string) {
  try{
    const saved=storedDraft();
    saveDraft({...INITIAL,...saved,step:1,mode:'upload',documentId,caseId:undefined,candidates:[],supplied:[],attentionPage:undefined,retrievalId:undefined,requestKey:undefined,jobId:undefined,jobKind:undefined});
    rememberSetupOpen(true);
  }catch{/* Without browser storage setup starts afresh; the marked document is listed in Library › Inputs. */}
}
const hasProgress=(draft:Draft)=>draft.step>0||!!(draft.caseId||draft.documentId||draft.jobId||draft.prompt.trim())||
  draft.model!==INITIAL.model||draft.role!==INITIAL.role||draft.focus!==INITIAL.focus||draft.mode!==INITIAL.mode||
  draft.system!==INITIAL.system||draft.fullAttention||draft.imageTokens!==undefined;

type PagePreset='evidence'|'neighbours'|'distractors';
const PAGE_PRESETS:{id:PagePreset;label:string;text:string}[]=[
  {id:'evidence',label:'Evidence pages',text:'Only the designated evidence pages.'},
  {id:'neighbours',label:'Evidence + neighbours',text:'Evidence pages and the pages either side of them.'},
  {id:'distractors',label:'Evidence + distractors',text:'Evidence pages and evenly spaced non-evidence pages.'},
];
/** Deterministic selections built from designated support, kept in document order within the model limit. */
function presetPages(preset:PagePreset,pages:string[],support:string[],limit:number) {
  const chosen=new Set(pages.filter(id=>support.includes(id)).slice(0,limit));
  const rest=pages.filter(id=>!chosen.has(id)),fill=Math.min(Math.max(limit-chosen.size,0),rest.length);
  const extra=preset==='neighbours'?pages.flatMap((id,i)=>chosen.has(id)?[pages[i-1],pages[i+1]]:[]).filter(id=>id!==undefined&&!chosen.has(id)):
    preset==='distractors'?Array.from({length:fill},(_,i)=>rest[Math.floor((i+.5)*rest.length/fill)]):[];
  for(const id of extra)if(chosen.size<limit)chosen.add(id);
  return pages.filter(id=>chosen.has(id));
}
function defaultPages(c:Case,limit:number,fallback:number) {
  const evidence=presetPages('evidence',c.page_ids,c.support_page_ids,limit);
  return evidence.length?evidence:c.page_ids.slice(0,fallback);
}

function PageSelection({pages,support,selected,limit,ranking,label,onSelect}:{pages:string[];support:string[];selected:string[];limit:number;ranking:boolean;label:(id:string)=>string;onSelect:(ids:string[])=>void}) {
  const evidence=pages.filter(id=>support.includes(id));
  if(!evidence.length)return <p>This question has no designated evidence pages. Select pages manually.</p>;
  const active=PAGE_PRESETS.find(p=>{const ids=presetPages(p.id,pages,support,limit);return ids.length===selected.length&&ids.every(id=>selected.includes(id));});
  const included=selected.filter(id=>support.includes(id)).length;
  return <div className="docattentionlens-evidence-presets">
    <p><strong className="docattentionlens-evidence-badge">Evidence</strong><strong>{evidence.map(label).join(' · ')}</strong> {evidence.length===1?'is':'are'} designated by the dataset. The annotation guides this selection; it is never supplied to the model.</p>
    <div className="docattentionlens-actions" role="group" aria-label="Page selection">{PAGE_PRESETS.map(p=><button type="button" key={p.id} className="docattentionlens-button" aria-pressed={active?.id===p.id} disabled={p.id!=='evidence'&&evidence.length>=limit} onClick={()=>onSelect(presetPages(p.id,pages,support,limit))}>{p.label}</button>)}</div>
    <p className="docattentionlens-muted">{active?.text??'Custom selection.'} {selected.length} of {limit} pages selected · {included} of {evidence.length} evidence pages included{evidence.length>limit?` (at most ${limit} fit)`:''}.{ranking&&selected.length===1?' One candidate is scored but not ranked; add distractors to test whether the evidence ranks first.':''}</p>
  </div>;
}

export function InvestigationSetup({initialCaseId}:{initialCaseId?:string}) {
  const navigate=useNavigate();
  const [draft,setDraft]=useState(()=>readDraft(initialCaseId));
  // A case link or a running job goes straight to the steps; otherwise the entry screen offers examples or a new run.
  const [open,setOpen]=useState(()=>!!initialCaseId||!!draft.jobId||setupWasOpen());
  const [error,setError]=useState<string>();
  const [busy,setBusy]=useState(false);
  const [progress,setProgress]=useState('');
  const [modelVersion,setModelVersion]=useState(0);
  const [checkpointName,setCheckpointName]=useState('My checkpoint');
  const [showExamples,setShowExamples]=useState(false);
  const [storageError,setStorageError]=useState(false);
  const [allPages,setAllPages]=useState(false);
  useEffect(()=>{try{saveDraft(draft);setStorageError(false);}catch{setStorageError(true);}},[draft]);
  useEffect(()=>rememberSetupOpen(open),[open]);
  const models=useRemote(useCallback(async()=>unwrap(await api.GET('/api/v1/setup/models')),[modelVersion]));
  const generatorPlan=useRemote(useCallback(async()=>unwrap(await api.GET('/api/v1/generation/plan')),[]),5000);
  // The plan checks the retriever the run will submit: the selected model, or ColPali when it retrieves for a generator.
  const planRetriever=draft.role==='retrieval'?draft.model:'builtin-retrieval';
  const retrievalPlan=useRemote(useCallback(async()=>unwrap(await api.GET('/api/v1/retrieval/plan',{params:{query:{model_registration_id:planRetriever}}})),[planRetriever]),5000);
  const record=useRemote(useCallback(async()=>draft.caseId?unwrap(await api.GET('/api/v1/cases/{case_id}',{params:{path:{case_id:draft.caseId}}})):undefined,[draft.caseId]));
  const documentId=record.data?.document_id??draft.documentId;
  const document=useRemote(useCallback(async()=>documentId?unwrap(await api.GET('/api/v1/documents/{document_id}',{params:{path:{document_id:documentId}}})):undefined,[documentId]));
  const datasetId=document.data?.ref.dataset_id;
  const inputManifest=useRemote(useCallback(async()=>datasetId?unwrap(await api.GET('/api/v1/datasets/{dataset_id}/summary',{params:{path:{dataset_id:datasetId}}})):undefined,[datasetId]));
  const pageSources=inputManifest.data?.source_manifest.page_sources;
  const sources=(Array.isArray(pageSources)?pageSources:[]) as {page_id:string;filename:string;original_page_number:number}[];
  const sourceLabel=(id:string)=>{const source=sources.find(p=>p.page_id===id);return source?`${source.filename} · source page ${source.original_page_number}`:`Page ${(document.data?.page_ids.indexOf(id)??-1)+1}`;};
  const retrieval=useRemote(useCallback(async()=>draft.retrievalId?unwrap(await api.GET('/api/v1/retrieval/captures/{capture_id}',{params:{path:{capture_id:draft.retrievalId}}})):undefined,[draft.retrievalId]));
  const savedRetrieval=useRemote(useCallback(async()=>draft.caseId?unwrap(await api.GET('/api/v1/retrieval/captures',{params:{query:{case_id:draft.caseId,limit:25}}})):undefined,[draft.caseId,draft.retrievalId]));
  const examples=useRemote(useCallback(async()=>{
    if(!showExamples)return undefined;
    const saved=unwrap(await api.GET('/api/v1/generation/captures',{params:{query:{limit:25}}}));
    // Full-attention captures come first, so an example offers every method; otherwise newest first.
    const checked=saved.items.filter(c=>c.validation.status==='passed_for_this_capture').sort((a,b)=>Number(!!b.full_attention)-Number(!!a.full_attention));
    const cases=new Map<string,Case|undefined>(),items:typeof checked=[];
    // Bound case lookups; old captures may refer to inputs no longer registered.
    for(const c of checked){
      if(items.some(item=>item.case_id===c.case_id))continue;
      if(!cases.has(c.case_id)){try{cases.set(c.case_id,unwrap(await api.GET('/api/v1/cases/{case_id}',{params:{path:{case_id:c.case_id}}})));}catch{cases.set(c.case_id,undefined);}}
      if(cases.get(c.case_id)?.support_page_ids.some(id=>c.context.selected_page_ids.includes(id)))items.push(c);
      if(items.length===3)break;
    }
    return items;
  },[showExamples]));
  const job=useRemote(useCallback(async()=>draft.jobId?unwrap(await api.GET('/api/v1/jobs/{job_id}',{params:{path:{job_id:draft.jobId}}})):undefined,[draft.jobId]),draft.jobId?2000:undefined);
  const completing=useRef<string|undefined>(undefined);
  const steps=useRef<HTMLElement>(null);
  const shownStep=useRef(draft.step);
  // Moving between steps starts the new step at the top rather than at the old scroll position.
  useEffect(()=>{if(shownStep.current!==draft.step){shownStep.current=draft.step;steps.current?.scrollIntoView({block:'start'});}},[draft.step]);
  const locked=busy||!!draft.jobId;
  const isRetriever=draft.role==='retrieval';
  // Table grounding questions locate a table on their one page; the page is the model input.
  const tableCase=!!record.data?.table_grounding;
  // Ranking one page cannot change what the generator sees, so a table question supplies its page directly.
  const skipsRetrieval=tableCase&&!isRetriever&&draft.focus!=='generation';
  const needsRetrieval=!isRetriever&&draft.focus!=='generation'&&!tableCase;
  const runFocus:Focus=skipsRetrieval?'generation':draft.focus;
  const ranked=[...(retrieval.data?.pages??[])].sort((a,b)=>b.score-a.score);
  const availablePages=needsRetrieval?retrieval.data?.pages.map(p=>p.page_id)??[]:record.data?.page_ids??[];
  const model=models.data?.find(m=>m.registration_id===draft.model&&m.role===draft.role);
  const visibleModels=models.data?.filter(m=>m.role===draft.role)??[];
  const pageLimit=model?.capabilities.limits?.page_limit??2;
  const candidateLimit=isRetriever?model?.capabilities.limits?.candidate_limit??8:retrievalPlan.data?.candidate_limit??8;
  const support=record.data?.support_page_ids??[];
  const evidenceMark=(id:string)=>support.includes(id)?' · evidence':'';
  const focusedPages=support.length>0&&!allPages;
  const visiblePages=(selected:string[])=>record.data?.page_ids.filter(id=>!focusedPages||selected.includes(id)||support.includes(id))??[];
  const attentionPage=draft.candidates.includes(draft.attentionPage??'')?draft.attentionPage!:draft.candidates.find(id=>support.includes(id))??draft.candidates[0];
  const selectedPages=isRetriever?draft.candidates:draft.supplied;
  // Pixel sizes of the supplied pages give their image positions; ranked pages carry them, others are looked up.
  const suppliedKey=needsRetrieval||isRetriever?'':draft.supplied.join(',');
  const suppliedRecords=useRemote(useCallback(async()=>suppliedKey?Promise.all(suppliedKey.split(',').map(async id=>unwrap(await api.GET('/api/v1/pages/{page_id}',{params:{path:{page_id:id}}})))):undefined,[suppliedKey]));
  const pageSize=(id:string)=>(needsRetrieval?retrieval.data?.pages:suppliedRecords.data)?.find(p=>p.page_id===id);
  const tablePage=tableCase?record.data!.page_ids[0]:undefined;
  const pageTables=useRemote(useCallback(async()=>tablePage?unwrap(await api.GET('/api/v1/pages/{page_id}/tables',{params:{path:{page_id:tablePage}}})):undefined,[tablePage]));
  // One supplied page may use the finer image plan; table questions default to it.
  const singlePage=!isRetriever&&!needsRetrieval&&draft.supplied.length===1;
  const imageTokens:ImagePositions=singlePage?draft.imageTokens??(tableCase?1024:512):512;
  const geometry=modelGeometry(model);
  // Before pages are chosen, the estimate is for one page of unknown size.
  const estimatePages=draft.supplied.length?draft.supplied.map(pageSize):[undefined];
  const fullAttentionEstimate=isRetriever?undefined:estimateFullAttention(geometry&&{...geometry,maxPagePositions:imageTokens},estimatePages,draft.system+(draft.prompt||record.data?.query||''));
  // A page may hold several tables; the case's own table sets the resolution.
  const caseTable=pageTables.data?.items.find(t=>t.table_id===record.data?.table_grounding?.table_id)??pageTables.data?.items[0];
  const tableResolution=tableCase&&geometry&&suppliedRecords.data?.[0]&&caseTable?resolutionLabel(suppliedRecords.data[0],caseTable,geometry.cell,imageTokens):undefined;
  const fullAttentionScope=draft.supplied.length?`${draft.supplied.length} supplied page${draft.supplied.length===1?'':'s'}`:'one page; pages are chosen in the next step';
  useEffect(()=>{if(record.data&&record.data.case_id===draft.caseId&&!draft.prompt)setDraft(s=>({...s,prompt:record.data!.query,candidates:defaultPages(record.data!,candidateLimit,8),supplied:defaultPages(record.data!,pageLimit,1)}));},[record.data,draft.caseId,draft.prompt,candidateLimit,pageLimit]);
  const selectionValid=!!record.data&&record.data.case_id===draft.caseId&&!!model&&(isRetriever?
    draft.candidates.length>0&&draft.candidates.length<=candidateLimit&&draft.candidates.every(id=>record.data!.page_ids.includes(id)):
    draft.supplied.length>0&&draft.supplied.length<=pageLimit&&draft.supplied.every(id=>availablePages.includes(id))&&(!needsRetrieval||retrieval.data?.case_id===record.data.case_id));
  const promptEdited=!!draft.prompt&&draft.prompt!==record.data?.query;
  const runsUnavailable=!!generatorPlan.data&&!!retrievalPlan.data&&!generatorPlan.data.available&&!retrievalPlan.data.available;
  const summary=[model?.name,document.data?.filename,record.data?.case_id===draft.caseId&&record.data?`${selectedPages.length} ${isRetriever?'candidate':'supplied'} page${selectedPages.length===1?'':'s'}`:draft.step===1?'Choose a document and question':undefined].filter(Boolean).join(' · ');
  const reviewBlocker=promptEdited?'Use the edited question to continue':!record.data?'Choose a question to continue':!model?'Choose a model to continue':!selectionValid?needsRetrieval&&!retrieval.data?'Retrieve pages and choose the generator input to continue':'Select valid pages to continue':undefined;
  const update=(next:Partial<Draft>)=>setDraft(s=>({...s,...next}));

  useEffect(()=>{
    if(job.data?.job_id!==draft.jobId)return;
    if(job.data?.state==='failed'||job.data?.state==='cancelled'){
      setError(job.data.error_message??'Capture cancelled. Your inputs and completed retrieval are preserved.');
      setDraft(s=>({...s,jobId:undefined,jobKind:undefined,requestKey:undefined}));return;
    }
    if(job.data?.state!=='completed'||!job.data.output_resource_id||completing.current===draft.jobId)return;
    completing.current=draft.jobId;
    if(draft.jobKind==='generation') {
      const id=job.data.output_resource_id;
      setDraft(s=>({...s,jobId:undefined,jobKind:undefined,requestKey:undefined}));
      navigate(`/attention-tool?generation=${id}&focus=${runFocus}&lens=${runFocus==='context'?'context':'generation'}`);
    }else{
      const id=job.data.output_resource_id;
      if(draft.role==='retrieval'){
        setDraft(s=>({...s,jobId:undefined,jobKind:undefined,requestKey:undefined}));
        navigate(`/attention-tool?capture=${id}&role=retrieval`);return;
      }
      void api.GET('/api/v1/retrieval/captures/{capture_id}',{params:{path:{capture_id:id}}}).then(unwrap).then(capture=>{
        setDraft(s=>({...s,retrievalId:id,supplied:[...capture.pages].sort((a,b)=>b.score-a.score).slice(0,1).map(p=>p.page_id),jobId:undefined,jobKind:undefined,requestKey:undefined}));
      }).catch(reason=>{setError(errorMessage(reason));update({jobId:undefined,jobKind:undefined});}).finally(()=>{completing.current=undefined;});
    }
  },[job.data,draft.jobId,draft.jobKind,runFocus,draft.role,navigate]);

  function chooseRole(role:ModelRole){
    setError(undefined);
    update({role,model:models.data?.find(m=>m.role===role)?.registration_id??(role==='generation'?'builtin':'builtin-retrieval'),requestKey:undefined});
  }

  function chooseCase(c:Case){setError(undefined);setAllPages(false);setDraft(s=>({...s,caseId:c.case_id,documentId:c.document_id,prompt:c.query,candidates:defaultPages(c,candidateLimit,8),supplied:defaultPages(c,pageLimit,1),attentionPage:undefined,retrievalId:undefined,requestKey:undefined}));}
  async function uploadFiles(kind:'documents'|'model',files:File[]){
    if(!files.length)return;
    setBusy(true);setError(undefined);
    try{
      if(new Set(files.map(f=>f.name)).size!==files.length)throw new Error('Choose files with distinct filenames.');
      if(kind==='documents'&&files.reduce((n,f)=>n+f.size,0)>50*1024**2)throw new Error('The document collection must fit within 50 MiB.');
      const supported=kind==='model'?files.filter(f=>/^(config|generation_config|preprocessor_config|processor_config|tokenizer|tokenizer_config|special_tokens_map|vocab|chat_template)\.json$|^merges\.txt$|^chat_template\.jinja$|^model(?:-\d{5}-of-\d{5})?\.safetensors$|^model\.safetensors\.index\.json$/.test(f.name)):files;
      if(!supported.length)throw new Error('No compatible files found in the selection.');
      const session=unwrap(await api.POST('/api/v1/setup/uploads',{body:{kind}}));
      for(let i=0;i<supported.length;i++){
        setProgress(`Uploading ${i+1} of ${supported.length}: ${supported[i].name}`);
        const response=await fetch(`/api/v1/setup/uploads/${session.upload_id}/files?filename=${encodeURIComponent(supported[i].name)}`,{method:'PUT',headers:{'Content-Type':'application/octet-stream'},body:supported[i]});
        if(!response.ok)throw await response.json();
      }
      setProgress(kind==='documents'?'Rendering and registering pages…':'Checking checkpoint architecture and file integrity…');
      const body={filenames:supported.map(f=>f.name),name:checkpointName,role:draft.role};
      if(kind==='documents'){
        const doc=unwrap(await api.POST('/api/v1/setup/uploads/{upload_id}/documents',{params:{path:{upload_id:session.upload_id}},body}));
        update({documentId:doc.document_id,caseId:undefined,retrievalId:undefined,candidates:[],supplied:[]});
      }else{
        const imported=unwrap(await api.POST('/api/v1/setup/uploads/{upload_id}/model',{params:{path:{upload_id:session.upload_id}},body}));
        update({model:imported.registration_id});setModelVersion(v=>v+1);
      }
    }catch(reason){setError(errorMessage(reason));}finally{setBusy(false);setProgress('');}
  }
  async function freezePrompt(){
    if(!documentId)return;setBusy(true);setError(undefined);
    try{chooseCase(unwrap(await api.POST('/api/v1/setup/cases',{body:{document_id:documentId,query:draft.prompt}})));}catch(reason){setError(errorMessage(reason));}finally{setBusy(false);}
  }
  async function launch(kind:'retrieval'|'generation'){
    if(!record.data)return;setBusy(true);setError(undefined);
    const requestKey=draft.requestKey??crypto.randomUUID();
    update({requestKey});
    try{
      const accepted=kind==='retrieval'?unwrap(await api.POST('/api/v1/retrieval/runs',{body:{case_id:record.data.case_id,page_ids:draft.candidates,capture_attention:isRetriever,capture_joint_attention:isRetriever,...(isRetriever?{attention_page_id:attentionPage}:{}),model_registration_id:isRetriever?draft.model:'builtin-retrieval',idempotency_key:requestKey}})):
        unwrap(await api.POST('/api/v1/generation/runs',{body:{...(needsRetrieval?{retrieval_capture_id:draft.retrievalId}:{case_id:record.data.case_id}),page_ids:draft.supplied,model_registration_id:draft.model,system_prompt:draft.system,capture_full_attention:draft.fullAttention===true,image_token_limit:imageTokens,idempotency_key:requestKey}}));
      update({jobId:accepted.job_id,jobKind:kind});
    }catch(reason){setError(errorMessage(reason));}finally{setBusy(false);}
  }
  function toggle(id:string,key:'candidates'|'supplied',limit:number){setDraft(s=>({...s,[key]:s[key].includes(id)?s[key].filter(p=>p!==id):s[key].length<limit?[...s[key],id]:s[key],...(key==='candidates'?{retrievalId:undefined,supplied:[],requestKey:undefined}:{requestKey:undefined})}));}
  function selectPages(key:'candidates'|'supplied',ids:string[]){setDraft(s=>({...s,[key]:ids,...(key==='candidates'?{retrievalId:undefined,supplied:[],requestKey:undefined}:{requestKey:undefined})}));}
  const showAllToggle=support.length>0&&<button type="button" className="docattentionlens-button docattentionlens-page-scope" onClick={()=>setAllPages(v=>!v)}>{allPages?'Show selected and evidence pages':`Choose from all ${record.data?.page_ids.length??0} pages`}</button>;
  async function cancel(){if(!draft.jobId)return;try{unwrap(await api.POST('/api/v1/jobs/{job_id}/cancel',{params:{path:{job_id:draft.jobId}}}));}catch(reason){setError(errorMessage(reason));}}
  function openCapture(c:components['schemas']['GenerationCapture']){navigate(`/attention-tool?generation=${c.capture_id}&focus=both`);}
  function startNew(){
    if(locked||hasProgress(draft)&&!window.confirm('Starting a new investigation will replace your unfinished setup. Discard it and start a new investigation?'))return;
    setDraft(INITIAL);setError(undefined);setAllPages(false);
    setShowExamples(false);setOpen(true);
  }
  function startOver(){
    if(locked||hasProgress(draft)&&!window.confirm('Discard this investigation setup and return to the start?'))return;
    // Write storage now: leaving a case link remounts this component before effects would run.
    try{saveDraft(INITIAL);}catch{/* The storage notice already covers this. */}
    rememberSetupOpen(false);
    setDraft(INITIAL);setError(undefined);setAllPages(false);setOpen(false);
    if(initialCaseId)navigate('/attention-tool');
  }

  if(!open) return <div className="docattentionlens-setup">
    <section className="docattentionlens-setup-home">
      <h1 className="docattentionlens-title">Investigate document attention</h1>
      <p className="docattentionlens-setup-lead">Explore model attention on document pages.</p>
      {hasProgress(draft)&&<div className="docattentionlens-setup-resume" role="region" aria-label="Unfinished setup"><div><span className="docattentionlens-setup-draft-label">Unfinished setup</span>{document.data?.filename&&<p>{document.data.filename}</p>}{model&&<small>{model.name}</small>}</div><button className="docattentionlens-button" onClick={()=>setOpen(true)}>Resume <ChevronRight/></button></div>}
      {storageError&&<Message>Your browser cannot save this draft. Keep this page open until the capture is saved.</Message>}
      {runsUnavailable&&<p className="docattentionlens-muted" role="status">Model runs are currently unavailable. You can prepare a setup or open an example.</p>}
      <div className="docattentionlens-actions"><button className="docattentionlens-button docattentionlens-primary" disabled={locked} onClick={startNew}><Plus/> New investigation</button></div>
      <div className="docattentionlens-setup-links">
        <button className="docattentionlens-quiet" aria-expanded={showExamples} aria-controls="docattentionlens-setup-examples" onClick={()=>setShowExamples(v=>!v)}>Open an example</button>
        <Link className="docattentionlens-quiet" to="/artifacts/investigations">Saved investigations</Link>
      </div>
      {showExamples&&<section id="docattentionlens-setup-examples" className="docattentionlens-section docattentionlens-saved-captures" aria-label="Saved examples"><h2>Examples</h2><p>Open saved attention without a model run. Capture checks passed; answer correctness has not been evaluated.</p>{examples.error&&<Message error>{examples.error}</Message>}{!examples.data&&!examples.error&&<p>Checking available examples…</p>}<ul className="docattentionlens-choice-list">{examples.data?.map(c=><li key={c.capture_id}><button className="docattentionlens-choice" onClick={()=>openCapture(c)}><strong>{c.context.query}</strong><span>{c.context.pages.length} supplied pages · {String(c.provenance.model_id)} · capture checks passed{c.full_attention?' · full attention':''}</span></button></li>)}</ul>{examples.data?.length===0&&<p>No checked examples with designated evidence are available yet. <Link to="/artifacts/runs">Browse saved runs</Link> or start a new investigation.</p>}</section>}
    </section>
  </div>;

  function resetInputs(documentId?:string){update({documentId,caseId:undefined,prompt:'',supplied:[],candidates:[],attentionPage:undefined,retrievalId:undefined,requestKey:undefined});}
  const questionPicker=<QuestionPicker document={document.data} record={record.data} onChoose={chooseCase} onDocument={doc=>resetInputs(doc.document_id)} onDatasetChange={()=>resetInputs()}/>;

  return <div className="docattentionlens-setup docattentionlens-setup-wizard">
    <nav ref={steps} className="docattentionlens-setup-steps" aria-label="Investigation setup steps">{STEPS.map((name,i)=><button key={name} disabled={locked||(i===2&&!selectionValid)} aria-current={draft.step===i?'step':undefined} onClick={()=>update({step:i})}><span aria-hidden="true">{i<draft.step?'✓':i+1}</span>{name}</button>)}</nav>
    <header className="docattentionlens-setup-heading"><h1 className="docattentionlens-title">{TITLES[draft.step]}</h1><div className="docattentionlens-setup-summary"><p>{summary||'Choose a model'}</p><small role="status">{storageError?'Draft not saved':'Draft saved'}</small></div></header>
    {error&&<Message error>{error}</Message>}{record.error&&<Message error>{record.error}</Message>}{models.error&&<Message error>{models.error}</Message>}{storageError&&<Message>Your browser cannot save this draft. Keep this page open until the capture is saved.</Message>}
    {locked&&<Message>{progress||`${draft.jobKind==='retrieval'?'Retrieving candidate pages':'Generating answer and capturing attention'} · ${job.data?.state??'submitting'}`}{draft.jobId&&<><Link to={`/artifacts/jobs/${draft.jobId}`}> View job</Link><button className="docattentionlens-button" onClick={()=>void cancel()}>Cancel capture</button></>}{job.error&&<p>{job.error} · checking again; the draft retains the job ID.</p>}</Message>}
    <section className="docattentionlens-setup-form"><fieldset disabled={locked}>
      {draft.step===0&&<><span className="docattentionlens-kicker">Step 1 of 3</span><h2>What do you want to explain?</h2>
        <div className="docattentionlens-setup-focus docattentionlens-model-role" role="radiogroup" aria-label="Model type">
          <label className={!isRetriever?'is-selected':''}><input type="radio" name="model-role" value="generation" checked={!isRetriever} onChange={()=>chooseRole('generation')}/><small>Generation model</small><strong>How an answer was produced</strong><span>The model reads your question and pages, then writes an answer. See which parts of each page it attended to while predicting.</span></label>
          <label className={isRetriever?'is-selected':''}><input type="radio" name="model-role" value="retrieval" checked={isRetriever} onChange={()=>chooseRole('retrieval')}/><small>Retrieval model</small><strong>How pages were ranked</strong><span>The model encodes your question and each page to score how well they match. See the similarity behind the ranking and, where the model supports it, its attention.</span></label>
        </div>
        <label>{isRetriever?'Retrieval model':'Generation model'}<select aria-label={isRetriever?'Retrieval model':'Generation model'} value={draft.model} onChange={e=>update({model:e.target.value,requestKey:undefined})}>{visibleModels.map(m=><option key={m.registration_id} value={m.registration_id}>{m.name}{m.capabilities.can_run?'':' · adapter required'}</option>)}</select></label>
        <ModelCapabilities model={model} fullAttention={!isRetriever&&draft.fullAttention}/>
        {!isRetriever&&<label className="docattentionlens-checkbox docattentionlens-setup-option"><input type="checkbox" checked={draft.fullAttention} onChange={e=>update({fullAttention:e.target.checked,requestKey:undefined})}/><span><strong>Save full attention matrices</strong> (optional)<small>{FULL_ATTENTION_ENABLES} {FULL_ATTENTION_KEEPS} Estimated extra storage: {fullAttentionEstimate?`${fullAttentionEstimate.label} for ${model?.name} with ${fullAttentionScope}`:model?'unknown; this model does not declare its layer, head and patch dimensions':'shown once the model catalogue loads'}.</small></span></label>}
        {!isRetriever&&<label className="docattentionlens-checkbox docattentionlens-setup-option"><input type="checkbox" checked={draft.focus!=='generation'} onChange={e=>update({focus:e.target.checked?'both':'generation',requestKey:undefined})}/><span><strong>Let a retriever choose the pages first</strong> (optional)<small>ColPali ranks up to {candidateLimit} pages you pick, then you choose which of them the generator sees. This adds a retrieval run before review.</small></span></label>}
        <details><summary>Use your own checkpoint</summary><p>Select a local folder with safetensors weights, configuration, tokenizer and any image processor files. The model is registered under the selected type. Its layer, head and patch details come from the checkpoint configuration.</p><p>Launching requires a compatible extraction adapter. A model can appear in this catalogue before its adapter is available; its description will show that status.</p><label>Checkpoint name<input value={checkpointName} maxLength={100} onChange={e=>setCheckpointName(e.target.value)}/></label><label>Checkpoint folder<input ref={node=>{node?.setAttribute('webkitdirectory','');}} type="file" multiple onChange={e=>void uploadFiles('model',Array.from(e.target.files??[]))}/></label><p>Full floating-point safetensors packages up to 12 GiB. Files are copied into the local model store; full loading and capture validation occur at launch.</p></details>
      </>}
      {draft.step===1&&<><div className="docattentionlens-setup-input-tabs" role="group" aria-label="Question source"><button className="docattentionlens-quiet" aria-pressed={draft.mode==='dataset'} onClick={()=>update({mode:'dataset'})}>Dataset question</button><button className="docattentionlens-quiet" aria-pressed={draft.mode==='upload'} onClick={()=>update({mode:'upload'})}>Your own documents</button></div>
        {draft.mode==='dataset'&&<details className="docattentionlens-picker-disclosure" data-selected={!!record.data} open={!record.data}><summary hidden={!record.data}>Choose a different dataset question</summary>{questionPicker}</details>}
        {draft.mode==='upload'&&<><label>Upload documents<input type="file" multiple accept=".pdf,.png,.jpg,.jpeg,.webp" onChange={e=>void uploadFiles('documents',Array.from(e.target.files??[]))}/></label><p>PDFs and images · up to 50 MiB / 64 pages. Multiple files form one page collection in selection order. Pages arrive without elements; you can mark them after uploading.</p></>}
        {document.data&&<><strong>{document.data.filename} · {document.data.total_pages} pages</strong>{document.data.ref.domain==='uploaded'&&<MarkingCard document={document.data}/>}<label>Question / prompt<textarea rows={3} maxLength={4000} value={draft.prompt||record.data?.query||''} onChange={e=>update({prompt:e.target.value})}/></label>{(!record.data||draft.prompt!==record.data.query)&&<button className="docattentionlens-button" disabled={!draft.prompt.trim()} onClick={()=>void freezePrompt()}>Use this question</button>}</>}
        {record.data&&<><p className="docattentionlens-frozen-prompt">Question for this run: {record.data.query}</p>{promptEdited&&<Message>Select Use this question to continue with the edited wording. Each question is saved unchanged with its capture, so editing creates a new case.</Message>}
          {skipsRetrieval&&<Message>Table questions have one page, so there is nothing for a retriever to rank. The generator gets the page directly.</Message>}
          {record.data.table_grounding?.binding_status==='needs_source_qa_review'&&<Message>Answer/cell mismatch: the saved answer “{record.data.table_grounding.accepted_answers[0]}” differs from the text in its PDF cell (“{record.data.table_grounding.pdf_answer_text??'none'}”). You can still run it; grounding comparisons exclude this binding by default.</Message>}
          {isRetriever&&!tableCase&&<><h3>Pages to rank</h3><p>Select up to {candidateLimit} pages for {model?.name} to encode and rank against the question.</p>
            <PageSelection pages={record.data.page_ids} support={support} selected={draft.candidates} limit={candidateLimit} ranking label={sourceLabel} onSelect={ids=>selectPages('candidates',ids)}/>
            <div className={`docattentionlens-setup-pages${focusedPages?' is-focused':''}`}>{visiblePages(draft.candidates).map(id=><label key={id} className={`${draft.candidates.includes(id)?'is-selected':''}${support.includes(id)?' is-evidence':''}`}>{support.includes(id)&&<strong className="docattentionlens-evidence-badge">Evidence</strong>}<img loading="lazy" src={`/api/v1/pages/${id}/image`} alt={sourceLabel(id)}/><span><input type="checkbox" checked={draft.candidates.includes(id)} disabled={!draft.candidates.includes(id)&&draft.candidates.length>=candidateLimit} onChange={()=>toggle(id,'candidates',candidateLimit)}/>{sourceLabel(id)}</span></label>)}</div>{showAllToggle}
            <label>Document attention page<select aria-label="Document attention page" value={attentionPage??''} onChange={e=>update({attentionPage:e.target.value,requestKey:undefined})}>{draft.candidates.map(id=><option key={id} value={id}>{sourceLabel(id)}{evidenceMark(id)}</option>)}</select></label><p className="docattentionlens-muted">Full attention is kept for the question and for this one page. All selected pages are ranked. Vision-tower attention is outside this capture.</p></>}
          {needsRetrieval&&!tableCase&&<><h3>Pages for the retriever to rank</h3><p>Choose up to {candidateLimit} pages. ColPali ranks them, then you choose which ones the generator sees. Attention is captured from the generator afterwards.</p><PageSelection pages={record.data.page_ids} support={support} selected={draft.candidates} limit={candidateLimit} ranking label={sourceLabel} onSelect={ids=>selectPages('candidates',ids)}/><div className="docattentionlens-setup-page-checks">{record.data.page_ids.map(id=><label className="docattentionlens-checkbox" key={id}><input type="checkbox" checked={draft.candidates.includes(id)} disabled={!draft.candidates.includes(id)&&draft.candidates.length>=candidateLimit} onChange={()=>toggle(id,'candidates',candidateLimit)}/>Page {(document.data?.page_ids.indexOf(id)??-1)+1}{support.includes(id)&&<strong className="docattentionlens-evidence-badge">Evidence</strong>}</label>)}</div><button className="docattentionlens-button" disabled={!retrievalPlan.data?.available||!draft.candidates.length} onClick={()=>void launch('retrieval')}>Retrieve selected pages</button><p>{retrievalPlan.error??retrievalPlan.data?.reason}</p>{savedRetrieval.data?.items.length? <label>Or reuse a saved ranking<select aria-label="Saved context ranking" value={draft.retrievalId??''} onChange={e=>update({retrievalId:e.target.value||undefined,supplied:[],requestKey:undefined})}><option value="">Choose a ranking for this question</option>{savedRetrieval.data.items.map(c=><option key={c.capture_id} value={c.capture_id}>{new Date(c.created_at).toLocaleString()} · {c.pages.length} candidate pages</option>)}</select></label>:null}</>}
          {tableCase&&isRetriever&&<><h3>Page the retriever will encode</h3><p>Table questions have one page, so there is nothing to rank: {model?.name} scores this page on its own. The capture keeps its similarity to the question and, where the model supports it, its attention on the page.</p>
            <div className="docattentionlens-setup-pages docattentionlens-review-pages">{draft.candidates.map(id=><figure key={id}><img loading="lazy" src={`/api/v1/pages/${id}/image`} alt={sourceLabel(id)}/><figcaption>{sourceLabel(id)}{evidenceMark(id)}</figcaption></figure>)}</div></>}
          {tableCase&&!isRetriever&&<><h3>Page the generator will see</h3>{pageTables.error&&<Message error>{pageTables.error}</Message>}
            {suppliedRecords.data?.[0]&&pageTables.data?<TablePageInput page={suppliedRecords.data[0]} tables={pageTables.data.items} grounding={record.data.table_grounding} cell={geometry?.cell}
              positions={imageTokens} onPositions={p=>update({imageTokens:p,requestKey:undefined})} sourceLabel={sourceLabel(record.data.page_ids[0])}/>:<p>Loading the page and its table…</p>}</>}
          {!isRetriever&&!tableCase&&(!needsRetrieval||retrieval.data)&&<><h3>Pages the generator will see</h3><p>Select up to {pageLimit} pages. They are given to the model in the order you select them.</p>{needsRetrieval?support.length>0&&<p className="docattentionlens-muted">{ranked.some(p=>support.includes(p.page_id))?`Evidence rank: ${ranked.flatMap((p,i)=>support.includes(p.page_id)?[`${sourceLabel(p.page_id)} ${i+1} of ${ranked.length}`]:[]).join(' · ')}.`:'No designated evidence page is among the ranked candidates.'}</p>:<PageSelection pages={record.data.page_ids} support={support} selected={draft.supplied} limit={pageLimit} ranking={false} label={sourceLabel} onSelect={ids=>selectPages('supplied',ids)}/>}<div className={`docattentionlens-setup-pages${focusedPages&&!needsRetrieval?' is-focused':''}`}>{(needsRetrieval?ranked.map(p=>({id:p.page_id,number:p.page_number,score:p.score})):visiblePages(draft.supplied).map(id=>({id,number:(document.data?.page_ids.indexOf(id)??-1)+1,score:undefined}))).map(p=><label className={`${draft.supplied.includes(p.id)?'is-selected':''}${support.includes(p.id)?' is-evidence':''}`} key={p.id}>{support.includes(p.id)&&<strong className="docattentionlens-evidence-badge">Evidence</strong>}<img loading="lazy" src={`/api/v1/pages/${p.id}/image`} alt={`Source page ${p.number}`}/><span><input type="checkbox" checked={draft.supplied.includes(p.id)} disabled={!draft.supplied.includes(p.id)&&draft.supplied.length>=pageLimit} onChange={()=>toggle(p.id,'supplied',pageLimit)}/>Page {p.number}{draft.supplied.includes(p.id)?` · supplied ${draft.supplied.indexOf(p.id)+1}`:' · omitted'}<small>{sourceLabel(p.id)}</small>{p.score!==undefined&&<small>Retrieval score {p.score.toFixed(4)}</small>}</span></label>)}</div>{!needsRetrieval&&showAllToggle}</>}
        </>}
      </>}
      {draft.step===2&&<><span className="docattentionlens-kicker">Step 3 of 3</span><h2>{isRetriever?'Check what the retriever will see':'Check what the generator will see'}</h2><p>{isRetriever?model?.capabilities.input_description:'One run captures attention for the answer and for the supplied pages. The model also runs once without capturing attention, to check that capturing does not change the output.'}</p>
        <dl className="docattentionlens-setup-review"><dt>Analysed model</dt><dd>{model?.name} · {model?.revision.slice(0,12)}</dd><dt>Model type</dt><dd>{isRetriever?'Retrieval':'Generation'}</dd><dt>Investigation</dt><dd>{isRetriever?'Retrieval encoding and ranking':needsRetrieval?'Generation from retriever-ranked pages':'Generation from pages you chose'}</dd><dt>Question</dt><dd>{record.data?.query}</dd>{(tableCase||(document.data?.element_count??0)>0)&&<><dt>Grounding</dt><dd>{INPUT_GROUNDING[tableCase?'table_grounding':'document_layout'].label}</dd></>}<dt>{isRetriever?'Candidate pages':'Supplied image order'}</dt><dd>{selectedPages.map(id=>(sources.some(p=>p.page_id===id)?`Page ${(document.data?.page_ids.indexOf(id)??-1)+1}: ${sourceLabel(id)}`:sourceLabel(id))+evidenceMark(id)).join(isRetriever?' · ':' → ')}</dd>{support.length>0&&<><dt>Designated evidence</dt><dd>{(()=>{const included=support.filter(id=>selectedPages.includes(id)).length;return `${included} of ${support.length} included · ${support.map(sourceLabel).join(' · ')} · evaluation annotation, not model input`;})()}</dd></>}{!isRetriever&&<><dt>Context source</dt><dd>{needsRetrieval?'Reviewed retrieval candidates':'Directly supplied pages'}{needsRetrieval&&` · ${(retrieval.data?.pages.length??0)-draft.supplied.length} retrieved pages omitted`}</dd></>}{isRetriever&&<><dt>Document attention page</dt><dd>{attentionPage?sourceLabel(attentionPage)+evidenceMark(attentionPage):''}</dd></>}{tableCase&&record.data?.table_grounding&&<><dt>Input</dt><dd>Full page · {sourceLabel(record.data.page_ids[0])} · the question was written against the cropped table</dd>
          <dt>Question grounding</dt><dd>Answer cell, {record.data.table_grounding.row_header_path.length}-step row header path and {record.data.table_grounding.column_header_path.length}-step column header path · evaluation annotation, not model input{record.data.table_grounding.binding_status==='needs_source_qa_review'?' · answer/cell mismatch':''}</dd></>}
          {!isRetriever&&<><dt>Image resolution</dt><dd data-testid="review-resolution">{imageTokens===1024?'Table detail':'Standard'} · at most {imageTokens.toLocaleString()} image positions per page{tableResolution?` · ${tableResolution.isolable} of ${tableResolution.cells} table cells have a position of their own (estimate)`:''}</dd></>}
          <dt>Captured evidence</dt><dd>{captureSummary(model,{fullAttention:!isRetriever&&draft.fullAttention})}</dd>{!isRetriever&&<><dt>Full attention matrices</dt><dd>{draft.fullAttention?`Saved · ${fullAttentionEstimate?`${fullAttentionEstimate.label} extra`:'size unknown'}`:'Not saved · prediction and question rows only'}</dd></>}</dl>
        <div className="docattentionlens-setup-pages docattentionlens-review-pages" aria-label="Selected source pages">{selectedPages.map((id,index)=><figure key={id}><img loading="lazy" src={`/api/v1/pages/${id}/image`} alt={`Selected source page ${index+1}`}/><figcaption>{index+1}. {sourceLabel(id)}{evidenceMark(id)}</figcaption></figure>)}</div><p className="docattentionlens-muted">These are the source pages, in the order the model receives them, shown before model preprocessing. The saved capture keeps the exact input the model consumed.</p>
        {!isRetriever&&<details><summary>Advanced settings</summary><label>System instruction<textarea rows={4} maxLength={4000} value={draft.system} onChange={e=>update({system:e.target.value,requestKey:undefined})}/></label>
          {!tableCase&&<label className="docattentionlens-checkbox"><input type="checkbox" checked={imageTokens===1024} disabled={!singlePage} onChange={e=>update({imageTokens:e.target.checked?1024:512,requestKey:undefined})}/><span>Use up to 1,024 image positions for this page<small>{singlePage?'Finer detail for one supplied page. The default is 512.':'Available when exactly one page is supplied directly.'}</small></span></label>}{model?.capabilities.limits?.input_positions&&<p>Each page is resized to at most {model.capabilities.limits.image_positions_per_page} merged image positions. Inputs over {model.capabilities.limits.input_positions.toLocaleString()} positions are rejected, with no silent truncation. Reference answers, support labels and OCR are excluded.</p>}</details>}
        {isRetriever&&!model?.capabilities.attention_available&&<Message>{model?.capabilities.summary}</Message>}
        <div className="docattentionlens-setup-launch"><button className="docattentionlens-button docattentionlens-primary" disabled={!selectionValid||!model?.capabilities.can_run||(!isRetriever&&!draft.system.trim())||!(isRetriever?retrievalPlan.data?.attention_available:generatorPlan.data?.available)||promptEdited} onClick={()=>void launch(isRetriever?'retrieval':'generation')}>Run investigation</button><p>{model?.capabilities.can_run?(isRetriever?retrievalPlan.error??retrievalPlan.data?.reason:generatorPlan.error??generatorPlan.data?.reason):model?.capabilities.reason}</p></div>
        <p className="docattentionlens-muted">{model?.capabilities.attention_available?'Head, layer, method and display controls open in the right panel after the run.':'Available retrieval evidence and display controls open in the right panel. Attention controls require captured weights.'} You can leave this page while it runs; the draft remembers the job.</p>
      </>}
      <div className="docattentionlens-setup-next"><button type="button" className="docattentionlens-quiet docattentionlens-setup-restart" onClick={startOver}>Start over</button><div className="docattentionlens-setup-next-actions">{draft.step===1&&reviewBlocker&&<small id="docattentionlens-review-blocker">{reviewBlocker}</small>}<div>{draft.step>0&&<button className="docattentionlens-button" onClick={()=>update({step:draft.step-1})}>Back</button>}{draft.step<2&&<button className="docattentionlens-button docattentionlens-primary" aria-describedby={draft.step===1&&reviewBlocker?'docattentionlens-review-blocker':undefined} disabled={draft.step===0&&!model||draft.step===1&&!!reviewBlocker} onClick={()=>update({step:draft.step+1})}>{draft.step===1?'Review setup':'Continue'}</button>}</div></div></div>
    </fieldset></section>
  </div>;
}
