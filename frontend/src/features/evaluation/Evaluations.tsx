import { useCallback, useState } from 'react';
import { Link, useNavigate, useParams, useSearchParams } from 'react-router-dom';
import type { components } from '../../api/generated/schema';
import { api, errorMessage, unwrap } from '../../api/client';
import { useRemote } from '../../app/remote';
import { RecordId } from '../artifacts/RecordId';
import { Heading, Message } from '../../app/App';

type Study = components['schemas']['StudyRecord'];
type StudyCase = components['schemas']['StudyCase'];
type Experiment = components['schemas']['ExperimentRecord'];
type Trial = components['schemas']['UtilityTrialRequest'];
type Review = components['schemas']['AnswerReviewRequest'];
type Evidence = components['schemas']['EvaluationEvidence'];
type Measurement = {arm:string;answer:string|null;answer_assessment:{verdict:string;basis:string;disagreement?:boolean};support_recall_at_2_within_candidates?:number|null;context_support_page_recall?:number|null;changed_page_score?:number};
type Report = {model_evaluation:{planned_acquisitions:number;completed_acquisitions:number;pending_or_missing_acquisitions:number;cases:{protocol_id:string;document_id:string;split:string;rubric:string;support_bearing:boolean;observations:{repetition:number;result_id:string;retrospective:boolean;measurements:Measurement[]}[];summary:unknown[]}[]};runs:{protocol_id:string;repetition:number;job_id:string|null;state:string|null;result_id:string|null;output_resource_id:string|null}[];evidence:Evidence[];utility:{recorded_trials:number;complete_pairs:number;unpaired_groups:number;participants:number;paired_time_difference_median_seconds:number|null;baseline_first_pairs:number;linked_first_pairs:number;conditions:unknown};scientific_validation:string;limitations:string[]};
const readable = (value:string) => value.replaceAll('_',' ');
const percent = (value:number|null|undefined) => value == null ? 'Unmeasured / no eligible support' : `${(value*100).toFixed(1)}%`;

function CaseReference({protocol}:{protocol:Experiment}) {
  const source = useRemote(useCallback(async()=>unwrap(await api.GET('/api/v1/cases/{case_id}',{params:{path:{case_id:protocol.source_case_id}}})),[protocol.source_case_id]));
  return <>{source.error&&<Message error>{source.error}</Message>}{source.data&&<p><strong>Evaluation reference:</strong> {source.data.reference_answer}<br/>{source.data.support_element_ids.length} designated elements across {source.data.support_page_ids.length} pages. <Link to={`/attention-tool?case=${source.data.case_id}`}>Inspect original annotations</Link></p>}</>;
}

export function SavedEvaluations() {
  const navigate = useNavigate();
  const [query] = useSearchParams();
  const requestedProtocol=query.get('protocol');
  const requested=useRemote(useCallback(async()=>requestedProtocol?unwrap(await api.GET('/api/v1/experiments/{experiment_id}',{params:{path:{experiment_id:requestedProtocol}}})):undefined,[requestedProtocol]));
  const [cursor,setCursor] = useState<string>();
  const [protocolCursor,setProtocolCursor] = useState<string>();
  const records = useRemote(useCallback(async()=>unwrap(await api.GET('/api/v1/evaluation/studies',{params:{query:{limit:10,before:cursor}}})),[cursor]));
  const protocols = useRemote(useCallback(async()=>unwrap(await api.GET('/api/v1/experiments',{params:{query:{limit:25,before:protocolCursor}}})),[protocolCursor]));
  const [selected,setSelected] = useState<string[]>(query.get('protocol')?[query.get('protocol')!]:[]);
  const [choices,setChoices] = useState<Record<string,{split:StudyCase['split'];rubric:StudyCase['rubric']}>>({});
  const [name,setName] = useState('');
  const [reviewer,setReviewer] = useState('');
  const [role,setRole] = useState<StudyCase['reviewer_role']>('engineering');
  const [kind,setKind] = useState<'engineering_pilot'|'researcher_evaluation'>('engineering_pilot');
  const [notes,setNotes] = useState('');
  const [answerCriteria,setAnswerCriteria] = useState('');
  const [criteria,setCriteria] = useState('');
  const [exclusions,setExclusions] = useState('');
  const [baseline,setBaseline] = useState('');
  const [tasks,setTasks] = useState('');
  const [repetitions,setRepetitions] = useState(2);
  const [checked,setChecked] = useState(false);
  const [busy,setBusy] = useState(false);
  const [error,setError] = useState<string>();
  const defaults = {split:'validation' as const,rubric:'human_adjudication_v1' as const};
  const availableProtocols=[...(protocols.data?.items??[]),...(requested.data&&!(protocols.data?.items??[]).some(p=>p.experiment_id===requested.data?.experiment_id)?[requested.data]:[])];
  async function create(){
    setBusy(true);setError(undefined);
    try {
      const cases = selected.map(protocol_id=>({protocol_id,...(choices[protocol_id]??defaults),reviewer,reviewer_role:role,review_notes:notes,answer_criteria:answerCriteria,support_labels_checked:true as const,reference_answer_checked:true as const}));
      const record=unwrap(await api.POST('/api/v1/evaluation/studies',{body:{name,cases,repetitions,selection_criteria:criteria,exclusions,baseline_instructions:baseline,task_instructions:tasks,study_kind:kind}}));
      navigate(`/evaluations/${record.study_id}`);
    } catch(reason){setError(errorMessage(reason));}finally{setBusy(false);}
  }
  return <><h2>Research evaluation</h2><p>Freeze reviewed cases, repeat actual acquisitions and record answer assessments or researcher task outcomes. Empty measurements remain pending.</p>
    {records.error?<Message error>{records.error}</Message>:!records.data?<Message>Loading saved evaluations…</Message>:<ul className="docattentionlens-record-list">{records.data.items.map(s=><li key={s.study_id}><Link to={`/evaluations/${s.study_id}`}><strong>{s.specification.name}</strong><span>{readable(s.specification.study_kind)} · {s.specification.cases.length} cases · {s.specification.repetitions} planned acquisitions per case</span></Link><RecordId value={s.study_id} label="evaluation"/></li>)}</ul>}
    {records.data?.items.length===0&&<Message>No evaluation plans yet.</Message>}
    {records.data?.next_cursor&&<button className="docattentionlens-button" onClick={()=>setCursor(records.data?.next_cursor??undefined)}>Older evaluations</button>}{cursor&&<button className="docattentionlens-button" onClick={()=>setCursor(undefined)}>Latest evaluations</button>}
    <section className="docattentionlens-section docattentionlens-evaluation-form"><h2>Prepare an evaluation plan</h2><p>Select finalized layout protocols. Inspect the original support annotations and reference answers before declaring review. Freeze selection criteria before new measurements. An engineering pilot can explicitly attach earlier results.</p>
      <label>Evaluation name<input value={name} onChange={e=>setName(e.target.value)} maxLength={200}/></label>
      <label>Evaluation scope<select value={kind} onChange={e=>setKind(e.target.value as typeof kind)}><option value="engineering_pilot">Engineering pilot</option><option value="researcher_evaluation">Researcher evaluation · independent review declaration required</option></select></label>
      {protocols.error&&<Message error>{protocols.error}</Message>}
      {requested.error&&<Message error>{requested.error}</Message>}
      {availableProtocols.filter(p=>p.record_kind==='experiment_protocol').map(p=><fieldset key={p.experiment_id}><legend><label><input type="checkbox" checked={selected.includes(p.experiment_id)} disabled={!selected.includes(p.experiment_id)&&selected.length>=8} onChange={e=>setSelected(v=>e.target.checked?[...v,p.experiment_id]:v.filter(id=>id!==p.experiment_id))}/>{p.name}</label></legend><Link to={`/experiments/${p.experiment_id}`}>Review protocol and actual images</Link>{selected.includes(p.experiment_id)&&<><CaseReference protocol={p}/><label>Document split<select value={choices[p.experiment_id]?.split??defaults.split} onChange={e=>setChoices(v=>({...v,[p.experiment_id]:{...(v[p.experiment_id]??defaults),split:e.target.value as StudyCase['split']}}))}><option value="discovery">Discovery</option><option value="validation">Validation</option></select></label><label>Answer rubric<select value={choices[p.experiment_id]?.rubric??defaults.rubric} onChange={e=>setChoices(v=>({...v,[p.experiment_id]:{...(v[p.experiment_id]??defaults),rubric:e.target.value as StudyCase['rubric']}}))}><option value="human_adjudication_v1">Human adjudication · long or ambiguous answers</option><option value="boolean_strict_v1">Strict true/false · true-or-false questions</option><option value="normalized_exact_v1">Normalized exact match · short questions</option></select></label></>}</fieldset>)}
      {protocols.data?.next_cursor&&<button className="docattentionlens-button" onClick={()=>setProtocolCursor(protocols.data?.next_cursor??undefined)}>Older protocols</button>}{protocolCursor&&<button className="docattentionlens-button" onClick={()=>setProtocolCursor(undefined)}>Latest protocols</button>}
      <p>{selected.length} protocols selected. Discovery and validation must use disjoint documents.</p>
      <label>Planned acquisitions per case<select value={repetitions} onChange={e=>setRepetitions(Number(e.target.value))}>{[1,2,3,4,5].map(n=><option key={n}>{n}</option>)}</select></label>
      <label>Case selection criteria<textarea value={criteria} onChange={e=>setCriteria(e.target.value)} rows={3} maxLength={5000}/></label>
      <label>Attempted cases and exclusions<textarea value={exclusions} onChange={e=>setExclusions(e.target.value)} rows={2} maxLength={5000}/></label>
      <label>Baseline instructions<textarea value={baseline} onChange={e=>setBaseline(e.target.value)} rows={3} maxLength={5000}/></label><p>Declare the comparison interface/materials, page similarity, heatmaps and manual annotations available, plus matched cases and compute access. DocAttentionLens records this protocol; it does not create an alternative baseline interface.</p>
      <label>Researcher task instructions and assessment criteria<textarea value={tasks} onChange={e=>setTasks(e.target.value)} rows={3} maxLength={5000}/></label>
      <label>Answer assessment criteria<textarea value={answerCriteria} onChange={e=>setAnswerCriteria(e.target.value)} rows={3} maxLength={3000}/></label><label>Case reviewer<input value={reviewer} onChange={e=>setReviewer(e.target.value)} maxLength={200}/></label>
      <label>Case reviewer role<select value={role} onChange={e=>setRole(e.target.value as typeof role)}><option value="engineering">Engineering review</option><option value="independent_researcher">Declared independent researcher</option></select></label>
      <label>Case review notes<textarea value={notes} onChange={e=>setNotes(e.target.value)} rows={3} maxLength={5000}/></label>
      <label><input type="checkbox" checked={checked} onChange={e=>setChecked(e.target.checked)}/>I checked support labels and reference answers against the original pages for every selected case.</label>
      <button className="docattentionlens-button docattentionlens-primary" disabled={busy||!selected.length||!checked||!name.trim()||!reviewer.trim()||[notes,criteria,baseline,tasks,answerCriteria].some(s=>s.trim().length<20)} onClick={()=>void create()}>Freeze evaluation plan</button>{error&&<Message error>{error}</Message>}
    </section></>;
}

function AnswerReviewForm({studyId,resultId,refresh}:{studyId:string;resultId:string;refresh:()=>void}) {
  const record = useRemote(useCallback(async()=>unwrap(await api.GET('/api/v1/experiments/{experiment_id}',{params:{path:{experiment_id:resultId}}})),[resultId]));
  const [reviewer,setReviewer]=useState('');const [role,setRole]=useState<Review['reviewer_role']>('engineering');
  const [judgements,setJudgements]=useState<Record<string,{verdict:'correct'|'incorrect'|'uncertain';rationale:string}>>({});
  const [error,setError]=useState<string>();const [busy,setBusy]=useState(false);
  async function save(){setBusy(true);setError(undefined);try{
    const rows=record.data!.arms.map(a=>({arm:a.name as Review['judgements'][number]['arm'],...judgements[a.name]}));
    unwrap(await api.POST('/api/v1/evaluation/studies/{study_id}/answer-reviews',{params:{path:{study_id:studyId}},body:{result_id:resultId,reviewer,reviewer_role:role,judgements:rows}}));refresh();
  }catch(reason){setError(errorMessage(reason));}finally{setBusy(false);}}
  return <details className="docattentionlens-section docattentionlens-evaluation-form"><summary>Record answer adjudication · {resultId}</summary>{record.error&&<Message error>{record.error}</Message>}<label>Answer assessor<input value={reviewer} onChange={e=>setReviewer(e.target.value)} maxLength={200}/></label><label>Answer assessor role<select value={role} onChange={e=>setRole(e.target.value as typeof role)}><option value="engineering">Engineering</option><option value="independent_researcher">Declared independent researcher</option></select></label>{(record.data?.outcomes??[]).map(o=>{const arm=String(o.arm),value=judgements[arm]??{verdict:'uncertain',rationale:''};return <fieldset key={arm}><legend>{readable(arm)}</legend><p>Observed answer: {String(o.answer)}</p><label>Verdict<select value={value.verdict} onChange={e=>setJudgements(v=>({...v,[arm]:{...value,verdict:e.target.value as typeof value.verdict}}))}><option value="uncertain">Uncertain / cannot adjudicate</option><option value="correct">Correct under the declared criteria</option><option value="incorrect">Incorrect under the declared criteria</option></select></label><label>Answer rationale<textarea value={value.rationale} onChange={e=>setJudgements(v=>({...v,[arm]:{...value,rationale:e.target.value}}))} maxLength={3000}/></label></fieldset>})}<button className="docattentionlens-button" disabled={busy||!reviewer.trim()||!record.data||record.data.arms.some(a=>(judgements[a.name]?.rationale.trim().length??0)<10)} onClick={()=>void save()}>Save immutable answer review</button>{error&&<Message error>{error}</Message>}</details>;
}

function UtilityForm({study,refresh}:{study:Study;refresh:()=>void}) {
  const [protocol,setProtocol]=useState(study.specification.cases[0].protocol_id);
  const [participant,setParticipant]=useState('');const [experience,setExperience]=useState('');
  const [task,setTask]=useState<Trial['task']>('locate_support');const [condition,setCondition]=useState<Trial['condition']>('baseline');
  const [order,setOrder]=useState<1|2>(1);const [duration,setDuration]=useState('');const [started,setStarted]=useState<number>();
  const [completed,setCompleted]=useState(false);const [diagnostic,setDiagnostic]=useState('unknown');const [reproduction,setReproduction]=useState('unknown');const [causal,setCausal]=useState('unknown');
  const [assessor,setAssessor]=useState('');const [role,setRole]=useState<Trial['assessor_role']>('engineering');const [notes,setNotes]=useState('');const [matched,setMatched]=useState(false);
  const [error,setError]=useState<string>();const [busy,setBusy]=useState(false);
  const bool=(v:string)=>v==='unknown'?null:v==='yes';
  async function save(){setBusy(true);setError(undefined);try{
    unwrap(await api.POST('/api/v1/evaluation/studies/{study_id}/utility-trials',{params:{path:{study_id:study.study_id}},body:{protocol_id:protocol,participant,experience,task,condition,order,elapsed_seconds:Number(duration),completed,diagnostic_correct:bool(diagnostic),reproduction_success:bool(reproduction),unsupported_causal_assertion:bool(causal),assessor,assessor_role:role,notes,matched_cases_and_compute_confirmed:true}}));
    setDuration('');setStarted(undefined);setMatched(false);refresh();
  }catch(reason){setError(errorMessage(reason));}finally{setBusy(false);}}
  return <details className="docattentionlens-section docattentionlens-evaluation-form"><summary>Record a completed researcher task</summary><p>Use participant pseudonyms. Record actual observed work under the frozen instructions. Timing is observer-reported; the optional timer stays local until saved. Give paired conditions opposite orders and counterbalance order across participants.</p>
    <label>Task case<select value={protocol} onChange={e=>setProtocol(e.target.value)}>{study.specification.cases.map(c=><option key={c.protocol_id}>{c.protocol_id}</option>)}</select></label>
    <label>Participant pseudonym<input value={participant} onChange={e=>setParticipant(e.target.value)} maxLength={100}/></label><label>Relevant experience<input value={experience} onChange={e=>setExperience(e.target.value)} maxLength={1000}/></label>
    <label>Task<select value={task} onChange={e=>setTask(e.target.value as typeof task)}><option value="locate_support">Locate designated supporting evidence</option><option value="distinguish_context">Distinguish retrieved from supplied context</option><option value="reproduce_investigation">Explain and reopen a saved investigation</option></select></label>
    <label>Interface condition<select value={condition} onChange={e=>setCondition(e.target.value as typeof condition)}><option value="baseline">Declared baseline</option><option value="docattentionlens">DocAttentionLens</option></select></label><label>Presentation order<select value={order} onChange={e=>setOrder(Number(e.target.value) as 1|2)}><option value={1}>First</option><option value={2}>Second</option></select></label>
    <div className="docattentionlens-actions"><button className="docattentionlens-button" disabled={started!==undefined} onClick={()=>{setStarted(performance.now());setDuration('');}}>Start task timer</button><button className="docattentionlens-button" disabled={started===undefined} onClick={()=>{setDuration(((performance.now()-started!)/1000).toFixed(2));setStarted(undefined);}}>Stop task timer</button></div>
    <label>Elapsed seconds<input type="number" min="0.01" max="86400" step="0.01" value={duration} onChange={e=>setDuration(e.target.value)}/></label><label><input type="checkbox" checked={completed} onChange={e=>setCompleted(e.target.checked)}/>Task completed</label>
    {([{label:'Diagnosis correct',value:diagnostic,set:setDiagnostic},{label:'Saved work reproduced',value:reproduction,set:setReproduction},{label:'Unsupported causal assertion observed',value:causal,set:setCausal}]).map(f=><label key={f.label}>{f.label}<select value={f.value} onChange={e=>f.set(e.target.value)}><option value="unknown">Unmeasured / not applicable</option><option value="yes">Yes</option><option value="no">No</option></select></label>)}
    <label>Task assessor<input value={assessor} onChange={e=>setAssessor(e.target.value)} maxLength={200}/></label><label>Task assessor role<select value={role} onChange={e=>setRole(e.target.value as typeof role)}><option value="engineering">Engineering</option><option value="independent_researcher">Declared independent researcher</option></select></label><label>Task observations and participant feedback<textarea value={notes} onChange={e=>setNotes(e.target.value)} rows={3} maxLength={5000}/></label>
    <label><input type="checkbox" checked={matched} onChange={e=>setMatched(e.target.checked)}/>The declared baseline and linked condition use the same cases and matched compute access.</label><button className="docattentionlens-button" disabled={busy||started!==undefined||!matched||!participant.trim()||!assessor.trim()||experience.trim().length<5||notes.trim().length<10||!(Number(duration)>0)} onClick={()=>void save()}>Save immutable task record</button>{error&&<Message error>{error}</Message>}
  </details>;
}

export function EvaluationView() {
  const {studyId=''}=useParams();const [revision,setRevision]=useState(0);const refresh=()=>setRevision(v=>v+1);
  const study=useRemote(useCallback(async()=>unwrap(await api.GET('/api/v1/evaluation/studies/{study_id}',{params:{path:{study_id:studyId}}})),[studyId]));
  const report=useRemote(useCallback(async()=>unwrap(await api.GET('/api/v1/evaluation/studies/{study_id}/report',{params:{path:{study_id:studyId}}})) as Report,[studyId,revision]),3000);
  const [error,setError]=useState<string>();const [busy,setBusy]=useState(false);const [attached,setAttached]=useState<Record<string,string>>({});
  async function run(protocol_id:string,repetition:number){setBusy(true);setError(undefined);try{unwrap(await api.POST('/api/v1/evaluation/studies/{study_id}/runs',{params:{path:{study_id:studyId}},body:{protocol_id,repetition}}));refresh();}catch(reason){setError(errorMessage(reason));}finally{setBusy(false);}}
  async function attach(protocol_id:string,repetition:number,key:string){setBusy(true);setError(undefined);try{unwrap(await api.POST('/api/v1/evaluation/studies/{study_id}/observations',{params:{path:{study_id:studyId}},body:{protocol_id,repetition,result_id:attached[key]}}));refresh();}catch(reason){setError(errorMessage(reason));}finally{setBusy(false);}}
  const s=study.data,r=report.data;
  return <><Heading kicker="Artifacts / Research evaluation" title="Evaluation"/><div className="docattentionlens-content"><Link to="/artifacts/evaluations">← Evaluation plans</Link>{study.error?<Message error>{study.error}</Message>:!s?<Message>Loading frozen evaluation plan…</Message>:<>
    <h2>{s.specification.name}</h2><p>{readable(s.specification.study_kind)} · {s.specification.cases.length} frozen cases · {s.specification.repetitions} seed-zero acquisitions per case</p>
    <Message>Software verification does not establish research validity. Independent reviews are declared identities. Repetitions measure reproducibility on these cases; they do not increase the number of independent questions. Baseline and task judgements require real researcher observations.</Message>
    <details className="docattentionlens-section"><summary>Frozen selection, reviews and task instructions</summary><p><strong>Selection:</strong> {s.specification.selection_criteria}</p><p><strong>Exclusions:</strong> {s.specification.exclusions||'None recorded'}</p><p><strong>Baseline:</strong> {s.specification.baseline_instructions}</p><p><strong>Tasks:</strong> {s.specification.task_instructions}</p><pre>{JSON.stringify(s.frozen_cases,null,2)}</pre></details>
    {r&&<><h2>Model measurements</h2><p data-testid="evaluation-progress">{r.model_evaluation.completed_acquisitions} / {r.model_evaluation.planned_acquisitions} planned acquisitions completed. {r.model_evaluation.pending_or_missing_acquisitions} pending or missing.</p>
      {r.model_evaluation.cases.map(c=><section key={c.protocol_id} className="docattentionlens-section docattentionlens-evaluation-form"><h3>{readable(c.split)} · {c.support_bearing?'Moves designated supporting elements':'Moves non-support elements'}</h3><Link to={`/experiments/${c.protocol_id}`}>Open frozen protocol and images</Link><p>Answer rubric: {readable(c.rubric)}. Strict Boolean accepts true/false with an optional final full stop; exact matching normalizes Unicode, case and whitespace. Neither rubric accepts paraphrases. Human disagreements remain uncertain.</p>
        {Array.from({length:s.specification.repetitions},(_,i)=>i+1).map(n=>{const key=`${c.protocol_id}:${n}`,slot=r.runs.find(row=>row.protocol_id===c.protocol_id&&row.repetition===n);return <div className="docattentionlens-evaluation-slot" key={n}><strong>Acquisition {n}</strong>{slot?<p>{slot.job_id?<Link to={`/artifacts/jobs/${slot.job_id}`}>Job · {slot.state}</Link>:'Attached measured result'} · {slot.output_resource_id||slot.result_id?<Link to={`/experiments/${slot.output_resource_id??slot.result_id}`}>Open result</Link>:null}</p>:<><button className="docattentionlens-button" disabled={busy} onClick={()=>void run(c.protocol_id,n)}>Acquire repetition {n}</button>{s.specification.study_kind==='engineering_pilot'&&<div><label>Existing result for acquisition {n}<input value={attached[key]??''} onChange={e=>setAttached(v=>({...v,[key]:e.target.value}))} placeholder="experiment_…"/></label><button className="docattentionlens-button" disabled={busy||!attached[key]?.trim()} onClick={()=>void attach(c.protocol_id,n,key)}>Attach existing measured result</button></div>}</>}</div>})}
        {c.observations.map(o=><div key={o.result_id}><h4>Acquisition {o.repetition} · {o.retrospective?'retrospective engineering evidence':'acquired after the plan was frozen'}</h4><div className="docattentionlens-table-scroll"><table><thead><tr><th>Arm</th><th>Observed answer</th><th>Assessment</th><th>Support recall@2 within candidates</th><th>Context support-page coverage</th><th>Changed page similarity</th></tr></thead><tbody>{o.measurements.map(m=><tr key={m.arm}><td>{readable(m.arm)}</td><td>{m.answer??'No generation'}</td><td>{m.answer_assessment.verdict}{m.answer_assessment.disagreement?' · reviewer disagreement':''}</td><td>{percent(m.support_recall_at_2_within_candidates)}</td><td>{percent(m.context_support_page_recall)}</td><td>{m.changed_page_score==null?'Not measured':m.changed_page_score.toFixed(6)}</td></tr>)}</tbody></table></div>{c.rubric==='human_adjudication_v1'&&o.measurements.some(m=>m.answer!==null)&&<AnswerReviewForm key={o.result_id} studyId={studyId} resultId={o.result_id} refresh={refresh}/>}</div>)}
        {c.summary.length>0&&<details><summary>Repetition consistency and rubric denominators</summary><pre>{JSON.stringify(c.summary,null,2)}</pre></details>}
      </section>)}
      <section className="docattentionlens-section"><h2>Researcher task comparison</h2><p data-testid="utility-progress">{r.utility.recorded_trials} recorded trials · {r.utility.complete_pairs} complete baseline/DocAttentionLens pairs · {r.utility.unpaired_groups} unpaired task groups · {r.utility.participants} participants.</p><p>Baseline first: {r.utility.baseline_first_pairs} pairs. DocAttentionLens first: {r.utility.linked_first_pairs} pairs.</p><p>Median paired time difference (DocAttentionLens minus baseline): {r.utility.paired_time_difference_median_seconds===null?'Unmeasured':`${r.utility.paired_time_difference_median_seconds.toFixed(2)} seconds`}. Descriptive observer reports; no population or efficiency claim.</p><UtilityForm key={studyId} study={s} refresh={refresh}/><details><summary>Task outcomes and explicit denominators</summary><pre>{JSON.stringify(r.utility.conditions,null,2)}</pre></details></section>
      <details className="docattentionlens-section"><summary>Saved reviews and task records</summary>{r.evidence.length===0?<p>No researcher records submitted.</p>:<pre>{JSON.stringify(r.evidence,null,2)}</pre>}</details>
      <section className="docattentionlens-section"><h2>Reopen and verify</h2><p>Plans, measurements and reviews reopen without model execution. Reports are derived from immutable records; their underlying hashes and references remain visible.</p><a className="docattentionlens-button" href={`/api/v1/evaluation/studies/${studyId}/report`} target="_blank" rel="noreferrer">Open evaluation report JSON</a><details><summary>Plan identity and limitations</summary><p><code>{s.content_sha256}</code></p><ul>{s.limitations.map(line=><li key={line}>{line}</li>)}</ul></details></section>
    </>}{report.error&&<Message error>{report.error}</Message>}
  </>}{error&&<Message error>{error}</Message>}</div></>;
}
