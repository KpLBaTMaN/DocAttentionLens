import { useCallback, useEffect, useRef, useState } from 'react';
import { Link, useNavigate, useParams } from 'react-router-dom';
import { api, errorMessage, unwrap } from '../../api/client';
import { Heading, Message } from '../../app/App';
import { useRemote } from '../../app/remote';
import { SavedExperiments } from '../experiments/Experiments';
import { SavedEvaluations } from '../evaluation/Evaluations';
import { Inputs, SavedCases, SavedInvestigations } from '../inputs/Inputs';
import { RecordId } from './RecordId';
import '../../styles/library.css';

/** Library navigation: everyday sections first, the rest under "More". Planned sections stay off the list. */
const PRIMARY_SECTIONS = [['investigations', 'Investigations'], ['runs', 'Runs and captures'], ['inputs', 'Inputs']] as const;
const MORE_SECTIONS = [['cases', 'Cases'], ['experiments', 'Experiments'], ['evaluations', 'Evaluations'], ['jobs', 'Jobs']] as const;
const PLANNED_SECTIONS: Record<string, string> = {results: 'Results', exports: 'Exports'};
const readable = (value: string) => value.replaceAll('_', ' ');
const date = (value: string) => new Date(value).toLocaleString();
const jobLabel = (kind?: string) => kind === 'layout_experiment' ? 'Controlled layout experiment' : kind === 'generation_capture' ? 'Generator and attention' : kind === 'retrieval_capture' ? 'ColPali retrieval' : kind === 'dataset_import' ? 'Dataset registration' : 'CPU integrity check';

function SavedCaptures() {
  const [generatorCursor,setGeneratorCursor]=useState<string>();
  const [cursor,setCursor]=useState<string>();
  const records=useRemote(useCallback(async()=>unwrap(await api.GET('/api/v1/retrieval/captures',{params:{query:{limit:10,before:cursor}}})),[cursor]));
  const generations=useRemote(useCallback(async()=>unwrap(await api.GET('/api/v1/generation/captures',{params:{query:{limit:10,before:generatorCursor}}})),[generatorCursor]));
  return <><h2>Generator answers and attention</h2><p>Saved context, output and captured prediction rows can reopen without GPU execution.</p>{generations.error?<Message error>{generations.error}</Message>:!generations.data?<Message>Loading saved answers…</Message>:!generations.data.items.length?<Message>No generator captures yet. Open retrieval evidence and choose context pages.</Message>:<ul className="docattentionlens-record-list">{generations.data.items.map(c=><li key={c.capture_id}><Link to={`/attention-tool?generation=${c.capture_id}`}><strong>{c.context.query}</strong><span>{c.answer}</span><span>{c.context.pages.length} supplied pages · {c.steps.length} captured prediction rows</span><time dateTime={c.created_at}>{date(c.created_at)}</time></Link><RecordId value={c.capture_id} label="capture"/></li>)}</ul>}{generations.data?.next_cursor&&<button className="docattentionlens-button" onClick={()=>setGeneratorCursor(generations.data?.next_cursor??undefined)}>Older generator captures</button>}{generatorCursor&&<button className="docattentionlens-button" onClick={()=>setGeneratorCursor(undefined)}>Latest generator captures</button>}<h2>Retrieval runs and captures</h2><p>Saved ColPali evidence remains available when GPU execution is disconnected.</p>{records.error?<Message error>{records.error}</Message>:!records.data?<Message>Loading saved captures...</Message>:!records.data.items.length?<Message>No real retrieval captures yet. Open a case to run ColPali.</Message>:<ul className="docattentionlens-record-list">{records.data.items.map(c=><li key={c.capture_id}><Link to={`/attention-tool?capture=${c.capture_id}`}><strong>{c.query}</strong><span>{c.pages.length} candidates · late-interaction similarity</span><time dateTime={c.created_at}>{date(c.created_at)}</time></Link><RecordId value={c.capture_id} label="capture"/></li>)}</ul>}{cursor&&<button className="docattentionlens-button" onClick={()=>setCursor(undefined)}>Latest captures</button>}{records.data?.next_cursor&&<button className="docattentionlens-button" onClick={()=>setCursor(records.data?.next_cursor??undefined)}>Older captures</button>}</>;
}

function Jobs() {
  const navigate = useNavigate();
  const [busy, setBusy] = useState(false);
  const [error, setError] = useState<string>();
  const [cursor, setCursor] = useState<string>();
  const jobs = useRemote(useCallback(async () => unwrap(await api.GET('/api/v1/jobs', { params: { query: { limit: 25, before: cursor } } })), [cursor]), 2000);
  const runtime = useRemote(useCallback(async () => unwrap(await api.GET('/api/v1/runtime')), []), 5000);
  async function submit() {
    setBusy(true); setError(undefined);
    try {
      const job = unwrap(await api.POST('/api/v1/jobs', { body: { kind: 'integrity_check', resource_class: 'cpu', idempotency_key: crypto.randomUUID(), inputs: { text: 'DocAttentionLens CPU check' } } }));
      navigate(`/artifacts/jobs/${job.job_id}`);
    } catch (reason) { setError(errorMessage(reason)); } finally { setBusy(false); }
  }
  return <>
    <header className="docattentionlens-library-head"><div><h2>Jobs</h2><p>Execution records for captures, imports and experiments. Run CPU check submits a harmless checksum job; its receipt verifies durable execution and contains no model measurement.</p></div>
      <div className="docattentionlens-library-head-actions"><span className="docattentionlens-library-worker">{runtime.error ? 'Worker status unavailable' : runtime.data?.workers.some(w => w.available&&w.resource_class==='cpu') ? 'CPU worker connected' : 'No active worker — submitted jobs will stay queued'}</span><button className="docattentionlens-button" disabled={busy} onClick={() => void submit()}>{busy ? 'Submitting…' : 'Run CPU check'}</button></div></header>
    {error && <Message error>{error}</Message>}
    {jobs.error ? <Message error>{jobs.error}</Message> : !jobs.data ? <Message>Loading job history…</Message> : jobs.data.items.length === 0 ? <Message>No saved jobs yet.</Message> : <ul className="docattentionlens-record-list">{jobs.data.items.map(job => <li key={job.job_id}><Link to={`/artifacts/jobs/${job.job_id}`}><strong>{jobLabel(job.kind)}</strong><span className="docattentionlens-badge">{readable(job.state)}</span><time dateTime={job.created_at}>{date(job.created_at)}</time></Link><RecordId value={job.job_id} label="job"/></li>)}</ul>}
    <div className="docattentionlens-actions">{cursor && <button className="docattentionlens-button" onClick={() => setCursor(undefined)}>Latest jobs</button>}{jobs.data?.next_cursor && <button className="docattentionlens-button" onClick={() => setCursor(jobs.data?.next_cursor ?? undefined)}>Older jobs</button>}</div>
  </>;
}

function SectionLink({id, label, current}: {id: string; label: string; current: string}) {
  return <li><Link to={`/artifacts/${id}`} className={id === current ? 'active' : undefined} aria-current={id === current ? 'page' : undefined}>{label}</Link></li>;
}

export function ArtifactArea() {
  const { section = 'investigations' } = useParams();
  const nav = useRef<HTMLElement>(null);
  // On narrow screens the side list is a scrolling row; keep the current section in view.
  useEffect(() => {
    const list = nav.current, link = list?.querySelector<HTMLElement>('[aria-current="page"]');
    if (list && link && list.scrollWidth > list.clientWidth) list.scrollLeft = link.offsetLeft - list.offsetLeft - 12;
  }, [section]);
  const body = section === 'inputs' ? <Inputs/> : section === 'investigations' ? <SavedInvestigations/> : section === 'cases' ? <SavedCases/> : section === 'runs' ? <SavedCaptures/> : section === 'experiments' ? <SavedExperiments/> : section === 'evaluations' ? <SavedEvaluations/> : section === 'jobs' ? <Jobs/>
    : PLANNED_SECTIONS[section] ? <div className="docattentionlens-empty"><span className="docattentionlens-kicker">Planned</span><h2>{PLANNED_SECTIONS[section]}</h2><p>This record workflow is planned for a later milestone. No illustrative or synthetic records have been created.</p><Link className="docattentionlens-button" to="/artifacts/investigations">Open saved investigations</Link></div>
    : <Message error>Library section not found.</Message>;
  return <><Heading kicker="Saved work" title="Library"/><div className="docattentionlens-content docattentionlens-library">
    <nav className="docattentionlens-library-nav" aria-label="Library sections" ref={nav}>
      <ul>{PRIMARY_SECTIONS.map(([id, label]) => <SectionLink key={id} id={id} label={label} current={section}/>)}</ul>
      <p className="docattentionlens-library-group" id="library-more">More</p>
      <ul aria-labelledby="library-more">{MORE_SECTIONS.map(([id, label]) => <SectionLink key={id} id={id} label={label} current={section}/>)}</ul>
    </nav>
    <div className="docattentionlens-library-body">{body}</div>
  </div></>;
}

export function JobView() {
  const { jobId = '' } = useParams();
  const job = useRemote(useCallback(async () => unwrap(await api.GET('/api/v1/jobs/{job_id}', { params: { path: { job_id: jobId } } })), [jobId]), 1500);
  const [error, setError] = useState<string>();
  const [cancelling, setCancelling] = useState(false);
  async function cancel() {
    setCancelling(true);
    try { unwrap(await api.POST('/api/v1/jobs/{job_id}/cancel', { params: { path: { job_id: jobId } } })); }
    catch (reason) { setError(errorMessage(reason)); }
    finally { setCancelling(false); }
  }
  return <><Heading kicker="Library / Operational job" title={jobLabel(job.data?.kind)}/><div className="docattentionlens-content"><Link to="/artifacts/jobs">← Job history</Link>
    {job.error ? <Message error>{job.error}</Message> : !job.data ? <Message>Loading saved job…</Message> : <>
      <div className="docattentionlens-detail-lead"><span className="docattentionlens-badge" role="status">{readable(job.data.state)}</span><h2>{job.data.kind==='layout_experiment'?'Paired reviewed image manipulation':job.data.kind==='generation_capture'?'Frozen context, answer and prediction rows':job.data.kind==='retrieval_capture'?'Real query and page capture':job.data.kind==='dataset_import'?'Read-only source registration':'Durable execution receipt'}</h2><p>{job.data.kind==='layout_experiment'?'Four matched image arms are consumed by the selected RAG stages. Publication requires all outcomes to complete.':job.data.kind==='generation_capture'?'The GPU worker supplies selected retrieval page images and the frozen question, checks capture-on/off output equivalence and saves language prediction attention.':job.data.kind==='retrieval_capture'?'The GPU worker separately encodes the frozen query and selected original page images, then saves checked MaxSim scores and vectors.':job.data.kind==='dataset_import'?'The CPU worker verifies page assets and native annotations, records a content version and preserves raw source geometry.':'This infrastructure check does not execute a model or calculate a research result.'}</p><code>{job.data.job_id}</code></div>
      {job.data.output_resource_kind==='experiment_result'&&<Link className="docattentionlens-button docattentionlens-primary" to={`/experiments/${job.data.output_resource_id}`}>Open measured layout comparison</Link>}
      {job.data.output_resource_kind==='generation_capture'&&<Link className="docattentionlens-button docattentionlens-primary" to={`/attention-tool?generation=${job.data.output_resource_id}`}>Open answer and attention</Link>}
      {job.data.output_resource_kind==='retrieval_capture'&&<Link className="docattentionlens-button docattentionlens-primary" to={`/attention-tool?capture=${job.data.output_resource_id}`}>Open retrieval evidence</Link>}
      {job.data.output_resource_kind==='dataset' && <Link className="docattentionlens-button docattentionlens-primary" to="/artifacts/inputs">Browse registered documents</Link>}
      {job.data.result_id && <Link className="docattentionlens-button docattentionlens-primary" to={`/artifacts/operational_receipt/${job.data.result_id}`}>Open receipt</Link>}
      {['queued', 'running'].includes(job.data.state) && <button className="docattentionlens-button" disabled={cancelling} onClick={() => void cancel()}>{cancelling ? 'Requesting…' : 'Cancel job'}</button>}
      {job.data.state === 'cancellation_requested' && <Message>Cancellation requested. Waiting for the worker to reach a cancellation boundary.</Message>}
      {job.data.error_message && <Message error>{job.data.error_message} ({job.data.error_code})</Message>}
      <div className="docattentionlens-detail-grid"><section className="docattentionlens-section"><h2>Attempts</h2>{job.data.attempts.length ? <ul>{job.data.attempts.map(a => <li key={a.attempt_id}><strong>{a.state}</strong><p><code>{a.attempt_id}</code></p><p>Started {date(a.started_at)}{a.finished_at && ` · Finished ${date(a.finished_at)}`}</p></li>)}</ul> : <p>No worker has claimed this job.</p>}</section><section className="docattentionlens-section"><h2>Saved status history</h2><ol>{job.data.events.map(e => <li key={e.event_id}>{readable(e.state)} · {date(e.created_at)}</li>)}</ol></section></div>
      <details className="docattentionlens-section"><summary>Immutable input specification</summary><pre>{JSON.stringify(job.data.specification, null, 2)}</pre><p>Input SHA-256: <code>{job.data.input_hash}</code></p></details>
    </>}{error && <Message error>{error}</Message>}</div></>;
}

export function ReceiptView() {
  const { recordId = '' } = useParams();
  const receipt = useRemote(useCallback(async () => unwrap(await api.GET('/api/v1/artifacts/operational_receipt/{record_id}', { params: { path: { record_id: recordId } } })), [recordId]));
  return <><Heading kicker="Library / Verified immutable record" title="Operational receipt"/><div className="docattentionlens-content"><Link to="/artifacts/jobs">← Job history</Link>{receipt.error ? <Message error>{receipt.error}</Message> : !receipt.data ? <Message>Verifying receipt…</Message> : <>
    <div className="docattentionlens-detail-lead"><span className="docattentionlens-badge">{receipt.data.provenance.purpose==='controlled_layout_execution'?'Controlled layout experiment':receipt.data.provenance.purpose==='generation_execution'?'Generation execution':receipt.data.provenance.purpose==='retrieval_execution'?'Retrieval execution':receipt.data.provenance.purpose==='dataset_registration'?'Source registration':'Infrastructure verification'}</span><h2>{receipt.data.provenance.purpose==='controlled_layout_execution'?'Paired model outcomes completed':receipt.data.provenance.purpose==='generation_execution'?'Answer and attention capture completed':receipt.data.provenance.purpose==='retrieval_execution'?'ColPali capture completed':receipt.data.provenance.purpose==='dataset_registration'?'Dataset registration completed':'CPU check completed'}</h2><p>The saved record passed checksum verification on reopening.</p><code>{receipt.data.record_id}</code></div>
    <section className="docattentionlens-section"><h2>Receipt content</h2><pre>{JSON.stringify(receipt.data.output, null, 2)}</pre><p>Created {date(receipt.data.created_at)} · Producer {receipt.data.producer_version}</p><Link to={`/artifacts/jobs/${receipt.data.provenance.job_id}`}>Open originating job</Link></section>
    <details className="docattentionlens-section"><summary>Record provenance</summary><pre>{JSON.stringify(receipt.data.provenance, null, 2)}</pre></details>
  </>}</div></>;
}

