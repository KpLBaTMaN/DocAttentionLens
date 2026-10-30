import { useCallback, useEffect, useState } from 'react';
import { Link, useNavigate } from 'react-router-dom';
import { api, errorMessage, unwrap, type Document, type Case } from '../../api/client';
import { Message } from '../../app/App';
import { useRemote } from '../../app/remote';
import { RecordId } from '../artifacts/RecordId';
import { isTableGrounding } from '../tables/tableGeometry';
import { INPUT_GROUNDING } from './grounding';

export function Inputs({onChoose,onDocument}:{onChoose?:(record:Case)=>void;onDocument?:(document:Document)=>void} = {}) {
  const navigate = useNavigate();
  const datasets = useRemote(useCallback(async () => unwrap(await api.GET('/api/v1/datasets')), []), 3000);
  const sources = useRemote(useCallback(async () => unwrap(await api.GET('/api/v1/datasets/sources')), []));
  const [datasetId, setDataset] = useState('');
  const [document, setDocument] = useState<Document>();
  const [search, setSearch] = useState('');
  const [domain, setDomain] = useState('');
  const [cursor, setCursor] = useState<string>();
  const [questionCursor, setQuestionCursor] = useState<string>();
  const [variant, setVariant] = useState<'original'|'document_scoped'|'collection_scoped'>('collection_scoped');
  const [originalScope, setOriginalScope] = useState<'document'|'collection'>('document');
  const [busy, setBusy] = useState(false);
  const [error, setError] = useState<string>();
  const selected = datasets.data?.items.find(d => d.dataset_id === datasetId) ?? datasets.data?.items[0];
  const tableDataset = isTableGrounding(selected?.structure_kind);
  const docs = useRemote(useCallback(async () => selected ? unwrap(await api.GET('/api/v1/documents', {params:{query:{dataset_id:selected.dataset_id, search, domain:domain || undefined, before:cursor, limit:25}}})) : undefined, [selected?.dataset_id, search, domain, cursor]));
  const questions = useRemote(useCallback(async () => document ? unwrap(await api.GET('/api/v1/questions', {params:{query:{document_id:document.document_id, before:questionCursor, limit:25}}})) : undefined, [document?.document_id, questionCursor]));
  useEffect(() => { setDocument(undefined); setCursor(undefined); setQuestionCursor(undefined); }, [selected?.dataset_id, domain, search]);
  async function register(source: string) {
    setBusy(true); setError(undefined);
    try {
      const job = unwrap(await api.POST('/api/v1/datasets/register', {body:{source, idempotency_key:crypto.randomUUID()}}));
      navigate(`/artifacts/jobs/${job.job_id}`);
    } catch (reason) { setError(errorMessage(reason)); } finally { setBusy(false); }
  }
  async function open(questionId: string) {
    setBusy(true); setError(undefined);
    try {
      // Table questions have one wording and a table scope.
      const body = tableDataset ? {question_id:questionId, query_variant:'original' as const, task_scope:'table' as const}
        : {question_id:questionId, query_variant:variant, task_scope:variant === 'original' ? originalScope : variant === 'document_scoped' ? 'document' as const : 'collection' as const};
      const record = unwrap(await api.POST('/api/v1/cases', {body}));
      if(onChoose)onChoose(record);else navigate(`/attention-tool?case=${record.case_id}&configure=1`);
    } catch (reason) { setError(errorMessage(reason)); } finally { setBusy(false); }
  }
  return <>
    <div className="docattentionlens-job-launch"><div><h2>Original documents</h2><p>Register a supplied dataset, choose a document and freeze a question wording for inspection. Registration checks the source files in a CPU job.</p></div>
      <div className="docattentionlens-dataset-sources">{(sources.data?.items ?? [{source:'doclayout_mm_rag_bench', label:'Document layout benchmark', description:'', structure_kind:'document_layout' as const, present:true}]).filter(s => s.present).map(s => {
        const registered = datasets.data?.items.some(d => d.name === s.source);
        return <button key={s.source} className="docattentionlens-button" disabled={busy} title={s.description} data-source={s.source} onClick={() => void register(s.source)}>{registered ? `Check ${s.label.toLowerCase()} version` : `Register ${s.label.toLowerCase()}`}</button>;})}</div></div>
    {error && <Message error>{error}</Message>}
    {datasets.error ? <Message error>{datasets.error}</Message> : !datasets.data ? <Message>Loading registered inputs…</Message> : !selected ? <Message>No dataset registered. Registration checks the source assets and annotations in a CPU job.</Message> : <>
      {datasets.data.items.length > 1 && <label>Dataset<select value={selected.dataset_id} onChange={e => setDataset(e.target.value)}>{datasets.data.items.map(d => <option key={d.dataset_id} value={d.dataset_id}>{isTableGrounding(d.structure_kind) ? INPUT_GROUNDING.table_grounding.label : d.counts.elements>0 ? INPUT_GROUNDING.document_layout.label : 'Uploaded pages'} · {d.name} · {d.dataset_version.slice(0, 12)}</option>)}</select></label>}
      {(tableDataset||selected.counts.elements>0)&&<p className="docattentionlens-muted" data-testid="input-grounding">Grounding: {INPUT_GROUNDING[tableDataset?'table_grounding':'document_layout'].label}</p>}
      {tableDataset ? <div className="docattentionlens-input-summary" data-kind="table_grounding"><strong>Table grounding · {selected.counts.tables} tables on {selected.counts.pages} full pages · {selected.counts.cells} cells · {selected.counts.questions} questions</strong><span>Content version {selected.dataset_version.slice(0,12)}</span>
        <p>{INPUT_GROUNDING.table_grounding.description}</p>
        <details><summary>Geometry and provenance</summary><p>{selected.coordinate_convention}</p><p>Tables use the source export's crop-to-page transforms and projected logical cell boxes.</p></details></div>
      : <div className="docattentionlens-input-summary"><strong>{selected.counts.documents} documents · {selected.counts.pages} pages · {selected.counts.questions} questions · {selected.counts.elements} elements</strong><span>Content version {selected.dataset_version.slice(0,12)}</span><p>{selected.quality_counts.bbox_outside_or_degenerate} flagged boxes · {selected.quality_counts.empty_ocr_texts} empty OCR strings · {selected.quality_counts.multi_page_support_questions} questions with support across pages</p><details><summary>Dataset quality and vocabulary</summary><p>Types: {selected.element_types.join(', ')}</p><p>Categories: {selected.element_categories.join(', ')}</p><p>{selected.geometry_validation}</p><p>Manifest alias: pwc_documents → professional_services_reports. Source IDs are not inferred.</p></details></div>}
      <div className="docattentionlens-input-columns"><section className="docattentionlens-section"><h2>Choose a document</h2><div className="docattentionlens-input-filters"><label>Search filename<input value={search} onChange={e => setSearch(e.target.value)}/></label><label>Domain<select value={domain} onChange={e => setDomain(e.target.value)}><option value="">All domains</option>{Object.keys(selected.domain_counts).map(d => <option key={d}>{d}</option>)}</select></label></div>
        {docs.error ? <Message error>{docs.error}</Message> : !docs.data ? <Message>Loading documents…</Message> : docs.data.items.length === 0 ? <Message>No matching documents.</Message> : <><p>{docs.data.total} matching documents</p><ul className="docattentionlens-choice-list">{docs.data.items.map(doc => <li key={doc.key}><button className="docattentionlens-choice" aria-pressed={doc.document_id === document?.document_id} onClick={() => {setDocument(doc); setQuestionCursor(undefined);onDocument?.(doc);}}><strong>{doc.filename}</strong><span>{doc.total_pages} pages · {doc.question_count} questions · {doc.ref.domain}</span></button>{selected.adapter_version?.startsWith('uploaded_')&&<Link className="docattentionlens-quiet" to={`/documents/${doc.document_id}/elements`}>{doc.element_count?`Edit ${doc.element_count} elements`:'Mark elements'}</Link>}</li>)}</ul><div className="docattentionlens-actions">{cursor && <button className="docattentionlens-button" onClick={() => setCursor(undefined)}>First documents</button>}{docs.data.next_cursor && <button className="docattentionlens-button" onClick={() => setCursor(docs.data?.next_cursor ?? undefined)}>Next documents</button>}</div></>}
      </section><section className="docattentionlens-section"><h2>Choose a question</h2>{!document ? <p>Select a document to browse its native questions.</p> : <><p>{document.title ?? document.filename}</p>{!tableDataset && <><label>Question wording<select value={variant} onChange={e => setVariant(e.target.value as typeof variant)}><option value="collection_scoped">Collection scoped</option><option value="document_scoped">Document scoped</option><option value="original">Original</option></select></label>{variant === 'original' && <label>Task scope<select value={originalScope} onChange={e => setOriginalScope(e.target.value as typeof originalScope)}><option value="document">Document</option><option value="collection">Collection</option></select></label>}</>}
        {questions.error ? <Message error>{questions.error}</Message> : !questions.data ? <Message>Loading questions…</Message> : questions.data.items.length === 0 ? <Message>No questions in this document.</Message> : <><p>{questions.data.total} questions</p><ul className="docattentionlens-choice-list">{questions.data.items.map(q => <li key={q.key}><button className="docattentionlens-choice" disabled={busy} onClick={() => void open(q.question_id)}><strong>{tableDataset || variant === 'original' ? q.question : variant === 'document_scoped' ? q.document_scoped_question : q.collection_scoped_question}</strong><span>{q.table_grounding ? `${q.table_grounding.value_type} · table question${q.table_grounding.binding_status === 'needs_source_qa_review' ? ' · answer/cell mismatch' : ''}` : `${q.question_type} · ${q.support_page_ids.length} support pages`} · Open case →</span></button></li>)}</ul><div className="docattentionlens-actions">{questionCursor && <button className="docattentionlens-button" onClick={() => setQuestionCursor(undefined)}>First questions</button>}{questions.data.next_cursor && <button className="docattentionlens-button" onClick={() => setQuestionCursor(questions.data?.next_cursor ?? undefined)}>Next questions</button>}</div></>}
      </>}</section></div>
    </>}
    <p>Designated support and reference answers are evaluation annotations. They are excluded from the generator prompt.</p>
  </>;
}

export function SavedInvestigations() {
  const [cursor, setCursor] = useState<string>();
  const records = useRemote(useCallback(async () => unwrap(await api.GET('/api/v1/investigations', {params:{query:{limit:25,before:cursor}}})), [cursor]));
  return <><h2>Saved investigations</h2>{records.error ? <Message error>{records.error}</Message> : !records.data ? <Message>Loading investigations…</Message> : !records.data.items.length ? <Message>No saved investigations yet. Choose an input question and save its workspace.</Message> : <ul className="docattentionlens-record-list">{records.data.items.map(r => <li key={r.key}><Link to={`/attention-tool/${r.investigation_id}`}><strong>{r.name}</strong><span>Revision {r.revision}</span><time>{new Date(r.created_at).toLocaleString()}</time></Link><RecordId value={r.investigation_id} label="investigation"/></li>)}</ul>}<div className="docattentionlens-actions">{cursor && <button className="docattentionlens-button" onClick={() => setCursor(undefined)}>Latest investigations</button>}{records.data?.next_cursor && <button className="docattentionlens-button" onClick={() => setCursor(records.data?.next_cursor ?? undefined)}>Older investigations</button>}</div><Link className="docattentionlens-button" to="/artifacts/inputs">Choose inputs</Link></>;
}

export function SavedCases() {
  const [cursor, setCursor] = useState<string>();
  const records = useRemote(useCallback(async () => unwrap(await api.GET('/api/v1/cases', {params:{query:{limit:25,before:cursor}}})), [cursor]));
  return <><h2>Immutable cases</h2>{records.error ? <Message error>{records.error}</Message> : !records.data ? <Message>Loading cases…</Message> : !records.data.length ? <Message>No cases yet.</Message> : <ul className="docattentionlens-record-list">{records.data.map(r => <li key={r.key}><Link to={`/attention-tool?case=${r.case_id}`}><strong>{r.query}</strong><span>{r.query_variant.replaceAll('_',' ')} · {r.task_scope} scope · {r.page_ids.length} input pages</span></Link><RecordId value={r.case_id} label="case"/></li>)}</ul>}<div className="docattentionlens-actions">{cursor && <button className="docattentionlens-button" onClick={() => setCursor(undefined)}>Latest cases</button>}{records.data?.length === 25 && <button className="docattentionlens-button" onClick={() => setCursor(records.data?.at(-1)?.case_id)}>Older cases</button>}</div></>;
}
