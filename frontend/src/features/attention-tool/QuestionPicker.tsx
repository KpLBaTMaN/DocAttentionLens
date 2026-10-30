import { useCallback, useEffect, useState } from 'react';
import { Link } from 'react-router-dom';
import { api, errorMessage, unwrap, type Document, type Case } from '../../api/client';
import { Message } from '../../app/App';
import { useRemote } from '../../app/remote';
import { isTableGrounding } from '../tables/tableGeometry';

type Variant = 'original' | 'document_scoped' | 'collection_scoped';
const WORDING: Record<Variant, string> = {original: 'Original', document_scoped: 'Document scoped', collection_scoped: 'Collection scoped'};
/** Native row/column IDs are numbered from one ("row-0004"). */
const ordinal = (id: string) => Number(id.split('-').at(-1));

/** Linked document/table and question lists. Registration and provenance stay in Library. */
export function QuestionPicker({document, record, onChoose, onDocument, onDatasetChange}: {
  document?: Document; record?: Case; onChoose: (record: Case) => void;
  onDocument: (document: Document) => void; onDatasetChange: () => void;
}) {
  const datasets = useRemote(useCallback(async () => unwrap(await api.GET('/api/v1/datasets')), []), 3000);
  const [datasetId, setDataset] = useState('');
  const [search, setSearch] = useState('');
  const [questionSearch, setQuestionSearch] = useState('');
  const [cursor, setCursor] = useState<string>();
  const [questionCursor, setQuestionCursor] = useState<string>();
  const [variant, setVariant] = useState<Variant>(() => record?.query_variant === 'original' || record?.query_variant === 'document_scoped' ? record.query_variant : 'collection_scoped');
  const [originalScope, setOriginalScope] = useState<'document' | 'collection'>(record?.task_scope === 'collection' ? 'collection' : 'document');
  const [busy, setBusy] = useState(false);
  const [error, setError] = useState<string>();
  // Uploaded collections have no dataset questions. Follow the draft's dataset until explicitly changed.
  const choices = datasets.data?.items.filter(d => d.counts.questions > 0) ?? [];
  const selected = choices.find(d => d.dataset_id === (datasetId || document?.ref.dataset_id)) ?? choices[0];
  const current = document && document.ref.dataset_id === selected?.dataset_id ? document : undefined;
  const tableDataset = !!selected && isTableGrounding(selected.structure_kind ?? 'document_layout');
  const docs = useRemote(useCallback(async () => selected ? unwrap(await api.GET('/api/v1/documents', {params: {query: {dataset_id: selected.dataset_id, search: tableDataset ? '' : search, before: tableDataset ? undefined : cursor, limit: 25}}})) : undefined, [selected?.dataset_id, tableDataset, search, cursor]));
  const tables = useRemote(useCallback(async () => selected && tableDataset ? unwrap(await api.GET('/api/v1/tables', {params: {query: {dataset_id: selected.dataset_id}}})) : undefined, [selected?.dataset_id, tableDataset]));
  const questions = useRemote(useCallback(async () => current ? unwrap(await api.GET('/api/v1/questions', {params: {query: {document_id: current.document_id, search: questionSearch, before: questionCursor, limit: 25}}})) : undefined, [current?.document_id, questionSearch, questionCursor]));
  useEffect(() => { setCursor(undefined); setSearch(''); setError(undefined); }, [selected?.dataset_id]);
  useEffect(() => { setQuestionCursor(undefined); setQuestionSearch(''); }, [current?.document_id]);
  useEffect(() => {
    if (!record || record.task_scope === 'table') return;
    setVariant(record.query_variant === 'original' || record.query_variant === 'document_scoped' ? record.query_variant : 'collection_scoped');
    setOriginalScope(record.task_scope === 'collection' ? 'collection' : 'document');
  }, [record?.case_id, record?.query_variant, record?.task_scope]);

  async function open(questionId: string) {
    setBusy(true); setError(undefined);
    try {
      const body = tableDataset ? {question_id: questionId, query_variant: 'original' as const, task_scope: 'table' as const}
        : {question_id: questionId, query_variant: variant, task_scope: variant === 'original' ? originalScope : variant === 'document_scoped' ? 'document' as const : 'collection' as const};
      onChoose(unwrap(await api.POST('/api/v1/cases', {body})));
    } catch (reason) { setError(errorMessage(reason)); } finally { setBusy(false); }
  }
  function selectDataset(id: string) {
    setDataset(id); setCursor(undefined); setSearch(''); setQuestionCursor(undefined); setQuestionSearch('');
    onDatasetChange();
  }

  if (datasets.error) return <Message error>{datasets.error}</Message>;
  if (!datasets.data) return <Message>Loading dataset questions…</Message>;
  if (!selected) return <Message>No dataset questions are registered yet. Register the benchmark in <Link to="/artifacts/inputs">Library → Inputs</Link>, or use your own documents.</Message>;
  const count = tableDataset ? selected.counts.tables ?? selected.counts.documents : selected.counts.documents;
  const noun = tableDataset ? 'table' : 'document';
  // Include a version only when needed to distinguish registrations with the same name.
  const datasetLabel = (d: typeof selected) => d.name + (choices.some(other => other.dataset_id !== d.dataset_id && other.name === d.name) ? ` · ${d.dataset_version}` : '');

  return <div className="docattentionlens-question-picker" data-kind={tableDataset ? 'table_grounding' : 'document_layout'}>
    {error && <Message error>{error}</Message>}
    <div className="docattentionlens-picker-source">
      {choices.length > 1 ? <label>Dataset<select aria-label="Dataset" value={selected.dataset_id} disabled={busy} onChange={e => selectDataset(e.target.value)}>{choices.map(d => <option key={d.dataset_id} value={d.dataset_id}>{datasetLabel(d)}</option>)}</select></label>
        : <strong>{selected.name}</strong>}
      <p className="docattentionlens-picker-dataset"><span>{count} {noun}{count === 1 ? '' : 's'} · {selected.counts.questions.toLocaleString()} questions</span><Link to="/artifacts/inputs">Input details</Link></p>
    </div>
    <div className="docattentionlens-picker-columns">
      <section aria-label={tableDataset ? 'Tables' : 'Documents'}>
        <h3>{tableDataset ? 'Table' : 'Document'}<small>{tableDataset ? tables.data?.items.length : docs.data?.total} {search ? 'matching' : 'available'}</small></h3>
        {!tableDataset && <label>Search documents<input type="search" placeholder="Filename" value={search} onChange={e => {setSearch(e.target.value); setCursor(undefined);}}/></label>}
        {(tableDataset ? tables.error ?? docs.error : docs.error) ? <Message error>{tableDataset ? tables.error ?? docs.error : docs.error}</Message>
          : !docs.data || tableDataset && !tables.data ? <Message>Loading {noun}s…</Message>
          : tableDataset ? <ul className="docattentionlens-choice-list">{tables.data!.items.map(t => {
            const doc = docs.data!.items.find(d => d.document_id === t.document_id);
            return <li key={t.table_id}><button type="button" className="docattentionlens-choice" data-testid="table-choice" aria-pressed={t.document_id === current?.document_id} disabled={busy || !doc} onClick={() => doc && onDocument(doc)}>
              <strong>{t.title}</strong><span>Page {t.source_page_number} of {t.source_page_count} · {t.row_count} × {t.column_count} · {t.cell_count} cells · {t.question_count} questions</span></button></li>;
          })}</ul>
          : !docs.data.items.length ? <Message>No matching documents.</Message> : <>
            <ul className="docattentionlens-choice-list">{docs.data.items.map(doc => <li key={doc.key}><button type="button" className="docattentionlens-choice" aria-pressed={doc.document_id === current?.document_id} disabled={busy} onClick={() => onDocument(doc)}><strong>{doc.filename}</strong><span>{doc.total_pages} page{doc.total_pages === 1 ? '' : 's'} · {doc.question_count} question{doc.question_count === 1 ? '' : 's'}</span></button></li>)}</ul>
            {(cursor || docs.data.next_cursor) && <div className="docattentionlens-actions">{cursor && <button type="button" className="docattentionlens-button" onClick={() => setCursor(undefined)}>First documents</button>}{docs.data.next_cursor && <button type="button" className="docattentionlens-button" onClick={() => setCursor(docs.data?.next_cursor ?? undefined)}>Next documents</button>}</div>}
          </>}
      </section>
      <section aria-label="Questions">
        <h3>Question{questions.data && current && <small>{questions.data.total} {questionSearch ? 'matching' : 'available'}</small>}</h3>
        {!current ? <p className="docattentionlens-picker-empty">Choose a {noun} to see its questions.</p> : <>
          <p className="docattentionlens-picker-current">{current.filename}</p>
          <label>Search questions<input type="search" placeholder={`In ${current.filename}`} value={questionSearch} onChange={e => {setQuestionSearch(e.target.value); setQuestionCursor(undefined);}}/></label>
          {!tableDataset && <details className="docattentionlens-picker-wording"><summary>Question wording and scope <span>{WORDING[variant]}{variant === 'original' ? ` · ${originalScope}` : ''}</span></summary>
            <div><label>Question wording<select aria-label="Question wording" value={variant} onChange={e => setVariant(e.target.value as Variant)}>{Object.entries(WORDING).map(([value, label]) => <option key={value} value={value}>{label}</option>)}</select></label>
              {variant === 'original' && <label>Task scope<select aria-label="Task scope" value={originalScope} onChange={e => setOriginalScope(e.target.value as typeof originalScope)}><option value="document">Document</option><option value="collection">Collection</option></select></label>}</div>
          </details>}
          {questions.error ? <Message error>{questions.error}</Message> : !questions.data ? <Message>Loading questions…</Message>
            : !questions.data.items.length ? <Message>{questionSearch ? 'No matching questions.' : `No questions for this ${noun}.`}</Message> : <>
              <ul className="docattentionlens-choice-list">{questions.data.items.map(q => {
                const g = tableDataset ? q.table_grounding : undefined;
                return <li key={q.key}><button type="button" className="docattentionlens-choice" data-testid={tableDataset ? 'table-question' : undefined} disabled={busy} onClick={() => void open(q.question_id)}>
                  <strong>{tableDataset || variant === 'original' ? q.question : variant === 'document_scoped' ? q.document_scoped_question : q.collection_scoped_question}</strong>
                  <span>{g ? `${g.value_type} · header path ${g.row_header_path.length} × ${g.column_header_path.length} · row ${g.answer_row_ids.map(ordinal).join('–')}, column ${g.answer_column_ids.map(ordinal).join('–')}`
                    : `${q.question_type} · ${q.support_page_ids.length} evidence page${q.support_page_ids.length === 1 ? '' : 's'}`}
                    {g?.binding_status === 'needs_source_qa_review' && <em className="docattentionlens-chip is-flag">answer/cell mismatch</em>}</span>
                </button></li>;
              })}</ul>
              {(questionCursor || questions.data.next_cursor) && <div className="docattentionlens-actions">{questionCursor && <button type="button" className="docattentionlens-button" onClick={() => setQuestionCursor(undefined)}>First questions</button>}{questions.data.next_cursor && <button type="button" className="docattentionlens-button" onClick={() => setQuestionCursor(questions.data?.next_cursor ?? undefined)}>Next questions</button>}</div>}
            </>}
        </>}
      </section>
    </div>
    <p className="docattentionlens-picker-annotation">{tableDataset ? 'Answer cells and header paths' : 'Evidence labels and reference answers'} guide evaluation; they are excluded from model input.</p>
  </div>;
}
