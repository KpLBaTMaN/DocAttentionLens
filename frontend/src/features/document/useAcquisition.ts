import { useCallback, useEffect, useRef, useState, type Dispatch, type SetStateAction } from 'react';
import { api, unwrap, errorMessage, type Case } from '../../api/client';
import { useRemote } from '../../app/remote';
import type { components } from '../../api/generated/schema';
import type { Capture } from './RetrievalPanel';
import type { Generation } from './GenerationPanel';
import { defaultSource } from './useRetrieverAttention';
import { estimateFullAttention, modelGeometry } from '../attention-tool/fullAttentionEstimate';

export type View = components['schemas']['ViewState'];
export type UpdateView = Dispatch<SetStateAction<View>>;

// These hooks belong to Workspace, rather than a tab or drawer. Acquisition,
// drafts and polling consequently survive every presentation change.
/** `model` is the retriever that re-acquisition uses: the open capture's own model. */
export function useRetrievalAcquisition(record: Case, update: UpdateView, captureAttention=true, model='builtin-retrieval') {
  const [candidates, setCandidates] = useState(record.page_ids.slice(0, 1));
  const [attentionPage, setAttentionPage] = useState(record.page_ids[0]);
  // Attention captures always include the diagnostic Page + question pass, so every reading is there when the
  // capture opens. An additional pass whose prompt also carries the case's reference answer is opt-in.
  const [captureJointAnswer,setCaptureJointAnswer] = useState(false);
  const jointAnswer = captureAttention&&captureJointAnswer;
  const [jobId, setJobId] = useState<string>();
  const [busy, setBusy] = useState(false);
  const [error, setError] = useState<string>();
  const [cursor, setCursor] = useState<string>();
  const plan = useRemote(useCallback(async () => unwrap(await api.GET('/api/v1/retrieval/plan', { params: { query: { model_registration_id: model, capture_joint_attention:captureAttention, joint_answer:jointAnswer } } })), [model,captureAttention,jointAnswer]), 5000);
  const records = useRemote(useCallback(async () => unwrap(await api.GET('/api/v1/retrieval/captures', { params: { query: { case_id: record.case_id, limit: 10, before: cursor } } })), [record.case_id, cursor]), jobId ? 3000 : undefined);
  const job = useRemote(useCallback(async () => jobId ? unwrap(await api.GET('/api/v1/jobs/{job_id}', { params: { path: { job_id: jobId } } })) : undefined, [jobId]), jobId ? 2000 : undefined);
  const choose = useCallback((capture?: Capture, explicit = true) => {
    update(s => ({ ...s, retrieval_capture_id: capture?.capture_id ?? null,
      generation_capture_id: s.retrieval_capture_id === capture?.capture_id ? s.generation_capture_id : null,
      retrieval_encoding_id:s.retrieval_capture_id===capture?.capture_id?s.retrieval_encoding_id:capture?.attention_encodings?.find(e=>e.pass_kind!=='query')?.encoding_id??'document',
      retrieval_source_index:s.retrieval_capture_id===capture?.capture_id?s.retrieval_source_index:defaultSource(capture?.attention_encodings?.find(e=>e.pass_kind!=='query')),
      attention_layer:s.retrieval_capture_id===capture?.capture_id?s.attention_layer:0,
      attention_head:s.retrieval_capture_id===capture?.capture_id?s.attention_head:0,
      retrieval_attention_parameters:s.retrieval_capture_id===capture?.capture_id?s.retrieval_attention_parameters:{},
      query_position: s.retrieval_capture_id === capture?.capture_id ? s.query_position : 0,
      evidence_signal: explicit ? 'retrieval' : s.evidence_signal,
      page_id: capture?.pages.some(p => p.page_id === s.page_id) ? s.page_id : capture?.pages[0]?.page_id ?? s.page_id,
      selected_element_id: capture?.pages.some(p => p.page_id === s.page_id) ? s.selected_element_id : null }));
  }, [update]);
  const completing = useRef<string | undefined>(undefined);
  useEffect(() => {
    if (job.data?.state === 'completed' && job.data.output_resource_id && completing.current !== jobId) {
      completing.current = jobId;
      void (async () => {
        try { choose(unwrap(await api.GET('/api/v1/retrieval/captures/{capture_id}', { params: { path: { capture_id: job.data!.output_resource_id! } } })), false); }
        catch (reason) { setError(errorMessage(reason)); }
        finally { setJobId(undefined); completing.current = undefined; }
      })();
    }
    if (job.data?.state === 'failed' || job.data?.state === 'cancelled') { setError(job.data.error_message ?? 'Retrieval job cancelled.'); setJobId(undefined); }
  }, [job.data, jobId, choose]);
  async function run() {
    setBusy(true); setError(undefined);
    try { setJobId(unwrap(await api.POST('/api/v1/retrieval/runs', { body: { model_registration_id:model, case_id: record.case_id, page_ids: candidates, capture_attention:captureAttention,capture_joint_attention:captureAttention, ...(jointAnswer?{joint_answer_source:'reference' as const}:{}), ...(captureAttention?{attention_page_id:candidates.includes(attentionPage)?attentionPage:candidates[0]}:{}), idempotency_key: crypto.randomUUID() } })).job_id); }
    catch (reason) { setError(errorMessage(reason)); }
    finally { setBusy(false); }
  }
  return { candidates, setCandidates, attentionPage, setAttentionPage, captureAttention, captureJointAnswer, setCaptureJointAnswer, jointAnswer, jobId, busy, error: error ?? job.error, plan, records, cursor, setCursor, choose, run };
}

/** Quick acquisition always uses the pinned generator with its default instruction. */
export const QUICK_GENERATOR = 'builtin';
export const QUICK_SYSTEM = 'Answer the question using only the supplied document page images. If the images do not provide enough evidence, say so. Give a concise answer.';

export function useGenerationAcquisition(record: Case, retrieval: Capture | undefined, update: UpdateView) {
  const [selected, setSelected] = useState<string[]>([]);
  // Opt-in: complete prefill attention for every layer (rollout, flow, graphs, structure, image-patch sources).
  const [fullAttention, setFullAttention] = useState(false);
  const [jobId, setJobId] = useState<string>();
  const [busy, setBusy] = useState(false);
  const [error, setError] = useState<string>();
  const [cursor, setCursor] = useState<string>();
  const plan = useRemote(useCallback(async () => unwrap(await api.GET('/api/v1/generation/plan')), []), 5000);
  const models = useRemote(useCallback(async () => unwrap(await api.GET('/api/v1/setup/models')), []));
  const model = models.data?.find(m => m.registration_id === QUICK_GENERATOR && m.role === 'generation');
  const fullAttentionEstimate = estimateFullAttention(modelGeometry(model), (selected.length ? selected : [undefined]).map(id => retrieval?.pages.find(p => p.page_id === id)), QUICK_SYSTEM + record.query);
  const records = useRemote(useCallback(async () => unwrap(await api.GET('/api/v1/generation/captures', { params: { query: { case_id: record.case_id, limit: 10, before: cursor } } })), [record.case_id, cursor]), jobId ? 3000 : undefined);
  const job = useRemote(useCallback(async () => jobId ? unwrap(await api.GET('/api/v1/jobs/{job_id}', { params: { path: { job_id: jobId } } })) : undefined, [jobId]), jobId ? 2000 : undefined);
  useEffect(() => { setSelected(retrieval ? [...retrieval.pages].sort((a, b) => b.score - a.score).slice(0, 1).map(p => p.page_id) : []); }, [retrieval]);
  const choose = useCallback((capture: Generation, explicit = true) => {
    update(s => {
      const same = s.generation_capture_id === capture.capture_id;
      const keepPage = capture.context.selected_page_ids.includes(s.page_id ?? '');
      return { ...s, generation_capture_id: capture.capture_id, retrieval_capture_id: capture.retrieval_capture_id,
        attention_parameters: same ? s.attention_parameters : {}, attention_analysis_id: same ? s.attention_analysis_id : null,
      query_position: s.retrieval_capture_id === capture.retrieval_capture_id ? s.query_position : 0,
        evidence_signal: explicit ? 'attention' : s.evidence_signal,
        attention_step: same ? s.attention_step : capture.steps[0]?.step ?? 0,
        attention_layer: same ? s.attention_layer : 0, attention_head: same ? s.attention_head : 0,
        page_id: keepPage ? s.page_id : capture.context.selected_page_ids[0], selected_element_id: keepPage ? s.selected_element_id : null };
    });
  }, [update]);
  const completing = useRef<string | undefined>(undefined);
  useEffect(() => {
    if (job.data?.state === 'completed' && job.data.output_resource_id && completing.current !== jobId) {
      completing.current = jobId;
      void (async () => {
        try { choose(unwrap(await api.GET('/api/v1/generation/captures/{capture_id}', { params: { path: { capture_id: job.data!.output_resource_id! } } })), false); }
        catch (reason) { setError(errorMessage(reason)); }
        finally { setJobId(undefined); completing.current = undefined; }
      })();
    }
    if (job.data?.state === 'failed' || job.data?.state === 'cancelled') { setError(job.data.error_message ?? 'Generation job cancelled.'); setJobId(undefined); }
  }, [job.data, jobId, choose]);
  async function run() {
    if (!retrieval) return;
    setBusy(true); setError(undefined);
    try { setJobId(unwrap(await api.POST('/api/v1/generation/runs', { body: { retrieval_capture_id: retrieval.capture_id, page_ids: selected, model_registration_id: QUICK_GENERATOR, system_prompt: QUICK_SYSTEM, capture_full_attention: fullAttention, image_token_limit: 512, idempotency_key: crypto.randomUUID() } })).job_id); }
    catch (reason) { setError(errorMessage(reason)); }
    finally { setBusy(false); }
  }
  return { selected, setSelected, fullAttention, setFullAttention, fullAttentionEstimate, model, jobId, busy, error: error ?? job.error, plan, records, cursor, setCursor, choose, run };
}
