import { useCallback } from 'react';
import { Link, useParams, useSearchParams } from 'react-router-dom';
import { api, unwrap } from '../../api/client';
import { Heading, Message } from '../../app/App';
import { useRemote } from '../../app/remote';
import { Workspace } from '../document/Workspace';
import { InvestigationSetup } from './InvestigationSetup';

export function AttentionTool() {
  const { investigationId } = useParams();
  const [params] = useSearchParams();
  const caseId = params.get('case');
  const captureId = params.get('capture');
  const generationId=params.get('generation');
  const revision = params.get('revision');
  const data = useRemote(useCallback(async () => {
    if (!caseId && !investigationId && !captureId && !generationId) return null;
    const initialGeneration=generationId?unwrap(await api.GET('/api/v1/generation/captures/{capture_id}',{params:{path:{capture_id:generationId}}})):undefined;
    const initialCapture = captureId ? unwrap(await api.GET('/api/v1/retrieval/captures/{capture_id}',{params:{path:{capture_id:captureId}}})) : undefined;
    const saved = investigationId ? unwrap(await api.GET('/api/v1/investigations/{investigation_id}', {params:{path:{investigation_id:investigationId},query:{revision:revision?Number(revision):undefined}}})) : undefined;
    const record = unwrap(await api.GET('/api/v1/cases/{case_id}', {params:{path:{case_id:saved?.case_id ?? initialGeneration?.case_id ?? initialCapture?.case_id ?? caseId!}}}));
    const document = unwrap(await api.GET('/api/v1/documents/{document_id}', {params:{path:{document_id:record.document_id}}}));
    return {record,document,saved,initialCapture,initialGeneration};
  }, [caseId, investigationId, revision,captureId,generationId]));
  if (params.has('configure') || (!caseId && !investigationId && !captureId && !generationId)) return <InvestigationSetup key={caseId??'new'} initialCaseId={caseId??undefined}/>;
  if (data.error) return <><Heading kicker="Document investigation" title="Attention Tool"/><Message error>{data.error}</Message><Link className="docattentionlens-button" to="/artifacts/inputs">Open inputs</Link></>;
  return data.data ? <Workspace key={`${data.data.record.case_id}/${data.data.saved?.revision ?? generationId ?? captureId ?? 'new'}`} {...data.data} initialRole={params.get('role')==='generation'?'generation':params.get('role')==='retrieval'||data.data.initialCapture&&!data.data.initialGeneration?'retrieval':'generation'} initialLens={params.get('lens')==='context'?'context':'generation'} initialFocus={params.get('focus')==='context'?'context':params.get('focus')==='generation'?'generation':'both'}/> : <Message>Reopening case and saved selection…</Message>;
}
