import { useCallback, useRef } from 'react';
import { api, unwrap } from '../../api/client';
import { useRemote } from '../../app/remote';
import type { Capture } from './RetrievalPanel';
import { decodePiece } from './TokenStrip';
import type { View } from './useAcquisition';
import { retrieverMappingGap } from './perspective';
import { selectedComparison } from './documentGrouping';
import { rolloutIntervalGap } from './rolloutInterval';

type SourceEncoding = {attention_mask: number[]; causal?: boolean; pass_kind?: string; answer_source?: string | null;
  positions?: {index:number; role:string; eligible?:boolean; piece?:string}[]};

const active = (encoding: SourceEncoding, role: string) =>
  encoding.positions?.filter(p => p.role === role && p.eligible !== false && !!encoding.attention_mask[p.index]) ?? [];

/** The last word of the question in a Page + question reading, not its closing punctuation. */
export function lastQuestionWord(encoding?: SourceEncoding) {
  if (!encoding) return undefined;
  const question = active(encoding, 'query_text');
  const words = question.filter(p => /[\p{L}\p{N}]/u.test(decodePiece(p.piece ?? '')));
  return (words.length ? words : question).at(-1)?.index;
}

/**
 * "Last answer token" in a Page + question + reference answer reading: the position whose output
 * predicts the answer's final token, as an answer-target method reads it (`target` is that final token).
 */
export function lastAnswerToken(encoding?: SourceEncoding) {
  if (!encoding) return undefined;
  const target = active(encoding, 'answer_text').at(-1);
  if (!target || target.index < 1 || !encoding.attention_mask[target.index - 1]) return undefined;
  return {source: target.index - 1, target};
}

/** Source shown when an encoding opens: a late position for causal encodings, position 0 for bidirectional ones. */
export function defaultSource(encoding?: SourceEncoding) {
  if (encoding?.pass_kind === 'joint') {
    const start = (encoding.answer_source ? lastAnswerToken(encoding)?.source : undefined) ?? lastQuestionWord(encoding);
    if (start !== undefined) return start;
  }
  return encoding?.causal ? Math.max(0, encoding.attention_mask.lastIndexOf(1)) : 0;
}

export function useRetrieverAttention(capture:Capture|undefined,state:View) {
  const catalogue=useRemote(useCallback(async()=>unwrap(await api.GET('/api/v1/retrieval/attention-methods')),[]));
  const encoding=capture?.attention_encodings?.find(e=>e.encoding_id===(state.retrieval_encoding_id??'document'))
    ??capture?.attention_encodings?.find(e=>e.pass_kind!=='query');
  const method=catalogue.data?.find(m=>m.method_id===(state.retrieval_attention_method??'attention.row'));
  const properties=method?.parameters_schema.properties as Record<string,unknown>|undefined;
  const enabled=state.investigation_model_role==='retrieval'&&state.retrieval_attention!==false&&!!encoding;
  const originalPage=capture?.pages.find(p=>p.page_id===encoding?.page_id);
  const encodingPage=originalPage&&encoding?.positions?.length?{...originalPage,positions:encoding.positions,ownership:encoding.ownership??originalPage.ownership}:originalPage;
  // A document-aware method on an encoding without mapped elements has no element mass to show.
  const unavailable=retrieverMappingGap(method,encoding,encodingPage?.ownership,state.retrieval_source_index??0)
    ?? encodingMethodGap(method,encoding,encoding?.pass_kind==='query'?capture?.query_positions:encodingPage?.positions,state.retrieval_source_index??0)
    ?? (encoding ? rolloutIntervalGap(method?.method_id,encoding.layer_count,state.attention_layer??0,state.retrieval_attention_parameters?.start_layer??0) : undefined);
  const kSupport=encoding?.attention_mask.filter((active,i)=>active&&(!encoding.causal||i<=(state.retrieval_source_index??0))).length;
  const kRequested=state.retrieval_attention_parameters?.k;
  const kExceeds=!!properties?.k&&typeof kRequested==='number'&&kSupport!==undefined&&kRequested>kSupport;
  const body=enabled&&method&&capture&&!unavailable&&!kExceeds?{encoding_id:encoding.encoding_id,source_index:state.retrieval_source_index??0,
    method_id:method.method_id,parameters:{...state.retrieval_attention_parameters,layer:state.attention_layer??0,
      ...(properties?.head?{head:state.attention_head??0}:{})}}:undefined;
  const key=JSON.stringify([capture?.capture_id,body]);
  const latest=useRef(key);
  latest.current=key;
  const remote=useRemote(useCallback(async()=>{
    if(!body||!capture)return undefined;
    await new Promise(resolve=>setTimeout(resolve,150));
    if(latest.current!==key)return undefined;
    return {key,result:unwrap(await api.POST('/api/v1/retrieval/captures/{capture_id}/attention',{
      params:{path:{capture_id:capture.capture_id}},body}))};
  // The serialized selection includes every scientific parameter and capture identity.
  // eslint-disable-next-line react-hooks/exhaustive-deps
  },[key]));
  const result=remote.data?.key===key?remote.data.result:undefined;
  const row=result?.row??selectedComparison(result?.comparisons,state.retrieval_comparison_index)?.row;
  const page=encodingPage;
  const positions=encoding?.pass_kind==='query'?capture?.query_positions:page?.positions;
  return {catalogue,encoding,method,enabled,result,row,positions,page,unavailable,error:remote.error,loading:!!body&&!result&&!remote.error};
}

export type RetrieverAttention = ReturnType<typeof useRetrieverAttention>;

/** Source/geometry prerequisites for General methods, independent of annotations. */
export function encodingMethodGap(method: {requires:string[];method_id?:string}|undefined,encoding:{attention_mask:number[];causal?:boolean}|undefined,positions:{role:string;bbox?:number[]|null}[]|undefined,source:number) {
  if(!method||!encoding||!positions)return undefined;
  if(method.requires.includes('image_mask')&&!positions.some(p=>p.role==='visual_patch'&&p.bbox))return 'This method needs mapped image targets; choose a document or diagnostic joint encoding.';
  if(method.requires.includes('captured_image_sources')&&!positions.some(p=>p.role==='visual_patch'))return 'This encoding has no image-patch sources; choose a document or diagnostic joint encoding.';
  if(method.requires.includes('captured_image_sources')&&positions[source]?.role!=='visual_patch')return 'Select a captured image-patch source.';
  if(method.requires.includes('captured_text_sources')&&positions[source]?.role==='visual_patch')return 'Select a captured textual (non-image) source.';
  if(method.requires.includes('multiple_source_rows')&&encoding.attention_mask.filter(Boolean).length<2)return 'This method needs at least two captured source rows.';
  if(method.method_id&&/^attention\.(text|vision)_to_(text|vision)$/.test(method.method_id)) {
    const image=method.method_id.endsWith('_to_vision');
    if(!positions.some((p,i)=>!!encoding.attention_mask[i]&&(!encoding.causal||i<=source)&&(p.role==='visual_patch')===image))return 'This source has no valid targets in the selected modality block.';
  }
  return undefined;
}
