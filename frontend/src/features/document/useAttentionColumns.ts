import { useCallback } from 'react';
import { api, unwrap } from '../../api/client';
import { useRemote } from '../../app/remote';
import type { Generation } from './GenerationPanel';
import type { Capture } from './RetrievalPanel';
import type { View } from './useAcquisition';

/** "What attends to this": a generator image patch's column, or one patch's similarity to every query token. */
export function useAttentionColumns({state, generation, capture, pageId, generatorActive, similarityActive}: {
  state: View; generation?: Generation; capture?: Capture; pageId: string;
  generatorActive: boolean; similarityActive: boolean;
}) {
  const layer = state.attention_layer ?? 0, head = state.attention_head ?? 0;
  const column = state.attention_direction === 'column';
  const generatorTarget = generatorActive && column && generation ? state.attention_target_index ?? null : null;
  const similarityTarget = similarityActive && state.similarity_direction === 'column' && capture?.pages.some(p => p.page_id === pageId)
    ? state.similarity_target_index ?? null : null;
  const generator = useRemote(useCallback(async () => generatorTarget === null || !generation ? undefined
    : unwrap(await api.GET('/api/v1/generation/captures/{capture_id}/received', {params: {path: {capture_id: generation.capture_id},
      query: {target_index: generatorTarget, layer, head}}})), [generation, generatorTarget, layer, head]));
  const similarity = useRemote(useCallback(async () => similarityTarget === null || !capture ? undefined
    : unwrap(await api.GET('/api/v1/retrieval/captures/{capture_id}/columns', {params: {path: {capture_id: capture.capture_id},
      query: {page_id: pageId, position_index: similarityTarget}}})), [capture, pageId, similarityTarget]));
  return {generator, generatorTarget, similarity, similarityTarget};
}

/** Column values indexed by source position; sources that cannot see the target stay undefined. */
export function columnValues(column?: {sources: {source_index: number; value: number}[]}) {
  const values: number[] = [];
  column?.sources.forEach(s => { values[s.source_index] = s.value; });
  return values;
}
