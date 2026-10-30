import { useCallback } from 'react';
import { type DisplaySettings, type Method } from '../../api/client';
import type { components } from '../../api/generated/schema';
import type { Generation } from './GenerationPanel';
import type { Capture } from './RetrievalPanel';
import type { RetrieverAttention } from './useRetrieverAttention';
import type { UpdateView, View } from './useAcquisition';
import { isRolloutInterval } from './rolloutInterval';
import { directionFor } from './methodState';
import { choicesForRole, shortReason, type MethodChoice } from './methodChoices';
import { hasMappedElements, methodForPerspective, methodKey, remember, structureStatus, type DocumentStructure, type Perspective, type Role } from './perspective';
import { analysisPerspective, canGroupByElement, groupedDescriptor, groupingKey } from './documentGrouping';

type PreparedPage = components['schemas']['PreparedMethodPage'];

export type MethodSelectionInput = {
  /** Model whose attention is inspected; each role supplies its own evidence source. */
  role: Role;
  generation?: Generation; retrieval?: Capture; retriever: RetrieverAttention; state: View;
  /** Saved-capture availability with verified document structure for the case. */
  prepared: {data?: PreparedPage; error?: string};
  update: UpdateView;
  /**
   * Generator method change (the workspace's `selectMethod`). `grouped`: the method is chosen as the input of
   * Document-aware element totals, so its output is the element summary rather than its own result.
   */
  onMethod: (method: Method, options?: {preserveAxes?: boolean; grouped?: boolean}) => void;
  settings: DisplaySettings; onView: (next: DisplaySettings) => void;
};

/** Method selection rules shared by the canvas method picker and the Details panel. */
export function useMethodSelection({role, generation, retrieval, retriever, prepared, state, update, onMethod, settings, onView}: MethodSelectionInput) {
  const isRetrieval = role === 'retrieval';
  const structure: DocumentStructure | undefined = prepared.data?.structure ?? undefined;
  const encodings = retrieval?.attention_encodings ?? [];
  const encoding = retriever.encoding;
  // Retriever methods reuse the shared method files; availability depends on saved encodings and,
  // for document-aware methods, on an encoding whose patches map to annotated elements.
  const all: MethodChoice[] = choicesForRole(role, {prepared, retriever, encodingCount: encodings.length, sourceIndex: state.retrieval_source_index ?? 0});
  const catalogueReady = isRetrieval ? !!retriever.catalogue.data : !!prepared.data;
  const catalogueError = isRetrieval ? retriever.catalogue.error : prepared.error;
  const generatorKey = `${state.attention_method_id ?? 'attention.row'}/${state.attention_variant_id ?? 'captured_weights_v1'}`;
  const retrieverMethodId = state.retrieval_attention_method ?? 'attention.row';
  const selected = isRetrieval ? all.find(p => p.method.method_id === retrieverMethodId && p.method.status !== 'planned') : all.find(p => methodKey(p.method) === generatorKey);
  const selectedKey = selected ? methodKey(selected.method) : isRetrieval ? `${retrieverMethodId}/retriever_encoding_v1` : generatorKey;
  const perspective = analysisPerspective(selected?.method, settings, role);
  const grouped = perspective === 'document_aware' && canGroupByElement(selected?.method);
  const elementAnalysis = all.find(p => p.method.method_id === 'attention.element_allocation' && p.method.status !== 'planned');
  const primary = grouped ? elementAnalysis : selected;
  const primaryKey = primary ? methodKey(primary.method) : selectedKey;
  const inputs = all.filter(p => p.method.perspective === 'general' && canGroupByElement(p.method));
  const input = selected?.method.method_id === 'attention.element_allocation'
    ? inputs.find(p => p.method.method_id === 'attention.row') : selected;
  // Attention controls belong to the workspace, independently of the displayed task.
  // Using a control activates its evidence so the canvas matches it.
  const updateAttention: UpdateView = useCallback(next => update(s => ({...(typeof next === 'function' ? next(s) : next), ...(isRetrieval ? {retrieval_attention: true} : {evidence_signal: 'attention' as const})})), [update, isRetrieval]);
  /** `target` is the perspective the method is chosen in; it decides element grouping. */
  const chooseMethod = (method: Method, preserveAxes = false, groupingInput = false, target: Perspective = perspective) => {
    // Raw attention is represented by the historic element method while grouped.
    const resolved = groupingInput && method.method_id === 'attention.row'
      ? elementAnalysis?.method ?? method : method;
    // An unsupported General calculation never inherits an unrelated element overlay.
    if (method.perspective === 'general' && !canGroupByElement(method))
      onView({...settings, [groupingKey(role)]: 'positions', attention_grouped_comparison_index: 0});
    else onView({...settings, [groupingKey(role)]: target === 'document_aware' && canGroupByElement(method) ? 'elements' : 'positions', attention_grouped_comparison_index: 0});
    return isRetrieval
    // Retriever encodings keep their source, layer and head on every method change.
      ? updateAttention(s => ({...s, perspective_methods: remember(remember(s.perspective_methods, 'retrieval', retriever.method), 'retrieval', method),
          retrieval_attention_method: resolved.method_id, retrieval_attention_parameters: {}, retrieval_comparison_index: 0,
          attention_direction: directionFor(resolved.method_id, s.attention_direction)}))
      : onMethod(resolved, {preserveAxes, grouped: groupingInput || (target === 'document_aware' && canGroupByElement(method))});
  };
  const chooseInput = (method: Method) => {
    chooseMethod(method, true, true, 'document_aware');
    // Remember the actual input, including Raw, separately from its grouping alias.
    updateAttention(s => ({...s, perspective_methods: remember(s.perspective_methods, role, method)}));
  };
  const choosePrimary = (method: Method) => {
    if (perspective === 'document_aware' && method.method_id === 'attention.element_allocation') {
      const remembered = inputs.find(p => methodKey(p.method) === state.perspective_methods?.[`${role}:general`] && p.available);
      if (remembered) {chooseInput(remembered.method); return;}
    }
    chooseMethod(method, perspective === 'document_aware');
  };
  // Changing perspective re-analyses the same saved row; it never runs a model.
  const choosePerspective = (target: Perspective) => {
    if (target === perspective || !catalogueReady) return;
    // Element grouping switches General ↔ Document-aware in place; Table-aware chooses its own method.
    if (target !== 'table_aware' && perspective !== 'table_aware' && selected && canGroupByElement(selected.method)) {
      onView({...settings, [groupingKey(role)]: target === 'document_aware' ? 'elements' : 'positions'});
      // The raw element method is the historic alias of raw attention. Other methods
      // keep their exact descriptor, parameters and result when changing grouping.
      const alias = selected.method.method_id === 'attention.row' && target === 'document_aware' ? 'attention.element_allocation'
        : selected.method.method_id === 'attention.element_allocation' && target === 'general' ? 'attention.row' : undefined;
      if (alias) {
        const next = all.find(p => p.method.method_id === alias && p.method.status !== 'planned');
        if (next) {
          if (isRetrieval) updateAttention(s => ({...s, retrieval_attention_method: next.method.method_id, attention_direction: 'row'}));
          else onMethod(next.method, {preserveAxes: true});
        }
      } else updateAttention(s => ({...s, attention_direction: 'row'}));
      return;
    }
    const next = methodForPerspective(state.perspective_methods, role, target, all.filter(p => p.method.status !== 'planned'));
    if (next) chooseMethod(next.method, true);
    onView({...settings, [groupingKey(role)]: 'positions'});
  };
  /** A choice from the method picker; a method from the other perspective switches perspective. */
  const choose = (method: Method, {asGroupingInput = false}: {asGroupingInput?: boolean} = {}) => {
    if (asGroupingInput) return chooseInput(method);
    if (method.perspective === perspective) return choosePrimary(method);
    const groupable = !!selected && canGroupByElement(selected.method);
    if (method.perspective === 'document_aware' && method.method_id === 'attention.element_allocation' && groupable) return choosePerspective('document_aware');
    if (method.perspective === 'general' && grouped && selected && methodKey(method) === methodKey(selected.method)) return choosePerspective('general');
    return chooseMethod(method, true, false, method.perspective);
  };
  const returnTo = encodeURIComponent(window.location.pathname + window.location.search);
  // Retriever variants are derived from catalogue entries, so their details link to the shared method.
  const detailUrl = (method: Method) => `/toolbox/${encodeURIComponent(method.method_id)}?${isRetrieval ? '' : `variant=${encodeURIComponent(method.variant_id)}&`}returnTo=${returnTo}`;
  const reason = (p: MethodChoice) => shortReason(p, {role, encodingCount: encodings.length});
  const mappedElements = isRetrieval ? !!encoding && encoding.pass_kind !== 'query' && hasMappedElements(retriever.page?.ownership ?? [])
    : hasMappedElements(generation?.positions.map(p => p.ownership) ?? []);
  /** Readiness of a method for this capture, with the document structure it needs. */
  const readiness = (choice: MethodChoice) => structureStatus(choice, {mappedElements, structure, queryEncoding: isRetrieval && encoding?.pass_kind === 'query'});
  const status = selected && structureStatus(grouped ? {...selected, method: groupedDescriptor(selected.method)} : selected,
    {mappedElements, structure, queryEncoding: isRetrieval && encoding?.pass_kind === 'query'});
  const properties = selected?.method.parameters_schema.properties as Record<string, unknown> | undefined;
  const rollout = isRolloutInterval(selected?.method.method_id);
  const partialRollout = selected?.method.method_id === 'attention.partial_rollout';
  const partialChoice = all.find(p => p.method.method_id === 'attention.partial_rollout' && p.available);
  const inputParameters = (isRetrieval ? state.retrieval_attention_parameters : state.attention_parameters) ?? {};
  const chooseLayerRange = partialChoice ? () => {
    const residual = inputParameters.residual_coefficient;
    chooseMethod(partialChoice.method, true, grouped);
    const next = {start_layer: 0, ...(residual !== undefined ? {residual_coefficient: residual} : {})};
    updateAttention(s => isRetrieval ? {...s, retrieval_attention_parameters: next} : {...s, attention_parameters: next});
  } : undefined;
  /** The rollout interval's end layer and, for partial rollout, its start. */
  const changeLayerRange = (start: number, end: number) => updateAttention(s => isRetrieval
    ? {...s, attention_layer: end, retrieval_attention_parameters: {...s.retrieval_attention_parameters, ...(partialRollout ? {start_layer: start} : {})}, retrieval_comparison_index: 0}
    : {...s, attention_layer: end, attention_parameters: {...s.attention_parameters, ...(partialRollout ? {start_layer: start} : {})}});
  return {role, isRetrieval, all, structure, encodings, encoding, catalogueReady, catalogueError, selected, selectedKey, perspective, grouped,
    elementAnalysis, primary, primaryKey, inputs, input, updateAttention, chooseMethod, chooseInput, choosePrimary, choosePerspective, choose,
    detailUrl, reason, mappedElements, readiness, status, properties, rollout, partialRollout, partialChoice, inputParameters, chooseLayerRange, changeLayerRange};
}

export type MethodSelection = ReturnType<typeof useMethodSelection>;
