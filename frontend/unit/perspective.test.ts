import { describe, it, expect } from 'vitest';
import type { Method } from '../src/api/client';
import type { View } from '../src/features/document/useAcquisition';
import { switchMethod } from '../src/features/document/methodState';
import { QUERY_ENCODING_GAP, NO_MAPPED_ELEMENTS, bucketLabel, contributions, elementMassMaximum, methodForPerspective, quantityLabel,
  remember, retrieverMappingGap, structureStatus } from '../src/features/document/perspective';

const descriptor = (method_id: string, variant_id: string, perspective: Method['perspective'], requires: string[] = [], status: Method['status'] = 'implemented') =>
  ({method_id, variant_id, perspective, requires, status, parameters_schema: {properties: {step: {}, layer: {}, head: {}}}}) as unknown as Method;
const raw = descriptor('attention.row', 'captured_weights_v1', 'general');
const mean = descriptor('attention.mean_heads', 'captured_prediction_v1', 'general');
const elements = descriptor('attention.element_allocation', 'ownership_mass_v1', 'document_aware', ['attention_weights', 'validated_ownership_map', 'mapped_elements']);
const support = descriptor('attention.designated_support', 'annotated_support_mass_v1', 'document_aware', ['mapped_elements', 'designated_support_annotations'], 'planned');
const order = descriptor('attention.reading_order', 'annotated_order_profile_v1', 'document_aware', ['mapped_elements', 'reading_order_annotations'], 'planned');
const ready = (method: Method) => ({method, available: method.status !== 'planned', reasons: method.status === 'planned' ? ['Planned'] : []});
const catalogue = [raw, mean, elements, support].map(ready);
const structure = {provenance: 'dataset' as const, source: 'benchmark dataset annotations', adapter_version: 'a', annotation_version: '0123456789abcdef',
  element_count: 4, boxed_element_count: 4, flagged_element_count: 0, reading_order_count: 0, element_types: ['table'], support_element_count: 1};

describe('perspective switching', () => {
  const state = {generation_capture_id: 'generation_a', attention_method_id: raw.method_id, attention_variant_id: raw.variant_id,
    attention_step: 2, attention_layer: 12, attention_head: 3, attention_parameters: {},
    attention_parameter_drafts: {'generation_a/attention.element_allocation/ownership_mass_v1': {step: 0, layer: 0, head: 0}}} as unknown as View;

  it('keeps the prediction, layer and head while restoring only method settings', () => {
    const next = switchMethod(state, elements, raw, {preserveAxes: true});
    expect([next.attention_step, next.attention_layer, next.attention_head]).toEqual([2, 12, 3]);
    expect(next.attention_method_id).toBe('attention.element_allocation');
    // Ordinary method switching still restores that method's saved draft.
    expect(switchMethod(state, elements, raw).attention_layer).toBe(0);
  });

  it('reopens the last method of each perspective, then its default', () => {
    expect(methodForPerspective({}, 'generation', 'document_aware', catalogue)?.method).toBe(elements);
    const memory = remember(remember({}, 'generation', mean), 'generation', elements);
    expect(memory).toEqual({'generation:general': 'attention.mean_heads/captured_prediction_v1', 'generation:document_aware': 'attention.element_allocation/ownership_mass_v1'});
    expect(methodForPerspective(memory, 'generation', 'general', catalogue)?.method).toBe(mean);
    // Retrieval memory is separate from generation memory.
    expect(methodForPerspective(memory, 'retrieval', 'general', catalogue)?.method).toBe(raw);
  });
});

describe('document structure availability', () => {
  it('reports ready structure with what is available', () => {
    expect(structureStatus(ready(elements), {mappedElements: true, structure})).toEqual({state: 'ready', text: 'Ready: element boxes and patch mapping available.'});
    expect(structureStatus(ready(raw), {mappedElements: true, structure})).toBeUndefined();
  });

  it('names missing annotations instead of reporting zero', () => {
    expect(structureStatus(ready(elements), {mappedElements: false, structure})).toEqual({state: 'unavailable', text: 'Unavailable: element boxes and patch mapping missing.'});
    expect(structureStatus(ready(order), {mappedElements: true, structure})?.text).toBe('Planned: not implemented yet · reading-order annotations missing · element boxes and patch mapping available.');
    expect(structureStatus(ready(support), {mappedElements: true, structure})?.state).toBe('planned');
    expect(structureStatus(ready(elements), {mappedElements: false, structure, queryEncoding: true})?.text).toBe(`Unavailable: ${QUERY_ENCODING_GAP}`);
  });

  it('applies the retriever mapping rule only to document-aware methods', () => {
    const mappedPage: Record<string, number>[] = [{element_a: .5, background: .5}, {non_page: 1}];
    expect(retrieverMappingGap(elements, {pass_kind: 'query'}, [])).toBe(QUERY_ENCODING_GAP);
    expect(retrieverMappingGap(elements, {pass_kind: 'document'}, [{background: 1}])).toBe(NO_MAPPED_ELEMENTS);
    expect(retrieverMappingGap(elements, {pass_kind: 'document'}, mappedPage)).toBeUndefined();
    expect(retrieverMappingGap(raw, {pass_kind: 'query'}, [])).toBeUndefined();
  });

  it('does not call future-only element mappings measured zero in a causal encoding',()=>{
    const ownership:Record<string,number>[]=[{non_page:1},{element_a:1}];
    expect(retrieverMappingGap(elements,{pass_kind:'document',causal:true},ownership,0)).toContain('causal source');
    expect(retrieverMappingGap(elements,{pass_kind:'document',causal:true},ownership,1)).toBeUndefined();
  });
});

describe('element quantities', () => {
  it('lists contributing patches whose ownership-weighted values sum to the element mass', () => {
    const positions: {index: number; ownership: Record<string, number>}[] = [{index: 0, ownership: {element_a: 1}}, {index: 1, ownership: {element_a: .25, background: .75}}, {index: 2, ownership: {background: 1}}, {index: 3, ownership: {element_a: 1}}];
    const parts = contributions(positions, [.2, .4, .3, .1], 3, 'element_a');
    expect(parts.map(p => p.index)).toEqual([0, 1]);
    expect(parts.reduce((s, p) => s + p.contribution, 0)).toBeCloseTo(.3, 12);
    expect(elementMassMaximum([{element_id: 'element_a', mass: .3}, {element_id: null, mass: .9}])).toBe(.3);
  });

  it('labels distinct measurements and residual buckets explicitly', () => {
    expect(quantityLabel('post_softmax_attention_weight')).toBe('Patch attention weight');
    expect(quantityLabel('element_attention_mass')).toBe('Element attention mass');
    expect(quantityLabel('element_type_mass_and_area_adjusted_attention')).toBe('Area-adjusted attention');
    expect(quantityLabel('head_envelope')).toBe('Head envelope value');
    expect(quantityLabel('whole_query_image_allocation')).toBe('Image-conditioned source aggregate');
    expect(quantityLabel('modality_block_attention')).toBe('Modality-block attention weight');
    expect(quantityLabel('partial_rollout_connectivity')).toBe('Rollout connectivity');
    expect(bucketLabel('non_image:query')).toBe('Non-image · query tokens');
    expect(bucketLabel('unreviewed_geometry')).toBe('Unreviewed element geometry');
  });
});
