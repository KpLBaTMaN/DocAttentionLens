import { describe, expect, it } from 'vitest';
import type { Method } from '../src/api/client';
import { NO_RETRIEVER_ATTENTION, choicesForRole, filterChoices, generatorChoices, groupByFamily, groupingInputChoices, methodControls,
  retrieverChoices, shortReason, type MethodChoice } from '../src/features/document/methodChoices';
import { QUERY_ENCODING_GAP } from '../src/features/document/perspective';

// Descriptors shaped like the live catalogue (GET /api/v1/methods), trimmed to the fields the picker reads.
function descriptor(method_id: string, family: string, {perspective = 'general', name = method_id, summary = '', evidence_kind = 'attention_allocation',
  requires = ['attention_weights'], status = 'implemented', parameters = ['step', 'layer', 'head'], variant_id = 'captured_prediction_v1'}: {
  perspective?: Method['perspective']; name?: string; summary?: string; evidence_kind?: string; requires?: string[]; status?: Method['status'];
  parameters?: string[]; variant_id?: string;
} = {}): Method {
  return {method_id, variant_id, family, perspective, name, summary, evidence_kind, requires, status, quantity: 'attention_weights',
    parameters_schema: {properties: Object.fromEntries(parameters.map(p => [p, {type: 'integer'}]))},
    output: {schema: 'method_analysis', units: 'dimensionless mass', signed: false, support: 'valid causal targets', normalization: 'original causal target domain'},
  } as unknown as Method;
}
const ready = (method: Method, reasons: string[] = []): MethodChoice => ({method, available: !reasons.length && method.status !== 'planned', reasons});

const raw = descriptor('attention.row', 'direct_attention', {name: 'Raw Attention', variant_id: 'captured_weights_v1', summary: 'One captured head row.'});
const mean = descriptor('attention.mean_heads', 'aggregation', {name: 'Mean-Head Attention', summary: 'Arithmetic mean across the selected query heads.', parameters: ['step', 'layer', 'heads']});
const rollout = descriptor('attention.rollout', 'connectivity', {name: 'Attention Rollout', evidence_kind: 'attention_summary', requires: ['full_square_attention_matrices'], parameters: ['step', 'layer', 'residual_coefficient']});
const entropy = descriptor('attention.entropy', 'attention_statistics', {name: 'Attention Entropy', evidence_kind: 'attention_summary', parameters: ['step', 'layer', 'head', 'log_base']});
const textToVision = descriptor('attention.text_to_vision', 'multimodal_attention', {name: 'Text → Vision', requires: ['captured_text_sources', 'image_mask']});
const elements = descriptor('attention.element_allocation', 'aggregation', {perspective: 'document_aware', name: 'Element Attention Mass', variant_id: 'ownership_mass_v1',
  requires: ['attention_weights', 'mapped_elements']});
const support = descriptor('attention.designated_support', 'document_structure', {perspective: 'document_aware', name: 'Designated Support Mass',
  requires: ['mapped_elements', 'designated_support_annotations']});
const order = descriptor('attention.reading_order', 'document_structure', {perspective: 'document_aware', name: 'Reading Order Profile',
  requires: ['mapped_elements', 'reading_order_annotations'], status: 'planned'});
const atlas = descriptor('attention.question_atlas', 'document_structure', {perspective: 'document_aware', name: 'Question Atlas',
  requires: ['element_type_annotations', 'corpus_sample_manifest']});
const maxsim = descriptor('retrieval.maxsim', 'similarity_and_retrieval', {name: 'MaxSim', evidence_kind: 'retrieval_scoring', parameters: []});
const profile = descriptor('heads.profile_similarity', 'head_and_representation', {name: 'Head Profiles', evidence_kind: 'representation'});

const generator = [ready(raw), ready(mean), ready(rollout, ['Needs full square attention matrices.']), ready(entropy), ready(textToVision),
  ready(elements), ready(support), ready(order), ready(atlas, ['Needs corpus sample.']), ready(maxsim), ready(profile)];

describe('generator choices', () => {
  it('lists attention methods and head representations, not similarity or attribution', () => {
    const ids = generatorChoices({data: {items: generator}}).map(p => p.method.method_id);
    expect(ids).toContain('heads.profile_similarity');
    expect(ids).not.toContain('retrieval.maxsim');
    expect(generatorChoices({})).toEqual([]);
  });
});

describe('retriever choices', () => {
  const catalogue = [raw, elements, support, textToVision];
  const encoding = {pass_kind: 'document', causal: false, attention_mask: [1, 1, 1, 1]};
  const positions = [{role: 'query_text'}, {role: 'visual_patch', bbox: [0, 0, 1, 1]}, {role: 'visual_patch', bbox: [1, 0, 2, 1]}, {role: 'prompt'}];
  const ownership: Record<string, number>[] = [{non_page: 1}, {element_1: 1}, {background: 1}, {non_page: 1}];
  const structure = {provenance: 'dataset', source: 's', adapter_version: 'a', annotation_version: 'v', element_count: 2, boxed_element_count: 2,
    flagged_element_count: 0, reading_order_count: 0, element_types: ['table'], support_element_count: 1} as never;

  it('needs saved retriever encodings before any method is ready', () => {
    const choices = retrieverChoices({catalogue, encodingCount: 0, sourceIndex: 0});
    expect(choices.every(p => !p.available && p.reasons[0] === NO_RETRIEVER_ATTENTION)).toBe(true);
    expect(shortReason(choices[0], {role: 'retrieval', encodingCount: 0})).toBe('Needs saved retriever attention');
  });

  it('applies the encoding, source and structure rules of the attention panel', () => {
    const choices = retrieverChoices({catalogue, encodingCount: 1, encoding, positions, ownership, sourceIndex: 1, structure});
    expect(choices.map(p => [p.method.method_id, p.available])).toEqual([
      ['attention.row', true], ['attention.element_allocation', true], ['attention.designated_support', true],
      ['attention.text_to_vision', false]]);
    expect(choices[3].reasons).toEqual(['Select a captured textual (non-image) source.']);
    const query = retrieverChoices({catalogue, encodingCount: 1, encoding: {...encoding, pass_kind: 'query'}, positions, ownership, sourceIndex: 0, structure});
    expect(query[1].reasons).toEqual([QUERY_ENCODING_GAP]);
  });

  it('waits for, or reports, the document structure that annotation methods need', () => {
    const loading = retrieverChoices({catalogue: [support], encodingCount: 1, encoding, positions, ownership, sourceIndex: 1});
    expect(loading[0].reasons).toEqual(['Loading document structure…']);
    const failed = retrieverChoices({catalogue: [support], encodingCount: 1, encoding, positions, ownership, sourceIndex: 1, structureError: 'HTTP 502'});
    expect(failed[0].reasons).toEqual(['Document structure could not be loaded: HTTP 502']);
    expect(shortReason(failed[0], {role: 'retrieval', encodingCount: 1})).toBe('Structure could not be loaded');
  });

  it('lists planned descriptors from the prepared catalogue in the retriever role', () => {
    const choices = choicesForRole('retrieval', {prepared: {data: {items: [ready(order), ready(raw)]}}, retriever: {catalogue: {data: [raw]}},
      encodingCount: 1, sourceIndex: 0});
    expect(choices.map(p => p.method.method_id)).toEqual(['attention.row', 'attention.reading_order']);
    expect(shortReason(choices[1], {role: 'retrieval'})).toBe('Planned');
  });
});

describe('short reasons', () => {
  it('names the missing input in a few words', () => {
    expect(shortReason(ready(raw), {role: 'generation'})).toBe('Ready from this capture');
    expect(shortReason(ready(rollout, ['x']), {role: 'generation'})).toBe('Needs a full-matrix capture');
    const visionToText = descriptor('attention.vision_to_text', 'cross_modal', {requires: ['attention_weights', 'complete_declared_support', 'captured_image_sources']});
    expect(shortReason(ready(visionToText, ['x']), {role: 'generation'})).toBe('Needs a full-matrix capture');
    expect(shortReason(ready(atlas, ['x']), {role: 'generation'})).toBe('Run from ⋯ › Experiments › Document studies');
    expect(shortReason(ready(support, ['No annotated element boxes are mapped.']), {role: 'generation'})).toBe('Needs mapped document structure');
  });
});

describe('filtering', () => {
  const ids = (query: string, perspective?: Method['perspective']) => filterChoices(generator, query, perspective).map(p => p.method.method_id);
  it('matches name, summary, family label, evidence kind and id within a perspective', () => {
    expect(ids('mean-head', 'general')).toEqual(['attention.mean_heads']);
    expect(ids('arithmetic', 'general')).toEqual(['attention.mean_heads']);
    expect(ids('modality blocks', 'general')).toEqual(['attention.text_to_vision']);
    expect(ids('attention summary', 'general')).toEqual(['attention.rollout', 'attention.entropy']);
    expect(ids('reading_order')).toEqual(['attention.reading_order']);
    expect(ids('MEAN heads')).toEqual(['attention.mean_heads']);
    expect(ids('support', 'general')).toEqual([]);
    expect(ids('', 'document_aware')).toEqual(['attention.element_allocation', 'attention.designated_support', 'attention.reading_order', 'attention.question_atlas']);
  });

  it('groups by family in catalogue order with planned methods last', () => {
    const groups = groupByFamily(filterChoices(generator, '', 'document_aware'));
    expect(groups.map(g => g.family)).toEqual(['aggregation', 'document_structure']);
    expect(groups[1].choices.map(p => p.method.method_id)).toEqual(['attention.designated_support', 'attention.question_atlas', 'attention.reading_order']);
    expect(groups[0].label).toBe('Attention · Aggregation');
  });

  it('offers groupable General methods as document-aware inputs', () => {
    expect(groupingInputChoices(generator).map(p => p.method.method_id)).toEqual(['attention.row', 'attention.mean_heads', 'attention.rollout', 'attention.text_to_vision']);
  });
});

describe('controls', () => {
  it('reads layer, head and settings from the parameter schema', () => {
    expect(methodControls(raw)).toEqual({axes: ['Layer', 'One head'], settings: []});
    expect(methodControls(mean)).toEqual({axes: ['Layer', 'Head selection'], settings: []});
    expect(methodControls(rollout)).toEqual({axes: ['Layer'], settings: ['residual coefficient']});
    expect(methodControls(entropy)).toEqual({axes: ['Layer', 'One head'], settings: []});
  });
});

describe('element attention profile', () => {
  it('is opened from Details › Element and never listed as a page method', () => {
    const profile = {...descriptor('attention.element_profile', 'document_structure', {perspective: 'document_aware', name: 'Element attention profile', parameters: ['step']}),
      displays: ['element_profile']} as Method;
    const listed = generatorChoices({data: {items: [ready(raw), ready(profile)]}});
    expect(listed.map(c => c.method.method_id)).toEqual(['attention.row']);
  });
});
