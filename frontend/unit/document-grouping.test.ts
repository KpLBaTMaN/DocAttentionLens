import { describe, expect, it } from 'vitest';
import type { DisplaySettings, Method } from '../src/api/client';
import { analysisPerspective, canGroupByElement, elementQuantityLabel, selectedComparison } from '../src/features/document/documentGrouping';

const method = (method_id: string, perspective: Method['perspective'] = 'general', status: Method['status'] = 'implemented') =>
  ({method_id, perspective, status}) as Method;
const settings = {attention_grouping: 'elements', retrieval_grouping: 'positions'} as DisplaySettings;

describe('document grouping', () => {
  it('keeps native document methods compatible with older saved settings', () => {
    expect(analysisPerspective(method('attention.element_allocation', 'document_aware'), {} as DisplaySettings, 'generation')).toBe('document_aware');
    expect(analysisPerspective(method('attention.designated_support', 'document_aware'), {} as DisplaySettings, 'retrieval')).toBe('document_aware');
  });

  it('groups position and comparison results independently for each model role', () => {
    for (const id of ['attention.mean_heads', 'attention.whole_query', 'attention.partial_rollout', 'attention.head_by_head', 'attention.rollout_evolution']) {
      expect(analysisPerspective(method(id), settings, 'generation')).toBe('document_aware');
      expect(analysisPerspective(method(id), settings, 'retrieval')).toBe('general');
    }
  });

  it('does not substitute rows for statistics, graphs or planned calculations', () => {
    for (const id of ['attention.entropy', 'attention.concentration', 'attention.flow', 'attention.graph', 'heads.similarity', 'attention.spatial_distance']) {
      expect(canGroupByElement(method(id))).toBe(false);
      expect(analysisPerspective(method(id), settings, 'generation')).toBe('general');
    }
    expect(canGroupByElement(method('attention.rollout', 'general', 'planned'))).toBe(false);
  });

  it('preserves quantity distinctions in the element summary and legend', () => {
    expect(elementQuantityLabel('post_softmax_attention_weight')).toBe('Element attention mass');
    expect(elementQuantityLabel('head_maximum_envelope')).toBe('Head envelope allocation');
    expect(elementQuantityLabel('rollout_connectivity')).toBe('Rollout connectivity by element');
    expect(elementQuantityLabel('whole_query_image_allocation')).toBe('Image-conditioned source aggregate by element');
  });

  it('keeps the shown comparison aligned with the clamped selector', () => {
    expect(selectedComparison(['Layer 0', 'Layer 1'], 5)).toBe('Layer 1');
    expect(selectedComparison([], 0)).toBeUndefined();
  });
});
