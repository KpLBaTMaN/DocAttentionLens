import type { DisplaySettings, Method } from '../../api/client';
import { quantityLabel, type Perspective, type Role } from './perspective';

// These canonical methods return ownership-allocated position rows or comparison rows.
// A scalar/table/graph is not a position distribution and must not be grouped as one.
const GROUPABLE = new Set([
  'attention.row', 'attention.element_allocation', 'attention.mean_heads', 'attention.weighted_heads',
  'attention.max_heads', 'attention.min_heads', 'attention.whole_query',
  'attention.text_to_text', 'attention.text_to_vision', 'attention.vision_to_text', 'attention.vision_to_vision',
  'attention.rollout', 'attention.partial_rollout', 'attention.head_by_head',
  'attention.across_layers', 'attention.rollout_evolution',
]);

export const canGroupByElement = (method?: Pick<Method, 'method_id' | 'status'>) =>
  !!method && method.status === 'implemented' && GROUPABLE.has(method.method_id);

export const groupingKey = (role: Role) => role === 'retrieval' ? 'retrieval_grouping' : 'attention_grouping';

/** User-facing names for the input calculation, distinct from element aggregation. */
export function attentionInputLabel(method: Pick<Method, 'method_id' | 'name'>) {
  const labels: Record<string, string> = {
    'attention.row': 'Raw attention', 'attention.element_allocation': 'Raw attention',
    'attention.mean_heads': 'Mean across heads', 'attention.weighted_heads': 'Weighted mean across heads',
    'attention.max_heads': 'Maximum across heads', 'attention.min_heads': 'Minimum across heads',
    'attention.whole_query': 'Whole-query attention',
  };
  return labels[method.method_id] ?? method.name;
}

export function analysisPerspective(method: Method | undefined, settings: DisplaySettings, role: Role): Perspective {
  if (method?.perspective === 'table_aware') return 'table_aware';
  return method?.perspective === 'document_aware' || settings[groupingKey(role)] === 'elements' && canGroupByElement(method)
    ? 'document_aware' : 'general';
}

/** A view prerequisite, never sent as a substituted calculation descriptor. */
export function groupedDescriptor(method: Method): Method {
  return {...method, perspective: 'document_aware', requires: [...new Set([...method.requires, 'mapped_elements'])]};
}

export function elementQuantityLabel(quantity: string) {
  if (quantity === 'post_softmax_attention_weight') return 'Element attention mass';
  if (quantity.includes('envelope')) return 'Head envelope allocation';
  return `${quantityLabel(quantity)} by element`;
}

/** Changing a selected head/layer population can shorten a returned comparison. */
export function selectedComparison<T>(comparisons: T[] | null | undefined, index = 0): T | undefined {
  return comparisons?.[Math.max(0, Math.min(index, comparisons.length - 1))];
}
