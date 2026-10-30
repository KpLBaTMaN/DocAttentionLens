import { describe, expect, it } from 'vitest';
import type { Method } from '../src/api/client';
import { FACETS, NO_FACETS, facetOptions, facetValueLabel, groupMethods, matchesFacets, matchesSearch, type FacetSelection } from '../src/features/toolbox/methodFacets';

// Descriptors shaped like the live catalogue (GET /api/v1/methods), trimmed to the fields the facets read.
function descriptor(method_id: string, family: string, {perspective = 'general', name = method_id, summary = '', evidence_kind = 'attention_allocation',
  status = 'implemented'}: {perspective?: Method['perspective']; name?: string; summary?: string; evidence_kind?: string; status?: Method['status']} = {}): Method {
  return {method_id, variant_id: 'v1', family, perspective, name, summary, evidence_kind, status, requires: [], quantity: 'attention_weights'} as unknown as Method;
}

const raw = descriptor('attention.row', 'direct_attention', {name: 'Raw Attention', summary: 'One captured head row.'});
const mean = descriptor('attention.mean_heads', 'aggregation', {name: 'Mean-Head Attention', summary: 'Arithmetic mean across heads.'});
const elements = descriptor('attention.element_allocation', 'aggregation', {perspective: 'document_aware', name: 'Element Attention Mass'});
const rollout = descriptor('attention.rollout', 'connectivity', {name: 'Attention Rollout', evidence_kind: 'attention_summary'});
const flow = descriptor('attention.flow', 'connectivity', {name: 'Attention Flow', evidence_kind: 'attention_summary', status: 'planned'});
const order = descriptor('attention.reading_order', 'document_structure', {perspective: 'document_aware', name: 'Reading Order Profile', status: 'planned'});
const support = descriptor('attention.designated_support', 'document_structure', {perspective: 'document_aware', name: 'Designated Support Mass'});
const custom = descriptor('custom.probe', 'custom_probe_family', {name: 'Custom Probe'});
// Deliberately out of catalogue order, with planned variants ahead of implemented ones.
const catalogue = [custom, order, flow, support, elements, rollout, mean, raw];

const select = (partial: Partial<FacetSelection>): FacetSelection => ({...NO_FACETS, ...partial});
const counts = (options: {value: string; count: number}[]) => Object.fromEntries(options.map(o => [o.value, o.count]));

describe('facet options', () => {
  it('counts each value against the search and the other facets', () => {
    expect(counts(facetOptions(catalogue, 'family', '', NO_FACETS))).toEqual({
      direct_attention: 1, aggregation: 2, connectivity: 2, document_structure: 2, custom_probe_family: 1,
    });
    // Search narrows every facet's counts.
    expect(counts(facetOptions(catalogue, 'family', '  ROLLOUT ', NO_FACETS))).toMatchObject({connectivity: 1, aggregation: 0, document_structure: 0});
    // Another facet narrows the counts; the facet's own selection does not, so its siblings stay choosable.
    const documentAware = select({perspective: ['document_aware']});
    expect(counts(facetOptions(catalogue, 'family', '', documentAware))).toEqual({
      direct_attention: 0, aggregation: 1, connectivity: 0, document_structure: 2, custom_probe_family: 0,
    });
    expect(counts(facetOptions(catalogue, 'perspective', '', documentAware))).toEqual({general: 5, document_aware: 3, table_aware: 0});
    const aggregation = select({family: ['aggregation']});
    expect(counts(facetOptions(catalogue, 'family', '', aggregation))).toMatchObject({aggregation: 2, connectivity: 2});
    expect(counts(facetOptions(catalogue, 'status', '', aggregation))).toEqual({implemented: 2, validated: 0, planned: 0});
  });

  it('keeps zero-count statuses and perspectives listed in their fixed order', () => {
    const generalOnly = [raw, mean, rollout];
    expect(facetOptions(generalOnly, 'status', '', NO_FACETS)).toEqual([
      {value: 'implemented', label: 'Implemented', count: 3},
      {value: 'validated', label: 'Validated', count: 0},
      {value: 'planned', label: 'Planned', count: 0},
    ]);
    expect(facetOptions(generalOnly, 'perspective', '', NO_FACETS)).toEqual([
      {value: 'general', label: 'General', count: 3},
      {value: 'document_aware', label: 'Document-aware', count: 0},
      {value: 'table_aware', label: 'Table-aware', count: 0},
    ]);
    // Families and evidence kinds come only from the catalogue: nothing invented.
    expect(facetOptions(generalOnly, 'family', 'no such method', NO_FACETS).map(o => [o.value, o.count])).toEqual([
      ['direct_attention', 0], ['aggregation', 0], ['connectivity', 0],
    ]);
    expect(facetOptions(generalOnly, 'evidence_kind', '', NO_FACETS).map(o => o.label)).toEqual(['Attention allocation', 'Attention summary']);
  });

  it('orders families by the catalogue vocabulary with unknown families last', () => {
    const options = facetOptions(catalogue, 'family', '', NO_FACETS);
    expect(options.map(o => o.value)).toEqual(['direct_attention', 'aggregation', 'connectivity', 'document_structure', 'custom_probe_family']);
    expect(options.at(-1)?.label).toBe('Custom probe family');
    expect(facetValueLabel('family', 'aggregation')).toBe('Attention · Aggregation');
  });
});

describe('grouped methods', () => {
  it('groups matches by family order and lists planned variants last in each family', () => {
    const {count, groups} = groupMethods(catalogue, '', NO_FACETS);
    expect(count).toBe(catalogue.length);
    expect(groups.map(g => g.family)).toEqual(['direct_attention', 'aggregation', 'connectivity', 'document_structure', 'custom_probe_family']);
    expect(groups.find(g => g.family === 'connectivity')?.methods.map(m => m.method_id)).toEqual(['attention.rollout', 'attention.flow']);
    expect(groups.find(g => g.family === 'document_structure')?.methods.map(m => m.method_id)).toEqual(['attention.designated_support', 'attention.reading_order']);
    expect(groups.find(g => g.family === 'connectivity')?.label).toBe('Attention-derived · Propagation and flow');
  });

  it('applies search and facets together, OR within a facet and AND across facets', () => {
    const selection = select({family: ['connectivity', 'document_structure'], status: ['planned']});
    const {count, groups} = groupMethods(catalogue, '', selection);
    expect(count).toBe(2);
    expect(groups.map(g => [g.family, g.methods.map(m => m.method_id)])).toEqual([
      ['connectivity', ['attention.flow']], ['document_structure', ['attention.reading_order']],
    ]);
    expect(groupMethods(catalogue, 'reading order', selection).groups.map(g => g.family)).toEqual(['document_structure']);
    expect(groupMethods(catalogue, 'nothing matches this', NO_FACETS)).toEqual({count: 0, groups: []});
    // The perspective label is searchable as well as names, summaries and identifiers.
    expect(matchesSearch(elements, 'document-aware')).toBe(true);
    expect(matchesSearch(raw, 'captured head')).toBe(true);
  });

  it('shows the whole catalogue again once filters are cleared', () => {
    const narrowed = select({perspective: ['document_aware'], status: ['planned'], evidence_kind: ['attention_allocation']});
    expect(groupMethods(catalogue, 'order', narrowed).count).toBe(1);
    // Clearing one facet widens only that facet.
    expect(groupMethods(catalogue, '', {...narrowed, status: []}).count).toBe(3);
    // Clearing everything restores every method, and the shared empty selection is untouched.
    expect(groupMethods(catalogue, '', NO_FACETS).count).toBe(catalogue.length);
    expect(groupMethods(catalogue, '   ', NO_FACETS).count).toBe(catalogue.length);
    expect(catalogue.every(m => matchesFacets(m, NO_FACETS))).toBe(true);
    expect(NO_FACETS).toEqual(Object.fromEntries(FACETS.map(f => [f.id, []])));
  });
});
