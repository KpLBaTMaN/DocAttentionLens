import type { Method } from '../../api/client';
import { PERSPECTIVES, perspectiveLabel } from '../document/perspective';
import { FAMILY_LABELS, familyOrder } from './categories';

/** Facets of the Methods catalogue. Each one filters on a single descriptor field. */
export const FACETS = [
  {id: 'perspective', label: 'Perspective'},
  {id: 'family', label: 'Family'},
  {id: 'status', label: 'Status'},
  {id: 'evidence_kind', label: 'Evidence kind'},
] as const;
export type FacetId = typeof FACETS[number]['id'];
export type FacetSelection = Record<FacetId, string[]>;
export const NO_FACETS: FacetSelection = {perspective: [], family: [], status: [], evidence_kind: []};

const STATUS_ORDER = ['implemented', 'validated', 'planned'];
const readable = (value: string) => value.replaceAll('_', ' ');
const sentence = (value: string) => readable(value).replace(/^./, c => c.toUpperCase());

export const familyLabel = (family: string) => FAMILY_LABELS[family] ?? sentence(family);
const familyRank = (family: string) => { const i = familyOrder.indexOf(family); return i < 0 ? familyOrder.length : i; };

export function facetValueLabel(facet: FacetId, value: string) {
  if (facet === 'perspective') return perspectiveLabel(value as Method['perspective']);
  if (facet === 'family') return familyLabel(value);
  return sentence(value);
}

export const matchesSearch = (m: Method, search: string) =>
  `${m.name} ${m.summary} ${m.family} ${m.evidence_kind} ${m.method_id} ${perspectiveLabel(m.perspective)}`.toLowerCase().includes(search.trim().toLowerCase());

/** True when the method passes every selected facet except `skip` (used for counts). */
export const matchesFacets = (m: Method, selection: FacetSelection, skip?: FacetId) =>
  FACETS.every(({id}) => id === skip || !selection[id].length || selection[id].includes(m[id]));

/** Values offered by one facet, with the number of methods each would show. */
export function facetOptions(items: Method[], facet: FacetId, search: string, selection: FacetSelection) {
  const values = new Set(items.map(m => m[facet]));
  if (facet === 'status') STATUS_ORDER.forEach(s => values.add(s));
  if (facet === 'perspective') PERSPECTIVES.forEach(p => values.add(p.id));
  const pool = items.filter(m => matchesSearch(m, search) && matchesFacets(m, selection, facet));
  const order = (value: string) => facet === 'family' ? familyRank(value) : facet === 'status' ? STATUS_ORDER.indexOf(value) : facet === 'perspective' ? PERSPECTIVES.findIndex(p => p.id === value) : 0;
  return [...values]
    .sort((a, b) => order(a) - order(b) || facetValueLabel(facet, a).localeCompare(facetValueLabel(facet, b)))
    .map(value => ({value, label: facetValueLabel(facet, value), count: pool.filter(m => m[facet] === value).length}));
}

/** Matching methods grouped by family in catalogue order; planned variants sort last in each family. */
export function groupMethods(items: Method[], search: string, selection: FacetSelection) {
  const shown = items.filter(m => matchesSearch(m, search) && matchesFacets(m, selection));
  const families = [...new Set(shown.map(m => m.family))].sort((a, b) => familyRank(a) - familyRank(b));
  return {
    count: shown.length,
    groups: families.map(family => ({
      family,
      label: familyLabel(family),
      methods: shown.filter(m => m.family === family).sort((a, b) => Number(a.status === 'planned') - Number(b.status === 'planned')),
    })),
  };
}
