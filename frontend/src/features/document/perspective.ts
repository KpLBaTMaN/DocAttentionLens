import type { Method } from '../../api/client';
import type { components } from '../../api/generated/schema';
import type { View } from './useAcquisition';

/** Canonical method perspectives; documentGrouping.ts adds the grouped view without changing the descriptor. */
export type Perspective = Method['perspective'];
export type Role = 'generation' | 'retrieval';
export type DocumentStructure = components['schemas']['DocumentStructure'];
type MemoryKey = keyof NonNullable<View['perspective_methods']>;
type Prepared = {method: Method; available: boolean; reasons: string[]};

export const PERSPECTIVES: {id: Perspective; label: string; description: string}[] = [
  {id: 'general', label: 'General', description: 'Attention over model positions: tokens, image patches, heads and layers.'},
  {id: 'document_aware', label: 'Document-aware', description: 'The same attention, aggregated over annotated document regions.'},
  {id: 'table_aware', label: 'Table-aware', description: 'The same attention, aggregated over the table: cells, rows, columns, headers and the lookup path.'},
];

export const perspectiveLabel = (perspective?: Perspective) => PERSPECTIVES.find(p => p.id === perspective)?.label ?? 'General';

export const methodKey = (method: Pick<Method, 'method_id' | 'variant_id'>) => `${method.method_id}/${method.variant_id}`;

/** Opening method of a perspective that has not been used in this investigation. */
export const DEFAULT_METHODS: Record<Role, Partial<Record<Perspective, string>>> = {
  generation: {general: 'attention.row/captured_weights_v1', document_aware: 'attention.element_allocation/ownership_mass_v1', table_aware: 'table.cell_allocation/table_cell_partition_v1'},
  retrieval: {general: 'attention.row/retriever_encoding_v1', document_aware: 'attention.element_allocation/retriever_encoding_v1', table_aware: 'table.cell_allocation/retriever_encoding_v1'},
};

/** Records a method as its perspective's last choice. */
export function remember(memory: View['perspective_methods'], role: Role, method?: Pick<Method, 'method_id' | 'variant_id' | 'perspective'>): View['perspective_methods'] {
  return method ? {...memory, [`${role}:${method.perspective}` as MemoryKey]: methodKey(method)} : {...memory};
}

/** The method a perspective reopens: its last choice, its default, then its first ready method. */
export function methodForPerspective<T extends Prepared>(memory: View['perspective_methods'], role: Role, perspective: Perspective, methods: T[]): T | undefined {
  const candidates = methods.filter(p => p.method.perspective === perspective);
  const find = (key?: string) => key ? candidates.find(p => methodKey(p.method) === key) : undefined;
  return find(memory?.[`${role}:${perspective}` as MemoryKey]) ?? find(DEFAULT_METHODS[role][perspective])
    ?? candidates.find(p => p.available) ?? candidates[0];
}

type Ownership = {[bucket: string]: number};
export const hasMappedElements = (ownership: (Ownership | undefined)[]) => ownership.some(o => !!o && Object.keys(o).some(b => b.startsWith('element_')));

export const QUERY_ENCODING_GAP = 'The query encoding has no mapped document positions. Document-aware methods require a document encoding with a valid element mapping.';
export const NO_MAPPED_ELEMENTS = 'No annotated element boxes are mapped to the captured patches of this page.';

/** Why a retriever encoding cannot support a document-aware method; the server applies the same rule. */
export const SPATIAL_SOURCE_GAP = 'Spatial distance needs an image-patch source in a document encoding; query and prompt tokens have no page position';

export function retrieverMappingGap(method: Pick<Method, 'perspective' | 'method_id'> | undefined, encoding?: {pass_kind: string;causal?:boolean}, ownership: (Ownership | undefined)[] = [],sourceIndex?:number) {
  if (method?.method_id === 'attention.spatial_distance' && encoding && (encoding.pass_kind === 'query' || (sourceIndex !== undefined && 'non_page' in (ownership[sourceIndex] ?? {}))))
    return SPATIAL_SOURCE_GAP;
  if (method?.perspective !== 'document_aware' || !encoding) return undefined;
  if (encoding.pass_kind === 'query') return QUERY_ENCODING_GAP;
  if(!hasMappedElements(ownership))return NO_MAPPED_ELEMENTS;
  if(encoding.causal&&sourceIndex!==undefined&&!hasMappedElements(ownership.slice(0,sourceIndex+1)))return 'No annotated element is visible from this causal source position; select a later document position';
  return undefined;
}

type Facts = {mappedElements: boolean; structure?: DocumentStructure; queryEncoding?: boolean};
const STRUCTURE: Record<string, {label: string; present: (facts: Facts) => boolean}> = {
  mapped_elements: {label: 'element boxes and patch mapping', present: f => f.mappedElements},
  designated_support_annotations: {label: 'designated support annotations', present: f => !!f.structure?.support_element_count},
  element_type_annotations: {label: 'element-type annotations', present: f => !!f.structure?.element_types.length},
  reading_order_annotations: {label: 'reading-order annotations', present: f => !!f.structure?.reading_order_count},
  document_relation_profiles: {label: 'document relation profiles', present: () => false},
  reviewed_group_relation: {label: 'reviewed element-group relations', present: () => false},
};

export type StructureStatus = {state: 'ready' | 'unavailable' | 'planned'; text: string};

/** Dependency message for a document-aware method. Missing structure is never reported as a measured zero. */
export function structureStatus(prepared: Prepared, facts: Facts): StructureStatus | undefined {
  const {method} = prepared;
  if (method.perspective !== 'document_aware') return undefined;
  const needs = method.requires.filter(r => r in STRUCTURE);
  const present = needs.filter(r => STRUCTURE[r].present(facts)).map(r => STRUCTURE[r].label);
  const missing = needs.filter(r => !STRUCTURE[r].present(facts)).map(r => STRUCTURE[r].label);
  const list = (items: string[]) => items.join(', ');
  if (method.status === 'planned') return {state: 'planned', text: `Planned: not implemented yet${missing.length ? ` · ${list(missing)} missing` : ''}${present.length ? ` · ${list(present)} available` : ''}.`};
  if (facts.queryEncoding) return {state: 'unavailable', text: `Unavailable: ${QUERY_ENCODING_GAP}`};
  if (missing.length) return {state: 'unavailable', text: `Unavailable: ${list(missing)} missing.`};
  if (!prepared.available) return {state: 'unavailable', text: `Unavailable: ${prepared.reasons.join(' ')}`};
  return {state: 'ready', text: `Ready: ${list(present)} available.`};
}

const PROVENANCE: Record<DocumentStructure['provenance'], string> = {dataset: 'Dataset annotations', imported: 'Imported', inferred: 'Model-inferred'};

export function provenanceLabel(structure?: DocumentStructure, error?: string) {
  if (!structure) return error ? 'Structure provenance unavailable: the structure request failed.' : 'Structure provenance loading…';
  return `Structure: ${PROVENANCE[structure.provenance]} · ${structure.source} · annotation ${structure.annotation_version.slice(0, 12)} · ${structure.boxed_element_count} of ${structure.element_count} elements boxed${structure.flagged_element_count ? ` · ${structure.flagged_element_count} flagged` : ''}`;
}

/** The legend names the measured quantity; these are different measurements. */
export function quantityLabel(quantity?: string | null) {
  if (!quantity) return 'Attention value';
  if (quantity.includes('area_adjusted')) return 'Area-adjusted attention';
  if (quantity === 'element_attention_mass') return 'Element attention mass';
  if (quantity.includes('rollout')&&quantity.includes('connectivity')) return 'Rollout connectivity';
  if (quantity === 'whole_query_image_allocation') return 'Image-conditioned source aggregate';
  if (quantity === 'modality_block_attention') return 'Modality-block attention weight';
  if (quantity.includes('envelope')) return 'Head envelope value';
  return 'Patch attention weight';
}

const RESIDUALS: Record<string, string> = {
  background: 'Page background · no annotated element',
  unreviewed_geometry: 'Unreviewed element geometry',
  non_page: 'Non-page positions · prompt and special tokens',
};

export function bucketLabel(bucket: string) {
  if (bucket.startsWith('non_image:')) return `Non-image · ${bucket.slice('non_image:'.length)} tokens`;
  return RESIDUALS[bucket] ?? bucket.replaceAll('_', ' ');
}

export type Contribution = {index: number; weight: number; fraction: number; contribution: number};

/** Patches owned by an element and the share of each patch's weight that its mass includes. */
export function contributions(positions: {index: number; ownership?: Ownership | null}[], values: number[], targetCount: number, elementId: string): Contribution[] {
  return positions.filter(p => p.index < targetCount && (p.ownership?.[elementId] ?? 0) > 0)
    .map(p => {const fraction = p.ownership![elementId]; const weight = values[p.index]; return {index: p.index, weight, fraction, contribution: weight * fraction};})
    .sort((a, b) => b.contribution - a.contribution || a.index - b.index);
}

/** Display maximum shared by every element on every page of one row. */
export function elementMassMaximum(allocations: {element_id?: string | null; mass: number}[]) {
  return Math.max(...allocations.filter(a => a.element_id).map(a => a.mass), 0);
}
