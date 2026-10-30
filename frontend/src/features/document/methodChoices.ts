import type { Method } from '../../api/client';
import type { components } from '../../api/generated/schema';
import { FAMILY_LABELS, familyOrder } from '../toolbox/categories';
import { canGroupByElement } from './documentGrouping';
import { perspectiveLabel, retrieverMappingGap, type DocumentStructure, type Perspective, type Role } from './perspective';
import { encodingMethodGap } from './useRetrieverAttention';

/** Method picker rules as pure functions: which methods each role lists and whether the capture supports them. */
export type MethodChoice = {method: Method; available: boolean; reasons: string[]};
type PreparedPage = components['schemas']['PreparedMethodPage'];
type Ownership = {[bucket: string]: number};

/** Element attention profile opens from Details › Element for the selected element, never as the page method. */
export const isElementProfile = (method: Pick<Method, 'displays'>) => !!method.displays?.includes('element_profile');

/** Only attention descriptors (and head representations) are chosen on the page canvas. */
export const isAttentionMethod = (method: Pick<Method, 'evidence_kind' | 'family' | 'displays'>) =>
  !isElementProfile(method) && (method.evidence_kind.startsWith('attention_') || method.family === 'head_and_representation');

export const NO_RETRIEVER_ATTENTION = 'This saved capture has no internal retriever attention. Acquire a new attention capture to explore heads and layers.';

export const familyLabel = (family: string) => FAMILY_LABELS[family] ?? family.replaceAll('_', ' ');

/** Generator choices: the prepared catalogue for the open capture, attention methods only. */
export function generatorChoices<T extends MethodChoice>(prepared: {data?: {items: T[]}}): T[] {
  return (prepared.data?.items ?? []).filter(p => isAttentionMethod(p.method));
}

export type RetrieverChoiceInput = {
  /** Retriever attention catalogue (`retriever.catalogue.data`). */
  catalogue?: Method[];
  /** Number of saved retriever encodings in the capture (`retrieval.attention_encodings.length`). */
  encodingCount: number;
  /** The selected encoding (`retriever.encoding`). */
  encoding?: {pass_kind: string; causal?: boolean; attention_mask: number[]};
  /** Patch-to-element ownership of the encoding's page (`retriever.page?.ownership`). */
  ownership?: (Ownership | undefined)[];
  /** Encoding positions (`retriever.positions`). */
  positions?: {role: string; bbox?: number[] | null}[];
  /** Selected source position (`state.retrieval_source_index ?? 0`). */
  sourceIndex: number;
  /** Verified structure for the case (`prepared.data?.structure`). */
  structure?: DocumentStructure | null;
  /** Error from the prepared request that supplies the structure (`prepared.error`). */
  structureError?: string;
  /** Planned descriptors are model-independent and listed in both roles (`prepared.data.items`). */
  prepared?: MethodChoice[];
};

/** Retriever choices: shared method files gated by saved encodings, source and document structure. */
export function retrieverChoices(input: RetrieverChoiceInput): MethodChoice[] {
  const {catalogue = [], encodingCount, encoding, ownership, positions, sourceIndex, structure, structureError} = input;
  const choices = catalogue.map(method => {
    const gap = encodingCount ? retrieverMappingGap(method, encoding, ownership, sourceIndex)
      ?? encodingMethodGap(method, encoding, positions, sourceIndex) : undefined;
    const reasons = !encodingCount ? [NO_RETRIEVER_ATTENTION] : gap ? [gap] : [];
    const needsStructure = method.requires.some(r => r === 'designated_support_annotations' || r === 'element_type_annotations' || r === 'reading_order_annotations');
    if (needsStructure && !structure) reasons.push(structureError ? `Document structure could not be loaded: ${structureError}` : 'Loading document structure…');
    else {
      if (method.requires.includes('designated_support_annotations') && !structure?.support_element_count) reasons.push('Needs designated supporting-element annotations.');
      if (method.requires.includes('element_type_annotations') && !structure?.element_types.length) reasons.push('Needs element-type annotations.');
      if (method.requires.includes('reading_order_annotations') && !structure?.reading_order_count) reasons.push('Needs reading-order annotations.');
    }
    return {method, available: !reasons.length, reasons};
  });
  const planned = (input.prepared ?? []).filter(p => p.method.status === 'planned');
  return [...choices, ...planned].filter(p => isAttentionMethod(p.method));
}

/** Convenience wrapper with the objects the attention panel holds. */
export function choicesForRole(role: Role, inputs: {
  prepared: {data?: PreparedPage; error?: string};
  retriever?: {catalogue: {data?: Method[]}; encoding?: RetrieverChoiceInput['encoding']; page?: {ownership?: (Ownership | undefined)[] | null}; positions?: RetrieverChoiceInput['positions']};
  encodingCount?: number; sourceIndex?: number;
}): MethodChoice[] {
  const {prepared, retriever} = inputs;
  if (role === 'generation') return generatorChoices(prepared);
  return retrieverChoices({catalogue: retriever?.catalogue.data, encodingCount: inputs.encodingCount ?? 0, encoding: retriever?.encoding,
    ownership: retriever?.page?.ownership ?? undefined, positions: retriever?.positions, sourceIndex: inputs.sourceIndex ?? 0,
    structure: prepared.data?.structure, structureError: prepared.error, prepared: prepared.data?.items});
}

/** A short visible reason; the full reasons stay available to screen readers and the preview. */
export function shortReason(choice: MethodChoice, {role, encodingCount = 0}: {role: Role; encodingCount?: number}) {
  const {method, available, reasons} = choice;
  const isRetrieval = role === 'retrieval';
  return method.status === 'planned' ? 'Planned' : available ? 'Ready from this capture'
    : method.requires.some(r => r === 'corpus_sample_manifest' || r === 'paired_capture') ? 'Run from ⋯ › Experiments › Document studies'
    : method.requires.includes('retrieval_evidence') && reasons.some(r => /retrieval/i.test(r)) ? 'Needs a linked retrieval capture'
    : reasons.some(r => r.startsWith('Document structure could not be loaded')) ? 'Structure could not be loaded'
    // Generator captures carry these only when saved with "Save full attention matrices"; the backend's reason says so in full.
    : !isRetrieval && method.requires.some(r => r === 'full_square_attention_matrices' || r === 'captured_image_sources') ? 'Needs a full-matrix capture'
    : isRetrieval ? (encodingCount ? 'Not available for this encoding or source' : 'Needs saved retriever attention')
    : method.perspective === 'document_aware' && reasons.some(r => /structure|element|annotat|mapped/i.test(r)) ? 'Needs mapped document structure'
    : method.perspective === 'table_aware' && reasons.some(r => /table/i.test(r)) ? 'Needs a table grounding question'
    : 'Not available for this capture';
}

const familyRank = (family: string) => {const i = familyOrder.indexOf(family); return i < 0 ? familyOrder.length : i;};

/** Families in catalogue order (`familyOrder`), each with its choices; planned methods last. */
export function groupByFamily<T extends MethodChoice>(choices: T[]): {family: string; label: string; choices: T[]}[] {
  const families = [...new Set(choices.map(p => p.method.family))].sort((a, b) => familyRank(a) - familyRank(b));
  return families.map(family => ({family, label: familyLabel(family),
    choices: choices.filter(p => p.method.family === family).sort((a, b) => Number(a.method.status === 'planned') - Number(b.method.status === 'planned'))}));
}

/** Words a researcher might type for a method: name, summary, family, evidence kind, id and perspective. */
export function searchText(method: Method) {
  return [method.name, method.summary, familyLabel(method.family), method.family, method.evidence_kind, method.evidence_kind.replaceAll('_', ' '),
    method.method_id, method.variant_id, perspectiveLabel(method.perspective)].join(' ').toLowerCase();
}

/** Choices in a perspective whose text contains every word of the query (any order). */
export function filterChoices<T extends MethodChoice>(choices: T[], query: string, perspective?: Perspective): T[] {
  const words = query.toLowerCase().split(/\s+/).filter(Boolean);
  return choices.filter(p => (!perspective || p.method.perspective === perspective) && words.every(w => searchText(p.method).includes(w)));
}

/** General methods whose result can be grouped into document elements (the document-aware "input"). */
export const groupingInputChoices = <T extends MethodChoice>(choices: T[]) =>
  choices.filter(p => p.method.perspective === 'general' && canGroupByElement(p.method));

const AXES: Record<string, string> = {step: 'Prediction step', layer: 'Layer', head: 'One head', heads: 'Head selection', layers: 'Layer selection'};

/** Controls a method uses, read from its parameter schema: axes first, then its own settings. */
export function methodControls(method: Pick<Method, 'parameters_schema'>) {
  const properties = Object.keys((method.parameters_schema.properties as Record<string, unknown> | undefined) ?? {});
  const axes = Object.keys(AXES).filter(k => properties.includes(k) && k !== 'step').map(k => AXES[k]);
  const settings = properties.filter(k => !(k in AXES) && k !== 'log_base').map(k => k.replaceAll('_', ' '));
  return {axes, settings};
}
