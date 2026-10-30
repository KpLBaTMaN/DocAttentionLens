/**
 * Known-answer attention evidence shared by component and browser rendering tests.
 *
 * One white 400 × 300 page is cut into a 4 × 3 grid of 100 px patches, numbered
 * row-major from the top left. Two annotated elements cover known patches. Every
 * expected picture follows from these numbers by hand: no app code, model or
 * database is involved.
 */
import type { components } from '../src/api/generated/schema';

type S = components['schemas'];
export type Box = [number, number, number, number];
export type Row = S['AttentionRow'];

export const CASE_ID = 'case_fixture';
export const DOCUMENT_ID = 'document_fixture';
export const PAGE_ID = 'page_fixture';
export const GENERATION_ID = 'generation_fixture';
export const RETRIEVAL_ID = 'capture_fixture';
const HASH = '0'.repeat(64);

export const PAGE = {width: 400, height: 300} as const;
export const GRID = {columns: 4, rows: 3, cell: 100} as const;
export const PATCH_COUNT = GRID.columns * GRID.rows;

/** Source-pixel box of patch `k`, row-major: patch 0 is top left, patch 11 bottom right. */
export function patchBox(k: number): Box {
  const column = k % GRID.columns, row = Math.floor(k / GRID.columns);
  return [column * GRID.cell, row * GRID.cell, (column + 1) * GRID.cell, (row + 1) * GRID.cell];
}

/** Centre of patch `k` in source pixels, away from every element and patch edge. */
export function patchCentre(k: number): {x: number; y: number} {
  const [x1, y1, x2, y2] = patchBox(k);
  return {x: (x1 + x2) / 2, y: (y1 + y2) / 2};
}

export const TABLE_ID = 'element_fixture_table';
export const TEXT_ID = 'element_fixture_text';
/** Each element covers whole patches, so ownership is exact and easy to check by eye. */
export const ELEMENTS = [
  {element_id: TABLE_ID, native_id: 1, element_type: 'table', box: [0, 0, 200, 100] as Box, patches: [0, 1], reading_order: 1},
  {element_id: TEXT_ID, native_id: 2, element_type: 'text', box: [200, 100, 400, 300] as Box, patches: [6, 7, 10, 11], reading_order: 2},
] as const;

export function patchOwnership(k: number): Record<string, number> {
  const owner = ELEMENTS.find(e => (e.patches as readonly number[]).includes(k));
  return owner ? {[owner.element_id]: 1} : {background: 1};
}

/** Image values: the hot patch gets HOT, every other patch a small, distinct ramp value. */
export const HOT = 0.24;
const COLD_STEP = 0.004;
export function patchValues(hot: number, ramp = true): number[] {
  return Array.from({length: PATCH_COUNT}, (_, k) => k === hot ? HOT : ramp ? COLD_STEP * (k + 1) : 0);
}

// ---------------------------------------------------------------- generator capture

/** Generator positions: prompt, the 12 page patches, the question, then two generated tokens. */
export const FIRST_PATCH = 3;
export const generatorPatchIndex = (k: number) => FIRST_PATCH + k;
const generatorTokens: [string, S['GenerationPosition']['role']][] = [
  ['<|im_start|>', 'special'], ['system', 'prompt'], ['<|vision_start|>', 'special'],
  ...Array.from({length: PATCH_COUNT}, (): [string, 'image'] => ['<|image_pad|>', 'image']),
  ['<|vision_end|>', 'special'], ['Which', 'query'], ['Ġpatch', 'query'], ['Ġis', 'query'], ['Ġhot', 'query'],
  ['Ċ', 'prompt'], ['Top', 'generated'], ['Ġleft', 'generated'],
];
export const generationPositions: S['GenerationPosition'][] = generatorTokens.map(([piece, role], index) => {
  const patch = index - FIRST_PATCH;
  const image = role === 'image';
  return {index, token_id: 1000 + index, piece, role, page_id: image ? PAGE_ID : null,
    bbox: image ? patchBox(patch) : null, ownership: image ? patchOwnership(patch) : {[`non_image:${role}`]: 1}};
});
const QUESTION = [16, 17, 18, 19];
const PREDICTION_SOURCE = 20;

export const generationCapture: S['GenerationCapture'] = {
  schema_version: '0.6.0', record_kind: 'generation_capture', capture_id: GENERATION_ID, case_id: CASE_ID,
  retrieval_capture_id: null, created_at: '2026-10-13T00:00:00+00:00', producer_version: 'fixture', dependencies: [CASE_ID, PAGE_ID],
  context: {construction_version: 'selected_page_images_v1', representation: 'image', retrieved_page_ids: [], selected_page_ids: [PAGE_ID],
    omitted_page_ids: [], query: 'Which patch is hot?', prompt: 'Fixture prompt', system_prompt: 'Fixture', input_source: 'supplied_pages',
    pages: [{page_id: PAGE_ID, page_number: 1, image_sha256: HASH, source_width: PAGE.width, source_height: PAGE.height,
      resized_width: 112, resized_height: 84, grid_thw: [1, GRID.rows * 2, GRID.columns * 2],
      position_indices: Array.from({length: PATCH_COUNT}, (_, k) => generatorPatchIndex(k)), pixel_array: 'pixels_0',
      preview_url: `/api/v1/generation/captures/${GENERATION_ID}/pages/${PAGE_ID}/input-image`}],
    input_token_ids: generatorTokens.slice(0, PREDICTION_SOURCE + 1).map((_, i) => 1000 + i),
    input_attention_mask: generatorTokens.slice(0, PREDICTION_SOURCE + 1).map(() => 1),
    truncation: 'none; over-budget input rejected', evaluation_labels_consumed: false, ocr_consumed: false, image_token_limit: 512},
  positions: generationPositions, output_token_ids: [1021, 1022], answer: 'Top left', stop_reason: 'eos',
  steps: [
    {step: 0, pass_kind: 'prefill', source_index: 20, consumed_token_id: 1020, consumed_piece: 'Ċ', predicted_index: 21, predicted_token_id: 1021, predicted_piece: 'Top', target_count: 21, array: 'step_0'},
    {step: 1, pass_kind: 'decode', source_index: 21, consumed_token_id: 1021, consumed_piece: 'Top', predicted_index: 22, predicted_token_id: 1022, predicted_piece: 'Ġleft', target_count: 22, array: 'step_1'},
  ],
  source_rows: QUESTION.map(i => ({source_index: i, token_id: 1000 + i, piece: generatorTokens[i][0], predicted_index: i + 1,
    predicted_token_id: 1001 + i, predicted_piece: generatorTokens[i + 1][0], target_count: i + 1, array: `question_${i}`})),
  layer_count: 2, query_head_count: 4, kv_head_count: 2,
  provenance: {model_id: 'fixture/generator', adapter_version: 'fixture'}, coverage: {}, validation: {},
};

/**
 * Incoming raw column of generator position `target`: the weight each captured question and
 * answer row gives it. Values grow with the source position, so the last answer row is strongest.
 */
export function generatorColumn(target: number, {layer = 0, head = 0} = {}): S['ReceivedAttention'] {
  const sources: S['ReceivedSource'][] = [
    ...QUESTION.map((i, k) => ({source_index: i, kind: 'question' as const, label: generatorTokens[i][0], value: 0.01 * (k + 1)})),
    ...generationCapture.steps.map((s, k) => ({source_index: s.source_index, kind: 'prediction' as const, label: `step ${s.step}`, value: 0.05 + 0.03 * k})),
  ];
  return {capture_id: GENERATION_ID, target_index: target, layer, head, sources, maximum: Math.max(...sources.map(s => s.value)),
    quantity: 'attention_received', units: 'post-softmax weight within each source row', normalization: 'fixture column'};
}

// ---------------------------------------------------------------- retriever capture

/** Retriever page encoding: the 12 patches first (ColPali order), then the visual prompt. */
const retrieverPage: S['Position'][] = [
  ...Array.from({length: PATCH_COUNT}, (_, k) => ({index: k, token_id: 257152, piece: '<image>', role: 'visual_patch', eligible: true, bbox: patchBox(k), offsets: null})),
  ...[['<bos>', 'bos'], ['Describe', 'visual_prompt'], ['▁image', 'visual_prompt'], ['\n', 'newline']].map(([piece, role], i) =>
    ({index: PATCH_COUNT + i, token_id: 2 + i, piece, role, eligible: true, bbox: null, offsets: null})),
];
export const retrieverOwnership = retrieverPage.map(p => p.bbox ? patchOwnership(p.index) : {non_page: 1});
export const queryPositions: S['Position'][] = [['<bos>', 'prefix_or_special'], ['Query', 'prefix_or_special'], [':', 'prefix_or_special'],
  ['▁hot', 'query_text'], ['▁patch', 'query_text'], ['<pad>', 'augmentation']].map(([piece, role], index) =>
  ({index, token_id: 10 + index, piece, role, eligible: true, bbox: null, offsets: [index, index + 1]}));
/** Word "hot" (query position 3) matches patch 5 best; its signed similarities are a known ramp. */
export const SIMILARITY_WINNER = 5;
export function similarityValues(queryIndex: number): (number | null)[] {
  return retrieverPage.map(p => p.bbox ? (p.index === SIMILARITY_WINNER ? 0.9 : (p.index % 3 - 1) * 0.1 * (queryIndex + 1)) : p.role === 'bos' ? 0.2 : -0.05);
}

const encoding = (encoding_id: 'document' | 'query', count: number): S['EncodingAttention'] => ({encoding_id, pass_kind: encoding_id,
  component: 'retriever_language', page_id: encoding_id === 'document' ? PAGE_ID : null, layer_count: 2, query_head_count: 4,
  position_count: count, attention_mask: Array(count).fill(1), layer_arrays: [`attn_${encoding_id}_l0`, `attn_${encoding_id}_l1`],
  normalization: 'full encoding prefix; active keys only', causal: false, positions: [], ownership: []});

export const retrievalCapture: S['RetrievalCapture'] = {
  schema_version: '0.11.0', record_kind: 'retrieval_capture', capture_id: RETRIEVAL_ID, case_id: CASE_ID, created_at: '2026-10-13T00:00:00+00:00',
  producer_version: 'fixture', dependencies: [CASE_ID, PAGE_ID], dataset_version: 'fixture', annotation_version: 'fixture',
  query: 'Which patch is hot?', query_positions: queryPositions, query_array: 'query',
  pages: [{page_id: PAGE_ID, image_sha256: HASH, width: PAGE.width, height: PAGE.height, page_number: 1, score: 1.5, reference_score: 1.5,
    reconstruction_error: 0, positions: retrieverPage, vector_array: 'page_0', ownership: retrieverOwnership,
    contributions: {background: 1, non_page: 0.5}, winner_indices: queryPositions.map(q => q.index === 3 ? SIMILARITY_WINNER : PATCH_COUNT),
    query_maxima: queryPositions.map(q => q.index === 3 ? 0.9 : 0.2)}],
  arrays: [], provenance: {model_id: 'fixture/retriever', model_revision: 'fixture'}, scorer: {formula: 'sum_i max_j dot(q_i,d_j)'}, geometry: {}, validation: {},
  attention_encodings: [encoding('document', retrieverPage.length), encoding('query', queryPositions.length)],
};
export const retrieverPositions = retrieverPage;

// ---------------------------------------------------------------- rows and allocations

type RowPosition = {index: number; bbox?: number[] | null; page_id?: string | null; ownership: Record<string, number>};

/**
 * A complete attention row over `targetCount` positions. Page patches carry `image`;
 * the remaining mass is spread over non-image targets so the row sums to one.
 * Allocations follow ownership exactly, as the server reports them.
 */
export function attentionRow({positions, patchIndex, image, targetCount, capture_id, source_index, step = 0, layer = 0, head = 0,
  quantity = 'post_softmax_attention_weight', pageOf = () => PAGE_ID}: {
  positions: RowPosition[]; patchIndex: (k: number) => number; image: number[]; targetCount: number; capture_id: string;
  source_index: number | null; step?: number; layer?: number; head?: number; quantity?: Row['quantity']; pageOf?: (p: RowPosition) => string | null;
}): Row {
  const values = Array<number>(targetCount).fill(0);
  image.forEach((v, k) => { if (patchIndex(k) < targetCount) values[patchIndex(k)] = v; });
  const imageMass = values.reduce((sum, v) => sum + v, 0);
  const others = positions.filter(p => p.index < targetCount && !p.bbox);
  others.forEach(p => { values[p.index] = (1 - imageMass) / others.length; });
  const buckets = new Map<string, {bucket: string; page_id: string | null; element_id: string | null; mass: number}>();
  positions.filter(p => p.index < targetCount).forEach(p => Object.entries(p.ownership).forEach(([bucket, share]) => {
    const entry = buckets.get(bucket) ?? {bucket, page_id: p.bbox ? pageOf(p) : null, element_id: bucket.startsWith('element_') ? bucket : null, mass: 0};
    entry.mass += share * values[p.index];
    buckets.set(bucket, entry);
  }));
  return {capture_id, step, layer, head, source_index, predicted_index: null, target_count: targetCount, values, quantity,
    normalization: 'fixture: original domain', total_mass: values.reduce((sum, v) => sum + v, 0), image_mass: imageMass,
    allocations: [...buckets.values()], allocation_error: 0, display_max: Math.max(...values)};
}

/** Generator row for a prediction step or question source. */
export function generatorRow(image: number[], {step = 0, layer = 0, head = 0, source_index, quantity}: {step?: number; layer?: number; head?: number; source_index?: number | null; quantity?: Row['quantity']} = {}): Row {
  const question = generationCapture.source_rows!.find(r => r.source_index === source_index);
  const prediction = generationCapture.steps.find(s => s.step === step)!;
  const row = attentionRow({positions: generationPositions, patchIndex: generatorPatchIndex, image, capture_id: GENERATION_ID,
    targetCount: question?.target_count ?? prediction.target_count, source_index: question?.source_index ?? prediction.source_index, step, layer, head, quantity});
  return {...row, predicted_index: question?.predicted_index ?? prediction.predicted_index};
}

/** Retriever document-encoding row: every encoding position is a target. */
export function retrieverRow(image: number[], {source_index = 0, layer = 0, head = 0, quantity}: {source_index?: number; layer?: number; head?: number; quantity?: Row['quantity']} = {}): Row {
  return attentionRow({positions: retrieverPage.map((p, i) => ({...p, ownership: retrieverOwnership[i]})), patchIndex: k => k, image,
    capture_id: RETRIEVAL_ID, targetCount: retrieverPage.length, source_index, layer, head, quantity});
}

/** Element attention mass on the fixture page, by element id: the ownership-weighted patch sum. */
export function elementMasses(row: Row): Record<string, number> {
  return Object.fromEntries(row.allocations.filter(a => a.element_id && a.page_id === PAGE_ID).map(a => [a.element_id!, a.mass]));
}

// ---------------------------------------------------------------- document records

const ref = {dataset_id: 'dataset_fixture', dataset_version: 'fixture', domain: 'fixture', document_id: 1, page_number: null, element_id: null, question_id: null};

export const caseRecord: S['CaseRecord'] = {key: CASE_ID, schema_version: '0.2.0', case_id: CASE_ID, dataset_id: 'dataset_fixture', dataset_version: 'fixture',
  document_id: DOCUMENT_ID, question_id: 'question_fixture', query_variant: 'collection_scoped', task_scope: 'collection', query: 'Which patch is hot?',
  page_ids: [PAGE_ID], page_image_sha256: {[PAGE_ID]: HASH}, annotation_version: 'fixture', support_page_ids: [PAGE_ID],
  support_element_ids: [TABLE_ID], reference_answer: 'Top left', label_exposure: 'evaluation_only', producer_version: 'fixture', record_kind: 'case'};

export const documentRecord: S['DocumentRecord'] = {key: DOCUMENT_ID, document_id: DOCUMENT_ID, ref, native_id: 1, filename: 'known-answer.pdf',
  total_pages: 1, page_ids: [PAGE_ID], question_count: 1, element_count: ELEMENTS.length, annotation_version: 'fixture',
  source_pdf_sha256: HASH, ocr_model: 'none', ocr_status: 'complete'};

export const pageRecord: S['PageRecord'] = {key: PAGE_ID, page_id: PAGE_ID, document_id: DOCUMENT_ID, ref: {...ref, page_number: 1}, page_number: 1,
  width: PAGE.width, height: PAGE.height, image_sha256: HASH, image_url: `/api/v1/pages/${PAGE_ID}/image`, orientation: 'landscape', spread_status: 'not_annotated'};

export const elementRecords: S['ElementRecord'][] = ELEMENTS.map(e => {
  const [x1, y1, x2, y2] = e.box;
  return {key: e.element_id, element_id: e.element_id, page_id: PAGE_ID, document_id: DOCUMENT_ID, ref: {...ref, page_number: 1, element_id: e.native_id},
    native_id: e.native_id, page_number: 1, element_type: e.element_type, element_category: e.element_type.toUpperCase(), bbox: {x1, y1, x2, y2},
    level: 1, reading_order: e.reading_order, ocr: {status: 'empty', text: ''}, geometry_flags: [], display_bbox: {x1, y1, x2, y2}, display_geometry_policy: 'raw'};
});

export const documentStructure: S['DocumentStructure'] = {provenance: 'dataset', source: 'fixture annotations', adapter_version: 'fixture',
  annotation_version: 'fixture', element_count: ELEMENTS.length, boxed_element_count: ELEMENTS.length, flagged_element_count: 0,
  reading_order_count: ELEMENTS.length, element_types: ELEMENTS.map(e => e.element_type), support_element_count: 1};

// ---------------------------------------------------------------- expected display (independent of app code)

export const OVERLAY_RGB = [0xd9, 0x9a, 0x36] as const;
export const SIMILARITY_POSITIVE = '#b080d1';
export const SIMILARITY_NEGATIVE = '#247ed5';
export const SOURCE_OUTLINE = '#39a9e0';

/** Documented inferno stops, copied independently of attentionDisplay.ts. */
const STOPS: [number, number[]][] = [[0, [0, 0, 4]], [.13, [31, 12, 72]], [.25, [85, 15, 109]], [.38, [136, 34, 106]], [.5, [186, 54, 85]],
  [.63, [227, 89, 51]], [.75, [249, 140, 10]], [.88, [249, 201, 50]], [1, [252, 255, 164]]];
export function inferno(t: number): number[] {
  const x = Number.isFinite(t) ? Math.min(1, Math.max(0, t)) : 0;
  const i = STOPS.findIndex(([s]) => x <= s);
  if (i <= 0) return STOPS[0][1];
  const [t0, c0] = STOPS[i - 1], [t1, c1] = STOPS[i];
  return c0.map((c, k) => Math.round(c + (x - t0) / (t1 - t0) * (c1[k] - c)));
}

/** Overlay fill opacity: linear in weight up to the chosen opacity at the display maximum. */
export const overlayOpacity = (value: number, maximum: number, opacity = .65) => maximum > 0 ? Math.max(0, Math.min(opacity, value / maximum * opacity)) : 0;

/** Colour of a fill with `alpha` composited over a white page. */
export const overWhite = (rgb: readonly number[], alpha: number) => rgb.map(c => 255 + (c - 255) * alpha);
