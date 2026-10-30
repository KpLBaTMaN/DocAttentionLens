/**
 * A complete in-browser stand-in for the DocAttentionLens API, serving the known-answer fixture.
 * Every /api and /health request is answered here, so rendering tests need no database,
 * GPU or saved capture. An unexpected endpoint is recorded and answered with 404; tests
 * assert that none occurred, so a new frontend call cannot silently reach a real server.
 */
import { readFileSync } from 'node:fs';
import { deflateSync } from 'node:zlib';
import type { Page, Route } from '@playwright/test';
import type { components } from '../../src/api/generated/schema';
import * as F from '../../fixtures/attention';

type S = components['schemas'];
type Method = S['MethodDescriptor'];
/** A method result; normalization is filled in when it is served. */
type Calculation = Omit<S['Calculation'], 'normalization'>;

/** The Python-owned registry, exactly as /api/v1/methods/availability serves its descriptors. */
export const catalogue: Method[] = JSON.parse(readFileSync(new URL('../../../library/src/docattentionlens/methods/catalogue.json', import.meta.url), 'utf8')).methods;

/** Capabilities of the fixture generator capture and its annotated case (see backend availability). */
const GENERATOR_CAPABILITIES = new Set(['attention_weights', 'source_target_position_axes', 'masks', 'normalization_domain', 'coverage',
  'generator_prediction_row_capture', 'complete_declared_support', 'validated_ownership_map', 'mapped_elements', 'image_mask',
  'question_source_rows', 'captured_text_sources', 'multiple_source_rows', 'designated_support_annotations', 'element_type_annotations',
  'reading_order_annotations']);
/** Added when the generator capture keeps full attention matrices (methods/registry.prediction_capabilities). */
const FULL_ATTENTION_CAPABILITIES = ['full_square_attention_matrices', 'aligned_layer_axes', 'residual_policy', 'eligible_mask', 'captured_image_sources'];

/**
 * The fixture generator capture saved with "Save full attention matrices" (generation schema 0.7.0): every
 * position below the square limit has a complete causal row. Six layers give the rollout interval room.
 */
export function fullAttentionCapture(layers = 6): S['GenerationCapture'] {
  const n = F.generationCapture.positions.filter(p => p.role !== 'generated').length;
  const square_limit = n + Math.max(0, F.generationCapture.steps.length - 1);
  const heads = F.generationCapture.query_head_count;
  return {...F.generationCapture, schema_version: '0.7.0', layer_count: layers,
    full_attention: {layout: 'bf16_lower_triangle_rows_v1', position_count: n, square_limit, layer_arrays: Array.from({length: layers}, (_, i) => `full_l${i}`),
      query_head_count: heads, bytes: layers * heads * n * (n + 1)}};
}

/** Answer pieces of `longAnswerCapture`, in order (cycled for longer answers). */
const LONG_ANSWER = ['Top', 'Ġleft', 'Ġcorner', ',', 'Ġthe', 'Ġbright', 'Ġpatch', 'Ġin', 'Ġrow', 'Ġone', 'Ġof', 'Ġthree', '.'];

/**
 * The fixture prompt answered with `tokens` generated tokens, of which the first `retained` keep their prediction
 * step (decode row). A full-attention capture (generation schema 0.8.0) keeps every answer token's row, so by
 * default `retained` = `tokens` and the square limit is n + tokens − 1. An ordinary capture keeps the first eight
 * steps (`retained: 8`), and so did full-attention captures made before 0.8.0.
 */
export function longAnswerCapture({tokens = 12, retained = tokens, fullAttention = true, layers = 2}: {tokens?: number; retained?: number; fullAttention?: boolean; layers?: number} = {}): S['GenerationCapture'] {
  const prompt = F.generationPositions.filter(p => p.role !== 'generated');
  const n = prompt.length;
  const pieces = Array.from({length: tokens}, (_, k) => LONG_ANSWER[k % LONG_ANSWER.length]);
  const generated: S['GenerationPosition'][] = pieces.map((piece, k) => ({index: n + k, token_id: 2000 + k, piece, role: 'generated', page_id: null, bbox: null,
    ownership: {'non_image:generated': 1}}));
  const positions = [...prompt, ...generated];
  const kept = Math.min(retained, tokens);
  const steps: S['PredictionStep'][] = Array.from({length: kept}, (_, k) => ({step: k, pass_kind: k ? 'decode' : 'prefill', source_index: n + k - 1,
    consumed_token_id: positions[n + k - 1].token_id, consumed_piece: positions[n + k - 1].piece, predicted_index: n + k, predicted_token_id: generated[k].token_id,
    predicted_piece: pieces[k], target_count: n + k, array: `step_${k}`}));
  const heads = F.generationCapture.query_head_count;
  // All-step full-attention captures are schema 0.8.0; earlier full-attention captures kept 8 steps under 0.7.0.
  return {...F.generationCapture, schema_version: fullAttention && kept === tokens ? '0.8.0' : '0.7.0', layer_count: layers, positions, steps, output_token_ids: generated.map(p => p.token_id),
    answer: pieces.join('').replace(/Ġ/g, ' '), stop_reason: 'eos', coverage: {captured_steps: kept, generated_steps: tokens},
    full_attention: fullAttention ? {layout: 'bf16_lower_triangle_rows_v1', position_count: n, square_limit: n + Math.max(0, kept - 1),
      layer_arrays: Array.from({length: layers}, (_, i) => `full_l${i}`), query_head_count: heads, bytes: layers * heads * n * (n + 1)} : null};
}

// ---------------------------------------------------------------- table grounding case

/** Table-aware methods (methods/_table.py); each returns the captured row with its table aggregation. */
export const TABLE_METHODS = ['table.page_regions', 'table.mapping_audit', 'table.cell_allocation', 'table.lookup_path',
  'table.answer_distractors', 'table.matched_contrast'];
/**
 * A 2 × 4 table over the top two patch rows of the fixture page, one cell per patch: cell `cell-RC` lies
 * on patch 4R + C, so a cell's attention mass is exactly that patch's weight. Merged cells and shared
 * positions are covered by unit/table-aware.test.tsx.
 */
const TABLE_TEXT = [['Item', 'A', 'B', 'C'], ['Hot', '4', '1', '7']];
export const TABLE_CELL_IDS = TABLE_TEXT.flatMap((row, r) => row.map((_, c) => `cell-${r}${c}`));
/** The fixture patch under a table cell. */
export const tableCellPatch = (cellId: string) => 4 * Number(cellId[5]) + Number(cellId[6]);
const box = ([x1, y1, x2, y2]: F.Box) => ({x1, y1, x2, y2});

export const tableRecord: S['TableRecord'] = {key: 'table_fixture', table_id: 'table_fixture', native_id: 'pt1m-fixture', document_id: F.DOCUMENT_ID, page_id: F.PAGE_ID,
  ref: {dataset_id: 'dataset_fixture', dataset_version: 'fixture', domain: 'pubtables_1m', document_id: 1, table_id: 'pt1m-fixture'},
  title: 'Fixture table', revision: 'v0001', table_sha256: '0'.repeat(64), grounding_revision: 'v0001', grounding_sha256: '0'.repeat(64),
  geometry_frame: 'page_image_pixels', bbox: box([0, 0, 400, 200]), wrap_bbox: box([0, 0, 400, 230]), crop_width: 400, crop_height: 200,
  crop_sha256: '0'.repeat(64), crop_to_page: [[1, 0, 0], [0, 1, 0]], registration_correlation: 1, geometry_verification: 'Fixture: crop and page coincide.',
  row_count: 2, column_count: 4,
  rows: [0, 1].map(r => ({native_id: `row-000${r + 1}`, index: r, bbox: box([0, 100 * r, 400, 100 * (r + 1)]), crop_bbox: box([0, 100 * r, 400, 100 * (r + 1)])})),
  columns: [0, 1, 2, 3].map(c => ({native_id: `column-000${c + 1}`, index: c, bbox: box([100 * c, 0, 100 * (c + 1), 200]), crop_bbox: box([100 * c, 0, 100 * (c + 1), 200])})),
  cells: TABLE_CELL_IDS.map(id => {
    const r = Number(id[5]), c = Number(id[6]), patch = box(F.patchBox(tableCellPatch(id)));
    return {native_id: id, row_ids: [`row-000${r + 1}`], column_ids: [`column-000${c + 1}`], row_indices: [r], column_indices: [c], bbox: patch, crop_bbox: patch,
      pdf_bboxes: [], text: TABLE_TEXT[r][c], text_method: 'fixture', text_status: 'unreviewed', pdf_text: TABLE_TEXT[r][c],
      roles: [{role: r === 0 ? 'column_header' : c === 0 ? 'row_header' : 'data', status: 'proposed', method: 'fixture'}],
      native_flags: r === 0 ? ['column_header' as const] : []};
  }),
  quality_status: 'needs_review', publishable: false, quality_issues: [], source: {}};

/** "What is B for Hot?": the answer cell is row 2, column 3, reached through the row header "Hot" and the column header "B". */
export const tableGrounding: S['TableGrounding'] = {table_id: tableRecord.table_id, native_table_id: tableRecord.native_id, table_revision: 'v0001',
  table_sha256: tableRecord.table_sha256, answer_cell_ids: ['cell-12'], answer_row_ids: ['row-0002'], answer_column_ids: ['column-0003'],
  row_header_path: ['cell-10'], column_header_path: ['cell-02'], support_cell_ids: [], accepted_answers: ['1'], value_type: 'integer', unit: null,
  rationale: 'Fixture: the Hot row under column B.', binding_status: 'text_matched_not_human_approved', pdf_answer_text: '1',
  authored_input: 'table_crop', review_status: 'unreviewed', support_role: 'evaluation_annotation'};

export const tableCaseRecord: S['CaseRecord'] = {...F.caseRecord, query: 'What is B for Hot?', reference_answer: '1', support_element_ids: [], table_grounding: tableGrounding};

/** Two saved captures of other questions on the table: both form the question-swap baseline; the first is the matched question. */
const relatedCapture = (capture_id: string, query: string, answer_row: number, answer_column: number, relation: string): S['TableRelatedCapture'] => ({capture_id,
  case_id: `case_${capture_id}`, question_id: `question_${capture_id}`, query, answer: TABLE_TEXT[answer_row][answer_column], created_at: '2026-10-19T00:00:00+00:00',
  answer_row, answer_column, relation, binding_status: 'text_matched_not_human_approved', comparable: true});
export const tableRelated: S['TableRelatedPage'] = {table_id: tableRecord.table_id,
  items: [relatedCapture('generation_same_row', 'What is C for Hot?', 1, 3, 'same_row'), relatedCapture('generation_other', 'What is Item for Hot?', 1, 0, 'other')],
  baseline_capture_ids: ['generation_same_row', 'generation_other'], matched_capture_ids: ['generation_same_row']};

const tableStructure: S['TableStructure'] = {provenance: 'dataset', source: 'fixture table', adapter_version: 'fixture', table_id: tableRecord.table_id,
  title: tableRecord.title, revision: tableRecord.revision, cell_count: tableRecord.cells.length, row_count: 2, column_count: 4, quality_status: tableRecord.quality_status,
  grounding: true, binding_status: tableGrounding.binding_status};

/**
 * The table aggregation of a generator row, as methods/_table.py partitions the page: each cell takes its
 * patch's weight, table rows and columns sum their cells, and the bottom patch row is the rest of the page.
 */
export function tableAllocation(row: F.Row, selection: string): S['TableAllocation'] {
  const image = (k: number) => row.values[F.generatorPatchIndex(k)] ?? 0;
  const area = 1 / F.PATCH_COUNT;
  const enrichment = (mass: number, share: number) => row.image_mass > 0 ? mass / row.image_mass / share : null;
  const cells = TABLE_CELL_IDS.map(id => {
    const mass = image(tableCellPatch(id));
    return {cell_id: id, row_indices: [Number(id[5])], column_indices: [Number(id[6])], mass, area_share: area, enrichment: enrichment(mass, area),
      isolable: true, touching_positions: 1, largest_position_share: 1};
  });
  const band = (prefix: string, index: number, members: typeof cells) => {
    const mass = members.reduce((s, c) => s + c.mass, 0), share = members.length * area;
    return {band_id: `${prefix}-000${index + 1}`, index, mass, area_share: share, enrichment: enrichment(mass, share)};
  };
  const tableMass = cells.reduce((s, c) => s + c.mass, 0);
  const rest = [8, 9, 10, 11].reduce((s, k) => s + image(k), 0);
  return {table_id: tableRecord.table_id, native_table_id: tableRecord.native_id, policy: 'table_cell_partition_v1', selection,
    denominator: 'original captured domain', total_mass: row.total_mass, image_mass: row.image_mass, table_mass: tableMass, cells,
    rows: [0, 1].map(r => band('row', r, cells.filter(c => c.row_indices[0] === r))),
    columns: [0, 1, 2, 3].map(c => band('column', c, cells.filter(x => x.column_indices[0] === c))),
    regions: [{region: 'table', label: 'Table cells', mass: tableMass, area_share: 8 * area, enrichment: enrichment(tableMass, 8 * area)},
      {region: 'rest_of_page', label: 'Rest of the page (unannotated)', mass: rest, area_share: 4 * area, enrichment: enrichment(rest, 4 * area)},
      ...row.allocations.filter(a => a.bucket.startsWith('non_image:')).map(a => ({region: a.bucket, label: a.bucket.slice('non_image:'.length), mass: a.mass}))],
    resolution: {image_positions: F.PATCH_COUNT, positions_on_table: 8, cell_count: cells.length, isolable_cells: cells.length, rows_per_position: 1, columns_per_position: 1}};
}

/** Methods the retriever offers for saved encodings (retrieval/attention.py SUPPORTED). */
const RETRIEVER_SUPPORTED = new Set(['attention.row', 'attention.element_allocation', 'attention.mean_heads', 'attention.max_heads',
  'attention.min_heads', 'attention.weighted_heads', 'attention.entropy', 'attention.concentration', 'attention.topk_mass',
  'attention.head_by_head', 'attention.across_layers', 'attention.sinks', 'attention.modality_ratio', 'heads.distribution_similarity',
  'attention.element_type_profile', 'attention.designated_support', 'attention.reading_order', 'attention.layout_sinks',
  'attention.section_rollup', 'attention.modality_budget_depth', 'attention.rollout', 'attention.spatial_distance', 'attention.whole_query',
  'attention.text_to_text', 'attention.text_to_vision', 'attention.vision_to_text', 'attention.vision_to_vision', 'attention.spatial_direction',
  'attention.local_global', 'attention.row_structure', 'attention.column_structure', 'attention.sink_detection', 'heads.clustering',
  'heads.pattern_taxonomy', 'attention.partial_rollout', 'attention.rollout_evolution', 'attention.graph', 'attention.flow']);
const RETRIEVER_REQUIREMENTS = new Set(['image_mask', 'question_source_rows', 'multiple_source_rows', 'captured_text_sources', 'captured_image_sources', 'full_square_attention_matrices']);

/** Retriever descriptors derived from the registry as retrieval/attention.py `methods()` derives them. */
export const retrieverMethods: Method[] = catalogue.filter(m => RETRIEVER_SUPPORTED.has(m.method_id)).map(m => {
  const {step: _step, ...properties} = (m.parameters_schema.properties ?? {}) as Record<string, unknown>;
  return {...m, variant_id: 'retriever_encoding_v1', implementation_version: 'retriever_encoding_rows_v1',
    requires: ['retriever_encoding_attention', 'complete_declared_support', ...(m.perspective === 'document_aware' ? ['validated_ownership_map', 'mapped_elements'] : []),
      ...m.requires.filter(r => r.endsWith('_annotations') || RETRIEVER_REQUIREMENTS.has(r))],
    parameters_schema: {...m.parameters_schema, properties}};
});

/**
 * What each method's real response carries, recorded from the live API on 13 October 2026
 * for a generator capture and a ColPali document encoding (table methods on 19 October 2026,
 * for a table grounding capture). Every other method returns statistics or tables only, with no page row.
 */
export const ROW_METHODS = new Set(['attention.row', 'attention.element_allocation', 'attention.mean_heads', 'attention.max_heads',
  'attention.min_heads', 'attention.weighted_heads', 'attention.whole_query', 'attention.text_to_text', 'attention.text_to_vision',
  'attention.vision_to_text', 'attention.vision_to_vision', 'attention.spatial_distance', 'attention.rollout', 'attention.partial_rollout',
  ...TABLE_METHODS]);
export const COMPARISON_AXIS: Record<string, 'head' | 'layer'> = {'attention.head_by_head': 'head', 'attention.across_layers': 'layer', 'attention.rollout_evolution': 'layer'};
const QUANTITY: Record<string, S['AttentionRow']['quantity']> = {'attention.max_heads': 'head_maximum_envelope', 'attention.min_heads': 'head_minimum_envelope',
  'attention.whole_query': 'whole_query_image_allocation', 'attention.text_to_text': 'modality_block_attention', 'attention.text_to_vision': 'modality_block_attention',
  'attention.vision_to_text': 'modality_block_attention', 'attention.vision_to_vision': 'modality_block_attention', 'attention.rollout': 'rollout_connectivity',
  'attention.partial_rollout': 'partial_rollout_connectivity', 'attention.rollout_evolution': 'rollout_evolution_connectivity'};

/** Hot patches in serving order: a permutation, so consecutive methods never share a hot patch. */
const HOT_ORDER = [5, 2, 9, 0, 7, 11, 4, 1, 10, 3, 6, 8];
/** The element attention profile's hot patch at each layer and head: the next in HOT_ORDER, layer by layer. */
export const profileHot = (layer: number, head: number, heads: number) => HOT_ORDER[(layer * heads + head) % HOT_ORDER.length];

export type Served = {kind: 'row' | 'analysis' | 'retriever'; method_id: string; result: Calculation & {row?: S['AttentionRow'] | null};
  /** The request body of an analysis or retriever calculation. */
  request?: Record<string, unknown>};

export type MockOptions = {
  retrievalCapture?: S['RetrievalCapture'];
  /** The generator capture served for F.GENERATION_ID (e.g. `fullAttentionCapture()`). */
  generationCapture?: S['GenerationCapture'];
  /** Image weights for a method; by default each new method gets the next hot patch over a ramp. */
  image?: (method_id: string, hot: number) => number[];
  /** Serve the case as a table grounding question (`tableCaseRecord`) with its table, related captures and table capabilities. */
  table?: boolean;
};

/** A solid white PNG of the fixture page size, so composited colours are known exactly. */
export function whitePng(width: number = F.PAGE.width, height: number = F.PAGE.height): Buffer {
  const table = Array.from({length: 256}, (_, n) => { let c = n; for (let k = 0; k < 8; k++) c = c & 1 ? 0xedb88320 ^ (c >>> 1) : c >>> 1; return c >>> 0; });
  const crc = (bytes: Buffer) => { let c = 0xffffffff; for (const b of bytes) c = table[(c ^ b) & 0xff] ^ (c >>> 8); return (c ^ 0xffffffff) >>> 0; };
  const chunk = (type: string, data: Buffer) => {
    const body = Buffer.concat([Buffer.from(type, 'ascii'), data]);
    const out = Buffer.alloc(body.length + 8);
    out.writeUInt32BE(data.length, 0); body.copy(out, 4); out.writeUInt32BE(crc(body), body.length + 4);
    return out;
  };
  const header = Buffer.alloc(13);
  header.writeUInt32BE(width, 0); header.writeUInt32BE(height, 4); header.set([8, 2, 0, 0, 0], 8);
  const scanline = Buffer.concat([Buffer.from([0]), Buffer.alloc(width * 3, 255)]);
  return Buffer.concat([Buffer.from([137, 80, 78, 71, 13, 10, 26, 10]), chunk('IHDR', header),
    chunk('IDAT', deflateSync(Buffer.concat(Array(height).fill(scanline)))), chunk('IEND', Buffer.alloc(0))]);
}

export async function mockAttentionApi(page: Page, options: MockOptions = {}) {
  const retrievalCapture = options.retrievalCapture ?? F.retrievalCapture;
  const generationCapture = options.generationCapture ?? F.generationCapture;
  const capabilities = new Set([...GENERATOR_CAPABILITIES, ...(generationCapture.full_attention ? FULL_ATTENTION_CAPABILITIES : []),
    ...(options.table ? ['table_structure_annotations', 'table_grounding_annotations'] : [])]);
  const served: Served[] = [];
  const unexpected: string[] = [];
  const hots = new Map<string, number>();
  const hotFor = (method_id: string) => {
    if (!hots.has(method_id)) hots.set(method_id, HOT_ORDER[hots.size % HOT_ORDER.length]);
    return hots.get(method_id)!;
  };
  const imageFor = (method_id: string) => (options.image ?? ((_, hot) => F.patchValues(hot)))(method_id, hotFor(method_id));
  const png = whitePng();

  /** True when the served capture kept a prediction step `step` (the server refuses any other step). */
  const hasStep = (step: number) => generationCapture.steps.some(s => s.step === step);
  /**
   * The served capture's row for a prediction step or a retained question row, as `F.generatorRow` builds it for the
   * fixture (identical for the fixture capture), so captures with more answer steps answer every step they kept.
   */
  const capturedRow = (image: number[], {step = 0, layer = 0, head = 0, source_index, quantity}: {step?: number; layer?: number; head?: number; source_index?: number | null; quantity?: S['AttentionRow']['quantity']}) => {
    const question = generationCapture.source_rows?.find(r => r.source_index === source_index);
    const prediction = generationCapture.steps.find(s => s.step === step)!;
    const row = F.attentionRow({positions: generationCapture.positions, patchIndex: F.generatorPatchIndex, image, capture_id: F.GENERATION_ID,
      targetCount: question?.target_count ?? prediction.target_count, source_index: question?.source_index ?? prediction.source_index, step, layer, head, quantity});
    // A full-attention capture answers any position below its square limit with that position's causal row.
    const anyPosition = !question && generationCapture.full_attention && source_index != null && source_index < generationCapture.full_attention.square_limit;
    const exact = {...row, predicted_index: question?.predicted_index ?? prediction.predicted_index};
    return anyPosition ? {...exact, source_index, predicted_index: null, target_count: source_index + 1, values: exact.values.slice(0, source_index + 1)} : exact;
  };

  /** Element attention profile: each layer and head is its own known-answer row, allocated by the fixture ownership. */
  const elementProfile = (step: number, source_index?: number | null): S['ElementProfile'] => {
    const {layer_count, query_head_count} = generationCapture;
    const rows = Array.from({length: layer_count}, (_, layer) => Array.from({length: query_head_count}, (_, head) =>
      capturedRow(F.patchValues(profileHot(layer, head, query_head_count)), {step, layer, head, source_index})));
    return {policy: 'element_layer_head_mass_v1', layer_count, head_count: query_head_count, kv_head_count: generationCapture.kv_head_count,
      deepstack_layers: [], image_positions: F.PATCH_COUNT, image_mass: rows.map(r => r.map(row => row.image_mass)), significant_digits: 7,
      unmeasured_element_ids: [], denominator: 'fixture', elements: F.ELEMENTS.map(e => ({element_id: e.element_id, page_id: F.PAGE_ID,
        label: `${e.element_type} · ${e.native_id}`, reading_order: e.reading_order, area_share: e.patches.length / F.PATCH_COUNT,
        mass: rows.map(r => r.map(row => F.elementMasses(row)[e.element_id] ?? 0))}))};
  };

  const generatorResult = (method_id: string, parameters: Record<string, number>, source_index?: number | null): Calculation => {
    if (method_id === 'attention.element_profile') return {element_profile: elementProfile(parameters.step ?? 0, source_index)};
    const image = imageFor(method_id);
    const axes = {step: parameters.step ?? 0, layer: parameters.layer ?? 0, head: parameters.head ?? 0, source_index};
    if (COMPARISON_AXIS[method_id]) {
      const axis = COMPARISON_AXIS[method_id];
      const count = axis === 'head' ? generationCapture.query_head_count : generationCapture.layer_count;
      return {comparisons: Array.from({length: count}, (_, i) => ({label: `${axis === 'head' ? 'Head' : 'Layer'} ${i}`,
        row: capturedRow(F.patchValues(HOT_ORDER[(HOT_ORDER.indexOf(hotFor(method_id)) + i) % HOT_ORDER.length]), {...axes, [axis]: i, quantity: QUANTITY[method_id]})}))};
    }
    if (ROW_METHODS.has(method_id)) {
      const whole = method_id === 'attention.whole_query';
      const row = capturedRow(image, {...axes, quantity: QUANTITY[method_id]});
      if (TABLE_METHODS.includes(method_id)) return {row, table_allocation: tableAllocation(row, `step ${axes.step} · layer ${axes.layer} · head ${axes.head}`)};
      return {row: whole ? {...row, source_index: null} : row, source_indices: whole ? F.generationCapture.source_rows!.map(r => r.source_index) : undefined};
    }
    return {statistics: [{name: 'Fixture statistic', value: 0.5, units: 'dimensionless'}]};
  };

  const retrieverResult = (method_id: string, body: {encoding_id: string; source_index: number; parameters: Record<string, number>}): Calculation => {
    const axes = {source_index: body.source_index, layer: body.parameters.layer ?? 0, head: body.parameters.head ?? 0};
    const encoding = retrievalCapture.attention_encodings?.find(e => e.encoding_id === body.encoding_id);
    const positions = encoding?.pass_kind === 'query' ? retrievalCapture.query_positions : encoding?.positions?.length ? encoding.positions : F.retrieverPositions;
    const row = (image: number[], extra: Partial<typeof axes> = {}) => F.attentionRow({
      positions: positions.map((p, i) => ({...p, ownership: encoding?.ownership?.[i] ?? (p.bbox ? F.retrieverOwnership[i] : {non_page: 1})})),
      patchIndex: k => positions.filter(p => p.bbox)[k]?.index ?? Infinity, image, capture_id: F.RETRIEVAL_ID,
      targetCount: positions.length, ...axes, ...extra, quantity: QUANTITY[method_id],
    });
    if (COMPARISON_AXIS[method_id]) {
      const axis = COMPARISON_AXIS[method_id];
      return {comparisons: Array.from({length: 2}, (_, i) => ({label: `${axis === 'head' ? 'Head' : 'Layer'} ${i}`,
        row: row(F.patchValues(HOT_ORDER[(HOT_ORDER.indexOf(hotFor(method_id)) + i) % HOT_ORDER.length]), {[axis]: i})}))};
    }
    if (ROW_METHODS.has(method_id)) return {row: row(imageFor(method_id))};
    return {statistics: [{name: 'Fixture statistic', value: 0.5, units: 'dimensionless'}]};
  };

  const analysisRecord = (body: {method_id: string; variant_id: string; parameters: Record<string, number>; source_index?: number}, result: Calculation): S['AnalysisResult'] => ({
    ...result, normalization: 'fixture', schema_version: '0.9.0', analysis_id: `analysis_${body.method_id.replace(/\W/g, '_')}_${served.length}`,
    method_id: body.method_id, variant_id: body.variant_id, implementation_version: 'fixture', generation_capture_id: F.GENERATION_ID,
    capture_sha256: '0'.repeat(64), array_sha256: '0'.repeat(64), parameters: body.parameters, source_index: result.row?.source_index ?? body.source_index ?? 20,
    predicted_index: result.row?.predicted_index ?? 21, target_count: result.row?.target_count ?? 21,
    valid_target_indices: Array.from({length: result.row?.target_count ?? 21}, (_, i) => i), quantity: 'fixture', units: 'dimensionless mass', coverage: {selection: 'fixture'}});

  const available = (m: Method) => m.status === 'implemented' && m.requires.every(r => capabilities.has(r));
  const capturePage = <T,>(items: T[]) => ({items, next_cursor: null});

  async function handle(route: Route) {
    const request = route.request();
    const url = new URL(request.url());
    const path = url.pathname, method = request.method();
    const json = (body: unknown, status = 200) => route.fulfill({status, contentType: 'application/json', body: JSON.stringify(body)});
    const body = () => request.postDataJSON();
    const [, , , ...parts] = path.split('/'); // '', 'api', 'v1', ...
    const at = (...expected: string[]) => parts.length === expected.length && expected.every((e, i) => e === '*' || e === parts[i]);
    if (path === '/health/ready') return json({status: 'ready', dependencies: {metadata: 'ready', artifact_store: 'ready'}});
    if (method === 'GET') {
      if (at('cases', F.CASE_ID)) return json(options.table ? tableCaseRecord : F.caseRecord);
      if (at('documents', F.DOCUMENT_ID)) return json(F.documentRecord);
      if (at('pages', F.PAGE_ID)) return json(F.pageRecord);
      if (at('pages', F.PAGE_ID, 'elements')) return json({items: F.elementRecords, total: F.elementRecords.length});
      if (at('pages', F.PAGE_ID, 'image') || at('generation', 'captures', F.GENERATION_ID, 'pages', F.PAGE_ID, 'input-image'))
        return route.fulfill({status: 200, contentType: 'image/png', body: png});
      if (options.table && at('pages', F.PAGE_ID, 'tables')) return json({items: [tableRecord], total: 1, coordinate_convention: 'page_image_pixels'});
      if (options.table && at('generation', 'captures', F.GENERATION_ID, 'table-related')) return json(tableRelated);
      if (options.table && at('generation', 'captures', F.GENERATION_ID, 'answer-check'))
        return json({verdict: 'incorrect', basis: 'fixture exact match', matched_answer: null, accepted_answers: tableGrounding.accepted_answers, interpretation: 'Fixture: the answer is not an accepted answer.'});
      if (at('generation', 'captures', F.GENERATION_ID)) return json(generationCapture);
      if (at('generation', 'captures')) return json(capturePage([generationCapture]));
      if (at('generation', 'captures', F.GENERATION_ID, 'rows')) {
        const q = (k: string) => url.searchParams.has(k) ? Number(url.searchParams.get(k)) : undefined;
        const step = q('step') ?? 0, source_index = q('source_index') ?? null;
        // A step the capture did not keep has no row; the server refuses it, and so does the fixture.
        if (source_index == null && !hasStep(step)) {
          unexpected.push(`GET ${path}?step=${step} (step not captured)`);
          return json({error: {code: 'invalid_request', message: `Fixture capture has no prediction step ${step}`}}, 422);
        }
        // Raw attention and element allocation both read this saved row, as on the real server.
        const row = capturedRow(imageFor('attention.row'), {step, layer: q('layer'), head: q('head'), source_index});
        served.push({kind: 'row', method_id: 'attention.row', result: {row}});
        return json(row);
      }
      if (at('generation', 'captures', F.GENERATION_ID, 'received')) {
        const q = (k: string) => Number(url.searchParams.get(k) ?? 0);
        return json(F.generatorColumn(q('target_index'), {layer: q('layer'), head: q('head')}));
      }
      if (at('retrieval', 'captures', F.RETRIEVAL_ID)) return json(retrievalCapture);
      if (at('retrieval', 'captures')) return json(capturePage([retrievalCapture]));
      if (at('retrieval', 'captures', F.RETRIEVAL_ID, 'rows')) {
        const query_index = Number(url.searchParams.get('query_index'));
        const similarities = F.similarityValues(query_index);
        const winner_index = similarities.reduce<number>((best, v, i) => (v ?? -Infinity) > (similarities[best] ?? -Infinity) ? i : best, 0);
        return json({capture_id: F.RETRIEVAL_ID, page_id: F.PAGE_ID, query_index, similarities, winner_index, maximum: similarities[winner_index],
          units: 'dot product of captured vectors', normalization: 'fixture'});
      }
      if (at('retrieval', 'captures', F.RETRIEVAL_ID, 'columns')) {
        const position_index = Number(url.searchParams.get('position_index'));
        const similarities = F.queryPositions.map(q => F.similarityValues(q.index)[position_index]);
        return json({capture_id: F.RETRIEVAL_ID, page_id: F.PAGE_ID, position_index, similarities,
          winning_query_indices: position_index === F.SIMILARITY_WINNER ? [3] : [], maximum: Math.max(...similarities.map(v => v ?? -Infinity))});
      }
      if (at('retrieval', 'attention-methods')) return json(retrieverMethods);
      if (at('methods', 'availability')) {
        const retrievalOnly = !url.searchParams.has('generation_capture_id');
        return json({items: catalogue.map(m => ({method: m, available: !retrievalOnly && available(m), reasons: available(m) ? [] : ['Not available for the fixture capture.']})),
          case_id: F.CASE_ID, generation_capture_id: url.searchParams.get('generation_capture_id'), retrieval_capture_id: null, structure: F.documentStructure,
          table_structure: options.table ? tableStructure : null});
      }
      if (at('methods')) return json({items: catalogue});
      if (at('methods', '*')) {
        const variant = url.searchParams.get('variant_id');
        const found = catalogue.find(m => m.method_id === parts[1] && (!variant || m.variant_id === variant)) ?? catalogue.find(m => m.method_id === parts[1]);
        return found ? json(found) : json({error: {code: 'not_found', message: 'Unknown fixture method'}}, 404);
      }
      if (at('setup', 'models')) return json([]);
      if (at('experiments') || at('attention-interventions') || at('document-studies')) return json(capturePage([]));
      if (at('retrieval', 'plan')) return json({available: false, reason: 'Fixture: acquisition disabled', model_registration_id: 'builtin-retrieval', model_name: 'Fixture retriever', attention_available: false});
      if (at('generation', 'plan')) return json({available: false, reason: 'Fixture: acquisition disabled'});
    }
    if (method === 'POST') {
      if (at('methods', 'analyses')) {
        const request = body();
        if (request.source_index == null && !hasStep(request.parameters?.step ?? 0)) {
          unexpected.push(`POST ${path} step=${request.parameters?.step} (step not captured)`);
          return json({error: {code: 'invalid_request', message: `Fixture capture has no prediction step ${request.parameters?.step}`}}, 422);
        }
        const result = generatorResult(request.method_id, request.parameters ?? {}, request.source_index);
        served.push({kind: 'analysis', method_id: request.method_id, result, request});
        return json(analysisRecord(request, result));
      }
      if (at('retrieval', 'captures', F.RETRIEVAL_ID, 'attention')) {
        const request = body();
        const result = retrieverResult(request.method_id, request);
        served.push({kind: 'retriever', method_id: request.method_id, result, request});
        return json({...result, normalization: 'fixture'});
      }
    }
    unexpected.push(`${method} ${path}`);
    return json({error: {code: 'not_found', message: `Fixture API has no ${method} ${path}`}}, 404);
  }

  await page.route(url => url.pathname.startsWith('/api/') || url.pathname.startsWith('/health/'), handle);
  return {
    served, unexpected,
    /** Most recent result served for a method, in the role that requested it. */
    latest(method_id: string, kind?: Served['kind']) {
      return served.filter(s => s.method_id === method_id && (!kind || s.kind === kind)).at(-1);
    },
  };
}
