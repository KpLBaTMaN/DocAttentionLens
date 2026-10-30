import { expect, test, type Page } from '@playwright/test';
import * as F from '../../fixtures/attention';
import { chooseMethod, methodChoice, openCaptureMenu, openDetailsTab, openMethodPicker, closeMethodPicker } from '../workspaceHelpers';
import { fullAttentionCapture, mockAttentionApi } from './mockApi';

// A generator capture saved with "Save full attention matrices": the matrix methods become selectable in the
// generator role, use the layer scrubber's range mode and draw their returned row on the page.
const capture = fullAttentionCapture(6);
type Analysis = {method_id: string; variant_id: string; parameters: {step?: number; layer?: number; start_layer?: number}; source_index?: number};

const round = (v: number) => Math.round(v * 1e9) / 1e9;
async function patches(page: Page) {
  return page.getByTestId('page-canvas').first().getByTestId('attention-patch').evaluateAll(rects => rects.map(r => ({
    index: Number(r.getAttribute('data-position-index')), opacity: Math.round(Number(r.getAttribute('fill-opacity')) * 1e9) / 1e9})));
}
/** What the page must draw for a row, computed from the fixture alone (as in page-overlay-methods.spec.ts). */
function expectedPatches(row: F.Row) {
  const shown = F.generationPositions.filter(p => p.page_id === F.PAGE_ID && p.bbox && p.index < row.target_count);
  const maximum = Math.max(...shown.filter(p => p.role === 'image').map(p => row.values[p.index]), 0);
  return shown.map(p => ({index: p.index, opacity: round(F.overlayOpacity(row.values[p.index], maximum))}));
}

test('generator rollout on a full-attention capture uses the layer range and draws its row', async ({page}) => {
  const api = await mockAttentionApi(page, {generationCapture: capture});
  const requests: Analysis[] = [];
  page.on('request', r => {if (r.method() === 'POST' && r.url().endsWith('/methods/analyses')) requests.push(r.postDataJSON());});
  await page.setViewportSize({width: 1440, height: 900});
  await page.goto(`/attention-tool?generation=${F.GENERATION_ID}`);
  await expect(page.getByTestId('attention-patch').first()).toBeVisible();

  // The capture identifies itself as a full-attention capture with its stored size.
  await openCaptureMenu(page);
  await expect(page.getByTestId('full-attention-coverage')).toContainText(/^Full attention · .* GB stored/);
  await page.keyboard.press('Escape');

  // Rollout is offered in the generator role.
  const methods = await openMethodPicker(page);
  await methods.getByRole('combobox', {name: 'Find a method'}).fill('rollout');
  await expect(methods.locator('[role="option"][data-method-key="attention.rollout/residual_mean_square_v1"]')).not.toHaveAttribute('aria-disabled', 'true');
  await closeMethodPicker(page);
  await chooseMethod(page, 'attention.rollout/residual_mean_square_v1');
  await expect(methodChoice(page)).toHaveAttribute('data-method-key', 'attention.rollout/residual_mean_square_v1');

  // Range mode: rollout composes layers 0..end; the end layer covers the capture's six layers.
  const interval = page.getByRole('group', {name: 'Rollout layer interval'});
  await expect(interval).toBeVisible();
  await expect(page.getByLabel('End layer', {exact: true}).locator('option')).toHaveCount(capture.layer_count);
  await expect(page.getByLabel('Start layer', {exact: true})).toHaveCount(0);
  await page.getByLabel('End layer', {exact: true}).selectOption('4');
  await expect.poll(() => requests.at(-1)).toMatchObject({method_id: 'attention.rollout', parameters: {step: 0, layer: 4}});
  expect(requests.at(-1)?.source_index).toBeUndefined();

  // The page draws exactly the returned rollout row.
  await expect.poll(() => api.latest('attention.rollout', 'analysis')?.result.row?.layer).toBe(4);
  const rollout = api.latest('attention.rollout', 'analysis')!.result.row!;
  await expect.poll(() => patches(page)).toEqual(expectedPatches(rollout));

  // Partial rollout keeps the interval in the generator state as start_layer.
  await interval.getByRole('button', {name: 'Choose a layer range', exact: true}).click();
  await expect(methodChoice(page)).toHaveAttribute('data-method-key', 'attention.partial_rollout/captured_attention_v1');
  await page.getByLabel('Start layer', {exact: true}).selectOption('2');
  await expect(interval).toContainText('Layers 2–4 · 3 layers composed');
  await expect.poll(() => requests.at(-1)).toMatchObject({method_id: 'attention.partial_rollout', parameters: {step: 0, layer: 4, start_layer: 2}});
  await expect.poll(async () => (await patches(page)).length).toBeGreaterThan(0);
  const partial = api.latest('attention.partial_rollout', 'analysis')!.result.row!;
  await expect.poll(() => patches(page)).toEqual(expectedPatches(partial));

  // Rollout evolution returns one map per layer; the method sentence's chooser selects the map shown.
  await chooseMethod(page, 'attention.rollout_evolution/captured_attention_v1');
  const shown = page.getByLabel('Shown comparison', {exact: true});
  await expect(shown.locator('option')).toHaveCount(capture.layer_count);
  await shown.selectOption('3');
  await expect(shown).toHaveValue('3');

  // Image-source methods read an image patch's own row. A patch reaches only earlier positions, so the page's
  // last patch (which sees every patch on the page) becomes the source, and the sentence names it.
  await chooseMethod(page, 'attention.vision_to_text/captured_attention_v1');
  const lastPatch = F.generationPositions.filter(p => p.role === 'image' && p.page_id === F.PAGE_ID).at(-1)!.index;
  await expect.poll(() => requests.at(-1)).toMatchObject({method_id: 'attention.vision_to_text', source_index: lastPatch});
  await expect(page.getByTestId('prediction-alignment').first()).toContainText(`Image patch (position ${lastPatch})`);
  await expect(page.getByRole('group', {name: 'Shown on the page'})).toContainText(`What patch ${lastPatch} attends to`);
  // Details › Result states the causal scope of a patch row.
  await openDetailsTab(page, 'Result');
  await expect(page.getByTestId('causal-scope-note')).toContainText('Vision → text therefore measures attention to the preceding template text.');
  // Raw attention reads the retained rows again: the source returns to the prediction step.
  await chooseMethod(page, 'attention.row/captured_weights_v1');
  await expect(page.getByTestId('prediction-alignment').first()).toContainText('Step 0');
  expect(api.unexpected).toEqual([]);
});

test('the acquire drawer offers full attention with an estimate and sends the choice', async ({page}) => {
  const api = await mockAttentionApi(page, {generationCapture: {...F.generationCapture, retrieval_capture_id: F.RETRIEVAL_ID}});
  // Routes added later take precedence over the fixture API: an available plan, the pinned model's dimensions,
  // and a run endpoint that records the request and starts nothing.
  await page.route('**/api/v1/generation/plan', route => route.fulfill({json: {available: true, reason: 'Fixture: plan available'}}));
  await page.route('**/api/v1/setup/models', route => route.fulfill({json: [{registration_id: 'builtin', role: 'generation', name: 'Qwen2.5-VL-3B-Instruct',
    model_id: 'Qwen/Qwen2.5-VL-3B-Instruct', revision: 'fixture', validation: '', architecture: {model_type: 'qwen2_5_vl', architectures: [], embedding_dim: null, dtype: 'bfloat16',
      components: [{name: 'Language', layers: 36, query_heads: 16, kv_heads: 2}, {name: 'Vision', layers: 32, query_heads: 16, patch_size: 14, spatial_merge_size: 2}]},
    capabilities: {can_run: true, attention_available: true, limits: {image_positions_per_page: 512, captured_steps: 8, max_new_tokens: 64, full_attention_captured_steps: 64}}}]}));
  const sent: {capture_full_attention?: boolean}[] = [];
  await page.route('**/api/v1/generation/runs', route => {sent.push(route.request().postDataJSON()); return route.fulfill({status: 503, json: {error: {message: 'Fixture: no run started.'}}});});
  await page.setViewportSize({width: 1440, height: 900});
  await page.goto(`/attention-tool?generation=${F.GENERATION_ID}`);
  await expect(page.getByTestId('attention-patch').first()).toBeVisible();
  // Default captures show no full-attention line.
  await openCaptureMenu(page);
  await expect(page.getByTestId('full-attention-coverage')).toHaveCount(0);
  await page.getByTestId('capture-menu').getByRole('button', {name: 'Capture more evidence', exact: true}).click();
  const option = page.getByRole('checkbox', {name: /Save full attention matrices/});
  await expect(option).not.toBeChecked();
  const hint = page.locator('label', {has: option});
  // The option also keeps every answer token's row, so the estimate adds up to 64 decode rows to the prefill triangles.
  await expect(hint).toContainText("every answer token's attention row, so every answer token can be selected");
  await expect(hint).toContainText(/Estimated extra storage: up to about \d+(\.\d+)? GB for Qwen2\.5-VL-3B-Instruct with 1 page\./);
  // The generator's limits come from the catalogue: eight answer rows on an ordinary capture, every one with full attention.
  const limits = page.getByTestId('generator-limits');
  await expect(limits).toContainText('Up to 64 output tokens; attention rows for the first 8 answer tokens (prediction steps); 36 layers, 16 query heads.');
  const generate = page.getByRole('button', {name: 'Generate answer and capture attention', exact: true});
  await generate.click();
  await expect.poll(() => sent.at(-1)?.capture_full_attention).toBe(false);
  await option.check();
  await expect(limits).toContainText("Up to 64 output tokens; every answer token's attention row (up to 64); 36 layers, 16 query heads.");
  await generate.click();
  await expect.poll(() => sent.at(-1)?.capture_full_attention).toBe(true);
  expect(sent).toHaveLength(2);
  expect(api.unexpected).toEqual([]);
});
