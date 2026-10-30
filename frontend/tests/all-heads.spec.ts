import { test, expect, type Page, type Request } from '@playwright/test';
import { findGenerationCapture, requireSaved, type SavedGeneration } from './savedEvidence';
import { choosePerspective, methodChoice, saveInvestigation, methodBrowse } from './workspaceHelpers';

const RAW = 'attention.row/captured_weights_v1';
const MEAN = 'attention.mean_heads/captured_prediction_v1';

type AnalysisBody = {method_id: string; parameters: Record<string, unknown>};

/** Method analyses and raw rows the page requests, newest last. */
function recordRequests(page: Page) {
  const analyses: AnalysisBody[] = [], rows: URL[] = [];
  page.on('request', (request: Request) => {
    const url = new URL(request.url());
    if (request.method() === 'POST' && url.pathname === '/api/v1/methods/analyses') analyses.push(request.postDataJSON() as AnalysisBody);
    if (request.method() === 'GET' && /^\/api\/v1\/generation\/captures\/[^/]+\/rows$/.test(url.pathname)) rows.push(url);
  });
  return {analyses, rows};
}

test('All heads switches raw attention to the mean over every head and back, and survives saving', async ({page}) => {
  // Prefer a compact one-page capture at the standard image size with head cells (at most 24 heads): large table
  // or two-page captures make the save-and-reopen step slow without testing anything more.
  const capture = await findGenerationCapture(page.request, c => c.query_head_count > 5 && c.layer_count > 2 && c.context.representation === 'image',
    c => c.context.pages.length === 1 && (c.context.image_token_limit ?? 512) === 512 && c.query_head_count <= 24);
  requireSaved(!capture, 'Needs a saved image generator capture with more than five heads');
  const heads = Array.from({length: capture!.query_head_count}, (_, i) => i);
  const requests = recordRequests(page);
  let acquisitions = 0;
  await page.route(/\/api\/v1\/(generation|retrieval)\/runs$/, route => {acquisitions++; return route.abort();});
  await page.goto('/attention-tool?generation=' + capture!.capture_id);

  const picker = methodChoice(page);
  const scrubber = page.getByTestId('layer-scrubber');
  const all = scrubber.getByRole('button', {name: 'All', exact: true});
  const head = (n: number) => scrubber.getByRole('radio', {name: `Head ${n}`, exact: true});
  const layer = page.getByLabel('Attention layer', {exact: true});
  const saveState = page.locator('.docattentionlens-save-state');
  await expect(picker).toHaveAttribute('data-method-key', RAW);
  await expect(page.getByTestId('attention-patch').first()).toBeVisible({timeout: 30_000});
  await expect(all).toHaveAttribute('aria-pressed', 'false');

  // Choose a layer and head first: All keeps the layer and source, and leaving it restores the head.
  await layer.selectOption('2');
  await head(3).click();
  await expect(head(3)).toHaveAttribute('aria-checked', 'true');

  await all.click();
  await expect(picker).toHaveAttribute('data-method-key', MEAN);
  await expect(picker).toContainText(/Mean/);
  await expect(all).toHaveAttribute('aria-pressed', 'true');
  await expect(saveState).toHaveAttribute('data-dirty', 'true');
  // The layer track and the head cells stay; no numbered head is selected while All is.
  await expect(layer).toHaveValue('2');
  await expect(head(0)).toBeVisible();
  await expect(scrubber.getByRole('radio', {checked: true})).toHaveCount(0);
  await expect(page.getByTestId('prediction-alignment').first()).toContainText('all heads');
  // The request averages every head explicitly at the kept step and layer, with no single head.
  await expect.poll(() => requests.analyses.at(-1)?.method_id).toBe('attention.mean_heads');
  const request = requests.analyses.at(-1)!;
  expect(request.parameters.heads).toEqual(heads);
  expect(request.parameters.layer).toBe(2);
  expect(request.parameters).not.toHaveProperty('head');
  await expect(page.getByTestId('attention-patch').first()).toBeVisible({timeout: 30_000});
  await expect(page.locator('.docattentionlens-legend-share')).toContainText('of attention');

  // A toggles back to the single-head method, on the head shown before, and on again.
  await page.keyboard.press('a');
  await expect(picker).toHaveAttribute('data-method-key', RAW);
  await expect(all).toHaveAttribute('aria-pressed', 'false');
  await expect(head(3)).toHaveAttribute('aria-checked', 'true');
  await expect(layer).toHaveValue('2');
  await page.keyboard.press('a');
  await expect(picker).toHaveAttribute('data-method-key', MEAN);
  await expect(all).toHaveAttribute('aria-pressed', 'true');

  // A numbered head leaves All for raw attention on that head.
  await head(5).click();
  await expect(picker).toHaveAttribute('data-method-key', RAW);
  await expect(all).toHaveAttribute('aria-pressed', 'false');
  await expect(head(5)).toHaveAttribute('aria-checked', 'true');
  await expect.poll(() => requests.rows.at(-1)?.searchParams.get('head')).toBe('5');
  expect(requests.rows.at(-1)!.searchParams.get('layer')).toBe('2');
  await expect(page.getByTestId('attention-patch').first()).toBeVisible({timeout: 30_000});

  // Saved with All on, the investigation reopens on Mean across heads over every head.
  await page.keyboard.press('a');
  await expect(picker).toHaveAttribute('data-method-key', MEAN);
  await saveInvestigation(page);
  await expect(page).toHaveURL(/investigation_/);
  await expect(saveState).toHaveAttribute('data-dirty', 'false');
  const id = new URL(page.url()).pathname.split('/').pop()!;
  const saved = await page.request.get('/api/v1/investigations/' + id).then(r => r.json());
  expect(saved.view_state.attention_method_id).toBe('attention.mean_heads');
  expect(saved.view_state.attention_parameters.heads).toEqual(heads);
  expect(saved.view_state.attention_layer).toBe(2);
  expect(saved.view_state.attention_head).toBe(5);

  await page.reload();
  await expect(picker).toHaveAttribute('data-method-key', MEAN);
  await expect(all).toHaveAttribute('aria-pressed', 'true');
  await expect(layer).toHaveValue('2');
  await expect(page.getByTestId('attention-patch').first()).toBeVisible({timeout: 30_000});
  await expect(saveState).toHaveAttribute('data-dirty', 'false');
  // After reopening, leaving All returns to raw attention (the default single-head method).
  await head(1).click();
  await expect(picker).toHaveAttribute('data-method-key', RAW);
  await expect(head(1)).toHaveAttribute('aria-checked', 'true');
  await expect(saveState).toHaveAttribute('data-dirty', 'true');
  expect(acquisitions).toBe(0);
});

test('All heads in Document-aware keeps the element grouping: grouped raw attention ↔ grouped mean across heads', async ({page}) => {
  // The newest image capture whose elements map to patches (element totals) and that offers the head mean.
  const listed: SavedGeneration[] = (await page.request.get('/api/v1/generation/captures?limit=25').then(r => r.json())).items;
  let capture: SavedGeneration | undefined;
  // Head cells (the radios this test clicks) exist for at most 24 heads; more heads use a slider.
  for (const candidate of listed.filter(c => c.query_head_count > 3 && c.query_head_count <= 24 && c.context.representation === 'image')) {
    const availability = await page.request.get(`/api/v1/methods/availability?case_id=${candidate.case_id}&generation_capture_id=${candidate.capture_id}`).then(r => r.json());
    const ready = (id: string) => availability.items.some((p: {method: {method_id: string}; available: boolean}) => p.method.method_id === id && p.available);
    if (ready('attention.element_allocation') && ready('attention.mean_heads')) {capture = candidate; break;}
  }
  requireSaved(!capture, 'Needs a saved image generator capture with element totals and the head mean available');
  const requests = recordRequests(page);
  await page.goto('/attention-tool?generation=' + capture!.capture_id);
  const picker = methodChoice(page);
  const scrubber = page.getByTestId('layer-scrubber');
  const all = scrubber.getByRole('button', {name: 'All', exact: true});
  await expect(methodBrowse(page)).toBeEnabled();
  await choosePerspective(page, 'Document-aware');
  const grouped = 'attention.element_allocation/ownership_mass_v1';
  await expect(picker).toHaveAttribute('data-method-key', grouped);
  await scrubber.getByRole('radio', {name: 'Head 2', exact: true}).click();
  await all.click();
  await expect(all).toHaveAttribute('aria-pressed', 'true');
  await expect(picker).toHaveAttribute('data-perspective', 'document_aware');
  await expect(picker).toHaveAttribute('data-method-key', grouped);
  await expect(picker).toHaveAttribute('data-input-key', MEAN);
  await expect.poll(() => requests.analyses.at(-1)?.method_id).toBe('attention.mean_heads');
  expect(requests.analyses.at(-1)!.parameters.heads).toEqual(Array.from({length: capture!.query_head_count}, (_, i) => i));
  await page.keyboard.press('a');
  await expect(all).toHaveAttribute('aria-pressed', 'false');
  await expect(picker).toHaveAttribute('data-perspective', 'document_aware');
  await expect(picker).toHaveAttribute('data-method-key', grouped);
  await expect(picker).not.toHaveAttribute('data-input-key', MEAN);
  await expect(scrubber.getByRole('radio', {name: 'Head 2', exact: true})).toHaveAttribute('aria-checked', 'true');
});

test('retriever attention: All switches to the encoding head mean and a numbered head returns', async ({page}) => {
  const records = await page.request.get('/api/v1/retrieval/captures?limit=25').then(r => r.json());
  type Encoding = {pass_kind: string; query_head_count: number};
  const capture = records.items.find((c: {attention_encodings?: Encoding[]}) => c.attention_encodings?.some(e => e.pass_kind === 'document' && e.query_head_count > 2));
  requireSaved(!capture, 'Needs a saved retrieval capture with a document attention encoding');
  const count = (capture.attention_encodings as Encoding[]).find(e => e.pass_kind === 'document')!.query_head_count;
  const bodies: {method_id: string; parameters: Record<string, unknown>}[] = [];
  page.on('request', request => {
    if (request.method() === 'POST' && /\/api\/v1\/retrieval\/captures\/[^/]+\/attention$/.test(new URL(request.url()).pathname)) bodies.push(request.postDataJSON());
  });
  await page.goto(`/attention-tool?capture=${capture.capture_id}&role=retrieval`);
  const picker = methodChoice(page);
  const scrubber = page.getByTestId('layer-scrubber');
  const all = scrubber.getByRole('button', {name: 'All', exact: true});
  await expect(picker).toHaveAttribute('data-method-key', 'attention.row/retriever_encoding_v1');
  await expect(all).toHaveAttribute('aria-pressed', 'false');
  await all.click();
  await expect(picker).toHaveAttribute('data-method-key', 'attention.mean_heads/retriever_encoding_v1');
  await expect(all).toHaveAttribute('aria-pressed', 'true');
  await expect(page.getByTestId('retriever-attention-context').first()).toContainText('all heads');
  await expect.poll(() => bodies.at(-1)?.method_id).toBe('attention.mean_heads');
  expect(bodies.at(-1)!.parameters.heads).toEqual(Array.from({length: count}, (_, i) => i));
  expect(bodies.at(-1)!.parameters).not.toHaveProperty('head');
  await expect(page.getByTestId('attention-patch').first()).toBeVisible({timeout: 30_000});
  await scrubber.getByRole('radio', {name: 'Head 2', exact: true}).click();
  await expect(picker).toHaveAttribute('data-method-key', 'attention.row/retriever_encoding_v1');
  await expect(scrubber.getByRole('radio', {name: 'Head 2', exact: true})).toHaveAttribute('aria-checked', 'true');
  await expect.poll(() => bodies.at(-1)?.parameters.head).toBe(2);
});
