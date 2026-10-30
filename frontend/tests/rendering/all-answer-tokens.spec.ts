import { expect, test, type Page } from '@playwright/test';
import * as F from '../../fixtures/attention';
import { chooseMethod, chooseTokenView, openCaptureMenu, promptTokens } from '../workspaceHelpers';
import { longAnswerCapture, mockAttentionApi } from './mockApi';

// Full-attention captures (generation schema 0.8.0) keep the attention row of every answer token, up to 64, not only
// the first eight prediction steps. Every answer token is then a source; ordinary captures keep eight and say why the
// rest cannot be chosen.
type Analysis = {method_id: string; parameters: {step?: number; layer?: number}; source_index?: number};

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
const answerChips = (page: Page) => page.locator('.docattentionlens-source-row').getByRole('group', {name: 'Answer tokens', exact: true}).getByTestId('token-chip');
/** Steps requested from the saved-row endpoint (Raw attention), in order. */
function rowSteps(page: Page) {
  const steps: number[] = [];
  page.on('request', r => {
    const url = new URL(r.url());
    if (r.method() === 'GET' && url.pathname.endsWith(`/generation/captures/${F.GENERATION_ID}/rows`) && !url.searchParams.has('source_index'))
      steps.push(Number(url.searchParams.get('step')));
  });
  return steps;
}
/** Presses `,` or `.` and waits for the source line to show the expected answer step. */
async function stepTo(page: Page, key: ',' | '.', step: number) {
  await page.keyboard.press(key);
  await expect(answerChips(page).nth(step)).toHaveAttribute('aria-pressed', 'true');
}

test('every answer token of an all-step full-attention capture is a source, by click, keys and palette', async ({page}) => {
  const capture = longAnswerCapture({tokens: 12});
  const api = await mockAttentionApi(page, {generationCapture: capture});
  const steps = rowSteps(page);
  const analyses: Analysis[] = [];
  page.on('request', r => {if (r.method() === 'POST' && r.url().endsWith('/methods/analyses')) analyses.push(r.postDataJSON());});
  await page.setViewportSize({width: 1440, height: 900});
  await page.goto(`/attention-tool?generation=${F.GENERATION_ID}`);
  await expect(page.getByTestId('attention-patch').first()).toBeVisible();

  // The capture says it keeps every answer token.
  await openCaptureMenu(page);
  await expect(page.getByTestId('capture-coverage')).toContainText('Attention rows for every answer token (12 of 12 captured)');
  await expect(page.getByTestId('full-attention-coverage')).toContainText('every answer token');
  await page.keyboard.press('Escape');

  // All twelve answer tokens are in the source line and clickable.
  const answer = answerChips(page);
  await expect(answer).toHaveCount(12);
  for (const chip of await answer.all()) await expect(chip).toBeEnabled();

  // Choosing the last one (step 11) requests that step's saved row and draws it.
  const last = capture.steps[11];
  await answer.nth(11).click();
  await expect(answer.nth(11)).toHaveAttribute('aria-pressed', 'true');
  await expect.poll(() => steps.at(-1)).toBe(11);
  await expect(page.getByTestId('prediction-alignment').first()).toContainText(`Step 11 → “three” (position ${last.source_index} → ${last.predicted_index})`);
  await expect.poll(() => api.latest('attention.row', 'row')?.result.row?.step).toBe(11);
  const row = api.latest('attention.row', 'row')!.result.row!;
  expect(row.target_count).toBe(last.target_count);
  await expect.poll(() => patches(page)).toEqual(expectedPatches(row));

  // `,` and `.` step through every answer step; `.` stops at the last.
  await answer.nth(0).click();
  for (let k = 1; k < 12; k++) await stepTo(page, '.', k);
  await page.keyboard.press('.');
  await expect(answer.nth(11)).toHaveAttribute('aria-pressed', 'true');
  await expect.poll(() => steps.at(-1)).toBe(11);
  await stepTo(page, ',', 10);

  // The command palette finds any answer step by number.
  await page.keyboard.press('Control+k');
  const palette = page.getByTestId('command-palette');
  await palette.getByRole('combobox', {name: 'Command'}).fill('step 9');
  await expect(palette.getByRole('option').first()).toContainText('Step 9 · “one”');
  await page.keyboard.press('Enter');
  await expect(answer.nth(9)).toHaveAttribute('aria-pressed', 'true');
  await expect.poll(() => steps.at(-1)).toBe(9);

  // Method analyses read the chosen step: a head mean, and rollout (a full-matrix method) from the last answer token.
  await answer.nth(11).click();
  await chooseMethod(page, 'attention.mean_heads/captured_prediction_v1');
  await expect.poll(() => analyses.at(-1)).toMatchObject({method_id: 'attention.mean_heads', parameters: {step: 11}});
  await chooseMethod(page, 'attention.rollout/residual_mean_square_v1');
  await expect.poll(() => analyses.at(-1)).toMatchObject({method_id: 'attention.rollout', parameters: {step: 11}});
  expect(analyses.at(-1)?.source_index).toBeUndefined();
  await expect.poll(() => api.latest('attention.rollout', 'analysis')?.result.row?.step).toBe(11);
  await expect.poll(() => patches(page)).toEqual(expectedPatches(api.latest('attention.rollout', 'analysis')!.result.row!));
  await chooseMethod(page, 'attention.row/captured_weights_v1');

  // The full prompt: every answer token is a button, and none says it was not captured.
  await chooseTokenView(page, 'Prompt');
  const text = promptTokens(page).locator('.docattentionlens-prompt-tokens-text');
  await expect(text.locator('button.docattentionlens-prompt-token.is-answer')).toHaveCount(12);
  await expect(text.locator('span.docattentionlens-prompt-token.is-answer')).toHaveCount(0);
  await expect(text.locator('[title*="not captured"]')).toHaveCount(0);
  await text.locator(`button[data-token-index="${last.predicted_index}"]`).click();
  await expect(page.getByTestId('prediction-alignment').first()).toContainText('Step 11 →');
  expect(api.unexpected).toEqual([]);
});

test('a long answer scrolls in the source line and keeps the selected answer token in view', async ({page}) => {
  const capture = longAnswerCapture({tokens: 24});
  const api = await mockAttentionApi(page, {generationCapture: capture});
  await page.setViewportSize({width: 375, height: 812});
  await page.goto(`/attention-tool?generation=${F.GENERATION_ID}`);
  await expect(page.getByTestId('attention-patch').first()).toBeVisible();
  const row = page.locator('.docattentionlens-source-row');
  const answer = answerChips(page);
  await expect(answer).toHaveCount(24);
  // The row is wider than the phone: it scrolls sideways, and the page does not.
  expect(await row.evaluate(r => r.scrollWidth > r.clientWidth)).toBe(true);
  expect(await page.evaluate(() => document.documentElement.scrollWidth <= innerWidth + 1)).toBe(true);
  await expect(answer.nth(23)).not.toBeInViewport();
  // The keys walk to the last answer token; the row follows it.
  for (let k = 1; k < 24; k++) await stepTo(page, '.', k);
  await expect(answer.nth(23)).toBeInViewport();
  expect(await row.evaluate(r => r.scrollLeft)).toBeGreaterThan(0);
  expect(api.unexpected).toEqual([]);
});

test('an ordinary capture keeps eight answer steps; later answer tokens say why they cannot be chosen', async ({page}) => {
  const capture = longAnswerCapture({tokens: 12, retained: 8, fullAttention: false});
  const api = await mockAttentionApi(page, {generationCapture: capture});
  const steps = rowSteps(page);
  await page.setViewportSize({width: 1440, height: 900});
  await page.goto(`/attention-tool?generation=${F.GENERATION_ID}`);
  await expect(page.getByTestId('attention-patch').first()).toBeVisible();
  await openCaptureMenu(page);
  await expect(page.getByTestId('capture-coverage')).toContainText('8 captured predictions');
  await expect(page.getByTestId('full-attention-coverage')).toHaveCount(0);
  await page.keyboard.press('Escape');

  // The source line lists the eight captured steps; `.` stops at the eighth.
  const answer = answerChips(page);
  await expect(answer).toHaveCount(8);
  await answer.nth(0).click();
  for (let k = 1; k < 8; k++) await stepTo(page, '.', k);
  await page.keyboard.press('.');
  await expect(answer.nth(7)).toHaveAttribute('aria-pressed', 'true');
  expect(Math.max(...steps)).toBe(7);

  // In the full prompt, answer tokens after step 8 are not buttons and explain themselves.
  await chooseTokenView(page, 'Prompt');
  const text = promptTokens(page).locator('.docattentionlens-prompt-tokens-text');
  await expect(text.locator('button.docattentionlens-prompt-token.is-answer')).toHaveCount(8);
  const later = text.locator('span.docattentionlens-prompt-token.is-answer');
  await expect(later).toHaveCount(4);
  for (const token of await later.all())
    await expect(token).toHaveAttribute('title', /not captured: attention rows are kept for the first 8 prediction steps only · a capture with Save full attention matrices keeps every answer token/);
  expect(api.unexpected).toEqual([]);
});

test('a full-attention capture made before every answer token was kept names its eight captured steps', async ({page}) => {
  const capture = longAnswerCapture({tokens: 12, retained: 8, fullAttention: true});
  const api = await mockAttentionApi(page, {generationCapture: capture});
  await page.setViewportSize({width: 1440, height: 900});
  await page.goto(`/attention-tool?generation=${F.GENERATION_ID}`);
  await expect(page.getByTestId('attention-patch').first()).toBeVisible();
  await openCaptureMenu(page);
  await expect(page.getByTestId('capture-coverage')).toContainText('Attention rows for the first 8 answer tokens (8 of 12 captured)');
  await expect(page.getByTestId('full-attention-coverage')).toContainText('first 8 answer tokens');
  await page.keyboard.press('Escape');
  await expect(answerChips(page)).toHaveCount(8);
  await chooseTokenView(page, 'Prompt');
  const later = promptTokens(page).locator('.docattentionlens-prompt-tokens-text span.docattentionlens-prompt-token.is-answer');
  await expect(later).toHaveCount(4);
  await expect(later.first()).toHaveAttribute('title', /not captured: attention rows are kept for the first 8 prediction steps only · this full-attention capture predates keeping every answer token/);
  expect(api.unexpected).toEqual([]);
});
