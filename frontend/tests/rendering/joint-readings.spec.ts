import { expect, test } from '@playwright/test';
import * as F from '../../fixtures/attention';
import { mockAttentionApi } from './mockApi';
import { chooseTokenView, openTemplateRuns } from '../workspaceHelpers';
import { promptTokens, openDetailsTab, chooseDisplayView } from '../workspaceHelpers';

// Different token axes from the separate page/query passes, including trailing prompt tokens.
const jointPositions = [
  ...F.retrieverPositions,
  {index: 16, token_id: 80, piece: '▁hot', role: 'query_text', eligible: true, bbox: null},
  {index: 17, token_id: 81, piece: '▁patch', role: 'query_text', eligible: true, bbox: null},
  {index: 18, token_id: 82, piece: '<end>', role: 'visual_prompt', eligible: true, bbox: null},
];
const joint = {...F.retrievalCapture.attention_encodings![0], encoding_id: 'joint', pass_kind: 'joint' as const,
  causal: true, position_count: jointPositions.length, attention_mask: jointPositions.map(() => 1), positions: jointPositions,
  ownership: [...F.retrieverOwnership, {non_page: 1}, {non_page: 1}, {non_page: 1}]};

test('three readings preserve axes, select the joint question, and split token strips', async ({page}) => {
  const api = await mockAttentionApi(page, {retrievalCapture: {...F.retrievalCapture, attention_encodings: [...F.retrievalCapture.attention_encodings!, joint]}});
  const requests: {encoding_id: string; source_index: number}[] = [];
  page.on('request', r => {if (r.method() === 'POST' && r.url().endsWith('/attention')) requests.push(r.postDataJSON());});
  await page.setViewportSize({width: 1440, height: 900});
  await page.goto(`/attention-tool?capture=${F.RETRIEVAL_ID}&role=retrieval`);
  await openDetailsTab(page, 'Method');
  const readings = page.getByRole('group', {name: 'What the model read'});
  await expect(readings.getByRole('button', {name: 'Page', exact: true})).toHaveAttribute('aria-pressed', 'true');
  await page.getByLabel('Attention layer', {exact: true}).selectOption('1');
  await page.getByLabel('Attention head', {exact: true}).selectOption('2');
  await readings.getByRole('button', {name: 'Page + question', exact: true}).click();
  await expect(readings).toContainText(/excluded from the retrieval score/i);
  await expect(page.getByLabel('Attention source position')).toHaveValue('17');
  await expect(page.getByLabel('Attention layer', {exact: true})).toHaveValue('1');
  await expect(page.getByLabel('Attention head', {exact: true})).toHaveValue('2');
  // The source line shows the question tokens; the template tokens are in the full prompt.
  await chooseTokenView(page, 'Parts');
  const question = page.getByRole('group', {name: 'Question tokens', exact: true});
  await expect(question.getByTestId('token-chip')).toHaveCount(2);
  await expect(question.getByRole('button', {name: 'Token 17: patch', exact: true})).toHaveAttribute('aria-pressed', 'true');
  await expect(page.getByRole('group', {name: 'Template tokens', exact: true})).toHaveCount(0);
  await chooseTokenView(page, 'Prompt');
  await openTemplateRuns(page);
  const prompt = promptTokens(page).locator('.docattentionlens-prompt-segment[data-part="template"]');
  await expect(prompt.getByRole('button', {name: 'Token 16: hot', exact: true})).toHaveCount(0);
  await expect(prompt.getByTestId('token-chip').first()).toBeVisible();
  await chooseTokenView(page, 'Parts');
  await question.getByRole('button', {name: 'Token 16: hot', exact: true}).click();
  await expect.poll(() => requests.at(-1)).toMatchObject({encoding_id: 'joint', source_index: 16});
  await expect(page.getByTestId('attention-patch')).toHaveCount(F.PATCH_COUNT);
  // The token strip's colour scale is named in Details › Display › Scale.
  await openDetailsTab(page, 'Display');
  await chooseDisplayView(page, 'Scale');
  await expect(page.getByRole('complementary', {name: 'Retrieval panel', exact: true})).toContainText('Question & template tokens');
  await openDetailsTab(page, 'Method');
  await page.screenshot({path: '../var/screenshots/joint-readings-desktop.png'});
  await readings.getByRole('button', {name: 'Question', exact: true}).click();
  await expect.poll(() => requests.at(-1)?.encoding_id).toBe('query');
  await expect(page.getByTestId('attention-patch')).toHaveCount(0);
  await readings.getByRole('button', {name: 'Page', exact: true}).click();
  await expect.poll(() => requests.at(-1)?.encoding_id).toBe('document');
  await expect(question).toHaveCount(0);
  await page.setViewportSize({width: 390, height: 844});
  await openDetailsTab(page, 'Method');
  await readings.getByRole('button', {name: 'Page + question', exact: true}).click();
  await expect(readings).toBeInViewport();
  expect(await page.evaluate(() => document.documentElement.scrollWidth <= innerWidth + 1)).toBe(true);
  await page.screenshot({path: '../var/screenshots/joint-readings-mobile.png'});
  expect(api.unexpected).toEqual([]);
});

for (const legacyJoint of [false, true]) test(`legacy ${legacyJoint ? 'joint' : 'page'} capture shows the missing reading disabled, without a capture link`, async ({page}) => {
  const api = await mockAttentionApi(page, {retrievalCapture: {...F.retrievalCapture,
    attention_encodings: [F.retrievalCapture.attention_encodings![1], legacyJoint ? joint : F.retrievalCapture.attention_encodings![0]]}});
  let runs = 0;
  page.on('request', r => {if (/\/(retrieval|generation)\/runs$/.test(r.url())) runs++;});
  await page.goto(`/attention-tool?capture=${F.RETRIEVAL_ID}&role=retrieval`);
  // The readings are in Details › Method.
  await openDetailsTab(page, 'Method');
  const readings = page.getByRole('group', {name: 'What the model read'});
  const missing = readings.getByRole('button', {name: legacyJoint ? 'Page' : 'Page + question', exact: true});
  await expect(missing).toBeDisabled();
  await expect(missing).toHaveAttribute('title', /not saved in this older capture; new attention captures include it/);
  if (legacyJoint) await expect(page.getByLabel('Attention source position')).toHaveValue('17');
  // New captures always save every reading, so an older one offers no link to capture the missing one.
  await expect(readings).not.toContainText(/^(Page|Page \+ question) isn’t in this capture/m);
  await expect(readings.getByRole('button', {name: /^Capture (Page|Page \+ question) attention$/})).toHaveCount(0);
  expect(runs).toBe(0);
  expect(api.unexpected).toEqual([]);
});
