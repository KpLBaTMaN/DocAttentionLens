import { expect, test } from '@playwright/test';
import * as F from '../../fixtures/attention';
import { mockAttentionApi } from './mockApi';
import { chooseTokenView, detailsPanel, promptTokens, openDetailsTab } from '../workspaceHelpers';

// A three-reading capture whose Page + question pass also exists with the case's reference answer
// ("Top left") appended to the prompt: "hot patch\nAnswer: Top left".
const question = [
  {index: 16, token_id: 80, piece: '▁hot', role: 'query_text', eligible: true, bbox: null},
  {index: 17, token_id: 81, piece: '▁patch', role: 'query_text', eligible: true, bbox: null},
];
const tail = {index: 18, token_id: 82, piece: '<end>', role: 'visual_prompt', eligible: true, bbox: null};
const answerTail = [
  {index: 18, token_id: 83, piece: 'Answer', role: 'visual_prompt', eligible: true, bbox: null},
  {index: 19, token_id: 84, piece: ':', role: 'visual_prompt', eligible: true, bbox: null},
  {index: 20, token_id: 85, piece: '▁Top', role: 'answer_text', eligible: true, bbox: null},
  {index: 21, token_id: 86, piece: '▁left', role: 'answer_text', eligible: true, bbox: null},
  {...tail, index: 22},
];
const imagePiece = F.retrieverPositions[0].piece;
const encoding = (id: string, positions: typeof F.retrieverPositions, extra: object) => ({...F.retrievalCapture.attention_encodings![0],
  encoding_id: id, pass_kind: 'joint' as const, causal: true, position_count: positions.length, attention_mask: positions.map(() => 1), positions,
  ownership: positions.map((p, i) => p.bbox ? F.retrieverOwnership[i] : {non_page: 1}), ...extra});
const jointPositions = [...F.retrieverPositions, ...question, tail];
const answerPositions = [...F.retrieverPositions, ...question, ...answerTail];
const joint = encoding('joint', jointPositions, {prompt_text: `${imagePiece} hot patch<end>`});
const answered = encoding('joint_answer', answerPositions,
  {prompt_text: `${imagePiece} hot patch\nAnswer: Top left<end>`, answer_source: 'reference' as const, answer_text: 'Top left'});

test('the reference answer is an explicit prompt variant, shown in the prompt and opened on the last answer token', async ({page}) => {
  const api = await mockAttentionApi(page, {retrievalCapture: {...F.retrievalCapture,
    attention_encodings: [...F.retrievalCapture.attention_encodings!, joint, answered]}});
  const requests: {encoding_id: string; source_index: number}[] = [];
  page.on('request', r => {if (r.method() === 'POST' && r.url().endsWith('/attention')) requests.push(r.postDataJSON());});
  await page.setViewportSize({width: 1440, height: 900});
  await page.goto(`/attention-tool?capture=${F.RETRIEVAL_ID}&role=retrieval`);
  // Details › Method, or the method panel under the method bar in the three-column layout.
  const panel = await detailsPanel(page, 'Retrieval panel');
  const readings = page.getByRole('group', {name: 'What the model read'});
  const source = panel.getByLabel('Attention source position');

  // Page + question opens on its question-only prompt; the whole prompt is shown above the page, not in the panel.
  await readings.getByRole('button', {name: 'Page + question', exact: true}).click();
  const prompt = readings.getByRole('group', {name: 'Prompt', exact: true});
  await expect(prompt.getByRole('button', {name: 'Question only', exact: true})).toHaveAttribute('aria-pressed', 'true');
  await expect(page.getByTestId('joint-prompt')).toHaveCount(0);
  await readings.getByRole('button', {name: 'Show prompt above the page', exact: true}).click();
  await expect(promptTokens(page)).toBeVisible();
  const tokens = promptTokens(page), text = tokens.locator('.docattentionlens-prompt-tokens-text');
  await expect(tokens.getByTestId('prompt-image')).toContainText(`${F.PATCH_COUNT} patches`);
  await expect(text).toContainText('hot patch');
  await expect(tokens.getByRole('button', {name: 'Token 17: patch', exact: true})).toHaveAttribute('aria-pressed', 'true');
  await expect(text.locator('.is-answer')).toHaveCount(0);
  await expect(source).toHaveValue('17');
  await expect(panel.getByRole('group', {name: 'Source shortcuts'})).toHaveCount(0);

  // With the reference answer: it is labelled as an annotation, highlighted, and the source is the last answer token.
  await prompt.getByRole('button', {name: 'Question + reference answer', exact: true}).click();
  await expect.poll(() => requests.at(-1)).toMatchObject({encoding_id: 'joint_answer', source_index: 20});
  await expect(readings).toContainText('The answer is a case annotation added to the prompt');
  await expect(text.locator('.is-answer')).toHaveText(['Top', 'left']);
  await expect(text.locator('.is-last')).toHaveText('left');
  await expect(tokens.locator('.docattentionlens-prompt-tokens-key')).toContainText('reference answer');
  await expect(tokens.getByRole('button', {name: 'Token 20: Top', exact: true})).toHaveAttribute('aria-pressed', 'true');
  await expect(source.locator('option:checked')).toHaveText('Answer · “Top” (20) · predicts last answer token “left”');
  const shortcuts = panel.getByRole('group', {name: 'Source shortcuts'});
  await expect(shortcuts.getByRole('button', {name: 'Last answer token'})).toHaveAttribute('aria-pressed', 'true');
  // The source line shows question → reference answer and the page; template tokens stay in the full prompt.
  await chooseTokenView(page, 'Parts');
  await expect(page.getByRole('group', {name: 'Answer tokens · reference answer', exact: true}).getByTestId('token-chip')).toHaveCount(2);
  await expect(page.getByRole('group', {name: 'Question tokens', exact: true}).getByTestId('token-chip')).toHaveCount(2);
  await expect(page.getByRole('group', {name: 'Template tokens', exact: true})).toHaveCount(0);
  await expect(page.getByTestId('page-image-chip')).toContainText(`Page image · ${F.PATCH_COUNT} patches`);

  // The question word is one click away, and back again.
  await shortcuts.getByRole('button', {name: 'Last question word'}).click();
  await expect.poll(() => requests.at(-1)?.source_index).toBe(17);
  await shortcuts.getByRole('button', {name: 'Last answer token'}).click();
  await expect.poll(() => requests.at(-1)?.source_index).toBe(20);
  await expect(page.getByTestId('attention-patch')).toHaveCount(F.PATCH_COUNT);
  await page.screenshot({path: '../var/screenshots/joint-answer-desktop.png'});
  expect(api.unexpected).toEqual([]);
});

test('a capture without the answer variant offers to capture it, with the reference answer shown before running', async ({page}) => {
  const api = await mockAttentionApi(page, {retrievalCapture: {...F.retrievalCapture, attention_encodings: [...F.retrievalCapture.attention_encodings!, joint]}});
  let runs = 0;
  page.on('request', r => {if (/\/retrieval\/runs$/.test(r.url())) runs++;});
  await page.goto(`/attention-tool?capture=${F.RETRIEVAL_ID}&role=retrieval`);
  // The readings are in Details › Method.
  await openDetailsTab(page, 'Method');
  await page.getByRole('group', {name: 'What the model read'}).getByRole('button', {name: 'Page + question', exact: true}).click();
  await expect(page.getByRole('group', {name: 'Prompt', exact: true}).getByRole('button', {name: 'Question + reference answer', exact: true})).toBeDisabled();
  // Every attention capture includes Page + question; the reference-answer pass is the one opt-in.
  const plan = page.waitForRequest(r => /\/retrieval\/plan\?/.test(r.url()) && r.url().includes('capture_joint_attention=true') && r.url().includes('joint_answer=true'));
  await page.getByRole('button', {name: 'Capture Page + question attention with the reference answer', exact: true}).click();
  await plan;
  const acquisition = page.getByRole('region', {name: 'Retrieval acquisition'});
  await expect(acquisition).toContainText('Saves Question, Page and the diagnostic Page + question attention');
  const answer = page.getByRole('checkbox', {name: 'Also Page + question with the reference answer: “Top left”', exact: true});
  await expect(answer).toBeChecked();
  await expect(acquisition).toContainText('The answer comes from the case annotations');
  expect(runs).toBe(0);
  expect(api.unexpected).toEqual([]);
});
