import { chooseTokenView, clickPage, expectRowTotals, withPanelHidden, detailsPanel, sourceOptions, closeSourceOptions, promptTokens, openTemplateRuns, openDetailsTab } from './workspaceHelpers';
import { test, expect, type Page } from '@playwright/test';

/** The source line's notes (how to choose a source, the token scale) are in its ⋯. */
async function expectGuide(page:Page, text:RegExp) {
  await expect((await sourceOptions(page)).getByText(text)).toBeVisible();
  await closeSourceOptions(page);
}

async function retrieverCapture(page: Page) {
  const records = await page.request.get('/api/v1/retrieval/captures?limit=25').then(r => r.json());
  const capture = records.items.find((c: {attention_encodings?: unknown[]}) => c.attention_encodings?.length);
  expect(capture).toBeTruthy();
  return capture;
}

test('query tokens and page patches select the saved retriever source position', async ({page}) => {
  const capture = await retrieverCapture(page);
  await page.goto(`/attention-tool?capture=${capture.capture_id}&role=retrieval`);
  const panel = await detailsPanel(page, 'Retrieval panel');
  const source = panel.getByLabel('Attention source position');
  await expectRowTotals(page, '', 'retriever-attention-mass');

  // Document encoding: a plain click on the page makes the patch under it the source. There is no
  // picking mode or direction control, and a click never selects the layout element under the patch.
  await expect(page.getByRole('button', {name: /Pick patch|What attends to it|What it attends to/})).toHaveCount(0);
  await expect(page.getByTestId('page-canvas')).toHaveClass(/is-pickable/);
  await clickPage(page, .5, .5);
  await expect(source).not.toHaveValue('0');
  const picked = await source.inputValue();
  await expect(source.locator(`option[value="${picked}"]`)).toHaveText(`Patch ${picked}`);
  await expect(page.locator(`[data-testid=attention-patch][data-position-index="${picked}"]`)).toHaveAttribute('stroke', '#39a9e0', {timeout: 20_000});
  await expect(page.locator('rect.docattentionlens-box.is-selected')).toHaveCount(0);
  await clickPage(page, .3, .3);
  await expect(source).not.toHaveValue(picked);

  // Template tokens in the document encoding are selectable sources too; they are shown in the full prompt.
  await chooseTokenView(page, 'Prompt');
  await openTemplateRuns(page);
  const prompt = promptTokens(page).locator('.docattentionlens-prompt-segment[data-part="template"]').getByTestId('token-chip');
  const template = await prompt.first().getAttribute('data-token-index');
  await prompt.first().click();
  await expect(source).toHaveValue(template ?? '');
  // Without the full prompt, the source line names a template source in its own chip.
  await chooseTokenView(page, 'Parts');
  await expect(page.getByTestId('source-chip')).toContainText(`template`);
  await expect(page.getByTestId('source-chip')).toContainText(`position ${template}`);

  // Query encoding: click a query word, then step with the keyboard.
  await panel.getByRole('button',{name:'Question',exact:true}).click();
  const tokens = page.getByRole('group', {name: 'Query tokens'});
  // The first question word of whichever saved capture is newest; captures differ in wording.
  const firstWord = capture.query_positions.find((p: {role: string}) => p.role === 'query_text').index;
  const word = tokens.locator(`[data-token-index="${firstWord}"]`);
  await word.click();
  const index = await word.getAttribute('data-token-index');
  await expect(source).toHaveValue(index!);
  await expect(word).toHaveAttribute('aria-pressed', 'true');
  await word.press('ArrowRight');
  await expect(source).toHaveValue(String(Number(index) + 1));
  await withPanelHidden(page, () => expect(page.getByTestId('retriever-attention-context')).toContainText(`Source ${Number(index) + 1}`));
  // Heat on query tokens shares the page heatmap colour.
  await expect.poll(() => tokens.getByTestId('token-chip').evaluateAll(chips => chips.filter(c => (c as HTMLElement).style.background.includes('217, 154, 54')).length), {timeout: 20_000}).toBeGreaterThan(0);
  await expectGuide(page, /Token scale 0 →/);

  // MaxSim: query tokens select the similarity row, labelled as similarity.
  await page.getByLabel('Retrieval evidence type').selectOption('similarity');
  const similarity = page.getByRole('group', {name: 'Query tokens'});
  await expectGuide(page, /MaxSim similarity, not attention/);
  const second = similarity.getByTestId('token-chip').nth(4);
  await second.click();
  await expect(page.getByLabel('Overlay query position', {exact: true})).toHaveValue((await second.getAttribute('data-token-index'))!);
  await openDetailsTab(page, 'Result');
  await expect(page.getByTestId('retrieval-winner')).toBeVisible();
});

test('generator answer tokens select prediction steps and question tokens show their attention', async ({page}) => {
  const records = await page.request.get('/api/v1/generation/captures?limit=25').then(r => r.json());
  const capture = records.items.find((c: {steps: unknown[]}) => c.steps.length > 1);
  expect(capture).toBeTruthy();
  await page.goto(`/attention-tool?generation=${capture.capture_id}`);
  await expectRowTotals(page);
  await chooseTokenView(page, 'Parts');
  await detailsPanel(page);
  const answer = page.getByRole('group', {name: 'Answer tokens'}).getByTestId('token-chip');
  await expect(answer).toHaveCount(capture.steps.length);
  await answer.nth(1).click();
  await expect(page.getByLabel('Attention source', {exact: true})).toHaveValue('step:1');
  await withPanelHidden(page, () => expect(page.getByTestId('prediction-alignment')).toContainText('Step 1'));
  const question = page.getByRole('group', {name: 'Question tokens'});
  // Captures with retained question rows make question tokens selectable; older captures say why they are not.
  await expectGuide(page, /Click a question or answer token to see what it attends to|predates question-token rows/);
  await expect.poll(() => question.locator('.docattentionlens-token').evaluateAll(chips => chips.filter(c => (c as HTMLElement).style.background.includes('217, 154, 54')).length), {timeout: 20_000}).toBeGreaterThan(0);
});
