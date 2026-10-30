import { chooseTokenView, clickPage, expectRowTotals, withPanelHidden, detailsPanel, currentPage, sourceOptions, closeSourceOptions } from './workspaceHelpers';
import { test, expect, type Locator, type Page } from '@playwright/test';
import { requireSaved } from './savedEvidence';

async function retrieverCapture(page: Page) {
  const records = await page.request.get('/api/v1/retrieval/captures?limit=25').then(r => r.json());
  const capture = records.items.find((c: {attention_encodings?: unknown[]}) => c.attention_encodings?.length);
  expect(capture).toBeTruthy();
  return capture;
}

const heated = (chips: Locator, rgb: string) =>
  chips.evaluateAll((items, colour) => items.filter(c => (c as HTMLElement).style.background.includes(colour as string)).length, rgb);

type Source = {source_index: number; kind: string; value: number};
const strongest = (sources: Source[]) => sources.reduce((a, b) => a.value > b.value ? a : b);

test('clicking chooses what is shown: retriever positions show their own row, MaxSim patches their token similarity', async ({page}) => {
  const capture = await retrieverCapture(page);
  await page.goto(`/attention-tool?capture=${capture.capture_id}&role=retrieval`);
  await expectRowTotals(page, '', 'retriever-attention-mass');
  // Every retriever position has its own row: a patch click shows what it attends to, never a column.
  const panel = await detailsPanel(page, 'Retrieval panel');
  await clickPage(page, .5, .6);
  const source = panel.getByLabel('Attention source position');
  await expect(source).not.toHaveValue('0');
  await expect(page.getByTestId('column-context')).toHaveCount(0);
  await expect(page.getByTestId('display-scale')).toContainText('excluding the source patch');

  // MaxSim: a patch's signed similarity to every query token, never labelled attention.
  await page.getByLabel('Retrieval evidence type').selectOption('similarity');
  await clickPage(page, .62, .7);
  await expect(page.getByTestId('column-context')).toContainText('similarity to every query token', {timeout: 20_000});
  const patch = Number((await page.getByTestId('column-context').textContent())!.match(/Patch (\d+)/)![1]);
  const pageId = await currentPage(page);
  const similarity = await page.request.get(`/api/v1/retrieval/captures/${capture.capture_id}/columns?page_id=${pageId}&position_index=${patch}`).then(r => r.json());
  const chips = page.getByRole('group', {name: 'Query tokens'}).getByTestId('token-chip');
  const first = similarity.similarities.findIndex((v: number | null) => v !== null);
  await expect(chips.nth(first)).toHaveAttribute('title', new RegExp(`similarity ${similarity.similarities[first].toFixed(6)}`));
  await expect.poll(async () => await heated(chips, '176, 128, 209') + await heated(chips, '36, 126, 213')).toBeGreaterThan(0);
  // The source line names the clicked patch with a clear button; its ⋯ explains the reading.
  await expect(page.getByTestId('source-column')).toContainText(`Similarity of patch ${patch} to each query token`);
  await expect((await sourceOptions(page)).getByText(/signed MaxSim dot products, not attention/)).toBeVisible();
  await closeSourceOptions(page);
  // A query token click returns to that token's similarity to every patch.
  await chips.nth(first).click();
  await expect(page.getByTestId('column-context')).toHaveCount(0);
  await expect(page.getByTestId('similarity-patch').first()).toBeVisible();
});

test('question-token sources and the generator column use retained question rows and survive reopening', async ({page}) => {
  const records = await page.request.get('/api/v1/generation/captures?limit=25').then(r => r.json());
  const capture = records.items.find((c: {source_rows?: unknown[]}) => c.source_rows?.length);
  requireSaved(!capture, 'Needs a generator capture with question-token rows (schema 0.5.0)');
  const question = capture.source_rows[Math.floor(capture.source_rows.length / 2)];
  await page.goto(`/attention-tool?generation=${capture.capture_id}`);
  await expectRowTotals(page);
  const panel = await detailsPanel(page);
  await chooseTokenView(page, 'Parts');
  const questionChip = page.getByRole('group', {name: 'Question tokens'}).locator(`[data-token-index="${question.source_index}"]`);
  await questionChip.click();
  await expect(panel.getByLabel('Attention source', {exact: true})).toHaveValue(`token:${question.source_index}`);
  await withPanelHidden(page, () => expect(page.getByTestId('prediction-alignment')).toContainText(`(position ${question.source_index})`));
  // The page heatmap is the question token's own row.
  const row = await page.request.get(`/api/v1/generation/captures/${capture.capture_id}/rows?layer=0&head=0&source_index=${question.source_index}`).then(r => r.json());
  await expectRowTotals(page, row.total_mass.toFixed(6));
  await expect(page.getByTestId('attention-patch').first()).toBeVisible();
  // An answer token returns to prediction steps.
  await page.getByRole('group', {name: 'Answer tokens'}).getByTestId('token-chip').first().click();
  await expect(panel.getByLabel('Attention source', {exact: true})).toHaveValue(/^step:/);
  await questionChip.click();

  // A patch has no row of its own, so clicking it shows which answer and question tokens attend to it.
  await clickPage(page, .5, .5);
  await expect(page.getByTestId('column-context')).toContainText('image patch', {timeout: 20_000});
  const target = Number((await page.getByTestId('column-context').textContent())!.match(/position (\d+)/)![1]);
  const column = await page.request.get(`/api/v1/generation/captures/${capture.capture_id}/received?target_index=${target}&layer=0&head=0`).then(r => r.json());
  const questionSources = column.sources.filter((s: Source) => s.kind === 'question');
  expect(questionSources.length).toBeGreaterThan(0);
  const chip = page.getByRole('group', {name: 'Question tokens'}).locator(`[data-token-index="${strongest(questionSources).source_index}"]`);
  await expect(chip).toHaveAttribute('title', /attention weight/);
  await expect.poll(() => chip.evaluate(c => (c as HTMLElement).style.background)).toContain('217, 154, 54');
  // Column mode is named in the source line ("What attends to patch N") with a clear button.
  await expect(page.getByTestId('source-column')).toContainText(`What attends to patch ${target}`);

  await page.getByRole('button', {name: 'Save investigation', exact: true}).click();
  await expect(page).toHaveURL(/investigation_/);
  await page.reload();
  await expect(page.getByTestId('column-context')).toContainText(`position ${target}`, {timeout: 20_000});
  // A question token click shows that token's row again.
  await questionChip.click();
  await expect(page.getByTestId('column-context')).toHaveCount(0);
  await expect(panel.getByLabel('Attention source', {exact: true})).toHaveValue(`token:${question.source_index}`);
});
