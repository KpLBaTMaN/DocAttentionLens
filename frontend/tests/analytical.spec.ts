import { openTool, showElements, showEvidence, showRetrievalScores, expectRowTotals, withDisplaySettings, setFit, setDetails, openDisplayOptions, closeDisplaySettings, caseDetails, filmstrip, openDetailsTab, detailsTab, saveInvestigation, evidenceTab } from './workspaceHelpers';
import { test, expect, type Page } from '@playwright/test';
import { readFileSync } from 'node:fs';
import { findGenerationCapture, requireSaved, type SavedGeneration } from './savedEvidence';

/** Supporting retrieval scores exist only for captures acquired from retrieval, not from directly supplied pages. */
const fromRetrieval = (c: SavedGeneration) => Boolean(c.retrieval_capture_id);

/** The newest saved generator capture meeting `required`, preferring a two-page context; skips the test when none exists. */
async function savedGeneration(page: Page, required = fromRetrieval, need = 'acquired from a retrieval capture') {
  const capture = await findGenerationCapture(page.request, required, c => c.context.pages.length === 2);
  requireSaved(!capture, `Requires a saved generator capture ${need}; acquire one with scripts/generation_smoke.py`);
  return capture!;
}
/**
 * Scrolling workspace contract: the window may scroll vertically but never horizontally, and Fit page
 * places the whole page and its legend on the first screen.
 */
async function fits(page: Page) {
  await page.evaluate(() => window.scrollTo(0, 0));
  await expect.poll(() => page.evaluate(() => document.documentElement.scrollWidth <= innerWidth + 1)).toBe(true);
  expect(await page.getByTestId('page-canvas').first().evaluate(e => e.clientHeight)).toBeGreaterThan(100);
  await expect.poll(() => page.locator('.docattentionlens-canvas-legend').evaluate(e => e.getBoundingClientRect().bottom <= innerHeight + 1)).toBe(true);
}

/** The input view (original, consumed or both) is in Details › Display (the legend's Display button). */
async function imageView(page: Page) {
  const value = await (await openDisplayOptions(page)).getByLabel('Image view').inputValue();
  await closeDisplaySettings(page);
  return value;
}

/** With the window scrolled past the page, the top bar, source line, scrubber, method sentence and legend stay in view. */
async function instrumentsPinned(page: Page) {
  await setFit(page, 'width');
  await page.evaluate(() => {const canvas = document.querySelector('.docattentionlens-canvas')!.getBoundingClientRect(); window.scrollBy(0, canvas.bottom - innerHeight + 1);});
  await expect.poll(() => page.evaluate(() => scrollY)).toBeGreaterThan(100);
  for (const instrument of [page.getByRole('button',{name:'Save investigation',exact:true}), page.getByLabel('Attention layer',{exact:true}),
    page.getByRole('button',{name:'Source line options',exact:true}), page.getByTestId('display-scale'), page.getByTestId('image-share'), page.getByTestId('row-summary')]) await expect(instrument).toBeInViewport();
  await setFit(page, 'page');
  await page.evaluate(() => window.scrollTo(0, 0));
}

test('both tasks fit laptop and desktop viewports with persistent output, ranks and linked details', async ({page}) => {
  const capture = await savedGeneration(page);
  await page.goto('/attention-tool?generation=' + capture.capture_id);
  await expect(page.getByTestId('attention-patch').first()).toBeVisible({timeout: 20_000});
  for (const [width,height] of [[1366,768],[1440,900],[1920,1080]]) {
    await page.setViewportSize({width,height});
    await fits(page);
    // The answer tokens are in the source line above the page.
    await expect(page.getByRole('group',{name:'Answer tokens'})).toBeInViewport();
    // Two supplied pages: the filmstrip lists them beside the page.
    await expect(filmstrip(page)).toBeInViewport();
    await expect(filmstrip(page).locator('[data-page-id]')).toHaveCount(capture.context.pages.length + capture.context.omitted_page_ids.length);
    await expect(page.getByTestId('display-scale')).toBeInViewport();
    await instrumentsPinned(page);
    await showElements(page);
    await page.getByTestId('outline-element').first().click();
    await expect(page.getByTestId('element-attention-evidence')).toBeInViewport();
    // The method's own output is in Details › Result; the layer scrubber stays in view beside it.
    await showEvidence(page, 'Complete causal row');
    await expect(page.getByRole('tablist',{name:'Evidence view'})).toBeInViewport();
    await expect(page.getByLabel('Attention layer',{exact:true})).toBeInViewport();
    await setDetails(page, false);
    await showRetrievalScores(page);
    await expect(page.getByTestId('similarity-patch').first()).toBeVisible();
    // Retrieval candidates are ranked in the filmstrip; the winner ledger is in Details › Result.
    await expect(filmstrip(page).locator('[data-page-id]').first()).toContainText('#1');
    await expect(evidenceTab(page,'Query winner ledger')).toBeInViewport();
    await setDetails(page, false);
    // MaxSim compares the query alone: no answer tokens in the source line.
    await expect(page.getByRole('group',{name:'Answer tokens'})).toHaveCount(0);
    await fits(page);
    await page.getByRole('button',{name:'← Back to generation',exact:true}).click();
  }
  await page.setViewportSize({width:1440,height:900});
  await page.screenshot({path:'../var/screenshots/analytical-generation.png'});
  await showRetrievalScores(page);
  await page.screenshot({path:'../var/screenshots/analytical-retrieval.png'});
});

test('older saved investigations reopen, name their role and light theme preserves evidence', async ({page}) => {
  const metadata=JSON.parse(readFileSync('../var/verification/generation-smoke.json','utf8'));
  await page.setViewportSize({width:1440,height:900});
  await page.goto('/attention-tool/'+metadata.investigation_id);
  await expect(page.getByRole('group',{name:'Answer tokens'})).toBeVisible();
  expect(await imageView(page)).toBe('original');
  // The role lives in the model chip; there are no task tabs.
  await expect(page.getByTestId('capture-menu').locator('.docattentionlens-capture-role')).toHaveText('Generation');
  await expect(page.getByRole('tab',{name:'Generation',exact:true})).toHaveCount(0);
  await page.getByRole('button',{name:'Switch to White appearance',exact:true}).click();
  await expect(page.locator('.docattentionlens-app')).toHaveAttribute('data-theme','light');
  await expect(page.getByTestId('attention-patch').first()).toHaveAttribute('fill','#d99a36');
  await expect(page.getByRole('group',{name:'Answer tokens'})).toBeInViewport();
  await fits(page);
  await page.screenshot({path:'../var/screenshots/analytical-light.png'});
  await openTool(page, 'Layout experiment');
  await page.getByText('Prepare a layout experiment on this page',{exact:true}).click();
  await page.getByLabel('Experiment name').fill('Retained drawer draft');
  await page.getByRole('button',{name:'Close panel',exact:true}).click();
  await openTool(page, 'Layout experiment');
  await expect(page.getByLabel('Experiment name')).toHaveValue('Retained drawer draft');
});

test('captured tokens, fixed scales, linked input comparison and saved panel preferences retain scientific values', async ({page}) => {
  const capture = await savedGeneration(page, c => fromRetrieval(c) && c.layer_count > 35 && c.query_head_count > 15, 'acquired from a retrieval capture with layer 35 and head 15');
  await page.setViewportSize({width:1440,height:900});
  await page.goto('/attention-tool?generation=' + capture.capture_id);
  await expect(page.getByTestId('attention-patch').first()).toBeVisible({timeout: 20_000});
  await showElements(page);
  await page.getByTestId('outline-element').first().click();
  const selected = await page.getByTestId('selected-element').textContent();
  const step = capture.steps[capture.steps.length - 1];
  // Prediction tokens are a view of Details › Result.
  await showEvidence(page,'Answer and prediction tokens');
  await page.getByRole('button',{name:`Inspect prediction ${step.step}: ${step.predicted_piece}`,exact:true}).click();
  await setDetails(page,false);
  await page.getByLabel('Attention layer',{exact:true}).selectOption('35');
  await page.getByLabel('Attention head',{exact:true}).selectOption('15');
  const row = await page.request.get(`/api/v1/generation/captures/${capture.capture_id}/rows?step=${step.step}&layer=35&head=15`).then(r=>r.json());
  await expectRowTotals(page,row.total_mass.toFixed(6));
  await withDisplaySettings(page,async drawer=>{
    await drawer.getByLabel('Heatmap scale',{exact:true}).selectOption('fixed');
    await drawer.getByLabel('Fixed scale maximum',{exact:true}).fill('0.01');
  });
  await expect(page.getByTestId('display-scale')).toContainText('0.010000');
  await expectRowTotals(page,row.total_mass.toFixed(6));
  await (await openDisplayOptions(page)).getByLabel('Image view').selectOption('split');
  await closeDisplaySettings(page);
  await expect(page.getByTestId('page-canvas')).toHaveCount(2);
  await expect(page.getByTestId('attention-patch')).toHaveCount(capture.context.pages[0].position_indices.length*2);
  await page.getByRole('button',{name:'Zoom in',exact:true}).click();
  const position = capture.positions[capture.context.pages[0].position_indices[0]];
  for(const canvas of await page.getByTestId('page-canvas').all()) {
    const b = await canvas.evaluate((e,index) => {
      const image=e.querySelector('image')!.getBoundingClientRect();
      const patch=e.querySelector(`[data-position-index="${index}"]`)!.getBoundingClientRect();
      return {image:{x:image.x,y:image.y,width:image.width},patch:{x:patch.x,y:patch.y,width:patch.width}};
    },position.index);
    const scale=b.image.width/capture.context.pages[0].source_width;
    expect(Math.abs(b.patch.x-b.image.x-position.bbox![0]*scale)).toBeLessThan(1);
    expect(Math.abs(b.patch.y-b.image.y-position.bbox![1]*scale)).toBeLessThan(1);
  }
  await setFit(page,'page');
  await setDetails(page,true);
  await page.getByRole('separator',{name:'Resize Details'}).press('ArrowLeft');
  await showRetrievalScores(page);
  // MaxSim's query position is its source, in Details › Method.
  await openDetailsTab(page,'Method');
  await page.getByLabel('Overlay query position').selectOption('1');
  await page.getByRole('button',{name:'← Back to generation',exact:true}).click();
  await showElements(page);
  await expect(page.getByTestId('selected-element')).toHaveText(selected!);
  await openDetailsTab(page,'Method');
  await expect(page.getByLabel('Attention source',{exact:true})).toHaveValue(`step:${step.step}`);
  await openTool(page,'Notes & name');
  await page.getByLabel('Investigation name').fill('Analytical workspace browser verification');
  // Details left open on Result saves as an open inspector and an open evidence dock.
  await detailsTab(page,'Result').click();
  await saveInvestigation(page);
  await expect(page).toHaveURL(/investigation_/);
  await page.reload();
  await expect(page.getByTestId('page-canvas')).toHaveCount(2);
  expect(await imageView(page)).toBe('split');
  await expect(page.getByTestId('display-scale')).toContainText('0.010000');
  await expect(detailsTab(page,'Result')).toHaveAttribute('aria-selected','true');
  await openDetailsTab(page,'Method');
  await expect(page.getByLabel('Attention source',{exact:true})).toHaveValue(`step:${step.step}`);
  await expect(page.getByRole('separator',{name:'Resize Details'})).toHaveAttribute('aria-valuenow','376');
  await showRetrievalScores(page);
  await openDetailsTab(page,'Method');
  await expect(page.getByLabel('Overlay query position')).toHaveValue('1');
  await page.getByRole('button',{name:'← Back to generation',exact:true}).click();
  await page.screenshot({path:'../var/screenshots/analytical-input-comparison.png'});
  await page.setViewportSize({width:390,height:844});
  await openDetailsTab(page,'Method');
  await expect(page.getByLabel('Attention source',{exact:true})).toBeVisible();
  expect(await page.evaluate(()=>document.documentElement.scrollWidth<=innerWidth+1)).toBe(true);
  await page.screenshot({path:'../var/screenshots/analytical-mobile.png'});
});

test('acquisition draft and job tracking survive closing drawers and switching tasks (transport simulation)', async ({page}) => {
  const capture = await savedGeneration(page);
  let completed=false;
  await page.route('**/api/v1/generation/plan',async route=>{const response=await route.fetch();const body=await response.json();await route.fulfill({json:{...body,available:true}});});
  await page.route('**/api/v1/generation/runs',route=>route.fulfill({status:202,json:{job_id:'job_workspace_transport_check'}}));
  await page.route('**/api/v1/jobs/job_workspace_transport_check',route=>route.fulfill({json:{state:completed?'completed':'running',output_resource_id:completed?capture.capture_id:null}}));
  // A generation investigation of the retrieval capture (no generator capture yet) opens on the supporting
  // retrieval scores; the top bar's pill returns to generation, which acquires one.
  await page.goto('/attention-tool?capture='+capture.retrieval_capture_id+'&role=generation');
  await expect(page.getByTestId('capture-menu').locator('.docattentionlens-capture-role')).toHaveText('Retrieval');
  await page.getByRole('button',{name:'← Back to generation',exact:true}).click();
  await expect(page.getByRole('button',{name:'← Back to generation',exact:true})).toHaveCount(0);
  await openTool(page, 'Open / acquire evidence');
  const checkboxes=page.getByRole('dialog').locator('.docattentionlens-candidate-picker input');
  await expect(checkboxes.first()).toBeVisible();
  await checkboxes.nth(1).check();
  const contextDraft=await checkboxes.evaluateAll(es=>es.map(e=>(e as HTMLInputElement).checked));
  await page.getByRole('button',{name:'Close panel',exact:true}).click();
  await openTool(page, 'Open / acquire evidence');
  expect(await checkboxes.evaluateAll(es=>es.map(e=>(e as HTMLInputElement).checked))).toEqual(contextDraft);
  await page.getByRole('button',{name:'Generate answer and capture attention',exact:true}).click();
  await page.getByRole('button',{name:'Close panel',exact:true}).click();
  await expect(page.getByRole('link',{name:'Job details',exact:true})).toBeVisible();
  completed=true;
  await expect(page.getByRole('link',{name:'Job details',exact:true})).toHaveCount(0,{timeout:15000});
  await expect(page.getByTestId('capture-menu').locator('.docattentionlens-capture-role')).toHaveText('Generation');
  await expect((await caseDetails(page)).getByTestId('generated-answer')).toHaveText(capture.answer);
});
