import { displaySettingsButton, openCaptureMenu, withDisplaySettings, detailsPanel, methodChoice, chooseMethod, pickerOptions, openDetailsTab, showPageList, methodBrowse } from './workspaceHelpers';
import type { Page } from '@playwright/test';

/** A method's entry in the picker (searched by its id), for its availability. */
async function pickerEntry(page: Page, key: string) {
  return (await pickerOptions(page, key.split('/')[0])).and(page.locator(`[data-method-key="${key}"]`));
}
import { test, expect } from '@playwright/test';

test('a case opened for generation can select saved attention directly in the shared panel', async ({page}) => {
  const capture = (await page.request.get('/api/v1/generation/captures').then(r => r.json())).items[0];
  expect(capture, 'Requires a real saved attention capture').toBeTruthy();
  let acquisitions = 0;
  await page.route(/\/api\/v1\/(generation|retrieval)\/runs$/, route => {acquisitions++; return route.abort();});
  await page.goto('/attention-tool?case=' + capture.case_id);
  const panel = await detailsPanel(page);
  const picker = methodChoice(page);
  await expect(page.getByTestId('capture-menu').locator('.docattentionlens-capture-role')).toHaveText('Generation');
  await expect(picker).toBeVisible();
  await expect(await pickerEntry(page, 'attention.mean_heads/captured_prediction_v1')).toHaveAttribute('aria-disabled', 'true');
  await page.keyboard.press('Escape');
  await expect(panel.getByLabel('Attention opacity', {exact:true})).toHaveCount(0);
  await expect(page.getByLabel('Attention opacity', {exact:true})).toHaveCount(0);
  await expect(displaySettingsButton(page)).toBeDisabled();
  await openCaptureMenu(page);
  const sources = page.getByLabel('Saved attention capture', {exact:true});
  await expect(sources.locator(`option[value="${capture.capture_id}"]`)).toHaveCount(1);
  await sources.selectOption(capture.capture_id);
  await sources.press('Escape');
  await expect(await pickerEntry(page, 'attention.mean_heads/captured_prediction_v1')).not.toHaveAttribute('aria-disabled', 'true');
  await page.keyboard.press('Escape');
  await expect(page.getByLabel('Attention layer', {exact:true})).toBeVisible();
  await expect(page.getByLabel('Attention head', {exact:true})).toBeVisible();
  await expect(page.getByTestId('attention-patch').first()).toBeVisible();
  await chooseMethod(page, 'attention.mean_heads/captured_prediction_v1');
  await openDetailsTab(page,'Result');
  const result = page.getByTestId('method-result');
  await expect(result).toHaveAttribute('data-analysis-id', /^analysis_/);
  const analysis = await page.request.get('/api/v1/methods/analyses/' + await result.getAttribute('data-analysis-id')).then(r => r.json());
  expect(analysis.generation_capture_id).toBe(capture.capture_id);
  expect(analysis.capture_sha256).toBe(capture.content_sha256);
  expect(acquisitions).toBe(0);
  await page.screenshot({path:'../var/screenshots/shared-attention-from-case.png'});
});

test('generation keeps attention methods, axes, parameters, context attention and saved selections accessible', async ({page}) => {
  const capture = (await page.request.get('/api/v1/generation/captures').then(r => r.json())).items[0];
  let acquisitions = 0;
  await page.route(/\/api\/v1\/(generation|retrieval)\/runs$/, route => {acquisitions++; return route.abort();});
  await page.goto('/attention-tool?generation=' + capture.capture_id);
  await detailsPanel(page);
  const picker = methodChoice(page);
  await expect(page.getByTestId('attention-patch').first()).toBeVisible({timeout: 20_000});
  await expect(methodBrowse(page)).toBeEnabled();
  await expect(page.getByLabel('Attention head', {exact:true})).toBeVisible();
  // Details › Run lists the attention to each supplied page (as does the filmstrip), with no lens to choose.
  await showPageList(page);
  await expect(page.getByRole('region', {name:'Generator attention to context'})).toBeVisible();
  await page.getByLabel('Attention head slider', {exact:true}).focus();
  await page.getByLabel('Attention head slider', {exact:true}).press('Tab');
  await expect(page.getByRole('region', {name:'Generation analysis'})).toBeVisible();
  await page.getByLabel('Attention layer', {exact:true}).selectOption('2');
  await expect(page.getByRole('region', {name:'Generation analysis'})).toBeVisible();
  await openDetailsTab(page,'Method');
  await expect(page.getByLabel('Attention source', {exact:true})).toHaveValue(/^step:/);
  await withDisplaySettings(page, drawer=>drawer.getByLabel('Attention opacity', {exact:true}).fill('25'));
  await chooseMethod(page, 'attention.topk_mass/captured_prediction_v1');
  await page.getByLabel('Top-k targets', {exact:true}).fill('7');
  await openDetailsTab(page,'Result');
  await expect(page.getByTestId('method-result').getByText('Top-7 attention mass', {exact:true})).toBeVisible();
  await openDetailsTab(page,'Method');
  await expect(page.getByLabel('Top-k targets', {exact:true})).toHaveValue('7');
  await expect(page.getByRole('region', {name:'Generation analysis'})).toBeVisible();
  await page.getByRole('button', {name:'Save investigation', exact:true}).click();
  await expect(page).toHaveURL(/investigation_/);
  await page.reload();
  await expect(page.getByRole('region', {name:'Generation analysis'})).toBeVisible();
  await expect(picker).toHaveAttribute('data-method-key', 'attention.topk_mass/captured_prediction_v1');
  await openDetailsTab(page,'Method');
  await expect(page.getByLabel('Top-k targets', {exact:true})).toHaveValue('7');
  await expect(page.getByLabel('Attention layer', {exact:true})).toHaveValue('2');
  const id=new URL(page.url()).pathname.split('/').at(-1)!;
  const saved=await page.request.get('/api/v1/investigations/'+id).then(r=>r.json());
  expect(saved.display_settings.attention_opacity).toBe(.25);
  expect(saved.view_state.generation_capture_id).toBe(capture.capture_id);
  await expect(page.getByRole('img', {name:/Original page .* with layout overlays/})).toBeVisible();
  await page.locator('.docattentionlens-details-body').evaluate(e => e.scrollTo({top:0}));
  await page.screenshot({path:'../var/screenshots/shared-attention-retrieval.png'});
  await page.setViewportSize({width:390, height:844});
  await openDetailsTab(page,'Result');
  await expect(picker).toBeInViewport();
  expect(await page.evaluate(() => document.documentElement.scrollWidth <= innerWidth + 1)).toBe(true);
  await page.screenshot({path:'../var/screenshots/shared-attention-mobile.png'});
  expect(acquisitions).toBe(0);
});

test('missing attention evidence explains requirements and opens the correct acquisition drawer for generation', async ({page}) => {
  const capture = (await page.request.get('/api/v1/generation/captures').then(r => r.json())).items[0];
  await page.route('**/api/v1/generation/captures?*', route => route.fulfill({json:{items:[], next_cursor:null}}));
  let acquisitions = 0;
  await page.route(/\/api\/v1\/(generation|retrieval)\/runs$/, route => {acquisitions++; return route.abort();});
  await page.goto('/attention-tool?case=' + capture.case_id);
  const panel = await detailsPanel(page);
  const picker = methodChoice(page);
  await expect(picker).toBeVisible();
  await expect(await pickerEntry(page, 'attention.entropy/shannon_row_v1')).toHaveAttribute('aria-disabled', 'true');
  await page.keyboard.press('Escape');
  await expect(panel.getByText(/Choose a saved attention capture to explore prediction steps, layers and heads/)).toBeVisible();
  await panel.getByRole('button', {name:'Acquire attention', exact:true}).click();
  await expect(page.getByRole('dialog').getByLabel('Saved generator capture', {exact:true})).toBeVisible();
  await expect(page.getByRole('region', {name:'Generation analysis'})).toBeVisible();
  await page.getByRole('dialog').getByRole('button', {name:'Close panel', exact:true}).click();
  // A new capture is configured from the model chip, beside capturing more evidence.
  await openCaptureMenu(page);
  await expect(page.getByRole('link', {name:'Configure new capture'})).toBeVisible();
  await expect(page.getByRole('button', {name:'Capture more evidence'})).toBeVisible();
  expect(acquisitions).toBe(0);
});
