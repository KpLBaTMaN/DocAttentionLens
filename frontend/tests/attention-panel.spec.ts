import { test, expect, type Locator } from '@playwright/test';
import { showElements, expectRowTotals, pageColouring, withDisplaySettings, methodChoice, chooseMethod, openMethodPicker, openTool, elementList, saveInvestigation, openDetailsTab, methodBrowse, closeMethodPicker } from './workspaceHelpers';
import { findGenerationCapture, requireSaved } from './savedEvidence';

async function setSlider(slider:Locator, value:number) {
  await slider.press('Home');
  for(let i=0;i<value;i++)await slider.press('ArrowRight');
}

test('an older row response cannot replace a newer head and layer selection', async ({page}) => {
  // A compact one-page capture at the standard image size keeps the race test independent of large captures.
  const capture = (await findGenerationCapture(page.request, c => c.layer_count > 2,
    c => c.context.pages.length === 1 && (c.context.image_token_limit ?? 512) === 512))!;
  expect(capture.layer_count).toBeGreaterThan(2);
  const expected = await page.request.get(`/api/v1/generation/captures/${capture.capture_id}/rows?step=0&layer=2&head=0`).then(r=>r.json());
  let started = false;
  let release!:()=>void;
  let finished!:()=>void;
  const gate = new Promise<void>(resolve=>{release=resolve;});
  const delivered = new Promise<void>(resolve=>{finished=resolve;});
  await page.route('**/generation/captures/*/rows?*', async route=>{
    if(new URL(route.request().url()).searchParams.get('layer') !== '1')return route.continue();
    const response = await route.fetch();
    started = true;
    await gate;
    await route.fulfill({response});
    finished();
  });
  await page.goto('/attention-tool?generation='+capture.capture_id);
  await expect(page.getByTestId('attention-patch').first()).toBeVisible({timeout: 20_000});
  await page.getByLabel('Attention layer',{exact:true}).selectOption('1');
  await expect.poll(()=>started).toBe(true);
  await page.getByLabel('Attention layer',{exact:true}).selectOption('2');
  const position = capture.context.pages[0].position_indices[0];
  const patchTitle = page.locator(`[data-testid="attention-patch"][data-position-index="${position}"] title`);
  await expect(patchTitle).toHaveText(`Image position ${position}: patch attention weight ${expected.values[position].toFixed(8)}`);
  release();
  await delivered;
  await expect(patchTitle).toHaveText(`Image position ${position}: patch attention weight ${expected.values[position].toFixed(8)}`);
  await expect(page.getByLabel('Attention layer',{exact:true})).toHaveValue('2');
});

test('attention methods, the layer scrubber and Details preserve real evidence and saved selections', async ({page}) => {
  // The element steps need supplied-page patches mapped to annotated element boxes.
  const mapped = (c: {positions: {ownership?: Record<string, number> | null}[]}) => c.positions.some(p => Object.keys(p.ownership ?? {}).some(k => k.startsWith('element_')));
  // Without full attention, so the matrix methods (rollout) are unavailable and explain why.
  const found = await findGenerationCapture(page.request, c => Boolean(c.retrieval_capture_id) && mapped(c) && !c.full_attention);
  requireSaved(!found, 'Requires a saved generator capture acquired from a retrieval capture whose page has annotated elements; acquire one with scripts/generation_smoke.py');
  const capture = found!;
  const rowUrl = `/api/v1/generation/captures/${capture.capture_id}/rows?step=0&layer=1&head=1`;
  const original = await page.request.get(rowUrl).then(r => r.json());
  let acquisitions = 0;
  await page.route(/\/api\/v1\/(generation|retrieval)\/runs$/, route => {acquisitions++; return route.abort();});
  await page.setViewportSize({width:1280,height:800});
  await page.goto('/attention-tool?generation=' + capture.capture_id);
  const panel = page.getByRole('complementary', {name:'Attention panel',exact:true});
  // The method is named on the page (method sentence); its name opens the method picker.
  const picker = methodChoice(page);
  await expect(picker).toHaveAttribute('data-method-key', 'attention.row/captured_weights_v1');
  await expect(methodBrowse(page)).toBeEnabled();
  await expect(picker).toBeInViewport();
  // Layer and head are on the scrubber under the page, not in the panel.
  await expect(page.getByLabel('Attention layer', {exact:true})).toBeVisible();
  await expect(page.getByLabel('Attention head', {exact:true})).toBeVisible();
  await expect(page.getByLabel('Attention layer', {exact:true})).toBeInViewport();
  await expect(page.getByLabel('Attention head', {exact:true})).toBeInViewport();
  await expect(panel.getByLabel('Attention layer')).toHaveCount(0);
  const methods = await openMethodPicker(page);
  await methods.getByRole('combobox',{name:'Find a method'}).fill('rollout');
  const rollout = methods.locator('[role="option"][data-method-key="attention.rollout/residual_mean_square_v1"]');
  await expect(rollout).toHaveAttribute('aria-disabled','true');
  // Unavailable methods explain their requirement on the picker entry itself.
  await expect(rollout).toContainText('Needs a full-matrix capture');
  await expect(rollout).toContainText(/full attention matrices: run a new generator capture with 'Save full attention matrices'/);
  await closeMethodPicker(page);
  // Keyboard sliders and exact selectors share the same captured axes.
  await page.getByLabel('Attention layer slider', {exact:true}).press('ArrowRight');
  await expect(page.getByLabel('Attention layer', {exact:true})).toHaveValue('1');
  await page.getByLabel('Attention head slider', {exact:true}).press('ArrowRight');
  await expect(page.getByLabel('Attention head', {exact:true})).toHaveValue('1');
  await expectRowTotals(page, original.total_mass.toFixed(6));
  await showElements(page);
  await page.getByTestId('outline-element').first().click();
  const elementMass = await page.getByTestId('element-attention-evidence').textContent();
  // General lists no element aggregation; choosing it in Document-aware groups the same row into elements.
  const general = await openMethodPicker(page);
  await expect(general.getByRole('radio', {name:'General',exact:true})).toHaveAttribute('aria-checked','true');
  await general.getByRole('combobox',{name:'Find a method'}).fill('element_allocation');
  await expect(general.locator('[role="option"][data-method-key="attention.element_allocation/ownership_mass_v1"]')).toHaveCount(0);
  await closeMethodPicker(page);
  await chooseMethod(page, 'attention.element_allocation/ownership_mass_v1');
  await expect(picker).toHaveAttribute('data-method-key', 'attention.element_allocation/ownership_mass_v1');
  // Chosen in Details › Method, the switch to Document-aware keeps Details there; Element lists each element's
  // value and the element summary (the full table) follows it in the same tab.
  await openDetailsTab(page, 'Element');
  await expect(elementList(page).getByTestId('element-value').first()).toBeVisible();
  await expect(page.getByRole('region', {name:'Element summary',exact:true})).toHaveCount(1);
  await expect(page.getByRole('complementary', {name:'Linked element summary',exact:true})).toHaveCount(0);
  await expect(page.getByTestId('element-attention-evidence')).toHaveText(elementMass!);
  // Use a fixed, saturated patch to verify alpha, not a synthetic overlay.
  await withDisplaySettings(page, async drawer=>{
    await pageColouring(drawer, 'Attention patches').click();
    await drawer.getByLabel('Heatmap scale', {exact:true}).selectOption('fixed');
    await drawer.getByLabel('Fixed scale maximum', {exact:true}).fill('0.00000001');
    await setSlider(drawer.getByLabel('Attention opacity', {exact:true}),35);
  });
  // Patches at or above the fixed maximum saturate at exactly the chosen opacity. Some captures give the page almost
  // nothing at this layer and head (image weights below 1e-8, the smallest allowed maximum), so the expected peak
  // opacity follows the row's own image maximum.
  const imageMax = Math.max(...capture.positions.filter((p: {role: string; bbox?: unknown; index: number}) => p.role === 'image' && p.bbox && p.index < original.target_count).map((p: {index: number}) => original.values[p.index]));
  const peak = 0.35 * Math.min(1, imageMax / 1e-8);
  await expect.poll(() => page.getByTestId('attention-patch').evaluateAll(patches => Math.max(...patches.map(p => Number(p.getAttribute('fill-opacity')))))).toBeCloseTo(peak, 3);
  expect(await page.getByTestId('attention-patch').evaluateAll(patches => patches.every(p => Number(p.getAttribute('fill-opacity')) <= 0.35))).toBe(true);
  await expect(page.getByTestId('element-attention-evidence')).toHaveText(elementMass!);
  expect(await page.request.get(rowUrl).then(r => r.json())).toEqual(original);
  await expect(methodBrowse(page)).toBeEnabled();
  await expect(page.getByLabel('Attention head', {exact:true})).toBeVisible();
  await expect(page.getByTestId('attention-patch').first()).toBeVisible();
  await expect(picker).toHaveAttribute('data-method-key', 'attention.element_allocation/ownership_mass_v1');
  await withDisplaySettings(page, drawer=>expect(drawer.getByLabel('Attention opacity', {exact:true})).toHaveValue('35'));
  await openTool(page, 'Notes & name');
  await page.getByLabel('Investigation name').fill('Attention panel browser verification');
  await saveInvestigation(page);
  await expect(page).toHaveURL(/investigation_/);
  const savedUrl = page.url();
  await page.reload();
  await expect(picker).toHaveAttribute('data-method-key', 'attention.element_allocation/ownership_mass_v1');
  await expect(page.getByLabel('Attention layer', {exact:true})).toHaveValue('1');
  await expect(page.getByLabel('Attention head', {exact:true})).toHaveValue('1');
  await showElements(page);
  await expect(elementList(page).getByTestId('element-value').first()).toBeVisible();
  await withDisplaySettings(page, async drawer=>{
    await expect(drawer.getByLabel('Attention opacity', {exact:true})).toHaveValue('35');
    await expect(pageColouring(drawer, 'Attention patches')).toHaveAttribute('aria-checked','true');
  });
  const reopened = await page.request.get('/api/v1/generation/captures/' + capture.capture_id).then(r=>r.json());
  expect(reopened.content_sha256).toBe(capture.content_sha256);
  expect(acquisitions).toBe(0);
  await withDisplaySettings(page, async drawer=>{
    await drawer.getByLabel('Heatmap scale', {exact:true}).selectOption('auto');
    await setSlider(drawer.getByLabel('Attention opacity', {exact:true}),65);
  });
  await page.locator('.docattentionlens-details-body').evaluate(e=>e.scrollTo({top:0}));
  await page.screenshot({path:'../var/screenshots/attention-panel-desktop.png'});
  await page.setViewportSize({width:390,height:844});
  await openDetailsTab(page, 'Method');
  await expect(picker).toBeInViewport();
  await expect(panel.getByLabel('Attention source', {exact:true})).toBeVisible();
  await expect(page.getByLabel('Attention layer', {exact:true})).toBeVisible();
  await page.locator('.docattentionlens-details-body').evaluate(e=>e.scrollTo({top:0}));
  expect(await page.evaluate(()=>document.documentElement.scrollWidth <= innerWidth + 1)).toBe(true);
  await page.screenshot({path:'../var/screenshots/attention-panel-mobile.png'});
  await page.getByRole('button', {name:'Switch to White appearance',exact:true}).click();
  await page.screenshot({path:'../var/screenshots/attention-panel-light.png'});
  // The saved investigation restores the same scientific selection.
  page.on('dialog', dialog=>dialog.accept());
  await page.goto(savedUrl);
  await expect(page.getByRole('region', {name:'Generation analysis'})).toBeVisible();
  await withDisplaySettings(page, drawer=>expect(drawer.getByLabel('Attention opacity', {exact:true})).toHaveValue('35'));
  await openDetailsTab(page, 'Method');
  await expect(panel.getByLabel('Attention source', {exact:true})).toBeVisible();
  await expect(page.getByLabel('Attention layer', {exact:true})).toHaveValue('1');
});
