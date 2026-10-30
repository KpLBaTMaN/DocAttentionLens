import { expect, test } from '@playwright/test';
import { detailsPanel, chooseMethod, choosePerspective, methodChoice, openDetailsTab } from './workspaceHelpers';

test('real joint rollout saves and reopens its layer interval in Document-aware and phone views', async ({page}) => {
  test.setTimeout(120_000);
  const captures = (await page.request.get('/api/v1/retrieval/captures?limit=25').then(r => r.json())).items;
  const capture = captures.find((c: {provenance: {model_registration_id?:string}; attention_encodings?:{pass_kind:string}[]}) =>
    c.provenance.model_registration_id === 'builtin-colqwen2.5' && c.attention_encodings?.some(e => e.pass_kind === 'joint'));
  expect(capture, 'Requires a saved ColQwen2.5 joint encoding').toBeTruthy();
  let acquisitions = 0;
  await page.route(/\/api\/v1\/(retrieval|generation)\/runs$/, route => {acquisitions++;return route.abort();});
  const responses: {request: {method_id:string; parameters:{start_layer?:number;layer:number}}; status:number; result:{normalization?:string}}[] = [];
  const requested: {parameters:{start_layer?:number;layer:number}}[] = [];
  page.on('request', r => {if(r.method()==='POST'&&r.url().endsWith('/attention'))requested.push(r.postDataJSON());});
  page.on('response', async r => {if(r.request().method()==='POST'&&r.url().endsWith('/attention')) {
    responses.push({request:r.request().postDataJSON(),status:r.status(),result:await r.json()});
  }});
  await page.setViewportSize({width:1440,height:900});
  await page.goto(`/attention-tool?capture=${capture.capture_id}&role=retrieval`);
  const panel = await detailsPanel(page, 'Retrieval panel');
  await panel.getByRole('button', {name:'Page + question',exact:true}).click();
  const source = await panel.getByLabel('Attention source position').inputValue();
  await chooseMethod(page, 'attention.rollout/retriever_encoding_v1');
  // The rollout interval is on the layer scrubber under the page; its settings stay in Details.
  await page.getByLabel('End layer', {exact:true}).selectOption('4');
  await panel.getByLabel('Residual Coefficient', {exact:true}).fill('0.3');
  await page.getByRole('button', {name:'Choose a layer range',exact:true}).click();
  await expect(page.getByLabel('End layer', {exact:true})).toHaveValue('4');
  await expect(panel.getByLabel('Residual Coefficient', {exact:true})).toHaveValue('0.3');
  await page.getByLabel('Start layer', {exact:true}).selectOption('2');
  await expect.poll(() => responses.at(-1)?.result.normalization, {timeout:30_000}).toContain('matrices composed from 2 through 4');
  await expect(panel.getByLabel('Attention source position')).toHaveValue(source);
  await choosePerspective(page, 'Document-aware');
  await expect(methodChoice(page)).toHaveAttribute('data-input-key', 'attention.partial_rollout/retriever_encoding_v1');
  await expect(page.getByTestId('element-region').first()).toBeVisible({timeout:20_000});
  await page.getByRole('button', {name:'Save investigation',exact:true}).click();
  await expect(page).toHaveURL(/\/attention-tool\/investigation_/);
  await page.reload();
  await expect(page.getByLabel('Start layer', {exact:true})).toHaveValue('2');
  await expect(page.getByLabel('End layer', {exact:true})).toHaveValue('4');
  await openDetailsTab(page,'Method');
  await expect(panel.getByLabel('Residual Coefficient', {exact:true})).toHaveValue('0.3');
  await expect(panel.getByLabel('Attention source position')).toHaveValue(source);
  await expect(panel.getByRole('button', {name:'Page + question',exact:true})).toHaveAttribute('aria-pressed','true');
  await expect(methodChoice(page)).toHaveAttribute('data-input-key', 'attention.partial_rollout/retriever_encoding_v1');
  // The reopened interval's own result arrives before the phone view changes it again.
  await expect.poll(() => responses.at(-1)?.result.normalization, {timeout:30_000}).toContain('matrices composed from 2 through 4');
  await page.screenshot({path:'../var/screenshots/rollout-controls-real-desktop.png'});
  await page.setViewportSize({width:390,height:844});
  await openDetailsTab(page,'Method');
  await page.getByLabel('Start layer', {exact:true}).scrollIntoViewIfNeeded();
  await expect(page.getByLabel('Start layer', {exact:true})).toBeInViewport();
  await page.getByLabel('End layer', {exact:true}).scrollIntoViewIfNeeded();
  await expect(page.getByLabel('End layer', {exact:true})).toBeInViewport();
  await page.getByLabel('End layer', {exact:true}).selectOption('1');
  await expect(page.getByLabel('Start layer', {exact:true})).toHaveValue('1');
  // The interval is requested and its own result is shown (read from the page: the response listener can lag behind it).
  await expect.poll(() => requested.at(-1)?.parameters, {timeout:30_000}).toMatchObject({start_layer:1, layer:1});
  await openDetailsTab(page,'Result');
  await expect(page.getByTestId('method-result')).toContainText('matrices composed from 1 through 1', {timeout:30_000});
  expect(responses.map(r=>r.status)).not.toContain(422);
  expect(acquisitions).toBe(0);
  expect(await page.evaluate(() => document.documentElement.scrollWidth <= innerWidth+1)).toBe(true);
  await page.screenshot({path:'../var/screenshots/rollout-controls-real-mobile.png'});
});
