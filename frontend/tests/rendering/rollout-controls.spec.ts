import { expect, test } from '@playwright/test';
import * as F from '../../fixtures/attention';
import { chooseMethod, choosePerspective, methodChoice, openDetailsTab } from '../workspaceHelpers';
import { mockAttentionApi } from './mockApi';
import type { components } from '../../src/api/generated/schema';

const capture = {...F.retrievalCapture, attention_encodings: F.retrievalCapture.attention_encodings!.map(e => ({...e,
  layer_count: 36, query_head_count: 16, layer_arrays: Array.from({length:36}, (_,i) => `attn_${e.encoding_id}_l${i}`)}))};
const key = (id: string) => `${id}/retriever_encoding_v1`;
type Request = {method_id:string; encoding_id:string; source_index:number; parameters:{layer:number; start_layer?:number; residual_coefficient?:number}};

test('paired rollout endpoints send valid intervals and retain source, residual and grouping', async ({page}) => {
  const api = await mockAttentionApi(page, {retrievalCapture:capture});
  const requests: Request[] = [];
  page.on('request', r => {if(r.method()==='POST'&&r.url().endsWith('/attention'))requests.push(r.postDataJSON());});
  await page.setViewportSize({width:1440,height:900});
  await page.goto(`/attention-tool?capture=${F.RETRIEVAL_ID}&role=retrieval`);
  await openDetailsTab(page, 'Method');
  await page.getByLabel('Attention source position').selectOption('13');
  await chooseMethod(page, key('attention.rollout'));
  await expect(page.getByLabel('End layer', {exact:true}).locator('option')).toHaveCount(36);
  await expect(page.getByLabel('Start layer', {exact:true})).toHaveCount(0);
  await page.getByLabel('End layer', {exact:true}).selectOption('20');
  await page.getByLabel('Residual Coefficient', {exact:true}).fill('0.3');
  await page.getByRole('button', {name:'Choose a layer range',exact:true}).click();
  await expect(methodChoice(page)).toHaveAttribute('data-method-key', key('attention.partial_rollout'));
  await expect(page.getByLabel('Start layer', {exact:true})).toHaveValue('0');
  await expect(page.getByLabel('End layer', {exact:true})).toHaveValue('20');
  await expect(page.getByLabel('Residual Coefficient', {exact:true})).toHaveValue('0.3');
  await expect(page.getByLabel('Start Layer', {exact:true})).toHaveCount(0); // old duplicate number box
  await page.getByLabel('Start layer', {exact:true}).selectOption('10');
  await expect(page.getByRole('group', {name:'Rollout layer interval'})).toContainText('Layers 10–20 · 11 layers composed');
  await expect.poll(() => requests.at(-1)).toMatchObject({method_id:'attention.partial_rollout',source_index:13,
    parameters:{start_layer:10,layer:20,residual_coefficient:0.3}});
  // Exact selectors and keyboard sliders share the same paired interval rules.
  await page.getByLabel('Start layer slider', {exact:true}).press('End');
  await expect(page.getByLabel('Start layer', {exact:true})).toHaveValue('35');
  await expect(page.getByLabel('End layer', {exact:true})).toHaveValue('35');
  await expect(page.getByRole('group', {name:'Rollout layer interval'})).toContainText('1 layer composed');
  await page.getByLabel('End layer slider', {exact:true}).press('Home');
  await expect(page.getByLabel('Start layer', {exact:true})).toHaveValue('0');
  await expect(page.getByLabel('End layer', {exact:true})).toHaveValue('0');
  await page.getByLabel('End layer', {exact:true}).selectOption('20');
  await page.getByLabel('Start layer', {exact:true}).selectOption('10');
  await choosePerspective(page, 'Document-aware');
  await expect(methodChoice(page)).toHaveAttribute('data-input-key', key('attention.partial_rollout'));
  await expect(page.getByLabel('Start layer', {exact:true})).toHaveValue('10');
  await expect(page.getByTestId('element-region')).toHaveCount(F.ELEMENTS.length);
  await page.getByLabel('End layer', {exact:true}).selectOption('8');
  await expect(page.getByLabel('Start layer', {exact:true})).toHaveValue('8');
  await expect.poll(() => requests.at(-1)?.parameters).toMatchObject({start_layer:8,layer:8});
  await page.screenshot({path:'../var/screenshots/rollout-controls-desktop.png'});
  await page.setViewportSize({width:390,height:844});
  await openDetailsTab(page, 'Method');
  await page.getByLabel('Start layer', {exact:true}).scrollIntoViewIfNeeded();
  await expect(page.getByLabel('Start layer', {exact:true})).toBeInViewport();
  await page.getByLabel('End layer', {exact:true}).scrollIntoViewIfNeeded();
  await expect(page.getByLabel('End layer', {exact:true})).toBeInViewport();
  expect(await page.evaluate(() => document.documentElement.scrollWidth <= innerWidth+1)).toBe(true);
  await page.screenshot({path:'../var/screenshots/rollout-controls-mobile.png'});
  for(const r of requests.filter(r => r.method_id === 'attention.partial_rollout')) {
    expect(r.parameters.start_layer??0).toBeLessThanOrEqual(r.parameters.layer);
    expect(r.parameters.layer).toBeLessThan(36);
  }
  expect(api.unexpected).toEqual([]);
});

test('invalid saved intervals wait for explicit repair and do not request a calculation', async ({page}) => {
  const api = await mockAttentionApi(page, {retrievalCapture:capture});
  const requests: Request[] = [];
  page.on('request', r => {if(r.method()==='POST'&&r.url().endsWith('/attention'))requests.push(r.postDataJSON());});
  // Earlier saved views can omit fields that the current UI fills with defaults.
  const investigation: Omit<components['schemas']['InvestigationRecord'],'view_state'|'display_settings'> & {view_state:Partial<components['schemas']['ViewState']>} = {schema_version:'0.4.0', record_kind:'investigation',
    investigation_id:'investigation_rollout_saved', key:'investigation_rollout_saved', revision_id:'revision_rollout_saved', content_sha256:'0'.repeat(64),
    revision:1, name:'Saved layer interval', notes:'', created_at:'2026-10-14T00:00:00Z', case_id:F.CASE_ID,
    view_state:{investigation_model_role:'retrieval',retrieval_capture_id:F.RETRIEVAL_ID,retrieval_encoding_id:'document',retrieval_source_index:13,
      retrieval_attention_method:'attention.partial_rollout',retrieval_attention_parameters:{start_layer:127},attention_layer:20,page_id:F.PAGE_ID},
    dependencies:[], producer_version:'fixture', provenance:{}};
  await page.route('**/api/v1/investigations/investigation_rollout_saved', route => route.fulfill({json:investigation}));
  await page.goto('/attention-tool/investigation_rollout_saved');
  const interval = page.getByRole('group', {name:'Rollout layer interval'});
  await expect(interval.getByRole('alert')).toContainText('saved interval 127–20');
  await page.waitForTimeout(300); // longer than the saved-row debounce
  expect(requests).toEqual([]);
  await interval.getByRole('button', {name:'Use layers 20–20',exact:true}).click();
  await expect.poll(() => requests.at(-1)?.parameters).toMatchObject({start_layer:20,layer:20});
  expect(api.unexpected).toEqual([]);
});
