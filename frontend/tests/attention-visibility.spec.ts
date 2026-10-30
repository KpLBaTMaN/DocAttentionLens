import { showElements, expectRowTotals, readRowTotals, withDisplaySettings, chooseColourMode, choosePage } from './workspaceHelpers';
import { test, expect } from '@playwright/test';
import { readFileSync } from 'node:fs';

test('real saved generator attention is visible without altering weights or page comparisons', async ({page}) => {
  const meta=JSON.parse(readFileSync('../var/verification/generation-smoke.json','utf8'));
  const saved=await page.request.get('/api/v1/investigations/'+meta.investigation_id).then(r=>r.json());
  const capture=await page.request.get('/api/v1/generation/captures/'+saved.view_state.generation_capture_id).then(r=>r.json());
  const {attention_step:step,attention_layer:layer,attention_head:head}=saved.view_state;
  const row=await page.request.get(`/api/v1/generation/captures/${capture.capture_id}/rows?step=${step}&layer=${layer}&head=${head}`).then(r=>r.json());
  const imagePositions=capture.positions.filter((p:{role:string;index:number;bbox:unknown})=>p.role==='image'&&p.bbox&&p.index<row.target_count);
  const maximum=Math.max(...imagePositions.map((p:{index:number})=>row.values[p.index]));
  expect(maximum/row.display_max*.65).toBeLessThan(.03); // This real capture reproduced the visibility fault.
  await page.setViewportSize({width:1440,height:900});
  await page.goto('/attention-tool/'+meta.investigation_id);
  await withDisplaySettings(page,drawer=>expect(drawer.getByLabel('Heatmap scale',{exact:true})).toHaveValue('auto'));
  await expect(page.getByTestId('display-scale')).toContainText('image maximum across supplied pages');
  const assertPage=async(pageId:string)=>{
    await choosePage(page,pageId);
    const positions=imagePositions.filter((p:{page_id:string})=>p.page_id===pageId);
    await expect(page.getByTestId('attention-patch')).toHaveCount(positions.length);
    for(const p of [positions[0],positions.reduce((a:{index:number},b:{index:number})=>row.values[a.index]>row.values[b.index]?a:b)]) {
      await expect(page.locator(`[data-testid="attention-patch"][data-position-index="${p.index}"]`)).toHaveAttribute('fill-opacity',String(row.values[p.index]/maximum*.65));
    }
    await expectRowTotals(page,row.image_mass.toFixed(6));
  };
  for(const context of capture.context.pages) await assertPage(context.page_id);
  await assertPage(saved.view_state.page_id);
  const before=await readRowTotals(page);
  const alpha=await page.getByTestId('attention-patch').evaluateAll(es=>Math.max(...es.map(e=>Number(e.getAttribute('fill-opacity')))));
  expect(alpha).toBeGreaterThan(.1);
  if(saved.view_state.selected_element_id) {
    const element=await page.request.get('/api/v1/elements/'+saved.view_state.selected_element_id).then(r=>r.json());
    await showElements(page);
    await page.locator(`[data-testid="outline-element"][data-native-id="${element.native_id}"]`).click();
    const selected=page.getByTestId('page-canvas').locator(`rect.docattentionlens-box[data-element-id="${saved.view_state.selected_element_id}"]`);
    await expect(selected).toHaveClass(/is-selected/);
    expect(await selected.evaluate(e=>getComputedStyle(e).fill)).toBe('rgba(0, 0, 0, 0)');
    await expect(page.getByTestId('element-attention-evidence')).toBeVisible();
  }
  await page.screenshot({path:'../var/screenshots/attention-investigation-after.png'});
  await withDisplaySettings(page,drawer=>drawer.getByLabel('Heatmap scale',{exact:true}).selectOption('causal'));
  await expect(page.getByTestId('display-scale')).toContainText(row.display_max.toFixed(6));
  await expect(page.getByText('Full-row scale makes image weights faint.',{exact:false})).toBeVisible();
  const faint=await page.getByTestId('attention-patch').evaluateAll(es=>Math.max(...es.map(e=>Number(e.getAttribute('fill-opacity')))));
  expect(faint).toBeLessThan(.03);
  await withDisplaySettings(page,drawer=>drawer.getByLabel('Heatmap scale',{exact:true}).selectOption('auto'));
  await expectRowTotals(page,before);
  await chooseColourMode(page,'Original image');
  await expect(page.getByTestId('attention-patch')).toHaveCount(0);
  await expect(page.getByText('Original view: attention is hidden.',{exact:false})).toBeVisible();
  await chooseColourMode(page,'Overlay');
  await page.getByLabel('Attention head',{exact:true}).selectOption(head===0?'1':'0');
  const changed=await page.request.get(`/api/v1/generation/captures/${capture.capture_id}/rows?step=${step}&layer=${layer}&head=${head===0?1:0}`).then(r=>r.json());
  await expectRowTotals(page,changed.image_mass.toFixed(6));
  expect(changed.values).not.toEqual(row.values);
  // Save a new verification record, preserving the researcher's original saved view.
  const response=await page.request.post('/api/v1/investigations',{data:{name:'Attention visibility scale verification',case_id:saved.case_id,view_state:saved.view_state,display_settings:{...saved.display_settings,attention_scale_mode:'causal'}}});
  expect(response.ok()).toBeTruthy();
  const verification=await response.json();
  page.once('dialog',d=>d.accept());
  await page.goto('/attention-tool/'+verification.investigation_id);
  await withDisplaySettings(page,drawer=>expect(drawer.getByLabel('Heatmap scale',{exact:true})).toHaveValue('causal'));
});
