import { choosePage, filmstrip, openTool } from './workspaceHelpers';
import { test,expect } from '@playwright/test';
import fs from 'node:fs';

test('real page experiment preview records review without enabling unreviewed execution',async({page})=>{
  test.setTimeout(120_000);
  const body=JSON.parse(fs.readFileSync('../var/verification/experiment-request.json','utf8'));
  const c=await page.request.get('/api/v1/cases/'+body.case_id).then(r=>r.json());
  await page.goto('/attention-tool?case='+body.case_id);
  // Pages are chosen in the filmstrip (or Details › Run's page list); the canvas then shows the chosen page.
  if(c.page_ids.length>1)await expect(filmstrip(page)).toBeVisible({timeout:20_000});
  await choosePage(page,body.page_id);
  await openTool(page, 'Layout experiment');
  const panel=page.locator('.docattentionlens-experiment-launch');
  await panel.getByText('Prepare a layout experiment on this page',{exact:true}).click();
  await panel.getByLabel('Experiment name').fill('Browser: actual layout preview');
  for(const native of [549,550])await panel.getByLabel('Group element '+native,{exact:true}).check();
  for(const native of [551,552])await panel.getByLabel('Control element '+native,{exact:true}).check();
  await panel.getByRole('button',{name:'Build original / variant preview',exact:true}).click();
  // The server renders the real original and variant page images before the draft is saved.
  const link=panel.getByRole('link',{name:'Review the pages and finalize the protocol'});await expect(link).toBeVisible({timeout:60_000});
  page.on('dialog',d=>d.accept());await link.click();
  await expect(page.getByRole('heading',{name:'Layout comparison',exact:true})).toBeVisible();
  await expect(page.locator('.docattentionlens-experiment-arm')).toHaveCount(4);
  await expect(page.getByRole('button',{name:'Finalize reviewed protocol',exact:true})).toBeDisabled();
  await page.getByLabel('Reviewer',{exact:true}).fill('Browser automation; not independent human review');
  await page.getByLabel('Review notes',{exact:true}).fill('Engineering transport check of known intact group and matched control. Not a model finding or independent semantic review.');
  for(const label of ['The selected heading and table belong','Content, association and meaning','Source and moved content are readable','I have reviewed the rendering'])await page.getByLabel(new RegExp(label)).check();
  await expect(page.getByRole('button',{name:'Finalize reviewed protocol',exact:true})).toBeEnabled();
  await page.getByRole('button',{name:'Finalize reviewed protocol',exact:true}).click();
  await expect(page.getByRole('heading',{name:'Recorded review',exact:true})).toBeVisible();
  await expect(page.getByRole('heading',{name:'Run the paired experiment',exact:true})).toBeVisible();
  expect(c.page_ids).toContain(body.page_id);
  // Reload preserves declared review, variants, hashes and the source case link.
  await page.reload();await expect(page.getByText('Browser automation; not independent human review:',{exact:false})).toBeVisible();
});

test('measured layout comparison reopens actual outputs, context and independently mapped attention',async({page})=>{
  const meta=JSON.parse(fs.readFileSync('../var/verification/experiment-smoke.json','utf8'));
  expect(meta.result_id,'Run scripts/experiment_smoke.py to completion before this acceptance check').toBeTruthy();
  const result=await page.request.get('/api/v1/experiments/'+meta.result_id).then(r=>r.json());
  await page.setViewportSize({width:1600,height:1100});await page.goto('/experiments/'+meta.result_id);
  await expect(page.getByRole('heading',{name:'Measured paired outcomes',exact:true})).toBeVisible();
  await expect(page.getByTestId('experiment-answer')).toHaveCount(4);
  await expect(page.getByTestId('experiment-attention-mass')).toHaveCount(4);
  await page.getByLabel('Attention layer',{exact:true}).selectOption('35');await page.getByLabel('Attention head',{exact:true}).selectOption('15');
  const row=await page.request.get(`/api/v1/experiments/${meta.result_id}/arms/variant/attention?step=0&layer=35&head=15`).then(r=>r.json());
  const variant=page.locator('.docattentionlens-experiment-arm').filter({has:page.getByRole('heading',{name:'variant',exact:true})});
  await expect(variant.getByTestId('experiment-attention-mass')).toContainText(row.total_mass.toFixed(6));
  const group=result.protocol.element_ids[0];await page.getByLabel('Linked element selection').selectOption(group);
  await expect(variant.getByText('Selected element · persistent source identity',{exact:true})).toBeVisible();
  await expect(variant.getByTestId('element-retrieval-evidence')).toContainText('signed full-query MaxSim contribution');
  await page.getByLabel('Show actual attention').check();
  const gen=await page.request.get(`/api/v1/experiments/${meta.result_id}/arms/variant/generation`).then(r=>r.json());
  const arm=result.arms.find((a:{name:string})=>a.name==='variant');const positions=gen.positions.filter((p:{page_id:string})=>p.page_id===arm.page.page_id);
  await expect(variant.getByTestId('experiment-attention-patch')).toHaveCount(positions.length);
  expect(positions.length).toBeGreaterThan(0); // real changed image reached generator
  const image=await page.request.get(`/api/v1/experiments/${meta.result_id}/arms/variant/input-image/${arm.page.page_id}`);expect(image.status()).toBe(200);
  await page.screenshot({path:'../var/screenshots/experiment-desktop.png',fullPage:true});
  await page.setViewportSize({width:390,height:844});await expect(page.locator('.docattentionlens-experiment-arm')).toHaveCount(4);
  expect(await page.evaluate(()=>document.documentElement.scrollWidth<=innerWidth+1)).toBeTruthy();
  await page.screenshot({path:'../var/screenshots/experiment-mobile.png',fullPage:true});
  await page.reload();await expect(page.getByTestId('experiment-answer')).toHaveCount(4);
});
