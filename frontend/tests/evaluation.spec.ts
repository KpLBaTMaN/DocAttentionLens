import { expect, test } from '@playwright/test';
import { readFileSync } from 'node:fs';

const evidence = () => JSON.parse(readFileSync('../var/verification/evaluation-smoke.json','utf8'));

test('real saved evaluation shows rubric outcomes and no invented participants',async({page})=>{
  const saved=evidence();
  await page.goto('/evaluations/'+saved.study_id);
  await expect(page.getByRole('heading',{name:'Evaluation',exact:true})).toBeVisible();
  await expect(page.getByTestId('evaluation-progress')).toContainText('2 / 2');
  await expect(page.getByTestId('utility-progress')).toContainText('0 recorded trials');
  await expect(page.getByText('Strict Boolean accepts true/false',{exact:false})).toBeVisible();
  await expect(page.locator('tbody tr')).toHaveCount(8);
  await expect(page.locator('tbody tr').first()).toContainText('incorrect');
  const hash=await page.request.get('/api/v1/evaluation/studies/'+saved.study_id).then(r=>r.json());
  expect(hash.content_sha256).toBe(saved.study_sha256);
  await page.reload();
  await expect(page.getByTestId('evaluation-progress')).toContainText('2 / 2');
  await page.screenshot({path:'../var/screenshots/evaluation-desktop.png',fullPage:true});
  await page.setViewportSize({width:390,height:844});
  await expect(page.getByTestId('utility-progress')).toBeVisible();
  const fits=await page.evaluate(()=>document.documentElement.scrollWidth<=window.innerWidth+1);
  expect(fits).toBeTruthy();
  await page.screenshot({path:'../var/screenshots/evaluation-mobile.png',fullPage:true});
});

test('evaluation navigation and review gates use real protocols',async({page})=>{
  const saved=evidence();
  await page.goto('/artifacts/evaluations?protocol='+saved.protocol_id);
  await expect(page.getByRole('heading',{name:'Research evaluation',exact:true})).toBeVisible();
  await expect(page.getByRole('button',{name:'Freeze evaluation plan'})).toBeDisabled();
  await expect(page.getByText('Evaluation reference:',{exact:false}).first()).toBeVisible();
  await page.getByRole('link',{name:'Engineering reproducibility: Air Partner profit question',exact:false}).first().click();
  await expect(page).toHaveURL(new RegExp('/evaluations/'+saved.study_id));
  await expect(page.getByTestId('utility-progress')).toContainText('0 participants');
  await page.getByText('Record a completed researcher task',{exact:true}).click();
  await expect(page.getByRole('button',{name:'Save immutable task record'})).toBeDisabled();
  await page.getByRole('button',{name:'Start task timer'}).click();
  await expect(page.getByRole('button',{name:'Stop task timer'})).toBeEnabled();
  await page.getByRole('button',{name:'Stop task timer'}).click();
  await expect(page.getByLabel('Elapsed seconds')).not.toHaveValue('');
  // Timing alone never creates a participant observation or review.
  const report=await page.request.get('/api/v1/evaluation/studies/'+saved.study_id+'/report').then(r=>r.json());
  expect(report.utility.recorded_trials).toBe(0);
});

test('support-bearing candidate keeps independent review and execution pending',async({page})=>{
  const candidate=JSON.parse(readFileSync('../var/verification/evaluation-support-candidate.json','utf8'));
  const record=await page.request.get('/api/v1/experiments/'+candidate.draft_id).then(r=>r.json());
  expect(record.record_kind).toBe('experiment_draft');
  expect(record.review).toBeNull();
  expect(record.protocol.element_ids.some((id:string)=>record.provenance.source_case.support_element_ids.includes(id))).toBeTruthy();
  await page.goto('/experiments/'+candidate.draft_id);
  await expect(page.locator('.docattentionlens-experiment-arm')).toHaveCount(4);
  await expect(page.getByRole('button',{name:'Finalize reviewed protocol',exact:true})).toBeDisabled();
  await expect(page.getByRole('button',{name:'Run paired model experiment',exact:true})).toHaveCount(0);
  await page.getByLabel('Linked element selection').selectOption({label:'477 · table'});
  await expect(page.locator('.docattentionlens-experiment-arm').getByText('477 · table',{exact:true})).toHaveCount(4);
  await page.screenshot({path:'../var/screenshots/support-bearing-candidate.png',fullPage:true});
});
