import { test, expect } from '@playwright/test';
import { displaySettingsButton, withDisplaySettings, chooseMethod, methodChoice, setDetails, openTool, saveInvestigation, openDetailsTab, methodBrowse } from './workspaceHelpers';

test('twelve saved-row tools expose appropriate controls and preserve capture identity',async({page})=>{
  const capture=(await page.request.get('/api/v1/generation/captures').then(r=>r.json())).items[0];
  expect(capture,'Requires a saved real capture').toBeTruthy();
  let acquisitions=0;
  await page.route(/\/api\/v1\/(generation|retrieval)\/runs$/,route=>{acquisitions++;return route.abort();});
  await page.goto('/attention-tool?generation='+capture.capture_id);
  await expect(methodBrowse(page)).toBeEnabled();
  // The output is in Details › Result; method settings are in Details › Method.
  await setDetails(page,true);
  await openDetailsTab(page,'Result');
  const result=page.getByTestId('method-result');
  const methods=['attention.mean_heads','attention.max_heads','attention.min_heads','attention.weighted_heads','attention.entropy',
    'attention.concentration','attention.topk_mass','attention.sinks','attention.modality_ratio','attention.head_by_head','attention.across_layers','heads.distribution_similarity'];
  for(const id of methods){
    const variant=id==='attention.entropy'?'shannon_row_v1':'captured_prediction_v1';
    await chooseMethod(page,`${id}/${variant}`);
    await expect(result).toHaveAttribute('data-analysis-id',/^analysis_/);
    const reference=await page.request.get('/api/v1/methods/analyses/'+await result.getAttribute('data-analysis-id')).then(r=>r.json());
    expect(reference.method_id).toBe(id);
    expect(reference.capture_sha256).toBe(capture.content_sha256);
    expect(reference.target_count).toBe(capture.steps[0].target_count);
    if(['attention.mean_heads','attention.max_heads','attention.min_heads','attention.weighted_heads','attention.head_by_head','heads.distribution_similarity'].includes(id)){
      await expect(page.getByLabel('Attention head',{exact:true})).toHaveCount(0);
      await openDetailsTab(page,'Method');
      await expect(page.getByLabel('Include head 0',{exact:true})).toBeVisible();
      await openDetailsTab(page,'Result');
    }
    if(reference.row){
      await expect(page.getByTestId('attention-patch').first()).toBeVisible();
      if(id==='attention.max_heads'||id==='attention.min_heads')expect(reference.row.quantity).toContain('envelope');
    }else if(reference.comparisons?.length>1){
      // One map per layer or head: the page draws the chosen map and the method sentence offers every map.
      await expect(page.getByTestId('attention-patch').first()).toBeVisible();
      await expect(page.getByLabel('Shown comparison',{exact:true}).locator('option')).toHaveCount(reference.comparisons.length);
    }else await expect(page.getByTestId('attention-patch')).toHaveCount(0);
    if(id==='attention.entropy'){
      await expect(result.getByText('Shannon entropy',{exact:true})).toBeVisible();
      // Entropy is not an attention weight, so the colour bar's display choices are unavailable.
      await expect(displaySettingsButton(page)).toBeDisabled();
    }
    if(id==='attention.head_by_head'||id==='attention.across_layers')await expect(result.locator('figure')).toHaveCount(8);
  }
  expect(acquisitions).toBe(0);
  const reopened=await page.request.get('/api/v1/generation/captures/'+capture.capture_id).then(r=>r.json());
  expect(reopened.content_sha256).toBe(capture.content_sha256);
});

test('head weights, top-k drafts and immutable results survive saving and reopening',async({page})=>{
  const capture=(await page.request.get('/api/v1/generation/captures').then(r=>r.json())).items[0];
  await page.goto('/attention-tool?generation='+capture.capture_id);
  const picker=methodChoice(page);
  await expect(methodBrowse(page)).toBeEnabled();
  // Settings are set in Details › Method and the output read in Result.
  await setDetails(page,true);
  await chooseMethod(page,'attention.weighted_heads/captured_prediction_v1');
  await openDetailsTab(page,'Method');
  await page.getByLabel('Weight for head 0',{exact:true}).fill('3');
  const result=page.getByTestId('method-result');
  await openDetailsTab(page,'Result');
  await expect(result).toHaveAttribute('data-analysis-id',/^analysis_/);
  await expect.poll(async()=>page.request.get('/api/v1/methods/analyses/'+await result.getAttribute('data-analysis-id')).then(r=>r.json()).then(r=>r.parameters?.weights?.[0])).toBe(3);
  await chooseMethod(page,'attention.topk_mass/captured_prediction_v1');
  await openDetailsTab(page,'Method');
  await page.getByLabel('Top-k targets',{exact:true}).fill('8');
  await openDetailsTab(page,'Result');
  await expect(result.getByText('Top-8 attention mass',{exact:true})).toBeVisible();
  await chooseMethod(page,'attention.weighted_heads/captured_prediction_v1');
  await openDetailsTab(page,'Method');
  await expect(page.getByLabel('Weight for head 0',{exact:true})).toHaveValue('3');
  await openDetailsTab(page,'Result');
  await expect(result).toHaveAttribute('data-analysis-id',/^analysis_/);
  const identifier=await result.getAttribute('data-analysis-id');
  await withDisplaySettings(page,drawer=>drawer.getByLabel('Attention opacity',{exact:true}).fill('20'));
  await expect(result).toHaveAttribute('data-analysis-id',identifier!);
  await openTool(page,'Notes & name');
  await page.getByLabel('Investigation name').fill('Saved method implementation browser verification');
  await saveInvestigation(page);
  await expect(page).toHaveURL(/investigation_/);
  await page.reload();
  await expect(picker).toHaveAttribute('data-method-key','attention.weighted_heads/captured_prediction_v1');
  await openDetailsTab(page,'Method');
  await expect(page.getByLabel('Weight for head 0',{exact:true})).toHaveValue('3');
  await openDetailsTab(page,'Result');
  await expect(result).toHaveAttribute('data-analysis-id',identifier!);
  await chooseMethod(page,'attention.topk_mass/captured_prediction_v1');
  await openDetailsTab(page,'Method');
  await expect(page.getByLabel('Top-k targets',{exact:true})).toHaveValue('8');
  await openDetailsTab(page,'Result');
  await expect(result.getByText('Top-8 attention mass',{exact:true})).toBeVisible();
  await page.screenshot({path:'../var/screenshots/saved-methods-statistics.png'});
  await chooseMethod(page,'attention.head_by_head/captured_prediction_v1');
  await expect(result.locator('figure')).toHaveCount(8);
  await page.screenshot({path:'../var/screenshots/saved-methods-heads.png'});
  await page.setViewportSize({width:390,height:844});
  await openDetailsTab(page,'Result');
  await expect(picker).toBeInViewport();
  expect(await page.evaluate(()=>document.documentElement.scrollWidth<=innerWidth+1)).toBe(true);
  await page.screenshot({path:'../var/screenshots/saved-methods-mobile.png'});
});

test('a delayed derived response cannot replace a newer method parameter result',async({page})=>{
  const capture=(await page.request.get('/api/v1/generation/captures').then(r=>r.json())).items[0];
  let started=false;
  let release!:()=>void;
  const gate=new Promise<void>(resolve=>release=resolve);
  await page.route('**/methods/analyses',async route=>{
    const body=route.request().postDataJSON();
    if(body.method_id!=='attention.topk_mass'||body.parameters.k!==3)return route.continue();
    const response=await route.fetch();started=true;await gate;await route.fulfill({response});
  });
  await page.goto('/attention-tool?generation='+capture.capture_id);
  await expect(methodBrowse(page)).toBeEnabled();
  await setDetails(page,true);
  await chooseMethod(page,'attention.topk_mass/captured_prediction_v1');
  await openDetailsTab(page,'Method');
  await page.getByLabel('Top-k targets',{exact:true}).fill('3');
  await expect.poll(()=>started).toBe(true);
  await page.getByLabel('Top-k targets',{exact:true}).fill('5');
  await openDetailsTab(page,'Result');
  const result=page.getByTestId('method-result');
  await expect(result.getByText('Top-5 attention mass',{exact:true})).toBeVisible();
  const identifier=await result.getAttribute('data-analysis-id');
  release();
  await expect(result).toHaveAttribute('data-analysis-id',identifier!);
  await expect(result.getByText('Top-3 attention mass',{exact:true})).toHaveCount(0);
});
