import { test,expect } from '@playwright/test';
import { chooseMethod, choosePerspective, openTool, methodBrowse } from './workspaceHelpers';

test('document calculations use saved annotations and new-run controls only prepare',async({page})=>{
  test.setTimeout(120000);
  // The pinned document case, read directly (newer captures push it out of the newest 25); table cases have no
  // Document-aware view, so a fallback must be a document (image_token_limit 512) capture.
  const pinned=await page.request.get('/api/v1/generation/captures/generation_5db62722a76640e4bd14acd308263c22');
  const captures=(await page.request.get('/api/v1/generation/captures?limit=25').then(r=>r.json())).items;
  const capture=pinned.ok()?await pinned.json():captures.find((c:{context:{image_token_limit?:number}})=>(c.context.image_token_limit??512)===512)??captures[0];
  let acquisitions=0;
  await page.route(/\/api\/v1\/(generation|retrieval)\/runs$/,r=>{acquisitions++;return r.abort();});
  await page.route(/\/api\/v1\/document-studies\/[^/]+\/runs$/,r=>{acquisitions++;return r.abort();});
  await page.goto('/attention-tool?generation='+capture.capture_id);
  await expect(methodBrowse(page)).toBeEnabled();
  await choosePerspective(page,'Document-aware');
  for(const [method,variant] of [['attention.element_type_profile','type_mass_area_v1'],['attention.designated_support','annotated_support_mass_v1'],['attention.reading_order','annotated_order_profile_v1'],['attention.layout_sinks','document_saved_rows_v1']]){
    await chooseMethod(page,`${method}/${variant}`);
    const result=page.getByTestId('method-result');
    await expect(result).toBeVisible();await expect(result).toHaveAttribute('data-analysis-id',/^analysis_/);
    const saved=await page.request.get('/api/v1/methods/analyses/'+await result.getAttribute('data-analysis-id')).then(r=>r.json());
    expect(saved.method_id).toBe(method);expect(saved.input_manifest.inputs[capture.capture_id].annotation_sha256).toMatch(/^[a-f0-9]{64}$/);
  }
  await openTool(page,'Document studies');
  const dialog=page.getByRole('dialog',{name:'Document studies',exact:true});
  await expect(dialog).toBeVisible();
  await dialog.getByLabel('Document study method',{exact:true}).selectOption('attention.modality_budget_depth');
  await dialog.getByRole('button',{name:'Calculate from saved captures'}).click();
  await expect(dialog.getByTestId('document-study-result')).toBeVisible();
  await expect(dialog.locator('svg').first()).toHaveAttribute('aria-label','Attention budget by decoder layer');
  await dialog.getByLabel('Document study method',{exact:true}).selectOption('document.image_text_contrast');
  await expect(dialog.getByRole('checkbox',{name:'image',exact:true})).toBeChecked();
  await expect(dialog.getByRole('checkbox',{name:'ocr',exact:true})).toBeChecked();
  await expect(dialog.getByRole('checkbox',{name:'structured',exact:true})).toBeChecked();
  await dialog.getByRole('button',{name:'Prepare experiment and preview arms'}).click();
  await expect(dialog.getByRole('button',{name:'Run prepared experiment'})).toBeEnabled();
  await expect(dialog.getByText('ocr · ocr',{exact:true})).toBeVisible();
  expect(acquisitions).toBe(0);
  await page.setViewportSize({width:390,height:844});
  expect((await dialog.boundingBox())!.width).toBeLessThanOrEqual(390);
  await page.screenshot({path:'../var/screenshots/document-studies-mobile.png',fullPage:true});
});

test('question atlas uses explicit saved samples and survives reopening',async({page})=>{
  test.setTimeout(120000);
  const captures=(await page.request.get('/api/v1/generation/captures?limit=25').then(r=>r.json())).items;
  const mapped=(c:{positions:{role:string;ownership:Record<string,number>}[]})=>c.positions.some(p=>p.role==='image'&&Object.keys(p.ownership).some(i=>i.startsWith('element_')));
  const selected=captures.filter(mapped);
  const primary=selected.find((c:{capture_id:string})=>c.capture_id==='generation_5db62722a76640e4bd14acd308263c22')??selected[0];
  const other=selected.find((c:{case_id:string;provenance:{model_id:string;model_revision:string}})=>c.case_id!==primary.case_id&&c.provenance.model_id===primary.provenance.model_id&&c.provenance.model_revision===primary.provenance.model_revision);
  expect(other).toBeTruthy();
  const methods=(await page.request.get('/api/v1/methods').then(r=>r.json())).items;
  const method=methods.find((m:{method_id:string})=>m.method_id==='attention.question_atlas');
  const response=await page.request.post('/api/v1/methods/analyses',{data:{generation_capture_id:primary.capture_id,method_id:method.method_id,variant_id:method.variant_id,
    parameters:{},sample_capture_ids:[primary.capture_id,other.capture_id]}});
  expect(response.ok(),await response.text()).toBeTruthy();
  const result=await response.json();expect(result.plots[0].points).toHaveLength(2);
  const reopened=await page.request.get('/api/v1/methods/analyses/'+result.analysis_id).then(r=>r.json());
  expect(reopened.content_sha256).toBe(result.content_sha256);expect(reopened.input_manifest.sample_capture_ids).toEqual([primary.capture_id,other.capture_id]);
});
