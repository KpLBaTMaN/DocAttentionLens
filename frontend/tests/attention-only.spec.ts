import { test, expect } from '@playwright/test';
import { openTool, detailsPanel, chooseMethod, showEvidence, methodBrowse, openDetailsTab } from './workspaceHelpers';

const additions=['attention.whole_query','attention.text_to_text','attention.text_to_vision','attention.vision_to_text','attention.vision_to_vision',
  'attention.spatial_direction','attention.local_global','attention.row_structure','attention.column_structure','attention.sink_detection',
  'heads.clustering','heads.pattern_taxonomy','attention.partial_rollout','attention.rollout_evolution','attention.graph','attention.flow'];

async function documentCapture(page:any) {
  const captures=(await page.request.get('/api/v1/retrieval/captures?limit=25').then((r:any)=>r.json())).items;
  const capture=captures.find((c:any)=>c.attention_encodings?.some((e:any)=>e.pass_kind==='document'&&!e.causal));
  expect(capture,'Requires a saved bidirectional document encoding').toBeTruthy();
  return capture;
}

test('new methods are grouped into the correct Toolbox categories',async({page})=>{
  await page.goto('/toolbox');
  await expect(page.getByText('60 method variants',{exact:true})).toBeVisible();
  for(const [category,name] of [
    ['Attention · Aggregation','Whole-Query Attention'],['Attention · Modality blocks','Text → Vision Attention'],
    ['Attention · Spatial structure','Local / Global Attention'],['Attention · Heads and patterns','Head Clustering'],
    ['Attention-derived · Propagation and flow','Attention Flow · pruned capacity'],['Attention · Raw edge graphs','Attention Graph'],
    ['Attention interventions · Controlled model runs','Head Ablation']]) {
    await expect(page.getByRole('region',{name:category,exact:true}).getByRole('heading',{name,exact:true})).toBeVisible();
  }
  await page.screenshot({path:'../var/screenshots/attention-only-toolbox.png',fullPage:true});
});

test('all sixteen saved-attention additions render their declared results',async({page})=>{
  test.setTimeout(180_000);
  const capture=await documentCapture(page);
  const encoding=capture.attention_encodings.find((e:any)=>e.pass_kind==='document');
  const positions=capture.pages.find((p:any)=>p.page_id===encoding.page_id).positions;
  const image=positions.find((p:any)=>p.role==='visual_patch'&&p.eligible).index;
  const text=positions.find((p:any)=>p.role!=='visual_patch'&&p.eligible).index;
  let acquisitions=0;
  await page.route(/\/api\/v1\/(generation|retrieval)\/runs$/,route=>{acquisitions++;return route.abort();});
  await page.goto('/attention-tool?capture='+capture.capture_id+'&role=retrieval');
  const panel=await detailsPanel(page,'Retrieval panel');
  const catalogue=await page.request.get('/api/v1/retrieval/attention-methods').then(r=>r.json());
  const result=page.getByTestId('method-result');
  for(const id of additions) {
    // The source is chosen in Details › Method; the output is read in Result.
    await openDetailsTab(page,'Method');
    await panel.getByLabel('Attention source position').selectOption(String(id==='attention.text_to_text'||id==='attention.text_to_vision'?text:image));
    const method=catalogue.find((m:any)=>m.method_id===id);
    const response=page.waitForResponse(r=>r.url().endsWith('/attention')&&r.request().method()==='POST'&&r.request().postDataJSON().method_id===id);
    await chooseMethod(page,id+'/'+method.variant_id);
    const http=await response;
    expect(http.status(),id).toBe(200);
    const calculated=await http.json();
    await openDetailsTab(page,'Result');
    await expect(result.getByRole('heading',{level:2}).first()).toContainText(method.name);
    if(id==='attention.whole_query') {
      expect(calculated.row.source_index).toBeNull();
      await expect(page.getByTestId('source-population-context')).toContainText(calculated.source_indices.length+' rows');
      await expect(page.locator('[data-testid="attention-patch"][stroke="#39a9e0"]')).toHaveCount(0);
    }
    if(id==='attention.graph'||id==='attention.flow') {
      await expect(result.getByRole('figure',{name:'Attention graph',exact:true})).toBeVisible();
      await expect(result.getByText(/displayed edges/)).toBeVisible();
      await result.screenshot({path:'../var/screenshots/'+id.replace('.','-')+'.png'});
    }
    expect(calculated.row||calculated.table||calculated.statistics||calculated.graph||calculated.comparisons).toBeTruthy();
  }
  expect(acquisitions).toBe(0);
});

test('all five interventions prepare reviewable frozen plans without model execution',async({page})=>{
  const capture=await documentCapture(page);
  let modelRuns=0;
  await page.route('**/attention-interventions/*/runs',route=>{modelRuns++;return route.abort();});
  await page.goto('/attention-tool?capture='+capture.capture_id+'&role=retrieval');
  await expect(methodBrowse(page)).toBeEnabled();
  await openTool(page,'Attention interventions');
  const drawer=page.getByRole('dialog',{name:'Attention interventions',exact:true});
  const ids=['attention.head_ablation','attention.layer_ablation','attention.edge_ablation','attention.modality_ablation','attention.region_masking'];
  for(const id of ids) {
    await drawer.getByLabel('Intervention method').selectOption(id);
    if(id==='attention.edge_ablation')await drawer.getByLabel('Target Index',{exact:true}).fill('1');
    if(id==='attention.region_masking') {
      const region=drawer.getByLabel('Region',{exact:true});
      await region.fill('0.1, 0.2, 0.8, 0.9');
      await region.press('Tab');
    }
    const response=page.waitForResponse(r=>r.url().endsWith('/attention-interventions')&&r.request().method()==='POST');
    await drawer.getByRole('button',{name:'Prepare attention intervention',exact:true}).click();
    const http=await response;expect(http.status(),id).toBe(201);
    const prepared=await http.json();expect(prepared.method_id).toBe(id);expect(prepared.kind).toBe('prepared');
    if(id==='attention.region_masking')expect(prepared.parameters.region).toEqual([.1,.2,.8,.9]);
    await expect(drawer.getByRole('heading',{name:'Prepared experiment',exact:true})).toBeVisible();
    await expect(drawer.getByRole('button',{name:'Run attention intervention on GPU',exact:true})).toBeEnabled();
  }
  expect(modelRuns).toBe(0);
  await drawer.getByRole('heading',{name:'Prepared experiment',exact:true}).scrollIntoViewIfNeeded();
  await page.screenshot({path:'../var/screenshots/attention-intervention-plan.png'});
});

test('diagnostic joint question attention uses its own aligned page positions',async({page})=>{
  const captures=(await page.request.get('/api/v1/retrieval/captures?limit=25').then(r=>r.json())).items;
  const capture=captures.find((c:any)=>c.attention_encodings?.some((e:any)=>e.pass_kind==='joint'));
  expect(capture,'Requires a saved diagnostic joint capture').toBeTruthy();
  const encoding=capture.attention_encodings.find((e:any)=>e.pass_kind==='joint');
  const source=encoding.positions.find((p:any)=>p.role==='query_text'&&p.eligible).index;
  await page.goto('/attention-tool?capture='+capture.capture_id+'&role=retrieval');
  const panel=await detailsPanel(page,'Retrieval panel');
  await panel.getByRole('button',{name:'Page + question',exact:true}).click();
  await expect(panel.getByRole('button',{name:'Page + question',exact:true})).toHaveAttribute('aria-pressed','true');
  await panel.getByLabel('Attention source position').selectOption(String(source));
  await chooseMethod(page,'attention.text_to_vision/retriever_encoding_v1');
  await openDetailsTab(page,'Result');
  await expect(page.getByTestId('method-result')).toContainText('diagnostic joint page + question');
  await expect(page.getByTestId('attention-patch')).toHaveCount(encoding.positions.filter((p:any)=>p.bbox).length);
  await page.screenshot({path:'../var/screenshots/attention-joint-question.png',fullPage:true});
});

test('generator whole-query aggregation identifies its captured population',async({page})=>{
  test.setTimeout(60_000);
  const captures=(await page.request.get('/api/v1/generation/captures?limit=25').then(r=>r.json())).items;
  const capture=captures.find((c:any)=>c.source_rows?.length&&c.provenance.model_registration_id==='builtin-qwen2-vl-2b'&&!c.provenance.attention_experiment_id);
  expect(capture,'Requires saved question source rows').toBeTruthy();
  await page.goto('/attention-tool?generation='+capture.capture_id);
  await expect(methodBrowse(page)).toBeEnabled();
  await chooseMethod(page,'attention.whole_query/captured_attention_v1');
  // The method's own output is in Details › Result.
  await showEvidence(page,'Method result');
  const result=page.getByTestId('method-result');
  await expect(result.getByRole('heading',{name:/Whole-Query Attention/}).first()).toBeVisible({timeout:30_000});
  await expect(page.getByTestId('source-population-context')).toContainText('Captured source population');
  await expect(page.locator('[data-testid="attention-patch"][stroke="#39a9e0"]')).toHaveCount(0);
});

test('a completed GPU intervention opens the saved result in the drawer',async({page})=>{
  test.setTimeout(180_000);
  const capture=await documentCapture(page);
  await page.goto('/attention-tool?capture='+capture.capture_id+'&role=retrieval');
  await expect(methodBrowse(page)).toBeEnabled();
  await openTool(page,'Attention interventions');
  const drawer=page.getByRole('dialog',{name:'Attention interventions',exact:true});
  await drawer.getByRole('button',{name:'Prepare attention intervention',exact:true}).click();
  await expect(drawer.getByRole('heading',{name:'Prepared experiment',exact:true})).toBeVisible();
  await drawer.getByRole('button',{name:'Run attention intervention on GPU',exact:true}).click();
  await expect(drawer.getByRole('heading',{name:'Saved experiment result',exact:true})).toBeVisible({timeout:160_000});
  await expect(drawer.getByText('Experiment job: completed',{exact:true})).toBeVisible();
  await expect(drawer.getByRole('table').first()).toBeVisible();
  await expect(drawer.getByRole('button',{name:'Export attention experiment',exact:true})).toBeEnabled();
  await drawer.getByRole('heading',{name:'Saved experiment result',exact:true}).scrollIntoViewIfNeeded();
  await page.screenshot({path:'../var/screenshots/attention-intervention-result.png'});
});
