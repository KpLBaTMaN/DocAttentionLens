import { expectRowTotals, openDetailsTab, detailsPanel, methodChoice } from './workspaceHelpers';
import { test, expect, type Locator } from '@playwright/test';

type Component = {name:string;layers?:number|null;query_heads?:number|null;patch_size?:number|null;spatial_merge_size?:number|null;image_size?:number|null};
type Registration = {registration_id:string;name:string;role:'generation'|'retrieval';architecture:{components?:Component[]};capabilities:{can_run:boolean;summary:string}};

/** The details describe this model from its own declared architecture, never another model's. */
async function expectDescribed(details:Locator,model:Registration,models:Registration[]) {
  const components=model.architecture.components??[];
  for(const c of components) {
    if(c.layers)await expect(details).toContainText(`${c.layers} layers`);
    if(c.query_heads)await expect(details).toContainText(`${c.query_heads} ${c.name==='Language'?'query ':''}heads`);
    if(c.patch_size)await expect(details).toContainText(`${c.patch_size} px patches`);
    if(c.spatial_merge_size)await expect(details).toContainText(`${c.spatial_merge_size} × ${c.spatial_merge_size} spatial merge`);
    if(c.image_size)await expect(details).toContainText(`${c.image_size} × ${c.image_size} input`);
  }
  await expect(details).toContainText(model.capabilities.summary);
  const own=new Set(components.map(c=>c.layers));
  for(const other of models) {
    const layers=other.architecture.components?.find(c=>c.name==='Language')?.layers;
    if(layers&&!own.has(layers))await expect(details).not.toContainText(`${layers} layers`);
  }
}

test('model type lists every registered model and describes each from its own architecture',async({page})=>{
  // Driven by the registrations themselves, so a newly added model is covered here too.
  const models:Registration[]=await page.request.get('/api/v1/setup/models').then(r=>r.json());
  await page.goto('/attention-tool');
  await page.getByRole('button',{name:'New investigation',exact:true}).click();
  const details=page.getByRole('region',{name:'Selected model details'});
  for(const [role,label,radio] of [['retrieval','Retrieval model',/Retrieval/],['generation','Generation model',/Generation/]] as const) {
    const listed=models.filter(m=>m.role===role);
    expect(listed.length,`registered ${role} models`).toBeGreaterThan(0);
    await page.getByRole('radiogroup',{name:'Model type'}).getByRole('radio',{name:radio}).check();
    const picker=page.getByLabel(label,{exact:true});
    // Each role opens on its first registration.
    await expect(picker).toHaveValue(listed[0].registration_id);
    await expect(picker.locator('option')).toHaveText(listed.map(m=>m.name+(m.capabilities.can_run?'':' · adapter required')));
    for(const model of listed) {
      await picker.selectOption(model.registration_id);
      await expectDescribed(details,model,models);
    }
    await picker.selectOption(listed[0].registration_id);
    if(role==='retrieval') {
      await page.reload();
      await expect(picker).toHaveValue(listed[0].registration_id);
      await page.setViewportSize({width:390,height:844});
      expect(await page.evaluate(()=>document.documentElement.scrollWidth<=innerWidth+1)).toBe(true);
      await page.screenshot({path:'../var/screenshots/model-type-retrieval-mobile.png',fullPage:true});
      await page.setViewportSize({width:1280,height:900});
    } else {
      await page.screenshot({path:'../var/screenshots/model-type-generation.png',fullPage:true});
    }
  }
});

test('another model supplies its own description and cannot launch an unsupported adapter',async({page})=>{
  const existing=await page.request.get('/api/v1/setup/models').then(r=>r.json());
  await page.route('**/api/v1/setup/models',route=>route.fulfill({json:[...existing,{
    registration_id:'metadata-test',name:'Another retrieval architecture',model_id:'fixture-only',revision:'fixture',role:'retrieval',validation:'UI metadata fixture only',
    architecture:{model_type:'fixture',components:[{name:'Language',layers:24,query_heads:12,kv_heads:3},{name:'Vision',layers:6,query_heads:4,patch_size:16,image_size:384}]},
    capabilities:{can_run:false,attention_available:false,signal_description:'Checkpoint metadata',input_description:'Adapter-dependent inputs',summary:'Metadata only until an adapter is available.',reason:'No compatible retrieval extraction adapter is installed.',limits:{}}
  }]}));
  await page.goto('/attention-tool');
  await page.getByRole('button',{name:'New investigation',exact:true}).click();
  await page.getByRole('radiogroup',{name:'Model type'}).getByRole('radio',{name:/Retrieval/}).check();
  await page.getByLabel('Retrieval model',{exact:true}).selectOption('metadata-test');
  const details=page.getByRole('region',{name:'Selected model details'});
  await expect(details).toContainText('24 layers');
  await expect(details).toContainText('12 query heads');
  await expect(details).toContainText('16 px patches');
  await expect(details).not.toContainText('18 layers');
  // The launch blocker stays visible while the full description is collapsed.
  await expect(details.getByRole('status')).toContainText('No compatible retrieval extraction adapter');
  await expect(page.getByRole('checkbox',{name:/Let a retriever choose the pages first/})).toHaveCount(0);
  await page.getByRole('button',{name:/^(Continue|Review setup)$/}).click();
  await expect(page.getByRole('heading',{name:'Choose a question and pages'})).toBeVisible();
});

test('retrieval setup launches the selected retriever and preserves model role after saving',async({page})=>{
  test.setTimeout(180_000);
  const old=await page.request.get('/api/v1/retrieval/captures?limit=25').then(r=>r.json());
  const capture=old.items[0];expect(capture).toBeTruthy();
  await page.goto(`/attention-tool?case=${capture.case_id}&configure=1`);
  await page.locator('.docattentionlens-setup-steps').getByRole('button',{name:/Model/}).click();
  await page.getByRole('radiogroup',{name:'Model type'}).getByRole('radio',{name:/Retrieval/}).check();
  await page.getByRole('button',{name:/^(Continue|Review setup)$/}).click();
  const candidates=page.locator('.docattentionlens-setup-pages input[type=checkbox]');
  await expect(candidates.first()).toBeChecked();
  for(let i=1;i<await candidates.count();i++)await candidates.nth(i).uncheck();
  await page.getByRole('button',{name:/^(Continue|Review setup)$/}).click();
  await expect(page.getByRole('heading',{name:'Check what the retriever will see'})).toBeVisible();
  await expect(page.getByText('System instruction',{exact:true})).toHaveCount(0);
  await expect(page.locator('.docattentionlens-setup-review')).toContainText('ColPali v1.3');
  await page.evaluate(()=>scrollTo(0,0));
  await page.screenshot({path:'../var/screenshots/model-type-retrieval-review.png',fullPage:true});
  const submission=page.waitForRequest('**/api/v1/retrieval/runs');
  let generations=0;page.on('request',r=>{if(r.url().endsWith('/generation/runs'))generations++;});
  await page.getByRole('button',{name:'Run investigation',exact:true}).click();
  expect((await submission).postDataJSON().model_registration_id).toBe('builtin-retrieval');
  await expect(page).toHaveURL(/capture=.*role=retrieval/,{timeout:160_000});
  await expect(page.getByTestId('capture-menu').locator('.docattentionlens-capture-role')).toHaveText('Retrieval');
  await expect(page.getByRole('tab',{name:'Generation',exact:true})).toHaveCount(0);
  await expectRowTotals(page,'','retriever-attention-mass');
  await expect(await detailsPanel(page,'Retrieval panel')).toBeVisible();
  await expect(page.getByLabel('Saved attention capture',{exact:true})).toHaveCount(0);
  await page.getByRole('button',{name:'Save investigation',exact:true}).click();
  await expect(page).toHaveURL(/investigation_/);await page.reload();
  await expect(page.getByTestId('capture-menu').locator('.docattentionlens-capture-role')).toHaveText('Retrieval');
  expect(generations).toBe(0);
});

test('retrieval-only links and saved views keep generator controls out of the retrieval panel',async({page})=>{
  const records=await page.request.get('/api/v1/retrieval/captures?limit=25').then(r=>r.json());
  const capture=records.items.find((c:{attention_encodings?:unknown[]})=>!c.attention_encodings?.length);expect(capture).toBeTruthy();
  let runs=0;
  await page.route(/\/api\/v1\/(generation|retrieval)\/runs$/,route=>{runs++;return route.abort();});
  await page.goto(`/attention-tool?capture=${capture.capture_id}`);
  const panel=await detailsPanel(page,'Retrieval panel');
  await expect(panel).toBeVisible();
  await expect(panel.getByText(/This saved capture has no internal retriever attention/).first()).toBeVisible();
  // Without saved encodings no attention method can be chosen: the page names MaxSim and offers no method picker.
  await expect(page.getByRole('group',{name:'Shown on the page'})).toContainText('MaxSim similarity (not attention)');
  await expect(methodChoice(page)).toHaveCount(0);
  await expect(page.getByRole('dialog',{name:'Choose analysis method'})).toHaveCount(0);
  await expect(page.getByTestId('capture-menu').locator('.docattentionlens-capture-role')).toHaveText('Retrieval');
  for(const name of ['Saved attention capture','Attention layer','Attention head','Prediction step','Attention opacity']) {
    await expect(page.getByLabel(name,{exact:true})).toHaveCount(0);
  }
  // MaxSim is not attention: its query position is Details › Method's source; its display and ledgers are in Result.
  await expect(page.getByLabel('Overlay query position',{exact:true})).toBeVisible();
  await openDetailsTab(page,'Result');
  await page.getByLabel('Similarity opacity',{exact:true}).fill('35');
  await expect(page.getByTestId('capture-menu').locator('.docattentionlens-capture-role')).toHaveText('Retrieval');
  await openDetailsTab(page,'Method');
  await panel.getByRole('button',{name:'Acquire retriever attention',exact:true}).click();
  await expect(page.getByRole('dialog').getByLabel('Saved retrieval capture',{exact:true})).toBeVisible();
  await expect(page.getByLabel('Saved generator capture',{exact:true})).toHaveCount(0);
  await page.getByRole('button',{name:'Close panel',exact:true}).click();
  await page.getByRole('button',{name:'Save investigation',exact:true}).click();
  await expect(page).toHaveURL(/investigation_/);await page.reload();
  await expect(panel).toBeVisible();
  await openDetailsTab(page,'Result');
  await expect(page.getByLabel('Similarity opacity',{exact:true})).toHaveValue('35');
  await expect(page.getByLabel('Saved attention capture',{exact:true})).toHaveCount(0);
  await expect(page.getByTestId('retrieval-winner')).toBeVisible();
  await page.screenshot({path:'../var/screenshots/retrieval-scoped-controls.png'});
  await page.setViewportSize({width:390,height:844});
  await openDetailsTab(page,'Method');
  await expect(panel.getByRole('tablist',{name:'Details sections'})).toBeInViewport();
  // MaxSim's query position is in Details › Method.
  await page.getByLabel('Overlay query position',{exact:true}).scrollIntoViewIfNeeded();
  await expect(page.getByLabel('Overlay query position',{exact:true})).toBeInViewport();
  expect(await page.evaluate(()=>document.documentElement.scrollWidth<=innerWidth+1)).toBe(true);
  expect(runs).toBe(0);
});
