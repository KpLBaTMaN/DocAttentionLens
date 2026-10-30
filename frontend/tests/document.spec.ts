import { showElements, showMarkers, openTool, setFit, caseDetails, saveInvestigation, setDetails } from './workspaceHelpers';
import { test, expect, type APIRequestContext } from '@playwright/test';
import { readFileSync } from 'node:fs';

const fixture = (name:string) => JSON.parse(readFileSync(`../tests/fixtures/datasets/${name}.json`,'utf8'));
async function createCase(request:APIRequestContext, documentId:string, pageId?:string) {
  const questions=await request.get(`/api/v1/questions?document_id=${documentId}`).then(r=>r.json());
  const response=await request.post('/api/v1/cases',{data:{question_id:questions.items[0].question_id,query_variant:'document_scoped',task_scope:'document',...(pageId?{page_ids:[pageId]}:{})}});
  expect(response.ok()).toBeTruthy();
  return response.json();
}

test('real input workflow links selections and reopens exact saved wording', async ({page})=>{
  test.setTimeout(60000);
  await page.goto('/artifacts/inputs');
  await expect(page.getByText('80 documents · 1849 pages · 1559 questions · 25760 elements')).toBeVisible({timeout:20000});
  await page.locator('.docattentionlens-input-columns .docattentionlens-choice').first().click();
  await page.getByLabel('Question wording').selectOption('document_scoped');
  await page.locator('.docattentionlens-input-columns .docattentionlens-section').nth(1).locator('.docattentionlens-choice').first().click();
  await expect(page).toHaveURL(/configure=1/);
  await expect(page.getByText(/Question for this run:/)).toBeVisible();
  // Existing immutable cases remain directly inspectable without inference.
  const caseId=new URL(page.url()).searchParams.get('case');
  await page.goto('/attention-tool?case='+caseId);
  await expect(page.getByTestId('page-canvas').locator('image')).toBeVisible();
  await showElements(page);
  await expect(page.getByTestId('outline-element').first()).toBeVisible();
  const question=await page.getByTestId('frozen-question').textContent();
  await showElements(page);
  await page.getByTestId('outline-element').first().click();
  await expect(page.getByTestId('selected-element')).toBeVisible();
  await showMarkers(page);
  await page.getByRole('button',{name:'Mark inspection source',exact:true}).click();
  await showElements(page);
  await page.getByTestId('outline-element').nth(1).click();
  await page.getByRole('button',{name:'Mark inspection target',exact:true}).click();
  await expect(page.locator('[data-testid=outline-element][aria-pressed=true]')).toHaveCount(1);
  await expect(page.locator('.docattentionlens-box.is-source')).toHaveCount(1);
  await expect(page.locator('.docattentionlens-box.is-target')).toHaveCount(1);
  const selected=await page.getByTestId('selected-element').textContent();
  await openTool(page,'Notes & name');
  await page.getByLabel('Investigation name').fill('Browser recovery check');
  await page.getByLabel('Notes',{exact:true}).fill('Real annotations inspected. No model calculation.');
  await saveInvestigation(page);
  await expect(page).toHaveURL(/\/attention-tool\/investigation_/);
  await page.reload();
  await expect(page.getByTestId('frozen-question')).toHaveText(question!);
  await showElements(page);
  await expect(page.getByTestId('selected-element')).toHaveText(selected!);
  await openTool(page,'Notes & name');
  await expect(page.getByLabel('Notes',{exact:true})).toHaveValue('Real annotations inspected. No model calculation.');
  await expect(page.locator('.docattentionlens-box.is-source')).toHaveCount(1);
  await expect(page.locator('.docattentionlens-box.is-target')).toHaveCount(1);
  await setDetails(page,false);
  await page.screenshot({path:'../var/screenshots/document-workspace.png',fullPage:true});
  const iid=page.url().split('/').at(-1)!;
  const saved=await page.request.get('/api/v1/investigations/'+iid).then(r=>r.json());
  const changed=await page.request.patch('/api/v1/investigations/'+iid,{data:{name:saved.name,case_id:saved.case_id,notes:'Concurrent revision',view_state:saved.view_state,display_settings:saved.display_settings,expected_revision:saved.revision}});
  expect(changed.ok()).toBeTruthy();
  await openTool(page,'Notes & name');
  await page.getByLabel('Notes',{exact:true}).fill('Unsaved local note');
  await saveInvestigation(page);
  await expect(page.getByRole('alert')).toContainText('newer revision');
  await expect(page.getByLabel('Notes',{exact:true})).toHaveValue('Unsaved local note');
});

for(const name of ['portrait','landscape','spread-candidate']) {
  test(`real ${name} image and boxes share one transform through zoom and resize`,async ({page})=>{
    const source=fixture(name);
    const record=await createCase(page.request,source.document.document_id,source.page.page_id);
    await page.setViewportSize({width:1600,height:1000});
    await page.goto('/attention-tool?case='+record.case_id);
    await expect(page.locator('.docattentionlens-box').first()).toBeVisible();
    const id=source.elements.items.find((e:{display_bbox:unknown})=>e.display_bbox).element_id;
    const element=source.elements.items.find((e:{element_id:string})=>e.element_id===id);
    const assertAlignment=async()=>{
      const values=await page.evaluate(element=>{
        const image=document.querySelector('.docattentionlens-canvas image')!.getBoundingClientRect();
        const box=document.querySelector(`[data-element-id="${element.element_id}"]`)!.getBoundingClientRect();
        return {image:{left:image.left,top:image.top,width:image.width,height:image.height},box:{left:box.left,top:box.top,width:box.width,height:box.height}};
      },element);
      const scale=values.image.width/source.page.width;
      expect(Math.abs(values.image.height/source.page.height-scale)).toBeLessThan(.001);
      expect(Math.abs(values.box.left-(values.image.left+element.display_bbox.x1*scale))).toBeLessThan(1);
      expect(Math.abs(values.box.top-(values.image.top+element.display_bbox.y1*scale))).toBeLessThan(1);
      expect(Math.abs(values.box.width-(element.display_bbox.x2-element.display_bbox.x1)*scale)).toBeLessThan(1);
    };
    await assertAlignment();
    await page.getByRole('button',{name:'Zoom in',exact:true}).click();
    await assertAlignment();
    const viewport=await page.getByTestId('page-canvas').boundingBox();
    await page.mouse.move(viewport!.x+viewport!.width/2,viewport!.y+viewport!.height/2);
    await page.mouse.down();
    await page.mouse.move(viewport!.x+viewport!.width/2+25,viewport!.y+viewport!.height/2+20,{steps:3});
    await page.mouse.up();
    await assertAlignment();
    await page.setViewportSize({width:1100,height:900});
    await expect.poll(async()=>page.getByTestId('page-canvas').evaluate(node=>node.clientWidth)).toBeLessThan(1100);
    await assertAlignment();
    await setFit(page,'page');
    await page.screenshot({path:`../var/screenshots/document-${name}.png`,fullPage:true});
    if(name==='portrait') {
      await page.setViewportSize({width:390,height:844});
      await expect(page.getByRole('heading',{name:'Attention Tool',exact:true})).toBeVisible();
      await expect(page.getByTestId('page-canvas').locator('image')).toBeVisible();
      await expect(page.getByText('The view could not open.',{exact:false})).toHaveCount(0);
      await assertAlignment();
      expect(await page.evaluate(()=>document.documentElement.scrollWidth>window.innerWidth)).toBe(false);
      await page.screenshot({path:'../var/screenshots/document-mobile.png',fullPage:true});
    }
  });
}

test('raw flagged geometry and empty OCR have explicit inspection states',async ({page})=>{
  page.on('dialog',dialog=>dialog.accept());
  for(const name of ['flagged-geometry','empty-ocr']) {
    const source=fixture(name);
    const record=await createCase(page.request,source.element.document_id,source.page.page_id);
    await page.goto('/attention-tool?case='+record.case_id);
    await showElements(page);
    await page.locator(`[data-testid="outline-element"][data-native-id="${source.element.native_id}"]`).click();
    await page.getByText(name==='empty-ocr'?'OCR':'Geometry',{exact:true}).click();
    if(name==='empty-ocr') await expect(page.getByTestId('ocr-text')).toHaveText('OCR record present; text is empty');
    else {
      await expect(page.getByText('Geometry flagged:',{exact:false})).toBeVisible();
      await expect(page.locator('.docattentionlens-coordinates')).toContainText(String(source.element.bbox.y2));
    }
  }
});

test('switching documents clears local selections even when native element IDs repeat',async ({page})=>{
  const dataset=await page.request.get('/api/v1/datasets').then(r=>r.json());
  const documents=await page.request.get('/api/v1/documents?dataset_id='+dataset.items[0].dataset_id).then(r=>r.json());
  const first=await createCase(page.request,documents.items[0].document_id);
  await page.goto('/attention-tool?case='+first.case_id);
  await showElements(page);
  await page.getByTestId('outline-element').first().click();
  await showMarkers(page);
  await page.getByRole('button',{name:'Mark inspection source',exact:true}).click();
  page.once('dialog',dialog=>dialog.accept());
  await (await caseDetails(page)).getByRole('link',{name:'Choose another case',exact:false}).click();
  await page.locator('.docattentionlens-input-columns .docattentionlens-section').first().locator('.docattentionlens-choice').nth(1).click();
  await page.locator('.docattentionlens-input-columns .docattentionlens-section').nth(1).locator('.docattentionlens-choice').first().click();
  await expect(page).toHaveURL(/configure=1/);
  await page.goto('/attention-tool?case='+new URL(page.url()).searchParams.get('case'));
  await showElements(page);
  await expect(page.getByTestId('outline-element').first()).toBeVisible();
  await expect(page.getByTestId('selected-element')).toHaveCount(0);
  await expect(page.getByRole('button',{name:'Clear source',exact:true})).toHaveCount(0);
  await expect(page.locator('.docattentionlens-box.is-source')).toHaveCount(0);
});

test('unsaved note drafts recover explicitly without replacing the saved workspace',async ({page})=>{
  const source=fixture('portrait');
  const record=await createCase(page.request,source.document.document_id,source.page.page_id);
  const response=await page.request.post('/api/v1/investigations',{data:{name:'Note draft check',case_id:record.case_id,notes:'Saved note',view_state:{page_id:source.page.page_id}}});
  expect(response.ok()).toBeTruthy();
  const saved=await response.json();
  await page.goto('/attention-tool/'+saved.investigation_id);
  await openTool(page,'Notes & name');
  await page.getByLabel('Notes',{exact:true}).fill('Recover this unsaved note');
  await expect.poll(()=>page.evaluate(id=>localStorage.getItem('docattentionlens-note-draft:'+id),saved.investigation_id)).toContain('Recover this unsaved note');
  page.once('dialog',dialog=>dialog.accept());
  await page.reload();
  await openTool(page,'Notes & name');
  await expect(page.getByLabel('Notes',{exact:true})).toHaveValue('Saved note');
  await page.getByRole('complementary',{name:'Attention panel',exact:true}).getByRole('button',{name:'Restore note draft',exact:true}).click();
  await expect(page.getByLabel('Notes',{exact:true})).toHaveValue('Recover this unsaved note');
  await saveInvestigation(page);
  await expect(page.getByText('Revision 2',{exact:true})).toBeVisible();
  expect(await page.evaluate(id=>localStorage.getItem('docattentionlens-note-draft:'+id),saved.investigation_id)).toBeNull();
});

