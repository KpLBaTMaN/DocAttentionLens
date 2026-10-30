import { showRetrievalScores, openTool, showElements, expectRowTotals, withPanelHidden, caseDetails, setFit, setDetails, choosePage, saveInvestigation, openDetailsTab } from './workspaceHelpers';
import { test,expect } from '@playwright/test';
import { findGenerationCapture, requireSaved } from './savedEvidence';

test('real generator evidence preserves context, causal attention axes and saved view',async({page})=>{
  // Supporting retrieval scores need a capture acquired from retrieval; the row checks read layer 35, head 15.
  const found=await findGenerationCapture(page.request,c=>Boolean(c.retrieval_capture_id)&&c.layer_count>35&&c.query_head_count>15,c=>c.context.pages.length===2);
  requireSaved(!found,'Requires a saved generator capture acquired from a retrieval capture with layer 35 and head 15; acquire one with scripts/generation_smoke.py');
  const capture=found!;
  await page.setViewportSize({width:1600,height:1000});
  await page.goto('/attention-tool?generation='+capture.capture_id);
  await expect(page.getByTestId('capture-menu').locator('.docattentionlens-capture-role')).toHaveText('Generation');
  // The full answer is in the case details (breadcrumb ▾); its tokens are in the source line.
  await expect((await caseDetails(page)).getByTestId('generated-answer')).toHaveText(capture.answer);
  await page.keyboard.press('Escape');
  await expect(page.getByRole('group',{name:'Answer tokens'}).getByTestId('token-chip')).toHaveCount(capture.steps.length);
  const contextPage=capture.context.pages[0];
  await expect(page.getByTestId('attention-patch')).toHaveCount(contextPage.position_indices.length);
  await expect(page.getByTestId('similarity-patch')).toHaveCount(0);
  await expect(page.getByLabel('Attention layer',{exact:true})).toHaveValue('0');
  await expect(page.getByLabel('Attention head',{exact:true})).toHaveValue('0');
  await withPanelHidden(page,()=>expect(page.getByTestId('prediction-alignment')).toContainText(`(position ${capture.steps[0].source_index} →`));
  const assertAlignment=async()=>{
    const patch=capture.positions[contextPage.position_indices[0]];
    const bounds=await page.evaluate(index=>{
      const image=document.querySelector('.docattentionlens-canvas image')!.getBoundingClientRect();
      const patch=document.querySelector(`[data-testid="attention-patch"][data-position-index="${index}"]`)!.getBoundingClientRect();
      return {image:{left:image.left,top:image.top,width:image.width},patch:{left:patch.left,top:patch.top,width:patch.width}};
    },patch.index);
    const scale=bounds.image.width/contextPage.source_width;
    expect(Math.abs(bounds.patch.left-bounds.image.left-patch.bbox![0]*scale)).toBeLessThan(1);
    expect(Math.abs(bounds.patch.top-bounds.image.top-patch.bbox![1]*scale)).toBeLessThan(1);
    expect(Math.abs(bounds.patch.width-(patch.bbox![2]-patch.bbox![0])*scale)).toBeLessThan(1);
  };
  await assertAlignment();
  await page.getByRole('button',{name:'Zoom in',exact:true}).click();await assertAlignment();
  await setFit(page,'page');
  await page.getByLabel('Attention layer',{exact:true}).selectOption('35');
  await page.getByLabel('Attention head',{exact:true}).selectOption('15');
  const selected=capture.steps[capture.steps.length-1];
  await openDetailsTab(page,'Method');
  await page.getByLabel('Attention source',{exact:true}).selectOption(`step:${selected.step}`);
  const row=await page.request.get(`/api/v1/generation/captures/${capture.capture_id}/rows?step=${selected.step}&layer=35&head=15`).then(r=>r.json());
  await expectRowTotals(page,row.total_mass.toFixed(6));
  await showElements(page);
  await page.getByTestId('outline-element').first().click();
  await expect(page.getByTestId('element-context-evidence')).toContainText('Page supplied as an image');
  await expect(page.getByTestId('element-attention-evidence')).toContainText('original-domain mass');
  await withPanelHidden(page,()=>expect(page.getByTestId('prediction-alignment')).toContainText(`→ ${selected.predicted_index})`));
  const patch=page.getByTestId('attention-patch').first();
  const imageMaximum=Math.max(...capture.positions.filter(p=>p.role==='image'&&p.bbox&&p.index<row.target_count).map((p:{index:number})=>row.values[p.index]));
  await expect(patch).toHaveAttribute('fill-opacity',String(row.values[contextPage.position_indices[0]]/imageMaximum*.65));
  await showRetrievalScores(page);
  await expect(page.getByTestId('similarity-patch')).toHaveCount(1024);
  await expect(page.getByTestId('attention-patch')).toHaveCount(0);
  await page.getByRole('button',{name:'← Back to generation',exact:true}).click();
  await expect(page.getByTestId('attention-patch')).toHaveCount(contextPage.position_indices.length);
  await openTool(page,'Notes & name');
  await page.getByLabel('Investigation name').fill('Browser: real generator recovery');
  await saveInvestigation(page);
  await expect(page).toHaveURL(/\/attention-tool\/investigation_/);
  await page.reload();
  await expect(page.getByLabel('Attention layer',{exact:true})).toHaveValue('35');
  await expect(page.getByLabel('Attention head',{exact:true})).toHaveValue('15');
  await openDetailsTab(page,'Method');
  await expect(page.getByLabel('Attention source',{exact:true})).toHaveValue(`step:${selected.step}`);
  await expectRowTotals(page,row.total_mass.toFixed(6));
  await openTool(page, 'Provenance');
  // Provenance and receipt is Details › Run.
  await expect(page.getByAltText(`Consumed model input for page ${contextPage.page_number}`)).toBeVisible();
  await setDetails(page,false);
  if(capture.context.omitted_page_ids.length){
    await choosePage(page,capture.context.omitted_page_ids[0]);
    await expect(page.getByTestId('attention-patch')).toHaveCount(0);
    await expect(page.getByText('The displayed document page was not supplied to this generator.',{exact:false})).toBeVisible();
    await choosePage(page,contextPage.page_id);
  }
  await page.screenshot({path:'../var/screenshots/generation-desktop.png',fullPage:true});
  await page.setViewportSize({width:390,height:844});
  expect(await page.evaluate(()=>document.documentElement.scrollWidth>innerWidth)).toBe(false);
  await page.screenshot({path:'../var/screenshots/generation-mobile.png',fullPage:true});
});

test('generator launch consumes chosen retrieval context and returns real attention',async({page})=>{
  test.setTimeout(240000);
  const plan=await page.request.get('/api/v1/generation/plan').then(r=>r.json());
  test.skip(!plan.available,'Generator GPU execution unavailable; saved-evidence test still runs');
  const records=await page.request.get('/api/v1/retrieval/captures').then(r=>r.json());
  // The generator is supplied pages chosen from a saved retrieval ranking, so the capture must rank at least one page.
  const retrieval=records.items?.find((c:{pages?:unknown[]})=>(c.pages?.length??0)>0);
  requireSaved(!retrieval,'A saved real retrieval capture with at least one ranked page is required');
  // A generation investigation of that retrieval capture (no generator capture yet) opens on the supporting
  // retrieval scores; the top bar's pill returns to generation, which acquires one.
  await page.goto('/attention-tool?capture='+retrieval.capture_id+'&role=generation');
  await page.getByRole('button',{name:'← Back to generation',exact:true}).click();
  await openTool(page, 'Open / acquire evidence');
  await expect(page.getByText('Choose context pages and generate an answer',{exact:true})).toBeVisible();
  await page.getByRole('button',{name:'Generate answer and capture attention',exact:true}).click();
  await page.getByRole('button',{name:'Close panel',exact:true}).click();
  await expect(page.getByRole('group',{name:'Answer tokens'})).toBeVisible({timeout:180000});
  await expect(page.getByTestId('capture-menu').locator('.docattentionlens-capture-role')).toHaveText('Generation');
  await expect(page.getByTestId('attention-patch').first()).toBeVisible();
  await openTool(page, 'Open / acquire evidence');
  await page.getByText('Execution budget',{exact:true}).click();
  await expect(page.getByText('Evaluation labels and OCR are excluded.',{exact:false})).toBeVisible();
});
