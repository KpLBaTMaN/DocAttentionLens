import { test, expect } from '@playwright/test';
import { chooseMethod, detailsPanel, methodChoice, openTool, showPageList, methodBrowse, openDetailsTab } from './workspaceHelpers';
import { findGenerationCapture, requireSaved } from './savedEvidence';

test('entry offers examples or a new run, persists the retriever option and reopens saved attention without inference',async({page})=>{
  let acquisitions=0;
  await page.route(/\/api\/v1\/(generation|retrieval)\/runs$/,route=>{acquisitions++;return route.abort();});
  await page.goto('/attention-tool');
  await expect(page.getByRole('heading',{name:'Investigate document attention'})).toBeVisible();
  await expect(page.getByLabel('Generation model',{exact:true})).toHaveCount(0);
  await page.screenshot({path:'../var/screenshots/investigation-setup-entry.png',fullPage:true});
  await page.getByRole('button',{name:'New investigation',exact:true}).click();
  await expect(page.getByLabel('Generation model',{exact:true})).toHaveValue('builtin');
  await page.screenshot({path:'../var/screenshots/investigation-setup-model.png',fullPage:true});
  // Pages are supplied directly unless the researcher asks a retriever to rank them first.
  const retrieverFirst=page.getByRole('checkbox',{name:/Let a retriever choose the pages first/});
  await expect(retrieverFirst).not.toBeChecked();
  await retrieverFirst.check();
  // A reload in the same tab keeps the open setup and the stored choice.
  await page.reload();
  await expect(retrieverFirst).toBeChecked();
  await page.setViewportSize({width:390,height:844});
  expect(await page.evaluate(()=>document.documentElement.scrollWidth<=innerWidth+1)).toBe(true);
  await page.screenshot({path:'../var/screenshots/investigation-setup-mobile.png',fullPage:true});
  page.once('dialog',dialog=>dialog.accept());
  await page.getByRole('button',{name:'Start over',exact:true}).click();
  await expect(page.getByRole('heading',{name:'Investigate document attention'})).toBeVisible();
  await page.getByRole('link',{name:'Library',exact:true}).click();
  await page.getByRole('link',{name:'Runs and captures',exact:true}).click();
  await expect(page).toHaveURL(/\/artifacts\/runs$/);
  await page.locator('.docattentionlens-record-list a').first().click();
  await expect(page).toHaveURL(/(generation|capture)=/);
  expect(acquisitions).toBe(0);
});

test('dataset questions default to designated evidence pages with labelled alternatives',async({page})=>{
  let acquisitions=0;
  await page.route(/\/api\/v1\/(generation|retrieval)\/runs$/,route=>{acquisitions++;return route.abort();});
  const captures=(await page.request.get('/api/v1/generation/captures?limit=25').then(r=>r.json())).items;
  let record:{case_id:string;page_ids:string[];support_page_ids:string[]}|undefined;
  for(const capture of captures){
    const response=await page.request.get(`/api/v1/cases/${capture.case_id}`);
    const candidate=response.ok()?await response.json():undefined;
    if(candidate?.support_page_ids.length===1&&candidate.page_ids.length>8){record=candidate;break;}
  }
  requireSaved(!record,'Requires a saved case with one designated evidence page in a document of more than 8 pages');
  await page.goto(`/attention-tool?case=${record!.case_id}&configure=1`);
  await page.locator('.docattentionlens-setup-steps').getByRole('button',{name:/Model/}).click();
  await page.getByRole('radiogroup',{name:'Model type'}).getByRole('radio',{name:/Retrieval/}).check();
  await page.locator('.docattentionlens-setup-steps').getByRole('button',{name:/Question and pages/}).click();
  const selection=page.getByRole('group',{name:'Page selection'});
  const cards=page.locator('.docattentionlens-setup-pages label');
  await expect(selection.getByRole('button',{name:'Evidence pages'})).toHaveAttribute('aria-pressed','true');
  await expect(cards).toHaveCount(1);
  await expect(cards.first()).toHaveClass(/is-evidence/);
  await expect(cards.first().locator('input')).toBeChecked();
  await expect(page.getByLabel('Document attention page',{exact:true}).locator('option:checked')).toContainText('evidence');
  await page.screenshot({path:'../var/screenshots/investigation-setup-evidence.png',fullPage:true});
  await selection.getByRole('button',{name:'Evidence + distractors'}).click();
  await expect(page.locator('.docattentionlens-evidence-presets')).toContainText(/(\d+) of \1 pages selected/);
  await expect(page.locator('.docattentionlens-setup-pages .is-evidence input')).toBeChecked();
  // Click once: the filtered view removes an unchecked distractor.
  await page.locator('.docattentionlens-setup-pages label:not(.is-evidence) input:checked').first().click();
  await expect(page.locator('.docattentionlens-evidence-presets')).toContainText('Custom selection.');
  await page.getByRole('button',{name:/^Choose from all \d+ pages$/}).click();
  await expect(cards).toHaveCount(record!.page_ids.length);
  await page.getByRole('button',{name:/^(Continue|Review setup)$/}).click();
  await expect(page.locator('.docattentionlens-setup-review')).toContainText('1 of 1 included');
  expect(acquisitions).toBe(0);
});

test('generator context attention, method and axes survive saving and the supporting retrieval scores',async({page})=>{
  // Supporting retrieval scores need a generator capture acquired from a retrieval capture; the axes checks use layer 2, head 3.
  const found=await findGenerationCapture(page.request,c=>Boolean(c.retrieval_capture_id)&&c.layer_count>2&&c.query_head_count>3);
  requireSaved(!found,'Requires a saved generator capture acquired from a retrieval capture (with at least 3 layers and 4 query heads); generate one from a retrieval capture in the workspace or with scripts/generation_smoke.py');
  const capture=found!;
  let acquisitions=0;
  await page.route(/\/api\/v1\/(generation|retrieval)\/runs$/,route=>{acquisitions++;return route.abort();});
  await page.goto('/attention-tool?generation='+capture.capture_id);
  const picker=methodChoice(page);
  await expect(methodBrowse(page)).toBeEnabled();
  await page.getByLabel('Attention layer',{exact:true}).selectOption('2');
  await page.getByLabel('Attention head',{exact:true}).selectOption('3');
  await expect(page.getByTestId('capture-menu').locator('.docattentionlens-capture-role')).toHaveText('Generation');
  await showPageList(page);
  await expect(page.getByRole('region',{name:'Generator attention to context'})).toBeVisible();
  await expect(page.getByTestId('attention-patch').first()).toBeVisible();
  await expect(page.getByTestId('retrieval-winner')).toHaveCount(0);
  await expect(page.getByLabel('Attention layer',{exact:true})).toHaveValue('2');
  await expect(page.getByLabel('Attention head',{exact:true})).toHaveValue('3');
  await chooseMethod(page,'attention.topk_mass/captured_prediction_v1');
  await detailsPanel(page);
  await page.getByLabel('Top-k targets',{exact:true}).fill('7');
  await openDetailsTab(page,'Result');
  await expect(page.getByTestId('method-result').getByText('Top-7 attention mass',{exact:true})).toBeVisible();
  await expect(picker).toHaveAttribute('data-method-key','attention.topk_mass/captured_prediction_v1');
  await openDetailsTab(page,'Method');
  await expect(page.getByLabel('Top-k targets',{exact:true})).toHaveValue('7');
  await page.getByRole('button',{name:'Save investigation',exact:true}).click();
  await expect(page).toHaveURL(/investigation_/);await page.reload();
  await expect(picker).toHaveAttribute('data-method-key','attention.topk_mass/captured_prediction_v1');
  await expect(page.getByLabel('Attention layer',{exact:true})).toHaveValue('2');
  await expect(page.getByLabel('Attention head',{exact:true})).toHaveValue('3');
  // Details opens closed after a reload; the method's settings are in Details › Method.
  await detailsPanel(page);
  await expect(page.getByLabel('Top-k targets',{exact:true})).toHaveValue('7');
  await page.screenshot({path:'../var/screenshots/generator-context-lens.png'});
  await openTool(page,'Provenance');
  await page.getByRole('button',{name:'Inspect supporting retrieval scores'}).click();
  await expect(page.getByTestId('retrieval-winner')).toBeVisible();
  await expect(page.getByTestId('capture-menu').locator('.docattentionlens-capture-role')).toHaveText('Retrieval');
  await page.getByRole('button',{name:'← Back to generation',exact:true}).click();
  await expect(page.getByTestId('retrieval-winner')).toHaveCount(0);
  await expect(page.getByTestId('capture-menu').locator('.docattentionlens-capture-role')).toHaveText('Generation');
  await expect(picker).toHaveAttribute('data-method-key','attention.topk_mass/captured_prediction_v1');
  // Details › Method's own button returns the same way.
  await openTool(page,'Provenance');
  await page.getByRole('button',{name:'Inspect supporting retrieval scores'}).click();
  await openDetailsTab(page,'Method');
  await page.getByRole('button',{name:'Show captured attention'}).click();
  await expect(page.getByRole('region',{name:'Generation analysis'})).toBeVisible();
  expect(acquisitions).toBe(0);
});

test('document upload freezes a custom question and reviews direct context without retrieval',async({page})=>{
  await page.goto('/attention-tool');
  await page.getByRole('button',{name:'New investigation',exact:true}).click();
  await expect(page.getByRole('checkbox',{name:/Let a retriever choose the pages first/})).not.toBeChecked();
  await page.getByRole('button',{name:/^(Continue|Review setup)$/}).click();
  await page.getByRole('button',{name:'Your own documents'}).click();
  const png=Buffer.from('iVBORw0KGgoAAAANSUhEUgAAAAEAAAABCAQAAAC1HAwCAAAAC0lEQVR42mP8/x8AAwMCAO+jP2kAAAAASUVORK5CYII=','base64');
  await page.getByLabel('Upload documents',{exact:true}).setInputFiles({name:'setup-browser-fixture.png',mimeType:'image/png',buffer:png});
  await expect(page.getByLabel('Question / prompt')).toBeVisible();
  await page.getByLabel('Question / prompt').fill('What is shown in this uploaded page?');
  await page.getByRole('button',{name:'Use this question'}).click();
  await expect(page.getByText('Question for this run: What is shown in this uploaded page?',{exact:true})).toBeVisible();
  await expect(page.getByRole('button',{name:'Retrieve selected pages'})).toHaveCount(0);
  await page.getByRole('button',{name:/^(Continue|Review setup)$/}).click();
  const review=page.getByRole('heading',{name:'Check what the generator will see'});
  await expect(review).toBeVisible();
  await expect(page.getByText('Directly supplied pages',{exact:true})).toBeVisible();
  await expect(page.locator('.docattentionlens-setup-review')).toContainText('Page 1');
  await expect(page.locator('.docattentionlens-review-pages figure')).toHaveCount(1);
  await page.evaluate(()=>scrollTo(0,0));
  await page.screenshot({path:'../var/screenshots/investigation-setup-review.png',fullPage:true});
  await page.reload();
  await expect(review).toBeVisible();
  // A later visit offers the unfinished draft on the entry screen instead of reopening it.
  await page.evaluate(()=>sessionStorage.clear());
  await page.reload();
  await expect(page.getByRole('heading',{name:'Investigate document attention'})).toBeVisible();
  await page.getByRole('region',{name:'Unfinished setup'}).getByRole('button',{name:'Resume',exact:true}).click();
  await expect(review).toBeVisible();
  // A failed submission preserves the reviewed prompt, pages and request identity.
  const sent:{capture_full_attention?:boolean}[]=[];
  await page.route('**/api/v1/generation/runs',route=>{sent.push(route.request().postDataJSON());return route.fulfill({status:503,json:{error:{message:'Verification fixture: worker unavailable.'}}});});
  await page.getByRole('button',{name:'Run investigation',exact:true}).click();
  await expect(page.getByRole('alert')).toContainText('Verification fixture');
  // Full attention matrices are opt-in: an unchanged setup asks for the default, small capture.
  expect(sent.at(-1)?.capture_full_attention).toBe(false);
  await expect(page.locator('.docattentionlens-setup-review')).toContainText('What is shown in this uploaded page?');
});

test('full attention matrices are opt-in, estimated, persisted, reviewed and sent with the run',async({page})=>{
  const capture=await findGenerationCapture(page.request,c=>c.context.pages.length>=1);
  requireSaved(!capture,'Requires a saved generator capture for its case');
  // The plan is fixed so the launch button is enabled; the run request itself is intercepted and never reaches the worker.
  await page.route(/\/api\/v1\/generation\/plan/,route=>route.fulfill({json:{available:true,reason:'Verification fixture: plan available.'}}));
  const sent:{capture_full_attention?:boolean;case_id?:string}[]=[];
  await page.route(/\/api\/v1\/(generation|retrieval)\/runs$/,route=>{sent.push(route.request().postDataJSON());return route.fulfill({status:503,json:{error:{message:'Verification fixture: no run started.'}}});});
  // An earlier v2 draft without the option opens with it unchecked.
  await page.goto('/attention-tool');
  await page.evaluate(caseId=>{localStorage.setItem('docattentionlens-investigation-setup-v2',JSON.stringify({step:0,model:'builtin',role:'generation',focus:'generation',mode:'dataset',caseId,prompt:'',candidates:[],supplied:[],system:'Answer the question using only the supplied document page images. If the images do not provide enough evidence, say so. Give a concise answer.'}));},capture!.case_id);
  await page.goto(`/attention-tool?case=${capture!.case_id}&configure=1`);
  await page.locator('.docattentionlens-setup-steps').getByRole('button',{name:/Model/}).click();
  const option=page.getByRole('checkbox',{name:/Save full attention matrices/});
  await expect(option).not.toBeChecked();
  const hint=page.locator('label',{has:option});
  await expect(hint).toContainText('Enables rollout, attention flow and graphs, attention structure, and image-patch sources.');
  // It also keeps every answer token's row: the estimate is an upper bound with up to 64 decode rows.
  await expect(hint).toContainText("every answer token's attention row, so every answer token can be selected");
  await expect(hint).toContainText(/Estimated extra storage: up to about \d+(\.\d+)? GB for Qwen2\.5-VL-3B-Instruct with \d supplied pages?\./);
  const summary=page.getByTestId('capture-summary');
  await expect(summary).toContainText(/Attention rows for the first \d+ answer tokens/);
  await page.setViewportSize({width:1440,height:900});
  await hint.scrollIntoViewIfNeeded();
  await page.screenshot({path:'../var/screenshots/investigation-setup-full-attention.png'});
  await option.check();
  await expect(summary).toContainText(/Every answer token's attention row \(up to \d+\)/);
  await page.reload();
  await expect(option).toBeChecked();
  await expect(hint).toContainText(/up to about \d+(\.\d+)? GB/);
  await page.setViewportSize({width:375,height:812});
  await hint.scrollIntoViewIfNeeded();
  await expect(hint).toBeInViewport();
  expect(await page.evaluate(()=>document.documentElement.scrollWidth<=innerWidth+1)).toBe(true);
  await page.screenshot({path:'../var/screenshots/investigation-setup-full-attention-mobile.png'});
  await page.setViewportSize({width:1440,height:900});
  await page.getByRole('button',{name:/^(Continue|Review setup)$/}).click();
  await page.getByRole('button',{name:/^(Continue|Review setup)$/}).click();
  await expect(page.locator('.docattentionlens-setup-review')).toContainText(/Full attention matrices\s*Saved · up to about \d+(\.\d+)? GB extra/);
  await expect(page.locator('.docattentionlens-setup-review')).toContainText(/Captured evidence\s*Every answer token's attention row/);
  await page.getByRole('button',{name:'Run investigation',exact:true}).click();
  await expect(page.getByRole('alert')).toContainText('Verification fixture: no run started.');
  expect(sent.at(-1)).toMatchObject({case_id:capture!.case_id,capture_full_attention:true});
  // Turning it off again sends the default.
  await page.locator('.docattentionlens-setup-steps').getByRole('button',{name:/Model/}).click();
  await option.uncheck();
  await page.locator('.docattentionlens-setup-steps').getByRole('button',{name:/Review/}).click();
  await expect(page.locator('.docattentionlens-setup-review')).toContainText('Not saved · prediction and question rows only');
  await page.getByRole('button',{name:'Run investigation',exact:true}).click();
  await expect.poll(()=>sent.at(-1)?.capture_full_attention).toBe(false);
  expect(sent.length).toBe(2);
});
