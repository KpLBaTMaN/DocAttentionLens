import { test,expect,type Page } from '@playwright/test';
import { readFile } from 'node:fs/promises';
import { chooseColourMode, clickPage, openTool, methodChoice, chooseMethod, openMethodPicker, showEvidence, openDetailsTab } from './workspaceHelpers';

async function checkedCapture(page:Page) {
  const captures=(await page.request.get('/api/v1/generation/captures?limit=25').then(r=>r.json())).items;
  for(const capture of captures){
    const response=await page.request.get(`/api/v1/cases/${capture.case_id}`);
    if(!response.ok())continue;
    const record=await response.json();
    if(record.support_page_ids.some((id:string)=>capture.context.selected_page_ids.includes(id)))return capture;
  }
  throw new Error('Requires a saved capture with a supplied support page');
}

test('checked examples open saved evidence, and review shows the exact pages before launch',async({page})=>{
  let acquisitions=0;
  await page.route(/\/api\/v1\/(generation|retrieval)\/runs$/,route=>{acquisitions++;return route.abort();});
  await page.goto('/attention-tool');
  await page.getByRole('button',{name:'Open an example',exact:true}).click();
  const examples=page.getByRole('region',{name:'Saved examples'});
  await expect(examples.getByRole('button').first()).toBeVisible();
  await expect(page.getByLabel('Generation model',{exact:true})).toHaveCount(0);
  await examples.getByRole('button').first().click();
  await expect(methodChoice(page)).toHaveAttribute('data-method-key',/^attention\.row\//);
  const captureId=new URL(page.url()).searchParams.get('generation')!;
  const capture=await page.request.get(`/api/v1/generation/captures/${captureId}`).then(r=>r.json());
  page.on('dialog',d=>d.accept());
  await page.goto(`/attention-tool?case=${capture.case_id}&configure=1`);
  // A case link opens on its question and pages, supplied directly by default.
  await expect(page.getByRole('heading',{name:'Choose a question and pages'})).toBeVisible();
  const next=page.getByRole('button',{name:/^(Continue|Review setup)$/});
  await expect(next).toBeEnabled();
  await next.click();
  // The review step is the confirmation: it lists every supplied page, in order, beside the launch button.
  const selected=page.locator('.docattentionlens-setup-pages input:checked');
  await expect(page.locator('.docattentionlens-review-pages figure').first()).toBeVisible();
  await expect(page.getByRole('checkbox',{name:/I reviewed/})).toHaveCount(0);
  await expect(page.getByRole('button',{name:'Run investigation',exact:true})).toBeEnabled();
  await page.getByRole('button',{name:'Back',exact:true}).click();
  // Evidence defaults can supply more than one page.
  while(await selected.count())await selected.first().uncheck();
  await expect(next).toBeDisabled();
  expect(acquisitions).toBe(0);
});

test('expanded comparisons link viewports, share colours and restore presentation when saved',async({page})=>{
  const capture=await checkedCapture(page);
  await page.setViewportSize({width:1440,height:1000});
  await page.goto('/attention-tool?generation='+capture.capture_id);
  await chooseMethod(page,'attention.head_by_head/captured_prediction_v1');
  // The method's own output is in Details › Result (left open, it reopens there).
  await showEvidence(page,'Method result');
  await expect(page.getByTestId('method-result')).toBeVisible();
  await chooseColourMode(page,'Heatmap');
  await page.getByRole('button',{name:'Expand comparisons',exact:true}).click();
  const dialog=page.getByRole('dialog',{name:'Compare attention maps'});
  await expect(dialog.locator('figure')).toHaveCount(2);
  expect((await dialog.getByRole('img').first().boundingBox())!.height).toBeGreaterThanOrEqual(450);
  await dialog.getByLabel('Maps per view').selectOption('4');
  await expect(dialog.locator('figure')).toHaveCount(4);
  expect((await dialog.getByRole('img').first().boundingBox())!.height).toBeGreaterThanOrEqual(260);
  const display=dialog.getByRole('group',{name:'Attention display',exact:true});
  await display.getByRole('radio',{name:'Overlay',exact:true}).click();
  await display.getByLabel('Heatmap scale').selectOption('fixed');
  await display.getByLabel('Fixed scale maximum').fill('0.006');
  await expect(dialog.getByTestId('comparison-scale')).toContainText('fixed bound');
  await dialog.getByLabel('Linked comparison zoom').selectOption('2');
  const maps=dialog.getByRole('img');
  const boxes=await maps.evaluateAll(nodes=>nodes.map(n=>n.getAttribute('viewBox')));
  expect(new Set(boxes).size).toBe(1);
  const box=(await maps.first().boundingBox())!;
  await page.mouse.move(box.x+box.width*.5,box.y+box.height*.5);await page.mouse.down();await page.mouse.move(box.x+box.width*.6,box.y+box.height*.55);await page.mouse.up();
  const after=await maps.evaluateAll(nodes=>nodes.map(n=>n.getAttribute('viewBox')));
  expect(new Set(after).size).toBe(1);expect(after[0]).not.toBe(boxes[0]);
  const comparisonDownload=page.waitForEvent('download');
  await dialog.getByRole('button',{name:'Export comparison figure (SVG)'}).click();
  const svg=await readFile((await (await comparisonDownload).path())!,'utf8');
  expect(svg).toContain('visible_comparisons');expect(svg).toContain('Consumed model input');expect(svg).toContain(capture.capture_id);
  expect(svg.match(/href="data:image\//g)?.length).toBeGreaterThanOrEqual(4);
  await dialog.getByLabel('Linked comparison zoom').selectOption('1');
  await page.screenshot({path:'../var/screenshots/usability-comparisons.png'});
  await dialog.getByLabel('Linked comparison zoom').selectOption('2');
  await dialog.getByRole('button',{name:'Close panel'}).click();
  await page.getByRole('button',{name:'Save investigation',exact:true}).click();
  await expect(page).toHaveURL(/investigation_/);await page.reload();
  await page.getByRole('button',{name:'Expand comparisons',exact:true}).click();
  await expect(dialog.getByLabel('Maps per view')).toHaveValue('4');
  await expect(dialog.getByLabel('Linked comparison zoom')).toHaveValue('2');
  expect(await dialog.getByRole('img').first().getAttribute('viewBox')).toBe(after[0]);
  await page.setViewportSize({width:390,height:844});
  expect(await page.evaluate(()=>document.documentElement.scrollWidth<=innerWidth+1)).toBe(true);
  await expect(dialog.getByLabel('Maps per view')).toBeVisible();
});

test('exports include the displayed selection and portable page image with scientific context',async({page})=>{
  const capture=await checkedCapture(page);
  await page.goto('/attention-tool?generation='+capture.capture_id);
  await expect(page.getByTestId('attention-patch').first()).toBeVisible({timeout: 20_000});
  await expect(page.getByTestId('column-context')).toHaveCount(0);
  await openTool(page,'Provenance');
  const receipt=page.getByRole('region',{name:'Research receipt'});
  const jsonDownload=page.waitForEvent('download');
  await receipt.getByRole('button',{name:'Export receipt (JSON)'}).click();
  const json=JSON.parse(await readFile((await (await jsonDownload).path())!,'utf8'));
  expect(json.evidence.capture_id).toBe(capture.capture_id);
  expect(json.analysis.method_id).toBe('attention.row');
  expect(json.evidence.arrays).toEqual(capture.arrays);
  const svgDownload=page.waitForEvent('download');
  await receipt.getByRole('button',{name:'Export page figure (SVG)'}).click();
  const svg=await readFile((await (await svgDownload).path())!,'utf8');
  expect(svg).toContain('<metadata>');expect(svg).toContain(capture.capture_id);
  expect(svg).toMatch(/href="data:image\//);expect(svg).not.toMatch(/href="\/api\//);
  // A generator patch has no row of its own: clicking it shows its incoming raw column (Details › Result).
  await clickPage(page,.5,.5);
  await openDetailsTab(page,'Result');
  await expect(page.getByTestId('incoming-evidence')).toContainText('values across sources do not sum to one');
  await expect(page.getByTestId('column-context')).toContainText('image patch');
  await expect(page.getByTestId('incoming-summary')).toContainText('values across sources do not sum to one');
  // Incoming columns have one evidence view, so the view tabs are replaced by its name.
  await expect(page.getByRole('tablist',{name:'Evidence view'})).toHaveCount(0);
});


test('method browsing exposes canonical operations, availability and statistical output',async({page})=>{
  const capture=await checkedCapture(page);
  await page.goto('/attention-tool?generation='+capture.capture_id);
  // The method bar opens Details › Method; its preview says what the highlighted method reports.
  const browser=await openMethodPicker(page);
  const about=browser.getByTestId('method-preview');
  await expect(about).toContainText('Raw attention');
  await expect(about.getByRole('link',{name:/Full description in Methods/})).toHaveAttribute('href',/\/toolbox\/attention\.row\?variant=/);
  await browser.getByRole('combobox',{name:'Find a method'}).fill('entropy');
  await expect(about.getByTestId('method-readiness')).toContainText('Ready');
  await about.getByRole('button',{name:/Use.*entropy/i}).click();
  await expect(page.getByTestId('result-elsewhere')).toContainText('Result: statistics');
  await openDetailsTab(page,'Result');
  await expect(page.getByTestId('method-result')).toContainText('Shannon entropy');
  await expect(page.getByTestId('attention-patch')).toHaveCount(0);
  await page.screenshot({path:'../var/screenshots/usability-statistics.png'});
});
