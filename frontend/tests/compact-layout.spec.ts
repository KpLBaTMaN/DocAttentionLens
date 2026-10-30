import { test, expect } from '@playwright/test';
import { readFileSync, writeFileSync } from 'node:fs';
import { showElements, expectRowTotals, readRowTotals, pageArea, setDetails, setFit, canvasSizing, choosePage, openTool, openDisplayOptions, closeDisplaySettings, showEvidence, filmstrip, openDetailsTab, detailsTab, saveInvestigation, bottomBar } from './workspaceHelpers';

const meta=JSON.parse(readFileSync('../var/verification/generation-smoke.json','utf8'));
const baseline=JSON.parse(readFileSync('../var/verification/compact-layout-baseline.json','utf8'));

test('the page area gets most of the first screen with Details closed, on desktop and phone',async({page})=>{
  await page.setViewportSize({width:1440,height:900});
  await page.goto('/attention-tool?generation='+meta.capture_id);
  await expect(page.getByTestId('attention-patch').first()).toBeVisible({timeout: 20_000});
  await setDetails(page,false);
  // Redesign target: at least 75% of the viewport height, starting high on the screen.
  await expect.poll(async()=>(await pageArea(page)).height).toBeGreaterThanOrEqual(900*.75);
  expect((await pageArea(page)).top).toBeLessThan(200);
  expect(await page.evaluate(()=>document.documentElement.scrollWidth<=innerWidth+1)).toBe(true);
  await page.setViewportSize({width:375,height:812});
  await page.evaluate(()=>window.scrollTo(0,0));
  await expect.poll(async()=>(await pageArea(page)).top).toBeLessThan(140);
  expect((await pageArea(page)).height).toBeGreaterThan(812*.4);
  expect(await page.evaluate(()=>document.documentElement.scrollWidth<=innerWidth+1)).toBe(true);
});

test('Fit page gives the page the first screen; evidence and focus keep its size and panels recover',async({page})=>{
  await page.setViewportSize({width:1440,height:900});
  await page.goto('/attention-tool/'+meta.investigation_id);
  await expect(page.getByTestId('attention-patch').first()).toBeVisible({timeout: 20_000});
  const image=page.getByTestId('page-canvas').locator('image');
  // The locked layout fitted this page at about 410 px; the scrolling layout gives it the first screen below the pinned bands.
  await expect.poll(async()=>image.evaluate(e=>e.getBoundingClientRect().height)).toBeGreaterThan(520);
  await expect.poll(()=>page.locator('.docattentionlens-canvas-legend').evaluate(e=>e.getBoundingClientRect().bottom<=innerHeight+1)).toBe(true);
  // A saved investigation whose inspector was open reopens with Details open.
  await setDetails(page,true);
  await showElements(page);
  const before=await image.boundingBox();
  const canvasWidth=await page.getByTestId('page-canvas').evaluate(e=>e.clientWidth);
  const selection=await page.getByTestId('selected-element').textContent();
  const mass=await readRowTotals(page);
  const query=await page.getByTestId('frozen-question').textContent();
  await page.screenshot({path:'../var/screenshots/compact-layout-generation.png'});
  const after={viewport:baseline.viewport,image:{width:before!.width,height:before!.height},heightRatio:before!.height/baseline.image.height};
  writeFileSync('../var/verification/compact-layout-comparison.json',JSON.stringify({baseline,after},null,2));
  // The method's output is a Details tab, so reading it keeps the page's size.
  await showEvidence(page,'Complete causal row');
  expect(Math.abs((await image.boundingBox())!.height-before!.height)).toBeLessThan(2);
  await expect(filmstrip(page)).toBeVisible();
  await page.getByRole('button',{name:'Image focus',exact:true}).click();
  await expect(page.getByRole('button',{name:'Image focus',exact:true})).toHaveAttribute('aria-pressed','true');
  // Image focus keeps the method sentence, which names the shown answer token, and the question in the top bar.
  await expect(page.getByTestId('row-summary')).toContainText(/What “.+” attends to/);
  await expect(filmstrip(page)).toHaveCount(0);
  // Image focus gives the page the filmstrip's 72 px and Details' 360 px.
  await expect.poll(()=>page.getByTestId('page-canvas').evaluate(e=>e.clientWidth)).toBeGreaterThan(canvasWidth+400);
  await expect(page.getByTestId('frozen-question')).toHaveText(query!);
  // The image share stays on the page in image focus; exact totals are in Details, which image focus hides.
  await expect(page.getByTestId('image-share')).toBeInViewport();
  await expect(page.getByLabel('Attention layer',{exact:true})).toHaveValue('35');
  await expect(page.getByTestId('display-scale')).toBeInViewport();
  await expect(page.getByTestId('row-summary')).toBeInViewport();
  await page.screenshot({path:'../var/screenshots/compact-layout-image-focus.png'});
  await page.getByRole('button',{name:'Image focus',exact:true}).click();
  await expect(page.getByRole('button',{name:'Image focus',exact:true})).toHaveAttribute('aria-pressed','false');
  await showElements(page);
  await expect(page.getByTestId('selected-element')).toHaveText(selection!);
  await expectRowTotals(page,mass);
  // Details closes and reopens from its toggle and with D; the answer tokens stay in the source line.
  await setDetails(page,false);
  await expect(page.getByRole('complementary',{name:'Attention panel',exact:true})).toBeHidden();
  await page.keyboard.press('d');
  await expect(page.getByRole('complementary',{name:'Attention panel',exact:true})).toBeVisible();
  await expect(page.getByRole('group',{name:'Answer tokens'})).toBeVisible();
  // Menus close on Escape and return focus to their button.
  await page.getByRole('button',{name:'Investigation menu',exact:true}).click();
  await page.getByRole('menuitem',{name:'Provenance and receipt',exact:true}).focus();
  await page.keyboard.press('Escape');
  await expect(page.getByRole('menuitem',{name:'Provenance and receipt',exact:true})).toBeHidden();
  await expect(page.getByRole('button',{name:'Investigation menu',exact:true})).toBeFocused();
  await expect(page.getByTestId('frozen-question')).toHaveText(query!);
});

test('the page is sized by fit mode at laptop, desktop and phone widths while instruments stay pinned',async({page})=>{
  const results:Record<string,unknown>={};
  const imageBox=()=>page.getByTestId('page-canvas').first().locator('image').evaluate(e=>{const r=e.getBoundingClientRect();return {width:Math.round(r.width),height:Math.round(r.height)};});
  await page.setViewportSize({width:1440,height:900});
  await page.goto('/attention-tool/'+meta.investigation_id);
  await expect(page.getByTestId('attention-patch').first()).toBeVisible({timeout: 20_000});
  // Locked-layout baselines for this page: about 250, 410 and 520 px tall at these sizes.
  for(const [width,height,minimum] of [[1366,768,380],[1440,900,520],[1920,1080,700]] as const){
    await page.setViewportSize({width,height});
    await setFit(page,'page');
    await page.evaluate(()=>window.scrollTo(0,0));
    // Fit page: the whole page and its legend share the first screen.
    await expect.poll(async()=>(await imageBox()).height).toBeGreaterThan(minimum);
    await expect.poll(()=>page.locator('.docattentionlens-canvas-legend').evaluate(e=>e.getBoundingClientRect().bottom<=innerHeight+1)).toBe(true);
    const fitPage=await imageBox();
    // Fit width: the page takes the column width and the window scrolls.
    await setFit(page,'width');
    await expect.poll(()=>page.getByTestId('page-canvas').first().evaluate(e=>Math.abs(e.querySelector('image')!.getBoundingClientRect().width-e.clientWidth))).toBeLessThan(1);
    const fitWidth=await imageBox();
    await page.evaluate(()=>{const canvas=document.querySelector('.docattentionlens-canvas')!.getBoundingClientRect();window.scrollBy(0,canvas.bottom-innerHeight+1);});
    // The top bar, source line, method sentence, legend, zoom pill and layer scrubber stay pinned while the page scrolls.
    for(const instrument of [page.getByRole('button',{name:'Save investigation',exact:true}),page.getByLabel('Attention layer',{exact:true}),page.getByRole('group',{name:'Source tokens'}),page.getByRole('button',{name:'Source line options',exact:true}),page.getByTestId('display-scale'),page.getByTestId('row-summary'),page.getByRole('button',{name:'Zoom in',exact:true})])await expect(instrument).toBeInViewport();
    expect(await page.evaluate(()=>document.documentElement.scrollWidth<=innerWidth+1)).toBe(true);
    results[`${width}x${height}`]={fitPage,fitWidth};
    if(width===1440)await page.screenshot({path:'../var/screenshots/scrolling-workspace-fit-width.png'});
  }
  await setFit(page,'page');
  await page.setViewportSize({width:390,height:844});
  await page.evaluate(()=>window.scrollTo(0,0));
  // Phones: the page always gets the column width; controls open as a sheet with the page still visible above it.
  await expect.poll(async()=>(await imageBox()).width).toBeGreaterThan(330);
  const phone=await imageBox();
  expect(await page.evaluate(()=>document.documentElement.scrollWidth<=innerWidth+1)).toBe(true);
  // Layer and head are on the scrubber under the page; Details opens as a sheet with the page still visible above it.
  await expect(page.getByLabel('Attention layer',{exact:true})).toBeVisible();
  await bottomBar(page).getByRole('button',{name:'Details',exact:true}).click();
  await detailsTab(page,'Result').click();
  await expect(page.getByLabel('Attention source',{exact:true})).toBeVisible();
  // The scrubber stays between the sheet and the bottom bar.
  await expect(page.getByLabel('Attention layer',{exact:true})).toBeInViewport();
  const sheetTop=await page.getByRole('complementary',{name:'Attention panel',exact:true}).evaluate(e=>e.getBoundingClientRect().top);
  expect(sheetTop).toBeGreaterThan(200);
  await page.screenshot({path:'../var/screenshots/scrolling-workspace-phone-controls.png'});
  await page.keyboard.press('Escape');
  await expect(page.getByLabel('Attention source',{exact:true})).toBeHidden();
  results['390x844']={fitPage:phone,controlsSheetTop:Math.round(sheetTop)};
  writeFileSync('../var/verification/scrolling-workspace.json',JSON.stringify(results,null,2));
});

test('fit width and focus element preserve image/attention transforms and reset on page change',async({page})=>{
  const capture=await page.request.get('/api/v1/generation/captures/'+meta.capture_id).then(r=>r.json());
  const saved=await page.request.get('/api/v1/investigations/'+meta.investigation_id).then(r=>r.json());
  const element=await page.request.get('/api/v1/elements/'+saved.view_state.selected_element_id).then(r=>r.json());
  await page.setViewportSize({width:1440,height:900});
  await page.goto('/attention-tool?generation='+capture.capture_id);
  await expect(page.getByTestId('attention-patch').first()).toBeVisible({timeout: 20_000});
  await showElements(page);
  await page.locator(`[data-testid="outline-element"][data-native-id="${element.native_id}"]`).click();
  const mass=await readRowTotals(page);
  const alignment=async()=>{
    const patch=capture.positions[capture.context.pages[0].position_indices[0]];
    const rect=await page.getByTestId('page-canvas').evaluate((e,index)=>{const image=e.querySelector('image')!.getBoundingClientRect();const box=e.querySelector(`[data-position-index="${index}"]`)!.getBoundingClientRect();return {image:{x:image.x,y:image.y,width:image.width},box:{x:box.x,y:box.y}};},patch.index);
    const scale=rect.image.width/capture.context.pages[0].source_width;
    expect(Math.abs(rect.box.x-rect.image.x-patch.bbox[0]*scale)).toBeLessThan(1);
    expect(Math.abs(rect.box.y-rect.image.y-patch.bbox[1]*scale)).toBeLessThan(1);
  };
  await page.getByRole('button',{name:'Focus element',exact:true}).click();
  await expect(canvasSizing(page)).toHaveAttribute('data-fit','manual');
  await expect.poll(()=>page.getByTestId('page-canvas').evaluate(e=>{const rect=e.querySelector('.docattentionlens-box.is-selected')!.getBoundingClientRect();const canvas=e.getBoundingClientRect();return Math.abs(rect.x+rect.width/2-canvas.x-canvas.width/2)+Math.abs(rect.y+rect.height/2-canvas.y-canvas.height/2);})).toBeLessThan(2);
  await alignment();
  await expectRowTotals(page,mass);
  await page.screenshot({path:'../var/screenshots/compact-layout-element-focus.png'});
  await setFit(page,'width');
  await expect.poll(()=>page.getByTestId('page-canvas').evaluate(e=>Math.abs(e.querySelector('image')!.getBoundingClientRect().width-e.clientWidth))).toBeLessThan(1);
  await page.setViewportSize({width:1600,height:1000});
  await expect.poll(()=>page.getByTestId('page-canvas').evaluate(e=>Math.abs(e.querySelector('image')!.getBoundingClientRect().width-e.clientWidth))).toBeLessThan(1);
  await alignment();
  await expectRowTotals(page,mass);
  // The input view moved from the View box into Details › Display (the legend's Display button).
  await (await openDisplayOptions(page)).getByLabel('Image view').selectOption('split');
  await closeDisplaySettings(page);
  await expect(page.getByTestId('page-canvas')).toHaveCount(2);
  await expect.poll(()=>page.getByTestId('page-canvas').first().evaluate(e=>Math.abs(e.querySelector('image')!.getBoundingClientRect().width-e.clientWidth))).toBeLessThan(1);
  await choosePage(page,capture.context.pages[1].page_id);
  await expect(canvasSizing(page)).toHaveAttribute('data-fit','page');
  await expect(page.getByTestId('selected-element')).toHaveCount(0);
  await expect(page.getByTestId('attention-patch')).toHaveCount(capture.context.pages[1].position_indices.length*2);
});

test('Details open on Result saves as an open inspector and dock, and reopens there without changing immutable capture',async({page})=>{
  const capture=await page.request.get('/api/v1/generation/captures/'+meta.capture_id).then(r=>r.json());
  await page.setViewportSize({width:1440,height:900});
  await page.goto('/attention-tool?generation='+capture.capture_id);
  await expect(page.getByTestId('attention-patch').first()).toBeVisible({timeout: 20_000});
  await showElements(page);
  await page.getByTestId('outline-element').first().click();
  await openTool(page,'Notes & name');
  await expect(detailsTab(page,'Notes')).toHaveAttribute('aria-selected','true');
  await page.getByLabel('Investigation name').fill('Compact layout browser verification');
  await detailsTab(page,'Result').click();
  await saveInvestigation(page);
  await expect(page).toHaveURL(/investigation_/);
  // Details open on Result is saved in the existing fields: inspector open, evidence dock open.
  const saved=await page.request.get('/api/v1/investigations/'+page.url().split('/').pop()).then(r=>r.json());
  expect(saved.display_settings).toMatchObject({inspector_collapsed:false,evidence_dock_open:true});
  await page.reload();
  await expect(page.getByRole('complementary',{name:'Attention panel',exact:true})).toBeVisible();
  await expect(detailsTab(page,'Result')).toHaveAttribute('aria-selected','true');
  await openDetailsTab(page,'Element');
  await expect(page.getByTestId('selected-element')).toBeVisible();
  const reopened=await page.request.get('/api/v1/generation/captures/'+capture.capture_id).then(r=>r.json());
  expect(reopened.content_sha256).toBe(capture.content_sha256);
  await page.setViewportSize({width:390,height:844});
  // Phones: the bottom bar opens Details as a sheet.
  await expect(bottomBar(page)).toBeVisible();
  await openDetailsTab(page,'Result');
  await expect(page.getByLabel('Evidence view')).toBeVisible();
  expect(await page.evaluate(()=>document.documentElement.scrollWidth<=innerWidth+1)).toBe(true);
  await page.screenshot({path:'../var/screenshots/compact-layout-mobile.png'});
});
