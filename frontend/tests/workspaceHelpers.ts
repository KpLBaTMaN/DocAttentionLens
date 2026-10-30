import { expect, type Locator, type Page } from '@playwright/test';

/** The investigation menu (⋯ in the top bar) replaced the Tools menu; old tool names map to its items. */
const MENU_ITEMS: Record<string, string> = {
  'Provenance': 'Provenance and receipt', 'Display': 'Display settings', 'Notes & name': 'Rename and notes',
};

/** Opens an item of the investigation menu (⋯ in the top bar): Provenance, Display, Document studies, ... */
export async function openTool(page:Page, name:string) {
  if(name==='Open / acquire evidence'){
    await openCaptureMenu(page);
    await page.getByTestId('capture-menu').getByRole('button',{name:'Capture more evidence',exact:true}).click();
    return;
  }
  await page.getByRole('button',{name:'Investigation menu',exact:true}).click();
  await page.getByRole('menu',{name:'Investigation menu'}).getByRole('menuitem',{name:MENU_ITEMS[name]??name,exact:true}).click();
}

/** A plain click on the page image at a fraction of its width and height; the canvas also contains letterboxing. */
export async function clickPage(page:Page, fx:number, fy:number) {
  await expect(page.locator('.docattentionlens-canvas-loading')).toHaveCount(0,{timeout:20_000});
  const b=(await page.getByTestId('page-canvas').first().locator('image').boundingBox())!;
  await page.mouse.click(b.x+b.width*fx,b.y+b.height*fy);
}

export async function showRetrievalScores(page:Page) {
  await openTool(page,'Provenance');
  // Generation investigations inspect the supporting retrieval; retrieval investigations show their own evidence.
  await page.getByRole('button',{name:/^(Inspect supporting retrieval scores|Show retrieval evidence)$/}).click();
}

/** Table grounding cases name the Element tab Table. */
export type DetailsTabName='Method'|'Element'|'Table'|'Result'|'Display'|'Run'|'Notes';

/** A Details tab (Method, Element, Result, Display, Run, Notes). */
export function detailsTab(page:Page, name:DetailsTabName) {
  return page.getByRole('tablist',{name:'Details sections'}).getByRole('tab',{name,exact:true});
}

/**
 * Whether the workspace has three columns (1200 px and wider, not a table grounding case): the element ledger left
 * of the page, the element inspector right of it, the method panel under the method bar and Details as a drawer
 * with Result, Display, Run and Notes.
 */
export async function columnsLayout(page:Page) {
  const workspace=page.locator('.docattentionlens-analysis');
  await expect(workspace).toBeVisible({timeout:20_000});
  return await workspace.getAttribute('data-layout')==='columns';
}

/** The three-column layout's element inspector, shown by closing the Details drawer over it (and the method panel, as the Element tab replaced Method). */
export async function showInspector(page:Page) {
  const inspector=page.locator('#workspace-inspector');
  await closeMethodPanel(page);
  await expect(async()=>{
    if(!await inspector.isVisible()){
      const close=page.locator('#workspace-details').getByRole('button',{name:'Close details',exact:true});
      if(await close.isVisible())await close.click({timeout:2_000});
    }
    await expect(inspector).toBeVisible({timeout:2_000});
  }).toPass({timeout:20_000});
  return inspector;
}

/** The method step: opens Details › Method, or the method panel in the three-column layout (the reading step on retrieval). */
function methodStep(page:Page) {
  return page.locator('.docattentionlens-method-step[data-step="reading"], .docattentionlens-method-step[data-step="method"], .docattentionlens-method-step[data-step="signal"]').first();
}

/** Opens the three-column layout's method panel (what the method reads, its settings and the method list) and returns it. */
export async function openMethodPanel(page:Page) {
  const panel=page.locator('#method-panel');
  if(!await panel.isVisible())await methodStep(page).click();
  await expect(panel).toBeVisible();
  return panel;
}

export async function closeMethodPanel(page:Page) {
  const panel=page.locator('#method-panel');
  if(await panel.isVisible())await methodStep(page).click();
  await expect(panel).toBeHidden();
}

/**
 * Opens Details (a sheet on phones) on a tab and returns the panel. In the three-column layout Method is the
 * method panel and Element the element inspector.
 */
export async function openDetailsTab(page:Page, name:DetailsTabName) {
  if(await columnsLayout(page)){
    if(name==='Method')return openMethodPanel(page);
    if(name==='Element')return showInspector(page);
  }
  const panel=page.locator('#workspace-details');
  // The canvas toggle on desktop, the bottom bar on phones. Retried as a whole: right after a viewport change
  // the panel can still be leaving its desktop column for the phone sheet.
  const toggle=detailsButton(page), bar=bottomBar(page).getByRole('button',{name:'Details',exact:true});
  await expect(toggle.or(bar).first()).toBeVisible({timeout:20_000});
  await expect(async()=>{
    if(!await panel.isVisible()){
      if(await toggle.isVisible())await toggle.click({timeout:2_000});
      else await bar.click({timeout:2_000});
    }
    await expect(panel).toBeVisible({timeout:2_000});
    const tab=detailsTab(page,name);
    if(await tab.getAttribute('aria-selected',{timeout:2_000})!=='true')await tab.click({timeout:2_000});
    await expect(tab).toHaveAttribute('aria-selected','true',{timeout:2_000});
  }).toPass({timeout:20_000});
  return panel;
}

/** The page's elements: Details › Element, or the element ledger and inspector of the three-column layout. */
export async function showElements(page:Page) {
  await openDetailsTab(page,'Element');
}

/** The page list (Details › Run): supplied context, candidate comparison or attention to context. */
export async function showPageList(page:Page) {
  return openDetailsTab(page,'Run');
}

/** The element list in Details › Element, or the element ledger left of the page. */
export function elementList(page:Page) {
  return page.locator('#details-elements');
}

/** The filmstrip (shown only when there is more than one page to choose). */
export function filmstrip(page:Page) {
  return page.getByTestId('filmstrip');
}

/** Turns to a page from the filmstrip, or from Details › Run's page list when there is no filmstrip. */
export async function choosePage(page:Page, pageId:string) {
  const strip=filmstrip(page).locator(`[data-page-id="${pageId}"]`);
  if(await strip.isVisible())await strip.click();
  else {
    await showPageList(page);
    await page.locator(`#workspace-details [data-page-id="${pageId}"]`).click();
  }
  await expect(page.locator('.docattentionlens-image-views')).toHaveAttribute('data-page-id',pageId);
}

/** The page shown on the canvas. */
export async function currentPage(page:Page) {
  return (await page.locator('.docattentionlens-image-views').getAttribute('data-page-id'))!;
}

/** Saves the investigation with the top bar's Save. */
export async function saveInvestigation(page:Page) {
  await page.getByRole('button',{name:'Save investigation',exact:true}).click();
}

/** Fit page or Fit width from the zoom pill (its middle button toggles between them). */
export async function setFit(page:Page, fit:'page'|'width') {
  const toggle=page.locator('.docattentionlens-zoom-pill .docattentionlens-fit-toggle');
  for(let i=0;i<2&&await toggle.getAttribute('data-fit')!==fit;i++)await toggle.click();
  await expect(toggle).toHaveAttribute('data-fit',fit);
}

/** Canvas sizing as the zoom pill reports it: page, width or manual. */
export function canvasSizing(page:Page) {
  return page.locator('.docattentionlens-zoom-pill .docattentionlens-fit-toggle');
}

/** A Details › Result view tab: Method result, Answer and prediction tokens, Complete causal row, Query winner ledger, ... */
export function evidenceTab(page:Page, name:string) {
  return page.getByRole('tablist',{name:'Evidence view'}).getByRole('tab',{name,exact:true});
}

/** Opens a view of the method's own output in Details › Result. */
export async function showEvidence(page:Page, name:string) {
  await openDetailsTab(page,'Result');
  await evidenceTab(page,name).click();
  await expect(evidenceTab(page,name)).toHaveAttribute('aria-selected','true');
}

export async function showMarkers(page:Page) {
  const summary=page.getByText('Inspection markers',{exact:true});
  if(!await summary.evaluate(e=>(e.parentElement as HTMLDetailsElement).open))await summary.click();
}

/** Runs `read` with Details open on a tab, then restores the panel's open state and tab. */
async function inDetailsTab<T>(page:Page, name:DetailsTabName, read:()=>Promise<T>):Promise<T> {
  const panel=page.locator('#workspace-details');
  const wasOpen=await panel.isVisible();
  const previous=wasOpen?await page.getByRole('tablist',{name:'Details sections'}).locator('[aria-selected="true"]').textContent():null;
  await openDetailsTab(page,name);
  const result=await read();
  if(previous&&previous!==name)await detailsTab(page,previous as DetailsTabName).click();
  if(!wasOpen)await page.getByRole('button',{name:'Close details',exact:true}).click();
  return result;
}

/** Details › Display shows its settings or the scale readout, switched at its top. */
export async function chooseDisplayView(page:Page, view:'Settings'|'Scale') {
  const tab=page.getByRole('tablist',{name:'Display view',exact:true}).getByRole('tab',{name:view,exact:true});
  if(await tab.getAttribute('aria-selected')!=='true')await tab.click();
  await expect(tab).toHaveAttribute('aria-selected','true');
}

/** Details › Display rounds values by default; tests compare the exact ones. */
async function exactValues(page:Page) {
  const exact=page.getByRole('radiogroup',{name:'Values',exact:true}).getByRole('radio',{name:'Exact',exact:true});
  if(await exact.isVisible()&&await exact.getAttribute('aria-checked')!=='true')await exact.click();
}

/** Exact totals of the shown row are in Details › Display (with allocation error); the legend carries the image share. */
export async function expectRowTotals(page:Page, text:string|RegExp='', testId='attention-mass') {
  await inDetailsTab(page,'Display',async()=>{
    await chooseDisplayView(page,'Scale');
    await expect(page.getByTestId(testId)).toBeVisible({timeout:20_000});
    await exactValues(page);
    await expect(page.getByTestId(testId)).toContainText(text,{timeout:20_000});
  });
}

export async function readRowTotals(page:Page, testId='attention-mass') {
  return inDetailsTab(page,'Display',async()=>{
    await chooseDisplayView(page,'Scale');
    await expect(page.getByTestId(testId)).toBeVisible({timeout:20_000});
    await exactValues(page);
    return (await page.getByTestId(testId).textContent())!;
  });
}

/** Opens the model and capture chip in the top bar, where saved captures are chosen. */
export async function openCaptureMenu(page:Page) {
  const menu=page.getByTestId('capture-menu');
  if(!await menu.evaluate(e=>(e as HTMLDetailsElement).open))await menu.locator('summary').click();
}

/**
 * Opens the finer display choices and returns where they are: Details › Display, opened from the legend's
 * Display button (the workspace), otherwise the ⚙ menu in a comparison toolbar.
 */
export async function openDisplayOptions(scope:Page|Locator):Promise<Locator> {
  const page='goto' in scope?scope:scope.page();
  if('goto' in scope||await scope.locator('.docattentionlens-canvas-legend').count()){
    const panel=page.getByTestId('display-settings');
    if(!await panel.isVisible()){
      // The chosen tab renders even while Details is closed, so both are restored afterwards.
      const details=page.locator('#workspace-details');
      const tab=await details.locator('.docattentionlens-details-tabs [role=tab][aria-selected=true]').textContent() as DetailsTabName;
      detailsBeforeDisplay.set(page,{open:await details.isVisible(),tab});
      await displaySettingsButton(page).click();
    }
    await expect(panel).toBeVisible();
    await chooseDisplayView(page,'Settings');
    return panel;
  }
  const more=scope.getByLabel('More display options',{exact:true}).first();
  if(!await more.evaluate(e=>(e.parentElement as HTMLDetailsElement).open))await more.click();
  return scope;
}

/** The legend's Display button: its name starts with "Display settings" and states the colour mode and scale. */
export function displaySettingsButton(page:Page) {
  return page.locator('.docattentionlens-canvas-legend').getByRole('button',{name:/^Display settings/});
}

/** Details as it was before `openDisplayOptions` switched it to Display: open or closed, and its tab. */
const detailsBeforeDisplay=new WeakMap<Page,{open:boolean;tab:DetailsTabName}>();

/** Leaves Details › Display, returning Details to the tab it showed before, or closing it if it was closed. */
export async function closeDisplaySettings(page:Page) {
  const drawer=page.getByTestId('display-settings');
  const before=detailsBeforeDisplay.get(page);
  detailsBeforeDisplay.delete(page);
  if(await drawer.isVisible()){
    if(before&&before.tab!=='Display')await detailsTab(page,before.tab).click();
    if(!before?.open)await page.locator('#workspace-details').getByRole('button',{name:'Close details',exact:true}).click();
  }
  await expect(drawer).toBeHidden();
}

/** Opens Details › Display, makes the change there and returns Details to how it was. */
export async function withDisplaySettings<T>(page:Page, change:(drawer:Locator)=>Promise<T>):Promise<T> {
  const result=await change(await openDisplayOptions(page));
  await closeDisplaySettings(page);
  return result;
}

/** A Colour mode choice in Display settings: Original image, Heatmap, Overlay or Blend. */
export function colourMode(drawer:Locator, name:string) {
  return drawer.getByRole('radiogroup',{name:'Colour mode',exact:true}).getByRole('radio',{name,exact:true});
}

export async function chooseColourMode(page:Page, name:string) {
  await withDisplaySettings(page,drawer=>colourMode(drawer,name).click());
}

/** A Page colouring choice in Display settings (document-aware element rows): Element totals, Attention patches or Selected patches. */
export function pageColouring(drawer:Locator, name:string) {
  return drawer.getByRole('radiogroup',{name:'Page attention values',exact:true}).getByRole('radio',{name,exact:true});
}

export async function choosePageColouring(page:Page, name:string) {
  await withDisplaySettings(page,drawer=>pageColouring(drawer,name).click());
}

/** Opens the source line's ⋯ (full prompt and the explanatory notes); returns the popover. */
export async function sourceOptions(page:Page) {
  const button=page.getByRole('button',{name:'Source line options',exact:true});
  if(await button.getAttribute('aria-expanded')!=='true')await button.click();
  const panel=page.getByRole('dialog',{name:'Source line options'});
  await expect(panel).toBeVisible();
  return panel;
}

export async function closeSourceOptions(page:Page) {
  const button=page.getByRole('button',{name:'Source line options',exact:true});
  if(await button.getAttribute('aria-expanded')==='true')await button.click();
  await expect(page.getByRole('dialog',{name:'Source line options'})).toHaveCount(0);
}

/**
 * The source line above the page: Parts is its default row (question → answer, page chip), "Source tokens" in its
 * view switch; Prompt is "Full prompt", the whole sequence with template tokens. MaxSim similarity has no full prompt
 * (one Query tokens row). The full prompt is opened if it was hidden to its header.
 */
export async function chooseTokenView(page:Page, view:'Prompt'|'Parts') {
  const button=page.getByRole('group',{name:'Source line view'}).getByRole('button',{name:view==='Prompt'?'Full prompt':'Source tokens',exact:true});
  if(await button.getAttribute('aria-pressed')!=='true')await button.click();
  await expect(button).toHaveAttribute('aria-pressed','true');
  const show=promptTokens(page).getByRole('button',{name:'Show prompt',exact:true});
  if(view==='Prompt'&&await show.count())await show.click();
}

/** Opens every folded template run in the full prompt, so its tokens can be chosen. */
export async function openTemplateRuns(page:Page) {
  const folded=promptTokens(page).getByRole('button',{name:/^Show \d+ template tokens$/});
  while(await folded.count())await folded.first().click();
}

/** The full prompt's inline token sequence (a group named "Prompt", like the panel's prompt-variant choice). */
export function promptTokens(page:Page) {
  return page.locator('.docattentionlens-prompt-tokens');
}

/** Image focus hides the filmstrip, Details and the source line; the method sentence and legend stay on the page. */
export async function withPanelHidden(page:Page, check:()=>Promise<void>) {
  const focus=page.getByRole('button',{name:'Image focus',exact:true});
  await focus.click();
  await expect(focus).toHaveAttribute('aria-pressed','true');
  try {await check();} finally {await focus.click();}
}

/** The Details toggle: on the canvas's top-right corner, or at the end of the method bar in the three-column layout. */
export function detailsButton(page:Page) {
  return page.locator('.docattentionlens-canvas-instruments, .docattentionlens-method-bar').getByRole('button',{name:'Details',exact:true});
}

/** The phone bottom bar: Page, Details, Method and (with more than one page) Pages. */
export function bottomBar(page:Page) {
  return page.getByRole('navigation',{name:'Workspace views'});
}

/** Opens or closes the Details panel (closed by default for new investigations). */
export async function setDetails(page:Page, open:boolean) {
  const toggle=detailsButton(page);
  await expect(toggle).toBeVisible({timeout:20_000});
  if((await toggle.getAttribute('aria-expanded'))!==String(open))await toggle.click();
  await expect(toggle).toHaveAttribute('aria-expanded',String(open));
}

/**
 * Opens Details on Method (signal, reading, source, grouping, readiness, the method and its settings) and returns
 * it: the "Attention panel" (generation) or "Retrieval panel" (retrieval role). The output is in Result
 * (`methodResult`).
 */
export async function detailsPanel(page:Page, name:'Attention panel'|'Retrieval panel'='Attention panel') {
  // The three-column layout keeps the Method content in the method panel under the method bar.
  if(await columnsLayout(page))return openMethodPanel(page);
  await setDetails(page,true);
  const panel=page.getByRole('complementary',{name,exact:true});
  await expect(panel).toBeVisible();
  const tab=detailsTab(page,'Method');
  if(await tab.getAttribute('aria-selected')!=='true')await tab.click();
  return panel;
}

/** Opens Details › Result and returns the method's output (`method-result`). */
export async function methodResult(page:Page) {
  await openDetailsTab(page,'Result');
  return page.getByTestId('method-result');
}

/** The page image area (both inputs in a split view) in viewport coordinates. */
export async function pageArea(page:Page) {
  return page.locator('.docattentionlens-image-views').evaluate(e=>{const r=e.getBoundingClientRect();return {top:r.top,height:r.height,width:r.width};});
}

/**
 * The shown method: the method bar under the source line (wide layouts) or, on phones, the method named in the
 * method sentence. Its `data-method-key` is `method_id/variant_id` (element aggregation while grouped) and
 * `data-input-key` the grouped input.
 */
export function methodChoice(page:Page) {
  return page.getByTestId('method-choice');
}

/** The method bar's Browse methods button: opens Details › Method with the search focused. */
export function methodBrowse(page:Page) {
  return methodChoice(page).getByRole('button',{name:/Browse methods/});
}

/** Details as it was before `openMethodPicker` switched it to Method: open or closed, and its tab. */
const detailsBeforeMethod=new WeakMap<Page,{open:boolean;tab:DetailsTabName}>();
/** Pages whose method panel `openMethodPicker` opened (three-column layout), so `closeMethodPicker` closes it again. */
const methodPanelOpened=new WeakSet<Page>();

/**
 * Opens Details › Method (the docked method picker) from the method bar, or the bottom bar on phones, and returns it.
 * In the three-column layout the picker is in the method panel under the method bar.
 */
export async function openMethodPicker(page:Page) {
  const browser=page.getByTestId('method-browser');
  if(!await browser.isVisible()&&await columnsLayout(page)){
    methodPanelOpened.add(page);
    await methodBrowse(page).click();
  }
  if(!await browser.isVisible()){
    const details=page.locator('#workspace-details');
    const tab=await details.locator('.docattentionlens-details-tabs [role=tab][aria-selected=true]').textContent() as DetailsTabName;
    detailsBeforeMethod.set(page,{open:await details.isVisible(),tab});
    const browse=methodBrowse(page);
    if(await browse.isVisible())await browse.click();
    else await bottomBar(page).getByRole('button',{name:'Method',exact:true}).click();
  }
  await expect(browser).toBeVisible();
  await browser.getByRole('combobox',{name:'Find a method'}).fill('');
  // The catalogue loads with the capture; perspective and method choices wait for it.
  await expect(browser.locator('.docattentionlens-mp-family, [role="option"]').first()).toBeVisible({timeout:20_000});
  return browser;
}

/** Leaves Details › Method, returning Details to the tab it showed before, or closing it if it was closed. */
export async function closeMethodPicker(page:Page) {
  if(methodPanelOpened.delete(page)){await closeMethodPanel(page);return;}
  const browser=page.getByTestId('method-browser');
  const before=detailsBeforeMethod.get(page);
  detailsBeforeMethod.delete(page);
  // Nothing to restore when Method was already showing before the picker was used.
  if(!before||before.tab==='Method')return;
  if(await browser.isVisible()){
    // The chosen tab renders even while Details is closed, so the tab is restored before closing.
    if(before)await detailsTab(page,before.tab).click();
    if(!before?.open)await page.locator('#workspace-details').getByRole('button',{name:'Close details',exact:true}).click();
  }
  await expect(browser).toBeHidden();
}

/**
 * Chooses a method in Details › Method (opened from the method bar), then returns Details to how it was.
 * `input` chooses a General method as the document-aware view's grouped input ("General attention · grouped into elements").
 */
export async function chooseMethod(page:Page, key:string, {input=false}:{input?:boolean}={}) {
  const picker=await openMethodPicker(page);
  await picker.getByRole('combobox',{name:'Find a method'}).fill(key.split('/')[0]);
  const option=picker.locator(`[role="option"][data-method-key="${key}"]${input?'[data-grouping-input]':':not([data-grouping-input])'}`);
  if(!await option.count()){
    const other=picker.getByRole('radiogroup',{name:'Analysis perspective'}).locator('[aria-checked="false"]');
    await other.click();
  }
  await option.first().click();
  // The docked picker stays open: the choice is made once its row is marked current.
  await expect(picker.locator(`[role="option"][aria-selected="true"][data-method-key="${key}"]${input?'[data-grouping-input]':''}`).first()).toBeVisible();
  await closeMethodPicker(page);
}

/** Opens the case details popover from the breadcrumb (document / question ▾) and returns it. */
export async function caseDetails(page:Page) {
  const panel=page.getByRole('dialog',{name:'Case details'});
  if(!await panel.count())await page.locator('.docattentionlens-crumb-button').click();
  await expect(panel).toBeVisible();
  return panel;
}

/**
 * General or Document-aware: the method picker's perspective filter re-analyses the same saved row in that
 * perspective (as the panel's perspective tabs did), then Details returns to how it was.
 */
export async function choosePerspective(page:Page, name:'General'|'Document-aware'|'Table-aware') {
  const picker=await openMethodPicker(page);
  const radio=picker.getByRole('radiogroup',{name:'Analysis perspective'}).getByRole('radio',{name,exact:true});
  if(await radio.getAttribute('aria-checked')!=='true')await radio.click();
  await expect(methodChoice(page)).toHaveAttribute('data-perspective',({'General':'general','Document-aware':'document_aware','Table-aware':'table_aware'} as const)[name]);
  await closeMethodPicker(page);
}

/** Method keys listed by the picker for a search, in the current perspective view. */
export async function pickerOptions(page:Page, search:string) {
  const picker=await openMethodPicker(page);
  await picker.getByRole('combobox',{name:'Find a method'}).fill(search);
  return picker.locator('[role="option"]');
}

/** Keys of the available methods the picker lists in the current perspective (families opened; grouped inputs aside). */
export async function enabledMethodKeys(page:Page) {
  const picker=await openMethodPicker(page);
  const closed=picker.locator('.docattentionlens-mp-family[aria-expanded="false"]');
  while(await closed.count())await closed.first().click();
  const keys=await picker.locator('.docattentionlens-mp-section [role="option"]:not([aria-disabled="true"]):not([data-grouping-input])')
    .evaluateAll(os=>os.map(o=>o.getAttribute('data-method-key')!));
  await closeMethodPicker(page);
  return [...new Set(keys)];
}
