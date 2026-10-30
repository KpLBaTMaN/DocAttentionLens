import { expect, test, type Page } from '@playwright/test';
import * as F from '../../fixtures/attention';
import { catalogue, mockAttentionApi } from './mockApi';
import { chooseColourMode, choosePageColouring, chooseTokenView, colourMode, pageColouring, withDisplaySettings } from '../workspaceHelpers';
import { chooseMethod, choosePerspective, enabledMethodKeys, methodChoice, pickerOptions, setDetails, openDetailsTab, showEvidence } from '../workspaceHelpers';

const key = (id: string, retrieval: boolean) => `${id}/${retrieval ? 'retriever_encoding_v1' : catalogue.find(m => m.method_id === id)!.variant_id}`;
/** Details › Element's element list, where element values sit beside each element. */
const rail = (page: Page) => page.locator('#details-elements');
const railElement = (page: Page, id: string) => rail(page).locator(`[data-testid="outline-element"][data-native-id="${F.ELEMENTS.find(e => e.element_id === id)!.native_id}"]`);
/** The full element table and patch contributions follow the list in Details › Element. */
async function elementSummary(page: Page) {
  await openDetailsTab(page, 'Element');
  await rail(page).getByRole('button', {name: 'Patch contributions and full table ↓', exact: true}).click();
  return page.getByTestId('element-summary');
}
/**
 * The element summary when the list has no element values to open it from: Details › Element for a generator,
 * and also Details › Result's Element summary view for a retriever encoding.
 */
async function evidenceSummary(page: Page, retrieval: boolean) {
  if (retrieval) await showEvidence(page, 'Element summary');
  else await openDetailsTab(page, 'Element');
  return page.getByTestId('element-summary');
}
const masses = (page: Page) => page.getByTestId('element-region').evaluateAll(rs => Object.fromEntries(rs.map(r => [r.getAttribute('data-element-id')!, Number(r.getAttribute('data-mass'))])));

for (const retrieval of [false, true]) test(`${retrieval ? 'retriever' : 'generator'}: preserve the analysis and show its background patches`, async ({page}) => {
  const api = await mockAttentionApi(page);
  const parameters: {heads?: number[]}[] = [];
  page.on('request', r => {if(r.method()==='POST'&&r.url().endsWith(retrieval?'/attention':'/analyses'))parameters.push(r.postDataJSON().parameters);});
  await page.setViewportSize({width: 1440, height: 900});
  await page.goto(`/attention-tool?${retrieval ? 'capture=' + F.RETRIEVAL_ID : 'generation=' + F.GENERATION_ID}`);
  await openDetailsTab(page, 'Method');
  await chooseMethod(page, key('attention.mean_heads', retrieval));
  await expect(page.getByTestId('attention-patch')).toHaveCount(F.PATCH_COUNT);
  await page.getByLabel('Attention layer', {exact: true}).selectOption('1');
  await page.getByLabel('Include head 0', {exact: true}).uncheck();
  const headCount = retrieval ? F.retrievalCapture.attention_encodings![0].query_head_count : F.generationCapture.query_head_count;
  const heads = Array.from({length: headCount}, (_, i) => i).slice(1);
  await expect.poll(() => parameters.at(-1)?.heads).toEqual(heads);
  const kind = retrieval ? 'retriever' : 'analysis';
  await expect.poll(() => api.latest('attention.mean_heads', kind)?.result.row?.layer).toBe(1);
  const before = api.served.length;
  const row = api.latest('attention.mean_heads', kind)!.result.row!;
  await choosePerspective(page, 'Document-aware');
  await expect(methodChoice(page)).toHaveAttribute('data-method-key', key('attention.element_allocation', retrieval));
  await expect(methodChoice(page)).toHaveAttribute('data-input-key', key('attention.mean_heads', retrieval));
  expect((await enabledMethodKeys(page)).some(k => k.startsWith('attention.mean_heads/'))).toBe(false);
  await expect(page.getByLabel('Attention layer', {exact: true})).toHaveValue('1');
  await expect.poll(() => masses(page)).toEqual(F.elementMasses(row));
  // Details › Element lists each element with its value (a switch made in Details › Method stays there).
  await openDetailsTab(page, 'Element');
  await expect(rail(page).getByTestId('element-value')).toHaveCount(2);
  await openDetailsTab(page, 'Method');
  await expect(page.getByLabel('Include head 0', {exact: true})).not.toBeChecked();
  await expect(page.getByRole('complementary', {name: 'Linked element summary'})).toHaveCount(0);
  await expect(await elementSummary(page)).toContainText('Mean-Head Attention');
  await expect(page.getByTestId('attention-patch')).toHaveCount(0);
  expect(api.served.length).toBe(before);

  // Page colouring sits in Display settings and keeps its arrow-key choice.
  await withDisplaySettings(page, async drawer => {
    await pageColouring(drawer, 'Element totals').focus();
    await pageColouring(drawer, 'Element totals').press('ArrowRight');
    await expect(pageColouring(drawer, 'Attention patches')).toBeFocused();
    await expect(page.getByTestId('attention-patch')).toHaveCount(F.PATCH_COUNT);
    await pageColouring(drawer, 'Attention patches').press('ArrowLeft');
    await expect(pageColouring(drawer, 'Element totals')).toHaveAttribute('aria-checked','true');
  });
  expect(api.served.length).toBe(before);

  // This does not read raw attention or condition on annotated elements.
  await openDetailsTab(page, 'Element');
  await chooseColourMode(page, 'Original image');
  await page.getByRole('button', {name: 'Show attention patches including background'}).click();
  await withDisplaySettings(page, async drawer => {
    await expect(colourMode(drawer, 'Overlay')).toHaveAttribute('aria-checked','true');
    await expect(pageColouring(drawer, 'Attention patches')).toHaveAttribute('aria-checked','true');
  });
  await expect(page.getByTestId('attention-patch')).toHaveCount(F.PATCH_COUNT);
  await expect(page.getByTestId('element-region')).toHaveCount(0);
  const background = retrieval ? 5 : F.generatorPatchIndex(5);
  const backgroundPatch = page.getByTestId('attention-patch').filter({has: page.locator('title', {hasText: `position ${background}:`})});
  await expect(backgroundPatch).toHaveCount(1);
  const allPatches = await page.getByTestId('attention-patch').evaluateAll(rs => rs.map(r => Number(r.getAttribute('data-position-index'))));
  expect(allPatches).toContain(background);
  await expect(page.getByTestId('element-summary')).toContainText('Page background');
  await expect(rail(page).getByTestId('outside-elements')).toContainText('background');
  expect(api.served.length).toBe(before);

  await withDisplaySettings(page, drawer => expect(pageColouring(drawer, 'Selected patches')).toHaveCount(0));
  await railElement(page, F.TABLE_ID).click();
  await choosePageColouring(page, 'Selected patches');
  // A retriever source patch may also be retained as an outline.
  const drawn = await page.getByTestId('attention-patch').evaluateAll(rs => rs.map(r => Number(r.getAttribute('data-position-index'))));
  expect(drawn).toContain(retrieval ? 0 : F.generatorPatchIndex(0));
  expect(drawn).not.toContain(background);

  await choosePageColouring(page, 'Element totals');
  await chooseMethod(page, key('attention.max_heads', retrieval), {input: true});
  await expect(page.getByTestId('element-key')).toContainText('Head envelope allocation');
  await expect(await elementSummary(page)).toContainText('need not sum to one');
  await choosePerspective(page, 'General');
  await expect(methodChoice(page)).toHaveAttribute('data-method-key', key('attention.max_heads', retrieval));
  await expect(page.getByTestId('attention-patch')).toHaveCount(F.PATCH_COUNT);
  await expect(page.getByTestId('element-region')).toHaveCount(0);
  expect(api.unexpected).toEqual([]);
});

for (const retrieval of [false, true]) test(`${retrieval ? 'retriever' : 'generator'}: nested input survives native analyses and explicitly remembers raw`, async ({page}) => {
  const api = await mockAttentionApi(page);
  await page.setViewportSize({width:1440,height:900});
  await page.goto(`/attention-tool?${retrieval ? 'capture=' + F.RETRIEVAL_ID : 'generation=' + F.GENERATION_ID}`);
  await openDetailsTab(page, 'Method');
  await page.getByLabel('Attention layer', {exact:true}).selectOption('1');
  await page.getByLabel('Attention head', {exact:true}).selectOption('1');
  await chooseMethod(page, key('attention.mean_heads', retrieval));
  await page.getByLabel('Include head 0', {exact:true}).uncheck();
  await choosePerspective(page, 'Document-aware');
  await expect(methodChoice(page)).toHaveAttribute('data-input-key', key('attention.mean_heads', retrieval));
  await chooseMethod(page, key('attention.element_type_profile', retrieval));
  await expect(methodChoice(page)).not.toHaveAttribute('data-input-key', /./);
  await expect(page.getByTestId('document-grouping')).toHaveCount(0);
  await expect.poll(() => !!api.latest('attention.element_type_profile', retrieval ? 'retriever' : 'analysis')).toBe(true);
  await chooseMethod(page, key('attention.element_allocation', retrieval));
  await expect(methodChoice(page)).toHaveAttribute('data-input-key', key('attention.mean_heads', retrieval));
  await expect(page.getByLabel('Attention layer', {exact:true})).toHaveValue('1');
  if (!retrieval) await expect(page.getByLabel('Include head 0', {exact:true})).not.toBeChecked();

  await chooseMethod(page, key('attention.row', retrieval), {input: true});
  await expect(page.getByTestId('document-grouping')).toHaveText('Raw attention → Element totals');
  await expect(page.getByLabel('Attention head', {exact:true})).toHaveValue('1');
  await chooseMethod(page, key('attention.element_type_profile', retrieval));
  await chooseMethod(page, key('attention.element_allocation', retrieval));
  await expect(methodChoice(page)).toHaveAttribute('data-input-key', key('attention.row', retrieval));
  await chooseMethod(page, key('attention.element_type_profile', retrieval));
  await choosePerspective(page, 'General');
  await expect(methodChoice(page)).toHaveAttribute('data-method-key', key('attention.row', retrieval));
  await expect(methodChoice(page)).not.toHaveAttribute('data-input-key', /./);
  await expect(methodChoice(page)).toHaveAttribute('data-perspective', 'general');
  expect(api.unexpected).toEqual([]);
});

for (const retrieval of [false, true]) test(`${retrieval ? 'retriever' : 'generator'}: group comparison rows separately`, async ({page}) => {
  const api = await mockAttentionApi(page);
  await page.setViewportSize({width: 1440, height: 900});
  await page.goto(`/attention-tool?${retrieval ? 'capture=' + F.RETRIEVAL_ID : 'generation=' + F.GENERATION_ID}`);
  await setDetails(page, true);
  await chooseMethod(page, key('attention.across_layers', retrieval));
  await expect.poll(() => api.latest('attention.across_layers', retrieval ? 'retriever' : 'analysis')?.result.comparisons?.length ?? 0).toBeGreaterThan(1);
  await choosePerspective(page, 'Document-aware');
  await expect(methodChoice(page)).toHaveAttribute('data-method-key', key('attention.element_allocation', retrieval));
  await expect(methodChoice(page)).toHaveAttribute('data-input-key', key('attention.across_layers', retrieval));
  await openDetailsTab(page, 'Result');
  const before = api.served.length;
  const rows = api.latest('attention.across_layers', retrieval ? 'retriever' : 'analysis')!.result.comparisons!;
  await expect.poll(() => masses(page)).toEqual(F.elementMasses(rows[0].row));
  await page.getByLabel('Grouped comparison').selectOption('1');
  await expect.poll(() => masses(page)).toEqual(F.elementMasses(rows[1].row));
  await expect(page.getByLabel('Grouped comparison')).toHaveValue('1');
  expect(api.served.length).toBe(before);
  expect(api.unexpected).toEqual([]);
});

test('retriever: grouped query encodings disclose missing document positions', async ({page}) => {
  const api = await mockAttentionApi(page);
  await page.goto(`/attention-tool?capture=${F.RETRIEVAL_ID}`);
  await setDetails(page, true);
  await chooseMethod(page, key('attention.mean_heads', true));
  await choosePerspective(page, 'Document-aware');
  await openDetailsTab(page, 'Method');
  await page.getByRole('button', {name: 'Question', exact: true}).click();
  await expect(page.getByTestId('structure-gap')).toContainText('query encoding has no mapped document positions');
  await openDetailsTab(page, 'Element');
  await expect(rail(page).getByTestId('outline-element').first()).toBeVisible();
  await expect(rail(page).getByTestId('element-value')).toHaveCount(0);
  await expect((await evidenceSummary(page, true)).getByTestId('element-summary-unavailable')).toBeVisible();
  await expect(page.getByTestId('attention-patch')).toHaveCount(0);
  await expect(page.getByTestId('element-region')).toHaveCount(0);
  // Neither the document-aware list nor its grouped General inputs offer entropy, a scalar.
  await expect((await pickerOptions(page, 'entropy')).filter({hasText: 'Attention entropy'})).toHaveCount(0);
  await page.keyboard.press('Escape');
  expect(api.unexpected).toEqual([]);
});

test('mobile: page values and grouped summary remain reachable without horizontal overflow', async ({page}) => {
  const api = await mockAttentionApi(page);
  await page.setViewportSize({width: 390, height: 844});
  await page.goto(`/attention-tool?generation=${F.GENERATION_ID}`);
  await openDetailsTab(page, 'Result');
  await chooseMethod(page, key('attention.mean_heads', false));
  await choosePerspective(page, 'Document-aware');
  await expect(methodChoice(page)).toHaveAttribute('data-input-key', key('attention.mean_heads', false));
  // The element values are on Element; the source and method settings are on Result.
  await openDetailsTab(page, 'Element');
  await openDetailsTab(page, 'Method');
  const source = page.getByLabel('Attention source', {exact:true});
  await source.scrollIntoViewIfNeeded();
  await expect(source).toBeInViewport();
  const layer = page.getByLabel('Attention layer', {exact:true});
  await layer.scrollIntoViewIfNeeded();
  await expect(layer).toBeInViewport();
  await layer.selectOption('1');
  const head = page.getByLabel('Include head 0', {exact:true});
  await head.scrollIntoViewIfNeeded();
  await expect(head).toBeInViewport();
  await head.uncheck();
  await page.getByRole('button', {name: 'Close details', exact: true}).click();
  await choosePageColouring(page, 'Attention patches');
  await expect(page.getByTestId('attention-patch')).toHaveCount(F.PATCH_COUNT);
  await openDetailsTab(page, 'Element');
  await page.getByTestId('element-summary').scrollIntoViewIfNeeded();
  await expect(page.getByTestId('element-summary')).toContainText('Mean-Head Attention');
  expect(await page.evaluate(() => document.documentElement.scrollWidth <= innerWidth + 1)).toBe(true);
  expect(api.unexpected).toEqual([]);
});

async function clickPatch(page: Page, k: number) {
  await expect(page.locator('.docattentionlens-canvas-loading')).toHaveCount(0);
  const image = (await page.getByTestId('page-canvas').locator('image').boundingBox())!;
  const {x, y} = F.patchCentre(k);
  await page.mouse.click(image.x + x / F.PAGE.width * image.width, image.y + y / F.PAGE.height * image.height);
}

test('a generator patch click shows what attends to it; the next token click restores the method', async ({page}) => {
  const api = await mockAttentionApi(page);
  await page.setViewportSize({width: 1440, height: 900});
  await page.goto(`/attention-tool?generation=${F.GENERATION_ID}`);
  await chooseMethod(page, key('attention.mean_heads', false));
  await expect(page.getByTestId('attention-patch')).toHaveCount(F.PATCH_COUNT);
  // There is no direction or picking control: the page itself is the control.
  await expect(page.getByRole('button', {name: /What attends to it|What it attends to|Pick patch/})).toHaveCount(0);
  await chooseTokenView(page, 'Parts');
  await clickPatch(page, 5);
  const target = F.generatorPatchIndex(5);
  await expect(page.getByTestId('column-context')).toContainText(`What attends to position ${target} · image patch`);
  // Incoming columns are raw attention, read from the saved capture.
  await expect(methodChoice(page)).toHaveAttribute('data-method-key', key('attention.row', false));
  // Column mode is named in the source line, with a clear button.
  await expect(page.getByTestId('source-column')).toContainText(`What attends to patch ${target}`);
  const column = F.generatorColumn(target);
  const strongest = column.sources.filter(s => s.kind === 'question').reduce((a, b) => a.value > b.value ? a : b);
  await expect(page.getByRole('group', {name: 'Question tokens'}).locator(`[data-token-index="${strongest.source_index}"]`)).toHaveAttribute('title', /attention weight/);
  await expect(page.locator(`[data-testid=attention-patch][data-position-index="${target}"]`)).toHaveAttribute('stroke', '#39a9e0');
  // A plain click never selects the layout element under the patch.
  await expect(page.locator('rect.docattentionlens-box.is-selected')).toHaveCount(0);

  await page.getByRole('group', {name: 'Answer tokens'}).locator('[data-token-index="1"]').click();
  await expect(methodChoice(page)).toHaveAttribute('data-method-key', key('attention.mean_heads', false));
  await expect(page.getByTestId('column-context')).toHaveCount(0);
  await expect(page.getByTestId('source-column')).toHaveCount(0);
  await expect(page.getByTestId('attention-patch')).toHaveCount(F.PATCH_COUNT);
  expect(api.unexpected).toEqual([]);
});

test('the Document-aware generator page shows element masses; a click chooses an element, never a patch', async ({page}) => {
  const api = await mockAttentionApi(page);
  await page.goto(`/attention-tool?generation=${F.GENERATION_ID}`);
  await chooseMethod(page, key('attention.mean_heads', false));
  await choosePerspective(page, 'Document-aware');
  await expect(page.getByTestId('element-region').first()).toBeVisible();
  await expect(page.getByTestId('page-canvas')).not.toHaveClass(/is-pickable/);
  await clickPatch(page, 5);
  await expect(methodChoice(page)).toHaveAttribute('data-perspective', 'document_aware');
  await expect(page.getByTestId('column-context')).toHaveCount(0);
  // The patch lies in the table: the click selects that element and shows it (in the element inspector beside the
  // page, or Details › Element in the tabbed layout).
  await clickPatch(page, 0);
  await expect(page.locator(`rect.docattentionlens-box.is-selected[data-element-id="${F.TABLE_ID}"]`)).toHaveCount(1);
  await expect(page.getByTestId('selected-element')).toBeVisible();
  expect(api.unexpected).toEqual([]);
});

for (const [layout, width] of [['three columns', 1440], ['tabbed Details', 1100]] as const) test(`a causal retriever source states its page reach and offers the whole page (${layout})`, async ({page}) => {
  const api = await mockAttentionApi(page);
  // ColQwen2-like: each document position sees only itself and earlier positions.
  await page.route(`**/api/v1/retrieval/captures/${F.RETRIEVAL_ID}`, route => route.fulfill({json: {...F.retrievalCapture,
    attention_encodings: F.retrievalCapture.attention_encodings!.map(e => e.encoding_id === 'document' ? {...e, causal: true} : e)}}));
  await page.setViewportSize({width, height: 900});
  await page.goto(`/attention-tool?capture=${F.RETRIEVAL_ID}`);
  await setDetails(page, true);
  await choosePerspective(page, 'Document-aware');
  const source = page.getByLabel('Attention source position');
  const last = String(F.retrieverPositions.length - 1);
  // A causal encoding opens on its last position, which reaches every page patch.
  await openDetailsTab(page, 'Element');
  await expect(rail(page).getByTestId('element-value')).toHaveCount(2);
  await expect(rail(page).getByTestId('rail-reach')).toHaveCount(0);
  await openDetailsTab(page, 'Method');
  await expect(source).toHaveValue(last);
  // Patch 6 reaches patches 0–6: the table (0, 1) fully, the text (6, 7, 10, 11) through one patch.
  await source.selectOption('6');
  if (width < 1200) {
    // Away from the element list (another Details tab), the page states the reach.
    await expect(page.getByTestId('page-reach')).toContainText(`Source 6 sees 7 of ${F.PATCH_COUNT} page patches`);
    await openDetailsTab(page, 'Element');
  }
  await expect(rail(page).getByTestId('rail-reach')).toContainText(`Source 6 sees 7 of ${F.PATCH_COUNT} patches`);
  await expect(railElement(page, F.TEXT_ID)).toContainText('partial 1/4');
  await expect(railElement(page, F.TABLE_ID)).not.toContainText('partial');
  // The element list sits beside the page, so no page notice repeats the reach.
  await expect(page.getByTestId('page-reach')).toHaveCount(0);
  await rail(page).getByRole('button', {name: 'Use the last token', exact: true}).click();
  await expect(rail(page).getByTestId('rail-reach')).toHaveCount(0);
  await openDetailsTab(page, 'Method');
  await expect(source).toHaveValue(last);
  await expect(methodChoice(page)).toHaveAttribute('data-perspective', 'document_aware');
  expect(api.unexpected).toEqual([]);
});

test('missing element mapping is unavailable rather than measured zero', async ({page}) => {
  const api = await mockAttentionApi(page);
  await page.route(`**/api/v1/generation/captures/${F.GENERATION_ID}`, route => route.fulfill({json: {
    ...F.generationCapture, positions: F.generationCapture.positions.map(p => ({...p, ownership: p.role === 'image' ? {background: 1} : p.ownership})),
  }}));
  await page.goto(`/attention-tool?generation=${F.GENERATION_ID}`);
  await chooseMethod(page, key('attention.mean_heads', false));
  await choosePerspective(page, 'Document-aware');
  await expect((await evidenceSummary(page, false)).getByTestId('element-summary-unavailable')).toContainText('No annotated element boxes are mapped');
  await expect(page.getByTestId('element-region')).toHaveCount(0);
  await expect(page.getByTestId('element-summary-row')).toHaveCount(0);
  expect(api.unexpected).toEqual([]);
});
