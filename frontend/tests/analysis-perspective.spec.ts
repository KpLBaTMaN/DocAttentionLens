import { test, expect, type Page } from '@playwright/test';
import { requireSaved } from './savedEvidence';
import { choosePageColouring, pageColouring, withDisplaySettings, detailsPanel, methodChoice, choosePerspective, chooseMethod, pickerOptions, currentPage, choosePage, detailsTab, elementList, openDetailsTab, showElements, showEvidence } from './workspaceHelpers';

// General and Document-aware analyse the same saved attention. Switching never runs a model,
// keeps the capture, source, layer/head and compatible calculation. Page colouring (in Display
// settings) can independently show elements, the full underlying patch result, or selected patches.

type Position = {index: number; page_id?: string | null; bbox?: number[] | null; ownership?: Record<string, number>};
type Allocation = {bucket: string; element_id?: string | null; page_id?: string | null; mass: number};

// The app's numeric label, written independently: six decimals, scientific below 1e-6.
const label = (v: number) => v !== 0 && Math.abs(v) < 1e-6 ? v.toExponential(3) : v.toFixed(6);
const mapped = (p: Position) => Object.keys(p.ownership ?? {}).some(k => k.startsWith('element_'));

async function guardRuns(page: Page) {
  let runs = 0;
  await page.route(/\/api\/v1\/(generation|retrieval)\/runs$/, route => {runs++; return route.abort();});
  return () => runs;
}

async function drawnRegions(page: Page) {
  return page.getByTestId('element-region').evaluateAll(rs => rs.map(r => ({id: r.getAttribute('data-element-id')!,
    mass: Number(r.getAttribute('data-mass')), opacity: Number(r.getAttribute('fill-opacity'))})));
}

test('generation: perspectives re-analyse one saved row, keep selections and link regions with the summary', async ({page}) => {
  test.setTimeout(120_000);
  const items = (await page.request.get('/api/v1/generation/captures?limit=25').then(r => r.json())).items;
  const capture = items.find((c: {positions: Position[]}) => c.positions.some(mapped));
  requireSaved(!capture, 'Requires a saved generator capture whose pages map annotated elements');
  const runs = await guardRuns(page);
  const models = await page.request.get('/api/v1/setup/models').then(r => r.json());
  await page.setViewportSize({width: 1440, height: 900});
  await page.goto('/attention-tool?generation=' + capture.capture_id);
  const panel = await detailsPanel(page);
  // The method sentence on the page names the method; its picker holds the General | Document-aware perspective.
  const picker = methodChoice(page);
  await expect(picker).toHaveAttribute('data-method-key', 'attention.row/captured_weights_v1');
  await expect(picker).toHaveAttribute('data-perspective', 'general');
  // 1. The model and capture are the top bar's model chip; Details repeats no selection.
  await expect(page.getByTestId('capture-menu')).toContainText(models.find((m: {model_id: string}) => m.model_id === capture.provenance.model_id)?.name ?? capture.provenance.model_id.split('/').pop());
  await expect(panel.getByTestId('attention-identity')).toHaveCount(0);
  await expect(panel.getByLabel('Attention source', {exact: true})).toHaveValue('step:0');
  // General lists only general methods; raw attention has no settings section.
  await expect((await pickerOptions(page, 'element_allocation')).and(page.locator('[data-method-key="attention.element_allocation/ownership_mass_v1"]'))).toHaveCount(0);
  await expect((await pickerOptions(page, 'mean_heads')).and(page.locator('[data-method-key="attention.mean_heads/captured_prediction_v1"]'))).toHaveCount(1);
  await page.keyboard.press('Escape');
  await expect(panel.locator('.docattentionlens-method-settings')).toHaveCount(0);
  await expect(panel.getByText('Method options', {exact: true})).toHaveCount(0);
  await expect(picker).toHaveAttribute('data-perspective', 'general');
  await page.getByLabel('Attention layer', {exact: true}).selectOption('2');
  await page.getByLabel('Attention head', {exact: true}).selectOption('3');
  await expect(page.getByTestId('attention-patch').first()).toBeVisible();
  const generalHeight = (await page.getByTestId('page-canvas').boundingBox())!.height;

  await choosePerspective(page, 'Document-aware');
  await expect(picker).toHaveAttribute('data-method-key', 'attention.element_allocation/ownership_mass_v1');
  await expect(page.getByLabel('Attention layer', {exact: true})).toHaveValue('2');
  await expect(page.getByLabel('Attention head', {exact: true})).toHaveValue('3');
  // Switching to Document-aware shows the element list, where each element carries its value (the ledger beside the
  // page, or Details › Element in the tabbed layout); the page keeps its width.
  await expect(elementList(page)).toBeVisible();
  await expect(elementList(page).getByTestId('element-value').first()).toBeVisible();
  await expect(page.getByRole('complementary', {name: 'Linked element summary', exact: true})).toHaveCount(0);
  await openDetailsTab(page, 'Method');
  await expect(panel.getByLabel('Attention source', {exact: true})).toHaveValue('step:0');
  await expect(picker).toHaveAttribute('data-perspective', 'document_aware');
  // The method sentence names the method and perspective; no separate analysis banner repeats them.
  await expect(page.getByTestId('active-analysis')).toHaveCount(0);
  await expect(panel.getByTestId('structure-status')).toHaveText('Ready: element boxes and patch mapping available.');
  await expect(panel.getByTestId('structure-provenance')).toContainText('Structure: Dataset annotations');
  await expect(panel.getByRole('region', {name: 'Element aggregation settings', exact: true})).toBeVisible();
  await expect(page.getByTestId('element-region').first()).toBeVisible();
  // Page colouring lives in Display settings, so the colour bar does not grow and the page keeps its height.
  await expect(page.getByRole('radiogroup', {name: 'Page attention values', exact: true})).toHaveCount(0);
  const groupedHeight = (await page.getByTestId('page-canvas').boundingBox())!.height;
  expect(groupedHeight).toBeGreaterThanOrEqual(generalHeight - 40);
  expect(groupedHeight).toBeGreaterThanOrEqual(400);
  // Document calculations are now executable when this capture has their verified annotations.
  for (const value of ['attention.designated_support/annotated_support_mass_v1', 'attention.element_type_profile/type_mass_area_v1', 'attention.reading_order/annotated_order_profile_v1']) {
    const option = (await pickerOptions(page, value.split('/')[0])).and(page.locator(`[data-method-key="${value}"]`));
    await expect(option).toHaveCount(1);
    await expect(option).not.toContainText('Planned');
  }
  await page.keyboard.press('Escape');

  // Regions draw the saved row's element masses with the patch palette and their own maximum.
  const row = await page.request.get(`/api/v1/generation/captures/${capture.capture_id}/rows?step=0&layer=2&head=3`).then(r => r.json());
  const pageId = await currentPage(page);
  const allocations: Allocation[] = row.allocations;
  const masses = Object.fromEntries(allocations.filter(a => a.element_id && a.page_id === pageId).map(a => [a.element_id!, a.mass]));
  const maximum = Math.max(...allocations.filter(a => a.element_id).map(a => a.mass));
  await expect(page.getByTestId('element-region')).toHaveCount(Object.keys(masses).length);
  for (const region of await drawnRegions(page)) {
    expect(region.mass).toBeCloseTo(masses[region.id], 12);
    expect(region.opacity).toBeCloseTo(Math.min(.65, masses[region.id] / maximum * .65), 9);
  }
  await expect(page.getByTestId('element-key')).toContainText('Element attention mass');
  await expect(page.getByTestId('element-scale')).toContainText(`0 → ${label(maximum)}`);
  await expect(page.getByTestId('attention-patch')).toHaveCount(0);
  await withDisplaySettings(page, drawer => expect(pageColouring(drawer, 'Element totals')).toHaveAttribute('aria-checked','true'));

  // Element list → canvas: the element's box is selected and its contributing patches appear.
  const [top, second] = Object.entries(masses).sort((a, b) => b[1] - a[1]).map(([id]) => id);
  await showElements(page);
  const rail = elementList(page);
  const natives = new Map<string, number>((await page.request.get(`/api/v1/pages/${pageId}/elements`).then(r => r.json())).items.map((e: {element_id: string; native_id: number}) => [e.element_id, e.native_id]));
  const railElement = (id: string) => rail.locator(`[data-testid="outline-element"][data-native-id="${natives.get(id)}"]`);
  await railElement(top).click();
  await choosePageColouring(page, 'Selected patches');
  await expect(page.locator(`rect.docattentionlens-box.is-selected[data-element-id="${top}"]`)).toHaveCount(1);
  const owned = (capture.positions as Position[]).filter(p => p.page_id === pageId && p.index < row.target_count && (p.ownership?.[top] ?? 0) > 0);
  await expect(page.getByTestId('attention-patch')).toHaveCount(owned.length);
  const contribution = owned.reduce((sum, p) => sum + row.values[p.index] * p.ownership![top], 0);
  expect(contribution).toBeCloseTo(masses[top], 9);
  // The full table and its patch contributions follow the list in Details › Element.
  await rail.getByRole('button', {name: 'Patch contributions and full table ↓', exact: true}).click();
  const summary = page.getByTestId('element-summary');
  await expect(summary.getByTestId('element-contributions')).toContainText(`${owned.length} patches · Σ weight × ownership = ${label(contribution)} (element mass ${label(masses[top])})`);
  await expect(page.getByTestId('display-scale')).toContainText('General patch scale');
  // Element list → summary: selecting an element in the list selects its summary row (page clicks choose patches).
  if (second) {
    await railElement(second).click();
    await expect(summary.locator(`[data-testid="element-summary-row"][data-element-id="${second}"]`)).toHaveAttribute('aria-selected', 'true');
    await expect(summary.locator(`[data-testid="element-summary-row"][data-element-id="${top}"]`)).toHaveAttribute('aria-selected', 'false');
  }
  await page.screenshot({path: '../var/screenshots/perspective-generation-document-aware.png'});

  // Back to General: raw attention over every patch, with the same row selection.
  await choosePerspective(page, 'General');
  await expect(picker).toHaveAttribute('data-method-key', 'attention.row/captured_weights_v1');
  await expect(page.getByLabel('Attention layer', {exact: true})).toHaveValue('2');
  await expect(page.getByLabel('Attention head', {exact: true})).toHaveValue('3');
  await expect(page.getByTestId('element-region')).toHaveCount(0);
  await expect(page.getByTestId('attention-patch')).toHaveCount((capture.positions as Position[]).filter(p => p.page_id === pageId && p.bbox && p.index < row.target_count).length);
  // Compatible analyses carry through to Document-aware without reverting to raw.
  await chooseMethod(page, 'attention.mean_heads/captured_prediction_v1');
  await expect(panel.getByRole('region', {name: 'Head aggregation settings', exact: true})).toBeVisible();
  await choosePerspective(page, 'Document-aware');
  await expect(picker).toHaveAttribute('data-method-key', 'attention.element_allocation/ownership_mass_v1');
  await expect(picker).toHaveAttribute('data-input-key', 'attention.mean_heads/captured_prediction_v1');
  await openDetailsTab(page, 'Method');
  await expect(panel.getByTestId('document-grouping')).toContainText('Mean across heads');
  await choosePerspective(page, 'General');
  await expect(picker).toHaveAttribute('data-method-key', 'attention.mean_heads/captured_prediction_v1');
  await expect(page.getByLabel('Attention layer', {exact: true})).toHaveValue('2');
  await choosePerspective(page, 'Document-aware');
  await choosePageColouring(page, 'Element totals');
  await expect(page.getByTestId('element-region').first()).toBeVisible();
  // Turning to the page again clears the element selection.
  await choosePage(page, pageId);
  await withDisplaySettings(page, drawer => expect(pageColouring(drawer, 'Selected patches')).toHaveCount(0));
  await page.setViewportSize({width:1920,height:1080});
  await page.evaluate(()=>window.scrollTo(0,0));
  await page.screenshot({path: '../var/screenshots/nested-attention-elements.png'});
  await showElements(page);
  await elementList(page).getByRole('button', {name: 'Patch contributions and full table ↓', exact: true}).click();
  await page.getByRole('button', {name: 'Show attention patches including background'}).click();
  await expect(page.getByTestId('attention-patch').first()).toBeVisible();
  await expect(page.getByTestId('element-summary')).toContainText('Mean-Head Attention');
  await page.screenshot({path: '../var/screenshots/nested-attention-background.png'});
  await page.getByRole('button', {name: 'Save investigation', exact: true}).click();
  await expect(page).toHaveURL(/investigation_/);
  await page.reload();
  await expect(picker).toHaveAttribute('data-method-key', 'attention.element_allocation/ownership_mass_v1');
  await expect(picker).toHaveAttribute('data-input-key', 'attention.mean_heads/captured_prediction_v1');
  await expect(picker).toHaveAttribute('data-perspective', 'document_aware');
  await withDisplaySettings(page, drawer => expect(pageColouring(drawer, 'Attention patches')).toHaveAttribute('aria-checked','true'));
  await expect(page.getByTestId('attention-patch').first()).toBeVisible();
  await choosePerspective(page, 'General');
  await expect(picker).toHaveAttribute('data-method-key', 'attention.mean_heads/captured_prediction_v1');
  expect(runs()).toBe(0);
});

test('retrieval: document-aware needs a mapped document encoding; the query encoding explains why', async ({page}) => {
  test.setTimeout(120_000);
  const items = (await page.request.get('/api/v1/retrieval/captures?limit=25').then(r => r.json())).items;
  // The newest capture whose document encoding's page maps annotated elements (some newer captures have no mapping).
  type Saved = {attention_encodings?: {pass_kind: string; page_id?: string | null; ownership?: (Record<string, number> | null)[] | null}[]; pages: {page_id: string; ownership: (Record<string, number> | null)[]}[]};
  const mappedDocument = (c: Saved) => c.attention_encodings?.some(e => e.pass_kind === 'document'
    && (e.ownership?.length ? e.ownership : c.pages.find(p => p.page_id === e.page_id)?.ownership ?? []).some(o => Object.keys(o ?? {}).some(k => k.startsWith('element_'))));
  const capture = items.find((c: Saved) => mappedDocument(c) && c.attention_encodings!.some(e => e.pass_kind === 'query'));
  requireSaved(!capture, 'Requires a saved retrieval capture with query and document encodings whose page maps annotated elements');
  const runs = await guardRuns(page);
  const requested: {encoding_id: string; method_id: string}[] = [];
  page.on('request', r => {if (r.method() === 'POST' && /\/retrieval\/captures\/[^/]+\/attention$/.test(r.url())) requested.push(r.postDataJSON());});
  await page.setViewportSize({width: 1440, height: 900});
  await page.goto(`/attention-tool?capture=${capture.capture_id}&role=retrieval`);
  const panel = await detailsPanel(page, 'Retrieval panel');
  const picker = methodChoice(page);
  await expect(page.getByTestId('capture-menu')).toContainText(capture.capture_id.slice(-8));
  await expect(panel.getByRole('button', {name: 'Page', exact: true})).toHaveAttribute('aria-pressed', 'true');
  await expect(page.getByTestId('attention-patch').first()).toBeVisible({timeout: 20_000});
  await page.getByLabel('Attention layer', {exact: true}).selectOption('5');
  await page.getByLabel('Attention head', {exact: true}).selectOption('2');
  const source = Number(await panel.getByLabel('Attention source position').inputValue());

  await choosePerspective(page, 'Document-aware');
  await expect(picker).toHaveAttribute('data-method-key', 'attention.element_allocation/retriever_encoding_v1');
  await expect(page.getByLabel('Attention layer', {exact: true})).toHaveValue('5');
  await expect(page.getByLabel('Attention head', {exact: true})).toHaveValue('2');
  await expect(elementList(page)).toBeVisible();
  await expect(elementList(page).getByTestId('element-value').first()).toBeVisible();
  await openDetailsTab(page, 'Method');
  await expect(panel.getByLabel('Attention source position')).toHaveValue(String(source));
  await expect(panel.getByTestId('structure-status')).toHaveText('Ready: element boxes and patch mapping available.');
  const result = await page.request.post(`/api/v1/retrieval/captures/${capture.capture_id}/attention`,
    {data: {encoding_id: 'document', source_index: source, method_id: 'attention.element_allocation', parameters: {layer: 5, head: 2}}}).then(r => r.json());
  const encodingPage = capture.attention_encodings.find((e: {pass_kind: string}) => e.pass_kind === 'document').page_id;
  const masses = Object.fromEntries((result.row.allocations as Allocation[]).filter(a => a.element_id && a.page_id === encodingPage).map(a => [a.element_id!, a.mass]));
  await expect(page.getByTestId('element-region')).toHaveCount(Object.keys(masses).length, {timeout: 20_000});
  for (const region of await drawnRegions(page)) expect(region.mass).toBeCloseTo(masses[region.id], 12);
  const top = Object.entries(masses).sort((a, b) => b[1] - a[1])[0][0];
  const topNative = (await page.request.get(`/api/v1/pages/${encodingPage}/elements`).then(r => r.json())).items.find((e: {element_id: string}) => e.element_id === top).native_id;
  await showElements(page);
  await elementList(page).locator(`[data-testid="outline-element"][data-native-id="${topNative}"]`).click();
  await choosePageColouring(page, 'Selected patches');
  await expect(page.locator(`rect.docattentionlens-box.is-selected[data-element-id="${top}"]`)).toHaveCount(1);
  const pageEvidence = capture.pages.find((p: {page_id: string}) => p.page_id === encodingPage);
  const drawn = new Set<number>(pageEvidence.positions.filter((p: Position, i: number) => p.bbox && (pageEvidence.ownership[i]?.[top] ?? 0) > 0).map((p: Position) => p.index));
  if (pageEvidence.positions[source]?.bbox) drawn.add(source);
  await expect(page.getByTestId('attention-patch')).toHaveCount(drawn.size);
  await page.screenshot({path: '../var/screenshots/perspective-retrieval-document-aware.png'});

  // A query encoding has no document positions: unavailable, never a zero element mass.
  await openDetailsTab(page, 'Method');
  await panel.getByRole('button', {name: 'Question', exact: true}).click();
  await expect(picker).toHaveAttribute('data-method-key', 'attention.element_allocation/retriever_encoding_v1');
  await expect(panel.getByTestId('structure-status')).toHaveText(/^Unavailable: The query encoding has no mapped document positions\. Document-aware methods require a document encoding with a valid element mapping\.$/);
  await expect(page.getByTestId('structure-gap')).toBeVisible();
  await showElements(page);
  await expect(elementList(page).getByTestId('outline-element').first()).toBeVisible();
  await expect(elementList(page).getByTestId('element-value')).toHaveCount(0);
  await showEvidence(page, 'Element summary');
  await expect(page.getByTestId('element-summary-unavailable')).toBeVisible();
  await expect(page.getByTestId('element-region')).toHaveCount(0);
  await expect(page.getByTestId('element-summary-row')).toHaveCount(0);
  expect(requested.some(r => r.encoding_id === 'query' && r.method_id === 'attention.element_allocation')).toBe(false);
  await page.screenshot({path: '../var/screenshots/perspective-retrieval-query-unavailable.png'});
  await choosePerspective(page, 'General');
  await expect(picker).toHaveAttribute('data-method-key', 'attention.row/retriever_encoding_v1');
  await expect(page.getByLabel('Attention layer', {exact: true})).toHaveValue('5');
  await expect(page.getByLabel('Attention head', {exact: true})).toHaveValue('2');
  await expect(page.getByTestId('structure-gap')).toHaveCount(0);
  expect(runs()).toBe(0);
});

test('narrow layouts keep the perspective control reachable and show the linked summary in Details › Element', async ({page}) => {
  const items = (await page.request.get('/api/v1/generation/captures?limit=25').then(r => r.json())).items;
  const capture = items.find((c: {positions: Position[]}) => c.positions.some(mapped));
  requireSaved(!capture, 'Requires a saved generator capture whose pages map annotated elements');
  const runs = await guardRuns(page);
  await page.setViewportSize({width: 390, height: 844});
  await page.goto('/attention-tool?generation=' + capture.capture_id);
  // The method sentence stays on the page; its picker (a bottom sheet here) holds the perspective.
  await expect(methodChoice(page)).toBeInViewport();
  await choosePerspective(page, 'Document-aware');
  await expect(methodChoice(page)).toHaveAttribute('data-method-key', 'attention.element_allocation/ownership_mass_v1');
  await openDetailsTab(page, 'Method');
  const panel = page.getByRole('complementary', {name: 'Attention panel', exact: true});
  await expect(panel.getByTestId('document-grouping')).toContainText('Raw attention');
  await page.screenshot({path:'../var/screenshots/nested-attention-mobile.png'});
  const source=panel.getByLabel('Attention source',{exact:true});
  await source.scrollIntoViewIfNeeded();
  await expect(source).toBeInViewport();
  const layer=page.getByLabel('Attention layer',{exact:true});
  await layer.scrollIntoViewIfNeeded();
  await expect(layer).toBeInViewport();
  await expect(page.getByRole('complementary', {name: 'Linked element summary', exact: true})).toHaveCount(0);
  await detailsTab(page, 'Element').click();
  await page.getByTestId('element-summary-row').first().scrollIntoViewIfNeeded();
  await expect(page.getByTestId('element-summary-row').first()).toBeInViewport();
  expect(await page.evaluate(() => document.documentElement.scrollWidth <= innerWidth + 1)).toBe(true);
  await page.screenshot({path: '../var/screenshots/perspective-mobile-summary.png'});
  expect(runs()).toBe(0);
});
