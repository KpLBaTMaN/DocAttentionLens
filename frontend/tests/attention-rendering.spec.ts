import { chooseColourMode, chooseTokenView, clickPage, expectRowTotals, withDisplaySettings, withPanelHidden, currentPage, choosePage, chooseMethod, methodChoice, openDetailsTab, setDetails, chooseDisplayView } from './workspaceHelpers';
import { test, expect, type Page } from '@playwright/test';
import { requireSaved } from './savedEvidence';

// Rendered page overlays must equal the saved values they claim to draw. No acquisitions
// run and no investigation is saved; viewing a method result may add a cached analysis.

type Position = {index: number; role: string; page_id?: string | null; bbox: number[] | null; eligible?: boolean};

// Independent copy of the documented inferno stops; the app's own mapping is not reused here.
const STOPS: [number, number[]][] = [[0, [0, 0, 4]], [.13, [31, 12, 72]], [.25, [85, 15, 109]], [.38, [136, 34, 106]], [.5, [186, 54, 85]],
  [.63, [227, 89, 51]], [.75, [249, 140, 10]], [.88, [249, 201, 50]], [1, [252, 255, 164]]];
function inferno(t: number) {
  const x = Math.min(1, Math.max(0, t));
  const i = STOPS.findIndex(([s]) => x <= s);
  if (i <= 0) return STOPS[0][1];
  const [t0, c0] = STOPS[i - 1], [t1, c1] = STOPS[i];
  return c0.map((c, k) => Math.round(c + (x - t0) / (t1 - t0) * (c1[k] - c)));
}

async function blockAcquisition(page: Page) {
  let attempts = 0;
  await page.route(/\/api\/v1\/(generation|retrieval)\/runs$/, route => {attempts++; return route.abort();});
  return () => attempts;
}

// The legend's share formatting: one decimal below 10%, whole percent above.
const percent = (x: number) => x > 0 && x < .001 ? '<0.1%' : `${(x * 100).toFixed(x < .1 ? 1 : 0)}%`;

async function drawn(page: Page) {
  return page.getByTestId('attention-patch').evaluateAll(rects => rects.map(r => ({
    index: Number(r.getAttribute('data-position-index')), fill: r.getAttribute('fill'), opacity: Number(r.getAttribute('fill-opacity')),
    box: ['x', 'y', 'width', 'height'].map(a => Number(r.getAttribute(a))), stroke: r.getAttribute('stroke')})));
}

test('generator overlay, heatmap and smoothed raster draw each supplied page from the saved row', async ({page}) => {
  const items = (await page.request.get('/api/v1/generation/captures?limit=25').then(r => r.json())).items;
  const capture = items.find((c: {context: {pages: unknown[]}}) => c.context.pages.length === 2) ?? items[0];
  requireSaved(!capture, 'Requires a saved generator capture');
  const acquisitions = await blockAcquisition(page);
  const row = await page.request.get(`/api/v1/generation/captures/${capture.capture_id}/rows?step=0&layer=0&head=0`).then(r => r.json());
  const images: Position[] = capture.positions.filter((p: Position) => p.role === 'image' && p.bbox && p.index < row.target_count);
  // Image contrast shares one maximum across every supplied page so pages stay comparable.
  const maximum = Math.max(...images.map(p => row.values[p.index]));
  await page.setViewportSize({width: 1440, height: 900});
  await page.goto('/attention-tool?generation=' + capture.capture_id);
  await expect(page.getByTestId('attention-patch').first()).toBeVisible({timeout: 20_000});
  await expect(page.getByTestId('display-scale')).toContainText(maximum.toFixed(6));
  // Image contrast magnifies small weights, so the legend states the image share of the row.
  const share = row.image_mass / row.total_mass;
  const firstPage = await currentPage(page);
  const pageShare = images.filter(p => p.page_id === firstPage).reduce((sum, p) => sum + row.values[p.index], 0) / row.total_mass;
  const several = capture.context.pages.length > 1;
  await expect(page.getByTestId('image-share')).toHaveText(`${several ? 'Pages get' : 'Page gets'} ${percent(share)} of attention` + (several ? ` · this page ${percent(pageShare)}` : ''));
  // Details › Display › Scale states the same share against the row mass.
  await openDetailsTab(page, 'Display');
  await chooseDisplayView(page, 'Scale');
  await expect(page.locator('.docattentionlens-scale-mass')).toContainText(`Image ${percent(share)} of row mass` + (several ? ` · this page ${percent(pageShare)}` : ''));
  await setDetails(page, false);
  // A small image share is marked, and its explanation opens on click (not hover).
  const why = page.getByRole('button', {name: 'Why the image share is small', exact: true});
  if (share > 0 && share < .05) {
    await why.click();
    await expect(page.getByTestId('low-mass-note')).toHaveText('Most attention is off the page; Image contrast magnifies small weights.');
    await why.click();
  } else await expect(why).toHaveCount(0);
  await page.getByRole('button', {name: 'Image focus', exact: true}).click();
  await expect(page.getByTestId('image-share')).toBeVisible();
  await expect(page.getByTestId('display-scale')).toBeVisible();
  await page.getByRole('button', {name: 'Image focus', exact: true}).click();
  for (const supplied of capture.context.pages) {
    await choosePage(page, supplied.page_id);
    await chooseColourMode(page, 'Overlay');
    const expected = images.filter(p => p.page_id === supplied.page_id);
    await expect(page.getByTestId('attention-patch')).toHaveCount(expected.length);
    const overlay = await drawn(page);
    expect(overlay.map(r => r.index).sort((a, b) => a - b)).toEqual(expected.map(p => p.index));
    for (const rect of overlay) {
      const position = capture.positions[rect.index];
      expect(rect.opacity).toBeCloseTo(Math.min(.65, row.values[rect.index] / maximum * .65), 9);
      const [x1, y1, x2, y2] = position.bbox;
      expect(rect.box).toEqual([x1, y1, x2 - x1, y2 - y1].map(v => expect.closeTo(v, 6)));
    }
    await chooseColourMode(page, 'Heatmap');
    const heat = await drawn(page);
    for (const rect of heat) expect(rect.fill).toBe(`rgb(${inferno(row.values[rect.index] / maximum).join(',')})`);
    expect(heat.every(r => r.opacity === 1)).toBe(true);
    await withDisplaySettings(page, drawer => drawer.getByLabel('Smooth', {exact: true}).check());
    const raster = page.getByTestId('attention-raster');
    await expect(raster).toHaveCount(1);
    const columns = supplied.grid_thw[2] / 2;
    const pixels = await raster.evaluate(async (element, size) => {
      const image = new Image(); image.src = element.getAttribute('href')!; await image.decode();
      const canvas = document.createElement('canvas'); canvas.width = image.width; canvas.height = image.height;
      const context = canvas.getContext('2d')!; context.drawImage(image, 0, 0);
      return {width: image.width, height: image.height, data: [...context.getImageData(0, 0, size.columns, size.rows).data],
        box: ['x', 'y', 'width', 'height'].map(a => Number(element.getAttribute(a)))};
    }, {columns, rows: supplied.grid_thw[1] / 2});
    expect([pixels.width, pixels.height]).toEqual([columns, supplied.grid_thw[1] / 2]);
    expect(pixels.box).toEqual([0, 0, supplied.source_width, supplied.source_height].map(v => expect.closeTo(v, 6)));
    supplied.position_indices.forEach((index: number, k: number) => {
      expect(pixels.data.slice(k * 4, k * 4 + 3)).toEqual(inferno(row.values[index] / maximum));
    });
    await withDisplaySettings(page, drawer => drawer.getByLabel('Smooth', {exact: true}).uncheck());
  }
  expect(acquisitions()).toBe(0);
});

test('MaxSim overlay draws signed saved similarities and marks the recorded winner', async ({page}) => {
  const items = (await page.request.get('/api/v1/retrieval/captures?limit=25').then(r => r.json())).items;
  const capture = items.find((c: {attention_encodings?: unknown[]}) => !c.attention_encodings?.length);
  requireSaved(!capture, 'Requires a saved score-only retrieval capture');
  const acquisitions = await blockAcquisition(page);
  await page.setViewportSize({width: 1440, height: 900});
  await page.goto('/attention-tool?capture=' + capture.capture_id);
  await expect(page.getByTestId('similarity-patch').first()).toBeVisible({timeout: 20_000});
  const pageId = await currentPage(page);
  const evidence = capture.pages.find((p: {page_id: string}) => p.page_id === pageId);
  // Position 0 is <bos>, whose winner is always the page's <bos>; a new view opens on the first question word.
  const word = capture.query_positions.find((p: {eligible: boolean; role: string}) => p.eligible && p.role === 'query_text')?.index ?? 0;
  await expect(page.getByRole('group', {name: 'Query tokens'}).locator('[aria-pressed="true"]')).toHaveAttribute('data-token-index', String(word));
  const row = await page.request.get(`/api/v1/retrieval/captures/${capture.capture_id}/rows?page_id=${pageId}&query_index=${word}`).then(r => r.json());
  await expect(page.getByTestId('retrieval-winner')).toContainText(`dot product ${row.maximum.toFixed(6)}`);
  const scale = Math.max(...row.similarities.map((v: number | null) => Math.abs(v ?? 0)));
  const rects = await page.getByTestId('similarity-patch').evaluateAll(rs => rs.map(r => ({index: Number(r.getAttribute('data-position-index')),
    fill: r.getAttribute('fill'), opacity: Number(r.getAttribute('fill-opacity')), stroke: r.getAttribute('stroke')})));
  expect(rects).toHaveLength(evidence.positions.filter((p: Position) => p.bbox && p.eligible).length);
  for (const rect of rects) {
    const value = row.similarities[rect.index];
    expect(rect.fill).toBe(value >= 0 ? '#b080d1' : '#247ed5');
    expect(rect.opacity).toBeCloseTo(Math.min(.65, Math.abs(value) / scale * .65), 9);
  }
  const winners = rects.filter(r => r.stroke === '#ffc400').map(r => r.index);
  if (evidence.positions[row.winner_index].bbox) expect(winners).toEqual([row.winner_index]);
  else {
    expect(winners).toEqual([]);
    await expect(page.getByTestId('retrieval-winner')).toContainText('Non-page winner has no image box');
  }
  expect(acquisitions()).toBe(0);
});

test('retriever encoding overlay draws the saved document row and outlines its source patch', async ({page}) => {
  const items = (await page.request.get('/api/v1/retrieval/captures?limit=25').then(r => r.json())).items;
  // A bidirectional document encoding (ColPali) opens on source 0, a page patch; causal encoders open on their last position.
  const capture = items.find((c: {attention_encodings?: {encoding_id: string; causal?: boolean}[]}) => c.attention_encodings?.some(e => e.encoding_id === 'document' && !e.causal));
  requireSaved(!capture, 'Requires a saved retrieval capture with bidirectional encoder attention');
  const acquisitions = await blockAcquisition(page);
  const encoding = capture.attention_encodings.find((e: {encoding_id: string}) => e.encoding_id === 'document');
  const evidence = capture.pages.find((p: {page_id: string}) => p.page_id === encoding.page_id);
  const result = await page.request.post(`/api/v1/retrieval/captures/${capture.capture_id}/attention`,
    {data: {encoding_id: 'document', source_index: 0, method_id: 'attention.row', parameters: {layer: 0, head: 0}}}).then(r => r.json());
  const visual: Position[] = evidence.positions.filter((p: Position) => p.bbox);
  // Source 0 is a patch; its attention to itself is excluded from the Image contrast maximum.
  const maximum = Math.max(...visual.filter(p => p.index !== 0).map(p => result.row.values[p.index]));
  await page.setViewportSize({width: 1440, height: 900});
  await page.goto('/attention-tool?capture=' + capture.capture_id);
  await withPanelHidden(page, () => expect(page.getByTestId('retriever-attention-context')).toContainText('Document encoding · Source 0 · layer 0 · head 0'));
  await expect(page.getByTestId('attention-patch')).toHaveCount(visual.length);
  const rects = await drawn(page);
  for (const rect of rects) expect(rect.opacity).toBeCloseTo(Math.min(.65, result.row.values[rect.index] / maximum * .65), 9);
  expect(rects.filter(r => r.stroke === '#39a9e0').map(r => r.index)).toEqual([0]);
  await expect(page.getByTestId('display-scale')).toContainText(`${maximum.toFixed(6)} · image maximum excluding the source patch`);
  await expect(page.getByTestId('image-share')).toContainText(`Page gets ${percent(result.row.image_mass / result.row.total_mass)} of attention`);
  await expectRowTotals(page, result.row.image_mass.toFixed(6), 'retriever-attention-mass');
  expect(acquisitions()).toBe(0);
});

test('head comparisons follow the page display mode and colour mapping', async ({page}) => {
  const items = (await page.request.get('/api/v1/generation/captures?limit=25').then(r => r.json())).items;
  const capture = items.find((c: {context: {pages: unknown[]}}) => c.context.pages.length === 1);
  requireSaved(!capture, 'Requires a saved one-page generator capture');
  const acquisitions = await blockAcquisition(page);
  const heads = Array.from({length: capture.query_head_count}, (_, h) => h);
  const rows = await Promise.all(heads.map(h => page.request.get(`/api/v1/generation/captures/${capture.capture_id}/rows?step=0&layer=0&head=${h}`).then(r => r.json())));
  const images: Position[] = capture.positions.filter((p: Position) => p.role === 'image' && p.bbox);
  // Unreduced comparisons share one scale: the image maximum over every compared head.
  const scale = Math.max(...rows.flatMap(row => images.map(p => row.values[p.index])));
  await page.setViewportSize({width: 1440, height: 900});
  await page.goto('/attention-tool?generation=' + capture.capture_id);
  await expect(page.getByTestId('attention-patch').first()).toBeVisible({timeout: 20_000});
  await chooseMethod(page, 'attention.head_by_head/captured_prediction_v1');
  await chooseColourMode(page, 'Heatmap');
  const figures = page.getByTestId('method-result').locator('figure');
  await expect(figures).toHaveCount(Math.min(8, heads.length));
  for (const h of [0, 1]) {
    const patches = await figures.nth(h).getByTestId('comparison-patch').evaluateAll(rs => rs.map(r => ({fill: r.getAttribute('fill'), opacity: r.getAttribute('fill-opacity')})));
    expect(patches).toHaveLength(images.length);
    patches.forEach((patch, k) => expect(patch).toEqual({fill: `rgb(${inferno(rows[h].values[images[k].index] / scale).join(',')})`, opacity: '1'}));
  }
  await chooseColourMode(page, 'Overlay');
  const overlay = await figures.nth(0).getByTestId('comparison-patch').evaluateAll(rs => rs.map(r => ({fill: r.getAttribute('fill'), opacity: Number(r.getAttribute('fill-opacity'))})));
  overlay.forEach((patch, k) => {expect(patch.fill).toBe('#d99a36'); expect(patch.opacity).toBeCloseTo(Math.min(.65, rows[0].values[images[k].index] / scale * .65), 9);});
  expect(acquisitions()).toBe(0);
});

test('the page and the selected method always describe the same source and quantity', async ({page}) => {
  const items = (await page.request.get('/api/v1/generation/captures?limit=25').then(r => r.json())).items;
  const capture = items.find((c: {source_rows?: unknown[]}) => c.source_rows?.length);
  requireSaved(!capture, 'Requires a saved generator capture with question rows');
  const acquisitions = await blockAcquisition(page);
  await page.goto('/attention-tool?generation=' + capture.capture_id);
  await expect(page.getByTestId('attention-patch').first()).toBeVisible({timeout: 20_000});
  const method = methodChoice(page);
  // A patch click shows what attends to it: an incoming column, which is raw attention.
  await clickPage(page, .5, .5);
  await expect(page.getByTestId('column-context')).toContainText('raw attention', {timeout: 20_000});
  // A row statistic reads the prediction row, so the page returns to that row.
  await chooseMethod(page, 'attention.entropy/shannon_row_v1');
  await expect(page.getByTestId('column-context')).toHaveCount(0);
  await expect(page.getByTestId('result-elsewhere')).toContainText('statistics');
  await expect(page.getByTestId('method-result')).toContainText('Shannon entropy');
  // Clicking a patch again selects Raw attention; the next token click restores the statistic.
  await clickPage(page, .5, .5);
  await expect(method).toHaveAttribute('data-method-key', 'attention.row/captured_weights_v1');
  await expect(page.getByTestId('column-context')).toContainText('raw attention', {timeout: 20_000});
  await chooseTokenView(page, 'Parts');
  await page.getByRole('group', {name: 'Answer tokens'}).getByTestId('token-chip').first().click();
  await expect(method).toHaveAttribute('data-method-key', 'attention.entropy/shannon_row_v1');
  await expect(page.getByTestId('column-context')).toHaveCount(0);
  expect(acquisitions()).toBe(0);
});
