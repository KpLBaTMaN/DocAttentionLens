import { test, expect, type Page } from '@playwright/test';
import * as F from '../../fixtures/attention';
import { chooseMethod, choosePerspective, methodChoice, openDetailsTab } from '../workspaceHelpers';
import { mockAttentionApi, TABLE_CELL_IDS, tableCellPatch, tableRelated } from './mockApi';

// A table grounding question on the known-answer page: a 2 × 4 table whose cells each cover one patch
// (mockApi.ts `tableRecord`). The Table-aware workspace must draw the method's row as page patches,
// show the same weights per cell in Details › Table, outline a chosen cell on the page, and send the
// saved captures a comparison reads with its request.

const round = (v: number) => Math.round(v * 1e9) / 1e9;

async function patches(page: Page) {
  return page.getByTestId('page-canvas').first().getByTestId('attention-patch').evaluateAll(rects => rects.map(r => ({
    index: Number(r.getAttribute('data-position-index')), opacity: Math.round(Number(r.getAttribute('fill-opacity')) * 1e9) / 1e9,
    box: ['x', 'y', 'width', 'height'].map(a => Number(r.getAttribute(a)))})));
}

/** Patch opacities for a generator row on the fixture page, scaled to its image maximum. */
function expectedPatches(row: F.Row) {
  const pagePatches = F.generationPositions.filter(p => p.page_id === F.PAGE_ID && p.bbox && p.index < row.target_count);
  const maximum = Math.max(...pagePatches.map(p => row.values[p.index]), 0);
  return pagePatches.map(p => {
    const [x1, y1, x2, y2] = p.bbox!;
    return {index: p.index, opacity: round(F.overlayOpacity(row.values[p.index], maximum)), box: [x1, y1, x2 - x1, y2 - y1]};
  });
}

/** Each cell's mass is the weight of the patch it covers. */
const cellMass = (row: F.Row, cellId: string) => round(row.values[F.generatorPatchIndex(tableCellPatch(cellId))]);

async function openTableCase(page: Page) {
  const api = await mockAttentionApi(page, {table: true});
  await page.setViewportSize({width: 1440, height: 900});
  await page.goto(`/attention-tool?generation=${F.GENERATION_ID}`);
  await expect(page.getByTestId('attention-patch').first()).toBeVisible();
  await choosePerspective(page, 'Table-aware');
  return api;
}

async function latestRequest(api: Awaited<ReturnType<typeof mockAttentionApi>>, method_id: string) {
  await expect.poll(() => !!api.latest(method_id, 'analysis'), {message: `${method_id} was never requested`}).toBe(true);
  return api.latest(method_id, 'analysis')!;
}

test('cell allocation: the row on the page, its cells in Details › Table and a chosen cell outlined', async ({page}) => {
  const api = await openTableCase(page);
  // Table-aware opens on Cell allocation.
  await expect(methodChoice(page)).toHaveAttribute('data-method-key', 'table.cell_allocation/table_cell_partition_v1');
  const {result: {row}} = await latestRequest(api, 'table.cell_allocation');
  await expect.poll(() => patches(page), {message: 'cell allocation page patches'}).toEqual(expectedPatches(row!));

  const panel = await openDetailsTab(page, 'Table');
  const heatmap = panel.getByTestId('table-heatmap');
  await expect(heatmap).toBeVisible();
  await expect.poll(() => heatmap.getByRole('gridcell').evaluateAll(cells => cells.map(c => [c.getAttribute('data-cell-id'), Math.round(Number(c.getAttribute('data-mass')) * 1e9) / 1e9])),
    {message: 'cell masses in the heatmap'}).toEqual(TABLE_CELL_IDS.map(id => [id, cellMass(row!, id)]));

  // Choosing the answer cell (row 2, column 3, on patch 6) outlines it on the page and describes it.
  const outline = page.getByTestId('page-canvas').first().getByTestId('table-cell-highlight');
  await expect(outline).toHaveCount(0);
  const answer = heatmap.locator('[data-cell-id="cell-12"]');
  await answer.click();
  await expect(answer).toHaveAttribute('aria-pressed', 'true');
  await expect(outline).toHaveCount(1);
  await expect(outline).toHaveAttribute('data-cell-id', 'cell-12');
  const [x1, y1, x2, y2] = F.patchBox(6);
  expect(await outline.evaluate(r => ['x', 'y', 'width', 'height'].map(a => Number(r.getAttribute(a))))).toEqual([x1, y1, x2 - x1, y2 - y1]);
  await expect(heatmap.getByTestId('table-cell-detail')).toContainText('row 2, column 3 · 1');
  // Choosing it again clears the outline.
  await answer.click();
  await expect(outline).toHaveCount(0);
  expect(api.unexpected).toEqual([]);
});

test('lookup path and matched contrast send the saved captures they compare with', async ({page}) => {
  const api = await openTableCase(page);
  await openDetailsTab(page, 'Method');

  await chooseMethod(page, 'table.lookup_path/table_cell_partition_v1');
  const baseline = page.getByTestId('table-baseline');
  await expect(baseline).toContainText('Compare with 2 saved captures of other questions on this table');
  // The baseline is on by default: the request lists this capture first, then the baseline captures.
  await expect.poll(async () => (await latestRequest(api, 'table.lookup_path')).request?.sample_capture_ids)
    .toEqual([F.GENERATION_ID, ...tableRelated.baseline_capture_ids]);
  // Off, the same method reads this capture alone.
  const before = api.served.length;
  await baseline.getByRole('checkbox').uncheck();
  await expect.poll(() => api.served.slice(before).some(s => s.method_id === 'table.lookup_path')).toBe(true);
  expect(api.latest('table.lookup_path', 'analysis')!.request).not.toHaveProperty('sample_capture_ids');

  await chooseMethod(page, 'table.matched_contrast/table_cell_partition_v1');
  await expect(page.getByLabel('Matched question capture')).toHaveValue('generation_same_row');
  const contrast = await latestRequest(api, 'table.matched_contrast');
  expect(contrast.request?.comparison_capture_id).toBe('generation_same_row');
  expect(contrast.request).not.toHaveProperty('sample_capture_ids');
  expect(api.unexpected).toEqual([]);
});
