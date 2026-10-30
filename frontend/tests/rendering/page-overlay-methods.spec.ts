import { test, expect, type Page } from '@playwright/test';
import * as F from '../../fixtures/attention';
import { chooseMethod, choosePerspective, enabledMethodKeys, openDetailsTab } from '../workspaceHelpers';
import { catalogue, COMPARISON_AXIS, mockAttentionApi, ROW_METHODS, TABLE_METHODS, type Served } from './mockApi';

// Every attention method the workspace offers, selected the way a researcher selects it,
// against a fake API serving the known-answer fixture. For each method the page must show
// exactly the result that method returned: patch weights where it returns a row, element
// regions where it is document-aware, and nothing left over where it returns only statistics.
// Each new method gets a different hot patch, so a stale overlay from the previous method fails.

test.describe.configure({mode: 'serial'});

/** Methods whose page drawing was verified here, by role. The final test checks catalogue coverage. */
const verified = new Map<string, Set<string>>();
const mark = (method_id: string, how: string) => verified.set(method_id, new Set([...(verified.get(method_id) ?? []), how]));

/**
 * Catalogue entries that declare a page overlay the workspace cannot draw today, with the reason.
 * The coverage test fails if one of these starts drawing, so this list cannot go stale.
 */
const KNOWN_GAPS: Record<string, string> = {
  'attention.designated_support': 'Returns statistics and a rank table but no attention row in either role, so the page has nothing to draw.',
};

const round = (v: number) => Math.round(v * 1e9) / 1e9;

async function patches(page: Page, testId = 'attention-patch') {
  return page.getByTestId('page-canvas').first().getByTestId(testId).evaluateAll(rects => rects.map(r => ({
    index: Number(r.getAttribute('data-position-index')), opacity: Math.round(Number(r.getAttribute('fill-opacity')) * 1e9) / 1e9,
    box: ['x', 'y', 'width', 'height'].map(a => Number(r.getAttribute(a))), outlined: r.getAttribute('stroke') !== 'none'})));
}

async function regions(page: Page) {
  return page.getByTestId('element-region').evaluateAll(rects => rects.map(r => ({id: r.getAttribute('data-element-id'),
    mass: Math.round(Number(r.getAttribute('data-mass')) * 1e9) / 1e9, opacity: Math.round(Number(r.getAttribute('fill-opacity')) * 1e9) / 1e9})));
}

/** What the page must draw for one row, computed from the fixture alone. */
function expectedPatches(row: F.Row, positions: {index: number; bbox?: number[] | null; page_id?: string | null}[], maximum: number) {
  return positions.filter(p => p.bbox && p.index < row.target_count).map(p => {
    const [x1, y1, x2, y2] = p.bbox!;
    return {index: p.index, opacity: round(F.overlayOpacity(row.values[p.index], maximum)), box: [x1, y1, x2 - x1, y2 - y1], outlined: p.index === row.source_index};
  });
}
const generatorMaximum = (rows: F.Row[]) => Math.max(...rows.flatMap(row => F.generationPositions.filter(p => p.role === 'image' && p.bbox && p.index < row.target_count).map(p => row.values[p.index])), 0);
const retrieverMaximum = (rows: F.Row[], excluded: number | null) =>
  Math.max(...rows.flatMap(row => F.retrieverPositions.filter(p => p.bbox && p.index !== excluded && p.index < row.target_count).map(p => row.values[p.index] ?? 0)), 1e-8);

function expectedRegions(row: F.Row) {
  const maximum = Math.max(...row.allocations.filter(a => a.element_id).map(a => a.mass), 0);
  return Object.entries(F.elementMasses(row)).map(([id, mass]) => ({id, mass: round(mass), opacity: round(F.overlayOpacity(mass, maximum))}));
}

async function served(api: Awaited<ReturnType<typeof mockAttentionApi>>, method_id: string, kind: Served['kind']) {
  await expect.poll(() => !!api.latest(method_id, kind), {message: `${method_id} was never requested`}).toBe(true);
  return api.latest(method_id, kind)!.result;
}

async function methodOptions(page: Page) {
  return enabledMethodKeys(page);
}

test('generator: each method draws its own result on the page', async ({page}) => {
  test.setTimeout(180_000);
  const api = await mockAttentionApi(page);
  await page.setViewportSize({width: 1440, height: 900});
  await page.goto(`/attention-tool?generation=${F.GENERATION_ID}`);
  await expect(page.getByTestId('attention-patch').first()).toBeVisible();
  // Methods whose result is not a page map show it in Details › Result, which stays on that tab across method changes.
  await openDetailsTab(page, 'Result');
  for (const perspective of ['General', 'Document-aware'] as const) {
    await choosePerspective(page, perspective);
    for (const key of await methodOptions(page)) {
      const method_id = key.split('/')[0];
      await test.step(`${perspective}: ${method_id}`, async () => {
        await chooseMethod(page, key);
        const documentAware = perspective === 'Document-aware';
        // Raw attention and element allocation read the saved row endpoint; every other method its analysis.
        const result = ['attention.row', 'attention.element_allocation'].includes(method_id)
          ? await served(api, 'attention.row', 'row') : await served(api, method_id, 'analysis');
        if (result.row && documentAware) {
          await expect.poll(() => regions(page), {message: `${method_id} element regions`}).toEqual(expectedRegions(result.row));
          await expect(page.getByTestId('attention-patch')).toHaveCount(0);
          mark(method_id, 'generator regions');
        } else if (result.row) {
          const pagePositions = F.generationPositions.filter(p => p.page_id === F.PAGE_ID);
          await expect.poll(() => patches(page), {message: `${method_id} page patches`})
            .toEqual(expectedPatches(result.row, pagePositions, generatorMaximum([result.row])));
          mark(method_id, 'generator patches');
        } else if (result.comparisons?.length) {
          // Comparison maps share one scale: the image maximum over every compared row.
          const scale = generatorMaximum(result.comparisons.map(c => c.row));
          const figures = page.getByTestId('method-result').locator('figure');
          await expect(figures).toHaveCount(result.comparisons.length);
          for (const [i, comparison] of result.comparisons.entries()) {
            await expect.poll(() => figures.nth(i).getByTestId('comparison-patch').evaluateAll(rs => rs.map(r => Math.round(Number(r.getAttribute('fill-opacity')) * 1e9) / 1e9)))
              .toEqual(expectedPatches(comparison.row, F.generationPositions.filter(p => p.page_id === F.PAGE_ID), scale).map(p => p.opacity));
          }
          // The page draws the map chosen in the method sentence (the first when a method opens), on the shared scale.
          await expect.poll(() => patches(page), {message: `${method_id} chosen map on the page`})
            .toEqual(expectedPatches(result.comparisons[0].row, F.generationPositions.filter(p => p.page_id === F.PAGE_ID), scale));
          mark(method_id, 'generator comparison maps');
        } else {
          // Statistics only: nothing from an earlier method may remain on the page.
          await expect(page.getByTestId('method-result')).toBeVisible();
          await expect(page.getByTestId('attention-patch')).toHaveCount(0);
          await expect(page.getByTestId('element-region')).toHaveCount(0);
        }
      });
    }
  }
  expect(api.unexpected).toEqual([]);
});

test('retriever: each encoding method draws its own result on the page', async ({page}) => {
  test.setTimeout(180_000);
  const api = await mockAttentionApi(page);
  await page.setViewportSize({width: 1440, height: 900});
  await page.goto(`/attention-tool?capture=${F.RETRIEVAL_ID}`);
  await expect(page.getByTestId('attention-patch').first()).toBeVisible();
  await openDetailsTab(page, 'Method');
  for (const perspective of ['General', 'Document-aware'] as const) {
    await choosePerspective(page, perspective);
    for (const key of await methodOptions(page)) {
      const method_id = key.split('/')[0];
      await test.step(`${perspective}: ${method_id}`, async () => {
        await chooseMethod(page, key);
        const result = await served(api, method_id, 'retriever');
        // Comparison rows drive the retriever page; the first is shown when a method opens.
        const row = result.row ?? result.comparisons?.[0]?.row;
        if (row && perspective === 'Document-aware') {
          await expect.poll(() => regions(page), {message: `${method_id} element regions`}).toEqual(expectedRegions(row));
          mark(method_id, 'retriever regions');
        } else if (row) {
          // The source patch attends strongly to itself, so it is outlined and left out of the scale.
          const rows = result.comparisons?.length ? result.comparisons.map(c => c.row) : [row];
          await expect.poll(() => patches(page), {message: `${method_id} page patches`})
            .toEqual(expectedPatches(row, F.retrieverPositions, retrieverMaximum(rows, row.source_index)));
          mark(method_id, result.comparisons?.length ? 'retriever comparison row' : 'retriever patches');
        } else {
          // Statistics and tables are named by the result notice, which links to Details › Result.
          await expect(page.getByTestId('result-elsewhere').or(page.getByText('The selected method reports a summary without a page overlay.', {exact: false}))).toBeVisible();
          await expect(page.getByTestId('attention-patch')).toHaveCount(0);
          await expect(page.getByTestId('element-region')).toHaveCount(0);
        }
      });
    }
  }

  await test.step('MaxSim similarity: signed patches and the recorded winner', async () => {
    await page.getByLabel('Retrieval evidence type').selectOption('similarity');
    // The capture opens on the first question word, position 3 ("hot").
    const similarities = F.similarityValues(3);
    const scale = Math.max(...similarities.map(v => Math.abs(v ?? 0)));
    await expect.poll(() => page.getByTestId('similarity-patch').evaluateAll(rs => rs.map(r => ({index: Number(r.getAttribute('data-position-index')),
      fill: r.getAttribute('fill'), opacity: Math.round(Number(r.getAttribute('fill-opacity')) * 1e9) / 1e9, winner: r.getAttribute('stroke') === '#ffc400'}))))
      .toEqual(F.retrieverPositions.filter(p => p.bbox).map(p => ({index: p.index, fill: similarities[p.index]! >= 0 ? F.SIMILARITY_POSITIVE : F.SIMILARITY_NEGATIVE,
        opacity: round(Math.min(.65, Math.abs(similarities[p.index]!) / scale * .65)), winner: p.index === F.SIMILARITY_WINNER})));
    mark('retrieval.maxsim', 'similarity patches');
  });

  await test.step('query–patch similarity: clicking a patch outlines it and reads its column', async () => {
    // No mode to enter: a plain click on the page chooses the patch.
    await expect(page.getByTestId('page-canvas')).toHaveClass(/is-pickable/);
    const image = await page.getByTestId('page-canvas').locator('image').boundingBox();
    const target = 9, {x, y} = F.patchCentre(target);
    const column = page.waitForRequest(r => r.url().includes(`/columns?`) && r.url().includes(`position_index=${target}`));
    await page.mouse.click(image!.x + x / F.PAGE.width * image!.width, image!.y + y / F.PAGE.height * image!.height);
    await column;
    await expect.poll(() => patches(page)).toEqual(F.retrieverPositions.filter(p => p.bbox).map(p => {
      const [x1, y1, x2, y2] = p.bbox!;
      return {index: p.index, opacity: 0, box: [x1, y1, x2 - x1, y2 - y1], outlined: p.index === target};
    }));
    await expect(page.getByTestId('column-context')).toContainText(`Patch ${target}`);
    mark('retrieval.query_patch_similarity', 'selected patch outline');
    // A query token click returns to that token's similarity to every patch.
    await page.getByRole('group', {name: 'Query tokens'}).locator('[data-token-index="3"]').click();
    await expect(page.getByTestId('column-context')).toHaveCount(0);
    await expect(page.getByTestId('similarity-patch')).toHaveCount(F.retrieverPositions.filter(p => p.bbox).length);
  });
  expect(api.unexpected).toEqual([]);
});

test('table case: each Table-aware method draws its row on the page', async ({page}) => {
  test.setTimeout(120_000);
  const api = await mockAttentionApi(page, {table: true});
  await page.setViewportSize({width: 1440, height: 900});
  await page.goto(`/attention-tool?generation=${F.GENERATION_ID}`);
  await expect(page.getByTestId('attention-patch').first()).toBeVisible();
  await choosePerspective(page, 'Table-aware');
  const keys = await methodOptions(page);
  expect(keys.map(k => k.split('/')[0]).sort(), 'Table-aware methods offered for the table case').toEqual([...TABLE_METHODS].sort());
  for (const key of keys) {
    const method_id = key.split('/')[0];
    await test.step(`Table-aware: ${method_id}`, async () => {
      await chooseMethod(page, key);
      // Table-aware results draw the captured row as patches; the table itself is in Details › Table (table-aware.spec.ts).
      const result = await served(api, method_id, 'analysis');
      expect(result.table_allocation, `${method_id} table allocation`).toBeTruthy();
      await expect.poll(() => patches(page), {message: `${method_id} page patches`})
        .toEqual(expectedPatches(result.row!, F.generationPositions.filter(p => p.page_id === F.PAGE_ID), generatorMaximum([result.row!])));
      mark(method_id, 'generator table patches');
    });
  }
  expect(api.unexpected).toEqual([]);
});

test('every catalogue method that declares a page overlay is drawn by one of the roles above', () => {
  const declared = catalogue.filter(m => m.status === 'implemented' && m.displays.includes('page_overlay')).map(m => m.method_id);
  const report = declared.map(id => ({method: id, verified: [...(verified.get(id) ?? [])].join(', ') || KNOWN_GAPS[id] && `known gap: ${KNOWN_GAPS[id]}` || 'NOT VERIFIED'}));
  test.info().annotations.push(...report.map(r => ({type: r.method, description: r.verified})));
  expect(report.filter(r => r.verified === 'NOT VERIFIED').map(r => r.method), 'page-overlay methods never verified on the page').toEqual([]);
  expect(Object.keys(KNOWN_GAPS).filter(id => verified.has(id)), 'known gaps that now draw: remove them from KNOWN_GAPS').toEqual([]);
  // The fake API must know the response shape of every declared overlay (retrieval similarity has its own endpoints).
  expect(declared.filter(id => !KNOWN_GAPS[id] && !id.startsWith('retrieval.') && !ROW_METHODS.has(id) && !COMPARISON_AXIS[id]), 'add these to ROW_METHODS or COMPARISON_AXIS in mockApi.ts').toEqual([]);
});
