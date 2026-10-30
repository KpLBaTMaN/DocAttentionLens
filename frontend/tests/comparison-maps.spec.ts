import { test, expect } from '@playwright/test';
import { chooseMethod, methodChoice } from './workspaceHelpers';
import { findGenerationCapture, requireSaved } from './savedEvidence';

/**
 * Methods that return one map per layer or head (no single row) draw the chosen map on the page, on the scale
 * shared by every map; the method sentence chooses which. Before this the page was blank in the General view.
 */
for (const [key, axis, count] of [
  ['attention.across_layers/captured_prediction_v1', 'Layer', 'layer_count'],
  ['attention.head_by_head/captured_prediction_v1', 'Head', 'query_head_count'],
] as const) {
  test(`${key.split('/')[0]} draws the chosen ${axis.toLowerCase()} map on the page`, async ({page}) => {
    const capture = await findGenerationCapture(page.request, c => c.layer_count > 2 && c.query_head_count > 2);
    requireSaved(!capture, 'Requires a saved generator capture');
    await page.route(/\/api\/v1\/(generation|retrieval)\/runs$/, route => route.abort());
    await page.setViewportSize({width: 1440, height: 900});
    await page.goto('/attention-tool?generation=' + capture!.capture_id);
    await expect(page.getByTestId('attention-patch').first()).toBeVisible({timeout: 20_000});
    const analyses = page.waitForResponse(r => r.url().endsWith('/api/v1/methods/analyses') && r.request().method() === 'POST');
    await chooseMethod(page, key);
    const result = await (await analyses).json();
    expect(result.row ?? null).toBeNull();
    const maps: {label: string; row: {values: number[]; image_mass: number; total_mass: number}}[] = result.comparisons;
    expect(maps.length).toBe(capture![count]);
    await expect(methodChoice(page)).toHaveAttribute('data-method-key', key);
    // The page draws the first map and names it; every map is offered.
    const shown = page.getByLabel('Shown comparison', {exact: true});
    await expect(shown).toHaveValue('0');
    await expect(shown.locator('option')).toHaveCount(maps.length);
    await expect(shown.locator('option').first()).toHaveText(maps[0].label);
    await expect(page.getByTestId('attention-patch').first()).toBeVisible();
    await expect(page.getByTestId('image-share')).toBeVisible();
    // Choosing another map redraws the page with that map's share of attention.
    const at = maps.length - 1;
    await shown.selectOption(String(at));
    await expect(shown).toHaveValue(String(at));
    const share = maps[at].row.image_mass / maps[at].row.total_mass;
    if (share >= 0.001) await expect(page.getByTestId('image-share')).toContainText(`${(share * 100).toFixed(share < .1 ? 1 : 0)}%`);
    await page.getByRole('button', {name: 'Previous map', exact: true}).click();
    await expect(shown).toHaveValue(String(at - 1));
  });
}
