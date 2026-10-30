import { test, expect } from '@playwright/test';
import * as F from '../../fixtures/attention';
import { mockAttentionApi } from './mockApi';
import { openDetailsTab } from '../workspaceHelpers';

for (const table of [true, false]) {
  test(`${table ? 'table' : 'element layout'} grounding belongs in case details, including on phones`, async ({page}) => {
    const api = await mockAttentionApi(page, {table});
    await page.setViewportSize({width: 1440, height: 900});
    await page.goto(`/attention-tool?generation=${F.GENERATION_ID}`);
    await expect(page.getByTestId('attention-patch').first()).toBeVisible();
    const label = table ? 'Table grounding' : 'Element layout grounding';
    const header = page.locator('.docattentionlens-ws-bar');
    await expect(header).not.toContainText(label);
    await expect(page.locator('body')).not.toContainText(/draft annotations|not yet human-reviewed/i);

    await header.locator('.docattentionlens-crumb-button').click();
    const details = page.getByRole('dialog', {name: 'Case details', exact: true});
    await expect(details.getByTestId('case-grounding')).toContainText(label);
    if (table) await expect(details.getByTestId('reference-answer')).toHaveText('1');
    await page.keyboard.press('Escape');
    await expect(details).toBeHidden();

    if (table) {
      const panel = await openDetailsTab(page, 'Table');
      await expect(panel.getByTestId('table-details')).toBeVisible();
      await panel.getByText('Geometry and text sources', {exact: true}).click();
      await expect(panel).not.toContainText(/draft annotations|human.review/i);
    }
    await page.setViewportSize({width: 390, height: 844});
    await header.locator('.docattentionlens-crumb-button').click();
    await expect(details.getByTestId('case-grounding')).toContainText(label);
    expect(await page.evaluate(() => document.documentElement.scrollWidth <= innerWidth + 1)).toBe(true);
    expect(api.unexpected).toEqual([]);
  });
}

test('an input without layout elements is not labelled as grounded', async ({page}) => {
  const api = await mockAttentionApi(page);
  await page.route(`**/api/v1/documents/${F.DOCUMENT_ID}`, route => route.fulfill({json: {...F.documentRecord, element_count: 0}}));
  await page.goto(`/attention-tool?generation=${F.GENERATION_ID}`);
  await expect(page.getByTestId('attention-patch').first()).toBeVisible();
  await page.locator('.docattentionlens-crumb-button').click();
  await expect(page.getByRole('dialog', {name: 'Case details', exact: true})).toBeVisible();
  await expect(page.getByTestId('case-grounding')).toHaveCount(0);
  expect(api.unexpected).toEqual([]);
});
