import { expect, test, type Page } from '@playwright/test';
import * as F from '../../fixtures/attention';
import { mockAttentionApi, profileHot } from './mockApi';
import { openDetailsTab, setDetails } from '../workspaceHelpers';

// Element attention profile in Details › Element on the known-answer fixture (2 layers × 4 heads, 2 KV groups).
// Each layer and head is its own fixture row; expected values are computed here from the fixture, not the app.
const text = F.ELEMENTS.find(e => e.element_id === F.TEXT_ID)!;
const heads = F.generationCapture.query_head_count;
const layers = F.generationCapture.layer_count;
const textMass = (layer: number, head: number) => {
  const values = F.patchValues(profileHot(layer, head, heads));
  return text.patches.reduce((sum, k) => sum + values[k], 0);
};
const cells = Array.from({length: layers}, (_, l) => Array.from({length: heads}, (_, h) => ({l, h, v: textMass(l, h)}))).flat()
  .sort((a, b) => b.v - a.v || a.l - b.l || a.h - b.h);
const element = (page: Page, nativeId: number) => page.locator('#details-elements').locator(`[data-testid="outline-element"][data-native-id="${nativeId}"]`);

async function openProfile(page: Page) {
  await page.setViewportSize({width: 1440, height: 900});
  await page.goto(`/attention-tool?generation=${F.GENERATION_ID}`);
  await setDetails(page, true);
  await openDetailsTab(page, 'Element');
  await element(page, text.native_id).click();
  const profile = page.getByTestId('element-profile');
  await expect(profile.getByTestId('element-profile-grid')).toBeVisible();
  return profile;
}

test('Details › Element ranks every layer and head for the selected element and drives the scrubber', async ({page}) => {
  const api = await mockAttentionApi(page);
  const profile = await openProfile(page);
  await expect(profile.getByRole('gridcell')).toHaveCount(layers * heads);
  // One profile request per source: never repeated for a layer, head or element change.
  const requests = () => api.served.filter(s => s.method_id === 'attention.element_profile').length;
  expect(requests()).toBe(1);
  const top = cells[0];
  const percent = `${(top.v * 100).toFixed(2)}%`;
  await expect(profile.getByTestId('profile-top-heads').locator('li').first()).toContainText(`L${top.l} · H${top.h} ${percent}`);
  await profile.getByTestId('profile-top-heads').getByRole('button', {name: `L${top.l} · H${top.h}`}).click();
  await expect(page.getByLabel('Attention layer', {exact: true})).toHaveValue(String(top.l));
  await expect(page.getByRole('radio', {name: `Head ${top.h}`, exact: true})).toHaveAttribute('aria-checked', 'true');
  await expect(profile.getByTestId('element-profile-readout')).toContainText(`Layer ${top.l} · head ${top.h} · shown on the page`);
  await expect(profile.getByTestId('profile-mass')).toHaveText(`${percent} of the row`);
  // The page now reads that layer and head's row. (The mock's raw rows do not vary by axis, so their values are not
  // compared here; the server-side tests check each profile cell against element aggregation.)
  await expect.poll(() => api.latest('attention.row', 'row')?.result.row?.layer).toBe(top.l);
  expect(api.latest('attention.row', 'row')!.result.row!.head).toBe(top.h);
  // Selecting the other element reuses the same profile.
  await element(page, 1).click();
  await expect(profile.getByTestId('profile-rank')).toBeVisible();
  expect(requests()).toBe(1);
  expect(api.unexpected).toEqual([]);
});

test('Heads and layers is findable: Browse methods points to it and Details › Element says what it needs', async ({page}) => {
  const api = await mockAttentionApi(page);
  // Narrower windows keep the tabbed Details panel, where the element profile is a tab away from the method list.
  await page.setViewportSize({width: 1100, height: 900});
  await page.goto(`/attention-tool?generation=${F.GENERATION_ID}`);
  await page.getByRole('button', {name: /Browse methods/}).click();
  await expect(page.getByTestId('heads-layers-pointer')).toBeVisible();
  await page.getByRole('button', {name: 'Open Heads and layers', exact: true}).click();
  await expect(page.locator('#details-tabpanel')).toHaveAttribute('aria-label', 'Details: Element');
  await expect(page.getByTestId('element-profile-hint')).toContainText('Select an element in the list below.');
  await element(page, text.native_id).click();
  await expect(page.getByTestId('element-profile-grid')).toBeVisible();
  await expect(page.getByTestId('element-profile-hint')).toHaveCount(0);
  expect(api.unexpected).toEqual([]);
});

test('three columns: the element inspector keeps Heads and layers beside the page, filled from the ledger', async ({page}) => {
  const api = await mockAttentionApi(page);
  await page.setViewportSize({width: 1440, height: 900});
  await page.goto(`/attention-tool?generation=${F.GENERATION_ID}`);
  const inspector = page.locator('#workspace-inspector');
  await expect(inspector.getByTestId('element-profile-hint')).toContainText('Select an element in the list left of the page.');
  // The method list needs no pointer: the profile is already on screen.
  await page.getByRole('button', {name: /Browse methods/}).click();
  await expect(page.locator('#method-panel').getByTestId('method-browser')).toBeVisible();
  await expect(page.getByTestId('heads-layers-pointer')).toHaveCount(0);
  await page.keyboard.press('Escape');
  await expect(page.locator('#method-panel')).toHaveCount(0);
  await page.locator('.docattentionlens-ledger-column').locator(`[data-testid="outline-element"][data-native-id="${text.native_id}"]`).click();
  await expect(inspector.getByTestId('element-profile-grid')).toBeVisible();
  await expect(inspector.getByTestId('selected-element')).toContainText(`Element ${text.native_id}`);
  // The Details drawer covers the inspector while open and gives way to it when closed.
  await page.locator('.docattentionlens-method-bar').getByRole('button', {name: 'Details', exact: true}).click();
  await expect(page.getByRole('tablist', {name: 'Details sections'}).getByRole('tab')).toHaveText(['Result', 'Display', 'Run', 'Notes']);
  await expect(inspector).toHaveCount(0);
  await page.locator('#workspace-details').getByRole('button', {name: 'Close details', exact: true}).click();
  await expect(inspector.getByTestId('element-profile-grid')).toBeVisible();
  expect(api.unexpected).toEqual([]);
});

test('the keyboard moves a cursor; Enter shows that head and the scrubber bars carry the layer means', async ({page}) => {
  const api = await mockAttentionApi(page);
  const profile = await openProfile(page);
  const grid = profile.getByTestId('element-profile-grid');
  await grid.focus();
  await page.keyboard.press('ArrowDown');
  await page.keyboard.press('ArrowRight');
  await expect(page.getByLabel('Attention layer', {exact: true})).toHaveValue('0');
  await expect(grid).toHaveAttribute('aria-activedescendant', /-l1-h1$/);
  await page.keyboard.press('Enter');
  await expect(page.getByLabel('Attention layer', {exact: true})).toHaveValue('1');
  await expect(page.getByRole('radio', {name: 'Head 1', exact: true})).toHaveAttribute('aria-checked', 'true');
  const means = Array.from({length: layers}, (_, l) => Array.from({length: heads}, (_, h) => textMass(l, h)).reduce((a, b) => a + b) / heads);
  const heights = await page.locator('.docattentionlens-scrubber-bar').evaluateAll(bars => bars.map(b => (b as HTMLElement).style.height));
  expect(heights).toEqual(means.map(m => `${Math.round(20 + 80 * m / Math.max(...means))}%`));
  await expect(page.locator('.docattentionlens-scrubber-bars')).toHaveAttribute('title', /layer mean of text · 2/);
  expect(api.unexpected).toEqual([]);
});
