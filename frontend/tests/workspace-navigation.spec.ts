import { test, expect, type Page } from '@playwright/test';
import { readFileSync } from 'node:fs';
import { bottomBar, choosePerspective, columnsLayout, detailsTab, filmstrip, methodChoice, setDetails, currentPage, openDetailsTab } from './workspaceHelpers';

/** Wave 2b: Details tabs, filmstrip, phone bottom bar, command palette and keyboard shortcuts. */
const meta = JSON.parse(readFileSync('../var/verification/generation-smoke.json', 'utf8'));

/** A one-page generator capture: no filmstrip, no Pages button. */
async function onePageCapture(page: Page) {
  const items = (await page.request.get('/api/v1/generation/captures?limit=25').then(r => r.json())).items as {capture_id: string; context: {pages: unknown[]; omitted_page_ids: unknown[]}}[];
  return items.find(c => c.context.pages.length + c.context.omitted_page_ids.length === 1);
}

const noHorizontalScroll = (page: Page) => page.evaluate(() => document.documentElement.scrollWidth <= innerWidth + 1);

test('clicking an element on the page opens Details on Element, on desktop and as a phone sheet', async ({page}) => {
  await page.setViewportSize({width: 1440, height: 900});
  await page.goto('/attention-tool?generation=' + meta.capture_id);
  await expect(page.getByTestId('attention-patch').first()).toBeVisible({timeout: 20_000});
  // Document-aware: the page shows element masses and a click chooses an element (not a patch).
  await choosePerspective(page, 'Document-aware');
  await openDetailsTab(page, 'Display');
  await setDetails(page, false);
  const region = page.getByTestId('element-region').first();
  await expect(region).toBeVisible();
  const id = await region.getAttribute('data-element-id');
  const box = (await region.boundingBox())!;
  await page.mouse.click(box.x + box.width / 2, box.y + box.height / 2);
  // At this width the element inspector beside the page shows it; the tabbed layout opens Details on Element.
  if (await columnsLayout(page)) await expect(page.locator('#workspace-inspector')).toBeVisible();
  else {
    await expect(page.getByRole('complementary', {name: 'Attention panel', exact: true})).toBeVisible();
    await expect(detailsTab(page, 'Element')).toHaveAttribute('aria-selected', 'true');
  }
  await expect(page.getByTestId('selected-element')).toBeVisible();
  await expect(page.locator(`rect.docattentionlens-box.is-selected[data-element-id="${id}"]`)).toHaveCount(1);
  // Phones: the same click opens the Details sheet with the page still visible above it.
  await page.setViewportSize({width: 375, height: 812});
  await page.keyboard.press('Escape');
  await expect(page.locator('#workspace-details')).toBeHidden();
  await page.evaluate(() => window.scrollTo(0, 0));
  const small = (await page.getByTestId('element-region').first().boundingBox())!;
  await page.mouse.click(small.x + small.width / 2, small.y + small.height / 2);
  await expect(page.locator('#workspace-details')).toBeVisible();
  await expect(detailsTab(page, 'Element')).toHaveAttribute('aria-selected', 'true');
  const sheetTop = await page.locator('#workspace-details').evaluate(e => e.getBoundingClientRect().top);
  const pageTop = await page.locator('.docattentionlens-image-views').evaluate(e => e.getBoundingClientRect().top);
  expect(sheetTop - pageTop).toBeGreaterThan(80);
  expect(await noHorizontalScroll(page)).toBe(true);
});

test('Ctrl+K chooses a method and a layer; unavailable methods are listed with their reason', async ({page}) => {
  await page.setViewportSize({width: 1440, height: 900});
  await page.goto('/attention-tool?generation=' + meta.capture_id);
  await expect(page.getByTestId('attention-patch').first()).toBeVisible({timeout: 20_000});
  await expect(methodChoice(page)).toHaveAttribute('data-method-key', 'attention.row/captured_weights_v1');
  const trigger = page.getByRole('button', {name: 'Save investigation', exact: true});
  await trigger.focus();
  await page.keyboard.press('Control+k');
  const palette = page.getByRole('dialog', {name: 'Command palette'});
  await expect(palette).toBeVisible();
  const input = palette.getByRole('combobox', {name: 'Command'});
  await expect(input).toBeFocused();
  // An unavailable method stays listed, disabled, with its reason.
  await input.fill('rollout');
  const rollout = palette.getByRole('option', {name: /^Layer attention rollout/});
  await expect(rollout.first()).toHaveAttribute('aria-disabled', 'true');
  await input.fill('Mean-Head Attention');
  await expect(palette.getByRole('option').first()).toContainText('Mean-Head Attention');
  await expect(input).toHaveAttribute('aria-activedescendant', /.+/);
  await page.keyboard.press('Enter');
  await expect(palette).toHaveCount(0);
  await expect(methodChoice(page)).toHaveAttribute('data-method-key', 'attention.mean_heads/captured_prediction_v1');
  // Typed numbers find layers.
  await page.keyboard.press('Control+k');
  await input.fill('layer 3');
  await expect(palette.getByRole('option').first()).toHaveText(/^Layer 3/);
  await page.keyboard.press('Enter');
  await expect(page.getByLabel('Attention layer', {exact: true})).toHaveValue('3');
  // Escape closes and returns focus; the palette is also in the investigation menu.
  await trigger.focus();
  await page.keyboard.press('Control+k');
  await expect(palette).toBeVisible();
  await page.keyboard.press('Escape');
  await expect(palette).toHaveCount(0);
  await expect(trigger).toBeFocused();
  await page.getByRole('button', {name: 'Investigation menu', exact: true}).click();
  await page.getByRole('menuitem', {name: 'Command palette', exact: true}).click();
  await expect(palette).toBeVisible();
  // A Details tab is a command too.
  await input.fill('details display');
  await page.keyboard.press('Enter');
  await expect(detailsTab(page, 'Display')).toHaveAttribute('aria-selected', 'true');
});

test('the ? sheet lists the shortcuts; V cycles the display mode; keys are ignored while typing', async ({page}) => {
  await page.setViewportSize({width: 1440, height: 900});
  await page.goto('/attention-tool?generation=' + meta.capture_id);
  await expect(page.getByTestId('attention-patch').first()).toBeVisible({timeout: 20_000});
  await page.keyboard.press('?');
  const sheet = page.getByRole('dialog', {name: 'Keyboard shortcuts'});
  await expect(sheet).toBeVisible();
  for (const label of ['Previous or next layer', 'Previous or next head', 'Next display mode: Original image, Heatmap, Overlay, Blend', 'Image focus', 'Open or close Details', 'Browse methods', 'Save', 'This list of shortcuts'])
    await expect(sheet.getByText(label, {exact: true})).toBeVisible();
  // Shortcuts are off while a dialog is open.
  await page.keyboard.press('d');
  await expect(page.locator('#workspace-details')).toBeHidden();
  await page.keyboard.press('Escape');
  await expect(sheet).toHaveCount(0);
  await page.getByRole('button', {name: 'Investigation menu', exact: true}).click();
  await page.getByRole('menuitem', {name: 'Keyboard shortcuts', exact: true}).click();
  await expect(sheet).toBeVisible();
  await sheet.getByRole('button', {name: 'Close', exact: true}).click();
  // V: Overlay → Blend → Original image → Heatmap → Overlay.
  const settings = page.locator('.docattentionlens-canvas-legend').getByRole('button', {name: /^Display settings/});
  await expect(settings).toHaveAccessibleName(/^Display settings · Overlay/);
  for (const mode of ['Blend', 'Original image', 'Heatmap', 'Overlay']) {
    await page.keyboard.press('v');
    await expect(settings).toHaveAccessibleName(new RegExp(`^Display settings · ${mode}`));
  }
  // Typing in a field never triggers a shortcut; Details stays as it is.
  await openDetailsTab(page, 'Notes');
  await page.getByLabel('Notes', {exact: true}).fill('');
  await page.getByLabel('Notes', {exact: true}).pressSequentially('d]v');
  await expect(page.getByLabel('Notes', {exact: true})).toHaveValue('d]v');
  await expect(page.locator('#workspace-details')).toBeVisible();
  await expect(page.getByLabel('Attention layer', {exact: true})).toHaveValue('0');
  await expect(settings).toHaveAccessibleName(/^Display settings · Overlay/);
  // Outside a field, [ ] and D act.
  await page.locator('.docattentionlens-image-views').click({position: {x: 5, y: 5}});
  await page.keyboard.press(']');
  await expect(page.getByLabel('Attention layer', {exact: true})).toHaveValue('1');
  await page.keyboard.press('d');
  await expect(page.locator('#workspace-details')).toBeHidden();
});

test('the filmstrip appears only with more than one page and moves with the keyboard', async ({page}) => {
  await page.setViewportSize({width: 1440, height: 900});
  const single = await onePageCapture(page);
  if (single) {
    await page.goto('/attention-tool?generation=' + single.capture_id);
    await expect(page.getByTestId('attention-patch').first()).toBeVisible({timeout: 20_000});
    await expect(filmstrip(page)).toHaveCount(0);
    await page.setViewportSize({width: 375, height: 812});
    await expect(bottomBar(page).getByRole('button', {name: 'Pages', exact: true})).toHaveCount(0);
    await page.setViewportSize({width: 1440, height: 900});
  }
  const capture = await page.request.get('/api/v1/generation/captures/' + meta.capture_id).then(r => r.json());
  const order: string[] = [...capture.context.pages.map((p: {page_id: string}) => p.page_id), ...capture.context.omitted_page_ids];
  expect(order.length).toBeGreaterThan(1);
  await page.goto('/attention-tool?generation=' + meta.capture_id);
  await expect(page.getByTestId('attention-patch').first()).toBeVisible({timeout: 20_000});
  const tiles = filmstrip(page).locator('[data-page-id]');
  await expect(tiles).toHaveCount(order.length);
  expect(await tiles.evaluateAll(es => es.map(e => e.getAttribute('data-page-id')))).toEqual(order);
  // Supplied and omitted pages stay distinct.
  await expect(tiles.nth(0)).toContainText('Supplied');
  if (capture.context.omitted_page_ids.length) await expect(tiles.last()).toContainText('Omitted');
  expect(await filmstrip(page).evaluate(e => e.getBoundingClientRect().width)).toBeLessThanOrEqual(73);
  // Arrow keys move between pages; Enter chooses one.
  await tiles.nth(0).focus();
  await page.keyboard.press('ArrowDown');
  await expect(tiles.nth(1)).toBeFocused();
  await page.keyboard.press('Enter');
  await expect.poll(() => currentPage(page)).toBe(order[1]);
  await expect(tiles.nth(1)).toHaveAttribute('aria-current', 'page');
  // PageUp returns, in the same order.
  await page.locator('.docattentionlens-image-views').click({position: {x: 5, y: 5}});
  await page.keyboard.press('PageUp');
  await expect.poll(() => currentPage(page)).toBe(order[0]);
  // The filmstrip always shows each supplied page's attention mass; omitted pages are not measured.
  await expect(filmstrip(page).getByTestId('filmstrip-mass').first()).toHaveText(/^\d\.\d{6}$|e-/);
  if (capture.context.omitted_page_ids.length) await expect(tiles.last().getByTestId('filmstrip-mass')).toHaveText('not supplied');
});

test('the phone bottom bar opens Details, the method picker and the pages as sheets without overflow', async ({page}) => {
  await page.setViewportSize({width: 375, height: 812});
  await page.goto('/attention-tool?generation=' + meta.capture_id);
  await expect(page.getByTestId('attention-patch').first()).toBeVisible({timeout: 20_000});
  const bar = bottomBar(page);
  await expect(bar).toBeVisible();
  expect(await noHorizontalScroll(page)).toBe(true);
  const pageTop = await page.locator('.docattentionlens-image-views').evaluate(e => e.getBoundingClientRect().top);
  expect(pageTop).toBeLessThan(140);
  // The scrubber sits directly above the bottom bar.
  const scrubberBottom = await page.locator('.docattentionlens-scrubber-band').evaluate(e => e.getBoundingClientRect().bottom);
  const barTop = await bar.evaluate(e => e.getBoundingClientRect().top);
  expect(Math.abs(scrubberBottom - barTop)).toBeLessThan(2);
  // Details
  await bar.getByRole('button', {name: 'Details', exact: true}).click();
  const details = page.locator('#workspace-details');
  await expect(details).toBeVisible();
  expect(await details.evaluate(e => e.getBoundingClientRect().top)).toBeGreaterThan(pageTop + 80);
  await expect(page.getByLabel('Attention layer', {exact: true})).toBeInViewport();
  expect(await noHorizontalScroll(page)).toBe(true);
  await bar.getByRole('button', {name: 'Details', exact: true}).click();
  await expect(details).toBeHidden();
  // Pages (more than one page)
  await bar.getByRole('button', {name: 'Pages', exact: true}).click();
  const pages = page.getByRole('complementary', {name: 'Pages', exact: true});
  await expect(pages).toBeVisible();
  await expect(pages.getByTestId('filmstrip').locator('[data-page-id]').first()).toBeVisible();
  expect(await noHorizontalScroll(page)).toBe(true);
  const second = pages.getByTestId('filmstrip').locator('[data-page-id]').nth(1);
  const secondId = await second.getAttribute('data-page-id');
  await second.click();
  await expect(pages).toHaveCount(0);
  await expect.poll(() => currentPage(page)).toBe(secondId);
  // Method: Details › Method as a bottom sheet; the same button closes it.
  const method = bar.getByRole('button', {name: 'Method', exact: true});
  await method.click();
  const picker = page.getByTestId('method-browser');
  await expect(picker).toBeVisible();
  await expect(method).toHaveAttribute('aria-pressed', 'true');
  expect(await noHorizontalScroll(page)).toBe(true);
  await method.click();
  await expect(picker).toBeHidden();
  await page.screenshot({path: '../var/screenshots/workspace-phone-bottom-bar.png'});
});
