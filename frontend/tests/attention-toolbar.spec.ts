import { chooseColourMode, closeDisplaySettings, colourMode, displaySettingsButton, expectRowTotals, openDisplayOptions, readRowTotals, withDisplaySettings } from './workspaceHelpers';
import { test, expect } from '@playwright/test';

test('the legend keeps the colour key and one Display button; Details › Display changes only how saved weights are drawn', async ({page}) => {
  const capture = (await page.request.get('/api/v1/generation/captures').then(r => r.json())).items[0];
  expect(capture, 'Requires a real saved attention capture').toBeTruthy();
  let acquisitions = 0;
  await page.route(/\/api\/v1\/(generation|retrieval)\/runs$/, route => {acquisitions++; return route.abort();});
  await page.setViewportSize({width:1440, height:900});
  await page.goto('/attention-tool?generation=' + capture.capture_id);
  const toolbar = page.getByRole('group', {name:'Colour legend', exact:true});
  const settings = displaySettingsButton(page);
  const patch = page.getByTestId('attention-patch').first();
  await expect(patch).toBeVisible({timeout: 20_000});
  await expect(settings).toBeEnabled();
  // The legend on the page holds the colour key and one button naming the mode (and, for screen readers, the scale); every choice is in Details › Display.
  await expect(page.getByRole('complementary', {name:'Attention panel', exact:true}).getByLabel('Attention opacity', {exact:true})).toHaveCount(0);
  await expect(toolbar).toBeVisible();
  await expect(settings).toHaveCount(1);
  await expect(settings).toHaveAccessibleName('Display settings · Overlay · Image contrast');
  await expect(settings).toHaveText('Overlay');
  await expect(settings).toHaveAttribute('aria-expanded', 'false');
  for (const name of ['Attention display mode', 'Page attention values']) await expect(page.getByRole('radiogroup', {name, exact:true})).toHaveCount(0);
  for (const name of ['Attention opacity', 'Heatmap scale', 'Fixed scale maximum']) await expect(page.getByLabel(name, {exact:true})).toHaveCount(0);
  await expect(page.getByTestId('display-scale')).toBeVisible();
  // The legend floats on the canvas's bottom-left corner.
  const bar = (await toolbar.boundingBox())!, canvas = (await page.getByTestId('page-canvas').first().boundingBox())!;
  expect(bar.y + bar.height).toBeLessThanOrEqual(canvas.y + canvas.height + 1);
  expect(bar.y).toBeGreaterThan(canvas.y + canvas.height / 2);
  expect(bar.x).toBeLessThan(canvas.x + canvas.width / 2);
  const mass = await readRowTotals(page);

  const options = await openDisplayOptions(page);
  await expect(options).toBeVisible();
  // Details › Display sits beside the page, covering none of it, and does not dim it (no modal dialog or backdrop).
  await expect(page.getByRole('tab', {name: 'Display', exact: true})).toHaveAttribute('aria-selected', 'true');
  const drawerBox = (await options.boundingBox())!, pageBox = (await page.getByTestId('page-canvas').first().boundingBox())!;
  expect(drawerBox.x).toBeGreaterThanOrEqual(pageBox.x + pageBox.width - 1);
  await expect(page.locator('dialog[open]')).toHaveCount(0);
  await expect(settings).toHaveAttribute('aria-expanded', 'true');
  await expect(colourMode(options, 'Overlay')).toHaveAttribute('aria-checked', 'true');
  await expect(options.getByLabel('Attention opacity', {exact:true})).toBeVisible();
  await expect(options.getByLabel('Heatmap scale', {exact:true})).toHaveValue('auto');
  await expect(patch).toHaveAttribute('fill', '#d99a36');
  await colourMode(options, 'Heatmap').click();
  await expect(patch).toHaveAttribute('fill', /^rgb\(/);
  await expect(patch).toHaveAttribute('fill-opacity', '1');
  await expect(settings).toHaveAccessibleName('Display settings · Heatmap · Image contrast');
  // Opacity stays in the drawer for every mode.
  await expect(options.getByLabel('Attention opacity', {exact:true})).toBeVisible();
  await options.getByLabel('Heatmap scale', {exact:true}).selectOption('fixed');
  await expect(options.getByLabel('Fixed scale maximum', {exact:true})).toBeVisible();
  await expect(settings).toHaveAccessibleName('Display settings · Heatmap · Fixed');
  await options.getByLabel('Heatmap scale', {exact:true}).selectOption('auto');
  await expect(options.getByLabel('Fixed scale maximum', {exact:true})).toHaveCount(0);
  await options.getByLabel('Smooth', {exact:true}).check();
  await expect(page.getByTestId('attention-raster')).toHaveCount(1);
  await expect(patch).toHaveAttribute('fill-opacity', '0');
  await options.getByLabel('Patch grid', {exact:true}).check();
  await expect(page.getByTestId('attention-patch-grid')).toHaveCount(1);
  await page.screenshot({path:'../var/screenshots/attention-toolbar-heatmap.png'});
  // The legend button toggles Details on Display.
  await settings.click();
  await expect(options).toBeHidden();
  await expect(settings).toHaveAttribute('aria-expanded', 'false');
  await settings.click();
  await expect(options).toBeVisible();
  await closeDisplaySettings(page);
  await chooseColourMode(page, 'Original image');
  await expect(settings).toHaveAccessibleName('Display settings · Original image · Image contrast');
  await expect(page.getByTestId('attention-patch')).toHaveCount(0);
  await expect(page.getByTestId('attention-raster')).toHaveCount(0);
  await expect(page.getByTestId('attention-patch-grid')).toHaveCount(1);
  await expect(page.getByText('Original view: attention is hidden.', {exact:false})).toBeVisible();
  await chooseColourMode(page, 'Heatmap');
  await expectRowTotals(page, mass);
  // The colour map recolours the heatmap at once and is saved with the investigation.
  const before = await patch.getAttribute('fill');
  await withDisplaySettings(page, drawer => drawer.getByRole('radiogroup', {name: 'Colour map', exact: true}).getByRole('radio', {name: 'Viridis', exact: true}).click());
  await expect(page.getByTestId('attention-layer').first()).toHaveAttribute('data-colormap', 'viridis');
  await expect(patch).not.toHaveAttribute('fill', before!);
  await expectRowTotals(page, mass);

  await page.getByRole('button', {name:'Save investigation', exact:true}).click();
  await expect(page).toHaveURL(/investigation_/);
  await page.reload();
  await expect(settings).toHaveAccessibleName('Display settings · Heatmap · Image contrast');
  const reopened = await openDisplayOptions(page);
  await expect(colourMode(reopened, 'Heatmap')).toHaveAttribute('aria-checked', 'true');
  await expect(reopened.getByRole('radiogroup', {name: 'Colour map', exact: true}).getByRole('radio', {name: 'Viridis', exact: true})).toHaveAttribute('aria-checked', 'true');
  await expect(page.getByTestId('attention-layer').first()).toHaveAttribute('data-colormap', 'viridis');
  await expect(reopened.getByLabel('Smooth', {exact:true})).toBeChecked();
  await expect(reopened.getByLabel('Patch grid', {exact:true})).toBeChecked();
  await closeDisplaySettings(page);
  const id = new URL(page.url()).pathname.split('/').at(-1)!;
  const saved = await page.request.get('/api/v1/investigations/' + id).then(r => r.json());
  expect(saved.display_settings).toMatchObject({attention_display_mode:'heatmap', attention_colormap:'viridis', attention_smooth:true, attention_show_patches:true});
  await page.setViewportSize({width:390, height:844});
  await expect(toolbar).toBeVisible();
  await expect.poll(() => page.evaluate(() => document.documentElement.scrollWidth <= innerWidth + 1)).toBe(true);
  expect(acquisitions).toBe(0);
});
