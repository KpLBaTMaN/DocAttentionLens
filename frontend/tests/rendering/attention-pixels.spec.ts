import { test, expect, type Locator, type Page } from '@playwright/test';
import { inflateSync } from 'node:zlib';
import * as F from '../../fixtures/attention';
import { mockAttentionApi } from './mockApi';
import { chooseColourMode, closeDisplaySettings, colourMode, displaySettingsButton, openDisplayOptions, withDisplaySettings } from '../workspaceHelpers';
import { choosePerspective, setFit } from '../workspaceHelpers';
import { heatColor } from '../../src/features/document/attentionDisplay';

// What the researcher actually sees. The page image is plain white and only patch 6
// (row 1, column 2) carries image attention, so every expected screen colour is known:
// the hot patch shows the display colour at full strength and patch 9 shows the page.
// These read real screen pixels, so CSS, layering and image loading are all included.

const HOT_PATCH = 6, COLD_PATCH = 9;
const TOLERANCE = 3;

/** Decode the single pixel of a 1 × 1 PNG screenshot (any scanline filter leaves one pixel unchanged). */
function onePixel(png: Buffer): number[] {
  let offset = 8, colourType = 0;
  const data: Buffer[] = [];
  while (offset < png.length) {
    const length = png.readUInt32BE(offset), type = png.toString('ascii', offset + 4, offset + 8);
    const body = png.subarray(offset + 8, offset + 8 + length);
    if (type === 'IHDR') colourType = body[9];
    if (type === 'IDAT') data.push(body);
    offset += length + 12;
  }
  const raw = inflateSync(Buffer.concat(data));
  return [...raw.subarray(1, colourType === 6 ? 5 : 4)].slice(0, 3);
}

/** The original page image; the smoothing raster is a second image above it. */
const pageImage = (page: Page) => page.getByTestId('page-canvas').locator('image').first();

/** Screen point of a source-pixel point of the page, located through the page image's own on-screen box. */
async function screenAt(page: Page, source: {x: number; y: number}) {
  const box = (await pageImage(page).boundingBox())!;
  return {x: Math.round(box.x + source.x / F.PAGE.width * box.width), y: Math.round(box.y + source.y / F.PAGE.height * box.height)};
}

/** Screen colour at a source-pixel point of the page. The instruments floating over the canvas are hidden while a pixel is read. */
async function colourAt(page: Page, source: {x: number; y: number}) {
  const {x, y} = await screenAt(page, source);
  const style = await page.addStyleTag({content: '.docattentionlens-canvas-instruments {visibility: hidden !important;}'});
  try { return onePixel(await page.screenshot({clip: {x, y, width: 1, height: 1}})); }
  finally { await style.evaluate(e => e.parentNode?.removeChild(e)); }
}

function expectColour(actual: number[], expected: readonly number[], tolerance = TOLERANCE) {
  const off = Math.max(...actual.map((c, i) => Math.abs(c - expected[i])));
  expect(off, `screen colour ${actual} should be ${expected.map(Math.round)} ± ${tolerance}`).toBeLessThanOrEqual(tolerance);
}

/** The hot patch's on-screen rectangle must be exactly its share of the page image's rectangle. */
async function expectAligned(page: Page, patch: Locator, k: number) {
  const image = (await pageImage(page).boundingBox())!;
  const drawn = (await patch.boundingBox())!;
  const [x1, y1, x2, y2] = F.patchBox(k);
  const expected = {x: image.x + x1 / F.PAGE.width * image.width, y: image.y + y1 / F.PAGE.height * image.height,
    width: (x2 - x1) / F.PAGE.width * image.width, height: (y2 - y1) / F.PAGE.height * image.height};
  for (const key of ['x', 'y', 'width', 'height'] as const) expect(Math.abs(drawn[key] - expected[key]), `${key} of patch ${k}`).toBeLessThanOrEqual(.5);
}

async function open(page: Page) {
  // One-hot row: only HOT_PATCH has image weight, every other patch is exactly zero.
  const api = await mockAttentionApi(page, {image: () => F.patchValues(HOT_PATCH, false)});
  await page.setViewportSize({width: 1440, height: 900});
  await page.goto(`/attention-tool?generation=${F.GENERATION_ID}`);
  await expect(page.getByTestId('attention-patch')).toHaveCount(F.PATCH_COUNT);
  return api;
}

const toolbar = (page: Page) => page.getByRole('group', {name: 'Colour legend', exact: true});
/** Page snapshots record the page rendering, not the instruments that float over its corners. */
const CANVAS_ONLY = (page: Page) => ({mask: [page.locator('.docattentionlens-canvas-instruments .docattentionlens-instrument'), page.locator('.docattentionlens-canvas-notices p')]});
const hotPatch = (page: Page) => page.locator(`[data-testid="attention-patch"][data-position-index="${F.generatorPatchIndex(HOT_PATCH)}"]`);
const WHITE = [255, 255, 255];
const AMBER_65 = F.overWhite(F.OVERLAY_RGB, .65);

test('overlay: the hot patch is amber over the page and a zero patch shows the white page', async ({page}) => {
  const api = await open(page);
  await expectAligned(page, hotPatch(page), HOT_PATCH);
  expectColour(await colourAt(page, F.patchCentre(HOT_PATCH)), AMBER_65);
  expectColour(await colourAt(page, F.patchCentre(COLD_PATCH)), WHITE);
  await expect(page.getByTestId('page-canvas')).toHaveScreenshot('overlay.png', CANVAS_ONLY(page));
  expect(api.unexpected).toEqual([]);
});

test('heatmap: the hot patch is the brightest inferno colour and a zero patch the darkest', async ({page}) => {
  await open(page);
  await chooseColourMode(page, 'Heatmap');
  expectColour(await colourAt(page, F.patchCentre(HOT_PATCH)), F.inferno(1));
  expectColour(await colourAt(page, F.patchCentre(COLD_PATCH)), F.inferno(0));
  await expect(page.getByTestId('page-canvas')).toHaveScreenshot('heatmap.png', CANVAS_ONLY(page));
});

test('smoothed heatmap keeps each patch centre at its own colour', async ({page}) => {
  await open(page);
  await withDisplaySettings(page, async drawer => {
    await colourMode(drawer, 'Heatmap').click();
    await drawer.getByLabel('Smooth', {exact: true}).check();
  });
  await expect(page.getByTestId('attention-raster')).toHaveCount(1);
  // Interpolation blends neighbours between centres, so allow a little more at the centre itself.
  expectColour(await colourAt(page, F.patchCentre(HOT_PATCH)), F.inferno(1), 12);
  expectColour(await colourAt(page, F.patchCentre(COLD_PATCH)), F.inferno(0), 12);
  await expect(page.getByTestId('page-canvas')).toHaveScreenshot('heatmap-smoothed.png', CANVAS_ONLY(page));
});

test('blend: the colour map covers the whole page at one opacity, in the chosen map', async ({page}) => {
  await open(page);
  await chooseColourMode(page, 'Blend');
  // Unlike Overlay, a zero patch is not transparent: it takes the map's darkest colour.
  expectColour(await colourAt(page, F.patchCentre(HOT_PATCH)), F.overWhite(F.inferno(1), .65));
  expectColour(await colourAt(page, F.patchCentre(COLD_PATCH)), F.overWhite(F.inferno(0), .65));
  await withDisplaySettings(page, drawer => drawer.getByRole('radio', {name: 'Viridis'}).click());
  expectColour(await colourAt(page, F.patchCentre(HOT_PATCH)), F.overWhite(heatColor(1, 'viridis'), .65));
  expectColour(await colourAt(page, F.patchCentre(COLD_PATCH)), F.overWhite(heatColor(0, 'viridis'), .65));
});

test('the page key sits in the colour bar and reads the patch or element under the pointer', async ({page}) => {
  await open(page);
  const key = page.getByTestId('page-key'), readout = page.getByTestId('page-key-readout');
  // It lives in the legend on the canvas's bottom-left corner beside the one Display settings button.
  await expect(toolbar(page).getByTestId('page-key')).toBeVisible();
  await expect(displaySettingsButton(page)).toHaveAccessibleName('Display settings · Overlay · Image contrast');
  const canvas = (await page.getByTestId('page-canvas').boundingBox())!, drawn = (await key.boundingBox())!;
  expect(drawn.y).toBeGreaterThan(canvas.y + canvas.height / 2);
  expect(drawn.y + drawn.height).toBeLessThanOrEqual(canvas.y + canvas.height);
  await expect(page.getByTestId('page-key-floor')).toHaveText('0');
  await expect(readout).toHaveText('Hover the page');
  const hot = await screenAt(page, F.patchCentre(HOT_PATCH)), cold = await screenAt(page, F.patchCentre(COLD_PATCH));
  await page.mouse.move(hot.x, hot.y);
  await expect(readout).toHaveText(/^100% · /);
  await expect(page.getByTestId('page-key-marker')).toHaveAttribute('style', /left: 100%/);
  await page.mouse.move(cold.x, cold.y);
  await expect(readout).toHaveText('0% · 0.000000');
  await expect(page.getByTestId('page-key-marker')).toHaveAttribute('style', /left: 0%/);
  // Element fills carry the colour in Document-aware, so the element under the pointer is read.
  await choosePerspective(page, 'Document-aware');
  await expect(page.getByTestId('element-region')).toHaveCount(2);
  const text = F.ELEMENTS[1].box, inText = await screenAt(page, {x: (text[0] + text[2]) / 2 + 10, y: (text[1] + text[3]) / 2 + 10});
  await page.mouse.move(inText.x, inText.y);
  await expect(readout).toHaveText(/^100% · /);
  await expect(readout).toHaveAttribute('title', /^text /i);
  await chooseColourMode(page, 'Original image');
  await expect(key).toHaveCount(0);
});

test('Original shows the untouched page', async ({page}) => {
  await open(page);
  await chooseColourMode(page, 'Original image');
  await expect(page.getByTestId('attention-patch')).toHaveCount(0);
  expectColour(await colourAt(page, F.patchCentre(HOT_PATCH)), WHITE);
});

test('document-aware regions fill the element that owns the hot patch', async ({page}) => {
  await open(page);
  await choosePerspective(page, 'Document-aware');
  await expect(page.getByTestId('element-region')).toHaveCount(2);
  // The text element owns the hot patch and so holds the whole element maximum; the table holds none.
  const text = F.ELEMENTS[1], table = F.ELEMENTS[0];
  const inside = (box: F.Box) => ({x: (box[0] + box[2]) / 2 + 10, y: (box[1] + box[3]) / 2 + 10});
  expectColour(await colourAt(page, inside(text.box)), AMBER_65);
  expectColour(await colourAt(page, inside(table.box)), WHITE);
  await chooseColourMode(page, 'Heatmap');
  expectColour(await colourAt(page, inside(text.box)), F.overWhite(F.inferno(1), .75));
  expectColour(await colourAt(page, inside(table.box)), F.overWhite(F.inferno(0), .75));
  await expect(page.getByTestId('page-canvas')).toHaveScreenshot('regions-heatmap.png', CANVAS_ONLY(page));
});

// Colour maps on screen. Each map's darkest and brightest colours are copied from matplotlib here,
// independently of attentionDisplay.ts, so a choice that does not reach the page (or falls back to
// inferno) fails. Overlay is amber by design and must not change with the map.
const MAP_ENDS: Record<string, {low: number[]; high: number[]}> = {
  Inferno: {low: [0, 0, 4], high: [252, 255, 164]}, Magma: {low: [0, 0, 4], high: [252, 253, 191]},
  Viridis: {low: [68, 1, 84], high: [253, 231, 37]}, Plasma: {low: [13, 8, 135], high: [240, 249, 33]},
};
/** Inferno last, so each loop also checks that choosing the default again restores it. */
const MAP_ORDER = ['Magma', 'Viridis', 'Plasma', 'Inferno'];
const mapChoice = (drawer: Locator, name: string) => drawer.getByRole('radiogroup', {name: 'Colour map', exact: true}).getByRole('radio', {name, exact: true});
const pageKeyRamp = (page: Page) => page.getByTestId('page-key').locator('.docattentionlens-page-key-ramp');

for (const [mode, alpha] of [['Heatmap', 1], ['Blend', .65]] as const) test(`${mode.toLowerCase()}: every colour map recolours the page and its key as soon as it is chosen`, async ({page}) => {
  const api = await open(page);
  const drawer = await openDisplayOptions(page);
  await colourMode(drawer, mode).click();
  for (const name of MAP_ORDER) {
    await mapChoice(drawer, name).click();
    await expect(mapChoice(drawer, name)).toHaveAttribute('aria-checked', 'true');
    await expect(page.getByTestId('attention-layer')).toHaveAttribute('data-colormap', name.toLowerCase());
    const {low, high} = MAP_ENDS[name];
    expectColour(await colourAt(page, F.patchCentre(HOT_PATCH)), F.overWhite(high, alpha));
    expectColour(await colourAt(page, F.patchCentre(COLD_PATCH)), F.overWhite(low, alpha));
    // The key under the page draws the same map, ending in its brightest colour.
    await expect.poll(() => pageKeyRamp(page).getAttribute('style')).toContain(`rgb(${high.join(', ')}) 100%`);
  }
  await closeDisplaySettings(page);
  expect(api.unexpected).toEqual([]);
});

test('overlay stays amber whichever colour map is chosen, and the map applies once Heatmap is chosen', async ({page}) => {
  await open(page);
  const drawer = await openDisplayOptions(page);
  await expect(colourMode(drawer, 'Overlay')).toHaveAttribute('aria-checked', 'true');
  await expect(drawer.getByText('Overlay stays amber', {exact: true})).toBeVisible();
  for (const name of MAP_ORDER) {
    await mapChoice(drawer, name).click();
    await expect(page.getByTestId('attention-layer')).not.toHaveAttribute('data-colormap', /./);
    expectColour(await colourAt(page, F.patchCentre(HOT_PATCH)), AMBER_65);
    expectColour(await colourAt(page, F.patchCentre(COLD_PATCH)), WHITE);
  }
  // The map chosen while in Overlay is kept and shows as soon as the mode uses it.
  await mapChoice(drawer, 'Viridis').click();
  await colourMode(drawer, 'Heatmap').click();
  expectColour(await colourAt(page, F.patchCentre(HOT_PATCH)), MAP_ENDS.Viridis.high);
  await colourMode(drawer, 'Overlay').click();
  expectColour(await colourAt(page, F.patchCentre(HOT_PATCH)), AMBER_65);
  await colourMode(drawer, 'Blend').click();
  expectColour(await colourAt(page, F.patchCentre(HOT_PATCH)), F.overWhite(MAP_ENDS.Viridis.high, .65));
  await closeDisplaySettings(page);
});

test('element fills take the chosen colour map in Heatmap', async ({page}) => {
  await open(page);
  await choosePerspective(page, 'Document-aware');
  await expect(page.getByTestId('element-region')).toHaveCount(2);
  const text = F.ELEMENTS[1].box, inside = {x: (text[0] + text[2]) / 2 + 10, y: (text[1] + text[3]) / 2 + 10};
  const drawer = await openDisplayOptions(page);
  await colourMode(drawer, 'Heatmap').click();
  for (const name of MAP_ORDER) {
    await mapChoice(drawer, name).click();
    // The text element holds the element maximum, so it takes the map's brightest colour.
    expectColour(await colourAt(page, inside), F.overWhite(MAP_ENDS[name].high, .75));
  }
  await closeDisplaySettings(page);
});

test('zoom, pan and window resizing keep the overlay on its patch', async ({page}) => {
  await open(page);
  // This checks overlay alignment under zoom and resize. The view controls float over the page's
  // top-right corner and can cover the hot patch when zoomed, so they are hidden while a pixel is read.
  const withoutControls = async <T>(read: () => Promise<T>) => {
    const style = await page.addStyleTag({content: '.docattentionlens-canvas-instruments {visibility: hidden !important;}'});
    try { return await read(); } finally { await style.evaluate(e => e.parentNode?.removeChild(e)); }
  };
  for (const step of ['zoom in', 'zoom in again', 'narrow window', 'wide window', 'fit width']) {
    await test.step(step, async () => {
      if (step.startsWith('zoom')) await page.getByRole('button', {name: 'Zoom in', exact: true}).click();
      if (step === 'narrow window') await page.setViewportSize({width: 1024, height: 768});
      if (step === 'wide window') await page.setViewportSize({width: 1680, height: 1000});
      if (step === 'fit width') await setFit(page, 'width');
      await hotPatch(page).scrollIntoViewIfNeeded();
      await expect.poll(async () => {
        try { await expectAligned(page, hotPatch(page), HOT_PATCH); return true; } catch { return false; }
      }, {message: `hot patch aligned after ${step}`}).toBe(true);
      expectColour(await withoutControls(() => colourAt(page, F.patchCentre(HOT_PATCH))), AMBER_65);
    });
  }
  // Drag-pan: the page and its overlay move together.
  const canvas = (await page.getByTestId('page-canvas').boundingBox())!;
  await page.mouse.move(canvas.x + canvas.width / 2, canvas.y + canvas.height / 2);
  await page.mouse.down();
  await page.mouse.move(canvas.x + canvas.width / 2 - 60, canvas.y + canvas.height / 2 - 40, {steps: 5});
  await page.mouse.up();
  await expectAligned(page, hotPatch(page), HOT_PATCH);
});
