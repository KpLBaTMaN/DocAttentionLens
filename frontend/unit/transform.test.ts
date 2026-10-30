import assert from 'node:assert/strict';
import { test } from 'vitest';
import {
  boxToRect, fitContain, hitTest, orientation, pageRect, toDisplay, toSource, viewCentre, viewFromCentre, viewTransform, zoomAt,
  IDENTITY_VIEW, MAX_ZOOM, type Point, type Size, type Transform,
} from '../src/geometry/transform.ts';

const close = (actual: number, expected: number, message?: string) =>
  assert.ok(Math.abs(actual - expected) < 1e-9, `${message ?? ''} expected ${expected}, got ${actual}`);
const closePoint = (a: Point, b: Point) => { close(a.x, b.x, 'x'); close(a.y, b.y, 'y'); };

// Page sizes chosen to cover portrait, landscape and spread-like proportions.
const portrait: Size = { width: 1654, height: 2339 };
const landscape: Size = { width: 2339, height: 1654 };
const spread: Size = { width: 3308, height: 2339 };
const viewport: Size = { width: 900, height: 700 };

test('fit letterboxes and centres every page orientation', () => {
  for (const page of [portrait, landscape, spread]) {
    const t = fitContain(page, viewport)!;
    const r = pageRect(t, page);
    assert.ok(r.width <= viewport.width + 1e-9 && r.height <= viewport.height + 1e-9);
    assert.ok(Math.abs(r.width - viewport.width) < 1e-9 || Math.abs(r.height - viewport.height) < 1e-9, 'touches one edge');
    close(r.left * 2 + r.width, viewport.width, 'centred x');
    close(r.top * 2 + r.height, viewport.height, 'centred y');
    close(r.width / r.height, page.width / page.height, 'aspect');
  }
});

test('fit rejects unusable sizes', () => {
  assert.equal(fitContain({ width: 0, height: 10 }, viewport), null);
  assert.equal(fitContain(portrait, { width: 10, height: Number.NaN }), null);
});

test('toSource inverts toDisplay under zoom and pan', () => {
  const fit = fitContain(landscape, viewport)!;
  const t = viewTransform(fit, viewport, { zoom: 3.7, panX: -120, panY: 45 });
  for (const p of [{ x: 0, y: 0 }, { x: 2339, y: 1654 }, { x: 801.5, y: 17 }, { x: -40, y: 1700 }]) {
    closePoint(toSource(t, toDisplay(t, p)), p);
  }
});

test('boxes stay aligned with the page image after resize', () => {
  // Same source box, two viewport sizes: its position relative to the page rectangle is unchanged.
  const box = { x1: 200, y1: 300, x2: 900, y2: 600 };
  const relative = (t: Transform) => {
    const page = pageRect(t, portrait);
    const r = boxToRect(t, box);
    return [(r.left - page.left) / page.width, (r.top - page.top) / page.height, r.width / page.width, r.height / page.height];
  };
  const a = relative(fitContain(portrait, viewport)!);
  const b = relative(viewTransform(fitContain(portrait, { width: 390, height: 640 })!, { width: 390, height: 640 }, { zoom: 2, panX: 30, panY: -10 }));
  a.forEach((value, i) => close(value, b[i]));
  close(a[0], 200 / portrait.width);
});

test('boxToRect does not assume corner order', () => {
  const t: Transform = { scale: 2, offsetX: 10, offsetY: 20 };
  assert.deepEqual(boxToRect(t, { x1: 50, y1: 40, x2: 10, y2: 5 }), { left: 30, top: 30, width: 80, height: 70 });
});

test('zoomAt keeps the anchored source point under the cursor', () => {
  const fit = fitContain(portrait, viewport)!;
  const anchor = { x: 612, y: 233 };
  const before = toSource(viewTransform(fit, viewport, IDENTITY_VIEW), anchor);
  const view = zoomAt(fit, viewport, IDENTITY_VIEW, 2.5, anchor);
  close(view.zoom, 2.5);
  closePoint(toSource(viewTransform(fit, viewport, view), anchor), before);
});

test('zoom is clamped to the saved ViewState range', () => {
  const fit = fitContain(portrait, viewport)!;
  const view = zoomAt(fit, viewport, { zoom: 6, panX: 0, panY: 0 }, 4, { x: 450, y: 350 });
  assert.equal(view.zoom, MAX_ZOOM);
});

test('a saved centre reopens the same page region in a different window size', () => {
  const desktop = { width: 1200, height: 800 };
  const phone = { width: 390, height: 640 };
  const view = { zoom: 3, panX: -210, panY: 95 };
  const centre = viewCentre(fitContain(spread, desktop)!, desktop, view);
  const phoneFit = fitContain(spread, phone)!;
  const reopened = viewFromCentre(phoneFit, phone, view.zoom, centre);
  closePoint(viewCentre(phoneFit, phone, reopened), centre);
  close(reopened.zoom, 3);
});

test('hitTest prefers the smallest containing box and keeps order on ties', () => {
  const t: Transform = { scale: 0.5, offsetX: 100, offsetY: 0 };
  const candidates = [
    { item: 'section', box: { x1: 0, y1: 0, x2: 1000, y2: 1000 } },
    { item: 'paragraph', box: { x1: 100, y1: 100, x2: 400, y2: 300 } },
    { item: 'twin-a', box: { x1: 500, y1: 500, x2: 600, y2: 600 } },
    { item: 'twin-b', box: { x1: 600, y1: 600, x2: 500, y2: 500 } },
  ];
  assert.deepEqual(hitTest(t, toDisplay(t, { x: 200, y: 200 }), candidates), ['paragraph', 'section']);
  assert.deepEqual(hitTest(t, toDisplay(t, { x: 550, y: 550 }), candidates), ['twin-a', 'twin-b', 'section']);
  assert.deepEqual(hitTest(t, toDisplay(t, { x: 1200, y: 10 }), candidates), []);
});

test('orientation is a label from declared size only', () => {
  assert.equal(orientation(portrait), 'portrait');
  assert.equal(orientation(landscape), 'landscape');
  assert.equal(orientation(spread), 'landscape');
  assert.equal(orientation({ width: 5, height: 5 }), 'square');
});
