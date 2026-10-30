// Display geometry for page overlays: one transform maps source pixels to the visible image.

export type Point = { x: number; y: number };
export type Size = { width: number; height: number };
export type Box = { x1: number; y1: number; x2: number; y2: number };
export type Rect = { left: number; top: number; width: number; height: number };

/** display = source * scale + offset. Uniform scale keeps the page undistorted. */
export type Transform = { scale: number; offsetX: number; offsetY: number };

/** User view: zoom about the viewport centre, then pan in display pixels. */
export type View = { zoom: number; panX: number; panY: number };

export const MIN_ZOOM = 0.25;
export const MAX_ZOOM = 8;
export const IDENTITY_VIEW: View = { zoom: 1, panX: 0, panY: 0 };

function positive(size: Size) {
  return Number.isFinite(size.width) && Number.isFinite(size.height) && size.width > 0 && size.height > 0;
}

export function clampZoom(zoom: number) {
  return Math.min(MAX_ZOOM, Math.max(MIN_ZOOM, zoom));
}

/** Fit the whole page inside the viewport, centred, with letterboxing. Null when either size is unusable. */
export function fitContain(source: Size, viewport: Size): Transform | null {
  if (!positive(source) || !positive(viewport)) return null;
  const scale = Math.min(viewport.width / source.width, viewport.height / source.height);
  return {
    scale,
    offsetX: (viewport.width - source.width * scale) / 2,
    offsetY: (viewport.height - source.height * scale) / 2,
  };
}

/** Compose the fitted transform with a user view. */
export function viewTransform(fit: Transform, viewport: Size, view: View): Transform {
  const zoom = clampZoom(view.zoom);
  const cx = viewport.width / 2;
  const cy = viewport.height / 2;
  return {
    scale: fit.scale * zoom,
    offsetX: cx + (fit.offsetX - cx) * zoom + view.panX,
    offsetY: cy + (fit.offsetY - cy) * zoom + view.panY,
  };
}

export function toDisplay(t: Transform, p: Point): Point {
  return { x: p.x * t.scale + t.offsetX, y: p.y * t.scale + t.offsetY };
}

export function toSource(t: Transform, p: Point): Point {
  return { x: (p.x - t.offsetX) / t.scale, y: (p.y - t.offsetY) / t.scale };
}

/** Map a source box to a display rectangle. Corner order in the source is not assumed. */
export function boxToRect(t: Transform, box: Box): Rect {
  const a = toDisplay(t, { x: Math.min(box.x1, box.x2), y: Math.min(box.y1, box.y2) });
  const b = toDisplay(t, { x: Math.max(box.x1, box.x2), y: Math.max(box.y1, box.y2) });
  return { left: a.x, top: a.y, width: b.x - a.x, height: b.y - a.y };
}

/** The page image's own display rectangle. */
export function pageRect(t: Transform, source: Size): Rect {
  return boxToRect(t, { x1: 0, y1: 0, x2: source.width, y2: source.height });
}

/** Zoom by a factor while keeping the source point under `anchor` (display px) fixed. */
export function zoomAt(fit: Transform, viewport: Size, view: View, factor: number, anchor: Point): View {
  const before = viewTransform(fit, viewport, view);
  const sourcePoint = toSource(before, anchor);
  const zoom = clampZoom(view.zoom * factor);
  const unpanned = viewTransform(fit, viewport, { zoom, panX: 0, panY: 0 });
  const landed = toDisplay(unpanned, sourcePoint);
  return { zoom, panX: anchor.x - landed.x, panY: anchor.y - landed.y };
}

/**
 * The source point at the viewport centre. Saving this instead of display-pixel pan
 * reopens the same part of the page in a window of a different size.
 */
export function viewCentre(fit: Transform, viewport: Size, view: View): Point {
  return toSource(viewTransform(fit, viewport, view), { x: viewport.width / 2, y: viewport.height / 2 });
}

/** Rebuild a view that puts the saved source point at the viewport centre. */
export function viewFromCentre(fit: Transform, viewport: Size, zoom: number, centre: Point): View {
  const clamped = clampZoom(zoom);
  const landed = toDisplay(viewTransform(fit, viewport, { zoom: clamped, panX: 0, panY: 0 }), centre);
  return { zoom: clamped, panX: viewport.width / 2 - landed.x, panY: viewport.height / 2 - landed.y };
}

export type Hit<T> = { item: T; box: Box };

/**
 * Elements whose box contains the display point, smallest area first, so the most
 * specific element wins and overlapping boxes can be cycled. Ties keep input order.
 */
export function hitTest<T>(t: Transform, point: Point, candidates: readonly Hit<T>[]): T[] {
  const p = toSource(t, point);
  return candidates
    .map((candidate, index) => {
      const { box } = candidate;
      const x1 = Math.min(box.x1, box.x2), x2 = Math.max(box.x1, box.x2);
      const y1 = Math.min(box.y1, box.y2), y2 = Math.max(box.y1, box.y2);
      const inside = p.x >= x1 && p.x <= x2 && p.y >= y1 && p.y <= y2;
      return { candidate, index, inside, area: (x2 - x1) * (y2 - y1) };
    })
    .filter((entry) => entry.inside)
    .sort((a, b) => a.area - b.area || a.index - b.index)
    .map((entry) => entry.candidate.item);
}

/** Display label only; spreads are not inferred from proportions. */
export function orientation(size: Size): 'portrait' | 'landscape' | 'square' {
  if (size.width === size.height) return 'square';
  return size.width > size.height ? 'landscape' : 'portrait';
}
