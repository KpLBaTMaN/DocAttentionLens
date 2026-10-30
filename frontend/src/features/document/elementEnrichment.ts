import type { Attention } from './GenerationPanel';
import type { SourceReach } from './ElementSummary';
import { isImagePosition } from './massBreakdown';

type Position = {index: number; role?: string; bbox?: number[] | null; page_id?: string | null; ownership?: Record<string, number> | null};

/** An element that owns less than this many image positions' worth of area has an unstable Enrichment. */
export const MIN_POSITIONS = 1;

export type Enrichment = {
  /** Share of the row's image attention ÷ share of image positions; null when either denominator is zero. */
  value: number | null;
  imageShare: number;
  areaShare: number;
  /** Image positions' worth of area the bucket owns: Σ ownership over the measured image positions. */
  positions: number;
  /** The largest value the bucket can reach, with all image attention on it: 1 ÷ area share. */
  ceiling: number | null;
  /** Owns less than one image position, so the ratio rests on a fragment of one position. */
  belowOnePosition: boolean;
};

/** Enrichment of every ownership bucket on one page: image share over area share (E_b = p_b / s_b). */
export function pageEnrichment(row: Attention, positions: Position[], pageId: string, reach?: SourceReach): Map<string, Enrichment> {
  const owned = new Map<string, number>();
  let count = 0;
  for (const p of positions) {
    if (!isImagePosition(p) || p.index >= row.target_count || !(reach?.reaches(p.index) ?? true)) continue;
    count++;
    if (p.page_id === pageId) for (const [bucket, w] of Object.entries(p.ownership ?? {})) owned.set(bucket, (owned.get(bucket) ?? 0) + w);
  }
  const masses = new Map<string, number>();
  for (const a of row.allocations) if (a.page_id === pageId) masses.set(a.bucket, (masses.get(a.bucket) ?? 0) + a.mass);
  const result = new Map<string, Enrichment>();
  for (const [bucket, area] of owned) {
    if (!(area > 0)) continue;
    const areaShare = area / count, imageShare = row.image_mass > 0 ? (masses.get(bucket) ?? 0) / row.image_mass : 0;
    result.set(bucket, {value: row.image_mass > 0 ? imageShare / areaShare : null, imageShare, areaShare, positions: area,
      ceiling: 1 / areaShare, belowOnePosition: area < MIN_POSITIONS});
  }
  return result;
}

/** Where a value sits on the rail's bar, in percent: log₂ scale from ¼× (0%) through 1× (50%) to 4× (100%), clamped. */
export function enrichmentPosition(value: number) {
  return value > 0 ? 50 + 50 * Math.max(-1, Math.min(1, Math.log2(value) / 2)) : 0;
}
