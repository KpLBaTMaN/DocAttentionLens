import { describe, it, expect } from 'vitest';
import { attentionAlpha, attentionMode, attentionWeightLabel, heatColor, imageAttentionMaximum, patchGrid, roundedTimesLabel, roundedWeightLabel, sharePercent } from '../src/features/document/attentionDisplay';
import { scaleAgreement } from '../src/features/document/ScaleKey';
import { columnValues } from '../src/features/document/useAttentionColumns';
import type { DisplaySettings } from '../src/api/client';
import type { Attention, Generation } from '../src/features/document/GenerationPanel';

// WCAG relative luminance: a sequential colour scale must get monotonically lighter.
const luminance = ([r, g, b]: number[]) => {
  const f = (v: number) => {v /= 255; return v <= .03928 ? v / 12.92 : ((v + .055) / 1.055) ** 2.4;};
  return .2126 * f(r) + .7152 * f(g) + .0722 * f(b);
};

describe('attention colour and intensity mapping', () => {
  it('heatmap colours are a monotonic sequential scale clamped to its ends', () => {
    const ramp = Array.from({length: 201}, (_, i) => luminance(heatColor(i / 200)));
    ramp.slice(1).forEach((value, i) => expect(value).toBeGreaterThanOrEqual(ramp[i]));
    expect(heatColor(0)).toEqual([0, 0, 4]);
    expect(heatColor(1)).toEqual([252, 255, 164]);
    expect(heatColor(-3)).toEqual(heatColor(0));
    expect(heatColor(7)).toEqual(heatColor(1));
    expect(heatColor(NaN)).toEqual(heatColor(0));
  });

  it('overlay opacity is linear in weight, capped at the chosen opacity and zero without a scale', () => {
    expect(attentionAlpha(.25, 1, .8)).toBeCloseTo(.2);
    expect(attentionAlpha(.5, .5, .65)).toBeCloseTo(.65);
    expect(attentionAlpha(2, .5, .65)).toBe(.65);          // values above a fixed maximum saturate
    expect(attentionAlpha(-1, 1, .65)).toBe(0);
    expect(attentionAlpha(.3, 0, .65)).toBe(0);
    expect(attentionAlpha(.1, 1, .5) * 2).toBeCloseTo(attentionAlpha(.2, 1, .5));
  });

  it('weights keep their precision in labels', () => {
    expect(attentionWeightLabel(.5)).toBe('0.500000');
    expect(attentionWeightLabel(3e-8)).toBe('3.000e-8');
    expect(attentionWeightLabel(0)).toBe('0.000000');
  });

  it('rounded labels read at a glance without scientific notation', () => {
    expect(roundedWeightLabel(.095935)).toBe('0.096');
    expect(roundedWeightLabel(1)).toBe('1');
    expect(roundedWeightLabel(.0008)).toBe('0.001');
    expect(roundedWeightLabel(1e-8)).toBe('< 0.001');
    expect(roundedWeightLabel(3.331e-16)).toBe('≈ 0');
    expect(roundedWeightLabel(0)).toBe('0');
    expect(roundedWeightLabel(-.25)).toBe('−0.25');
    expect(roundedTimesLabel(9_593_546)).toBe('9.6 million');
    expect(roundedTimesLabel(2_000_000_000)).toBe('2 billion');
    expect(roundedTimesLabel(12_345)).toBe((12_345).toLocaleString());
    expect(scaleAgreement(1e-8, .095935, false, true)).toContain('9.6 million×');
    expect(scaleAgreement(1e-8, .095935, false)).toContain(`${Math.round(.095935 / 1e-8).toLocaleString()}×`);
  });

  it('share percentages never print a tiny positive share as 0%', () => {
    expect([0, 1e-12, .0005, .000999, .001, .0123, .0999, .1, .456, 1, -.0005, -.2].map(sharePercent))
      .toEqual(['0.0%', '<0.1%', '<0.1%', '<0.1%', '0.1%', '1.2%', '10.0%', '10%', '46%', '100%', '-0.1%', '-20.0%']);
  });

  it('older saved views hidden by the overlay flag reopen in Original', () => {
    const settings = {attention_display_mode: 'heatmap'} as DisplaySettings;
    expect(attentionMode(settings, false)).toBe('original');
    expect(attentionMode(settings, true)).toBe('heatmap');
    expect(attentionMode({} as DisplaySettings, null)).toBe('overlay');
  });
});

describe('image contrast maximum', () => {
  const positions = [
    {index: 0, role: 'special', bbox: null}, {index: 1, role: 'image', bbox: [0, 0, 1, 1]},
    {index: 2, role: 'image', bbox: [1, 0, 2, 1]}, {index: 3, role: 'query', bbox: null}, {index: 4, role: 'image', bbox: [0, 1, 1, 2]},
  ] as Generation['positions'];
  const capture = {positions} as Generation;
  it('uses only mapped image targets inside the causal row, never the dominant sink', () => {
    const row = {values: [.9, .02, .05, .03], target_count: 4} as Attention;
    expect(imageAttentionMaximum(capture, row)).toBe(.05);
    expect(imageAttentionMaximum(capture, {values: [.9, .02, .01, .03, .04], target_count: 3} as Attention)).toBe(.02);
    expect(imageAttentionMaximum(undefined, row)).toBe(1);
  });
});

describe('patch grid detection for smoothing', () => {
  const grid = (columns: number, rows: number, width: number, height: number) =>
    Array.from({length: rows * columns}, (_, k) => {const r = Math.floor(k / columns), c = k % columns;
      return [c * width / columns, r * height / rows, (c + 1) * width / columns, (r + 1) * height / rows];});
  it('recognises a regular row-major grid, including non-integer cell edges', () => {
    const boxes = grid(19, 26, 2480, 3509);
    expect(patchGrid(boxes)).toMatchObject({x1: 0, y1: 0, x2: 2480, y2: 3509});
    expect(patchGrid(boxes)!.columns).toHaveLength(19);
    expect(patchGrid(boxes)!.rows).toHaveLength(26);
  });
  it('declines layouts that are not one complete grid', () => {
    expect(patchGrid([])).toBeUndefined();
    expect(patchGrid(grid(4, 4, 100, 100).slice(1))).toBeUndefined();
    // Two pages of different sizes must never be smoothed as one raster.
    expect(patchGrid([...grid(2, 2, 100, 100), ...grid(3, 2, 90, 60)])).toBeUndefined();
  });
});

describe('column values', () => {
  it('are indexed by source position and leave sources that cannot see the target undefined', () => {
    const values = columnValues({sources: [{source_index: 5, value: .2}, {source_index: 2, value: .7}]});
    expect(values[5]).toBe(.2);
    expect(values[2]).toBe(.7);
    expect(values[3]).toBeUndefined();
    expect(columnValues(undefined)).toEqual([]);
  });
});
