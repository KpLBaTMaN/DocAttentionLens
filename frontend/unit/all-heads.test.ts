import { describe, it, expect } from 'vitest';
import { allHeadsState, everyHead, selectsEveryHead, singleHeadReturn, switchMethod, parametersFor, HEAD_MEAN_METHOD, RAW_ATTENTION } from '../src/features/document/methodState';
import type { Method } from '../src/api/client';
import type { View } from '../src/features/document/useAcquisition';

describe('All heads in the layer scrubber', () => {
  it('lists every head index explicitly', () => {
    expect(everyHead(4)).toEqual([0, 1, 2, 3]);
    expect(everyHead(0)).toEqual([]);
  });

  it('recognises a selection of every head in any order, and nothing less', () => {
    expect(selectsEveryHead([0, 1, 2, 3], 4)).toBe(true);
    expect(selectsEveryHead([3, 1, 0, 2], 4)).toBe(true);
    expect(selectsEveryHead([0, 1, 2], 4)).toBe(false);
    expect(selectsEveryHead([0, 1, 2, 2], 4)).toBe(false);
    expect(selectsEveryHead([0, 1, 2, 4], 4)).toBe(false);
    expect(selectsEveryHead([0, 1, 2, 1.5], 4)).toBe(false);
    expect(selectsEveryHead([], 4)).toBe(false);
    expect(selectsEveryHead(undefined, 4)).toBe(false);
    expect(selectsEveryHead([], 0)).toBe(false);
  });

  it('offers All for raw attention (also grouped by element) and shows it active for the explicit head mean', () => {
    expect(allHeadsState('attention.row', {}, 16)).toBe('single');
    expect(allHeadsState(undefined, undefined, 16)).toBe('single');
    expect(allHeadsState('attention.element_allocation', {}, 16)).toBe('single');
    expect(allHeadsState(HEAD_MEAN_METHOD, {heads: everyHead(16)}, 16)).toBe('all');
  });

  it('has no All cell for other methods, head subsets, the default selection or a capture without heads', () => {
    expect(allHeadsState('attention.entropy', {}, 16)).toBeUndefined();
    expect(allHeadsState('attention.max_heads', {heads: everyHead(16)}, 16)).toBeUndefined();
    expect(allHeadsState(HEAD_MEAN_METHOD, {heads: [0, 1]}, 16)).toBeUndefined();
    expect(allHeadsState(HEAD_MEAN_METHOD, {}, 16)).toBeUndefined();
    expect(allHeadsState(HEAD_MEAN_METHOD, {heads: []}, 16)).toBeUndefined();
    expect(allHeadsState('attention.row', {}, 0)).toBeUndefined();
  });

  it('returns to the single-head method All was chosen from, defaulting to raw attention', () => {
    expect(singleHeadReturn('attention.row')).toBe(RAW_ATTENTION.method_id);
    expect(singleHeadReturn(undefined)).toBe(RAW_ATTENTION.method_id);
    expect(singleHeadReturn('attention.entropy')).toBe(RAW_ATTENTION.method_id);
    // The element alias returns as raw attention; the perspective keeps the element grouping.
    expect(singleHeadReturn('attention.element_allocation')).toBe(RAW_ATTENTION.method_id);
  });

  it('switching to the head mean keeps step, layer and head, and the request carries every head but no single head', () => {
    const raw = {method_id: 'attention.row', variant_id: 'captured_weights_v1', parameters_schema: {properties: {step: {}, layer: {}, head: {}}}} as unknown as Method;
    const mean = {method_id: HEAD_MEAN_METHOD, variant_id: 'captured_prediction_v1', parameters_schema: {properties: {step: {}, layer: {}, heads: {}}}} as unknown as Method;
    const state = {generation_capture_id: 'generation_a', attention_method_id: raw.method_id, attention_variant_id: raw.variant_id,
      attention_step: 2, attention_layer: 7, attention_head: 5, attention_parameters: {}, attention_parameter_drafts: {
        // An earlier mean draft over a subset, at other axes, must not leak into All.
        [`generation_a/${mean.method_id}/${mean.variant_id}`]: {step: 0, layer: 1, heads: [0, 1]}}} as unknown as View;
    const switched = switchMethod(state, mean, raw, {preserveAxes: true});
    const all = {...switched, attention_parameters: {...switched.attention_parameters, heads: everyHead(8)}};
    expect([all.attention_step, all.attention_layer, all.attention_head]).toEqual([2, 7, 5]);
    expect(allHeadsState(all.attention_method_id, all.attention_parameters, 8)).toBe('all');
    expect(parametersFor(mean, all)).toEqual({heads: everyHead(8), step: 2, layer: 7});
    const back = switchMethod(all, raw, mean, {preserveAxes: true});
    expect(back.attention_method_id).toBe('attention.row');
    expect(back.attention_head).toBe(5);
    expect(back.attention_parameters).toEqual({});
    expect(back.attention_parameter_drafts?.[`generation_a/${mean.method_id}/${mean.variant_id}`]).toEqual({heads: everyHead(8), step: 2, layer: 7});
  });
});
