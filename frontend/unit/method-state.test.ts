import { describe, it, expect } from 'vitest';
import { selectionKey, switchMethod, switchCapture, parametersFor, directionFor, openedView, RAW_ATTENTION } from '../src/features/document/methodState';
import type { Method } from '../src/api/client';
import type { View } from '../src/features/document/useAcquisition';

const mean={method_id:'attention.mean_heads',variant_id:'captured_prediction_v1',parameters_schema:{properties:{step:{},layer:{},heads:{}}}} as unknown as Method;
const weighted={...mean,method_id:'attention.weighted_heads',parameters_schema:{properties:{step:{},layer:{},heads:{},weights:{}}}} as Method;
const state={generation_capture_id:'generation_a',attention_method_id:mean.method_id,attention_variant_id:mean.variant_id,attention_step:2,attention_layer:3,attention_head:4,attention_parameters:{heads:[1,2]},attention_analysis_id:null} as unknown as View;

describe('saved method drafts',()=>{
  it('restores exact parameters and source selection when switching back',()=>{
    const next=switchMethod(state,weighted,mean);
    const back=switchMethod({...next,attention_layer:5,attention_parameters:{heads:[0],weights:[2]}},mean,weighted);
    expect(back.attention_parameters).toEqual(state.attention_parameters);
    expect(back.attention_layer).toBe(3);
    expect(back.attention_parameter_drafts?.['generation_a/attention.weighted_heads/captured_prediction_v1']).toEqual({step:2,layer:5,heads:[0],weights:[2]});
  });
  it('isolates drafts between captures and omits a single head for aggregate methods',()=>{
    const next=switchMethod(state,weighted,mean);
    const other=switchMethod({...next,generation_capture_id:'generation_b'},mean,weighted);
    expect(other.attention_parameters).toEqual({});
    expect(parametersFor(mean,state)).toEqual({heads:[1,2],step:2,layer:3});
  });
  it('ignores display and page changes in the scientific request key',()=>{
    expect(selectionKey({...state,page_id:'page_b',attention_overlay:false})).toBe(selectionKey(state));
    expect(selectionKey({...state,attention_parameters:{heads:[0]}})).not.toBe(selectionKey(state));
  });
  it('stores the departing capture draft and restores it without transferring it to a new capture',()=>{
    const changed=switchCapture(state,{...state,generation_capture_id:'generation_b',attention_step:0,attention_layer:0,attention_head:0,attention_parameters:{}},mean);
    expect(changed.attention_parameters).toEqual({});
    const back=switchCapture(changed,{...changed,generation_capture_id:'generation_a'},mean);
    expect(back.attention_parameters).toEqual({heads:[1,2]});
    expect(back.attention_step).toBe(2);
    expect(back.attention_layer).toBe(3);
  });
});

describe('page and method describe one quantity',()=>{
  it('keeps incoming-attention columns only for raw attention',()=>{
    expect(directionFor('attention.row','column')).toBe('column');
    expect(directionFor(undefined,'column')).toBe('column');
    expect(directionFor('attention.entropy','column')).toBe('row');
    expect(directionFor('attention.element_allocation','column')).toBe('row');
    expect(directionFor('attention.mean_heads','row')).toBe('row');
  });
  it('switching to raw attention keeps the drafts of the method being left',()=>{
    const raw=switchMethod(state,RAW_ATTENTION,mean);
    expect([raw.attention_method_id,raw.attention_variant_id]).toEqual(['attention.row','captured_weights_v1']);
    expect(raw.attention_parameter_drafts?.['generation_a/attention.mean_heads/captured_prediction_v1']).toMatchObject({heads:[1,2],layer:3});
    expect(raw.attention_analysis_id).toBeNull();
  });
});

describe('clicking chooses what is shown',()=>{
  it('a saved retriever column reopens as the row of its target position',()=>{
    const saved={...state,investigation_model_role:'retrieval',attention_direction:'column',attention_target_index:41,retrieval_source_index:7} as View;
    expect(openedView(saved)).toMatchObject({attention_direction:'row',retrieval_source_index:41,attention_target_index:null});
    expect(openedView({...saved,attention_target_index:null})).toMatchObject({attention_direction:'row',retrieval_source_index:7});
  });
  it('a generator patch column, which has no row of its own, reopens unchanged',()=>{
    const saved={...state,investigation_model_role:'generation',attention_direction:'column',attention_target_index:41} as View;
    expect(openedView(saved)).toBe(saved);
  });
  it('a token click after a patch restores the replaced method on the clicked step',()=>{
    // Patch click: Raw attention replaces the mean; token click: step 5, then the mean returns with its own parameters.
    const incoming={...switchMethod(state,RAW_ATTENTION,mean),attention_direction:'column'} as View;
    const clicked={...incoming,attention_direction:'row',attention_step:5,attention_source_index:null} as View;
    const restored=switchMethod(clicked,mean,{...mean,...RAW_ATTENTION} as Method,{preserveAxes:true});
    expect([restored.attention_method_id,restored.attention_step,restored.attention_layer,restored.attention_head]).toEqual([mean.method_id,5,3,4]);
    expect(restored.attention_parameters).toEqual({heads:[1,2]});
  });
});
