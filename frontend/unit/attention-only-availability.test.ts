import { describe,it,expect } from 'vitest';
import { encodingMethodGap, defaultSource } from '../src/features/document/useRetrieverAttention';

describe('joint reading source defaults', () => {
  it('selects the final active question word, excluding trailing prompt and padding', () => {
    const encoding = {pass_kind: 'joint', causal: true, attention_mask: [1,1,1,1,0], positions: [
      {index: 0, role: 'visual_patch'}, {index: 1, role: 'query_text'}, {index: 2, role: 'query_text'},
      {index: 3, role: 'visual_prompt'}, {index: 4, role: 'query_text'},
    ]};
    expect(defaultSource(encoding)).toBe(2);
    expect(defaultSource({...encoding, causal: false})).toBe(2);
    expect(defaultSource({...encoding, positions: encoding.positions.map(p => ({...p, eligible: p.index !== 2}))})).toBe(1);
  });
  it('retains older document defaults and the fallback for a joint capture without question metadata', () => {
    expect(defaultSource({causal: true, attention_mask: [1,1,0]})).toBe(1);
    expect(defaultSource({causal: false, attention_mask: [1,1,0]})).toBe(0);
    expect(defaultSource({pass_kind: 'joint', causal: true, attention_mask: [1,1,0]})).toBe(1);
  });
});

const positions=[{role:'visual_patch',bbox:[0,0,1,1]},{role:'visual_patch',bbox:[1,0,2,1]},{role:'visual_prompt'},{role:'query_text'}];
describe('attention-only source availability',()=>{
  it('does not offer patch-source directions on text sources',()=>{
    const method={requires:['captured_image_sources']};
    expect(encodingMethodGap(method,{attention_mask:[1,1,1,1]},positions,3)).toContain('image-patch source');
    expect(encodingMethodGap(method,{attention_mask:[1,1,1,1]},positions,0)).toBeUndefined();
  });
  it('keeps non-image source and image-target requirements separate',()=>{
    expect(encodingMethodGap({requires:['captured_text_sources']},{attention_mask:[1,1,1,1]},positions,0)).toContain('textual');
    expect(encodingMethodGap({requires:['image_mask']},{attention_mask:[1]},[{role:'query_text'}],0)).toContain('mapped image');
  });
  it('does not present future text as a valid causal modality block',()=>{
    const method={method_id:'attention.vision_to_text',requires:['captured_image_sources']};
    expect(encodingMethodGap(method,{attention_mask:[1,1,1,1],causal:true},positions,0)).toContain('no valid targets');
    expect(encodingMethodGap(method,{attention_mask:[1,1,1,1],causal:false},positions,0)).toBeUndefined();
  });
  it('counts active source rows rather than padded positions',()=>{
    expect(encodingMethodGap({requires:['multiple_source_rows']},{attention_mask:[1,0,0,0]},positions,0)).toContain('two captured');
  });
});
