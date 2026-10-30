import { describe,it,expect } from 'vitest';
import { researchReceipt,type ReceiptInput } from '../src/features/document/receiptExport';

const input={record:{case_id:'case_a',dataset_version:'v1',annotation_version:'a1',query:'Question'},
  state:{generation_capture_id:'gen_a',attention_method_id:'attention.entropy',attention_variant_id:'shannon_row_v1',attention_analysis_id:'analysis_a'},
  settings:{attention_scale_mode:'auto',retrieval_scale_mode:'fixed',attention_smooth:true},
  generation:{capture_id:'gen_a',provenance:{model_id:'model_a',revision:'rev_a'},arrays:{sha256:'hash_a'},validation:{status:'passed'},context:{selected_page_ids:['page_a']}},
  capture:{capture_id:'retrieval_a',provenance:{model_id:'model_b'},pages:[{page_id:'page_b'}]},
  role:'generation',signal:'attention',methodName:'Entropy',scale:.002,
  result:{statistics:[{name:'Entropy',value:2,units:'nats'}],normalization:'original domain'},quantity:'attention_entropy',units:'nats'} as unknown as ReceiptInput;

describe('research receipt describes the displayed evidence',()=>{
  it('records the grouped page quantity and comparison without relabelling its canonical calculation',()=>{
    const receipt=researchReceipt({...input,state:{...input.state,attention_method_id:'attention.rollout_evolution'},
      quantity:'rollout_evolution_connectivity',pageQuantity:'Rollout connectivity by element',groupedComparison:'Layer 2',scale:.15});
    expect(receipt.analysis.method_id).toBe('attention.rollout_evolution');
    expect(receipt.analysis.quantity).toBe('rollout_evolution_connectivity');
    expect(receipt.presentation.page_quantity).toBe('Rollout connectivity by element');
    expect(receipt.presentation.grouped_comparison).toBe('Layer 2');
    expect(receipt.presentation.page_scale_max).toBe(.15);
  });
  it('retains model identity, immutable evidence and full statistical result separately from display',()=>{
    const receipt=researchReceipt(input);
    expect(receipt.analysis.quantity).toBe('attention_entropy');
    expect(receipt.analysis.units).toBe('nats');
    expect(receipt.analysis.result).toEqual(input.result);
    expect(receipt.evidence.capture_id).toBe('gen_a');
    expect(receipt.evidence.provenance).toEqual(input.generation?.provenance);
    expect(receipt.presentation.page_scale_max).toBe(.002);
    expect(receipt.analysis.normalization).toBe('original domain');
  });
  it('incoming columns export incoming sources rather than an unrelated outgoing analysis',()=>{
    const column={capture_id:'gen_a',target_index:8,layer:0,head:0,sources:[{source_index:9,value:.4,kind:'prediction' as const,label:'step 0'}],maximum:.4,quantity:'attention_received' as const,units:'post-softmax weight within each source row',normalization:'sources do not sum to one'};
    const receipt=researchReceipt({...input,state:{...input.state,attention_direction:'column'},column});
    expect(receipt.analysis.method_id).toBe('attention.row');
    expect(receipt.analysis.analysis_id).toBeNull();
    expect(receipt.analysis.result).toEqual(column);
    expect(receipt.analysis.quantity).toBe('attention_received');
    expect(receipt.interpretation.image_mass).toBeNull();
  });
  it('signed similarity retains the retrieval capture, scores and symmetric scale',()=>{
    const scores={similarities:[-.2,.8],maximum:.8,winner_index:1};
    const receipt=researchReceipt({...input,role:'retrieval',signal:'similarity',similarityResult:scores,scale:.8});
    expect(receipt.evidence.capture_id).toBe('retrieval_a');
    expect(receipt.analysis.analysis_id).toBeNull();
    expect(receipt.analysis.variant_id).toBeNull();
    expect(receipt.analysis.result).toEqual(scores);
    expect(receipt.analysis.normalization).toContain('not attention mass');
    expect(receipt.presentation.scale_mode).toBe('fixed');
    expect(receipt.presentation.page_scale_min).toBe(-.8);
  });
});
