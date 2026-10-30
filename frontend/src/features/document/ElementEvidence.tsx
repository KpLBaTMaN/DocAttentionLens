import type { Element } from '../../api/client';
import type { Capture } from './RetrievalPanel';
import type { Generation, Attention } from './GenerationPanel';
import { attentionWeightLabel } from './attentionDisplay';

export function ElementEvidence({element,retrieval,generation,attention}:{element:Element;retrieval?:Capture;generation?:Generation;attention?:Attention}) {
  const candidate=retrieval?.pages.find(p=>p.page_id===element.page_id);
  const retrievalMapped=candidate?.ownership.some(o=>element.element_id in o);
  const supplied=generation?.context.pages.find(p=>p.page_id===element.page_id);
  const allocation=attention?.allocations.find(a=>a.element_id===element.element_id&&a.page_id===element.page_id);
  return <section aria-label="Selected element generator evidence"><h3>Generator allocation</h3>
    <dl><dt>{attention&&attention.quantity!=='post_softmax_attention_weight'?'Envelope allocation':'Attention mass'}</dt><dd data-testid="element-attention-evidence">{!attention?'No allocation row for this method':!supplied?'Unavailable: page was not supplied':element.geometry_flags.length?'Unreviewed geometry: allocation held in residual':allocation?`${attentionWeightLabel(allocation.mass)} · ${attention.quantity==='post_softmax_attention_weight'?'original-domain mass':'head envelope units'}`:'No element support in captured target mapping'}</dd>
      <dt>Context</dt><dd data-testid="element-context-evidence">{!generation?'No generator capture selected':supplied?`Page supplied as an image · ${supplied.resized_width} × ${supplied.resized_height}`:'Page omitted from generator context'}</dd></dl>
    <details><summary>Retrieval provenance</summary><p data-testid="element-retrieval-evidence">{!candidate?'Page outside selected retrieval capture':!retrievalMapped?'No reviewed element mapping':`${(candidate.contributions[element.element_id]??0).toFixed(6)} · signed full-query MaxSim contribution`}</p></details>
    <details><summary>Meaning of these values</summary><p>Context inclusion refers to the resized page image. These quantities keep separate units; attention allocation is not output importance.</p></details>
  </section>;
}
