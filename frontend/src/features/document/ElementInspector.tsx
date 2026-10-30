import type { ReactNode } from 'react';
import type { Element } from '../../api/client';
import type { Capture } from './RetrievalPanel';
import type { Attention, Generation } from './GenerationPanel';
import type { UpdateView } from './useAcquisition';
import { ElementEvidence } from './ElementEvidence';

export function ElementInspector({selected,retrieval,generation,attention,isGeneration,sourceLabel,targetLabel,update,sourceId,targetId,profile}: {
  selected?:Element; retrieval?:Capture; generation?:Generation; attention?:Attention; isGeneration:boolean;
  sourceLabel:string; targetLabel:string; update:UpdateView; sourceId?:string|null; targetId?:string|null;
  /** The selected element's layer and head profile, directly under its generator allocation. */
  profile?:ReactNode;
}) {
  const candidate=retrieval?.pages.find(p=>p.page_id===selected?.page_id);
  return <section className="docattentionlens-element-inspection">
    <h2>Selected element</h2>
    {selected ? <>
      <h3 data-testid="selected-element">Element {selected.native_id} · {selected.element_type}</h3>
      {isGeneration ? <><ElementEvidence element={selected} retrieval={retrieval} generation={generation} attention={attention}/>{profile}</> : <>
        <h3>Retrieval contribution</h3>
        <p data-testid="element-retrieval-evidence">{!candidate?'Page outside selected retrieval capture':!candidate.ownership.some(o=>selected.element_id in o)?'No reviewed element mapping':`${(candidate.contributions[selected.element_id]??0).toFixed(6)} · signed full-query MaxSim contribution`}</p>
        {/* The retriever's own encoding attention to this element, by layer and head (separate from MaxSim). */}
        {profile}
      </>}
      <details><summary>OCR</summary><p className="docattentionlens-ocr" data-testid="ocr-text">{selected.ocr.status==='missing'?'OCR record missing':selected.ocr.status==='empty'?'OCR record present; text is empty':selected.ocr.text}</p></details>
      <details><summary>Geometry</summary><p className="docattentionlens-element-meta">{selected.element_category} · depth {selected.level??'unrecorded'} · order {selected.reading_order??'unrecorded'}</p><h3>Raw source box</h3><p className="docattentionlens-coordinates">({selected.bbox.x1}, {selected.bbox.y1}) → ({selected.bbox.x2}, {selected.bbox.y2})</p><p>Source pixels · top-left origin</p><pre>{JSON.stringify({ref:selected.ref,bbox:selected.bbox,display_bbox:selected.display_bbox,geometry_flags:selected.geometry_flags},null,2)}</pre></details>
      {!!selected.geometry_flags.length&&<p role="status">Geometry flagged: {selected.geometry_flags.join(', ')}. Unreviewed allocation remains in residuals.</p>}
    </> : <p>Select an element on the document, outline or contribution ledger.</p>}
    <details><summary>Inspection markers</summary>
      {selected&&<div className="docattentionlens-actions"><button className="docattentionlens-button" aria-label="Mark inspection source" onClick={()=>update(s=>({...s,source_element_id:selected.element_id}))}>Mark source</button><button className="docattentionlens-button" aria-label="Mark inspection target" onClick={()=>update(s=>({...s,target_element_id:selected.element_id}))}>Mark target</button></div>}
      <p>Inspection markers do not change the captured attention source.</p>
      <p>Source: {sourceLabel}</p>{sourceId&&<button className="docattentionlens-button" onClick={()=>update(s=>({...s,source_element_id:null}))}>Clear source</button>}
      <p>Target: {targetLabel}</p>{targetId&&<button className="docattentionlens-button" onClick={()=>update(s=>({...s,target_element_id:null}))}>Clear target</button>}
    </details>
    <details><summary>Signal interpretation</summary><p>{isGeneration?'Attention allocation is not output importance or causation. Nominal patch support is not a receptive field.':'Retrieval similarities are signed scores, not attention weights.'}</p></details>
  </section>;
}
