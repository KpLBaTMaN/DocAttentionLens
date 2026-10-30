import type { Case, DisplaySettings } from '../../api/client';
import type { components } from '../../api/generated/schema';
import type { Generation, Attention } from './GenerationPanel';
import type { Capture } from './RetrievalPanel';
import type { View } from './useAcquisition';
import { ATTENTION_FILL,mapGradient } from './attentionDisplay';

type Calculation = components['schemas']['Calculation'] & {quantity?:string;units?:string};
export type ReceiptInput = {record:Case;state:View;settings:DisplaySettings;generation?:Generation;capture?:Capture;
  role:'generation'|'retrieval';signal:'attention'|'similarity';methodName:string;scale:number;
  result?:Calculation;row?:Attention;column?:components['schemas']['ReceivedAttention'];tokenScale?:number;
  quantity?:string;units?:string;similarityResult?:unknown;displayMode?:string;pageQuantity?:string;groupedComparison?:string};

/** Exact current selection and immutable evidence, separate from presentation choices. */
export function researchReceipt(input:ReceiptInput) {
  const {record,state,settings,generation,capture,role,signal,result,row,column}=input;
  const incoming=signal==='attention'&&state.attention_direction==='column';
  const evidence=role==='generation'?generation:capture;
  // The retriever reading on screen, and the exact prompt of a Page + question reading (which may carry the reference answer).
  const encoding=role==='retrieval'&&signal==='attention'?capture?.attention_encodings?.find(e=>e.encoding_id===(state.retrieval_encoding_id??'document')):undefined;
  const answered=encoding?.answer_source==='reference';
  const method=signal==='similarity'?'retrieval.maxsim':incoming?'attention.row':role==='generation'?state.attention_method_id??'attention.row':state.retrieval_attention_method??'attention.row';
  return {receipt_version:'1.0.0',case:{case_id:record.case_id,dataset_version:record.dataset_version,annotation_version:record.annotation_version,question:record.query},
    evidence:{role,signal,capture_id:evidence?.capture_id??null,provenance:evidence?.provenance??null,arrays:evidence?.arrays??null,validation:evidence?.validation??null,
      input:role==='generation'?generation?.context??null:{query:capture?.query,page_ids:capture?.pages.map(p=>p.page_id),geometry:capture?.geometry},
      coverage:role==='generation'?generation?.coverage??null:capture?.attention_encodings??null,
      reading:encoding?{encoding_id:encoding.encoding_id,pass_kind:encoding.pass_kind,prompt_text:encoding.prompt_text??null,
        answer_source:encoding.answer_source??null,answer_text:encoding.answer_text??null}:null},
    analysis:{method_id:method,method_name:incoming?'Raw incoming attention':input.methodName,
      variant_id:signal==='similarity'?null:role==='generation'?state.attention_variant_id: 'retriever_encoding_v1',
      analysis_id:incoming||signal==='similarity'||role==='retrieval'?null:state.attention_analysis_id??null,selection:state,
      result:incoming?column??null:signal==='attention'?result??row??null:input.similarityResult??null,
      quantity:incoming?column?.quantity:signal==='similarity'?'signed_maxsim_similarity':result?.quantity??input.quantity??row?.quantity,
      units:incoming?column?.units:signal==='similarity'?'signed dot product of normalized embeddings':result?.units??input.units??(row?.quantity==='post_softmax_attention_weight'?'post-softmax attention weight':'head envelope value'),
      normalization:incoming?column?.normalization:signal==='similarity'?'signed similarity; not attention mass or probability':result?.normalization??row?.normalization},
    presentation:{settings,page_scale_min:signal==='similarity'?-input.scale:0,page_scale_max:input.scale,token_scale_max:input.tokenScale??null,
      page_quantity:input.pageQuantity??null,grouped_comparison:input.groupedComparison??null,
      display_mode:input.displayMode??settings.attention_display_mode,scale_mode:signal==='similarity'?settings.retrieval_scale_mode:settings.attention_scale_mode,
      palette:signal==='similarity'?{negative:'#247ed5',positive:'#b080d1',encoding:'linear opacity'}:(input.displayMode??settings.attention_display_mode)==='heatmap'?{colormap:settings.attention_colormap??'inferno',css_gradient:mapGradient(settings.attention_colormap??'inferno'),encoding:'opaque colour'}:(input.displayMode??settings.attention_display_mode)==='blend'?{colormap:settings.attention_colormap??'inferno',css_gradient:mapGradient(settings.attention_colormap??'inferno'),encoding:`colour map blended over the page at opacity ${settings.attention_opacity??.65}`}:{colour:ATTENTION_FILL,encoding:'linear opacity'},
      range_floor:signal==='similarity'?null:settings.attention_range_floor??'zero',
      token_palette:signal==='similarity'?{negative:'#247ed5',positive:'#b080d1'}:{colour:ATTENTION_FILL},
      smoothing:'display interpolation only; captured values unchanged',comparison_scale:'shared across compatible comparison rows'},
    interpretation:{attention:'Allocation weights describe the saved computation; they do not establish causation or answer correctness.',
      image_mass:incoming||signal==='similarity'?null:row?{image:row.image_mass,total:row.total_mass,share:row.total_mass>0?row.image_mass/row.total_mass:null}:null,
      evaluation_annotations:answered
        ?'This diagnostic reading’s prompt contains the case reference answer (an evaluation annotation). Question and Page readings and retrieval scores exclude it; designated support is excluded from model input.'
        :'Reference answers and designated support are evaluation annotations, excluded from model input.'}};
}

export type ResearchReceipt = ReturnType<typeof researchReceipt>;
export function downloadBlob(blob:Blob,filename:string) {
  const url=URL.createObjectURL(blob),a=document.createElement('a');a.href=url;a.download=filename;a.click();
  // Let the browser finish reading the blob before releasing it.
  setTimeout(()=>URL.revokeObjectURL(url),1000);
}

/** Portable SVG snapshot: retain the displayed viewport, inline local images and attach the receipt. */
export async function exportPageFigure(receipt:ResearchReceipt,svgs:SVGSVGElement[],caption:string,columns=svgs.length) {
  if(!svgs.length||svgs.some(svg=>svg.getBoundingClientRect().width===0))throw new Error('Open the document view before exporting its figure.');
  if(svgs.some(svg=>svg.closest('[data-testid="page-canvas"]')?.querySelector('.docattentionlens-canvas-loading')))throw new Error('Wait for the saved page image to finish loading before exporting.');
  const ns='http://www.w3.org/2000/svg',outer=document.createElementNS(ns,'svg');
  const width=1600,header=180,rows=Math.ceil(svgs.length/columns),panelWidth=width/columns,panelHeight=rows>1?500:700;
  outer.setAttribute('xmlns',ns);outer.setAttribute('viewBox',`0 0 ${width} ${header+rows*panelHeight}`);
  outer.setAttribute('width',String(width));outer.setAttribute('height',String(header+rows*panelHeight));
  const bg=document.createElementNS(ns,'rect');bg.setAttribute('width','100%');bg.setAttribute('height','100%');bg.setAttribute('fill','#fff');outer.append(bg);
  const metadata=document.createElementNS(ns,'metadata');metadata.textContent=JSON.stringify(receipt);outer.append(metadata);
  const lines=[caption,`${receipt.analysis.method_name} · ${receipt.analysis.quantity??'quantity unavailable'} · ${receipt.analysis.units??''}`,
    `Page scale: ${receipt.presentation.page_scale_min.toPrecision(6)} → ${receipt.presentation.page_scale_max.toPrecision(6)} · ${receipt.presentation.scale_mode} · ${receipt.presentation.display_mode==='heatmap'?'heatmap (dark → bright)':receipt.presentation.display_mode} · smoothing ${receipt.presentation.settings.attention_smooth&&receipt.evidence.signal==='attention'?'on (display only)':'off'}`,
    `Capture ${receipt.evidence.capture_id} · ${receipt.evidence.role} · ${receipt.evidence.signal}`,
    `${receipt.evidence.signal==='similarity'?(receipt.analysis.selection.similarity_direction==='column'?'Similarity patch '+receipt.analysis.selection.similarity_target_index:'Similarity query '+receipt.analysis.selection.query_position):receipt.analysis.selection.attention_direction==='column'?'Incoming target '+receipt.analysis.selection.attention_target_index:receipt.evidence.role==='generation'?'Prediction step '+(receipt.analysis.selection.attention_step??0)+' / source '+(receipt.analysis.selection.attention_source_index??'prediction row'):'Encoding '+(receipt.analysis.selection.retrieval_encoding_id??'document')+' / source '+(receipt.analysis.selection.retrieval_source_index??0)+(receipt.evidence.reading?.answer_source==='reference'?` · prompt includes the reference answer “${receipt.evidence.reading.answer_text}”`:'')} · selection and exact parameters in attached receipt`,
    receipt.interpretation.attention];
  for(let i=0;i<lines.length;i++){const text=document.createElementNS(ns,'text');text.setAttribute('x','16');text.setAttribute('y',String(24+i*24));text.setAttribute('font-size','15');text.setAttribute('font-family','sans-serif');text.setAttribute('fill','#171717');text.textContent=lines[i].slice(0,145);outer.append(text);}
  for(let index=0;index<svgs.length;index++){
    const source=svgs[index],clone=source.cloneNode(true) as SVGSVGElement;
    const original=[source,...source.querySelectorAll('*')],copied=[clone,...clone.querySelectorAll('*')];
    original.forEach((node,i)=>{const style=getComputedStyle(node);for(const property of ['fill','fill-opacity','stroke','stroke-width','stroke-dasharray','stroke-opacity','opacity','font-size','font-family','visibility','display'])copied[i].setAttribute(property,style.getPropertyValue(property));});
    for(const image of clone.querySelectorAll('image')){
      const href=image.getAttribute('href');if(!href||href.startsWith('data:'))continue;
      const url=new URL(href,location.href);if(url.origin!==location.origin)throw new Error('The figure contains an image outside this application.');
      const response=await fetch(url);if(!response.ok)throw new Error('Could not embed the saved page image.');
      const blob=await response.blob();const encoded=await new Promise<string>((resolve,reject)=>{const reader=new FileReader();reader.onload=()=>resolve(String(reader.result));reader.onerror=()=>reject(new Error('Could not read the saved image.'));reader.readAsDataURL(blob);});image.setAttribute('href',encoded);
    }
    const x=(index%columns)*panelWidth,y=header+Math.floor(index/columns)*panelHeight;
    const label=document.createElementNS(ns,'text');label.setAttribute('x',String(x+16));label.setAttribute('y',String(y+20));label.setAttribute('font-size','16');label.setAttribute('font-family','sans-serif');label.setAttribute('fill','#171717');label.textContent=source.closest('figure')?.querySelector('figcaption')?.textContent??source.getAttribute('aria-label')??`Map ${index+1}`;outer.append(label);
    clone.setAttribute('x',String(x));clone.setAttribute('y',String(y+30));clone.setAttribute('width',String(panelWidth));clone.setAttribute('height',String(panelHeight-30));outer.append(clone);
  }
  return new Blob([new XMLSerializer().serializeToString(outer)],{type:'image/svg+xml'});
}
