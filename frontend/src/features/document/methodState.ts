import type { Method } from '../../api/client';
import type { View } from './useAcquisition';

export function selectionKey(state:View) {
  return JSON.stringify([state.generation_capture_id,state.attention_method_id,state.attention_variant_id,
    state.attention_step,state.attention_source_index??null,state.attention_layer,state.attention_head,state.attention_parameters??{}]);
}

export function parametersFor(method:Method, state:View) {
  const properties=method.parameters_schema.properties as Record<string,unknown>??{};
  return {...state.attention_parameters,step:state.attention_step??0,layer:state.attention_layer??0,
    ...(properties.head?{head:state.attention_head??0}:{})};
}

/** "What attends to it" reads raw attention columns; every other method describes one source row. */
export const RAW_ATTENTION={method_id:'attention.row',variant_id:'captured_weights_v1'} as const;

/** The direction that keeps the page and the selected method on the same quantity. */
export function directionFor(methodId:string|null|undefined, direction:View['attention_direction']):View['attention_direction'] {
  return (methodId??RAW_ATTENTION.method_id)===RAW_ATTENTION.method_id?direction:'row';
}

/** The view a click opens: a generator image patch opens its incoming column, anything else its own row. */
export function openedView(view:View):View {
  return view.investigation_model_role==='retrieval'&&view.attention_direction==='column'
    ?{...view,attention_direction:'row',retrieval_source_index:view.attention_target_index??view.retrieval_source_index,attention_target_index:null}:view;
}

/**
 * Selects a method and restores its parameter draft. `preserveAxes` keeps the current
 * prediction, layer and head, as a perspective switch must: it re-analyses the same row.
 */
export function switchMethod(state:View, method:Pick<Method,'method_id'|'variant_id'>, previous?:Method, {preserveAxes=false}={}):View {
  const key=`${state.generation_capture_id}/${method.method_id}/${method.variant_id}`;
  const oldKey=`${state.generation_capture_id}/${state.attention_method_id}/${state.attention_variant_id}`;
  const drafts={...state.attention_parameter_drafts};
  if(previous&&state.generation_capture_id)drafts[oldKey]=parametersFor(previous,state);
  const restored=drafts[key]??{};
  const {step,layer,head,...parameters}=restored;
  const restore=!preserveAxes;
  return {...state,attention_method_id:method.method_id,attention_variant_id:method.variant_id,
    attention_step:restore&&typeof step==='number'?step:state.attention_step,
    attention_layer:restore&&typeof layer==='number'?layer:state.attention_layer,
    attention_head:restore&&typeof head==='number'?head:state.attention_head,
    attention_parameters:parameters,attention_parameter_drafts:drafts,attention_analysis_id:null};
}

export function switchCapture(previous:View, next:View, method?:Method):View {
  if(previous.generation_capture_id===next.generation_capture_id)return next;
  const drafts={...next.attention_parameter_drafts};
  if(previous.generation_capture_id&&method?.method_id===previous.attention_method_id&&method.variant_id===previous.attention_variant_id){
    drafts[`${previous.generation_capture_id}/${method.method_id}/${method.variant_id}`]=parametersFor(method,previous);
  }
  const key=`${next.generation_capture_id}/${next.attention_method_id}/${next.attention_variant_id}`;
  const {step,layer,head,...parameters}=drafts[key]??{};
  // Question-token sources and column targets are position indices of the previous capture.
  return {...next,attention_parameters:parameters,attention_parameter_drafts:drafts,attention_analysis_id:null,attention_source_index:null,attention_target_index:null,
    attention_step:typeof step==='number'?step:next.attention_step,
    attention_layer:typeof layer==='number'?layer:next.attention_layer,
    attention_head:typeof head==='number'?head:next.attention_head};
}

/** The head mean the scrubber's "All" cell switches to. */
export const HEAD_MEAN_METHOD='attention.mean_heads';
/**
 * Single-head methods whose mean over heads is a method of its own. The element method is Raw
 * attention grouped by element (its historic alias), so its mean is Mean across heads grouped the same way.
 */
const HEAD_MEAN_OF:Record<string,string>={'attention.row':HEAD_MEAN_METHOD,'attention.element_allocation':HEAD_MEAN_METHOD};

/** Every head index of a layer with `count` query heads: an explicit "all heads" selection. */
export const everyHead=(count:number)=>Array.from({length:Math.max(0,count)},(_,i)=>i);

/** True when `heads` names each of the `count` heads exactly once (in any order). */
export function selectsEveryHead(heads:unknown,count:number):boolean {
  if(!Array.isArray(heads)||count<1||heads.length!==count)return false;
  return heads.every(h=>Number.isInteger(h)&&h>=0&&h<count)&&new Set(heads).size===count;
}

/** What the scrubber's "All" cell means for the shown method, or undefined when it has none. */
export function allHeadsState(methodId:string|null|undefined,parameters:Record<string,unknown>|null|undefined,headCount:number):'single'|'all'|undefined {
  const id=methodId??RAW_ATTENTION.method_id;
  if(headCount<1)return undefined;
  if(HEAD_MEAN_OF[id])return 'single';
  return id===HEAD_MEAN_METHOD&&selectsEveryHead(parameters?.heads,headCount)?'all':undefined;
}

/**
 * The single-head method that leaving "All" returns to: the one the All cell was chosen from, or Raw
 * attention. The element alias returns as Raw attention: the perspective keeps the element grouping.
 */
export function singleHeadReturn(remembered?:string|null):string {
  return remembered&&HEAD_MEAN_OF[remembered]&&remembered!=='attention.element_allocation'?remembered:RAW_ATTENTION.method_id;
}
