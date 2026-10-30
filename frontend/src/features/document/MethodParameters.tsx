import { Fragment, useEffect, useState } from 'react';
import type { Method } from '../../api/client';

type Schema = {type?:string; enum?:string[]; const?:unknown; items?:Schema; minimum?:number; maximum?:number; default?:unknown; description?:string; title?:string};
type Parameters = Record<string,unknown>;

function NumericArray({label,value,schema,onChange}:{label:string;value:number[];schema:Schema;onChange:(values:number[])=>void}) {
  const serialized=value.join(', ');
  const [draft,setDraft]=useState(serialized),[error,setError]=useState(false);
  useEffect(()=>{setDraft(serialized);setError(false);},[serialized]);
  const commit=()=>{const values=draft.trim()?draft.split(',').map(s=>s.trim()?Number(s):NaN):[];
    if(values.every(v=>Number.isFinite(v)&&(schema.items?.type!=='integer'||Number.isInteger(v))&&v>=(schema.items?.minimum??-Infinity)&&v<=(schema.items?.maximum??Infinity))){onChange(values);setError(false);}else setError(true);};
  return <label>{label}<input aria-label={label} value={draft} onChange={e=>setDraft(e.target.value)} onBlur={commit} onKeyDown={e=>{if(e.key==='Enter')commit();}} placeholder="Comma-separated values"/>{schema.description&&<small>{schema.description}</small>}{error&&<small role="alert">Enter comma-separated numbers within the declared bounds.</small>}</label>;
}

/** Axis selections live with the source controls; entropy's fixed base is stated by its units (bits). */
const HIDDEN = ['step','layer','head','log_base'];
const TITLES: Record<string,string> = {
  'attention.element_allocation':'Element aggregation settings',
  'attention.mean_heads':'Head aggregation settings', 'attention.max_heads':'Head aggregation settings',
  'attention.min_heads':'Head aggregation settings', 'attention.weighted_heads':'Head aggregation settings',
  'attention.topk_mass':'Top-k settings', 'attention.sinks':'Target group settings',
  'attention.head_by_head':'Head comparison settings', 'heads.distribution_similarity':'Head comparison settings',
  'attention.across_layers':'Layer comparison settings',
};
const FIXED_VALUES: Record<string,string> = {single_row:'single selected row', original_capture_domain:'original captured domain · no renormalization'};

/** Settings that change the selected calculation, titled by what they configure; absent when there are none. */
export function MethodParameters({method, layerCount, headCount, maxTargets, parameters, onChange, domain='causal', hiddenParameters=[]}:{
  method:Method; layerCount:number; headCount:number; maxTargets?:number; parameters:Parameters;
  onChange:(next:(previous:Parameters)=>Parameters)=>void; domain?:'causal'|'encoding'; hiddenParameters?:string[];
}) {
  const properties = method.parameters_schema.properties as Record<string,Schema> | undefined;
  const set = (key:string, value:unknown) => onChange(p=>({...p,[key]:value}));
  const entries = Object.entries(properties ?? {}).filter(([key])=>!HIDDEN.includes(key)&&!hiddenParameters.includes(key));
  // Single-valued settings are fixed by the variant: disclosed, not editable.
  const fixed = entries.filter(([,schema])=>schema.const!==undefined);
  const controls = entries.filter(([,schema])=>schema.const===undefined).map(([key,schema])=>{
    if(key==='heads'||key==='layers') {
      const count = key==='heads'?headCount:layerCount;
      const all = Array.from({length:count},(_,i)=>i);
      const stored = (parameters[key]??schema.default) as number[] | undefined;
      const chosen = stored?.length?stored:all;
      return <fieldset className="docattentionlens-method-axis-set" key={key}><legend>{key==='heads'?'Selected heads':'Selected layers'} · {chosen.length}/{count}</legend>
        {/* Weighted heads keep one weight per selected head, so selecting all re-aligns the weights too. */}
        <button className="docattentionlens-button" onClick={()=>onChange(p=>key==='heads'&&properties?.weights?{...p,heads:all,weights:all.map(i=>(p.weights as number[]|undefined)?.[chosen.indexOf(i)]??1)}:{...p,[key]:all})}>Select all {key}</button>
        <div>{all.map(index=><label className="docattentionlens-checkbox" key={index}><input type="checkbox" aria-label={`Include ${key==='heads'?'head':'layer'} ${index}`} checked={chosen.includes(index)} disabled={chosen.length===1&&chosen.includes(index)}
          onChange={e=>{const selected=e.target.checked?all.filter(i=>chosen.includes(i)||i===index):chosen.filter(i=>i!==index);onChange(p=>({...p,[key]:selected,...(key==='heads'&&properties?.weights?{weights:selected.map(i=>{const old=(p.weights as number[]|undefined);return old?.[chosen.indexOf(i)]??1;})}:{})}));}}/>{index}</label>)}</div>
      </fieldset>;
    }
    if(key==='weights') {
      const heads = parameters.heads as number[] | undefined;
      const selected = heads?.length?heads:Array.from({length:headCount},(_,i)=>i);
      const weights = parameters.weights as number[] | undefined;
      return <fieldset className="docattentionlens-method-weights" key={key}><legend>Head weights</legend>{selected.map((head,index)=><label key={head}>Head {head}<input aria-label={`Weight for head ${head}`} type="number" min={0} max={1000000} step="any" value={weights?.[index]??1}
        onChange={e=>{const value=e.target.valueAsNumber;if(Number.isFinite(value)&&value>=0)set(key,selected.map((_,i)=>i===index?value:weights?.[i]??1));}}/></label>)}<p>Weights are normalized for the mean. At least one must be positive.</p></fieldset>;
    }
    if(key==='roles'&&schema.items?.enum) {
      const roles = (parameters.roles ?? schema.default ?? ['special']) as string[];
      return <fieldset key={key}><legend>Declared target roles</legend>{schema.items.enum.map(role=><label key={role} className="docattentionlens-checkbox"><input type="checkbox" aria-label={`Include ${role} targets`} checked={roles.includes(role)} disabled={roles.length===1&&roles.includes(role)} onChange={e=>set(key,e.target.checked?[...roles,role]:roles.filter(r=>r!==role))}/>{role}</label>)}</fieldset>;
    }
    if(schema.type==='integer'||schema.type==='number') {
      const value = Number(parameters[key] ?? schema.default ?? schema.minimum ?? 0);
      const maximum = key==='k'?maxTargets??schema.maximum:schema.maximum;
      const label = key==='k'?'Top-k targets':schema.title??key.replaceAll('_',' ');
      return <label key={key}>{label}<input aria-label={label} type="number" min={schema.minimum} max={maximum} step={schema.type==='integer'?1:'any'} value={value}
        onChange={e=>{const next=e.target.valueAsNumber;if(Number.isFinite(next)&&next>=(schema.minimum??-Infinity)&&next<=(maximum??Infinity)&&(!Number.isInteger(value)||schema.type!=='integer'||Number.isInteger(next)))set(key,next);}}/>
        {/* A stored value can outlive a source change that lowers the bound; say so instead of sending a failing request. */}
        {maximum!==undefined&&value>maximum&&<small role="alert">{label} {value} exceeds this source's {maximum} valid targets. <button className="docattentionlens-link-button" onClick={()=>set(key,maximum)}>Use {maximum}</button></small>}</label>;
    }
    if(schema.type==='string'&&schema.enum)return <label key={key}>{schema.title??key.replaceAll('_',' ')}<select aria-label={key} value={String(parameters[key]??schema.default??schema.enum[0])} onChange={e=>set(key,e.target.value)}>{schema.enum.map(v=><option key={v} value={v}>{v.replaceAll('_',' ')}</option>)}</select></label>;
    if(schema.type==='boolean')return <label key={key} className="docattentionlens-checkbox"><input type="checkbox" checked={Boolean(parameters[key]??schema.default)} onChange={e=>set(key,e.target.checked)}/>{schema.title??key.replaceAll('_',' ')}</label>;
    if(schema.type==='array'&&schema.items?.enum)return <fieldset key={key}><legend>{schema.title??key.replaceAll('_',' ')}</legend>{schema.items.enum.map(v=>{
      const chosen=(parameters[key]??schema.default??[]) as string[];
      return <label key={v} className="docattentionlens-checkbox"><input type="checkbox" checked={chosen.includes(v)} onChange={e=>set(key,e.target.checked?[...chosen,v]:chosen.filter(x=>x!==v))}/>{v}</label>;
    })}</fieldset>;
    if(schema.type==='array'&&(schema.items?.type==='integer'||schema.items?.type==='number'))return <NumericArray key={key} label={schema.title??key.replaceAll('_',' ')} value={(parameters[key]??schema.default??[]) as number[]} schema={schema} onChange={values=>set(key,values)}/>;
    return null;
  }).filter(Boolean);
  const ownership = method.method_id==='attention.element_allocation';
  if(!controls.length&&!fixed.length&&!ownership) return null;
  const title = TITLES[method.method_id] ?? `${method.name} settings`;
  return <section aria-label={title} className="docattentionlens-method-settings"><h2>{title}</h2>
    {controls}
    {(!!fixed.length||ownership)&&<dl className="docattentionlens-fixed-settings">
      {fixed.map(([key,schema])=><Fragment key={key}><dt>{key.replaceAll('_',' ')}</dt><dd>{FIXED_VALUES[String(schema.const)]??String(schema.const).replaceAll('_',' ')}</dd></Fragment>)}
      {ownership&&<><dt>element ownership</dt><dd>saved patch-to-element map on the original {domain} domain; background, unreviewed geometry and non-image mass kept as residuals</dd></>}
    </dl>}
  </section>;
}
