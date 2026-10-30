import { useCallback, useState } from 'react';
import { Link } from 'react-router-dom';
import { api, unwrap, errorMessage, type Case, type Element } from '../../api/client';
import { useRemote } from '../../app/remote';
import { Message } from '../../app/App';
import type { components } from '../../api/generated/schema';

type DraftRequest=components['schemas']['ExperimentDraftRequest'];

export function ExperimentPanel({record,pageId,elements}:{record:Case;pageId:string;elements:Element[]}) {
  const [group,setGroup]=useState<string[]>([]);
  const [control,setControl]=useState<string[]>([]);
  const [stage,setStage]=useState<DraftRequest['stage']>('retrieval_only');
  const [candidates,setCandidates]=useState<string[]>([pageId,...record.page_ids.filter(id=>id!==pageId).slice(0,2)]);
  const [context,setContext]=useState<string[]>([pageId]);
  const [name,setName]=useState('Intact heading/table movement');
  const [busy,setBusy]=useState(false);const [error,setError]=useState<string>();const [draft,setDraft]=useState<string>();
  const records=useRemote(useCallback(async()=>unwrap(await api.GET('/api/v1/experiments',{params:{query:{case_id:record.case_id,limit:10}}})),[record.case_id,draft]));
  function toggle(ids:string[],id:string){return ids.includes(id)?ids.filter(x=>x!==id):[...ids,id];}
  async function preview(){
    setBusy(true);setError(undefined);
    try{
      const next=unwrap(await api.POST('/api/v1/experiments/drafts',{body:{case_id:record.case_id,page_id:pageId,element_ids:group,non_support_element_ids:control,candidate_page_ids:candidates,fixed_context_page_ids:stage==='generation_fixed_evidence'?context:[pageId],stage,name}}));
      setDraft(next.experiment_id);
    }catch(reason){setError(errorMessage(reason));}finally{setBusy(false);}
  }
  return <section className="docattentionlens-section docattentionlens-experiment-launch"><span className="docattentionlens-kicker">Controlled layout experiments</span><h2>Move an intact heading and table</h2><p>Build real page images, review the group and compare actual model outcomes. Select every element in the rectangular group, including intervening text. The matched control must occupy an equally sized, disjoint rectangle and contain no designated support.</p>
    <details><summary>Prepare a layout experiment on this page</summary><label>Experiment name<input value={name} maxLength={200} onChange={e=>setName(e.target.value)}/></label>
    <label>Tested stage<select aria-label="Tested stage" value={stage} onChange={e=>setStage(e.target.value as DraftRequest['stage'])}><option value="retrieval_only">Retrieval only</option><option value="generation_fixed_evidence">Generation with fixed evidence</option><option value="end_to_end">Retrieval and generation</option></select></label>
    <p>The group moves horizontally into a white extension of the page. All four arms share the same doubled-width canvas and model budgets. Smaller text after resizing is a review requirement.</p>
    <fieldset><legend>Candidate pages (up to eight)</legend>{record.page_ids.map((id,i)=><label key={id}><input type="checkbox" checked={candidates.includes(id)} disabled={!candidates.includes(id)&&candidates.length>=8} onChange={()=>setCandidates(toggle(candidates,id))}/>Page {i+1}</label>)}</fieldset>
    {stage==='generation_fixed_evidence'&&<fieldset><legend>Fixed generator evidence (up to two, in selection order)</legend>{candidates.map(id=><label key={id}><input type="checkbox" checked={context.includes(id)} disabled={!context.includes(id)&&context.length>=2} onChange={()=>setContext(toggle(context,id))}/>Page {record.page_ids.indexOf(id)+1}</label>)}</fieldset>}
    <div className="docattentionlens-table-scroll"><table><thead><tr><th>Native element · current page</th><th>Intact group</th><th>Matched non-support control</th></tr></thead><tbody>{elements.map(e=><tr key={e.element_id}><td>{e.native_id} · {e.element_type}<small>{e.ocr.text?.slice(0,100)}</small></td><td><input type="checkbox" aria-label={`Group element ${e.native_id}`} checked={group.includes(e.element_id)} onChange={()=>setGroup(toggle(group,e.element_id))}/></td><td><input type="checkbox" aria-label={`Control element ${e.native_id}`} disabled={record.support_element_ids.includes(e.element_id)} checked={control.includes(e.element_id)} onChange={()=>setControl(toggle(control,e.element_id))}/></td></tr>)}</tbody></table></div>
    <button className="docattentionlens-button docattentionlens-primary" disabled={busy||group.length<2||!control.length||!candidates.includes(pageId)||!context.length||!name.trim()} onClick={()=>void preview()}>{busy?'Creating actual page variants…':'Build original / variant preview'}</button>
    </details>{error&&<Message error>{error}</Message>}{draft&&<Message>Preview saved. <Link to={`/experiments/${draft}`}>Review the pages and finalize the protocol</Link>.</Message>}
    {records.error&&<Message error>{records.error}</Message>}{!!records.data?.items.length&&<details><summary>Saved experiments for this case</summary><ul>{records.data.items.map(r=><li key={r.experiment_id}><Link to={`/experiments/${r.experiment_id}`}>{r.name} · {r.record_kind.replaceAll('_',' ')}</Link></li>)}</ul></details>}
  </section>;
}
