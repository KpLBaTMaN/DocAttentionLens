"""Real saved-input/controlled-run engineering checks; no research-validity claim."""
import argparse,json,time,urllib.request,urllib.error
from pathlib import Path

parser=argparse.ArgumentParser()
parser.add_argument("--base",default="http://127.0.0.1:8000/api/v1")
parser.add_argument("--capture",default="generation_5db62722a76640e4bd14acd308263c22")
parser.add_argument("--run-experiments",action="store_true")
parser.add_argument("--experiment-methods",nargs="+",choices=("document.image_text_contrast","document.element_masking","document.element_steering"),default=["document.image_text_contrast","document.element_masking","document.element_steering"])
parser.add_argument("--steering-strengths",nargs="+",type=float,default=[0,1])
parser.add_argument("--output",default="var/document-methods-smoke.json")
args=parser.parse_args(); report={"purpose":"engineering verification on real captures; not scientific experiment results","analyses":[],"studies":[]}


def request(path,body=None):
    try:
        req=urllib.request.Request(args.base+path,json.dumps(body).encode() if body is not None else None,{"Content-Type":"application/json"})
        return json.load(urllib.request.urlopen(req,timeout=120))
    except urllib.error.HTTPError as exc: raise RuntimeError(f"HTTP {exc.code}: {exc.read().decode()}") from None


def save():
    path=Path(args.output);path.parent.mkdir(parents=True,exist_ok=True);path.write_text(json.dumps(report,indent=2)+"\n",encoding="utf-8")


methods={m['method_id']:m for m in request('/methods')['items']}
capture=request('/generation/captures/'+args.capture)
case=request('/cases/'+capture['case_id'])
elements=[]
for page in capture['context']['selected_page_ids']:
    elements+=request('/pages/'+page+'/elements?limit=200')['items']
for identifier in ('attention.element_type_profile','attention.designated_support','attention.reading_order','attention.layout_sinks','attention.section_rollup',
                   'attention.model_input_legibility','attention.modality_budget_depth','attention.evidence_modality_matrix','attention.evidence_funnel','attention.spatial_prior_map','attention.head_specialisation'):
    body={'generation_capture_id':args.capture,'method_id':identifier,'variant_id':methods[identifier]['variant_id'],'parameters':{}}
    if 'corpus_sample_manifest' in methods[identifier]['requires']:body['sample_capture_ids']=[args.capture]
    result=request('/methods/analyses',body)
    assert request('/methods/analyses/'+result['analysis_id'])['content_sha256']==result['content_sha256']
    report['analyses'].append({'method':identifier,'analysis_id':result['analysis_id'],'input_manifest':result['input_manifest']})
    print(identifier,'saved and reopened',flush=True)
save()
if args.run_experiments:
    # Pick the smallest mapped, geometrically reviewed element to make a bounded
    # same-size control feasible. Annotation selection remains evaluation-only.
    owned={i for p in capture['positions'] for i in p['ownership'] if i.startswith('element_')}
    candidates=sorted((e for e in elements if e['element_id'] in owned and not e['geometry_flags']),
        key=lambda e:(e['bbox']['x2']-e['bbox']['x1'])*(e['bbox']['y2']-e['bbox']['y1']))
    if not candidates:raise RuntimeError('No mapped element for controlled tests')
    for identifier in args.experiment_methods:
        parameters={} if identifier.endswith('contrast') else {'element_ids':[candidates[0]['element_id']]}
        if identifier.endswith('steering'):parameters['strengths']=args.steering_strengths
        study=request('/document-studies',{'generation_capture_id':args.capture,'method_id':identifier,'variant_id':methods[identifier]['variant_id'],'parameters':parameters})
        entry={'method':identifier,'prepared_id':study['study_id'],'arms':study['arms']};report['studies'].append(entry);save()
        job=request('/document-studies/'+study['study_id']+'/runs',{'idempotency_key':'smoke-'+study['study_id']})
        entry['job_id']=job['job_id'];save();print(identifier,'running',len(study['arms']),'arms',flush=True)
        deadline=time.monotonic()+900
        while time.monotonic()<deadline:
            detail=request('/jobs/'+job['job_id'])
            if detail['state'] in ('completed','failed','cancelled'):break
            time.sleep(2)
        entry['state']=detail['state'];save()
        if detail['state']!='completed':raise RuntimeError(str(detail.get('error_message') or detail['state']))
        completed=request('/document-studies/'+detail['output_resource_id'])
        entry['result_id']=completed['study_id'];entry['outcomes']=completed['outcomes'];save()
        result=request('/methods/analyses',{'generation_capture_id':completed['outcomes'][0]['capture_id'],'method_id':identifier,
            'variant_id':methods[identifier]['variant_id'],'parameters':study['parameters'],'experiment_result_id':completed['study_id']})
        entry['analysis_id']=result['analysis_id'];save();print(identifier,'completed and analysed',flush=True)
print('Engineering checks saved to',args.output,flush=True)
