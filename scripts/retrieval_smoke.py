"""Acquire real evidence from three original pages, verify, save and reopen."""
import json
import time
import urllib.request
from pathlib import Path
from uuid import uuid4

BASE = "http://localhost:8000"
ROOT = Path(__file__).resolve().parents[1]


def request(path, body=None, method=None):
    data = json.dumps(body).encode() if body is not None else None
    with urllib.request.urlopen(urllib.request.Request(BASE+"/api/v1"+path,data=data,headers={"Content-Type":"application/json"},method=method),timeout=60) as response:
        return json.load(response)


def main():
    dataset=request('/datasets')['items'][0]
    docs=request('/documents?dataset_id='+dataset['dataset_id']+'&domain=annual_reports')['items']
    doc=next(d for d in docs if d['native_id']==1)
    question=request('/questions?document_id='+doc['document_id'])['items'][0]
    case=request('/cases',{'question_id':question['question_id'],'query_variant':'collection_scoped','task_scope':'collection','page_ids':doc['page_ids'][:3]})
    job=request('/retrieval/runs',{'case_id':case['case_id'],'page_ids':case['page_ids'],'idempotency_key':str(uuid4())})
    print('Real retrieval submitted: '+job['job_id'],flush=True)
    deadline=time.monotonic()+1200
    while time.monotonic()<deadline:
        detail=request('/jobs/'+job['job_id'])
        if detail['state'] in ('failed','cancelled'):
            raise RuntimeError(detail)
        if detail['state']=='completed': break
        time.sleep(3)
    else: raise TimeoutError('Model run did not finish')
    capture=request('/retrieval/captures/'+detail['output_resource_id'])
    assert len(capture['pages'])==3
    for page in capture['pages']:
        assert page['reconstruction_error']<=1e-4
        assert abs(sum(page['contributions'].values())-page['score'])<1e-4
        assert len([p for p in page['positions'] if p['bbox']])==1024
        assert all(abs(sum(w.values())-1)<1e-8 for w in page['ownership'])
        row=request('/retrieval/captures/'+capture['capture_id']+'/rows?page_id='+page['page_id']+'&query_index=0')
        assert abs(max(row['similarities'])-row['maximum'])<1e-6
    p=capture['pages'][0]
    diagnostic=request('/retrieval/captures/'+capture['capture_id']+'/fixed-vector-removal',{'page_id':p['page_id'],'removed_position_indices':[p['winner_indices'][0]]})
    assert diagnostic['score_delta']<=1e-4
    investigation=request('/investigations',{'name':'ColPali: real three-page capture','case_id':case['case_id'],'notes':'Real pinned ColPali run. Explicit first-three-page candidates; evaluation labels excluded from model inputs. Retrieval similarity is not attention.','view_state':{'page_id':p['page_id'],'retrieval_capture_id':capture['capture_id'],'query_position':0,'similarity_overlay':True}})
    reopened=request('/investigations/'+investigation['investigation_id'])
    assert reopened['view_state']['retrieval_capture_id']==capture['capture_id']
    evidence={'job_id':job['job_id'],'capture_id':capture['capture_id'],'case_id':case['case_id'],'investigation_id':investigation['investigation_id'],'workspace_url':'http://localhost:8080/attention-tool/'+investigation['investigation_id'],'model_revision':capture['provenance']['model_revision'],'pages':[{'page_number':p['page_number'],'score':p['score'],'reference_score':p['reference_score'],'reconstruction_error':p['reconstruction_error'],'allocated_total':sum(p['contributions'].values())} for p in capture['pages']],'validation':capture['validation'],'fixed_vector_removal':diagnostic}
    dest=ROOT/'var/verification/retrieval-smoke.json';dest.parent.mkdir(parents=True,exist_ok=True);dest.write_text(json.dumps(evidence,indent=2),encoding='utf-8')
    print(json.dumps(evidence,indent=2),flush=True)


if __name__=='__main__': main()
