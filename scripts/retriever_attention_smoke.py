"""Real GPU capture, saved analysis and role-isolated save/reopen verification."""
import json
import time
from pathlib import Path
from uuid import uuid4
from retrieval_smoke import request

ROOT=Path(__file__).resolve().parents[1]

def main():
    previous=request('/retrieval/captures?limit=1')['items'][0]
    case_id=previous['case_id']
    pages=[p['page_id'] for p in previous['pages']][:2]
    job=request('/retrieval/runs',{'case_id':case_id,'page_ids':pages,
        'capture_attention':True,'attention_page_id':pages[0],'idempotency_key':str(uuid4())})
    print('Retrieval attention job '+job['job_id'],flush=True)
    started=time.monotonic()
    while time.monotonic()-started<600:
        detail=request('/jobs/'+job['job_id'])
        if detail['state']=='completed':break
        if detail['state'] in ('failed','cancelled'):raise RuntimeError(detail)
        time.sleep(2)
    else:raise TimeoutError(job['job_id'])
    capture=request('/retrieval/captures/'+detail['output_resource_id'])
    assert len(capture['attention_encodings'])==2
    outputs={}
    for encoding in capture['attention_encodings']:
        assert encoding['layer_count']==18 and encoding['query_head_count']==8
        for method in request('/retrieval/attention-methods'):
            result=request('/retrieval/captures/'+capture['capture_id']+'/attention',{
                'encoding_id':encoding['encoding_id'],'source_index':0,'method_id':method['method_id'],'parameters':{'layer':17}})
            rows=([result['row']] if result['row'] else [])+[c['row'] for c in result['comparisons']]
            for row in rows:
                assert row['source_index']==0 and row['predicted_index'] is None
                assert row['target_count']==encoding['position_count']
                assert row['allocation_error']<1e-8
                if row['quantity']=='post_softmax_attention_weight':assert abs(row['total_mass']-1)<.02
            outputs[encoding['encoding_id']+'/'+method['method_id']]={'rows':len(rows),'statistics':len(result['statistics'])}
    view={'investigation_model_role':'retrieval','retrieval_capture_id':capture['capture_id'],
        'page_id':pages[0],'retrieval_attention':True,'retrieval_encoding_id':'document',
        'retrieval_source_index':10,'retrieval_attention_method':'attention.entropy','attention_layer':17,'attention_head':7}
    investigation=request('/investigations',{'name':'Retriever internal attention verification','case_id':case_id,'view_state':view})
    reopened=request('/investigations/'+investigation['investigation_id'])
    assert reopened['view_state']['retrieval_source_index']==10
    assert reopened['view_state']['generation_capture_id'] is None
    evidence={'capture_id':capture['capture_id'],'case_id':case_id,'investigation_id':investigation['investigation_id'],
        'elapsed_seconds':round(time.monotonic()-started,2),'validation':capture['validation'],
        'retained_bytes':sum(r['bytes'] for r in capture['arrays']),'encodings':capture['attention_encodings'],'methods':outputs}
    dest=ROOT/'var/verification/retriever-attention.json';dest.parent.mkdir(parents=True,exist_ok=True)
    dest.write_text(json.dumps(evidence,indent=2),encoding='utf-8')
    print(json.dumps({k:v for k,v in evidence.items() if k not in ('encodings','methods')},indent=2),flush=True)

if __name__=='__main__':main()
