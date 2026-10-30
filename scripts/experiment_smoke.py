"""Verify a real paired result. Acquisition/review happen through DocAttentionLens.

Uses existing metadata in var/verification/experiment-smoke.json by default.
No simulated model output and no implicit reviewer declaration.
"""
import argparse
from hashlib import sha256
from io import BytesIO
import json
from pathlib import Path
import time
from urllib.request import urlopen
from PIL import Image

ROOT=Path(__file__).resolve().parents[1]


def verify(base, identifier):
    def read(path): return urlopen(base+path,timeout=120).read()
    def get(path): return json.loads(read(path))
    r=get('/experiments/'+identifier)
    assert r['record_kind']=='experiment_result' and r['review'] is not None
    assert len(r['arms'])==4 and len(r['outcomes'])==4 and len(r['comparisons'])==3
    images={};rows={};consumed={};captures={}
    source_page_id=r['protocol']['page_id']
    for arm,outcome in zip(r['arms'],r['outcomes'],strict=True):
        assert arm['name']==outcome['arm'] and outcome['input_sha256']==arm['asset_sha256']
        raw=read(f"/experiments/{identifier}/arms/{arm['name']}/image")
        assert sha256(raw).hexdigest()==arm['asset_sha256']
        im=Image.open(BytesIO(raw)).convert('RGB')
        assert im.size==(arm['page']['width'],arm['page']['height'])
        assert sha256(im.tobytes()).hexdigest()==arm['pixel_sha256']
        images[arm['name']]=sha256(raw).hexdigest()
        if 'retrieval' in outcome:
            ret=get(f"/experiments/{identifier}/arms/{arm['name']}/retrieval")
            assert ret['content_sha256']==outcome['retrieval']['sha256'] and ret['case_id']==arm['case']['case_id']
            assert ret['query']==arm['case']['query']
            page=next(p for p in ret['pages'] if p['page_id']==arm['page']['page_id'])
            assert page['image_sha256']==arm['asset_sha256'] and page['reconstruction_error']<=ret['validation']['score_tolerance_absolute']
            captures[arm['name']+':retrieval']=ret['content_sha256']
            assert get('/retrieval/captures/'+ret['capture_id'])==ret
        if 'generation' in outcome:
            cap=get(f"/experiments/{identifier}/arms/{arm['name']}/generation")
            assert cap['content_sha256']==outcome['generation']['sha256'] and cap['case_id']==arm['case']['case_id']
            assert cap['answer']==outcome['answer'] and cap['validation']['capture_on_off_token_equality']
            assert cap['context']['evaluation_labels_consumed'] is False and cap['context']['ocr_consumed'] is False
            captures[arm['name']+':generation']=cap['content_sha256']
            assert get('/generation/captures/'+cap['capture_id'])==cap
            row=get(f"/experiments/{identifier}/arms/{arm['name']}/attention?step=0&layer=35&head=15")
            assert abs(row['total_mass']-1)<.02 and row['allocation_error']<1e-8
            assert row['source_index']+1==row['predicted_index']==row['target_count']
            rows[arm['name']]=row
            for p in cap['context']['pages']:
                raw=read(f"/experiments/{identifier}/arms/{arm['name']}/input-image/{p['page_id']}")
                assert Image.open(BytesIO(raw)).size==(p['resized_width'],p['resized_height'])
                consumed[arm['name']+':'+p['page_id']]=sha256(raw).hexdigest()
                assert sha256(read(p['preview_url'].removeprefix('/api/v1'))).hexdigest()==sha256(raw).hexdigest()
            if outcome['changed_page_supplied']:
                supplied=next(p for p in cap['context']['pages'] if p['page_id']==arm['page']['page_id'])
                assert supplied['image_sha256']==arm['asset_sha256']
            if r['protocol']['stage']=='generation_fixed_evidence' and arm['name']!='baseline':
                assert 'scores' not in outcome and 'retrieval' not in outcome
                assert outcome['fixed_candidate_selection']['record_kind']=='fixed_context_selection'
    assert r['arms'][0]['pixel_sha256']==r['arms'][1]['pixel_sha256']
    if 'score_deltas' in r['comparisons'][0]: assert all(abs(v)<=1e-4 for v in r['comparisons'][0]['score_deltas'].values())
    source=r['provenance']['source_case']
    assert source['case_id']==r['source_case_id'] and source_page_id in source['page_ids']
    return {'result_id':identifier,'result_sha256':r['content_sha256'],'manifest':r,'images':images,'rows':rows,'consumed_image_hashes':consumed,'capture_hashes':captures}


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--result');parser.add_argument('--base',default='http://127.0.0.1:8000/api/v1')
    args=parser.parse_args();path=ROOT/'var/verification/experiment-smoke.json'
    metadata=json.loads(path.read_text(encoding='utf-8')) if path.exists() else {}
    identifier=args.result or metadata.get('result_id')
    if not identifier and metadata.get('job_id'):
        deadline=time.monotonic()+1800
        while time.monotonic()<deadline:
            job=json.load(urlopen(args.base+'/jobs/'+metadata['job_id'],timeout=10))
            if job['state']=='completed':identifier=job['output_resource_id'];break
            if job['state'] in ('failed','cancelled'):raise RuntimeError(job['state'])
            print('Waiting for actual paired job:',job['state'],flush=True);time.sleep(5)
    if not identifier:parser.error('Supply --result or finish an experiment in DocAttentionLens first')
    evidence=verify(args.base,identifier)
    path.parent.mkdir(parents=True,exist_ok=True)
    path.write_text(json.dumps({**metadata,**evidence},indent=2,ensure_ascii=False),encoding='utf-8')
    print('Real paired manifest, consumed inputs, lineages, score controls and attention verified:',identifier)


if __name__=='__main__':main()
