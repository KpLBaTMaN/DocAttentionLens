"""Verify saved generator evidence, including consumed pixels, without a GPU."""
import argparse
from hashlib import sha256
import json
import os
from pathlib import Path
import subprocess
import time
import urllib.request

BASE='http://localhost:8080/api/v1'
COMPOSE=['docker','compose','-f','infra/compose.yaml','--profile','gpu']
ROOT=Path(__file__).resolve().parents[1]


def get(path):
    with urllib.request.urlopen(BASE+path,timeout=30) as response:return json.load(response)


def image(path):
    with urllib.request.urlopen('http://localhost:8080'+path,timeout=30) as response:return sha256(response.read()).hexdigest()


def main():
    parser=argparse.ArgumentParser();parser.add_argument('--browser',action='store_true');args=parser.parse_args()
    source=ROOT/'var/verification/generation-smoke.json'
    evidence=json.loads(source.read_text(encoding='utf-8'))
    path='/generation/captures/'+evidence['capture_id']
    capture=get(path)
    selected=evidence['view_state']
    row_path=path+f"/rows?step={selected['attention_step']}&layer={selected['attention_layer']}&head={selected['attention_head']}"
    row=get(row_path)
    def verify():
        assert get(path)==capture and get(row_path)==row
        assert get('/investigations/'+evidence['investigation_id'])['view_state']==evidence['view_state']
        assert all(image(p['preview_url'])==evidence['image_sha256'][p['page_id']] for p in capture['context']['pages'])
        assert get('/retrieval/captures/'+capture['retrieval_capture_id'])['capture_id']==capture['retrieval_capture_id']
    try:
        subprocess.run(COMPOSE+['stop','worker-gpu'],cwd=ROOT,check=True)
        for _ in range(20):
            if not get('/generation/plan')['available']:break
            time.sleep(1)
        else:raise RuntimeError('GPU worker still advertised available')
        verify();print('Answer, context, attention, image payloads and saved selection reopen with GPU disconnected',flush=True)
        if args.browser:
            subprocess.run(['npm.cmd' if os.name=='nt' else 'npm','run','test:e2e','--prefix','frontend','--','generation.spec.ts','--grep','real generator evidence'],cwd=ROOT,check=True)
        subprocess.run(COMPOSE+['restart','metadata','api','worker-cpu','frontend'],cwd=ROOT,check=True)
        for _ in range(40):
            try:verify();break
            except (OSError,ValueError):pass
            time.sleep(1)
        else:raise RuntimeError('Generator artifacts did not recover after restart')
        evidence['recovery']={'gpu_disconnected':True,'browser_offline_checked':args.browser,'services_restarted':True,'manifest_row_pixels_and_view_unchanged':True}
        source.write_text(json.dumps(evidence,indent=2)+'\n',encoding='utf-8')
        print('Saved generator evidence survived metadata/API/frontend restart unchanged',flush=True)
    finally:subprocess.run(COMPOSE+['start','worker-gpu'],cwd=ROOT,check=True)


if __name__=='__main__':main()
