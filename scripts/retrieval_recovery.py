"""Reopen actual persisted evidence with GPU stopped and after service restart."""
import json
from pathlib import Path
import subprocess
import time
import urllib.request

BASE='http://localhost:8080/api/v1'
COMPOSE=['docker','compose','-f','infra/compose.yaml','--profile','gpu']
ROOT=Path(__file__).resolve().parents[1]


def get(path):
    with urllib.request.urlopen(BASE+path,timeout=30) as response:return json.load(response)


def main():
    source=ROOT/'var/verification/retrieval-smoke.json'
    evidence=json.loads(source.read_text(encoding='utf-8'))
    path='/retrieval/captures/'+evidence['capture_id']
    capture=get(path)
    row_path=path+'/rows?page_id='+capture['pages'][0]['page_id']+'&query_index=0'
    row=get(row_path)
    try:
        subprocess.run(COMPOSE+['stop','worker-gpu'],cwd=ROOT,check=True)
        for _ in range(20):
            if not get('/retrieval/plan')['available']:break
            time.sleep(1)
        else:raise RuntimeError('GPU worker still advertised available')
        assert get(path)==capture and get(row_path)==row
        assert get('/investigations/'+evidence['investigation_id'])['view_state']['retrieval_capture_id']==capture['capture_id']
        print('Saved evidence verified with GPU disconnected',flush=True)
        subprocess.run(COMPOSE+['restart','metadata','api','worker-cpu','frontend'],cwd=ROOT,check=True)
        for _ in range(40):
            try:
                if get(path)==capture and get(row_path)==row:break
            except (OSError,ValueError):pass
            time.sleep(1)
        else:raise RuntimeError('Saved capture did not recover after restart')
        evidence['recovery']={'gpu_disconnected':True,'services_restarted':True,'manifest_and_row_unchanged':True}
        source.write_text(json.dumps(evidence,indent=2)+'\n',encoding='utf-8')
        print('Capture manifest and complete numeric row survived restart unchanged',flush=True)
    finally:
        subprocess.run(COMPOSE+['start','worker-gpu'],cwd=ROOT,check=True)


if __name__=='__main__':main()
