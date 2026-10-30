"""Verify immutable paired evidence with GPU stopped and services restarted."""
import argparse
import json
import os
from pathlib import Path
import subprocess
import time
from urllib.request import urlopen
from experiment_smoke import verify

BASE='http://127.0.0.1:8000/api/v1'
COMPOSE=['docker','compose','-f','infra/compose.yaml','--profile','gpu']
ROOT=Path(__file__).resolve().parents[1]


def main():
    parser=argparse.ArgumentParser();parser.add_argument('--browser',action='store_true');args=parser.parse_args()
    path=ROOT/'var/verification/experiment-smoke.json';metadata=json.loads(path.read_text(encoding='utf-8'))
    identifier=metadata['result_id'];reference=verify(BASE,identifier)
    try:
        subprocess.run(COMPOSE+['stop','worker-gpu'],cwd=ROOT,check=True)
        for _ in range(20):
            if not json.load(urlopen(BASE+'/generation/plan',timeout=5))['available']:break
            time.sleep(1)
        else:raise RuntimeError('GPU execution is still advertised')
        assert verify(BASE,identifier)==reference
        print('Paired manifest, source/variant images, captures, rows and consumed pixels reopen with GPU stopped',flush=True)
        if args.browser:
            subprocess.run(['npm.cmd' if os.name=='nt' else 'npm','run','test:e2e','--prefix','frontend','--','experiments.spec.ts','--grep','measured layout comparison'],cwd=ROOT,check=True)
        subprocess.run(COMPOSE+['restart','metadata','api','worker-cpu','frontend'],cwd=ROOT,check=True)
        for _ in range(40):
            try:
                assert verify(BASE,identifier)==reference
                break
            except OSError:time.sleep(1)
        else:raise RuntimeError('Paired artifacts did not recover')
        metadata['recovery']={'gpu_stopped':True,'browser_offline_checked':args.browser,'services_restarted':True,'manifest_images_capture_hashes_rows_and_consumed_pixels_unchanged':True}
        path.write_text(json.dumps(metadata,indent=2,ensure_ascii=False),encoding='utf-8')
        print('Paired evidence survived metadata/API/frontend restart unchanged',flush=True)
    finally:subprocess.run(COMPOSE+['start','worker-gpu'],cwd=ROOT,check=True)


if __name__=='__main__':main()
