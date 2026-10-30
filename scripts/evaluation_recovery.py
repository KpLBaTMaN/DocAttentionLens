"""Verify real evaluation reopening with GPU off and API/frontend restarted."""
import argparse
import json
import os
from pathlib import Path
import subprocess
import time
from evaluation_smoke import verify

BASE = 'http://127.0.0.1:8080/api/v1'
COMPOSE = ['docker', 'compose', '-f', 'infra/compose.yaml', '--profile', 'gpu']
ROOT = Path(__file__).resolve().parents[1]


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--browser', action='store_true')
    args = parser.parse_args()
    path = ROOT / 'var/verification/evaluation-smoke.json'
    metadata = json.loads(path.read_text(encoding='utf-8'))
    identifier = metadata['study_id']
    reference = verify(BASE, identifier)
    try:
        subprocess.run(COMPOSE + ['stop', 'worker-gpu'], cwd=ROOT, check=True)
        assert verify(BASE, identifier) == reference
        if args.browser:
            subprocess.run(['npm.cmd' if os.name == 'nt' else 'npm', 'run', 'test:e2e', '--prefix',
                            'frontend', '--', 'evaluation.spec.ts'], cwd=ROOT, check=True)
        subprocess.run(COMPOSE + ['restart', 'api', 'frontend'], cwd=ROOT, check=True)
        for _ in range(40):
            try:
                if verify(BASE, identifier) == reference:
                    break
            except OSError:
                pass
            time.sleep(1)
        else:
            raise RuntimeError('Evaluation did not recover identically after restart')
        metadata['recovery'] = {'gpu_off': True, 'api_frontend_restart': True, 'evidence_unchanged': True}
        path.write_text(json.dumps(metadata, indent=2) + '\n', encoding='utf-8')
        print('Evaluation plan, measurements, report and captures reopened identically with GPU off and after restart')
    finally:
        subprocess.run(COMPOSE + ['start', 'worker-gpu'], cwd=ROOT, check=True)


if __name__ == '__main__':
    main()
