"""Read-only integrity check of a real saved evaluation; never submits reviews."""
import argparse
from hashlib import sha256
import json
from pathlib import Path
from urllib.request import urlopen
from docattentionlens.datasets.adapter import canonical
from docattentionlens.evaluation.analysis import result_measurements
from docattentionlens.experiments.contracts import ExperimentRecord

ROOT = Path(__file__).resolve().parents[1]


def verify(base, identifier):
    def get(path):
        with urlopen(base + path, timeout=30) as response:
            return json.load(response)

    plan = get('/evaluation/studies/' + identifier)
    report = get('/evaluation/studies/' + identifier + '/report')
    digest = sha256(canonical({k: v for k, v in plan.items() if k != 'content_sha256'})).hexdigest()
    assert plan['content_sha256'] == report['study_sha256'] == digest
    evidence = report['evidence']
    for record in evidence:
        assert sha256(canonical({k: v for k, v in record.items() if k != 'content_sha256'})).hexdigest() == record['content_sha256']
    observations = []
    for case in report['model_evaluation']['cases']:
        frozen = next(c for c in plan['frozen_cases'] if c['protocol_id'] == case['protocol_id'])
        for observation in case['observations']:
            result = ExperimentRecord.model_validate(get('/experiments/' + observation['result_id']))
            assert result.provenance['protocol_sha256'] == frozen['protocol_sha256']
            assert result.content_sha256 == observation['result_sha256']
            reviews = [r['payload'] for r in evidence if r['record_kind'] == 'answer_review'
                       and r['payload']['result_id'] == result.experiment_id]
            assert result_measurements(result, frozen, reviews) == observation['measurements']
            captures = []
            for outcome in result.outcomes:
                if 'generation' in outcome:
                    capture = get('/generation/captures/' + outcome['generation']['capture_id'])
                    assert capture['content_sha256'] == outcome['generation']['sha256']
                    assert capture['answer'] == outcome['answer']
                    assert capture['output_token_ids'] == outcome['output_token_ids']
                    captures.append(capture['content_sha256'])
            observations.append({'result_id': result.experiment_id, 'result_sha256': result.content_sha256,
                                 'generation_capture_hashes': captures})
    count = report['model_evaluation']['completed_acquisitions']
    assert count == len(observations)
    assert len({o['result_id'] for o in observations}) == count
    return {'study_id': identifier, 'study_sha256': digest, 'observations': observations,
            'report_sha256': sha256(canonical(report)).hexdigest(),
            'completed_acquisitions': count, 'utility_trials': report['utility']['recorded_trials']}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--study')
    parser.add_argument('--base', default='http://127.0.0.1:8080/api/v1')
    args = parser.parse_args()
    path = ROOT / 'var/verification/evaluation-smoke.json'
    metadata = json.loads(path.read_text(encoding='utf-8')) if path.exists() else {}
    identifier = args.study or metadata['study_id']
    result = verify(args.base, identifier)
    metadata.update(result)
    path.write_text(json.dumps(metadata, indent=2) + '\n', encoding='utf-8')
    print(json.dumps(result, indent=2))


if __name__ == '__main__':
    main()
