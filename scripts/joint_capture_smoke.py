"""Verify three-reading capture on pinned retrievers against saved separate-pass vectors.

Uses the local API and real GPU worker; saves immutable captures and a JSON report.
Existing captures are never changed. Each model reuses the inputs of its newest
same-revision, same-adapter capture. Run after rebuilding the API/GPU worker.
"""
import argparse
import json
from datetime import UTC, datetime
from pathlib import Path
from uuid import uuid4

from model_matrix import Api, run_job
from docattentionlens.retrieval.contracts import RETRIEVER_PROFILES


def verify(api, registration, baseline):
    profile = RETRIEVER_PROFILES[registration]
    old_encoding = next((e for e in baseline.get('attention_encodings', []) if e['pass_kind'] != 'query'), None)
    page_id = old_encoding['page_id'] if old_encoding else baseline['pages'][0]['page_id']
    old_page = next(p for p in baseline['pages'] if p['page_id'] == page_id)
    identifier, error = run_job(api, '/retrieval/runs', {
        'case_id': baseline['case_id'], 'page_ids': [page_id], 'attention_page_id': page_id,
        'capture_attention': True, 'capture_joint_attention': True,
        'model_registration_id': registration, 'idempotency_key': 'joint-readings-' + uuid4().hex,
    }, 600, registration)
    if error:
        raise RuntimeError(error)
    capture = api.get('/retrieval/captures/' + identifier)
    encodings = {e['pass_kind']: e for e in capture['attention_encodings']}
    assert set(encodings) == {'query', 'document', 'joint'}
    assert capture['provenance']['joint_capture_policy'] == 'retain_document_v1'
    assert len(capture['provenance']['joint_attention_source_sha256']) == 64
    assert capture['pages'][0]['score'] == old_page['score'], 'separate-pass score changed'
    old_arrays = {a['name']: a for a in baseline['arrays']}
    new_arrays = {a['name']: a for a in capture['arrays']}
    assert old_arrays[baseline['query_array']]['sha256'] == new_arrays[capture['query_array']]['sha256'], 'query vectors changed'
    assert old_arrays[old_page['vector_array']]['sha256'] == new_arrays[capture['pages'][0]['vector_array']]['sha256'], 'page vectors changed'
    page = capture['pages'][0]
    visual = [p for p in page['positions'] if p['bbox']]
    joint_visual = [p for p in encodings['joint']['positions'] if p['bbox']]
    assert [p['bbox'] for p in visual] == [p['bbox'] for p in joint_visual]
    assert [page['ownership'][p['index']] for p in visual] == [encodings['joint']['ownership'][p['index']] for p in joint_visual]
    retained = sum(a['bytes'] for a in capture['arrays'] if a['name'].startswith('attn_'))
    plan = api.get('/retrieval/plan?model_registration_id=' + registration + '&capture_joint_attention=true')
    assert retained < plan['maximum_retained_attention_bytes']
    assert capture['validation']['attention']['candidate_ranking_equal']
    if profile['causal']:
        assert set(capture['validation']['attention']['embedding_kernel_errors']) == {'query', 'document'}
    sampled = []
    for kind, encoding in encodings.items():
        assert (encoding['layer_count'], encoding['query_head_count']) == (profile['layers'], profile['query_heads'])
        assert len(encoding['layer_arrays']) == profile['layers']
        assert all(name in new_arrays for name in encoding['layer_arrays'])
        positions = capture['query_positions'] if kind == 'query' else encoding['positions'] or page['positions']
        words = [p['index'] for p in positions if p['role'] == 'query_text' and p['eligible'] and encoding['attention_mask'][p['index']]]
        source = words[-1] if words else max(i for i, active in enumerate(encoding['attention_mask']) if active)
        for layer, head in ((0, 0), (profile['layers'] - 1, profile['query_heads'] - 1)):
            result = api.call('/retrieval/captures/' + identifier + '/attention', {
                'encoding_id': encoding['encoding_id'], 'source_index': source,
                'method_id': 'attention.row', 'parameters': {'layer': layer, 'head': head}})
            assert result[0] == 200, result
            row = result[1]['row']
            assert row['source_index'] == source and row['predicted_index'] is None
            assert abs(sum(row['values']) - 1) < .02
            assert abs(sum(a['mass'] for a in row['allocations']) - row['total_mass']) < 1e-6
            assert all(v == 0 for i, v in enumerate(row['values']) if not encoding['attention_mask'][i] or encoding['causal'] and i > source)
            sampled.append({'reading': kind, 'source': source, 'layer': layer, 'head': head, 'image_mass': row['image_mass']})
        if kind == 'joint':
            result = api.call('/retrieval/captures/' + identifier + '/attention', {
                'encoding_id': encoding['encoding_id'], 'source_index': source,
                'method_id': 'attention.rollout', 'parameters': {'layer': profile['layers'] - 1}})
            assert result[0] == 200, result
            row = result[1]['row']
            assert row['quantity'] == 'rollout_connectivity' and row['image_mass'] > 0
            assert abs(sum(row['values']) - 1) < 1e-6
    return {'capture_id': identifier, 'baseline_capture_id': baseline['capture_id'], 'page_id': page_id,
            'score': page['score'], 'vectors_and_score_equal': True, 'retained_attention_bytes': retained,
            'readings': {k: e['position_count'] for k, e in encodings.items()}, 'sampled_rows': sampled,
            'joint_rollout': 'passed', 'patch_mapping': 'equal'}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--base', default='http://127.0.0.1:8000')
    parser.add_argument('--models', nargs='+', choices=list(RETRIEVER_PROFILES), default=list(RETRIEVER_PROFILES))
    args = parser.parse_args()
    api = Api(args.base)
    records, before = [], ''
    for _ in range(5):
        result = api.get('/retrieval/captures?limit=25' + ('&before=' + before if before else ''))
        records.extend(result['items'])
        before = result.get('next_cursor')
        if not before:
            break
    report = {'created_at': datetime.now(UTC).isoformat(), 'models': {}}
    for registration in args.models:
        profile = RETRIEVER_PROFILES[registration]
        baseline = next((c for c in records if c['provenance'].get('model_id') == profile['model_id']
                         and c['provenance'].get('model_revision') == profile['revision']
                         and c['provenance'].get('adapter_version') == profile['adapter_version']), None)
        try:
            if baseline is None:
                raise ValueError('No saved same-revision separate-pass baseline')
            report['models'][registration] = {'status': 'passed', **verify(api, registration, baseline)}
            print(registration + ': passed', flush=True)
        except Exception as exc:
            report['models'][registration] = {'status': 'failed', 'reason': str(exc)}
            print(registration + ': ' + str(exc), flush=True)
    target = Path(__file__).resolve().parents[1] / 'var/verification/joint-capture-smoke.json'
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_text(json.dumps(report, indent=2) + '\n', encoding='utf-8')
    return int(any(v['status'] != 'passed' for v in report['models'].values()))


if __name__ == '__main__':
    raise SystemExit(main())
