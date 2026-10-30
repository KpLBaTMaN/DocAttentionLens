"""Each built-in generator has its own plan, availability gate and run specification."""
from fastapi.testclient import TestClient
from psycopg.types.json import Jsonb
from docattentionlens_api.main import create_app
from docattentionlens_api.repositories.jobs import JobRepository
from docattentionlens.generation.contracts import QWEN25_VL_3B, QWEN2_VL_2B, QWEN3_VL_8B


def test_builtin_generators_have_their_own_plans_gates_and_run_specifications(registered, snapshot):
    jobs = JobRepository(registered)
    with TestClient(create_app(registered)) as client:
        models = [m for m in client.get('/api/v1/setup/models').json() if m['role'] == 'generation']
        assert [m['registration_id'] for m in models][:3] == ['builtin', 'builtin-qwen2-vl-2b', 'builtin-qwen3-vl-8b']
        assert {m['registration_id']: m['capabilities']['adapter_id'] for m in models} == {
            p.registration_id: p.adapter_version for p in (QWEN25_VL_3B, QWEN2_VL_2B, QWEN3_VL_8B)}
        # The default plan is unchanged; other registrations describe their own model.
        assert client.get('/api/v1/generation/plan').json() == client.get('/api/v1/generation/plan', params={'model_registration_id': 'builtin'}).json()
        plan = client.get('/api/v1/generation/plan', params={'model_registration_id': 'builtin-qwen3-vl-8b'}).json()
        assert (plan['layers'], plan['query_heads'], plan['kv_heads'], plan['required_free_gpu_mib']) == (36, 32, 8, 21 * 1024)
        assert plan['model_revision'] == QWEN3_VL_8B.revision and not plan['available']
        assert client.get('/api/v1/generation/plan', params={'model_registration_id': 'builtin-retrieval'}).status_code == 422
        # A worker advertising only two adapters with 15 GiB serves the 2B model, not the 8B one.
        with jobs.connect() as conn:
            conn.execute("INSERT INTO workers(worker_id,resource_class,heartbeat_at,producer_version,capabilities) VALUES ('profile-fixture','gpu',clock_timestamp(),'test',%s)",
                         (Jsonb({'adapters': [QWEN25_VL_3B.adapter_version, QWEN2_VL_2B.adapter_version], 'free_mib': 15000}),))
        assert client.get('/api/v1/generation/plan', params={'model_registration_id': 'builtin-qwen2-vl-2b'}).json()['available']
        assert not client.get('/api/v1/generation/plan', params={'model_registration_id': 'builtin-qwen3-vl-8b'}).json()['available']
        case = client.post('/api/v1/cases', json={'question_id': snapshot.questions[0].question_id, 'page_ids': [snapshot.documents[0].page_ids[0]]}).json()
        run = {'case_id': case['case_id'], 'page_ids': case['page_ids'][:1]}
        refused = client.post('/api/v1/generation/runs', json={**run, 'model_registration_id': 'builtin-qwen3-vl-8b', 'idempotency_key': 'profile-8b'})
        assert refused.status_code == 503 and refused.json()['error']['code'] == 'model_unavailable'
        accepted = client.post('/api/v1/generation/runs', json={**run, 'model_registration_id': 'builtin-qwen2-vl-2b', 'idempotency_key': 'profile-2b'})
        assert accepted.status_code == 202, accepted.text
        with jobs.connect() as conn:
            spec = conn.execute("SELECT specification FROM jobs WHERE job_id=%s", (accepted.json()['job_id'],)).fetchone()['specification']
        assert (spec['adapter_version'], spec['model_revision'], spec['model_registration_id']) == (
            QWEN2_VL_2B.adapter_version, QWEN2_VL_2B.revision, 'builtin-qwen2-vl-2b')
        # The original default request keeps its replay specification: no registration field.
        default = client.post('/api/v1/generation/runs', json={**run, 'idempotency_key': 'profile-default'})
        with jobs.connect() as conn:
            spec = conn.execute("SELECT specification FROM jobs WHERE job_id=%s", (default.json()['job_id'],)).fetchone()['specification']
        assert spec['adapter_version'] == QWEN25_VL_3B.adapter_version and 'model_registration_id' not in spec
