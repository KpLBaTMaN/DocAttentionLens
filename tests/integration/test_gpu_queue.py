import pytest
from dataclasses import replace
from docattentionlens_api.repositories.jobs import JobRepository


@pytest.mark.parametrize("kind",["retrieval_capture","generation_capture","layout_experiment"])
def test_gpu_work_is_not_claimed_by_cpu_and_cancellation_fences_publication(database,kind):
    cpu=JobRepository(database)
    gpu=JobRepository(replace(database,worker_resource_class="gpu"))
    job=cpu.submit("gpu-queue-test",{"kind":kind,"case_id":"queue_fixture"})
    assert job["resource_class"]=="gpu"
    assert cpu.existing_submission("gpu-queue-test",{"kind":kind,"case_id":"queue_fixture"})["job_id"]==job["job_id"]
    assert cpu.claim("cpu-fixture") is None
    gpu.heartbeat("gpu-fixture")
    claimed=gpu.claim("gpu-fixture")
    assert claimed["job_id"]==job["job_id"]
    cpu.cancel(job["job_id"])
    assert gpu.finish(claimed,None,None,error_code="interrupted")
    result=cpu.get(job["job_id"])
    assert result["state"]=="cancelled" and result["output_resource_id"] is None
    assert not gpu.finish(claimed,None,None,error_code="superseded")
