"""The GPU worker reports free memory as CUDA gives it to the process, the figure adapters check before loading."""
import sys
from types import SimpleNamespace as NS
import pytest
from docattentionlens_api.repositories.jobs import gpu_memory

MIB = 2**20


def fake_torch(free, reserved):
    cuda = NS(mem_get_info=lambda: (free, 24564 * MIB), get_device_name=lambda: "Fixture GPU", memory_reserved=lambda: reserved)
    return NS(cuda=cuda)


def test_free_and_reclaimable_memory_come_from_the_cuda_runtime(monkeypatch):
    monkeypatch.setitem(sys.modules, "torch", fake_torch(17074 * MIB + 5, 5490 * MIB))
    assert gpu_memory() == ("Fixture GPU", 17074, 5490)


def test_a_missing_cuda_runtime_raises_for_the_heartbeat_to_report(monkeypatch):
    def unavailable():
        raise RuntimeError("No CUDA GPUs are available")
    monkeypatch.setitem(sys.modules, "torch", NS(cuda=NS(mem_get_info=unavailable)))
    with pytest.raises(RuntimeError):
        gpu_memory()
