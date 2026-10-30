"""Verify live CPU execution and saved-record recovery; never touch raw inputs."""
import argparse
import hashlib
import json
import subprocess
import time
import urllib.error
import urllib.request
from datetime import UTC, datetime
from pathlib import Path
from uuid import uuid4

from docattentionlens.methods.registry import list_methods


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--base-url", default="http://127.0.0.1:8080")
    parser.add_argument("--restart", action="store_true")
    parser.add_argument("--output", default="var/verification/smoke.json")
    args = parser.parse_args()
    root = Path(__file__).resolve().parents[1]
    base = args.base_url.rstrip("/")

    def request(path, body=None):
        encoded = json.dumps(body).encode() if body is not None else None
        req = urllib.request.Request(base + path, data=encoded, headers={"Content-Type": "application/json"} if encoded else {})
        with urllib.request.urlopen(req, timeout=5) as response:
            return json.load(response)

    def wait_for(check, timeout=60):
        deadline = time.monotonic() + timeout
        while time.monotonic() < deadline:
            try:
                result = check()
                if result:
                    return result
            except (urllib.error.URLError, TimeoutError, ConnectionError):
                pass
            time.sleep(0.5)
        raise RuntimeError("Deployment did not become ready within the timeout")

    assert request("/health/live")["status"] == "live"
    wait_for(lambda: request("/health/ready")["status"] == "ready")
    wait_for(lambda: any(w["available"] for w in request("/api/v1/runtime")["workers"]))
    catalogue = request("/api/v1/methods")
    expected = {(m.method_id, m.variant_id): (m.status, m.implementation_version) for m in list_methods()}
    actual = {(m["method_id"], m["variant_id"]): (m["status"], m["implementation_version"]) for m in catalogue["items"]}
    assert actual == expected
    specification = {"kind": "integrity_check", "idempotency_key": f"smoke-{uuid4().hex}", "inputs": {"text": "DocAttentionLens CPU check"}}
    accepted = request("/api/v1/jobs", specification)
    assert request("/api/v1/jobs", specification)["job_id"] == accepted["job_id"]
    job_path = f"/api/v1/jobs/{accepted['job_id']}"

    def completed():
        job = request(job_path)
        if job["state"] in ("failed", "cancelled"):
            raise RuntimeError(f"CPU check ended with {job['state']}")
        return job if job["state"] == "completed" else None

    job = wait_for(completed)
    receipt_path = f"/api/v1/artifacts/operational_receipt/{job['result_id']}"
    receipt = request(receipt_path)
    assert receipt["output"]["sha256"] == hashlib.sha256(b"DocAttentionLens CPU check").hexdigest()
    assert receipt["output"]["byte_count"] == len(b"DocAttentionLens CPU check")
    paths = ["/attention-tool", "/toolbox/retrieval.maxsim", f"/artifacts/jobs/{job['job_id']}", f"/artifacts/operational_receipt/{job['result_id']}"]
    for path in paths:
        with urllib.request.urlopen(base + path, timeout=5) as response:
            assert 'id="root"' in response.read().decode("utf-8")
    if args.restart:
        subprocess.run(["docker", "compose", "-f", "infra/compose.yaml", "restart", "metadata", "api", "worker-cpu", "frontend"], cwd=root, check=True, timeout=60)
        wait_for(lambda: request("/health/ready")["status"] == "ready")
        assert request(job_path)["state"] == "completed"
        assert request(receipt_path) == receipt
        wait_for(lambda: any(w["available"] for w in request("/api/v1/runtime")["workers"]))
        recovery = request("/api/v1/jobs", {**specification, "idempotency_key": f"smoke-restart-{uuid4().hex}"})

        def recovered_worker():
            current = request(f"/api/v1/jobs/{recovery['job_id']}")
            if current["state"] in ("failed", "cancelled"):
                raise RuntimeError(f"Post-restart CPU check ended with {current['state']}")
            return current if current["state"] == "completed" else None

        recovered = wait_for(recovered_worker)
        recovered_receipt = request(f"/api/v1/artifacts/operational_receipt/{recovered['result_id']}")
        assert recovered_receipt["output"] == receipt["output"]
    output = {"checked_at": datetime.now(UTC).isoformat(), "application_version": "0.1.0", "purpose": "deployment_smoke_verification", "job_id": job["job_id"], "receipt_id": job["result_id"], "cpu_job_completed": True, "idempotency_verified": True, "route_refresh_verified": paths, "restart_reopen_verified": args.restart, "restart_worker_execution_verified": args.restart, "registered_methods": len(catalogue["items"]), "scientific_methods_executed": 0}
    destination = root / args.output
    destination.parent.mkdir(parents=True, exist_ok=True)
    destination.write_text(json.dumps(output, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(output, indent=2))


if __name__ == "__main__":
    main()
