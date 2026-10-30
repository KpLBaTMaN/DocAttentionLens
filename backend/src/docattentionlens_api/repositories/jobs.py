"""PostgreSQL queue with attempt leases and publication fencing."""

from __future__ import annotations

import json
from hashlib import sha256
from typing import Any
from uuid import uuid4

import psycopg
from psycopg.rows import dict_row
from docattentionlens.generation.contracts import ADAPTER_VERSIONS as GENERATOR_ADAPTERS
from docattentionlens.retrieval.contracts import RETRIEVER_PROFILES
from psycopg.types.json import Jsonb

from docattentionlens_api.config import Settings
from docattentionlens_api.migrate import SCHEMA_VERSION
from docattentionlens import __version__
from docattentionlens.artifacts.records import OperationalReceipt


class ConflictError(Exception):
    pass


def input_hash(specification: dict[str, Any]) -> str:
    """Idempotency identity of a job specification: SHA-256 of its sorted, compact JSON."""
    return sha256(json.dumps(specification, sort_keys=True, separators=(",", ":"), ensure_ascii=False).encode()).hexdigest()


def gpu_memory() -> tuple[str, int, int]:
    """GPU name, free MiB and this worker's reclaimable MiB, as CUDA reports them to this process.

    Free memory is the figure every adapter checks before loading (torch.cuda.mem_get_info),
    so a plan admits exactly what a job can load. The single worker releases its resident
    adapter before switching, so its own Torch reservation is reclaimable; another process's is not.
    """
    import torch
    free, _total = torch.cuda.mem_get_info()
    return torch.cuda.get_device_name(), free // 2**20, torch.cuda.memory_reserved() // 2**20


class JobRepository:
    def __init__(self, settings: Settings):
        self.settings = settings

    def connect(self):
        return psycopg.connect(self.settings.database_url, row_factory=dict_row, connect_timeout=3)

    def ready(self) -> None:
        with self.connect() as conn:
            if not conn.execute("SELECT 1 FROM schema_migrations WHERE version=%s", (SCHEMA_VERSION,)).fetchone():
                raise RuntimeError("Metadata migration required")
            conn.execute("SELECT 1 FROM jobs LIMIT 1")

    @staticmethod
    def event(conn, job_id: str, state: str, attempt_id: str | None = None) -> None:
        conn.execute("INSERT INTO job_events(job_id,state,attempt_id) VALUES (%s,%s,%s)", (job_id, state, attempt_id))

    def submit(self, key: str, specification: dict[str, Any]) -> dict:
        digest = input_hash(specification)
        with self.connect() as conn:
            row = conn.execute(
                "INSERT INTO jobs(job_id,idempotency_key,input_hash,specification,kind,resource_class,state) VALUES (%s,%s,%s,%s,%s,%s,'queued') ON CONFLICT (idempotency_key) DO NOTHING RETURNING *",
                (f"job_{uuid4().hex}", key, digest, Jsonb(specification), specification["kind"], "gpu" if specification["kind"] in ("retrieval_capture", "generation_capture", "layout_experiment", "document_experiment", "attention_intervention") else "cpu"),
            ).fetchone()
            if row:
                self.event(conn, row["job_id"], "queued")
            else:
                row = conn.execute("SELECT * FROM jobs WHERE idempotency_key=%s", (key,)).fetchone()
                if row["input_hash"] != digest:
                    raise ConflictError("Idempotency key already names a different input specification.")
            return row

    def existing_submission(self, key: str, specification: dict[str, Any]) -> dict | None:
        digest=input_hash(specification)
        with self.connect() as conn:
            row=conn.execute("SELECT * FROM jobs WHERE idempotency_key=%s",(key,)).fetchone()
        if row and row["input_hash"]!=digest:
            raise ConflictError("Idempotency key already names a different input specification.")
        return row

    def get(self, job_id: str) -> dict | None:
        with self.connect() as conn:
            row = conn.execute("SELECT * FROM jobs WHERE job_id=%s", (job_id,)).fetchone()
            if row:
                row["attempts"] = conn.execute("SELECT * FROM attempts WHERE job_id=%s ORDER BY started_at,attempt_id", (job_id,)).fetchall()
                row["events"] = conn.execute("SELECT * FROM job_events WHERE job_id=%s ORDER BY event_id", (job_id,)).fetchall()
            return row

    def list(self, limit: int, before: str | None) -> list[dict]:
        with self.connect() as conn:
            return conn.execute(
                "SELECT * FROM jobs WHERE (%s::text IS NULL OR (created_at,job_id) < (SELECT created_at,job_id FROM jobs WHERE job_id=%s)) ORDER BY created_at DESC,job_id DESC LIMIT %s",
                (before, before, limit),
            ).fetchall()

    def cancel(self, job_id: str) -> dict | None:
        with self.connect() as conn:
            row = conn.execute("SELECT * FROM jobs WHERE job_id=%s FOR UPDATE", (job_id,)).fetchone()
            if not row:
                return None
            if row["state"] in ("queued", "running"):
                state = "cancelled" if row["state"] == "queued" else "cancellation_requested"
                row = conn.execute("UPDATE jobs SET state=%s,updated_at=clock_timestamp() WHERE job_id=%s RETURNING *", (state, job_id)).fetchone()
                self.event(conn, job_id, state, row["attempt_id"])
            return row

    def heartbeat(self, worker_id: str) -> None:
        capabilities = {}
        if self.settings.worker_resource_class == "gpu":
            try:
                name, free, reclaimable = gpu_memory()
                capabilities = {"adapter":"colpali_transformers_v1","retrieval_attention_adapter":"colpali_language_attention_v1","adapters":[*(p["adapter_version"] for p in RETRIEVER_PROFILES.values()),*GENERATOR_ADAPTERS],"retrieval_attention_adapters":[p["attention_adapter_version"] for p in RETRIEVER_PROFILES.values()],"gpu":name,"free_mib":free,"reclaimable_mib":reclaimable,"required_free_mib":8192}
                capabilities["document_experiment_operations"]=["document_studies_v1"]
                capabilities["attention_intervention_operations"]=["attention_interventions_v1"]
            except (ImportError, RuntimeError, AssertionError):
                capabilities = {"error":"gpu_preflight_unavailable"}
        with self.connect() as conn:
            conn.execute("INSERT INTO workers(worker_id,resource_class,heartbeat_at,producer_version,capabilities) VALUES (%s,%s,clock_timestamp(),%s,%s) ON CONFLICT (worker_id) DO UPDATE SET heartbeat_at=excluded.heartbeat_at,capabilities=excluded.capabilities", (worker_id, self.settings.worker_resource_class, __version__, Jsonb(capabilities)))

    def workers(self) -> list[dict]:
        with self.connect() as conn:
            return conn.execute("SELECT *, heartbeat_at > clock_timestamp() - interval '10 seconds' AS available FROM workers ORDER BY heartbeat_at DESC LIMIT 50").fetchall()

    def claim(self, worker_id: str) -> dict | None:
        with self.connect() as conn:
            # Expired attempts are retained. SKIP LOCKED allows independent workers.
            expired = conn.execute("SELECT job_id,attempt_id,state FROM jobs WHERE state IN ('running','cancellation_requested') AND lease_until <= clock_timestamp() FOR UPDATE SKIP LOCKED").fetchall()
            for old in expired:
                conn.execute("UPDATE attempts SET state='expired',finished_at=clock_timestamp(),error_code='lease_expired' WHERE attempt_id=%s", (old["attempt_id"],))
                state = "cancelled" if old["state"] == "cancellation_requested" else "queued"
                conn.execute("UPDATE jobs SET state=%s,attempt_id=NULL,worker_id=NULL,lease_until=NULL,updated_at=clock_timestamp() WHERE job_id=%s", (state, old["job_id"]))
                self.event(conn, old["job_id"], state, old["attempt_id"])
            row = conn.execute("SELECT * FROM jobs WHERE state='queued' AND resource_class=%s ORDER BY created_at,job_id LIMIT 1 FOR UPDATE SKIP LOCKED", (self.settings.worker_resource_class,)).fetchone()
            if not row:
                return None
            attempt_id = f"attempt_{uuid4().hex}"
            conn.execute("INSERT INTO attempts(attempt_id,job_id,worker_id,state) VALUES (%s,%s,%s,'running')", (attempt_id, row["job_id"], worker_id))
            row = conn.execute("UPDATE jobs SET state='running',attempt_id=%s,worker_id=%s,lease_until=clock_timestamp()+(%s * interval '1 second'),updated_at=clock_timestamp() WHERE job_id=%s RETURNING *", (attempt_id, worker_id, self.settings.lease_seconds, row["job_id"])).fetchone()
            self.event(conn, row["job_id"], "running", attempt_id)
            return row

    def renew(self, job_id: str, attempt_id: str) -> bool:
        with self.connect() as conn:
            return conn.execute("UPDATE jobs SET lease_until=clock_timestamp()+(%s * interval '1 second') WHERE job_id=%s AND attempt_id=%s AND state='running' AND lease_until > clock_timestamp() RETURNING job_id", (self.settings.lease_seconds, job_id, attempt_id)).fetchone() is not None

    def finish(self, job: dict, receipt: OperationalReceipt | None, checksum: str | None, error_code: str | None = None, dataset: dict | None = None, capture: dict | None = None, experiment: dict | None = None, error_message: str | None = None) -> bool:
        with self.connect() as conn:
            row = conn.execute("SELECT * FROM jobs WHERE job_id=%s AND attempt_id=%s AND state IN ('running','cancellation_requested') AND lease_until > clock_timestamp() FOR UPDATE", (job["job_id"], job["attempt_id"])).fetchone()
            if not row:
                return False  # An expired/superseded attempt cannot publish.
            state = "cancelled" if row["state"] == "cancellation_requested" else ("failed" if error_code else "completed")
            result_id = None
            output_id = None
            if state == "completed":
                if receipt is None or checksum is None:
                    raise ValueError("Completed publication requires a verified receipt")
                result_id = receipt.record_id
                conn.execute("INSERT INTO receipts(record_id,job_id,content_hash,manifest) VALUES (%s,%s,%s,%s)", (result_id, job["job_id"], checksum, Jsonb(receipt.model_dump(mode="json"))))
                if dataset:
                    output_id = dataset["summary"]["dataset_id"]
                    conn.execute("INSERT INTO datasets(dataset_id,content_hash,summary) VALUES (%s,%s,%s) ON CONFLICT (dataset_id) DO NOTHING", (output_id, dataset["content_hash"], Jsonb(dataset["summary"])))
                    existing = conn.execute("SELECT content_hash FROM datasets WHERE dataset_id=%s", (output_id,)).fetchone()
                    if existing["content_hash"] != dataset["content_hash"]:
                        raise ValueError("Dataset identity collision")
                if capture:
                    output_id = capture["capture_id"]
                    if capture.get("kind")=="generation_capture":
                        conn.execute("INSERT INTO generation_captures(capture_id,case_id,retrieval_capture_id,job_id,content_hash) VALUES (%s,%s,%s,%s,%s)", (output_id,capture["case_id"],capture["retrieval_capture_id"],job["job_id"],capture["content_hash"]))
                    else:
                        conn.execute("INSERT INTO retrieval_captures(capture_id,case_id,job_id,content_hash) VALUES (%s,%s,%s,%s)", (output_id,capture["case_id"],job["job_id"],capture["content_hash"]))
                if experiment and experiment.get("kind")=="attention_intervention":
                    output_id=experiment["experiment_id"]
                    conn.execute("INSERT INTO attention_experiments(experiment_id,source_case_id,kind,method_id,content_hash,job_id) VALUES (%s,%s,'result',%s,%s,%s)",
                                 (output_id,experiment["source_case_id"],experiment["method_id"],experiment["content_hash"],job["job_id"]))
                    for ref in experiment["capture_refs"]:
                        conn.execute("INSERT INTO attention_experiment_captures(capture_id,experiment_id,kind,arm,content_hash) VALUES (%s,%s,%s,%s,%s)",
                                     (ref["capture_id"],output_id,ref["kind"],ref["arm"],ref["sha256"]))
                elif experiment and experiment.get("kind")=="document_study":
                    output_id=experiment["experiment_id"]
                    conn.execute("INSERT INTO document_studies(study_id,source_case_id,kind,method_id,content_hash,job_id) VALUES (%s,%s,'result',%s,%s,%s)",
                                 (output_id,experiment["source_case_id"],experiment["method_id"],experiment["content_hash"],job["job_id"]))
                    for ref in experiment["capture_refs"]:
                        conn.execute("INSERT INTO document_study_captures(capture_id,study_id,arm,content_hash) VALUES (%s,%s,%s,%s)",(ref["capture_id"],output_id,ref["arm"],ref["sha256"]))
                elif experiment:
                    output_id=experiment["experiment_id"]
                    conn.execute("INSERT INTO experiments(experiment_id,source_case_id,record_kind,content_hash,name,job_id) VALUES (%s,%s,'experiment_result',%s,%s,%s)",(output_id,experiment["source_case_id"],experiment["content_hash"],experiment["name"],job["job_id"]))
                    for ref in experiment.get("capture_refs",[]):
                        conn.execute("INSERT INTO experiment_captures(capture_id,experiment_id,kind,arm,content_hash) VALUES (%s,%s,%s,%s,%s)",(ref["capture_id"],output_id,ref["kind"],ref["arm"],ref["sha256"]))
            conn.execute("UPDATE attempts SET state=%s,finished_at=clock_timestamp(),error_code=%s WHERE attempt_id=%s", (state, error_code, job["attempt_id"]))
            conn.execute("UPDATE jobs SET state=%s,result_id=%s,error_code=%s,error_message=%s,lease_until=NULL,updated_at=clock_timestamp() WHERE job_id=%s", (state, result_id, error_code, (error_message or "Execution failed; inspect worker logs using the job and attempt IDs.") if error_code else None, job["job_id"]))
            self.event(conn, job["job_id"], state, job["attempt_id"])
            conn.execute("UPDATE jobs SET output_resource_id=%s,output_resource_kind=%s WHERE job_id=%s", (output_id, "attention_intervention" if experiment and experiment.get("kind")=="attention_intervention" and output_id else "document_study" if experiment and experiment.get("kind")=="document_study" and output_id else "experiment_result" if experiment and output_id else capture.get("kind","retrieval_capture") if capture and output_id else "dataset" if output_id else None, job["job_id"]))
            return True

    def receipt(self, record_id: str) -> dict | None:
        with self.connect() as conn:
            return conn.execute("SELECT * FROM receipts WHERE record_id=%s", (record_id,)).fetchone()
