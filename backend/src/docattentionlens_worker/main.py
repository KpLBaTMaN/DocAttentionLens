import argparse
import logging
import signal
from datetime import UTC, datetime
from threading import Event, Thread
from uuid import uuid4

from docattentionlens_api.config import Settings
from docattentionlens_api.repositories.jobs import JobRepository
from docattentionlens_api.repositories.receipts import FileReceiptStore
from docattentionlens_worker.dispatch import execute
from docattentionlens_worker.health import worker_id_file
from docattentionlens import __version__
from docattentionlens.artifacts.records import OperationalReceipt
from docattentionlens.generation.contracts import GENERATORS as GENERATOR_PROFILES
from docattentionlens_api.repositories.documents import SnapshotStore

logger = logging.getLogger("docattentionlens.worker")


# Known adapter failures and the researcher-facing message each publishes instead of the generic one.
SAFE_FAILURES = {
    "Processed context exceeds plan or contains unsupported padding; no truncation applied": "The processed prompt exceeds 1,536 positions. Supply fewer pages or shorten the question or system instruction; no attention was published.",
    "Checkpoint weight loading is incomplete or incompatible": "The checkpoint has missing or incompatible weights. Upload a complete checkpoint for the supported architecture; no attention was published.",
    "Checkpoint bytes differ from the registered manifest": "Checkpoint integrity verification failed; no attention was published.",
    "Generator requires a BF16-capable CUDA device with 12 GiB free": "The generator needs a BF16 CUDA device with 12 GiB free memory. Your completed retrieval and inputs remain available.",
    **{f"Generator requires a BF16-capable CUDA device with {gib} GiB free": f"This generator needs a BF16 CUDA device with {gib} GiB free memory. Your completed retrieval and inputs remain available."
       for gib in sorted({p.required_free_gib for p in GENERATOR_PROFILES.values()} - {12})},
    "ColQwen2 requires a BF16-capable CUDA device": "ColQwen2 needs a BF16-capable CUDA device; no capture was published.",
    "Processor or vision ordering differs from the validated Qwen2-VL mapping": "The ColQwen2 processor or vision ordering differs from the validated mapping; no capture was published.",
    "ColQwen2.5 requires a BF16-capable CUDA device": "ColQwen2.5 needs a BF16-capable CUDA device; no capture was published.",
    "Processor or vision ordering differs from the validated Qwen2.5-VL mapping": "The ColQwen2.5 processor or vision ordering differs from the validated mapping; no capture was published.",
    "At least 12 GiB free GPU memory is required for this bounded plan": "ColQwen2.5 needs 12 GiB free GPU memory. Saved captures and inputs remain available.",
    "Runtime vision merge/window inversion differs from row-major mapping": "The vision tower's patch order did not invert to the validated row-major mapping; no capture was published.",
    "Actual page pixel mapping differs from independent reconstruction": "Page pixels did not match the independent patch reconstruction; no capture was published.",
    "Causal retriever attention includes future keys": "Captured retriever attention was not causal as expected; no attention was published.",
    "Capture-on/off token or coverage equivalence failed": "Capture validation failed: instrumentation did not preserve the output or expected coverage; no attention was published.",
    "Capture-on/off logits equivalence failed": "Capture validation failed: prediction logits differed with instrumentation; no attention was published.",
    "Full prefill attention is not a BF16 (1, heads, n, n) matrix": "Full attention capture failed: the decoder did not return BF16 prefill matrices of the expected shape; no attention was published.",
    "Full prefill attention has non-finite or negative weights": "Full attention capture failed: prefill attention contained non-finite or negative weights; no attention was published.",
    "Full prefill attention coverage or normalization failed": "Full attention capture failed: not every layer was kept or rows were not normalized; no attention was published.",
    "Full attention triangle differs from the retained prefill rows": "Full attention capture failed: the stored matrices did not reproduce the retained rows exactly; no attention was published.",
    "Attention values are not exact BF16; refusing lossy storage": "Full attention capture failed: values were not exact BF16, so lossless storage was refused; no attention was published.",
    "Causal prefill attention includes future keys": "Full attention capture failed: prefill attention was not causal as expected; no attention was published.",
    "Full attention capture is missing prediction steps": "Full attention capture failed: not every generated answer token's attention row was kept; no attention was published.",
    "Retained decode row differs from the full-attention axes": "Full attention capture failed: a generated step's attention row did not cover its causal positions; no attention was published.",
    "Retained decode row differs from normalized causal attention": "Full attention capture failed: a generated step's attention row was not normalized; no attention was published.",
    "Full-attention job predates every-step retention": "This full-attention run was requested before every answer step was kept. Submit the run again; no attention was published.",
}


def failure_message(exc):
    message = SAFE_FAILURES.get(str(exc))
    # Kernel-equivalence checks carry their measured values, so they cannot be listed verbatim above.
    if message is None and str(exc).startswith("Retriever eager/SDPA "):
        message = f"Attention extraction and scoring disagreed beyond the calibrated tolerance: {exc}. No capture was published."
    return message


def import_dataset(settings, job):
    """Snapshot a registered dataset source; returns the dataset row and the receipt output."""
    from docattentionlens_api.services.dataset_sources import read_source
    snapshot = read_source(settings.data_root, job["specification"].get("source"), job["specification"].get("operation_version"))
    snapshot_store = SnapshotStore(settings.artifact_root)
    snapshot_hash = snapshot_store.put(snapshot)
    snapshot_store.read(snapshot.summary.dataset_id, snapshot_hash)
    dataset = {"summary": snapshot.summary.model_dump(mode="json"), "content_hash": snapshot_hash}
    output = {"dataset_id": snapshot.summary.dataset_id, "dataset_version": snapshot.summary.dataset_version, "dataset_snapshot_sha256":snapshot_hash,
        "json_snapshot_sha256":snapshot.summary.json_snapshot_sha256,"adapter_version":snapshot.summary.adapter_version,
        "counts": snapshot.summary.counts, "quality_counts": snapshot.summary.quality_counts, "purpose": "dataset_registration"}
    return dataset, output


def capture_retrieval(settings, job, interrupted):
    """Run the specified retriever on the frozen case pages; returns the capture row and the receipt output."""
    from docattentionlens_api.repositories.documents import DocumentRepository
    from docattentionlens_api.repositories.captures import CaptureStore
    from importlib import import_module
    from docattentionlens.retrieval.contracts import retriever_for_adapter
    spec = job["specification"]
    documents = DocumentRepository(settings)
    case = documents.case(spec["case_id"])
    # The specification names the adapter; each built-in retriever pins its own revision.
    profile = retriever_for_adapter(spec["adapter_version"])
    if profile is None or spec["model_revision"]!=profile["revision"] or spec["case_sha256"]!=case.content_sha256:
        raise ValueError("Retrieval specification differs from frozen inputs")
    capture = import_module("docattentionlens.retrieval."+profile["module"]).capture
    snapshot = documents.snapshot(case.dataset_id)
    page_ids = spec["page_ids"]
    if not 1<=len(page_ids)<=8 or len(set(page_ids))!=len(page_ids) or not set(page_ids)<=set(case.page_ids):
        raise ValueError("Invalid retrieval candidate set")
    inputs = []
    for pid in page_ids:
        page = next(p for p in snapshot.pages if p.page_id==pid)
        inputs.append((page,documents.image_bytes(snapshot,page),[e for e in snapshot.elements if e.page_id==pid]))
    options = {}
    if spec.get("capture_attention"):
        if spec.get("attention_adapter_version") != profile["attention_adapter_version"]:
            raise ValueError("Retrieval attention adapter version differs")
        options = {"capture_attention":True,"attention_page_id":spec["attention_page_id"]}
        if spec.get("capture_joint_attention"):
            options["capture_joint_attention"] = True
            if spec.get("joint_answer_source"):
                options["joint_answer_source"] = spec["joint_answer_source"]
    record, arrays = capture(case,inputs,settings.model_cache,interrupted,**options)
    record = record.model_copy(update={"dependencies":[*record.dependencies,job["job_id"]],"provenance":{**record.provenance,"job_id":job["job_id"],"attempt_id":job["attempt_id"],"input_hash":job["input_hash"]}})
    capture_store = CaptureStore(settings.artifact_root)
    capture_hash = capture_store.put(record,arrays)
    stored = capture_store.read(record.capture_id,capture_hash)
    for ref in stored.arrays:
        capture_store.array(stored,ref.name)
    capture_metadata = {"capture_id":record.capture_id,"case_id":case.case_id,"content_hash":capture_hash}
    output = {**capture_metadata,"purpose":"retrieval_execution","candidate_pages":page_ids,"quantity":"late_interaction_similarity"}
    return capture_metadata, output


def capture_generation(settings, repository, job, interrupted):
    """Run the specified generator on the frozen case or retrieval inputs; returns the capture row and the receipt output."""
    from docattentionlens_api.repositories.documents import DocumentRepository
    from docattentionlens_api.repositories.captures import CaptureRepository
    from docattentionlens_api.repositories.generations import GenerationStore
    from docattentionlens.generation.qwen import capture as generate
    from docattentionlens.generation.contracts import GENERATORS, SYSTEM_PROMPT
    from docattentionlens_api.services.imports import ModelStore
    from docattentionlens_api.services.model_catalogue import generator_profile
    spec=job["specification"]
    documents=DocumentRepository(settings)
    case=documents.case(spec["case_id"])
    retrieval=CaptureRepository(repository).get(spec["retrieval_capture_id"]) if spec.get("retrieval_capture_id") else None
    checkpoint = ModelStore(settings.artifact_root).get(spec.get("model_registration_id", "builtin"))
    # Built-in profiles load their pinned hub revision; uploads load their verified files.
    uploaded = checkpoint["registration_id"] not in GENERATORS
    if uploaded:
        checkpoint["path"] = str(settings.artifact_root / "models" / checkpoint["registration_id"] / "files")
    profile = generator_profile(checkpoint)
    if profile is None or spec["adapter_version"]!=profile.adapter_version or spec["model_revision"]!=checkpoint["revision"] or spec["case_sha256"]!=case.content_sha256 or (retrieval and (retrieval.case_id!=case.case_id or spec["retrieval_sha256"]!=retrieval.content_sha256)):
        raise ValueError("Generator specification differs from frozen inputs")
    page_ids=spec["page_ids"]
    if not 1<=len(page_ids)<=2 or len(set(page_ids))!=len(page_ids) or not set(page_ids)<=(set(p.page_id for p in retrieval.pages) if retrieval else set(case.page_ids)):
        raise ValueError("Invalid generator context selection")
    snapshot=documents.snapshot(case.dataset_id)
    inputs=[]
    for pid in page_ids:
        page=next(p for p in snapshot.pages if p.page_id==pid)
        inputs.append((page,documents.image_bytes(snapshot,page),[e for e in snapshot.elements if e.page_id==pid]))
    full_attention=spec.get("capture_full_attention",False)
    image_token_limit=spec.get("image_token_limit",512)
    if not isinstance(full_attention,bool) or image_token_limit not in (512,1024) or (image_token_limit!=512 and len(page_ids)!=1):
        raise ValueError("Generator specification differs from frozen inputs")
    # Full-attention jobs keep every generated step. An earlier eight-step full-attention
    # specification is refused rather than silently answered with a different capture.
    if full_attention and spec.get("full_attention_steps")!="all":
        raise ValueError("Full-attention job predates every-step retention")
    if not full_attention and "full_attention_steps" in spec:
        raise ValueError("Generator specification differs from frozen inputs")
    record,arrays=generate(case,retrieval,inputs,settings.model_cache,interrupted,checkpoint=checkpoint if uploaded else None,system_prompt=spec.get("system_prompt", SYSTEM_PROMPT),profile=profile,full_attention=full_attention,image_token_limit=image_token_limit)
    if full_attention!=(record.full_attention is not None):
        raise ValueError("Full prefill attention coverage or normalization failed")
    if full_attention and (record.schema_version!="0.8.0" or [s.step for s in record.steps]!=list(range(len(record.output_token_ids)))
            or any(s.array not in arrays for s in record.steps)):
        raise ValueError("Full attention capture is missing prediction steps")
    record=record.model_copy(update={"dependencies":[*record.dependencies,job["job_id"]],"provenance":{**record.provenance,"job_id":job["job_id"],"attempt_id":job["attempt_id"],"input_hash":job["input_hash"]}})
    store=GenerationStore(settings.artifact_root)
    digest=store.put(record,arrays)
    stored=store.read(record.capture_id,digest)
    from docattentionlens.generation.inspection import attention_row
    for ref in stored.arrays:
        values=store.array(stored,ref.name)
        if ref.name.startswith("step_"):
            selected=next(s for s in stored.steps if s.array==ref.name)
            attention_row(stored,values,selected.step,0,0)
    capture_metadata={"capture_id":record.capture_id,"case_id":case.case_id,"retrieval_capture_id":retrieval.capture_id if retrieval else None,"content_hash":digest,"kind":"generation_capture"}
    output={**capture_metadata,"purpose":"generation_execution","supplied_pages":page_ids,"quantity":"post_softmax_attention_weight","answer":record.answer}
    return capture_metadata, output


def run_once(settings: Settings, worker_id: str) -> bool:
    repository = JobRepository(settings)
    repository.heartbeat(worker_id)
    job = repository.claim(worker_id)
    if job is None:
        return False
    done = Event()
    lost = Event()

    def renew():
        while not done.wait(settings.lease_seconds / 3):
            try:
                repository.heartbeat(worker_id)
                if not repository.renew(job["job_id"], job["attempt_id"]):
                    lost.set()
                    break
            except Exception:
                lost.set()
                logger.error("heartbeat_failed job=%s attempt=%s", job["job_id"], job["attempt_id"])
                break

    thread = Thread(target=renew, daemon=True)
    thread.start()
    try:
        dataset = capture_metadata = experiment_metadata = None
        if job["kind"] == "dataset_import":
            dataset, output = import_dataset(settings, job)
        elif job["kind"] == "retrieval_capture":
            capture_metadata, output = capture_retrieval(settings, job, lost.is_set)
        elif job["kind"] == "generation_capture":
            capture_metadata, output = capture_generation(settings, repository, job, lost.is_set)
        elif job["kind"] == "layout_experiment":
            from docattentionlens_worker.experiments import execute_experiment
            experiment_metadata=execute_experiment(settings,repository,job,lost.is_set)
            output={**experiment_metadata,"purpose":"controlled_layout_execution"}
        elif job["kind"] == "document_experiment":
            from docattentionlens_worker.document_studies import execute_document_study
            experiment_metadata=execute_document_study(settings,repository,job,lost.is_set)
            output={**experiment_metadata,"purpose":"controlled_document_execution"}
        elif job["kind"] == "attention_intervention":
            from docattentionlens_worker.attention_interventions import execute_attention_experiment
            experiment_metadata=execute_attention_experiment(settings,repository,job,lost.is_set)
            output={**experiment_metadata,"purpose":"direct_attention_intervention"}
        else:
            output = execute(job["specification"])
        if lost.is_set():
            return True
        receipt = OperationalReceipt(
            record_id=f"receipt_{uuid4().hex}", created_at=datetime.now(UTC),
            producer_version=__version__, dependencies=[job["job_id"]],
            provenance={"job_id": job["job_id"], "attempt_id": job["attempt_id"], "input_hash": job["input_hash"], "purpose": output["purpose"] if capture_metadata or experiment_metadata else "dataset_registration" if dataset else "infrastructure_verification"},
            output=output,
        )
        store = FileReceiptStore(settings.artifact_root)
        checksum = store.put(receipt)
        store.read(receipt.record_id, checksum)
        published = repository.finish(job, receipt, checksum, dataset=dataset, capture=capture_metadata, experiment=experiment_metadata)
        logger.info("job=%s attempt=%s published=%s", job["job_id"], job["attempt_id"], published)
    except Exception as exc:
        logger.exception("execution_failed job=%s attempt=%s", job["job_id"], job["attempt_id"])
        repository.finish(job, None, None, "execution_failed", error_message=failure_message(exc))
    finally:
        done.set()
        thread.join(timeout=5)
    return True


def main() -> None:
    parser = argparse.ArgumentParser(description="DocAttentionLens CPU/GPU worker")
    parser.add_argument("--once", action="store_true", help="Claim and execute at most one job")
    args = parser.parse_args()
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(name)s %(message)s")
    settings = Settings.from_env()
    repository = JobRepository(settings)
    repository.ready()
    FileReceiptStore(settings.artifact_root).probe()
    worker_id = f"worker_{uuid4().hex}"
    worker_id_file().write_text(worker_id, encoding="utf-8")
    stop = Event()
    signal.signal(signal.SIGTERM, lambda *_: stop.set())
    signal.signal(signal.SIGINT, lambda *_: stop.set())
    while not stop.is_set():
        try:
            run_once(settings, worker_id)
        except Exception:
            logger.error("metadata_unavailable worker=%s", worker_id)
        if args.once:
            break
        stop.wait(1)


if __name__ == "__main__":
    main()
