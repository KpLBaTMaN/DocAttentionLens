"""Baseline/intervention execution with frozen inputs, exact-query controls and fenced publication."""
from datetime import datetime,UTC
from importlib import import_module
from uuid import uuid4
import numpy as np
from docattentionlens_api.repositories.attention_interventions import AttentionExperimentRepository
from docattentionlens_api.repositories.generations import GenerationRepository,GenerationStore
from docattentionlens_api.repositories.captures import CaptureRepository,CaptureStore
from docattentionlens_api.repositories.documents import DocumentRepository
from docattentionlens_api.repositories.experiments import source_inputs
from docattentionlens.experiments.attention_interventions import validate_code,freeze,measurements,VERSION
from docattentionlens.generation.contracts import GENERATORS
from docattentionlens.retrieval.contracts import RETRIEVER_PROFILES

def execute_attention_experiment(settings,jobs,job,interrupted):
    repo=AttentionExperimentRepository(jobs); spec=job["specification"]; prepared=repo.get(spec["experiment_id"])
    validate_code(prepared)
    if prepared.kind!="prepared" or prepared.content_sha256!=spec["experiment_sha256"] or spec["operation_version"]!=VERSION:
        raise ValueError("Attention experiment differs from its immutable plan")
    source=(GenerationRepository(jobs) if prepared.role=="generation" else CaptureRepository(jobs)).get(prepared.source_capture_id)
    if source.content_sha256!=prepared.source_capture_sha256: raise ValueError("Source capture differs from the immutable plan")
    documents=DocumentRepository(settings)
    inputs=source_inputs(documents,prepared.case,prepared.page_ids)
    outcomes=[]; refs=[]; baseline=None; baseline_arrays=None; source_deltas={}
    for name in ("baseline","intervention"):
        if interrupted(): raise RuntimeError("Attention experiment interrupted")
        intervention=prepared.intervention if name=="intervention" else None
        if prepared.role=="generation":
            from docattentionlens.generation.qwen import capture
            profile=GENERATORS[prepared.model_registration_id]
            if profile.revision!=prepared.model_revision: raise ValueError("Pinned generator revision changed")
            retrieval=CaptureRepository(jobs).get(source.retrieval_capture_id) if source.retrieval_capture_id else None
            record,arrays=capture(prepared.case,retrieval,inputs,settings.model_cache,interrupted,
                profile=profile,system_prompt=prepared.provenance["system_prompt"],intervention=intervention)
            source_fingerprint=source.validation.get("prediction_logits_sha256")
            if name=="baseline" and (record.output_token_ids!=source.output_token_ids or
                (source_fingerprint is not None and record.validation["prediction_logits_sha256"]!=source_fingerprint)):
                raise ValueError("Unmodified intervention baseline does not reproduce source tokens/logits")
            kind="generation"; store=GenerationStore(settings.artifact_root)
            outcome={"name":name,"answer":record.answer,"output_token_ids":record.output_token_ids}
        else:
            profile=RETRIEVER_PROFILES[prepared.model_registration_id]
            if profile["revision"]!=prepared.model_revision: raise ValueError("Pinned retriever revision changed")
            capture=import_module("docattentionlens.retrieval."+profile["module"]).capture
            record,arrays=capture(prepared.case,inputs,settings.model_cache,interrupted,force_eager=True,intervention=intervention)
            if record.query_positions!=source.query_positions or [p.positions for p in record.pages]!=[p.positions for p in source.pages]:
                raise ValueError("Frozen retriever input positions differ on replay")
            if name=="baseline":
                source_deltas={p.page_id:p.score-next(s.score for s in source.pages if s.page_id==p.page_id) for p in record.pages}
                # Adapters declare their calibrated eager-versus-SDPA score tolerance; ColPali's stays 0.1.
                tolerance=record.validation.get("eager_score_kernel_tolerance",.1)
                if any(abs(v)>tolerance for v in source_deltas.values()):
                    raise ValueError(f"Eager baseline exceeds the recorded scorer kernel tolerance {tolerance:.4f}: score differences {source_deltas}")
            elif not np.array_equal(arrays[record.query_array],baseline_arrays[baseline.query_array]):
                raise ValueError("Query encoding changed between baseline and intervention")
            kind="retrieval"; store=CaptureStore(settings.artifact_root)
            outcome={"name":name,"pages":[{"page_id":p.page_id,"score":p.score} for p in record.pages]}
        record=record.model_copy(update={"dependencies":[*record.dependencies,prepared.experiment_id,job["job_id"]],
            "provenance":{**record.provenance,"attention_experiment_id":prepared.experiment_id,"arm":name,"job_id":job["job_id"],
                "attempt_id":job["attempt_id"],"intervention":intervention,"inputs_unchanged":True}})
        digest=store.put(record,arrays)
        stored=store.read(record.capture_id,digest)
        for ref in stored.arrays: store.array(stored,ref.name)
        outcome.update(capture_id=record.capture_id,capture_sha256=digest)
        outcomes.append(outcome); refs.append({"capture_id":record.capture_id,"sha256":digest,"kind":kind,"arm":name})
        if name=="baseline": baseline=record; baseline_arrays=arrays
    result=freeze(prepared.model_copy(update={"experiment_id":"attention_experiment_"+uuid4().hex,"kind":"result",
        "created_at":datetime.now(UTC).isoformat(),"outcomes":outcomes,"result":measurements(prepared.role,outcomes),
        "provenance":{**prepared.provenance,"prepared_experiment_id":prepared.experiment_id,"job_id":job["job_id"],
            "attempt_id":job["attempt_id"],"source_to_eager_baseline_score_deltas":source_deltas,
            **({"source_token_replay_equal":True,"source_logit_fingerprint_check":"exact match" if source.validation.get("prediction_logits_sha256") else "unavailable in this earlier source capture; fresh baseline fingerprint retained"} if prepared.role=="generation" else {})}}))
    digest=repo.write(result)
    return {"experiment_id":result.experiment_id,"source_case_id":result.source_case_id,"method_id":result.method_id,
        "content_hash":digest,"name":result.method_id,"kind":"attention_intervention","capture_refs":refs}
