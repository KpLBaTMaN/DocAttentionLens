"""Bounded paired document execution; cancelled/expired attempts publish no indexed results."""
from datetime import datetime,UTC
from hashlib import sha256
from uuid import uuid4
from docattentionlens_api.repositories.document_studies import DocumentStudyRepository
from docattentionlens_api.repositories.documents import DocumentRepository
from docattentionlens_api.repositories.generations import GenerationRepository,GenerationStore
from docattentionlens_api.repositories.experiments import source_inputs
from docattentionlens.datasets.adapter import canonical
from docattentionlens.experiments.document_studies import VERSION,StudyOutcome,masked_png,freeze,validate_execution_code
from docattentionlens.generation.contracts import GENERATORS


def execute_document_study(settings,jobs,job,interrupted):
    from docattentionlens.generation.qwen import capture as generate
    repo=DocumentStudyRepository(jobs); spec=job["specification"]; study=repo.get(spec["study_id"])
    profile=GENERATORS[study.model_registration_id]
    validate_execution_code(study)
    if study.kind!="prepared" or study.content_sha256!=spec["study_sha256"] or spec["operation_version"]!=VERSION or (profile.adapter_version,profile.revision)!=(spec["generator_adapter"],spec["generator_revision"]):
        raise ValueError("Document experiment differs from its immutable plan")
    source=GenerationRepository(jobs).get(study.source_generation_capture_id)
    if source.content_sha256!=study.source_capture_sha256: raise ValueError("Source capture differs from the prepared plan")
    documents=DocumentRepository(settings)
    inputs=source_inputs(documents,study.case,study.provenance["page_ids"])
    if study.method_id=="attribution.integrated_gradients":
        return execute_attribution(settings,repo,job,interrupted,study,profile,source,inputs)
    store=GenerationStore(settings.artifact_root); outcomes=[]; baseline=None; captures=[]
    for arm in study.arms:
        if interrupted(): raise RuntimeError("Document experiment interrupted")
        derived=[]; hashes={}
        for page,raw,elements in inputs:
            if arm.mask_box and arm.page_id==page.page_id and arm.strength is None:
                raw=masked_png(raw,arm.mask_box,arm.replacement)
                page=page.model_copy(update={"image_sha256":sha256(raw).hexdigest()})
            hashes[page.page_id]=page.image_sha256; derived.append((page,raw,elements))
        case=study.case.model_copy(update={"page_image_sha256":{**study.case.page_image_sha256,**hashes},"content_sha256":None})
        case=case.model_copy(update={"content_sha256":sha256(canonical(case.model_dump(mode="json",exclude={"content_sha256"}))).hexdigest()})
        steering=None
        if arm.strength is not None:
            steering={"layer":study.parameters["layer"],"head":study.parameters["head"],"strength":arm.strength,
                      "element_ids":study.parameters["element_ids"],"page_id":arm.page_id,
                      "control_box":arm.mask_box if arm.control else None,"source_population":"all prefill/decode sources at this decoder layer/head"}
        gen,arrays=generate(case,None,derived,settings.model_cache,interrupted,system_prompt=study.provenance["system_prompt"],profile=profile,representation=arm.representation,steering=steering)
        if arm.name=="baseline":
            baseline=gen
            if gen.output_token_ids!=source.output_token_ids: raise ValueError("Fresh baseline no longer reproduces the source capture output")
        if arm.strength==0 and (gen.output_token_ids!=baseline.output_token_ids or gen.validation["prediction_logits_sha256"]!=baseline.validation["prediction_logits_sha256"]):
            raise ValueError("Zero-strength intervention differs from baseline tokens/logits")
        gen=gen.model_copy(update={"dependencies":[*gen.dependencies,study.study_id,job["job_id"]],
            "provenance":{**gen.provenance,"document_study_id":study.study_id,"arm":arm.name,"source_case_sha256":study.case.content_sha256,
                          "case_image_version":"derived images retain semantic page/element identities; hashes identify exact consumed assets"}})
        digest=store.put(gen,arrays); saved=store.read(gen.capture_id,digest)
        for ref in saved.arrays: store.array(saved,ref.name)
        from docattentionlens.generation.inspection import attention_row
        step=next((s for s in gen.steps if s.step==study.parameters["step"]),None)
        allocation=attention_row(gen,arrays[step.array],step.step,study.parameters["layer"],study.parameters["head"]) if step else None
        selected_ids=set(study.parameters.get("element_ids",[]))
        selected_mass=sum(a.mass for a in allocation.allocations if a.element_id in selected_ids) if selected_ids and allocation else None
        outcomes.append(StudyOutcome(name=arm.name,capture_id=gen.capture_id,capture_sha256=digest,answer=gen.answer,
            output_token_ids=gen.output_token_ids,input_positions=len(gen.context.input_token_ids),strength=arm.strength,
            selected_element_mass=selected_mass,annotated_element_mass=sum(a.mass for a in allocation.allocations if a.element_id) if allocation else None))
        captures.append({"capture_id":gen.capture_id,"sha256":digest,"arm":arm.name})
    result=freeze(study.model_copy(update={"study_id":"document_study_"+uuid4().hex,"kind":"result","created_at":datetime.now(UTC).isoformat(),"outcomes":outcomes,
        "provenance":{**study.provenance,"prepared_study_id":study.study_id,"job_id":job["job_id"],"attempt_id":job["attempt_id"]}}))
    digest=repo.write(result)
    return {"experiment_id":result.study_id,"source_case_id":result.source_case_id,"content_hash":digest,"method_id":result.method_id,
            "kind":"document_study","name":result.method_id,"capture_refs":captures}


def execute_attribution(settings,repo,job,interrupted,study,profile,source,inputs):
    """One gradient run on the unchanged source inputs; publishes attribution only, never a new capture."""
    from docattentionlens.generation.qwen import integrated_gradients
    if interrupted(): raise RuntimeError("Document experiment interrupted")
    parameters=study.parameters
    measured=integrated_gradients(study.case,inputs,settings.model_cache,interrupted,source=source,step=parameters["step"],
        steps=parameters["steps"],baseline=parameters["baseline"],system_prompt=study.provenance["system_prompt"],profile=profile)
    if measured["target_token_id"]!=study.provenance["target"]["token_id"]:
        raise ValueError("Attributed target differs from the prepared plan")
    outcome=StudyOutcome(name="integrated_gradients",capture_id=source.capture_id,capture_sha256=study.source_capture_sha256,answer=source.answer,
        output_token_ids=source.output_token_ids,input_positions=len(source.context.input_token_ids),
        attribution=measured["attribution"],target_token_id=measured["target_token_id"],target_log_probability=measured["target_log_probability"],
        baseline_log_probability=measured["baseline_log_probability"],target_is_greedy=measured["target_is_greedy"],half_rule_totals=measured["half_rule_totals"])
    result=freeze(study.model_copy(update={"study_id":"document_study_"+uuid4().hex,"kind":"result","created_at":datetime.now(UTC).isoformat(),"outcomes":[outcome],
        "provenance":{**study.provenance,"prepared_study_id":study.study_id,"job_id":job["job_id"],"attempt_id":job["attempt_id"],"attribution_runtime":measured["runtime"]}}))
    digest=repo.write(result)
    return {"experiment_id":result.study_id,"source_case_id":result.source_case_id,"content_hash":digest,"method_id":result.method_id,
            "kind":"document_study","name":result.method_id,"capture_refs":[]}
