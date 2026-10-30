"""Orchestrate paired model execution; all scientific operations live in the library."""
from datetime import UTC, datetime
from uuid import uuid4
from docattentionlens_api.repositories.documents import DocumentRepository
from docattentionlens_api.repositories.experiments import ExperimentRepository, source_inputs
from docattentionlens_api.repositories.captures import CaptureStore
from docattentionlens_api.repositories.generations import GenerationStore
from docattentionlens.experiments.contracts import VERSION, FixedContextSelection
from docattentionlens.experiments.relayout import freeze, compare
from docattentionlens.retrieval.contracts import ADAPTER_VERSION as RA, MODEL_REVISION as RR
from docattentionlens.generation.contracts import ADAPTER_VERSION as GA, MODEL_REVISION as GR


def execute_experiment(settings, jobs, job, interrupted):
    from docattentionlens.retrieval.colpali import capture as retrieve
    from docattentionlens.generation.qwen import capture as generate
    from docattentionlens.generation.inspection import attention_row
    repo=ExperimentRepository(jobs); spec=job["specification"]; protocol=repo.get(spec["experiment_id"])
    if protocol.record_kind!="experiment_protocol" or protocol.review is None or protocol.content_sha256!=spec["protocol_sha256"] or spec["operation_version"]!=VERSION or (spec["retrieval_adapter"],spec["retrieval_revision"],spec["generator_adapter"],spec["generator_revision"])!=(RA,RR,GA,GR):
        raise ValueError("Experiment specification differs from reviewed frozen protocol")
    documents=DocumentRepository(settings); source=documents.case(protocol.source_case_id)
    if source.content_sha256!=protocol.provenance["source_case_sha256"]: raise ValueError("Original case lineage differs")
    originals=source_inputs(documents,source,protocol.protocol.candidate_page_ids)
    rs=CaptureStore(settings.artifact_root); gs=GenerationStore(settings.artifact_root)
    outcomes=[]; baseline_retrieval=None
    for arm in protocol.arms:
        if interrupted(): raise RuntimeError("Paired execution interrupted")
        raw=repo.store.image(protocol,arm.name)
        inputs=[(arm.page,raw,arm.elements) if p.page_id==protocol.protocol.page_id else (p,b,e) for p,b,e in originals]
        outcome={"arm":arm.name,"case_id":arm.case.case_id,"input_sha256":arm.asset_sha256,"consuming_stage":protocol.protocol.stage}
        if protocol.protocol.stage!="generation_fixed_evidence" or arm.name=="baseline":
            retrieval,arrays=retrieve(arm.case,inputs,settings.model_cache,interrupted)
            digest=rs.put(retrieval,arrays); retrieval=rs.read(retrieval.capture_id,digest)
            for ref in retrieval.arrays: rs.array(retrieval,ref.name)
            if arm.name=="baseline": baseline_retrieval=retrieval
            outcome["retrieval"]={"capture_id":retrieval.capture_id,"sha256":digest}
            if protocol.protocol.stage!="generation_fixed_evidence":
                outcome["scores"]={p.page_id:p.score for p in retrieval.pages}
                outcome["support_pages_in_top_two"]=[p.page_id for p in sorted(retrieval.pages,key=lambda p:(-p.score,p.page_number,p.page_id))[:2] if p.page_id in arm.case.support_page_ids]
        else:
            retrieval=FixedContextSelection(capture_id="selection_"+uuid4().hex,case_id=arm.case.case_id,query=arm.case.query,pages=[p for p,_,_ in inputs])
            outcome["fixed_candidate_selection"]={**retrieval.model_dump(mode="json"),"record_kind":"fixed_context_selection","baseline_retrieval_capture_id":baseline_retrieval.capture_id,"interpretation":"Candidate identities only; variant images were not re-encoded for retrieval and no scores are assigned"}
        if protocol.protocol.stage!="retrieval_only":
            if protocol.protocol.stage=="generation_fixed_evidence":
                ids=[arm.page.page_id if pid==protocol.protocol.page_id else pid for pid in protocol.protocol.fixed_context_page_ids]
            else:
                ids=[p.page_id for p in sorted(retrieval.pages,key=lambda p:(-p.score,p.page_number,p.page_id))[:2]]
            gen_inputs=[next(x for x in inputs if x[0].page_id==pid) for pid in ids]
            gen,arrays=generate(arm.case,retrieval,gen_inputs,settings.model_cache,interrupted)
            gen=gen.model_copy(update={"dependencies":[*gen.dependencies,protocol.experiment_id,job["job_id"]],"provenance":{**gen.provenance,"experiment_protocol_id":protocol.experiment_id,"arm":arm.name,"context_selection":"fixed reviewed evidence" if protocol.protocol.stage=="generation_fixed_evidence" else "top two by score; ties by page number then scoped ID"}})
            digest=gs.put(gen,arrays); gen=gs.read(gen.capture_id,digest)
            for ref in gen.arrays:
                values=gs.array(gen,ref.name)
                if ref.name.startswith("step_"):
                    s=next(s for s in gen.steps if s.array==ref.name); attention_row(gen,values,s.step,0,0)
            outcome.update(generation={"capture_id":gen.capture_id,"sha256":digest},answer=gen.answer,output_token_ids=gen.output_token_ids,stop_reason=gen.stop_reason,
                supplied_page_ids=ids,changed_page_supplied=arm.page.page_id in ids,omitted_page_ids=gen.context.omitted_page_ids,
                context_policy=gen.context.construction_version,capture_validation=gen.validation)
        outcomes.append(outcome)
    result=freeze(protocol.model_copy(update={"experiment_id":"experiment_"+uuid4().hex,"record_kind":"experiment_result","created_at":datetime.now(UTC).isoformat(),"outcomes":outcomes,
        "comparisons":compare(outcomes,protocol.protocol.page_id,protocol.arms),"dependencies":[*protocol.dependencies,protocol.experiment_id,job["job_id"]],
        "provenance":{**protocol.provenance,"protocol_id":protocol.experiment_id,"protocol_sha256":protocol.content_sha256,"job_id":job["job_id"],"attempt_id":job["attempt_id"],"input_hash":job["input_hash"],"repetitions":1,"seed":0,"generation_context_policy":"fixed reviewed selection" if protocol.protocol.stage=="generation_fixed_evidence" else "top two by score, page number/scoped ID ties"}}))
    digest=repo.store.put(result); repo.store.read(result.experiment_id,digest)
    refs=[{**o[k],"arm":o["arm"],"kind":k} for o in outcomes for k in ("retrieval","generation") if k in o]
    return {"experiment_id":result.experiment_id,"source_case_id":source.case_id,"content_hash":digest,"name":result.name,"kind":"experiment_result","capture_refs":refs}
