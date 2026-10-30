"""Immutable experiment records for direct attention interventions on frozen captures.

Prepares and freezes a plan from a saved capture, checks the execution code before a run and
tabulates outcomes. Plan validation and the hook applied inside the model adapters live in the
top-level `attention_interventions.py`.
"""
from datetime import datetime,UTC
from hashlib import sha256
from pathlib import Path
from typing import Any,Literal
from uuid import uuid4
from importlib.util import find_spec
from pydantic import BaseModel,ConfigDict,Field
from ..datasets.adapter import canonical
from ..datasets.contracts import CaseRecord
from ..methods.contracts import Calculation
from ..methods.registry import get_definition
from ..generation.contracts import GENERATORS
from ..retrieval.contracts import retriever_for_adapter
from ..attention_interventions import plan

VERSION="attention_interventions_v1"

class AttentionExperimentRequest(BaseModel):
    model_config=ConfigDict(extra="forbid")
    capture_id: str = Field(pattern=r"^(capture|generation)_[a-f0-9]{32}$")
    method_id: str
    variant_id: str = "captured_attention_v1"
    page_id: str | None = None
    parameters: dict[str,Any] = Field(default_factory=dict)

class AttentionExperiment(BaseModel):
    model_config=ConfigDict(extra="forbid")
    schema_version: Literal["1.0.0"]="1.0.0"
    experiment_id: str
    kind: Literal["prepared","result"]
    created_at: str
    source_capture_id: str
    source_capture_sha256: str
    source_case_id: str
    case: CaseRecord
    role: Literal["generation","retrieval"]
    method_id: str
    variant_id: str
    parameters: dict[str,Any]
    model_registration_id: str
    model_revision: str
    page_ids: list[str]
    intervention: dict[str,Any]
    outcomes: list[dict[str,Any]] = Field(default_factory=list)
    result: Calculation | None = None
    provenance: dict[str,Any]
    content_sha256: str | None = None

def freeze(record):
    return record.model_copy(update={"content_sha256":sha256(canonical(record.model_dump(mode="json",exclude={"content_sha256"}))).hexdigest()})

def code_hashes():
    root=Path(__file__).resolve().parents[1]
    names=["attention_interventions.py","experiments/attention_interventions.py","generation/qwen.py",
        "retrieval/colpali.py","retrieval/colqwen.py","retrieval/scoring.py"]
    names += ["methods/"+n+".py" for n in ("head_ablation","layer_ablation","edge_ablation","modality_ablation","region_attention_masking","_intervention_method")]
    names += ["methods/_attention_only.py","methods/_shared.py","methods/contracts.py","generation/contracts.py","retrieval/contracts.py"]
    result={n:sha256((root/n).read_bytes().replace(b"\r\n",b"\n")).hexdigest() for n in names}
    worker=find_spec("docattentionlens_worker.attention_interventions") if find_spec("docattentionlens_worker") else None
    if worker and worker.origin:
        result["worker/attention_interventions.py"]=sha256(Path(worker.origin).read_bytes().replace(b"\r\n",b"\n")).hexdigest()
    return result

def validate_code(record):
    if record.provenance.get("execution_code_sha256")!=code_hashes():
        raise ValueError("Experiment execution code changed; prepare a new immutable plan")

def prepare(body,capture,case):
    definition=get_definition(body.method_id,body.variant_id)
    if definition.descriptor.family!="attention_interventions": raise ValueError("Select a direct attention-intervention method")
    parameters=definition.parameters.model_validate(body.parameters).model_dump(mode="json")
    generation=capture.capture_id.startswith("generation_")
    if capture.case_id!=case.case_id or capture.provenance.get("attention_experiment_id"):
        raise ValueError("Use an original capture of this case, not a nested intervention")
    if case.query!=(capture.context.query if generation else capture.query) or (not generation and case.dataset_version!=capture.dataset_version):
        raise ValueError("Frozen case query/version differs from the source capture")
    if generation:
        if capture.provenance.get("document_study_id") or capture.provenance.get("experiment_protocol_id") or capture.context.representation!="image":
            raise ValueError("Attention interventions require an original image-input generator capture")
        registration=capture.provenance.get("model_registration_id")
        if registration not in GENERATORS: raise ValueError("Select a pinned built-in generator")
        profile=GENERATORS[registration]
        revision=profile.revision
        if capture.provenance.get("model_revision")!=revision: raise ValueError("Capture model revision differs from the pinned adapter")
        count=len(capture.context.input_token_ids)
        positions=capture.positions[:count]; mask=capture.context.input_attention_mask
        pages=capture.context.selected_page_ids
        layer_count=capture.layer_count; head_count=capture.query_head_count; causal=True
    else:
        profile=retriever_for_adapter(capture.provenance.get("adapter_version"))
        if profile is None or capture.provenance.get("model_revision")!=profile["revision"]:
            raise ValueError("Select a pinned supported retriever capture")
        registration=profile["registration_id"]; revision=profile["revision"]
        pages=[p.page_id for p in capture.pages]
        selected=body.page_id or pages[0]
        page=next((p for p in capture.pages if p.page_id==selected),None)
        if page is None: raise ValueError("Intervention page is outside the frozen retrieval candidates")
        positions=page.positions; mask=[int(p.eligible) for p in positions]
        layer_count=profile["layers"]; head_count=profile["query_heads"]; causal=profile["causal"]
    page_id=body.page_id or pages[0]
    if page_id not in pages: raise ValueError("Select a page from the frozen model inputs")
    if any(case.page_image_sha256[p]!=next(x.image_sha256 for x in (capture.context.pages if generation else capture.pages) if x.page_id==p) for p in pages):
        raise ValueError("Frozen page image hashes differ from the case")
    spec=plan(body.method_id,parameters,positions,mask,causal,layer_count,head_count,page_id)
    return freeze(AttentionExperiment(experiment_id="attention_experiment_"+uuid4().hex,kind="prepared",
        created_at=datetime.now(UTC).isoformat(),source_capture_id=capture.capture_id,source_capture_sha256=capture.content_sha256,
        source_case_id=case.case_id,case=case,role="generation" if generation else "retrieval",method_id=body.method_id,
        variant_id=body.variant_id,parameters=parameters,model_registration_id=registration,model_revision=revision,
        page_ids=pages,intervention=spec,provenance={"execution_code_sha256":code_hashes(),"page_image_sha256":{p:case.page_image_sha256[p] for p in pages},
            "system_prompt":capture.context.system_prompt if generation else None,
            "source_population":"all native prefill/decode sources at selected layers/heads; edge indices address the frozen input prefix",
            "inputs_unchanged":True,"evaluation_labels_consumed":False,
            "limitations":["Head/layer ablation zeros context outputs before projection.","Edge masks preserve padding and causal masks; all-masked rows are rejected.",
                "Autoregressive answer changes are outcomes; diverged histories are not compared by attention-row subtraction.",
                "Retrieval query encoding is fixed; only the selected candidate page is re-encoded under intervention."]}))

def measurements(role,outcomes):
    from ..methods._attention_only import table
    baseline=outcomes[0]; changed=outcomes[1]
    if role=="generation":
        a,b=baseline["output_token_ids"],changed["output_token_ids"]
        prefix=0
        for x,y in zip(a,b):
            if x!=y: break
            prefix+=1
        return Calculation(table=table("Generation intervention outcomes",["Output tokens","Matches baseline"],
            ["Baseline","Intervention"],[[len(a),1],[len(b),float(a==b)]],"tokens; exact-token agreement"),
            notices=[f"Baseline answer: {baseline['answer']}",f"Intervention answer: {changed['answer']}",
                f"Common output-token prefix: {prefix}. No answer-correctness judgement is inferred."])
    before={p["page_id"]:p for p in baseline["pages"]}; after={p["page_id"]:p for p in changed["pages"]}
    ranks_a={p["page_id"]:i+1 for i,p in enumerate(sorted(baseline["pages"],key=lambda p:-p["score"]))}
    ranks_b={p["page_id"]:i+1 for i,p in enumerate(sorted(changed["pages"],key=lambda p:-p["score"]))}
    return Calculation(table=table("Frozen-candidate retrieval intervention outcomes",["Baseline MaxSim","Intervention MaxSim","Score change","Baseline rank","Intervention rank"],
        list(before),[[before[i]["score"],after[i]["score"],after[i]["score"]-before[i]["score"],ranks_a[i],ranks_b[i]] for i in before],
        "signed MaxSim score; rank in frozen candidate set"),
        notices=["Ranks cover only the frozen candidate pages. Input pixels and query encoding are unchanged."])
