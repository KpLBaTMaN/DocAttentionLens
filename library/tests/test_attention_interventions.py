"""Plan semantics and effect reporting for direct attention interventions."""
from types import SimpleNamespace as NS
import numpy as np
import pytest
from docattentionlens.attention_interventions import plan,removed_edges
from docattentionlens.experiments.attention_interventions import prepare,AttentionExperimentRequest,validate_code,measurements
from docattentionlens.methods.registry import get_method,get_definition
from docattentionlens.generation.contracts import GENERATORS
from test_document_studies import inputs

def positions():
    return [NS(index=i,role=r,page_id="page_a" if r=="image" else None,
        bbox=[0,0,2,2] if i==1 else [2,0,4,2] if i==2 else None)
        for i,r in enumerate(["special","image","image","query","query"])]

def prepare_plan(identifier,**parameters):
    descriptor=get_method(identifier)
    definition=get_definition(identifier,descriptor.variant_id)
    p=definition.parameters.model_validate(parameters).model_dump(mode="json")
    return plan(identifier,p,positions(),[1]*5,True,2,3,"page_a")

@pytest.mark.parametrize("identifier",["attention.head_ablation","attention.layer_ablation","attention.edge_ablation","attention.modality_ablation","attention.region_masking"])
def test_each_intervention_has_a_real_validated_plan(identifier):
    parameters={"source_index":4,"target_index":1} if identifier=="attention.edge_ablation" else {}
    spec=prepare_plan(identifier,**parameters)
    assert spec["method_id"]==identifier
    assert spec["layers"]==([0,1] if identifier=="attention.modality_ablation" else [0])
    assert spec["heads"]==([0,1,2] if identifier in ("attention.layer_ablation","attention.modality_ablation") else [0])

def test_edge_removal_is_exact_and_modality_extends_generated_text():
    edge=prepare_plan("attention.edge_ablation",source_index=4,target_index=1)
    mask=removed_edges(edge,positions(),range(5),5)
    assert np.argwhere(mask).tolist()==[[4,1]]
    modality=prepare_plan("attention.modality_ablation")
    mask=removed_edges(modality,positions(),range(5,7),7)
    assert np.argwhere(mask).tolist()==[[0,1],[0,2],[1,1],[1,2]]

def test_region_masks_incoming_edges_from_outside_only():
    spec=prepare_plan("attention.region_masking",region=[0,0,.4,1])
    assert spec["region_targets"]==[1]
    mask=removed_edges(spec,positions(),range(5),5)
    assert not mask[1].any()
    assert np.flatnonzero(mask[:,1]).tolist()==[0,2,3,4]

def test_all_masked_rows_future_edges_invalid_axes_and_empty_regions_are_rejected():
    with pytest.raises(ValueError,match="every valid key"):prepare_plan("attention.modality_ablation",block="text_to_text")
    with pytest.raises(ValueError,match="no valid attention"):prepare_plan("attention.edge_ablation",source_index=0,target_index=1)
    with pytest.raises(ValueError,match="outside"):prepare_plan("attention.head_ablation",head=3)
    with pytest.raises(ValueError,match="positive unit"):prepare_plan("attention.region_masking",region=[.5,0,.4,1])
    with pytest.raises(ValueError,match="no patch"):prepare_plan("attention.region_masking",region=[.49,.1,.51,.2])

def test_generation_plan_freezes_identity_and_code_without_consuming_labels():
    capture,document=inputs()
    capture.case_id=document.case.case_id;capture.dataset_version=document.case.dataset_version
    capture.provenance["model_revision"]=GENERATORS["builtin"].revision
    capture.context.representation="image";capture.context.input_token_ids=list(range(5));capture.context.input_attention_mask=[1]*5
    capture.context.pages=[NS(page_id="page_a",image_sha256="a"*64)]
    capture.positions=positions()
    method=get_method("attention.head_ablation")
    request=AttentionExperimentRequest(capture_id=capture.capture_id,method_id=method.method_id,variant_id=method.variant_id)
    result=prepare(request,capture,document.case)
    assert result.kind=="prepared" and result.outcomes==[] and result.result is None
    assert result.provenance["inputs_unchanged"] and result.provenance["evaluation_labels_consumed"] is False
    assert result.case.reference_answer=="evaluation-only answer"
    validate_code(result)
    changed=result.model_copy(update={"provenance":{**result.provenance,"execution_code_sha256":{}}})
    with pytest.raises(ValueError,match="code changed"):validate_code(changed)

def test_effects_report_measurements_without_inventing_answer_correctness():
    generation=measurements("generation",[{"output_token_ids":[1,2],"answer":"a"},{"output_token_ids":[1,3],"answer":"b"}])
    assert generation.table.values==[[2,1],[2,0]]
    assert any("Common output-token prefix: 1" in n for n in generation.notices)
    assert all("correct" not in c.lower() for c in generation.table.columns)
    retrieval=measurements("retrieval",[
        {"pages":[{"page_id":"a","score":2},{"page_id":"b","score":1}]},
        {"pages":[{"page_id":"a","score":.5},{"page_id":"b","score":1}]}])
    assert retrieval.table.values[0]==[2,.5,-1.5,1,2]
