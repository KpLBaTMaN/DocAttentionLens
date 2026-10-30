"""Controlled transform, serializer and independent steering probability checks."""
from io import BytesIO
from types import SimpleNamespace as NS
import numpy as np
from PIL import Image
import pytest
from docattentionlens.datasets.contracts import CaseRecord
from docattentionlens.experiments.document_studies import prepare,StudyRequest,masked_png,area,intersects,VERSION,validate_execution_code
from docattentionlens.generation.document_inputs import serialize,text_ownership
from docattentionlens.generation.steering import biased_probabilities
from docattentionlens.methods.contracts import DocumentInputs
from docattentionlens.methods.registry import get_method


def inputs():
    case=CaseRecord(key="case_a",case_id="case_a",dataset_id="dataset_a",dataset_version="v1",document_id="doc_a",question_id="question_a",
        query_variant="document",task_scope="document",query="Question",page_ids=["page_a"],page_image_sha256={"page_a":"a"*64},annotation_version="v1",
        support_page_ids=["page_a"],support_element_ids=["element_b"],reference_answer="evaluation-only answer",producer_version="fixture",content_sha256="b"*64)
    page=NS(page_id="page_a",page_number=1,width=4,height=2)
    def element(i,x,kind):
        return NS(element_id=i,page_id="page_a",bbox=NS(x1=x,y1=0,x2=x+1,y2=1),geometry_flags=[],reading_order=x+1,
                  element_type=kind,ocr=NS(text="body "+i))
    elements=(element("element_a",0,"heading"),element("element_b",1,"table"))
    document=DocumentInputs(case,elements,(page,),"c"*64)
    capture=NS(capture_id="generation_"+"a"*32,content_sha256="d"*64,layer_count=2,query_head_count=3,
        steps=[NS(step=0)],
        context=NS(selected_page_ids=["page_a"],system_prompt="Instruction",query="Question"),
        provenance={"model_registration_id":"builtin","model_revision":"fixture"},positions=[NS(page_id="page_a",ownership={"element_a":1})])
    return capture,document


def request(identifier,**parameters):
    descriptor=get_method(identifier)
    return StudyRequest(generation_capture_id="generation_"+"a"*32,method_id=identifier,variant_id=descriptor.variant_id,parameters=parameters)


def test_frozen_representation_and_mask_plans_use_real_inputs_and_matched_controls():
    capture,document=inputs()
    representation=prepare(request("document.image_text_contrast"),capture,document)
    assert [a.representation for a in representation.arms]==["image","ocr","structured"]
    assert representation.provenance["operation_version"]==VERSION and representation.content_sha256
    validate_execution_code(representation)
    altered=representation.model_copy(update={"provenance":{**representation.provenance,"execution_code_sha256":{}}})
    with pytest.raises(ValueError,match="code differs"):validate_execution_code(altered)
    assert get_method("document.image_text_contrast").parameters_schema["properties"]["representations"]["default"]==["image","ocr","structured"]
    assert get_method("attention.head_specialisation").parameters_schema["properties"]["layers"]["default"]==[0]
    masking=prepare(request("document.element_masking",element_ids=["element_a"]),capture,document)
    target,control=masking.arms[1:]
    assert area(target.mask_box)==area(control.mask_box) and not intersects(target.mask_box,control.mask_box)
    support=[1,0,2,1]
    assert not intersects(control.mask_box,support)
    with pytest.raises(ValueError,match="Select distinct"):prepare(request("document.element_masking"),capture,document)
    with pytest.raises(ValueError,match="axes"):prepare(request("document.element_steering",layer=9,element_ids=["element_a"]),capture,document)
    with pytest.raises(ValueError,match="strengths"):prepare(request("document.element_steering",strengths=[99],element_ids=["element_a"]),capture,document)


def test_mask_changes_only_declared_pixels_and_zero_bias_preserves_masked_support():
    source=np.full((2,4,3),20,dtype=np.uint8); stream=BytesIO();Image.fromarray(source).save(stream,format="PNG")
    masked=np.asarray(Image.open(BytesIO(masked_png(stream.getvalue(),[0,0,1,1],"white"))))
    assert (masked[0,0]==255).all() and np.array_equal(masked[:,1:],source[:,1:]) and np.array_equal(masked[1],source[1])
    p=np.array([.1,.2,0,.7]); w=np.array([1,.5,1,0])
    assert biased_probabilities(p,w,0)==pytest.approx(p)
    positive=biased_probabilities(p,w,2); negative=biased_probabilities(p,w,-2)
    assert positive[0]>p[0]>negative[0] and positive[2]==0 and positive.sum()==pytest.approx(1)
    # Compare with direct log-odds, independently of the adapter implementation.
    assert np.log(positive[0]/positive[3])-np.log(p[0]/p[3])==pytest.approx(2)


def test_serialized_text_has_element_ownership_but_no_evaluation_labels():
    _,document=inputs(); page=document.pages[0]
    text,spans=serialize(page,document.elements,"structured")
    assert "evaluation-only answer" not in text and "support" not in text and "reference_answer" not in text
    assert all(e.element_id in text for e in document.elements)
    a,b,i,pid=spans[0]
    owned,page_id=text_ownership((a,b),spans)
    assert owned=={i:1} and page_id==pid
    owned,_=text_ownership((a-2,a+2),spans)
    assert owned[i]==.5 and sum(owned.values())==1
    assert text_ownership((0,1),spans)==({},None)


def test_experiment_calculations_report_observed_effects_and_never_invent_correctness():
    from docattentionlens.methods.contracts import Context
    from docattentionlens.methods.registry import execute
    capture,_=inputs()
    capture.steps=[NS(step=0,target_count=1)]
    record=NS(outcomes=[NS(name="baseline",answer="A",input_positions=8,output_token_ids=[1],strength=None),
                        NS(name="strength_0",answer="A",input_positions=8,output_token_ids=[1],strength=0)])
    for identifier in ("document.image_text_contrast","document.element_masking","document.element_steering"):
        m=get_method(identifier); context=Context(capture,np.ones((2,3,1),dtype=np.float32),experiment=record)
        result=execute(m.method_id,m.variant_id,{},context)
        assert result.table.values[1][1 if identifier.endswith('steering') else 0] == 0 and any("baseline: A"==s for s in result.notices)


def test_integrated_gradients_plan_freezes_target_path_and_unchanged_inputs():
    capture,document=inputs()
    capture.steps=[NS(step=0,predicted_token_id=11,predicted_piece="True")]
    capture.context.representation="image"
    capture.positions=[NS(page_id="page_a",role="image",ownership={"element_a":1})]
    plan=prepare(request("attribution.integrated_gradients",steps=16,baseline="white"),capture,document)
    assert [a.name for a in plan.arms]==["integrated_gradients"] and plan.arms[0].mask_box is None
    assert plan.provenance["target"]=={"kind":"token_log_probability","step":0,"token_id":11,"piece":"True",
        "conditioning":"prompt and earlier output tokens fixed (teacher forcing)"}
    assert plan.provenance["path"]=={"kind":"straight_line","approximation":"midpoint Riemann sum","steps":16,"baseline":"white",
        "input":"visual token embeddings entering the language model","baseline_input":"vision-encoder output for a constant image of the consumed size"}
    assert "methods/integrated_gradients.py" in plan.provenance["execution_code_sha256"]
    validate_execution_code(plan)
    with pytest.raises(ValueError,match="not captured"):prepare(request("attribution.integrated_gradients",step=3),capture,document)
    with pytest.raises(Exception):prepare(request("attribution.integrated_gradients",layer=0),capture,document)
    capture.context.representation="ocr"
    with pytest.raises(ValueError,match="page images"):prepare(request("attribution.integrated_gradients"),capture,document)
