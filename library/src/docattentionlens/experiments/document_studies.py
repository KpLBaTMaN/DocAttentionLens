"""Immutable fixed-evidence experiment plans; no execution is implied by preparation."""
from hashlib import sha256
from io import BytesIO
from pathlib import Path
from math import floor,ceil
from typing import Any,Literal
from uuid import uuid4
from datetime import datetime,UTC
from PIL import Image,ImageFilter
from pydantic import BaseModel,ConfigDict,Field
from ..datasets.contracts import CaseRecord
from ..datasets.adapter import canonical
from ..methods.registry import get_definition

VERSION="document_studies_v1"
EXPERIMENT_METHODS={"document.image_text_contrast","document.element_masking","document.element_steering","attribution.integrated_gradients"}


def execution_code_hashes():
    root=Path(__file__).resolve().parents[1]
    names=["experiments/document_studies.py","generation/qwen.py","generation/contracts.py",
           "generation/document_inputs.py","generation/steering.py","methods/integrated_gradients.py"]
    return {name:sha256((root/name).read_bytes().replace(b"\r\n",b"\n")).hexdigest() for name in names}


def validate_execution_code(study):
    if study.provenance.get("execution_code_sha256")!=execution_code_hashes():
        raise ValueError("Prepared experiment code differs from the current adapter; prepare a new immutable plan")


class StudyRequest(BaseModel):
    model_config=ConfigDict(extra="forbid")
    generation_capture_id: str = Field(pattern=r"^generation_[a-f0-9]{32}$")
    method_id: str
    variant_id: str
    parameters: dict[str,Any]=Field(default_factory=dict)


class StudyArm(BaseModel):
    name: str
    representation: Literal["image","ocr","structured"]="image"
    page_id: str | None=None
    mask_box: list[int] | None=None
    replacement: Literal["white","blur"]="white"
    strength: float | None=None
    control: bool=False


class StudyOutcome(BaseModel):
    name: str
    capture_id: str
    capture_sha256: str
    answer: str
    output_token_ids: list[int]
    input_positions: int
    strength: float | None=None
    selected_element_mass: float | None=None
    annotated_element_mass: float | None=None
    # Integrated gradients: per input position (None outside consumed image patches), with its target and completeness terms.
    attribution: list[float | None] | None=None
    target_token_id: int | None=None
    target_log_probability: float | None=None
    baseline_log_probability: float | None=None
    target_is_greedy: bool | None=None
    # Attribution totals of the two interleaved half-step rules; their gap measures convergence.
    half_rule_totals: list[float] | None=None


class DocumentStudy(BaseModel):
    model_config=ConfigDict(extra="forbid")
    schema_version: Literal["1.0.0"]="1.0.0"
    study_id: str
    kind: Literal["prepared","result"]
    created_at: str
    source_generation_capture_id: str
    source_capture_sha256: str
    source_case_id: str
    case: CaseRecord
    method_id: str
    variant_id: str
    parameters: dict[str,Any]
    model_registration_id: str
    model_revision: str
    arms: list[StudyArm]
    outcomes: list[StudyOutcome]=Field(default_factory=list)
    provenance: dict[str,Any]
    content_sha256: str | None=None


def freeze(record):
    return record.model_copy(update={"content_sha256":sha256(canonical(record.model_dump(mode="json",exclude={"content_sha256"}))).hexdigest()})


def area(box):
    return (box[2]-box[0])*(box[3]-box[1])


def intersects(a,b):
    return min(a[2],b[2])>max(a[0],b[0]) and min(a[3],b[3])>max(a[1],b[1])


def boxes(elements,ids,pages):
    known={e.element_id:e for e in elements}
    if not ids or len(set(ids))!=len(ids) or not set(ids)<=set(known):
        raise ValueError("Select distinct mapped elements for the experiment")
    selected=[known[i] for i in ids]
    if len({e.page_id for e in selected})!=1 or any(e.geometry_flags for e in selected):
        raise ValueError("Choose reviewed geometry on one supplied page")
    pid=selected[0].page_id; page=next((p for p in pages if p.page_id==pid),None)
    if page is None: raise ValueError("Selected element page was not supplied")
    box=[floor(min(e.bbox.x1 for e in selected)),floor(min(e.bbox.y1 for e in selected)),
         ceil(max(e.bbox.x2 for e in selected)),ceil(max(e.bbox.y2 for e in selected))]
    if area(box)<=0 or box[0]<0 or box[1]<0 or box[2]>page.width or box[3]>page.height:
        raise ValueError("Selected mask region lies outside the page")
    return pid,box,page


def matched_control(page,box,elements,support):
    width,height=box[2]-box[0],box[3]-box[1]
    excluded=[box]+[[e.bbox.x1,e.bbox.y1,e.bbox.x2,e.bbox.y2] for e in elements if e.page_id==page.page_id and e.element_id in support]
    # Deterministic bounded search, including far edges. No claimed semantic matching.
    xs=sorted({0,page.width-width,*[round(i*(page.width-width)/20) for i in range(21)]})
    ys=sorted({0,page.height-height,*[round(i*(page.height-height)/20) for i in range(21)]})
    for y in ys:
        for x in xs:
            candidate=[x,y,x+width,y+height]
            if not any(intersects(candidate,b) for b in excluded): return candidate
    raise ValueError("No disjoint same-size non-support control region fits this page; choose a smaller element")


def prepare(body,capture,document):
    if body.method_id not in EXPERIMENT_METHODS: raise ValueError("Method is not a document experiment family")
    definition=get_definition(body.method_id,body.variant_id)
    parameters=definition.parameters.model_validate(body.parameters)
    if body.method_id=="attribution.integrated_gradients":
        return prepare_attribution(body,capture,document,parameters)
    if parameters.layer>=capture.layer_count or parameters.head>=capture.query_head_count: raise ValueError("Selected intervention axes are outside this model")
    if not any(s.step==parameters.step for s in capture.steps): raise ValueError("Selected source prediction step was not captured")
    pages=[p for p in document.pages if p.page_id in capture.context.selected_page_ids]
    arms=[StudyArm(name="baseline")]
    if body.method_id=="document.image_text_contrast":
        if "image" not in parameters.representations: raise ValueError("Representation contrast requires its image baseline")
        if len(set(parameters.representations))!=len(parameters.representations): raise ValueError("Representation arms must be distinct")
        for mode in parameters.representations:
            if mode!="image": arms.append(StudyArm(name=mode,representation=mode))
        if len(arms)<2: raise ValueError("Select at least one OCR or structured-text arm")
        if not any(e.ocr.text for e in document.elements if e.page_id in capture.context.selected_page_ids): raise ValueError("Supplied pages have no OCR text")
    else:
        pid,box,page=boxes(document.elements,parameters.element_ids,pages)
        selected=set(parameters.element_ids)
        if not any(any(i in p.ownership for i in selected) for p in capture.positions if p.page_id==pid):
            raise ValueError("Selected elements have no model-input mapping")
        control=matched_control(page,box,document.elements,set(document.case.support_element_ids))
        if body.method_id=="document.element_masking":
            arms.extend([StudyArm(name="masked",page_id=pid,mask_box=box,replacement=parameters.replacement),
                         StudyArm(name="matched_non_support_mask",page_id=pid,mask_box=control,replacement=parameters.replacement,control=True)])
        else:
            if any(not -8<=value<=8 for value in parameters.strengths) or len(set(parameters.strengths))!=len(parameters.strengths):
                raise ValueError("Steering strengths must be distinct finite values from -8 to 8")
            for strength in parameters.strengths:
                arms.append(StudyArm(name=f"strength_{strength:g}",page_id=pid,mask_box=box,strength=strength))
                if strength: arms.append(StudyArm(name=f"control_{strength:g}",page_id=pid,mask_box=control,strength=strength,control=True))
    return freeze(DocumentStudy(study_id="document_study_"+uuid4().hex,kind="prepared",created_at=datetime.now(UTC).isoformat(),
        source_generation_capture_id=capture.capture_id,source_capture_sha256=capture.content_sha256,source_case_id=document.case.case_id,
        case=document.case,method_id=body.method_id,variant_id=body.variant_id,parameters=parameters.model_dump(mode="json"),
        model_registration_id=capture.provenance["model_registration_id"],model_revision=capture.provenance["model_revision"],arms=arms,
        provenance={"operation_version":VERSION,"annotation_sha256":document.annotation_sha256,"execution_code_sha256":execution_code_hashes(),
                    "stage":"generation_fixed_evidence","system_prompt":capture.context.system_prompt,"page_ids":capture.context.selected_page_ids,
                    "control":"disjoint same-size non-support rectangle; content/location differ", "source_query":capture.context.query,
                    "limitations":["fixed-evidence generation only; no end-to-end retrieval effect","answer text change is not semantic correctness","rectangle interventions may affect intervening or overlapping content"]}))


def prepare_attribution(body,capture,document,parameters):
    """One gradient run on the unchanged source inputs; no generation arms and no new capture."""
    step=next((s for s in capture.steps if s.step==parameters.step),None)
    if step is None: raise ValueError("Selected source prediction step was not captured")
    if capture.context.representation!="image" or not any(p.role=="image" for p in capture.positions):
        raise ValueError("Integrated gradients needs consumed page images")
    from ..generation.contracts import GENERATORS
    profile=GENERATORS.get(capture.provenance.get("model_registration_id"))
    if profile is not None and profile.deepstack_visual_indexes:
        raise ValueError("Integrated gradients does not support DeepStack generators: their visual features also enter later decoder layers")
    return freeze(DocumentStudy(study_id="document_study_"+uuid4().hex,kind="prepared",created_at=datetime.now(UTC).isoformat(),
        source_generation_capture_id=capture.capture_id,source_capture_sha256=capture.content_sha256,source_case_id=document.case.case_id,
        case=document.case,method_id=body.method_id,variant_id=body.variant_id,parameters=parameters.model_dump(mode="json"),
        model_registration_id=capture.provenance["model_registration_id"],model_revision=capture.provenance["model_revision"],
        arms=[StudyArm(name="integrated_gradients")],
        provenance={"operation_version":VERSION,"annotation_sha256":document.annotation_sha256,"execution_code_sha256":execution_code_hashes(),
                    "stage":"generation_fixed_evidence","system_prompt":capture.context.system_prompt,"page_ids":capture.context.selected_page_ids,
                    "source_query":capture.context.query,"target":{"kind":"token_log_probability","step":step.step,"token_id":step.predicted_token_id,
                    "piece":step.predicted_piece,"conditioning":"prompt and earlier output tokens fixed (teacher forcing)"},
                    "path":{"kind":"straight_line","approximation":"midpoint Riemann sum","steps":parameters.steps,"baseline":parameters.baseline,
                            "input":"visual token embeddings entering the language model","baseline_input":"vision-encoder output for a constant image of the consumed size"},
                    "limitations":["output attribution for one declared target and baseline, not attention","attribution is to visual-token embeddings, not pixels; text and earlier output tokens are fixed",
                                   "float32 arithmetic; completeness and the interleaved half-rule disagreement are reported"]}))


def masked_png(raw,box,replacement):
    with Image.open(BytesIO(raw)) as source:
        image=source.convert("RGB")
    if not (0<=box[0]<box[2]<=image.width and 0<=box[1]<box[3]<=image.height): raise ValueError("Mask is outside the immutable input image")
    image.paste("white",tuple(box)) if replacement=="white" else image.paste(image.crop(tuple(box)).filter(ImageFilter.GaussianBlur(12)),(box[0],box[1]))
    stream=BytesIO(); image.save(stream,format="PNG"); return stream.getvalue()
