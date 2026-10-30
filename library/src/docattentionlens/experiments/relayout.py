"""Lossless pixel translation; annotation associations are supplied by review."""
from datetime import UTC, datetime
from hashlib import sha256
from io import BytesIO
from math import floor, ceil
from pathlib import Path
from uuid import uuid4
import numpy as np
from PIL import Image
from docattentionlens import __version__
from docattentionlens.datasets.adapter import canonical
from docattentionlens.datasets.contracts import Box
from .contracts import VERSION, ArmInput, ExperimentRecord


def freeze(record):
    return record.model_copy(update={"content_sha256": sha256(canonical(record.model_dump(mode="json", exclude={"content_sha256"}))).hexdigest()})


def png(image):
    stream = BytesIO(); image.save(stream, format="PNG"); return stream.getvalue()


def rectangle(elements, ids, size, dimensions=None):
    if len(set(ids)) != len(ids) or not set(ids) <= {e.element_id for e in elements}:
        raise ValueError("Group elements must be distinct elements of the changed page")
    selected = [e for e in elements if e.element_id in ids]
    if any(e.geometry_flags or e.display_geometry_policy != "raw" for e in selected):
        raise ValueError("Flagged geometry cannot define a reviewed group")
    x1 = max(0, floor(min(e.bbox.x1 for e in selected))-8); y1 = max(0, floor(min(e.bbox.y1 for e in selected))-8)
    x2 = min(size[0], ceil(max(e.bbox.x2 for e in selected))+8); y2 = min(size[1], ceil(max(e.bbox.y2 for e in selected))+8)
    if dimensions:
        x2, y2 = x1 + dimensions[0], y1 + dimensions[1]
    if x1 < 0 or y1 < 0 or x2 > size[0] or y2 > size[1] or x2 <= x1 or y2 <= y1:
        raise ValueError("Matched crop does not fit inside the original page")
    touched = []
    for e in elements:
        b = e.bbox
        if min(x2, b.x2) > max(x1, b.x1) and min(y2, b.y2) > max(y1, b.y1):
            if e.geometry_flags or not (x1 <= b.x1 <= b.x2 <= x2 and y1 <= b.y1 <= b.y2 <= y2):
                raise ValueError("Crop would cut an annotated element; include the whole group")
            touched.append(e.element_id)
    if set(touched) != set(ids):
        raise ValueError("Include every element inside the crop, including text between heading and table")
    return [x1, y1, x2, y2]


def build_draft(request, case, page, raw, elements):
    if request.case_id != case.case_id or request.page_id not in case.page_ids or request.page_id != page.page_id or page.image_sha256 != case.page_image_sha256[page.page_id] or sha256(raw).hexdigest() != case.page_image_sha256[page.page_id]:
        raise ValueError("Changed page differs from frozen source case")
    if len(set(request.candidate_page_ids)) != len(request.candidate_page_ids) or not set(request.candidate_page_ids) <= set(case.page_ids) or page.page_id not in request.candidate_page_ids:
        raise ValueError("Candidates must be distinct case pages including the changed page")
    if len(set(request.fixed_context_page_ids)) != len(request.fixed_context_page_ids) or not set(request.fixed_context_page_ids) <= set(request.candidate_page_ids):
        raise ValueError("Fixed context must be distinct selected candidates")
    if request.stage == "generation_fixed_evidence" and page.page_id not in request.fixed_context_page_ids:
        raise ValueError("Fixed generation context must consume the changed page")
    selected = [e for e in elements if e.element_id in request.element_ids]
    if not any("heading" in e.element_type.lower() or "title" in e.element_type.lower() for e in selected) or not any("table" in e.element_type.lower() for e in selected):
        raise ValueError("The first family requires an explicitly reviewed heading and table group")
    with Image.open(BytesIO(raw)) as opened:
        if opened.size != (page.width, page.height) or opened.getexif().get(274, 1) != 1:
            raise ValueError("Source size/orientation differs")
        image = opened.convert("RGB")
    group = rectangle(elements, request.element_ids, image.size)
    dimensions = (group[2]-group[0], group[3]-group[1])
    control = rectangle(elements, request.non_support_element_ids, image.size, dimensions)
    if set(request.non_support_element_ids) & set(case.support_element_ids) or set(request.element_ids) & set(request.non_support_element_ids):
        raise ValueError("Matched control must be disjoint and contain no designated support elements")
    if min(group[2], control[2]) > max(group[0], control[0]) and min(group[3], control[3]) > max(group[1], control[1]):
        raise ValueError("Group and matched-control rectangles must not overlap")
    # Reject visible marks crossing crop boundaries even when unannotated.
    pixels = np.asarray(image)
    for box in (group, control):
        x1,y1,x2,y2 = box
        edge = np.concatenate((pixels[y1,x1:x2],pixels[y2-1,x1:x2],pixels[y1:y2,x1],pixels[y1:y2,x2-1]))
        if np.any(edge < 245):
            raise ValueError("Crop boundary crosses visible pixels; a clean whitespace boundary is required")
    baseline = Image.new("RGB", (page.width*2, page.height), "white"); baseline.paste(image, (0,0))
    identifier = "experiment_"+uuid4().hex
    arms=[]; assets={}
    for name, box, moving in (("baseline",None,[]),("render_control",None,[]),("variant",group,request.element_ids),("non_support_control",control,request.non_support_element_ids)):
        transformed=baseline.copy()
        if box:
            crop=image.crop(tuple(box)); transformed.paste("white",tuple(box)); transformed.paste(crop,(box[0]+page.width,box[1]))
            if not np.array_equal(np.asarray(transformed.crop((box[0]+page.width,box[1],box[2]+page.width,box[3]))),np.asarray(crop)):
                raise ValueError("Moved pixels were not preserved")
        encoded=png(transformed)
        if name=="render_control":
            with Image.open(BytesIO(encoded)) as reopened: encoded=png(reopened.convert("RGB"))
        digest=sha256(encoded).hexdigest(); assets[name]=encoded
        pid="page_"+sha256((identifier+name+page.page_id).encode()).hexdigest()
        new_width=page.width*2
        newpage=page.model_copy(update={"key":pid,"page_id":pid,"width":new_width,"image_sha256":digest,"image_url":f"/api/v1/experiments/{identifier}/arms/{name}/image","orientation":"square" if new_width==page.height else "landscape" if new_width>page.height else "portrait"})
        lineage={}; derived=[]
        for e in elements:
            eid="element_"+sha256((identifier+name+e.element_id).encode()).hexdigest(); lineage[e.element_id]=eid
            dx=page.width if e.element_id in moving else 0
            b=e.bbox; update={"key":eid,"element_id":eid,"page_id":pid}
            if dx:
                update.update(bbox=Box(x1=b.x1+dx,y1=b.y1,x2=b.x2+dx,y2=b.y2),display_bbox=Box(x1=b.x1+dx,y1=b.y1,x2=b.x2+dx,y2=b.y2))
            derived.append(e.model_copy(update=update))
        cid="case_"+uuid4().hex
        image_hashes={pid if k==page.page_id else k:v for k,v in case.page_image_sha256.items() if k in request.candidate_page_ids}; image_hashes[pid]=digest
        newcase=case.model_copy(update={"key":cid,"case_id":cid,"created_at":datetime.now(UTC).isoformat(),"page_ids":[pid if p==page.page_id else p for p in request.candidate_page_ids],"page_image_sha256":image_hashes,
            "support_page_ids":[pid if p==page.page_id else p for p in case.support_page_ids],"support_element_ids":[lineage.get(e,e) for e in case.support_element_ids],
            "annotation_version":sha256(canonical([e.model_dump(mode="json") for e in derived])).hexdigest(),"dependencies":[case.case_id,identifier],
            "provenance":{**case.provenance,"source_case_id":case.case_id,"experimental_arm":name,"native_refs":"retained source provenance; scoped IDs identify derived records"},"content_sha256":None})
        newcase=freeze(newcase)
        arms.append(ArmInput(name=name,case=newcase,page=newpage,elements=derived,asset_sha256=digest,pixel_sha256=sha256(np.asarray(transformed).tobytes()).hexdigest(),element_lineage=lineage,translation=[page.width,0] if box else [0,0],crop_box=box))
    record=ExperimentRecord(record_kind="experiment_draft",experiment_id=identifier,source_case_id=case.case_id,name=request.name,created_at=datetime.now(UTC).isoformat(),producer_version=__version__,dependencies=[case.case_id,page.page_id],
        provenance={"operation_version":VERSION,"source_asset_sha256":sha256(raw).hexdigest(),"source_case_sha256":case.content_sha256,"source_width":page.width,"source_height":page.height,"source_page":page.model_dump(mode="json"),"source_elements":[e.model_dump(mode="json") for e in elements],"source_case":case.model_dump(mode="json"),"transform_source_sha256":sha256(Path(__file__).read_bytes()).hexdigest()},
        protocol=request,arms=arms,validation={"crop_pixels_preserved":True,"no_annotated_content_cut":True,"clean_crop_boundaries":True,"render_control_pixels_equal":arms[0].pixel_sha256==arms[1].pixel_sha256,
            "control_matching":"identical crop dimensions, horizontal displacement, canvas, query, model and budgets; different original location/content", "family":"meaning_preserving_relayout",
            "changed_variables":["absolute group position","spatial distances","spatial ordering"],"shared_canvas":"white right margin doubles width in every arm; baseline is padded, not the original benchmark image",
            "limitations":["common padding changes processor scaling and may reduce readability","white replacement is conditional; unannotated content requires review","native reading order retained; image models consume pixels only","one deterministic run per arm; no uncertainty or general causal claim","answer correctness and attention subtraction are unavailable"]})
    return freeze(record),assets


def compare(outcomes, source_page_id, arms):
    base=outcomes[0]; results=[]
    for other in outcomes[1:]:
        result={"baseline":"baseline","other":other["arm"],"quantity":"paired_observed_task_outcomes","interpretation":"Effect under this declared image manipulation and controls; no isolated layout or causal grounding claim"}
        if "scores" in base:
            # Map derived changed-page identities back to the source before comparing.
            a={source_page_id if k==arms[0].page.page_id else k:v for k,v in base["scores"].items()}
            arm=next(x for x in arms if x.name==other["arm"])
            b={source_page_id if k==arm.page.page_id else k:v for k,v in other["scores"].items()}
            if set(a)!=set(b): raise ValueError("Paired retrieval candidate scopes differ")
            result.update(score_deltas={k:b[k]-a[k] for k in a},baseline_ranking=[k for k in sorted(a,key=lambda k:(-a[k],k))],other_ranking=[k for k in sorted(b,key=lambda k:(-b[k],k))])
        if "answer" in base:
            result.update(baseline_answer=base["answer"],other_answer=other["answer"],answer_text_changed=base["answer"]!=other["answer"],correctness="unevaluated; unchanged reference policy is recorded, not an accuracy score",attention_comparison="not computed: token/source/geometry alignment is required; inspect each actual row separately")
        results.append(result)
    return results
