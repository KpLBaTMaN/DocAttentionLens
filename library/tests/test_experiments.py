"""Known transformations; these fixtures are not model evidence."""
from hashlib import sha256
from io import BytesIO
import numpy as np
import pytest
from PIL import Image, ImageDraw
from docattentionlens.datasets.contracts import PageRecord, SourceRef, CaseRecord, ElementRecord, Box, OCRValue
from docattentionlens.experiments.contracts import ExperimentDraftRequest, Review
from docattentionlens.experiments.relayout import build_draft, compare, png


def fixture():
    ref=SourceRef(dataset_id='dataset',dataset_version='version',domain='fixture',document_id=1,page_number=1)
    image=Image.new('RGB',(300,300),'white'); draw=ImageDraw.Draw(image)
    draw.rectangle((30,30,80,40),fill='red'); draw.rectangle((30,60,100,100),fill='blue');draw.rectangle((170,30,195,50),fill='green')
    raw=png(image); h=sha256(raw).hexdigest()
    page=PageRecord(key='page',page_id='page',document_id='doc',ref=ref,page_number=1,width=300,height=300,image_sha256=h,image_url='fixture',orientation='square')
    es=[]
    for i,typ,box in ((1,'heading',[30,30,80,40]),(2,'table',[30,60,100,100]),(3,'text',[170,30,195,50])):
        es.append(ElementRecord(key=str(i),element_id='e'+str(i),page_id='page',document_id='doc',ref=ref,native_id=i,page_number=1,element_type=typ,element_category='fixture',bbox=Box(x1=box[0],y1=box[1],x2=box[2],y2=box[3]),level=None,reading_order=i,ocr=OCRValue(status='present',text=typ),geometry_flags=[],display_bbox=Box(x1=box[0],y1=box[1],x2=box[2],y2=box[3]),display_geometry_policy='raw'))
    case=CaseRecord(key='case',case_id='case',dataset_id='dataset',dataset_version='version',document_id='doc',question_id='question',query_variant='original',task_scope='document',query='fixture query',page_ids=['page'],page_image_sha256={'page':h},annotation_version='annotation',support_page_ids=['page'],support_element_ids=['e2'],reference_answer='fixture only',producer_version='test')
    req=ExperimentDraftRequest(case_id='case',page_id='page',element_ids=['e1','e2'],non_support_element_ids=['e3'],candidate_page_ids=['page'],fixed_context_page_ids=['page'],stage='retrieval_only',name='fixture')
    return req,case,page,raw,es


def test_intact_pixels_control_matching_and_element_lineage():
    record,assets=build_draft(*fixture())
    source=np.asarray(Image.open(BytesIO(fixture()[3])))
    base=np.asarray(Image.open(BytesIO(assets['baseline'])))
    moved=np.asarray(Image.open(BytesIO(assets['variant'])))
    assert np.array_equal(base[:,:300],source) and np.all(base[:,300:]==255)
    assert assets['baseline']==assets['render_control']
    assert record.arms[0].pixel_sha256==record.arms[1].pixel_sha256
    g=record.arms[2].crop_box;c=record.arms[3].crop_box
    assert g[2]-g[0]==c[2]-c[0] and g[3]-g[1]==c[3]-c[1]
    assert np.array_equal(moved[g[1]:g[3],g[0]+300:g[2]+300],source[g[1]:g[3],g[0]:g[2]])
    assert np.all(moved[g[1]:g[3],g[0]:g[2]]==255)
    changed=record.arms[2]; heading=next(e for e in changed.elements if e.native_id==1)
    assert heading.bbox.x1==330 and heading.native_id==1 and heading.reading_order==1
    assert changed.element_lineage['e1']==heading.element_id and changed.case.support_element_ids==[changed.element_lineage['e2']]
    assert changed.case.page_image_sha256[changed.page.page_id]==sha256(assets['variant']).hexdigest()


def test_clipping_overlap_support_and_unreviewed_geometry_refused():
    req,case,page,raw,es=fixture()
    for broken in (req.model_copy(update={'element_ids':['e1']}),req.model_copy(update={'non_support_element_ids':['e2']}),req.model_copy(update={'candidate_page_ids':['page','page']})):
        with pytest.raises(ValueError):build_draft(broken,case,page,raw,es)
    es[0]=es[0].model_copy(update={'geometry_flags':['out_of_bounds']})
    with pytest.raises(ValueError,match='Flagged'):build_draft(req,case,page,raw,es)
    req,case,page,raw,es=fixture();es[2]=es[2].model_copy(update={'bbox':Box(x1=90,y1=50,x2=180,y2=80)})
    with pytest.raises(ValueError,match='cut'):build_draft(req,case,page,raw,es)


def test_full_review_required_and_comparisons_use_explicit_page_lineage():
    with pytest.raises(ValueError):Review(reviewer='test',notes='fixture notes',relationship_confirmed=False,content_and_meaning_preserved=True,readable=True,answer_policy='unchanged_after_review',controls_and_resize_limitations_accepted=True)
    record,_=build_draft(*fixture())
    outcomes=[{'arm':a.name,'scores':{a.page.page_id:float(i),'unchanged':-1},'answer':'fixture'} for i,a in enumerate(record.arms)]
    pairs=compare(outcomes,'page',record.arms)
    assert pairs[1]['score_deltas']=={'page':2.,'unchanged':0.}
    assert pairs[1]['answer_text_changed'] is False and 'unevaluated' in pairs[1]['correctness']
    outcomes[1]['scores']['extra']=5
    with pytest.raises(ValueError,match='scopes'):compare(outcomes,'page',record.arms)
