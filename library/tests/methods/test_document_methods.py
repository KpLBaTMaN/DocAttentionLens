"""Known mathematical document fixtures; never scientific model results."""
from dataclasses import replace
from types import SimpleNamespace as NS
import numpy as np
import pytest
from docattentionlens.methods.contracts import DocumentInputs
from docattentionlens.methods.registry import get_method,execute
from docattentionlens.methods.retrieval_attention_agreement import spearman
from test_saved_rows import fixture


def document_fixture(identifier="generation_"+"a"*32,question="case_a"):
    context=fixture(); record=context.record; record.capture_id=identifier
    record.positions[0].ownership={"element_a":.5,"element_b":.5}
    record.positions[1].ownership={"element_b":1}
    for i in (0,1): record.positions[i].bbox=[i,0,i+1,1]
    record.context.pages=[NS(page_id="page_a",page_number=1,source_width=2,source_height=1,resized_width=20,resized_height=10)]
    record.context.selected_page_ids=["page_a"]; record.context.query="question"
    record.output_token_ids=[7]; record.steps[0].consumed_token_id=3; record.steps[0].predicted_piece="Answer"
    record.provenance={"model_id":"fixture","model_revision":"v1"}; record.case_id=question
    record.retrieval_capture_id=None; record.answer="True"
    page=NS(page_id="page_a",page_number=1,width=2,height=1)
    def element(i,kind,order,box,level=0):
        return NS(element_id=i,page_id="page_a",native_id=i,element_type=kind,element_category="TABLE" if kind=="table" else "TEXT",
                  reading_order=order,level=level,bbox=NS(x1=box[0],y1=box[1],x2=box[2],y2=box[3]),geometry_flags=[])
    elements=(element("element_a","heading",1,[0,0,1,1]),element("element_b","table",2,[1,0,2,1]),element("element_c","text",None,[0,0,1,1]))
    case=NS(case_id=question,document_id=question,query="question "+question,support_element_ids=["element_b","element_c"],reference_answer="True",question_type="true_or_false")
    return replace(context,document=DocumentInputs(case,elements,(page,),"fixture_annotation",question_type="true_or_false"))


def run(identifier,context=None,**parameters):
    method=get_method(identifier)
    return execute(identifier,method.variant_id,{"head":2,**parameters},context or document_fixture())


def test_type_area_support_order_and_size_reconstruct_known_values():
    result=run("attention.element_type_profile")
    assert result.table.labels==["heading","table"]
    assert np.asarray(result.table.values)==pytest.approx(np.array([[.125,.25,.25,1],[.375,.75,.75,1]]))
    assert result.tables[0].values==[[.5]]
    hit=run("attention.designated_support",top_k=1)
    stats={s.name:s.value for s in hit.statistics}
    assert stats["Designated support mass"]==.375 and stats["Precision at selected k"]==1
    assert stats["Unmeasured support elements"]==1
    order=run("attention.reading_order")
    assert order.table.element_ids==["element_a","element_b"]
    assert [p.y for p in order.plots[0].points]==[.125,.375]
    size=run("attention.model_input_legibility")
    assert size.table.values[0][:3]==[10,10,100]


def test_layout_and_inferred_sections_do_not_double_count():
    # No header, footer or page-number element is measured: unavailable, not zero.
    assert run("attention.layout_sinks").table.values==[[None,0],[None,0],[None,0],[0,None]]
    context=document_fixture(); context.document.elements[0].element_type="PageHeader"
    assert run("attention.layout_sinks",context).table.values[0]==[.125,1]
    sections=run("attention.section_rollup")
    assert sections.table.values==[[.5,.5]]
    assert sections.tables[0].values==[[None],[.5]]
    assert "inferred" in sections.notices[0].lower()


@pytest.mark.parametrize("identifier",["attention.element_type_profile","attention.designated_support","attention.reading_order",
                                       "attention.layout_sinks","attention.section_rollup","attention.model_input_legibility"])
def test_single_row_document_methods_return_their_attention_row(identifier):
    result=run(identifier)
    assert result.row is not None and result.row.head==2 and result.row.total_mass==pytest.approx(1)
    assert {a.element_id for a in result.row.allocations if a.element_id}=={"element_a","element_b"}


def test_unmeasured_designated_support_is_unavailable_not_zero():
    context=document_fixture(); context.document.case.support_element_ids=["element_c"]
    result=run("attention.designated_support",context)
    stats={s.name:s.value for s in result.statistics}
    assert stats["Designated support mass"] is None and stats["Precision at selected k"] is None
    assert stats["Other element mass"]==.5 and stats["Measured support elements"]==0 and stats["Unmeasured support elements"]==1
    assert result.table.values==[] and result.tables==[]
    assert "unavailable, not zero" in result.notices[0]


def test_sections_without_measured_elements_or_headings_are_explained():
    context=document_fixture()
    later=NS(element_id="element_d",page_id="page_a",native_id="element_d",element_type="heading",element_category="TEXT",
             reading_order=3,level=0,bbox=NS(x1=1,y1=0,x2=2,y2=1),geometry_flags=[])
    context=replace(context,document=replace(context.document,elements=(*context.document.elements,later)))
    sections=run("attention.section_rollup",context)
    assert sections.table.element_ids==["element_a","element_d"]
    assert sections.table.values==[[.5,.5],[None,None]]
    context=document_fixture(); context.document.elements[0].element_type="text"
    flat=run("attention.section_rollup",context)
    assert flat.table.values==[] and flat.tables[0].values==[[.5],[.5]]
    assert "No heading" in flat.notices[0] and "Outside inferred sections" in flat.notices[0]
    order=run("attention.reading_order")
    assert order.tables[0].values==[[None],[.5]]


def test_layer_budget_includes_all_valid_roles_and_spatial_map_conserves():
    depth=run("attention.modality_budget_depth")
    assert all(sum(values)==pytest.approx(1) for values in depth.table.values)
    spatial=run("attention.spatial_prior_map",grid_size=2)
    assert sum(p.value for p in spatial.plots[0].points)==pytest.approx(1)
    assert all(v[2]>=0 for v in spatial.table.values)


def test_comparison_rejects_diverged_history_and_reports_separate_statistics():
    a=document_fixture(); b=document_fixture("generation_"+"b"*32)
    result=run("attention.aligned_comparison",replace(a,comparison=b))
    stats={s.name:s.value for s in result.statistics}
    assert stats["Jensen–Shannon divergence on aligned element support"]==0
    assert stats["Top-k overlap fraction"]==1
    assert all(v[2]==0 for v in result.table.values)
    b.record.steps[0].consumed_token_id=8
    with pytest.raises(ValueError,match="sources diverge"):run("attention.aligned_comparison",replace(a,comparison=b))


@pytest.mark.parametrize("identifier",["attention.aligned_comparison","layout.move_group"])
def test_aligned_rows_are_readable_elements_with_pages(identifier):
    a=document_fixture(); b=document_fixture("generation_"+"b"*32)
    result=run(identifier,replace(a,comparison=b))
    assert result.table.labels==["heading · element_a","table · element_b"]
    assert result.table.element_ids==["element_a","element_b"]
    assert result.table.page_ids==["page_a","page_a"]


def test_disjoint_aligned_distributions_have_one_bit_divergence_and_no_top_one_overlap():
    a=document_fixture(); b=document_fixture("generation_"+"b"*32)
    for c in (a,b):c.record.positions[0].ownership={"element_a":1}
    a.values[:]=[1,0,0,0];b.values[:]=[0,1,0,0]
    result=run("attention.aligned_comparison",replace(a,comparison=b),top_k=1)
    stats={s.name:s.value for s in result.statistics}
    assert stats["Jensen–Shannon divergence on aligned element support"]==1
    assert stats["Top-k overlap fraction"]==0
    assert sorted(v[2] for v in result.table.values)==[-1,1]


def test_unmeasured_pages_never_become_zero_sections_and_axes_are_strict():
    from docattentionlens.methods.element_allocation import allocate_row
    context=document_fixture();context.record.context.input_attention_mask[1]=0;context.values[:]=[.5,0,0,.5]
    result=run("attention.element_type_profile",context)
    assert sum(v[0] for v in result.table.values)==.5
    with pytest.raises(ValueError):run("attention.head_specialisation",context,heads=[True])
    # A causal encoding source cannot measure elements that occur only in later keys.
    rec=NS(capture_id="capture_"+"a"*32,encoding_attention=True,causal=True,attention_mask=[1,1],
           positions=[NS(index=0,role="special",page_id=None,ownership={"non_page":1}),NS(index=1,role="image",page_id="page_a",ownership={"element_a":1})])
    step=NS(source_index=0,predicted_index=None,target_count=2)
    row=allocate_row(rec,np.array([1.,0.]),step,0,0,0)
    assert not any(a.element_id for a in row.allocations)


def test_rank_correlation_ties_constant_and_negative_scores():
    assert spearman([-2,-1,3],[.1,.2,.3])==pytest.approx(1)
    assert spearman([1,1,2],[2,2,1])==pytest.approx(-1)
    assert spearman([1,1],[1,2]) is None
    context=document_fixture(); context.record.retrieval_capture_id="retrieval"
    retrieval=NS(capture_id="retrieval",pages=[NS(contributions={"element_a":-1.,"element_b":2.},ownership=[{"element_a":1},{"element_b":1}])])
    assert run("attention.retrieval_agreement",replace(context,retrieval=retrieval)).statistics[0].value==pytest.approx(1)


def retrieval_agreement(contributions,ownership):
    context=document_fixture(); context.record.retrieval_capture_id="retrieval"
    retrieval=NS(capture_id="retrieval",pages=[NS(contributions=contributions,ownership=ownership)])
    result=run("attention.retrieval_agreement",replace(context,retrieval=retrieval))
    return {s.name:s.value for s in result.statistics},result.table


def test_retrieval_owned_elements_without_a_won_token_are_measured_zero():
    # Generator mass: element_a .125, element_b .375. element_b owns a retrieval patch but wins no token.
    stats,table=retrieval_agreement({"element_a":3.},[{"element_a":1},{"element_b":1}])
    assert stats["Aligned measured elements"]==2 and stats["Spearman rank correlation"]==pytest.approx(-1)
    assert table.values==[[3.,.125],[0.,.375]]
    assert table.labels==["heading · element_a","table · element_b"] and table.page_ids==["page_a","page_a"]
    # Without retrieval ownership an element is missing, never zero.
    stats,table=retrieval_agreement({"element_a":3.},[{"element_a":1},{"background":1}])
    assert stats["Aligned measured elements"]==1 and stats["Spearman rank correlation"] is None
    assert table.element_ids==["element_a"]


def with_unsupplied_figure(context,supplied=("page_a",)):
    """Add an element type that exists only on a page this capture did not supply."""
    figure=NS(element_id="element_f",page_id="page_b",native_id="element_f",element_type="figure",element_category="FIGURE",
              reading_order=None,level=0,bbox=NS(x1=0,y1=0,x2=1,y2=1),geometry_flags=[])
    context.record.context.selected_page_ids=list(supplied)
    page=NS(page_id="page_b",page_number=2,width=2,height=1)
    return replace(context,document=replace(context.document,elements=(*context.document.elements,figure),pages=(*context.document.pages,page)))


def test_type_vocabulary_excludes_types_only_on_unsupplied_pages():
    a=with_unsupplied_figure(document_fixture()); b=with_unsupplied_figure(document_fixture("generation_"+"b"*32,"case_b"))
    sample=replace(a,samples=(a,b))
    types=["heading","table","text"]
    heads=run("attention.head_specialisation",sample,layers=[0],heads=[2])
    assert heads.table.columns==["Layer","Head",*types,"Background/unreviewed image","Top-share gap"]
    atlas=run("attention.question_atlas",sample)
    assert atlas.table.columns==[*types,"Background/unreviewed image"]
    similarity=run("heads.profile_similarity",sample,layers=[0],heads=[0,2])
    assert similarity.tables[0].columns==[*types,"Background/unreviewed image"]
    # A type supplied to any sample is a measured column for every sample.
    b=with_unsupplied_figure(document_fixture("generation_"+"b"*32,"case_b"),("page_a","page_b"))
    atlas=run("attention.question_atlas",replace(a,samples=(a,b)))
    assert atlas.table.columns==["figure",*types,"Background/unreviewed image"]


def test_evidence_matrix_funnel_head_profiles_and_atlas():
    a=document_fixture(); b=document_fixture("generation_"+"b"*32,"case_b")
    sample=replace(a,samples=(a,b))
    matrix=run("attention.evidence_modality_matrix",sample)
    assert sum(v[-1] for v in matrix.table.values)==pytest.approx(2)
    funnel=run("attention.evidence_funnel",sample)
    assert funnel.table.values[0]==[2,None,None,2,1,1,1]
    heads=run("attention.head_specialisation",sample,layers=[0],heads=[2])
    assert sum(heads.table.values[0][2:-1])==pytest.approx(1)
    atlas=run("attention.question_atlas",sample)
    assert len(atlas.plots[0].points)==2
    assert all(p.x==0 and p.y==0 for p in atlas.plots[0].points)
    b.record.provenance["model_revision"]="different"
    with pytest.raises(ValueError,match="same model"):run("attention.question_atlas",sample)


def test_bridge_keeps_signed_retrieval_and_attention_edges_distinct():
    context=document_fixture(); context.record.retrieval_capture_id="retrieval"
    retrieval=NS(capture_id="retrieval",query_positions=[NS(index=0,piece="word",eligible=True)],
                 pages=[NS(winner_indices=[0],query_maxima=[-2.],ownership=[{"element_a":.5,"element_b":.5}])])
    result=run("attention.query_answer_bridge",replace(context,retrieval=retrieval))
    assert sum(e.value for e in result.edges if e.signal=="retrieval")==-2
    assert sum(e.value for e in result.edges if e.signal=="attention")==.5


def test_missing_annotations_are_errors_and_zero_image_enrichment_is_unavailable():
    with pytest.raises(ValueError,match="Verified case"):run("attention.element_type_profile",fixture())
    context=document_fixture(); context.values[:]=[0,0,.5,.5]
    result=run("attention.element_type_profile",context)
    assert all(v[1] is None and v[3] is None for v in result.table.values)


def profile(context=None):
    return execute("attention.element_profile",get_method("attention.element_profile").variant_id,{},context or document_fixture())


def test_element_profile_cells_match_element_allocation_support_and_head_mean():
    result=profile(); data=result.element_profile
    assert (data.layer_count,data.head_count)==(2,3)
    assert [e.element_id for e in data.elements]==["element_a","element_b"]
    assert data.unmeasured_element_ids==["element_c"]
    for layer in range(2):
        for head in range(3):
            row=run("attention.element_allocation",layer=layer,head=head).row
            allocated={a.element_id:a.mass for a in row.allocations if a.element_id}
            for element in data.elements:
                assert element.mass[layer][head]==pytest.approx(allocated[element.element_id],abs=1e-7)
            assert data.image_mass[layer][head]==pytest.approx(row.image_mass,abs=1e-7)
        mean=execute("attention.mean_heads",get_method("attention.mean_heads").variant_id,{"layer":layer,"heads":[0,1,2]},document_fixture()).row
        averaged={a.element_id:a.mass for a in mean.allocations if a.element_id}
        for element in data.elements:
            assert np.mean(element.mass[layer])==pytest.approx(averaged[element.element_id],abs=1e-7)
    # Designated support (element_b, plus unmapped element_c) is the summed support column.
    support=run("attention.designated_support").tables[0].values
    assert np.asarray(support)==pytest.approx(np.asarray(data.elements[1].mass),abs=1e-7)


def test_element_profile_area_ratio_peaks_and_source_only_parameters():
    data=profile().element_profile
    assert data.image_positions==2
    assert [e.area_share for e in data.elements]==pytest.approx([.25,.75])
    # Layer 0, head 2 attends uniformly: each element's within-image share equals its area share.
    a=data.elements[0]
    assert a.mass[0][2]/data.image_mass[0][2]/a.area_share==pytest.approx(1)
    peaks=profile().table
    assert peaks.labels==["heading · element_a","table · element_b"] and peaks.element_ids==["element_a","element_b"]
    assert peaks.values[1][:3]==pytest.approx([1,0,1])
    assert set(get_method("attention.element_profile").parameters_schema["properties"])=={"step"}
    with pytest.raises(Exception):
        execute("attention.element_profile",get_method("attention.element_profile").variant_id,{"layer":1},document_fixture())


def test_element_profile_missing_mapping_area_and_budget():
    context=document_fixture()
    for p in context.record.positions[:2]: p.role="query"
    data=profile(context).element_profile
    assert data.image_positions==0 and all(e.area_share is None for e in data.elements)
    unmapped=document_fixture()
    for p in unmapped.record.positions: p.ownership={"background":1}
    with pytest.raises(ValueError,match="No annotated element"):profile(unmapped)
    foreign=document_fixture(); foreign.record.positions[0].ownership={"element_z":1}
    with pytest.raises(ValueError,match="verified annotation"):profile(foreign)
    import docattentionlens.methods.element_profile as module
    original=module.PROFILE_BUDGET; module.PROFILE_BUDGET=5
    try:
        with pytest.raises(ValueError,match="bounded to 5"):profile()
    finally:
        module.PROFILE_BUDGET=original
