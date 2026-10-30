"""Independent mathematical oracles for the attention-only release; no model claims."""
from dataclasses import replace
from itertools import combinations
from types import SimpleNamespace as NS
import numpy as np
import pytest
from docattentionlens.methods.registry import get_method,list_methods,execute
from docattentionlens.methods.head_clustering import ward,silhouette
from docattentionlens.methods._max_flow import maximum_flow
from docattentionlens.retrieval.contracts import EncodingAttention,Position
from test_planned_methods import document_encoding,encoded,query_encoding
from test_saved_rows import fixture as prediction_fixture

IDS={"attention.whole_query","attention.text_to_text","attention.text_to_vision","attention.vision_to_text","attention.vision_to_vision",
     "attention.spatial_direction","attention.local_global","attention.row_structure","attention.column_structure","attention.sink_detection",
     "heads.clustering","heads.pattern_taxonomy","attention.partial_rollout","attention.rollout_evolution","attention.graph","attention.flow"}
INTERVENTIONS={"attention.head_ablation","attention.layer_ablation","attention.edge_ablation","attention.modality_ablation","attention.region_masking"}

def values(result):
    return {s.name:s.value for s in result.statistics}

def doc(heads=2,layers=2):
    return document_encoding(0,grid=2,prompt=2,heads=heads,layers=layers)[:2]

def test_all_21_owners_are_registered_and_correctly_classified():
    registered={m.method_id:m for m in list_methods()}
    assert IDS|INTERVENTIONS<=set(registered)
    assert all(registered[i].perspective=="general" and registered[i].status=="implemented" for i in IDS|INTERVENTIONS)
    assert all(registered[i].family=="attention_interventions" and registered[i].evidence_kind=="internal_intervention_effect" for i in INTERVENTIONS)
    assert registered["attention.graph"].family=="attention_graphs"
    assert all(registered[i].family=="multimodal_attention" for i in IDS if "_to_" in i)

@pytest.mark.parametrize("identifier",sorted(IDS))
def test_every_saved_method_executes_on_compatible_document_matrices(identifier):
    record,arrays=doc()
    source=5 if identifier in ("attention.text_to_text","attention.text_to_vision") else 0
    result=encoded(record,arrays,"document",identifier,source)
    assert result is not None
    assert len(result.model_dump_json())<4*1024**2
    assert any((result.row,result.comparisons,result.statistics,result.table,result.graph))

def test_whole_query_conditions_each_source_before_equal_source_mean():
    record,arrays=doc()
    for a in arrays.values():
        a[:,4]=[.1,.1,0,0,.8,0]
        a[:,5]=[0,0,.4,.4,.2,0]
    result=encoded(record,arrays,"document","attention.whole_query",5)
    assert result.row.values==pytest.approx([.25,.25,.25,.25,0,0])
    assert result.source_indices==[4,5]
    assert result.row.source_index is None and result.row.predicted_index is None
    assert np.allclose(result.table.values,[[.2,1],[.8,1]])
    assert any("not question-conditioned" in n for n in result.notices)
    envelope=encoded(record,arrays,"document","attention.whole_query",5,token_reduction="max")
    assert envelope.row.total_mass==pytest.approx(2)
    assert any("envelope" in n for n in envelope.notices)

def test_generator_multiple_question_rows_use_causal_padding_and_preserve_original_input():
    base=prediction_fixture(); record=base.record
    record.positions[3]=record.positions[3].model_copy(update={"role":"query"})
    record.source_rows=[NS(source_index=2),NS(source_index=3)]
    source2=np.broadcast_to(np.array([1,0,0],np.float32),(2,3,3)).copy()
    source3=np.broadcast_to(np.array([0,.5,.25,.25],np.float32),(2,3,4)).copy()
    original=source2.copy()
    context=replace(base,values=source3,source_record=record,
        read_source=lambda i:(source2 if i==2 else source3,np.ones(3 if i==2 else 4,dtype=bool)))
    method=get_method("attention.whole_query")
    result=execute(method.method_id,method.variant_id,{},context)
    assert result.row.values==pytest.approx([.5,.5,0,0])
    assert result.source_indices==[2,3]
    assert np.array_equal(source2,original)

def test_modality_block_retains_original_mass_and_reports_separate_shares():
    record,arrays=doc()
    arrays["l0"][:,5]=[0,0,.4,.4,.2,0]
    result=encoded(record,arrays,"document","attention.text_to_vision",5)
    assert result.row.total_mass==pytest.approx(.8)
    assert result.row.values==pytest.approx([0,0,.4,.4,0,0])
    assert np.asarray(result.table.values)[:,1].tolist()==pytest.approx([0,0,.5,.5])
    with pytest.raises(ValueError,match="textual"):encoded(record,arrays,"document","attention.text_to_vision",0)
    with pytest.raises(ValueError,match="image-patch"):encoded(record,arrays,"document","attention.vision_to_text",5)
    arrays["l0"][:,5]=[0,0,0,0,1,0]
    zero=encoded(record,arrays,"document","attention.text_to_vision",5)
    assert zero.row.total_mass==0
    assert all(v[1] is None for v in zero.table.values)

@pytest.mark.parametrize("causal",[False,True])
def test_partial_and_evolution_match_explicit_matrix_products(causal):
    record,arrays,mask=query_encoding(2,causal=causal)
    source=7
    valid=np.flatnonzero(mask)
    matrices=[]
    for l in range(3):
        m=arrays[f"l{l}"].mean(0,dtype=np.float64)[np.ix_(valid,valid)]
        m=.4*np.eye(len(valid))+.6*m
        matrices.append(m/m.sum(-1,keepdims=True))
    expected=np.zeros(len(mask)); expected[valid]=(matrices[2]@matrices[1])[valid.tolist().index(source)]
    partial=encoded(record,arrays,"query","attention.partial_rollout",source,layer=2,start_layer=1,residual_coefficient=.4)
    assert partial.row.values==pytest.approx(expected,abs=1e-9)
    evolution=encoded(record,arrays,"query","attention.rollout_evolution",source,layers=[1,2],start_layer=1,residual_coefficient=.4)
    assert evolution.comparisons[1].row.values==pytest.approx(expected,abs=1e-9)
    with pytest.raises(ValueError,match="interval"):encoded(record,arrays,"query","attention.partial_rollout",source,layer=0,start_layer=1)

def test_spatial_identity_and_uniform_baselines_exclude_self():
    record,arrays=doc()
    for a in arrays.values(): a[:]=np.eye(6,dtype=np.float32)[None]
    assert values(encoded(record,arrays,"document","attention.local_global",0,radius=0))["Local attention mass"]==1
    assert values(encoded(record,arrays,"document","attention.row_structure",0))["Same-row attention mass"]==0
    assert values(encoded(record,arrays,"document","attention.column_structure",0))["Same-column attention mass"]==0
    for a in arrays.values():
        a[:,:4]=0; a[:,:4,:4]=.25
    row=encoded(record,arrays,"document","attention.row_structure",0)
    column=encoded(record,arrays,"document","attention.column_structure",0)
    assert values(row)["Same-row attention mass"]==pytest.approx(.25)
    assert values(column)["Same-column attention mass"]==pytest.approx(.25)
    assert np.asarray(row.table.values)[:,1].tolist()==pytest.approx([.25]*4)
    assert values(encoded(record,arrays,"document","attention.local_global",0,radius=1))["Local attention mass"]==1

def test_spatial_direction_on_pure_horizontal_edges_is_one():
    record,arrays=doc()
    for a in arrays.values():
        a[:,:4]=0
        for source,target in ((0,1),(1,0),(2,3),(3,2)): a[:,source,target]=1
    result=encoded(record,arrays,"document","attention.spatial_direction",0)
    assert values(result)["Directional concentration"]==1
    assert result.table.values[0][4]==1
    assert result.table.values[1][8]==1

def test_sink_detector_is_not_selected_role_mass():
    record,arrays=doc()
    for a in arrays.values(): a[:]=0; a[:,:,0]=1
    result=encoded(record,arrays,"document","attention.sink_detection",0)
    assert values(result)["Strongest sink score"]==1
    assert values(result)["Layer sink-head rate"]==1
    assert result.source_indices==list(range(6))
    assert values(encoded(record,arrays,"document","attention.sinks",0))["Declared-group attention mass"]==0

def test_clustering_groups_identical_heads_and_taxonomy_discloses_heuristics():
    record,arrays=doc(heads=3)
    for a in arrays.values():
        a[:]=0; a[0,:,0]=1; a[1,:,0]=1; a[2,:,3]=1
    result=encoded(record,arrays,"document","heads.clustering",0,layers=[0])
    labels=[v[0] for v in result.table.values]
    assert labels[0]==labels[1] and labels[2]!=labels[0]
    assert values(result)["Mean silhouette"]==pytest.approx(2/3)
    taxonomy=encoded(record,arrays,"document","heads.pattern_taxonomy",0,layers=[0])
    assert all("vertical" in label for label in taxonomy.table.labels)
    assert all(v[0]==1 for v in taxonomy.table.values)
    assert any("threshold" in n for n in taxonomy.notices)
    linkage,cuts=ward(np.array([[0,.1,1],[.1,0,1],[1,1,0]]))
    assert linkage[0][:2]==[0,1]
    assert silhouette(np.array([[0,.1,1],[.1,0,1],[1,1,0]]),cuts[2])==pytest.approx(.6)

def test_graph_keeps_raw_edges_identities_and_deterministic_ties():
    record,arrays=doc()
    for a in arrays.values(): a[:]=np.eye(6)[None]
    result=encoded(record,arrays,"document","attention.graph",0,layers=[0],threshold=.9,max_edges=3)
    assert result.graph.total_edges==12
    assert [(e.layer,e.head,e.source,e.target) for e in result.graph.edges]==[
        (0,0,"L1:P0","L0:P0"),(0,0,"L1:P1","L0:P1"),(0,0,"L1:P2","L0:P2")]
    assert all(e.value==1 for e in result.graph.edges)

def min_cut(edges,source,target):
    interior=sorted({v for a,b,_ in edges for v in (a,b)}-{source,target})
    return min(sum(c for a,b,c in edges if a in ({source}|set(subset)) and b not in ({source}|set(subset)))
        for k in range(len(interior)+1) for subset in combinations(interior,k))

def test_float_maxflow_matches_exhaustive_min_cut():
    edges=[("s","a",.5),("s","b",.5),("a","t",.8),("b","t",.4),("a","b",.1)]
    assert maximum_flow(edges,"s","t")==pytest.approx(min_cut(edges,"s","t"))
    assert maximum_flow(edges,"s","unused")==0

def test_flow_exact_small_graph_and_pruning_lower_bound_are_not_probabilities():
    enc=EncodingAttention(encoding_id="query",pass_kind="query",layer_count=2,query_head_count=1,
        position_count=2,attention_mask=[1,1],layer_arrays=["l0","l1"])
    record=NS(capture_id="fixture",attention_encodings=[enc],pages=[],
        query_positions=[Position(index=i,token_id=i,piece=str(i),role="query_text",eligible=True) for i in range(2)])
    arrays={"l0":np.array([[[.8,.2],[.4,.6]]],np.float32),"l1":np.array([[[.6,.4],[.5,.5]]],np.float32)}
    exact=encoded(record,arrays,"query","attention.flow",1,layer=1,residual_coefficient=0,mass_coverage=1,max_edges_per_node=2,target_indices=[0,1])
    flow=np.asarray(exact.table.values)[:,1]
    assert flow==pytest.approx([.9,.7],abs=1e-7)
    assert flow.sum()>1
    pruned=encoded(record,arrays,"query","attention.flow",1,layer=1,residual_coefficient=0,mass_coverage=1,max_edges_per_node=1,target_indices=[0,1])
    assert np.all(np.asarray(pruned.table.values)[:,1]<=flow+1e-9)
    assert values(pruned)["Minimum per-node retained capacity"]<1
    assert any("unmeasured" in n for n in pruned.notices)

def test_full_matrix_methods_refuse_generator_rows():
    context=prediction_fixture()
    for identifier in ("attention.graph","attention.flow","attention.partial_rollout","attention.spatial_direction"):
        method=get_method(identifier)
        with pytest.raises(ValueError,match="complete saved attention matrices|has no attribute"):
            execute(identifier,method.variant_id,{},context)
