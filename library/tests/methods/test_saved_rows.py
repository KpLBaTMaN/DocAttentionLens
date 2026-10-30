"""Mathematical fixtures, never model captures or scientific validation evidence."""
from types import SimpleNamespace as NS
import numpy as np
import pytest
from docattentionlens.generation.contracts import GenerationPosition
from docattentionlens.methods.contracts import Context
from docattentionlens.methods.registry import execute, get_method, prepare_parameters


def fixture():
    positions = [GenerationPosition(index=i, token_id=i, piece=str(i), role=role,
        page_id="page_a" if role=="image" else None,
        ownership={"element_a":.75,"background":.25} if role=="image" else {"non_image:"+role:1})
        for i,role in enumerate(["image","image","query","special"])]
    record = NS(capture_id="generation_"+"a"*32, layer_count=2,query_head_count=3,
        positions=positions,steps=[NS(step=0,target_count=4,source_index=3,predicted_index=4,array="step_0")],
        context=NS(input_attention_mask=[1]*4,input_token_ids=list(range(4))),
        coverage={"all_causal_targets":True,"quantity":"post_softmax","ownership":"reviewed"})
    values=np.array([[[1,0,0,0],[0,1,0,0],[.25,.25,.25,.25]],
                     [[.1,.2,.3,.4],[.4,.3,.2,.1],[.25,.25,.25,.25]]],dtype=np.float32)
    return Context(record,values)


def run(identifier, context=None, **parameters):
    descriptor=get_method(identifier)
    return execute(identifier,descriptor.variant_id,parameters,context or fixture())


def stats(result):
    return {s.name:s.value for s in result.statistics}


def test_head_means_preserve_mass_ownership_and_input_and_ignore_head_order():
    context=fixture(); original=context.values.copy()
    mean=run("attention.mean_heads",context,heads=[0,1]).row
    reversed_mean=run("attention.mean_heads",context,heads=[1,0]).row
    assert mean.values==reversed_mean.values
    assert mean.total_mass==pytest.approx(1)
    assert sum(a.mass for a in mean.allocations)==pytest.approx(mean.total_mass)
    assert mean.image_mass==pytest.approx(1)
    assert mean.allocation_error<1e-8
    assert np.array_equal(context.values,original)
    weighted=run("attention.weighted_heads",context,heads=[0,1],weights=[1,1]).row
    assert weighted.values==mean.values
    selected=run("attention.weighted_heads",context,heads=[0,1],weights=[1,0]).row
    assert selected.values==run("attention.row",context,head=0).row.values
    huge=run("attention.weighted_heads",context,heads=[0,1],weights=[1e308,1e308]).row
    assert huge.values==mean.values


@pytest.mark.parametrize("weights",[[0,0],[-1,2],[1],[float('inf'),1],[float('nan'),1]])
def test_invalid_weighted_averages_are_rejected(weights):
    with pytest.raises(ValueError):run("attention.weighted_heads",heads=[0,1],weights=weights)


@pytest.mark.parametrize("heads",[[0,0],[3],[-1],[True],[1.5]])
def test_invalid_or_ambiguous_head_selections_are_rejected(heads):
    with pytest.raises(ValueError):run("attention.mean_heads",heads=heads)


def test_envelopes_are_not_probabilities_and_keep_conservation():
    maximum=run("attention.max_heads",heads=[0,1]).row
    minimum=run("attention.min_heads",heads=[0,1]).row
    assert maximum.total_mass==2
    assert minimum.total_mass==0
    assert maximum.quantity=="head_maximum_envelope"
    assert minimum.quantity=="head_minimum_envelope"
    assert "sums need not equal one" in maximum.normalization
    assert sum(a.mass for a in maximum.allocations)==maximum.total_mass


def test_entropy_concentration_and_valid_zero_weights_have_known_limits():
    uniform=stats(run("attention.entropy",head=2))
    point=stats(run("attention.entropy",head=0))
    assert uniform["Shannon entropy"]==2 and uniform["Normalized entropy"]==1
    assert point["Shannon entropy"]==0 and point["Valid targets"]==4
    concentration=stats(run("attention.concentration",head=2))
    assert concentration["Gini concentration"]==0
    assert concentration["Inverse Simpson effective support"]==4
    point=stats(run("attention.concentration",head=0))
    assert point["Gini concentration"]==.75
    assert point["Inverse Simpson effective support"]==1


def test_masks_come_from_capture_and_positive_weight_does_not_define_support():
    context=fixture();context.record.context.input_attention_mask[3]=0
    context.values[:]=[.5,.5,0,0]
    result=stats(run("attention.entropy",context,head=0))
    assert result["Valid targets"]==3
    assert result["Normalized entropy"]==pytest.approx(1/np.log2(3))
    with pytest.raises(ValueError,match="this source's 3 valid target positions"):run("attention.topk_mass",context,k=4)
    context.values[0,0]=[.5,0,0,.5]
    with pytest.raises(ValueError,match="masked support"):run("attention.entropy",context)


def test_topk_is_monotonic_and_bounded_by_valid_support():
    masses=[next(s.value for s in run("attention.topk_mass",head=2,k=k).statistics if s.name.startswith("Top-")) for k in range(1,5)]
    assert masses==sorted(masses) and masses[-1]==1
    with pytest.raises(ValueError,match="this source's 4 valid target positions"):run("attention.topk_mass",k=5)
    with pytest.raises(ValueError):run("attention.topk_mass",k=0)


def test_role_enrichment_reconciles_mass_and_absent_groups_are_unavailable():
    table=run("attention.modality_ratio",head=2).table
    assert sum(row[0] for row in table.values)==1
    assert table.values[table.labels.index("image")][3]==1
    assert table.values[table.labels.index("generated")][3] is None
    sink=stats(run("attention.sinks",head=2))
    assert sink["Declared-group attention mass"]==.25


def test_head_similarity_is_symmetric_bounded_and_distinguishes_disjoint_support():
    matrix=np.asarray(run("heads.distribution_similarity").table.values)
    assert np.allclose(matrix,matrix.T)
    assert np.allclose(np.diag(matrix),1)
    assert matrix[0,1]==0
    assert np.all((matrix>=0)&(matrix<=1))


def test_comparison_views_keep_axis_and_source_identity_without_reductions():
    context=fixture()
    heads=run("attention.head_by_head",context,heads=[1,0]).comparisons
    assert [c.row.head for c in heads]==[1,0]
    assert heads[0].row.values==run("attention.row",context,head=1).row.values
    layers=run("attention.across_layers",context,layers=[1,0],head=2).comparisons
    assert [c.row.layer for c in layers]==[1,0]
    assert all(c.row.source_index==3 for c in layers)
    with pytest.raises(ValueError):run("attention.across_layers",context,layers=[2])


def test_defaults_are_resolved_to_exact_axes_for_saved_results():
    method=get_method("attention.weighted_heads")
    parsed=prepare_parameters(method.method_id,method.variant_id,{},fixture().record)
    assert parsed.heads==[0,1,2] and parsed.weights==[1,1,1]
